"""Run a generated discovery program in a child process that reaches the web only through the parent's Fetcher.

The program was written by a model that read untrusted page text, so the child (docseek/codegen/_child.py):
- is a fresh interpreter started with an empty environment (no API keys), with OS limits on CPU time and memory,
- is a standalone script: it does not import docseek, and nothing re-runs the caller's main module,
- may import only an allowlist of modules (imported before the audit hook is installed),
- has an audit hook that refuses sockets, subprocesses, file opens and ctypes,
- asks the parent for every page; the parent applies the Fetcher's site, robots.txt and address rules.

This is a boundary, not a hardened jail: a determined escape from CPython is possible. It keeps honest mistakes
and casual injected instructions from touching anything but the target site. Generated programs are therefore off
unless PROGRAMS_DIR is set; run docseek in a container if strangers can reach it.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ALLOWED_MODULES = ('re', 'json', 'html', 'urllib.parse', 'collections', 'itertools', 'datetime', 'math',
                   'string', 'functools', 'unicodedata', 'html.parser')
PROGRAM_TIMEOUT_S = 300
MAX_DOCUMENTS = 5000
_CHILD = Path(__file__).resolve().parent / '_child.py'


class FetchRefused(Exception):
    """The guard said no; the message says why, so a program or the agent can adapt."""


def _child_env() -> dict:
    """What a fresh interpreter needs and nothing more: no keys, no proxies, no paths into the user's home."""
    return {k: os.environ[k] for k in ('SYSTEMROOT',) if k in os.environ}     # Windows cannot start Python without it


def run_program(code: str, fetcher, timeout_s: float = PROGRAM_TIMEOUT_S) -> dict:
    """{documents, error, fetch_failures, requests, renders, seconds}. Never raises for the program's own faults.
    `fetcher` is a docseek.codegen.fetcher.Fetcher (or anything with its fetch/render/post)."""
    started = time.monotonic()
    requests0, renders0 = fetcher.requests, fetcher.renders
    result = {'documents': [], 'error': None, 'fetch_failures': 0}
    proc = subprocess.Popen([sys.executable, '-I', str(_CHILD)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=_child_env(), text=True, encoding='utf-8', bufsize=1)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout] + [lines.put(None)], daemon=True).start()
    try:
        proc.stdin.write(json.dumps({'code': code, 'cpu_seconds': int(timeout_s) + 10, 'allowed': ALLOWED_MODULES,
                                     'max_documents': MAX_DOCUMENTS}) + '\n')
        proc.stdin.flush()
        while True:
            left = timeout_s - (time.monotonic() - started)
            if left <= 0:
                result['error'] = f'timed out after {timeout_s:.0f} s'
                break
            try:
                line = lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None:
                proc.wait(5)
                result['error'] = f'the program process exited ({proc.returncode}) without a result'
                break
            message = json.loads(line)
            if message['op'] == 'done':
                result['documents'] = message['documents']
                break
            if message['op'] == 'error':
                result['error'] = message['message']
                break
            ok, value = _serve(fetcher, message, result)
            proc.stdin.write(json.dumps({'ok': ok, 'value': value}) + '\n')
            proc.stdin.flush()
    except (BrokenPipeError, json.JSONDecodeError) as exc:
        result['error'] = f'the program process died ({type(exc).__name__})'
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(5)
    result.update(requests=fetcher.requests - requests0, renders=fetcher.renders - renders0,
                  seconds=round(time.monotonic() - started, 1))
    return result


def _serve(fetcher, message: dict, result: dict) -> tuple[bool, str]:
    """One page for the child: (True, body) or (False, why). A failure is counted - programs may swallow it."""
    url = message.get('url', '')
    try:
        if message['op'] == 'render':
            return True, fetcher.render(url)['html']
        page = fetcher.post(url, message.get('body') or {}) if message['op'] == 'post' else fetcher.fetch(url)
        if page['status'] >= 400:
            result['fetch_failures'] += 1
            return False, f'HTTP {page["status"]} for {url}'
        return True, page['text']
    except Exception as exc:  # noqa: BLE001 - a refused or failed fetch is the program's to handle
        result['fetch_failures'] += 1
        return False, f'{type(exc).__name__}: {exc}'[:300]
