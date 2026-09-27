"""Run a generated discovery program in a child process that reaches the web only through the parent's Fetcher.

The program was written by a model that read untrusted page text, so the child:
- starts with an empty environment (no API keys) and OS limits on CPU time and memory,
- may import only an allowlist of modules (imported before the audit hook is installed),
- has an audit hook that refuses sockets, subprocesses, file opens and ctypes,
- asks the parent for every page over a pipe; the parent applies the Fetcher's site, robots.txt and address rules.

This is a boundary, not a hardened jail: a determined escape from CPython is possible. It keeps honest mistakes
and casual injected instructions from touching anything but the target site. Generated programs are therefore off
unless PROGRAMS_DIR is set; run docseek in a container if strangers can reach it.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback

ALLOWED_MODULES = ('re', 'json', 'html', 'urllib.parse', 'collections', 'itertools', 'datetime', 'math',
                   'string', 'functools', 'unicodedata', 'html.parser')
PROGRAM_TIMEOUT_S = 300
MAX_DOCUMENTS = 5000
MEMORY_LIMIT_BYTES = 2 * 1024 ** 3
_BLOCKED_EVENTS = ('socket.', 'subprocess.', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn', 'os.fork',
                   'open', 'ctypes.', 'os.remove', 'os.rename', 'os.putenv', 'shutil.')
_BLOCKED_BUILTINS = ('open', 'exec', 'eval', 'compile', 'input', 'breakpoint', 'exit', 'quit', 'help')


class FetchRefused(Exception):
    """The guard said no; the message says why, so a program or the agent can adapt."""


def _limit_resources(cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:                                   # not a POSIX system: the wall-clock timeout still applies
        return
    for limit, value in ((resource.RLIMIT_CPU, cpu_seconds), (getattr(resource, 'RLIMIT_AS', None), MEMORY_LIMIT_BYTES)):
        if limit is None:
            continue
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):                     # macOS refuses RLIMIT_AS; keep what the platform allows
            pass


def _child(code: str, conn, cpu_seconds: int) -> None:
    import builtins
    import importlib
    import os
    import sys

    modules = {name: importlib.import_module(name) for name in ALLOWED_MODULES}
    modules['urllib'] = importlib.import_module('urllib')
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules or any(name == m.split('.')[0] for m in ALLOWED_MODULES):
            return real_import(name, globals, locals, fromlist, level)
        raise ImportError(f'module {name!r} is not available to a discovery program')

    def audit(event, args):
        if event.startswith(_BLOCKED_EVENTS):
            raise RuntimeError(f'{event} is not allowed in a discovery program')

    def ask(kind: str, payload):
        conn.send((kind, payload))
        ok, value = conn.recv()
        if not ok:
            raise FetchRefused(value)
        return value

    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins) if k not in _BLOCKED_BUILTINS}
    safe_builtins['__import__'] = guarded_import
    env = {'__builtins__': safe_builtins, '__name__': 'discovery_program', 'FetchRefused': FetchRefused}
    try:
        compiled = compile(code, 'discovery_program.py', 'exec')
        os.environ.clear()                                # the package loaded .env on import: no keys past here
        _limit_resources(cpu_seconds)
        sys.addaudithook(audit)
        exec(compiled, env)
        discover = env['discover']
        args = [lambda url: ask('fetch', url), lambda url: ask('render', url)]
        if discover.__code__.co_argcount >= 3:
            args.append(lambda url, body: ask('post', (url, body)))
        found = discover(*args)
        docs = [{'url': str(d['url']), 'name': str(d.get('name', ''))[:300], 'context': str(d.get('context', ''))[:400]}
                for d in list(found)[:MAX_DOCUMENTS] if isinstance(d, dict) and d.get('url')]
        conn.send(('done', docs))
    except BaseException as exc:  # noqa: BLE001 - the program's own failure is the result
        conn.send(('error', describe_error(exc, code)))


def describe_error(exc: BaseException, code: str) -> str:
    """The traceback without reading source files (the audit hook refuses open): the program's own lines are
    quoted from its code, library frames by name."""
    lines = code.splitlines()
    frames = []
    for frame, lineno in traceback.walk_tb(exc.__traceback__):
        name = frame.f_code.co_filename
        where = f'{name}:{lineno} in {frame.f_code.co_name}'
        if name == 'discovery_program.py' and 0 < lineno <= len(lines):
            where += f'\n    {lines[lineno - 1].strip()}'
        frames.append(where)
    return '\n'.join(frames[-6:] + [f'{type(exc).__name__}: {exc}'])[-3000:]


def run_program(code: str, fetcher, timeout_s: float = PROGRAM_TIMEOUT_S) -> dict:
    """{documents, error, fetch_failures, requests, renders, seconds}. Never raises for the program's own faults.
    `fetcher` is a docseek.codegen.fetcher.Fetcher (or anything with its fetch/render/post)."""
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(code, child, int(timeout_s) + 10), daemon=True)
    started = time.monotonic()
    requests0, renders0 = fetcher.requests, fetcher.renders
    proc.start()
    result = {'documents': [], 'error': None, 'fetch_failures': 0}
    try:
        while True:
            if time.monotonic() - started > timeout_s:
                result['error'] = f'timed out after {timeout_s:.0f} s'
                break
            if not parent.poll(1.0):
                if not proc.is_alive():
                    result['error'] = f'the program process exited ({proc.exitcode}) without a result'
                    break
                continue
            msg = parent.recv()
            if msg[0] == 'done':
                result['documents'] = msg[1]
                break
            if msg[0] == 'error':
                result['error'] = msg[1]
                break
            parent.send(_serve(fetcher, *msg, result))
    except EOFError:
        result['error'] = 'the program process died'
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
    result.update(requests=fetcher.requests - requests0, renders=fetcher.renders - renders0,
                  seconds=round(time.monotonic() - started, 1))
    return result


def _serve(fetcher, kind: str, payload, result: dict) -> tuple[bool, str]:
    """One page for the child: (True, body) or (False, why). A failure is counted - programs may swallow it."""
    try:
        if kind == 'post':
            url, body = payload
            page = fetcher.post(url, body)
        elif kind == 'render':
            url = payload
            return True, fetcher.render(url)['html']
        else:
            url = payload
            page = fetcher.fetch(url)
        if page['status'] >= 400:
            result['fetch_failures'] += 1
            return False, f'HTTP {page["status"]} for {url}'
        return True, page['text']
    except Exception as exc:  # noqa: BLE001 - a refused or failed fetch is the program's to handle
        result['fetch_failures'] += 1
        return False, f'{type(exc).__name__}: {exc}'[:300]
