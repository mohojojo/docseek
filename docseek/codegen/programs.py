"""Generated programs on disk, hybrid discovery, and the drift check.

A ProgramStore keeps one program per (start host, goal) in a directory (PROGRAMS_DIR):
  <key>.py             the program
  <key>.json           its metadata: goal, start_url, notes, data_hosts, post_endpoints, model, usage, health
  <key>.snapshot.json  what it returned when snapshotted, for the drift check
  <key>.log.json       the last generation's log (written even when it produced no program)

Hybrid discovery - the program first, the crawl as the safety net:
  program exists, verified and not stale -> run it (no model) -> judge -> healthy? answer with it
                                                                       -> not healthy: mark it stale, crawl
  no program, or an unverified one       -> crawl
After a crawl the caller may generate a (new) program for next time (generate_program).

A program is unhealthy when it errors, returns nothing where it used to find something, or keeps (accepted +
unsure) fewer than half of what it kept last time: the site changed under it. Health cannot see a program that
confidently returns the wrong slice of a site - only a crawl or a person can.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from ..jev_crawl import year_of
from ..judge import RelevanceJudge, verdict_for
from ..models import AgenticCrawlResult, AgenticDownload
from ..reach import bare_host
from ..series import date_of, period_of
from .fetcher import Fetcher
from .sandbox import run_program

MIN_KEPT_SHARE = 0.5        # below half of last time's kept documents, the program is presumed broken
DRIFT_KEEP_SHARE = 0.8      # the drift check: at least this share of the snapshot's documents still come back
_KEY_RE = re.compile(r'^[a-z0-9_-]+__[0-9a-f]{8}$')


def program_key(start_url: str, goal: str) -> str:
    """One program per (start host, goal)."""
    host = bare_host(urlparse(start_url).netloc)
    return f"{re.sub(r'[^a-z0-9-]', '_', host)}__{hashlib.sha256(goal.encode()).hexdigest()[:8]}"


@dataclass
class Program:
    key: str
    code: str
    meta: dict

    def fetcher(self, **kwargs) -> Fetcher:
        """A Fetcher that may reach what the site's own pages reached while the program was written."""
        return Fetcher(self.meta['start_url'], data_hosts=set(self.meta.get('data_hosts', [])),
                       post_endpoints=set(self.meta.get('post_endpoints', [])), **kwargs)


class ProgramStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str, suffix: str) -> Path:
        if not _KEY_RE.match(key):
            raise KeyError(f'not a program key: {key!r}')
        return self.root / f'{key}{suffix}'

    def load(self, key: str) -> Program | None:
        code, meta = self._path(key, '.py'), self._path(key, '.json')
        if not (code.exists() and meta.exists()):
            return None
        return Program(key, code.read_text(), json.loads(meta.read_text()))

    def save(self, program: Program) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(program.key, '.py').write_text(program.code.rstrip('\n') + '\n')
        self.save_meta(program)

    def save_meta(self, program: Program) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(program.key, '.json').write_text(json.dumps(program.meta, indent=1, ensure_ascii=False, default=str))

    def keys(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob('*.py') if _KEY_RE.match(p.stem)
                      and p.with_suffix('.json').exists()) if self.root.exists() else []

    def save_log(self, key: str, log: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(key, '.log.json').write_text(json.dumps(log, indent=1, ensure_ascii=False, default=str))

    def load_log(self, key: str) -> dict | None:
        path = self._path(key, '.log.json')
        return json.loads(path.read_text()) if path.exists() else None

    def last_attempt(self, key: str) -> float | None:
        """When a program was last generated for this key (successfully or not), as a POSIX time."""
        path = self._path(key, '.log.json')
        return path.stat().st_mtime if path.exists() else None

    def delete(self, key: str) -> bool:
        paths = [self._path(key, s) for s in ('.py', '.json', '.snapshot.json', '.log.json')]
        found = any(p.exists() for p in paths)
        for p in paths:
            p.unlink(missing_ok=True)
        return found

    def load_snapshot(self, key: str) -> dict | None:
        path = self._path(key, '.snapshot.json')
        return json.loads(path.read_text()) if path.exists() else None

    def save_snapshot(self, key: str, snapshot: dict) -> None:
        self._path(key, '.snapshot.json').write_text(json.dumps(snapshot, indent=1))


def health(error: str | None, kept: int, last_kept: int | None) -> str:
    """'healthy', or why not: 'error', 'empty', 'dropped'."""
    if error:
        return 'error'
    if kept == 0:
        return 'healthy' if last_kept == 0 else 'empty'   # a site with nothing to find stays empty
    if last_kept and kept < MIN_KEPT_SHARE * last_kept:
        return 'dropped'
    return 'healthy'


def as_candidates(documents: list[dict]) -> list[dict]:
    """A program's documents as the judge sees a crawl's links. A program's `context` is by definition where the
    document sits (its section, column or tab, its row, its date), so it goes where a crawl puts the section heading
    - shown with every link - not in the surrounding text, which a judge only sees for links with short names."""
    return [{'url': d['url'], 'name': d['name'], 'context': d.get('context', ''), 'section': d.get('context', '')}
            for d in documents]


def judge_documents(goal: str, start_url: str, documents: list[dict], judge: RelevanceJudge) -> list[AgenticDownload]:
    """The crawl's own relevance question over what a program returned."""
    candidates = as_candidates(documents)
    scores = judge.relevance(goal, f'documents found on {start_url}', candidates) if candidates else []
    return [AgenticDownload(
        url=d['url'], name=d['name'] or d['url'].rsplit('/', 1)[-1], reason='generated program', source_page=start_url,
        relevance=s, verdict=verdict_for(s), source='program', period=period_of(f"{d['name']} {d['url']}"),
        year=year_of(f"{d.get('context', '')} {d['name']} {d['url']}"), published=date_of(d.get('context', '')))
        for d, s in zip(documents, scores)]


def run_saved(program: Program, judge: RelevanceJudge, runner=run_program) -> tuple[dict, list[AgenticDownload]]:
    """Run a stored program and judge what it returned: (run, downloads)."""
    fetcher = program.fetcher()
    try:
        run = runner(program.code, fetcher)
    finally:
        fetcher.close()
    downloads = [] if run['error'] else judge_documents(program.meta['goal'], program.meta['start_url'],
                                                        run['documents'], judge)
    return run, downloads


def hybrid_discover(start_url: str, goal: str, *, store: ProgramStore, judge: RelevanceJudge,
                    crawl: Callable[[], AgenticCrawlResult], include_rejected: bool = False,
                    runner=run_program) -> AgenticCrawlResult:
    """The stored program's answer when it is healthy, else the crawl's. result.program says which and why:
    {key, path: 'program'|'crawl', reason: 'healthy'|'no_program'|'stale'|'error'|'empty'|'dropped'}."""
    key = program_key(start_url, goal)
    program = store.load(key)
    reason = ('no_program' if program is None else 'stale' if program.meta.get('stale')
              else 'unverified' if program.meta.get('verified') is False else None)
    if reason is None:
        began = time.monotonic()
        run, downloads = run_saved(program, judge, runner)
        kept = sum(1 for d in downloads if d.verdict != 'rejected')
        reason = health(run['error'], kept, program.meta.get('last_kept'))
        program.meta.update(last_run=time.strftime('%Y-%m-%dT%H:%M:%S'), last_reason=reason)
        if reason == 'healthy':
            program.meta['last_kept'] = kept
            store.save_meta(program)
            rejected = [d for d in downloads if d.verdict == 'rejected']
            return AgenticCrawlResult(
                start_url=start_url, goal=goal,
                downloads=downloads if include_rejected else [d for d in downloads if d.verdict != 'rejected'],
                decision_model='program', relevance_model=judge.model if downloads else None,
                rejected_count=len(rejected), stop_reason='program',
                program={'key': key, 'path': 'program', 'reason': reason, 'requests': run['requests'],
                         'renders': run['renders'], 'seconds': round(time.monotonic() - began, 1),
                         'fetch_failures': run['fetch_failures']})
        program.meta['stale'] = True
        program.meta['failures'] = program.meta.get('failures', 0) + 1
        store.save_meta(program)
    result = crawl()
    result.program = {'key': key, 'path': 'crawl', 'reason': reason}
    return result


def generate_program(store: ProgramStore, start_url: str, goal: str, *, llm, judge: RelevanceJudge,
                     explorer_cls=None) -> tuple[Program | None, dict]:
    """Write a program for (start_url, goal) and store it, replacing a stale one: (program, the agent's report).
    The program is None when the agent produced no code; the report then says how far it got. Takes minutes and a
    strong coding model; callers run it in the background."""
    from .explorer import Explorer
    result = (explorer_cls or Explorer)(start_url, goal, llm=llm, judge=judge).explore()
    report = {k: v for k, v in result.items() if k not in ('code', 'log')}
    key = program_key(start_url, goal)
    store.save_log(key, {'goal': goal, 'start_url': start_url, 'attempted': time.strftime('%Y-%m-%dT%H:%M:%S'),
                         **report, 'log': result.get('log', [])})
    if not result.get('code'):
        return None, report
    meta = {'goal': goal, 'start_url': start_url, **report, 'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'stale': False, 'last_kept': result.get('generated_kept')}
    program = Program(key, result['code'], meta)
    store.save(program)
    return program, report


def replay(program: Program, runner=run_program) -> dict:
    """What a program returns now, with no model: {urls, error, seconds, fetch_failures}."""
    fetcher = program.fetcher()
    try:
        run = runner(program.code, fetcher)
    finally:
        fetcher.close()
    return {'urls': sorted({d['url'] for d in run['documents']}), 'error': run['error'], 'seconds': run['seconds'],
            'fetch_failures': run['fetch_failures']}


def drift_status(now: dict, then: dict) -> tuple[str, float]:
    """ok / grew (the site published more) / shrank (under DRIFT_KEEP_SHARE of the snapshot) / broken."""
    if now['error'] or (not now['urls'] and then['urls']):
        return 'broken', 0.0
    kept = len(set(now['urls']) & set(then['urls'])) / len(then['urls']) if then['urls'] else 1.0
    if kept < DRIFT_KEEP_SHARE:
        return 'shrank', kept
    return ('grew' if len(now['urls']) > len(then['urls']) else 'ok'), kept


def snapshot(store: ProgramStore, key: str, runner=run_program) -> dict:
    """Record what a program returns today, for later drift checks."""
    now = {'date': time.strftime('%Y-%m-%d'), **replay(store.load(key), runner)}
    store.save_snapshot(key, now)
    return now


def check_drift(store: ProgramStore, keys: list[str] | None = None, runner=run_program) -> list[dict]:
    """Replay each program and compare it with its snapshot. No model and no judge: it measures the programs, not
    relevance. A program without a snapshot gets one and is reported as 'snapshot'."""
    rows = []
    for key in keys or store.keys():
        program = store.load(key)
        if program is None:
            continue
        then = store.load_snapshot(key)
        if then is None:
            now = snapshot(store, key, runner)
            rows.append({'key': key, 'status': 'snapshot', 'now': len(now['urls']), 'error': now['error']})
            continue
        now = replay(program, runner)
        state, kept = drift_status(now, then)
        rows.append({'key': key, 'status': state, 'kept': round(kept, 2), 'then': len(then['urls']),
                     'now': len(now['urls']), 'snapshot': then['date'], 'fetch_failures': now['fetch_failures'],
                     'error': (now['error'] or '')[-300:] or None})
    return rows
