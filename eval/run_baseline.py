"""Eval runner for the crawler: scores recall/precision/variance against user-supplied ground truth.

Reads ground_truth/*.json, runs agentic_crawl N times per site, scores each run by
normalized-URL match, and writes a report (JSON + console table).

Usage:
    .venv/bin/python -m eval.run_baseline                 # all sites, N=3
    .venv/bin/python -m eval.run_baseline --runs 1 --sites example.com
    .venv/bin/python -m eval.run_baseline --max-pages 15 [--model claude-haiku-4-5]

Match rule (ground_truth/README.md): strip query/fragment/trailing slash, case-fold host.
Applied symmetrically to expected and found URLs so it is a canonical key, not the fetch URL.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

_HERE = Path(__file__).resolve().parent
_GROUND_TRUTH = _HERE / 'ground_truth'
_REPORTS = _HERE / 'reports'


def start_url_for(site: str) -> str:
    return f'https://www.{site}/' if not site.startswith('www.') else f'https://{site}/'


def identity(url: str, keep_query: bool = False, identity_re: str | None = None) -> str:
    """normalize_url, unless the site declares what identifies a document (`identity_re`, one group): some
    sites serve one document as ?download=N:slug, ?download=N:slug&start=50 and /file/N-slug."""
    if identity_re:
        m = re.search(identity_re, url)
        if m:
            return f'{urlparse(url).hostname}#{m.group(1)}'
    return normalize_url(url, keep_query)


def normalize_url(url: str, keep_query: bool = False) -> str:
    """Canonical comparison key: lowercase scheme+host, no query, no fragment, no trailing slash.

    keep_query is for sites whose documents are told apart by the query string alone
    (`getfile.aspx?id=123`): dropping it would fold every document into one.
    """
    try:
        p = urlparse(url.strip())
    except Exception:
        return url.strip().lower()
    host = (p.hostname or '').lower()
    if p.port:
        host = f'{host}:{p.port}'
    path = p.path.rstrip('/') or '/'
    return urlunparse((p.scheme.lower(), host, path, '', p.query if keep_query else '', ''))


def load_ground_truth(sites_filter: list[str] | None) -> list[dict]:
    entries = []
    for f in sorted(_GROUND_TRUTH.glob('*.json')):
        d = json.loads(f.read_text())
        site = d['site']
        if sites_filter and site not in sites_filter:
            continue
        # example.json documents the format; it only runs when its site is named explicitly
        if f.name == 'example.json' and not sites_filter:
            continue
        keep_query = bool(d.get('match_query'))
        identity_re = d.get('identity_re')
        expected = {identity(doc['url'], keep_query, identity_re) for doc in d['expected_documents']}
        entries.append({
            'file': f.name,
            'site': site,
            'goal': d['goal'],
            'expected': expected,
            'expected_count': len(expected),
            # optional per-site settings: where to seed the crawl, whether it may leave the seed host,
            # whether the query string is part of a document's identity, and what shape the site tests
            'start_url': d.get('start_url'),
            'off_domain': bool(d.get('off_domain')),
            'keep_query': keep_query,
            'identity_re': identity_re,
            'goal_year': d.get('goal_year'),
            'profile': d.get('profile', 'generic'),   # the domain profile the judge words its questions with
            'shape': d.get('shape', ''),
        })
    return entries


def score(found: set[str], expected: set[str]) -> dict:
    tp = len(found & expected)
    # A site that holds nothing the goal asks for is scored too: missing nothing is full recall, and
    # any document returned there is a false accept.
    recall = tp / len(expected) if expected else 1.0
    precision = tp / len(found) if found else (0.0 if expected else 1.0)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
    return {
        'found_count': len(found),
        'true_positives': tp,
        'false_accepts': len(found - expected),
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'missed': sorted(expected - found),
        'extra': sorted(found - expected),
    }


def run_one(agentic_crawl, entry: dict, api_key: str, model: str,
            max_pages: int, max_depth: int) -> dict:
    start_url = entry.get('start_url') or start_url_for(entry['site'])
    t0 = time.time()
    downloads_seen: list[str] = []

    def on_event(ev):
        if ev.get('type') == 'agent_download':
            downloads_seen.append(ev.get('url', ''))

    result = agentic_crawl(
        start_url,
        entry['goal'],
        api_key=api_key,
        model=model,
        max_pages=max_pages,
        max_depth=max_depth,
        same_domain_only=not entry.get('off_domain'),
        on_event=on_event,
        enable_learning=False,   # baseline must not mutate the pattern store between runs
    )
    found = {identity(d.url, entry.get('keep_query', False), entry.get('identity_re')) for d in result.downloads}
    sc = score(found, entry['expected'])
    sc.update({
        'elapsed_s': round(time.time() - t0, 1),
        'pages_visited': result.pages_visited,
        'total_tokens': result.total_tokens,
    })
    return sc


def main() -> None:
    ap = argparse.ArgumentParser(prog='eval.run_baseline')
    ap.add_argument('--runs', type=int, default=3, help='Runs per site (N>=3 exposes nondeterminism)')
    ap.add_argument('--sites', nargs='*', default=None, help='Subset of site keys (e.g. example.com)')
    ap.add_argument('--model', default=None, help='Agent model (default: LLM_MODEL, or claude-haiku-4-5 on Anthropic)')
    ap.add_argument('--max-pages', type=int, default=15)
    ap.add_argument('--max-depth', type=int, default=3)
    ap.add_argument('--api-key', default=None)
    args = ap.parse_args()

    # The crawler package loads .env on import, but we read the key first, so load it here too.
    try:
        from dotenv import load_dotenv
        load_dotenv(_HERE.parent / '.env', override=False)
    except Exception:
        pass
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not (api_key or os.environ.get('LLM_PROVIDER')):
        print('No LLM configured: set LLM_PROVIDER / LLM_MODEL / LLM_API_KEY, or ANTHROPIC_API_KEY.', file=sys.stderr)
        sys.exit(1)

    from docseek.agent import agentic_crawl

    entries = load_ground_truth(args.sites)
    if not entries:
        print('No ground-truth entries matched.', file=sys.stderr)
        sys.exit(1)

    _REPORTS.mkdir(exist_ok=True)
    report = {'model': args.model, 'runs': args.runs, 'max_pages': args.max_pages,
              'max_depth': args.max_depth, 'sites': []}

    for entry in entries:
        print(f'\n=== {entry["site"]}  (expected {entry["expected_count"]})  goal: {entry["goal"]!r}',
              file=sys.stderr)
        runs = []
        for i in range(args.runs):
            print(f'  run {i + 1}/{args.runs} …', file=sys.stderr)
            try:
                runs.append(run_one(agentic_crawl, entry, api_key, args.model,
                                    args.max_pages, args.max_depth))
            except Exception as exc:
                print(f'  run {i + 1} FAILED: {exc}', file=sys.stderr)
                runs.append({'error': str(exc), 'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
                             'found_count': 0, 'true_positives': 0, 'missed': [], 'extra': []})
        recalls = [r['recall'] for r in runs]
        precisions = [r['precision'] for r in runs]
        site_summary = {
            'site': entry['site'],
            'goal': entry['goal'],
            'expected_count': entry['expected_count'],
            'recall_mean': round(statistics.mean(recalls), 3),
            'recall_stdev': round(statistics.pstdev(recalls), 3),
            'recall_min': round(min(recalls), 3),
            'recall_max': round(max(recalls), 3),
            'precision_mean': round(statistics.mean(precisions), 3),
            'precision_stdev': round(statistics.pstdev(precisions), 3),
            'runs': runs,
        }
        report['sites'].append(site_summary)
        print(f'  → recall {site_summary["recall_mean"]:.2f} '
              f'(±{site_summary["recall_stdev"]:.2f}, {site_summary["recall_min"]:.2f}-{site_summary["recall_max"]:.2f})  '
              f'precision {site_summary["precision_mean"]:.2f} (±{site_summary["precision_stdev"]:.2f})',
              file=sys.stderr)

    stamp = report_stamp()
    out = _REPORTS / f'baseline_{stamp}.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print('\n' + '=' * 72)
    print(f'{"SITE":<26}{"EXP":>5}{"RECALL":>16}{"PRECISION":>14}')
    print('-' * 72)
    for s in report['sites']:
        print(f'{s["site"]:<26}{s["expected_count"]:>5}'
              f'{s["recall_mean"]:>8.2f} ±{s["recall_stdev"]:<5.2f}'
              f'{s["precision_mean"]:>8.2f} ±{s["precision_stdev"]:<4.2f}')
    print('=' * 72)
    print(f'Report: {out}')


def report_stamp() -> str:
    # Avoid Date.now-style nondeterminism concerns in orchestration; wall clock is fine for a CLI.
    return time.strftime('%Y%m%d_%H%M%S')


if __name__ == '__main__':
    main()
