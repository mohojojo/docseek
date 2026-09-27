"""Jev decision layer eval: recall / precision per site against user-supplied ground truth.

    .venv/bin/python -m eval.run_jev --runs 2 --sites example.com example.org
    .venv/bin/python -m eval.run_jev --runs 1 --sites example.com --no-snapshots

Ground truth lives in eval/ground_truth/*.json; see eval/ground_truth/README.md for the format.

Every run is scored twice: `accepted` (the band the precision guardrail is about) and `returned`
(accepted + unsure + unscored - what a consumer of the API actually receives). `recall_at` is the
accepted recall after N pages, so a ranking change shows up as budget efficiency and not only as a
final number.

A run that ends `jev_unavailable` measures TypeSafe's availability, not the crawler. It is kept in
the JSON, marked `discarded`, and left out of every summary.

Snapshots (each page's HTML, harvested links and page-kind answers) go to eval/snapshots/, which is
not committed. They let a frontier or link-context change be replayed offline at no cost.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

from eval.run_baseline import identity, load_ground_truth, score, start_url_for

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE / 'reports'
_SNAPSHOTS = _HERE / 'snapshots'
RECALL_AT_PAGES = (10, 20, 40)
RETURNED_VERDICTS = {'accepted', 'unsure', 'unscored'}


def page_of(download, page_numbers: dict[str, int]) -> int:
    """The page a Candidate was found on. Sitemap and API Candidates exist before page 1."""
    if download.source in ('sitemap', 'api'):
        return 0
    return page_numbers.get(download.source_page, 0)


def score_run(downloads: list, page_numbers: dict[str, int], expected: set[str],
              keep_query: bool = False, identity_re: str | None = None, goal_year: str | None = None) -> dict:
    if goal_year:
        # what a consumer that asked for one year keeps: that year's documents and the undated ones
        downloads = [d for d in downloads if getattr(d, 'year', None) in (None, str(goal_year))]

    def pick(verdicts: set[str], max_page: int | None = None) -> set[str]:
        return {identity(d.url, keep_query, identity_re) for d in downloads if d.verdict in verdicts
                and (max_page is None or page_of(d, page_numbers) <= max_page)}

    hits = sorted(page_of(d, page_numbers) for d in downloads
                  if d.verdict == 'accepted' and identity(d.url, keep_query, identity_re) in expected)
    return {
        'accepted': score(pick({'accepted'}), expected),
        'returned': score(pick(RETURNED_VERDICTS), expected),
        'recall_at': {str(n): round(score(pick({'accepted'}, n), expected)['recall'], 3)
                      for n in RECALL_AT_PAGES},
        'pages_to_first_hit': hits[0] if hits else None,
    }


def run_one(jev_crawl, entry: dict, api_key: str, args, snapshot_dir: Path | None) -> dict:
    from docseek.judge import make_judge

    page_numbers: dict[str, int] = {}
    pages: list[dict] = []
    kinds: list[dict] = []
    harvests: list[str] = []
    rescued: list[int] = []
    recipe_events: list[dict] = []
    lock = threading.Lock()

    def on_event(ev: dict) -> None:
        if ev.get('type') in ('jev_recipe_recorded', 'jev_recipe_replayed'):
            recipe_events.append(ev)
        if ev.get('type') == 'jev_frontier_rescue':
            rescued.append(ev['after_pages'])
        if ev.get('type') == 'jev_page':
            page_numbers[ev['url']] = ev['page_no']
            pages.append({k: v for k, v in ev.items() if k != 'type'})

    def on_trace(ev: dict) -> None:
        with lock:
            if ev['type'] == 'page_kinds':
                # links classified before the first page was harvested came from the sitemap
                kinds.extend({**link, 'depth': ev['depth'], 'sitemap': not harvests} for link in ev['links'])
                return
            harvests.append(ev['url'])
            n = len(harvests)
            (snapshot_dir / f'harvest_{n:03d}.html').write_text(ev.pop('html'), encoding='utf-8')
            (snapshot_dir / f'harvest_{n:03d}.json').write_text(
                json.dumps(ev, ensure_ascii=False, indent=1), encoding='utf-8')

    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    result = jev_crawl(
        args.start_url or entry.get('start_url') or start_url_for(entry['site']), args.goal or entry['goal'],
        api_key=api_key, same_domain_only=not (args.off_domain or entry.get('off_domain')), include_rejected=True,
        max_pages=args.max_pages, max_seconds=args.max_seconds,
        **({'agent_token_cap': 0} if args.no_escalation else {}), frontier_policy=args.frontier,
        recipes_dir=args.recipes, judge=make_judge(args.judge, args.profile or entry.get('profile')),
        on_event=on_event, on_trace=on_trace if snapshot_dir else None,
    )
    run = score_run(result.downloads, page_numbers, entry['expected'], entry.get('keep_query', False),
                    entry.get('identity_re'), entry.get('goal_year'))
    run.update({
        'discarded': result.stop_reason == 'jev_unavailable',
        'stop_reason': result.stop_reason, 'pages': result.pages_visited,
        'elapsed_s': round(time.perf_counter() - began, 1),
        'agent_tokens': result.total_tokens + result.cache_read_tokens + result.cache_creation_tokens,
        'jev_usd': result.jev_cost_usd, 'jev_requests': result.jev_requests,
        'rejected': result.rejected_count, 'escalations': result.escalations,
        'escalations_skipped': len(result.escalations_skipped),
        'frontier_rescued_at': rescued[0] if rescued else None,
        'recipes': recipe_events,
        'decision_model': result.decision_model, 'trace': pages,
    })
    if snapshot_dir:
        (snapshot_dir / 'run.json').write_text(json.dumps({
            'site': entry['site'], 'goal': args.goal or entry['goal'], 'pages': pages, 'page_kinds': kinds,
            'candidates': [d.model_dump() for d in result.downloads],
        }, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    return run


def summarise(runs: list[dict]) -> dict | None:
    kept = [r for r in runs if 'error' not in r and not r['discarded']]
    if not kept:
        return None

    def stat(read) -> dict:
        values = [read(r) for r in kept]
        return {'mean': round(statistics.mean(values), 3), 'stdev': round(statistics.pstdev(values), 3),
                'min': round(min(values), 3)}

    return {
        'runs_kept': len(kept), 'runs_discarded': len(runs) - len(kept),
        'accepted_recall': stat(lambda r: r['accepted']['recall']),
        'accepted_precision': stat(lambda r: r['accepted']['precision']),
        'returned_recall': stat(lambda r: r['returned']['recall']),
        'returned_precision': stat(lambda r: r['returned']['precision']),
        'recall_at': {str(n): stat(lambda r, n=n: r['recall_at'][str(n)]) for n in RECALL_AT_PAGES},
        'pages': stat(lambda r: r['pages']), 'elapsed_s': stat(lambda r: r['elapsed_s']),
        'agent_tokens': stat(lambda r: r['agent_tokens']), 'jev_usd': stat(lambda r: r['jev_usd']),
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog='eval.run_jev')
    ap.add_argument('--runs', type=int, default=2)
    ap.add_argument('--sites', nargs='*', default=None, help='Subset of site keys (e.g. example.com)')
    ap.add_argument('--max-pages', type=int, default=40)
    ap.add_argument('--max-seconds', type=float, default=180.0)
    ap.add_argument('--off-domain', action='store_true',
                    help='same_domain_only=False: let the crawl follow links off the seed host. A site whose '
                         'documents live on another host scores 0 without it.')
    ap.add_argument('--start-url', default=None,
                    help='Seed one site somewhere other than its home page, e.g. example.com at '
                         'https://docs.example.com/reports (for when the home page never reaches the '
                         'documents). Use with a single --sites entry.')
    ap.add_argument('--goal', default=None, help='Override the ground-truth goal (scores then mean little)')
    ap.add_argument('--no-escalation', action='store_true',
                    help='Spend no agent tokens: for recording a site for offline replay (eval.frontier_replay)')
    ap.add_argument('--frontier', default='tier', choices=['tier', 'rescue', 'bandit'],
                    help="Frontier policy. 'tier' is what the service defaults to.")
    ap.add_argument('--recipes', default=None,
                    help='Directory for escalation recipes: record on the first crawl, replay on the next')
    ap.add_argument('--judge', choices=['jev', 'llm'], default=None,
                    help='Relevance judge (default: Jev when TYPESAFE_API_KEY is set, the LLM judge otherwise)')
    ap.add_argument('--profile', default=None, help="Override every site's domain profile (default: the site's own)")
    ap.add_argument('--no-snapshots', action='store_true')
    ap.add_argument('--label', default='', help='Short tag for the report name, e.g. before / after')
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(_HERE.parent / '.env', override=False)
    except Exception:
        pass
    api_key = os.environ.get('ANTHROPIC_API_KEY')   # None is fine: the agent then runs on LLM_* (docseek.llm)
    from docseek.judge import JudgeUnavailable, make_judge
    try:
        make_judge(args.judge, args.profile)             # fail now, not on every site, when the judge cannot run
    except (JudgeUnavailable, ValueError) as exc:
        print(f'No relevance judge: {exc}', file=sys.stderr)
        sys.exit(1)

    from docseek.jev_crawl import jev_crawl

    entries = load_ground_truth(args.sites)
    if not entries:
        print('No ground-truth entries matched.', file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime('%Y%m%d_%H%M%S') + (f'_{args.label}' if args.label else '')
    _REPORTS.mkdir(exist_ok=True)
    out = _REPORTS / f'jev_{stamp}.json'
    report = {'frontier': args.frontier, 'start_url': args.start_url, 'max_pages': args.max_pages, 'max_seconds': args.max_seconds, 'off_domain': args.off_domain,
              'runs': args.runs, 'sites': {}}
    for entry in entries:
        site = entry['site']
        runs: list[dict] = []
        report['sites'][site] = {'expected': entry['expected_count'], 'goal': entry['goal'], 'runs': runs}
        for i in range(args.runs):
            print(f'=== {site} run {i + 1}/{args.runs}', file=sys.stderr, flush=True)
            snapshot_dir = None if args.no_snapshots else _SNAPSHOTS / stamp / site / f'run{i + 1}'
            try:
                run = run_one(jev_crawl, entry, api_key, args, snapshot_dir)
            except Exception as exc:  # noqa: BLE001 - one failed run must not lose the others
                print(f'    FAILED: {exc}', file=sys.stderr, flush=True)
                run = {'error': str(exc)}
            runs.append(run)
            if 'error' not in run:
                acc, ret = run['accepted'], run['returned']
                print(f"    accepted R {acc['recall']:.2f} P {acc['precision']:.2f} | returned P "
                      f"{ret['precision']:.2f} | R@ {run['recall_at']} | pages {run['pages']} "
                      f"{run['stop_reason']}{'  DISCARDED' if run['discarded'] else ''}",
                      file=sys.stderr, flush=True)
            report['sites'][site]['summary'] = summarise(runs)
            out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))

    print(f'\n{"SITE":<24}{"ACC R":>8}{"ACC P":>8}{"RET P":>8}' + ''.join(f'{"R@" + str(n):>8}' for n in RECALL_AT_PAGES)
          + f'{"PAGES":>8}{"KEPT":>6}')
    for site, data in report['sites'].items():
        s = data['summary']
        if not s:
            print(f'{site:<24}  no usable run')
            continue
        print(f"{site:<24}{s['accepted_recall']['mean']:>8.2f}{s['accepted_precision']['mean']:>8.2f}"
              f"{s['returned_precision']['mean']:>8.2f}"
              + ''.join(f"{s['recall_at'][str(n)]['mean']:>8.2f}" for n in RECALL_AT_PAGES)
              + f"{s['pages']['mean']:>8.1f}{s['runs_kept']:>6}")
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
