"""Offline: replay a Frontier policy over a recorded site, at no cost.

Record a site once, deeper than a normal crawl and without escalations:

    .venv/bin/python -m eval.run_jev --runs 1 --max-pages 120 --max-seconds 900 --no-escalation \\
        --label rec --sites example.com

then replay any policy over it under the real 40-page budget:

    .venv/bin/python -m eval.frontier_replay eval/snapshots/<stamp>/<site>/run1 [more run dirs ...]

The replay uses docseek.frontier.Frontier itself and mirrors jev_crawl's loop: two pages per batch, the
variant cap, goal language, the no-progress stop. Every page's links, page kinds and verdicts come from
the recording, so only the order changes. A page the recording never visited is `unknown`: it costs a
page and pays nothing, which is pessimistic for whichever policy walks off the recorded part of the site.
"""
from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path
from urllib.parse import unquote

from docseek.frontier import Frontier
from docseek.jev_crawl import (
    MAX_VARIANTS_PER_PATH, canonical, goal_language, is_first_page, keeps_parent_facets, paging_identity,
    should_stop_for_no_progress, url_language,
)
from eval.run_baseline import load_ground_truth, normalize_url

CHECKPOINTS = (10, 20, 30, 40)


class Recording:
    def __init__(self, run_dir: Path):
        run = json.loads((run_dir / 'run.json').read_text())
        self.name = f"{run['site']} | {run['goal'][:48]}"
        self.goal, self.site = run['goal'], run['site']
        self.seed = min(run['pages'], key=lambda p: p['page_no'])['url']
        self.kinds = {canonical(k['url']): k for k in reversed(run['page_kinds'])}   # first answer wins
        self.sitemap = [k['url'] for k in run['page_kinds'] if k.get('sitemap')]
        self.verdict = {c['url']: c['verdict'] for c in run['candidates']}
        self.pages = {}
        for meta in sorted(run_dir.glob('harvest_*.json')):
            harvest = json.loads(meta.read_text())
            self.pages[canonical(harvest['url'])] = harvest
        truth = load_ground_truth([run['site']])
        same_goal = truth and truth[0]['goal'] == run['goal']
        self.expected = truth[0]['expected'] if same_goal else set()
        self.pre_crawl = {c['url'] for c in run['candidates'] if c['source'] in ('sitemap', 'api')}


def replay(rec: Recording, frontier: Frontier, max_pages: int = 40, parallel: int = 2,
           patient_stop: bool = False, max_depth: int | None = None) -> dict:
    """`max_depth=None` replays the crawl without depth tracking: every page at depth 0, no depth limit."""
    language = goal_language(rec.goal)
    classified: set[str] = set()
    paths_queued: collections.Counter[str] = collections.Counter()
    seen_docs = set(rec.pre_crawl)
    visited: list[str] = []
    found_at: list[tuple[int, str]] = [(0, u) for u in rec.pre_crawl if rec.verdict.get(u) == 'accepted']
    stale = unknown = 0

    def add(links: list[dict], parent: str | None, depth: int = 0) -> None:
        for link in links:
            key = canonical(link['url'])
            answer = rec.kinds.get(key)
            if key in classified or answer is None:
                continue                                  # never classified in the recording: not queued there either
            if '[' in unquote(link['url']) and not link.get('facet_chosen') and not keeps_parent_facets(link['url'], parent):
                continue                                  # a facet link, as the crawl refuses it (replayed: none chosen)
            classified.add(key)
            path = paging_identity(link['url'])
            if paths_queued[path] >= MAX_VARIANTS_PER_PATH or (paths_queued[path] and is_first_page(link['url'])):
                continue
            frontier.add(link['url'], kind=answer['kind'], probability=answer['probability'], depth=depth,
                         other_language=language is not None and url_language(link['url']) not in (None, language),
                         path_seen=paths_queued[path] > 0, group=link.get('path', ''), parent=parent,
                         chrome=bool(link.get('chrome')))
            paths_queued[path] += 1

    frontier.add_seed(rec.seed)
    classified.add(canonical(rec.seed))
    add([{'url': u} for u in rec.sitemap], None)
    while len(visited) < max_pages:
        # the time budget is not replayed, so its share of the stop rule is the page share
        if should_stop_for_no_progress(stale, len(visited), max_pages, 0.0, 1.0) \
                and not (patient_stop and frontier.has_untried_tier1()):
            break
        batch = [item for item in (frontier.pop() for _ in range(min(parallel, max_pages - len(visited)))) if item]
        if not batch:
            break
        for url, depth, _ in batch:
            visited.append(url)
            page = rec.pages.get(canonical(url))
            accepted = kept = 0
            if page is None:
                unknown += 1
            else:
                for doc in page['documents']:
                    if doc['url'] in seen_docs:
                        continue
                    seen_docs.add(doc['url'])
                    verdict = rec.verdict.get(doc['url'], 'rejected')
                    kept += verdict != 'rejected'
                    if verdict == 'accepted':
                        accepted += 1
                        found_at.append((len(visited), doc['url']))
                if max_depth is None:
                    add(page['page_links'], url)
                elif depth < max_depth:
                    add(page['page_links'], url, depth + 1)
            frontier.record(url, accepted)
            stale = 0 if kept else stale + 1
    out = {'pages': len(visited), 'unknown': unknown, 'visited': visited, 'rescued_at': frontier.rescued_at}
    for n in CHECKPOINTS:
        got = {u for page_no, u in found_at if page_no <= n}
        out[f'acc@{n}'] = len(got)
        if rec.expected:
            out[f'R@{n}'] = round(len({normalize_url(u) for u in got} & rec.expected) / len(rec.expected), 2)
    return out


POLICIES = {'tier (shipped)': {'policy': 'tier'}, 'tier menus level (pre-0.10)': {'policy': 'tier', 'content_first': False},
            'tier listing-first': {'policy': 'tier', 'listing_first': True}}
for _depth in (2, 3, 4):
    POLICIES[f'tier max_depth {_depth}'] = {'policy': 'tier', 'max_depth': _depth}
POLICIES['bandit'] = {'policy': 'bandit'}
for _dry, _patient in itertools.product((5, 6, 7, 8), (False, True)):
    POLICIES[f'rescue d{_dry}{" patient" if _patient else ""}'] = {
        'policy': 'rescue', 'dry_pages': _dry, 'patient_stop': _patient}


def main() -> None:
    recordings = [Recording(Path(p)) for p in sys.argv[1:]]
    for rec in recordings:
        ceiling = sum(1 for v in rec.verdict.values() if v == 'accepted')
        print(f'\n== {rec.name}  recorded pages {len(rec.pages)}, accepted in recording {ceiling}, '
              f'ground truth {len(rec.expected) or "n/a (other goal)"}')
        print(f"{'policy':<26}" + ''.join(f'{"acc@" + str(n):>8}' for n in CHECKPOINTS)
              + ''.join(f'{"R@" + str(n):>7}' for n in CHECKPOINTS) + f'{"pages":>7}{"unknown":>9}{"rescued":>9}')
        for name, params in POLICIES.items():
            params = dict(params)
            patient = params.pop('patient_stop', False)
            depth = params.pop('max_depth', None)
            r = replay(rec, Frontier(**params), patient_stop=patient, max_depth=depth)
            print(f'{name:<26}' + ''.join(f'{r[f"acc@{n}"]:>8}' for n in CHECKPOINTS)
                  + ''.join(f'{r.get(f"R@{n}", "-"):>7}' for n in CHECKPOINTS) + f'{r["pages"]:>7}{r["unknown"]:>9}{str(r["rescued_at"] or "-"):>9}')


if __name__ == '__main__':
    main()
