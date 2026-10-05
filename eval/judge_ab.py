"""Offline A/B of what the Relevance judge is told, over recorded sites - no crawling, one browser, no agent.

    .venv/bin/python -m eval.judge_ab eval/snapshots/<stamp>/<site>/run1 [more run dirs ...] --label row-date

Every candidate a recording harvested is judged three times with the same judge: as recorded (`off`), with
the change under test applied (`on`), and as recorded again (`off2`), so the change is read against the
judge's own run-to-run noise. `shipped()` rebuilds what the crawl now sends for a recorded candidate (it replays the harvest script over the
recorded HTML in a browser, so a changed dated-line rule or a date handed down from a listing row shows up exactly
as a crawl would see it), and `change()` applies the change under test on top; edit `change()` for a new one.

Scored at the accepted band against the site's ground truth (eval/ground_truth), per site and pooled, with
the mean score shift of each arm against `off`. Jev's verdict bands were calibrated on its shipped input:
a change that moves every score, even upward, needs this before it ships.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

from docseek.jev_crawl import _HARVEST_JS, date_of
from docseek.judge import make_judge, verdict_for
from eval.run_baseline import identity, load_ground_truth

_REPORTS = Path(__file__).resolve().parent / 'reports'
_LISTING_KINDS = ('seed', 'document_listing', 'category_or_overview')


def load_run(run_dir: Path) -> dict:
    run = json.loads((run_dir / 'run.json').read_text())
    pages = []
    for n in range(1, 10_000):
        meta = run_dir / f'harvest_{n:03d}.json'
        if not meta.exists():
            break
        pages.append({**json.loads(meta.read_text()), 'html': (run_dir / f'harvest_{n:03d}.html').read_text()})
    kinds = {k['url']: k['kind'] for k in run.get('page_kinds', [])}
    return {'site': run['site'], 'goal': run['goal'], 'pages': pages, 'kinds': kinds}


# the dated-line rule before 2026-10-04, replayed beside the shipped one so a changed rule is counted against
# itself and not against a recording made on a live page with its stylesheets
_OLD_DATED = (r'/\b20\d\d\. ?(?:\d{1,2}\.|[a-záéíóöőúüű]+ \d{1,2}\.)|\b\d{1,2}[./]\d{1,2}[./]20\d\d\b'
              r'|\b20\d\d-\d\d-\d\d\b/')
_NEW_DATED = _HARVEST_JS[_HARVEST_JS.index('.match(/(?<!') + len('.match('):_HARVEST_JS.index('/);', _HARVEST_JS.index('.match(/(?<!')) + 1]
_OLD_HARVEST_JS = _HARVEST_JS.replace(_NEW_DATED, _OLD_DATED)
assert _OLD_HARVEST_JS != _HARVEST_JS


def replay_harvest(browser, url: str, html: str) -> tuple[dict[str, str], dict[str, str]]:
    """The dated line of every link on a recorded page, by href: as the shipped harvest script reads it now,
    and as the old rule read it."""
    page = browser.new_page()
    try:
        page.route('**/*', lambda route: route.fulfill(body=html, content_type='text/html; charset=utf-8')
                   if route.request.url == url else route.abort())
        page.goto(url, wait_until='domcontentloaded', timeout=15_000)
        return ({link['href']: link.get('dated', '') for link in page.evaluate(_HARVEST_JS)},
                {link['href']: link.get('dated', '') for link in page.evaluate(_OLD_HARVEST_JS)})
    except Exception as exc:  # noqa: BLE001 - a page that will not load again keeps its recorded dates
        print(f'  replay failed for {url}: {exc}', file=sys.stderr)
        return {}, {}
    finally:
        page.close()


_GENERIC_LINK = {'download', 'downloads', 'file', 'files', 'document', 'documents', 'doc', 'view', 'open', 'pdf', 'show',
                 'get', 'dl', 'attachment', 'link', 'here', 'letöltés', 'megnyitás', 'megtekintés', 'lejupielādēt', 'skatīt',
                 'herunterladen', 'ansehen', 'öffnen', 'descargar', 'ver', 'pobierz', 'télécharger', 'télécharger le pdf'}
_FILE_NAME = re.compile(r'(?<![\w/])([^\s()]+\.(?:pdf|xlsx?|docx?|pptx?|zip|xhtml|csv))(?=[\s(]|$)', re.IGNORECASE)


def shipped(d: dict, when: str | None, own: str) -> dict:
    """What the crawl now sends for a recorded candidate: the listing row's date, where the page was linked from a
    dated row and the document shows no date of its own (shipped 2026-10-05)."""
    out = dict(d)
    if when and not own:
        out['context'] = f"{d.get('context', '')} - published {when}".lstrip(' -')
    return out


def change(d: dict) -> dict:
    """The change under test, applied on top of `shipped()`. Identity when nothing is under test.

    Tried and not shipped (2026-10-05): naming a link that says only "Download" by the file name beside it. Alone it
    cost csri 12 -> 2 true accepts, because a name with digits counts as strong and the row text - and with it the
    row's date - is then not sent; with the row kept it was 11 -> 9, within noise. The `_GENERIC_LINK` and
    `_FILE_NAME` rules above are what that attempt used.
    """
    return d


def treat(run: dict, browser) -> tuple[list[dict], dict]:
    """The recorded candidates, each as the crawl now sends it (`off`) and with the change under test on top (`on`)."""
    dated_now: dict[str, dict[str, str]] = {}      # page url -> href -> dated line, by the current script
    dated_old: dict[str, dict[str, str]] = {}      # ...and by the old rule, replayed the same way
    for p in run['pages']:
        dated_now[p['url']], dated_old[p['url']] = replay_harvest(browser, p['url'], p['html'])
    # the row's date for each page: from the first recorded page that linked to it
    row_date: dict[str, str] = {}
    for p in run['pages']:
        for link in p['page_links']:
            when = dated_now.get(p['url'], {}).get(link['url'], link.get('dated', ''))
            if when and link['url'] not in row_date:
                row_date[link['url']] = when
    candidates, stats = [], {'documents': 0, 'dated_changed': 0, 'treated': 0}
    for p in run['pages']:
        kind = run['kinds'].get(p['url'], 'seed' if p.get('depth') == 0 else 'other')
        when = date_of(row_date.get(p['url'], '')) if kind not in _LISTING_KINDS else None
        for d in p['documents']:
            stats['documents'] += 1
            own = dated_now.get(p['url'], {}).get(d['url'], d.get('dated', ''))
            stats['dated_changed'] += own != dated_old.get(p['url'], {}).get(d['url'], d.get('dated', ''))
            off = shipped(d, when, own)
            on = change(off)
            stats['treated'] += on != off
            candidates.append({'page': p['url'], 'off': off, 'on': on})
    return candidates, stats


def score(judge, goal: str, candidates: list[dict], arm: str) -> list[float | None]:
    by_page: dict[str, list[dict]] = {}
    for c in candidates:
        by_page.setdefault(c['page'], []).append(c)
    scores: dict[int, float | None] = {}
    for page_url, group in by_page.items():
        links = [{'url': c[arm]['url'], 'name': c[arm].get('name', ''), 'context': c[arm].get('context', ''),
                  'section': c[arm].get('section', ''), 'column': c[arm].get('column', '')} for c in group]
        for c, s in zip(group, judge.relevance(goal, page_url, links)):
            scores[id(c)] = s
    return [scores[id(c)] for c in candidates]


def band(scores: list[float | None], labels: list[bool]) -> dict:
    accepted = [verdict_for(s) == 'accepted' for s in scores]
    tp = sum(a and l for a, l in zip(accepted, labels))
    fp = sum(a and not l for a, l in zip(accepted, labels))
    pos = sum(labels)
    return {'tp': tp, 'fp': fp, 'positives': pos, 'P': round(tp / (tp + fp), 3) if tp + fp else None,
            'R': round(tp / pos, 3) if pos else None}


def main() -> None:
    ap = argparse.ArgumentParser(prog='eval.judge_ab')
    ap.add_argument('runs', nargs='+', help='snapshot run directories (eval/snapshots/<stamp>/<site>/run<n>)')
    ap.add_argument('--judge', choices=['jev', 'laya', 'llm'], default='jev')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=False)
    except Exception:
        pass
    from playwright.sync_api import sync_playwright

    truth = {e['site']: e for e in load_ground_truth(None)}
    report: dict = {'arms': {}, 'sites': {}, 'runs': args.runs}
    pooled = {arm: {'scores': [], 'labels': [], 'sites': []} for arm in ('off', 'on', 'off2')}
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for run_dir in args.runs:
            run = load_run(Path(run_dir))
            entry = truth.get(run['site'])
            if not entry:
                print(f'no ground truth for {run["site"]}: skipped', file=sys.stderr)
                continue
            candidates, stats = treat(run, browser)
            labels = [identity(c['off']['url'], entry.get('keep_query', False), entry.get('identity_re')) in entry['expected']
                      for c in candidates]
            judge = make_judge(args.judge, entry.get('profile'))
            print(f'=== {run["site"]}: {len(candidates)} candidates, {sum(labels)} labelled, '
                  f'{stats["treated"]} treated, dated line changed on {stats["dated_changed"]}', file=sys.stderr, flush=True)
            site = {'stats': stats, 'labelled': sum(labels)}
            scores = {}
            for arm, source in (('off', 'off'), ('on', 'on'), ('off2', 'off')):
                scores[arm] = score(judge, run['goal'], candidates, source)
                site[arm] = band(scores[arm], labels)
                pooled[arm]['scores'] += scores[arm]
                pooled[arm]['labels'] += labels
                print(f'    {arm:5} {site[arm]}', file=sys.stderr, flush=True)
            treated = [i for i, c in enumerate(candidates) if c['on'] is not c['off'] and c['on'] != c['off']]
            site['treated_rows'] = [{'url': candidates[i]['off']['url'], 'label': labels[i],
                                     'off': scores['off'][i], 'on': scores['on'][i], 'off2': scores['off2'][i]} for i in treated]
            report['sites'][run['site']] = site
        browser.close()

    def shift(a: str, b: str) -> float | None:
        pairs = [(x, y) for x, y in zip(pooled[a]['scores'], pooled[b]['scores']) if x is not None and y is not None]
        return round(statistics.mean(y - x for x, y in pairs), 4) if pairs else None

    for arm in ('off', 'on', 'off2'):
        report['arms'][arm] = {**band(pooled[arm]['scores'], pooled[arm]['labels']), 'shift_vs_off': shift('off', arm)}
        print(f'pooled {arm:5} {report["arms"][arm]}')
    stamp = time.strftime('%Y%m%d_%H%M%S') + (f'_{args.label}' if args.label else '')
    _REPORTS.mkdir(exist_ok=True)
    out = _REPORTS / f'judge_ab_{stamp}.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
