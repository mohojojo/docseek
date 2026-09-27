"""Offline: score a Relevance judge on a frozen, labelled candidate set - no crawling, no browser.

    .venv/bin/python -m eval.judge_compare <set.json> --judge jev
    .venv/bin/python -m eval.judge_compare <set.json> --judge llm --profile fund-reports [--model claude-haiku-4-5]

A set is a JSON list of candidates: {site, goal, batch (the page), context, text, url, row_text, section,
column, label (1 = a document the goal asks for)}. Candidates of one page are judged together, as in a crawl.
`--profile` (a profile name or a profile path, default 'generic') sets the wording every goal is judged with.
The set format is documented in eval/ground_truth/README.md.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from docseek.judge import LLMJudge, verdict_for
from docseek.llm import make_llm
from docseek.profile import load_profile

REPORTS = Path(__file__).resolve().parent / 'reports'

def make(judge: str, profile_name: str, model: str | None):
    profile = load_profile(profile_name)
    if judge == 'jev':
        from docseek.jev import JevClient
        return JevClient(profile=profile)
    llm = make_llm(model=model)
    if llm is None:
        raise SystemExit('no LLM configured (LLM_PROVIDER / LLM_MODEL / LLM_API_KEY or ANTHROPIC_API_KEY)')
    return LLMJudge(llm, profile)


def main() -> None:
    ap = argparse.ArgumentParser(prog='eval.judge_compare')
    ap.add_argument('set')
    ap.add_argument('--judge', choices=['jev', 'llm'], required=True)
    ap.add_argument('--profile', default='generic', help="'generic', 'fund-reports' or a profile path")
    ap.add_argument('--model', default=None, help='LLM model override for --judge llm')
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / '.env', override=False)
    except Exception:
        pass

    cands = json.loads(Path(args.set).read_text())
    pages: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        pages[(c['site'], c['goal'], c['batch'], c.get('context', ''))].append(c)
    name = args.profile
    judge = make(args.judge, name, args.model)
    judges = {name: judge}
    began = time.monotonic()
    for (site, goal, _, context), group in pages.items():
        links = [{'url': c['url'], 'name': c.get('text', ''), 'context': c.get('row_text', ''),
                  'section': c.get('section', ''), 'column': c.get('column', '')} for c in group]
        for c, score in zip(group, judge.relevance(goal, context, links)):
            c['score'], c['verdict'], c['profile'] = score, verdict_for(score), name

    per_site: dict[str, dict] = {}
    for site in sorted({c['site'] for c in cands}):
        rows = [c for c in cands if c['site'] == site]
        tp = sum(1 for c in rows if c['label'] and c['verdict'] == 'accepted')
        fp = sum(1 for c in rows if not c['label'] and c['verdict'] == 'accepted')
        pos = sum(c['label'] for c in rows)
        per_site[site] = {'P': round(tp / (tp + fp), 3) if tp + fp else None, 'R': round(tp / pos, 3) if pos else None,
                          'tp': tp, 'fp': fp, 'positives': pos, 'candidates': len(rows),
                          'unscored': sum(1 for c in rows if c['verdict'] == 'unscored')}
    tp = sum(s['tp'] for s in per_site.values())
    fp = sum(s['fp'] for s in per_site.values())
    pos = sum(s['positives'] for s in per_site.values())
    pooled = {'P': round(tp / (tp + fp), 3) if tp + fp else None, 'R': round(tp / pos, 3) if pos else None}
    usage = {name: {'model': j.model, 'requests': j.requests,
                    **({'input_tokens': j.llm.usage.input_tokens, 'output_tokens': j.llm.usage.output_tokens}
                       if hasattr(j, 'llm') else {'cost_usd': j.cost_usd})} for name, j in judges.items()}
    for site, s in per_site.items():
        print(f"{site:16} P={s['P']}  R={s['R']}  ({s['tp']} tp, {s['fp']} fp of {s['candidates']}; "
              f"{s['unscored']} unscored)")
    print(f"pooled           P={pooled['P']}  R={pooled['R']}  in {time.monotonic() - began:.0f} s  {usage}")
    stamp = time.strftime('%Y%m%d_%H%M%S') + (f'_{args.label}' if args.label else '')
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f'judge_compare_{stamp}.json'
    out.write_text(json.dumps({'set': args.set, 'judge': args.judge, 'profile': args.profile, 'per_site': per_site,
                               'pooled': pooled, 'usage': usage,
                               'predictions': [{k: c.get(k) for k in ('site', 'url', 'label', 'score', 'verdict', 'profile')}
                                               for c in cands]}, ensure_ascii=False, indent=1))
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
