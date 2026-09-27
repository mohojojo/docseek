"""Eval for generated discovery programs: generate once per site and goal, replay, score exactly like run_jev.

    .venv/bin/python -m eval.run_codegen --sites example.com
    .venv/bin/python -m eval.run_codegen --sites example.com --regenerate --replays 1

Programs are kept in eval/programs/ (or --programs-dir) and reused until --regenerate. Each replay runs the program
in the sandbox with no model, then the judge scores every returned document as a crawl would, so `acc` means the
same as in run_jev. Generation needs a coding model (CODEGEN_MODEL, or LLM_MODEL); replays need only the judge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from eval.run_baseline import load_ground_truth, start_url_for
from eval.run_jev import score_run

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE / 'reports'


def main() -> None:
    ap = argparse.ArgumentParser(prog='eval.run_codegen')
    ap.add_argument('--sites', nargs='*', default=None)
    ap.add_argument('--replays', type=int, default=2)
    ap.add_argument('--regenerate', action='store_true')
    ap.add_argument('--judge', choices=('jev', 'llm'), default=None)
    ap.add_argument('--programs-dir', default=str(_HERE / 'programs'))
    ap.add_argument('--label', default='')
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(_HERE.parent / '.env', override=False)
    except Exception:
        pass

    from docseek.codegen.explorer import codegen_llm
    from docseek.codegen.programs import ProgramStore, generate_program, program_key, run_saved
    from docseek.judge import JudgeUnavailable, make_judge

    store = ProgramStore(args.programs_dir)
    entries = load_ground_truth(args.sites)
    if not entries:
        print('No ground-truth entries matched.', file=sys.stderr)
        sys.exit(1)
    try:
        make_judge(args.judge, entries[0]['profile'])
    except (JudgeUnavailable, ValueError) as exc:
        print(f'No relevance judge: {exc}', file=sys.stderr)
        sys.exit(1)

    stamp = time.strftime('%Y%m%d_%H%M%S') + (f'_{args.label}' if args.label else '')
    _REPORTS.mkdir(exist_ok=True)
    out = _REPORTS / f'codegen_{stamp}.json'
    report = {'replays': args.replays, 'sites': {}}
    for entry in entries:
        site = entry['site']
        start_url = entry.get('start_url') or start_url_for(site)
        key = program_key(start_url, entry['goal'])
        generation = None
        if args.regenerate or store.load(key) is None:
            llm = codegen_llm()
            if llm is None:
                print(f'=== {site}: no program and no coding model (CODEGEN_MODEL / LLM_MODEL) - skipped', file=sys.stderr)
                continue
            print(f'=== {site}: generating', file=sys.stderr, flush=True)
            program = generate_program(store, start_url, entry['goal'], llm=llm,
                                       judge=make_judge(args.judge, entry['profile']))
            if program is None:
                print('    no program produced', file=sys.stderr)
                report['sites'][site] = {'goal': entry['goal'], 'generation': None, 'runs': []}
                continue
            generation = {k: v for k, v in program.meta.items() if k not in ('notes', 'runs')}
            print(f"    {'submitted' if generation['submitted'] else 'NOT submitted'} in {generation['turns']} turns, "
                  f"{generation['seconds']} s, tokens {generation['usage']}", file=sys.stderr)
        program = store.load(key)
        runs = []
        for i in range(args.replays):
            run, downloads = run_saved(program, make_judge(args.judge, entry['profile']))
            scored = score_run(downloads, {}, entry['expected'], entry.get('keep_query', False),
                               entry.get('identity_re'), entry.get('goal_year'))
            scored.update({'returned_documents': len(downloads), 'error': run['error'], 'seconds': run['seconds'],
                           'fetches': run['requests'], 'renders': run['renders'],
                           'fetch_failures': run['fetch_failures']})
            runs.append(scored)
            a = scored['accepted']
            print(f"    replay {i + 1}: acc R={a['recall']:.2f} P={a['precision']:.2f} | returned {len(downloads)}"
                  f" in {run['seconds']} s, {run['requests']} fetches{'  ERROR' if run['error'] else ''}",
                  file=sys.stderr, flush=True)
        report['sites'][site] = {'expected': entry['expected_count'], 'goal': entry['goal'], 'key': key,
                                 'generation': generation, 'runs': runs}
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f'Report: {out}', file=sys.stderr)


if __name__ == '__main__':
    main()
