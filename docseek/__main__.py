"""Command line.

  docseek URL GOAL [options]        the same crawl as POST /v1/discover, printed as JSON (or CSV with --format csv)
  docseek generate URL GOAL         write the site's discovery program (needs PROGRAMS_DIR and a coding model)
  docseek check [KEY ...]           replay programs and compare them with their snapshots (no model)

Progress goes to stderr.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time


def _programs_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--programs-dir', default=os.environ.get('PROGRAMS_DIR'),
                        help='Where generated programs live (default: PROGRAMS_DIR)')


def _judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--profile', default='generic', help='Domain profile name or path (default: generic)')
    parser.add_argument('--judge', choices=('jev', 'laya', 'llm'), default=None,
                        help='Relevance judge (default: jev when TYPESAFE_API_KEY is set, else llm)')


def write_csv(downloads: list, out) -> None:
    """One row per document, one column per field of AgenticDownload, in its order."""
    from .models import AgenticDownload

    writer = csv.DictWriter(out, fieldnames=list(AgenticDownload.model_fields), lineterminator='\n')
    writer.writeheader()
    for download in downloads:
        writer.writerow(download.model_dump())


def _fail(message: str) -> None:
    print(f'docseek: {message}', file=sys.stderr)
    sys.exit(1)


def _discover(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='docseek', description='Find the documents a goal asks for on a website.',
                                     epilog='Also: docseek generate URL GOAL, docseek check [KEY ...]')
    parser.add_argument('url', help='Start URL for the crawl')
    parser.add_argument('goal', help='What to find, in plain language')
    _judge_args(parser)
    parser.add_argument('--decision-layer', choices=('jev', 'agent'), default=None,
                        help='jev: the judge-driven crawl (with any judge); agent: the browsing agent alone')
    parser.add_argument('--model', default=None, help='LLM model override (default: LLM_MODEL)')
    parser.add_argument('--max-pages', type=int, default=10)
    parser.add_argument('--max-seconds', type=float, default=180)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--include-rejected', action='store_true', help='Also print rejected candidates')
    parser.add_argument('--latest', action='store_true', help='Keep only the newest document of each series')
    parser.add_argument('--format', choices=('json', 'csv'), default='json',
                        help='json: the whole result; csv: one row per document (default: json)')
    parser.add_argument('--programs', action=argparse.BooleanOptionalAction, default=None,
                        help="Answer from the site's generated program when it is healthy; after a crawl, write one "
                             '(default: on when a programs dir is set; --no-programs forces a crawl)')
    _programs_dir_arg(parser)
    args = parser.parse_args(argv)

    from fastapi import HTTPException

    from . import server

    server.PROGRAMS_DIR = args.programs_dir

    def on_event(ev: dict) -> None:
        if ev.get('type') == 'crawl_page_start':
            print(f'  -> {ev.get("url", "")}', file=sys.stderr)
        elif ev.get('type') == 'agent_download':
            print(f'     found: {ev.get("url", "")}', file=sys.stderr)

    payload = server.DiscoverRequest(
        url=args.url, goal=args.goal, profile=args.profile, judge=args.judge, decision_layer=args.decision_layer,
        model=args.model, max_pages=args.max_pages, max_seconds=args.max_seconds, max_depth=args.max_depth,
        include_rejected=args.include_rejected, latest=args.latest, programs=args.programs,
    )
    try:
        result = server._discover(payload, server._agent_llm(payload), on_event=on_event)
    except HTTPException as exc:
        _fail(exc.detail)
    if args.format == 'csv':
        write_csv(result.downloads, sys.stdout)
    else:
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    if (result.program or {}).get('generation') == 'started':
        print('docseek: writing a program for next time (this takes a few minutes)...', file=sys.stderr)
        while server._generating:
            time.sleep(1)


def _generate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='docseek generate', description="Write a site's discovery program.")
    parser.add_argument('url')
    parser.add_argument('goal')
    _judge_args(parser)
    _programs_dir_arg(parser)
    args = parser.parse_args(argv)
    if not args.programs_dir:
        _fail('set PROGRAMS_DIR or pass --programs-dir')

    from .codegen.explorer import codegen_llm
    from .codegen.programs import ProgramStore, generate_program
    from .judge import JudgeUnavailable, make_judge
    from .profile import UnknownProfile, load_profile

    llm = codegen_llm()
    if llm is None:
        _fail('no coding model configured: set CODEGEN_MODEL (or LLM_MODEL) and the provider settings')
    try:
        judge = make_judge(args.judge, load_profile(args.profile))
    except (JudgeUnavailable, UnknownProfile) as exc:
        _fail(str(exc))
    program, report = generate_program(ProgramStore(args.programs_dir), args.url, args.goal, llm=llm, judge=judge)
    if program is None:
        _fail(f"the agent produced no program ({report.get('stopped') or report.get('notes')}, "
              f"{report.get('turns')} turns)")
    meta = program.meta
    print(json.dumps({'key': program.key, **{k: meta.get(k) for k in (
        'submitted', 'turns', 'model', 'usage', 'seconds', 'generated_kept', 'data_hosts', 'post_endpoints', 'notes')}},
        indent=2, ensure_ascii=False))


def _check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog='docseek check',
                                     description='Replay generated programs and compare them with their snapshots.')
    parser.add_argument('keys', nargs='*', help='Program keys (default: every program)')
    _programs_dir_arg(parser)
    args = parser.parse_args(argv)
    if not args.programs_dir:
        _fail('set PROGRAMS_DIR or pass --programs-dir')

    from .codegen.programs import ProgramStore, check_drift

    rows = check_drift(ProgramStore(args.programs_dir), args.keys or None)
    for row in rows:
        detail = (f"kept {row['kept']:.0%}  {row['then']} -> {row['now']}" if 'kept' in row
                  else f"{row['now']} documents")
        print(f"{row['key']:45} {row['status']:8} {detail}" + (f"  {row['error'].splitlines()[-1][:80]}"
                                                               if row.get('error') else ''), file=sys.stderr)
    print(json.dumps(rows, indent=2))
    if any(row['status'] in ('broken', 'shrank') for row in rows):
        sys.exit(2)


def main() -> None:
    argv = sys.argv[1:]
    commands = {'generate': _generate, 'check': _check}
    if argv and argv[0] in commands:
        commands[argv[0]](argv[1:])
    else:
        _discover(argv)


if __name__ == '__main__':
    main()
