"""Command line: the same crawl as POST /v1/discover, printed as JSON; progress goes to stderr."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog='docseek', description='Find the documents a goal asks for on a website.')
    parser.add_argument('url', help='Start URL for the crawl')
    parser.add_argument('goal', help='What to find, in plain language')
    parser.add_argument('--profile', default='generic', help='Domain profile name or path (default: generic)')
    parser.add_argument('--judge', choices=('jev', 'llm'), default=None,
                        help='Relevance judge (default: jev when TYPESAFE_API_KEY is set, else llm)')
    parser.add_argument('--decision-layer', choices=('jev', 'agent'), default=None,
                        help='jev: the judge-driven crawl (with any judge); agent: the browsing agent alone')
    parser.add_argument('--model', default=None, help='LLM model override (default: LLM_MODEL)')
    parser.add_argument('--max-pages', type=int, default=10)
    parser.add_argument('--max-seconds', type=float, default=180)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--include-rejected', action='store_true', help='Also print rejected candidates')
    args = parser.parse_args()

    from fastapi import HTTPException

    from .server import DiscoverRequest, _agent_llm, _run_crawl

    def on_event(ev: dict) -> None:
        if ev.get('type') == 'crawl_page_start':
            print(f'  -> {ev.get("url", "")}', file=sys.stderr)
        elif ev.get('type') == 'agent_download':
            print(f'     found: {ev.get("url", "")}', file=sys.stderr)

    payload = DiscoverRequest(
        url=args.url, goal=args.goal, profile=args.profile, judge=args.judge, decision_layer=args.decision_layer,
        model=args.model, max_pages=args.max_pages, max_seconds=args.max_seconds, max_depth=args.max_depth,
        include_rejected=args.include_rejected,
    )
    try:
        result = _run_crawl(payload, _agent_llm(payload), on_event=on_event)
    except HTTPException as exc:
        print(f'docseek: {exc.detail}', file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
