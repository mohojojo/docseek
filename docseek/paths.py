"""Goal-to-path ranking: which of a site's many pages are worth the judge's question first.

A big sitemap lists thousands of pages, and a crawl can afford to ask the judge about a fraction of them. Asked
once, with the goal and a sample of the site's own path prefixes, a model names the prefixes that lead to what the
goal asks for and those that do not. The answer only orders pages: nothing is dropped on its account, because a
site's documents sometimes sit under a path no one would guess. The plan is kept per site in the patterns dir,
so a repeated goal costs no call.
"""
from __future__ import annotations

import collections
import logging
from urllib.parse import unquote, urlparse

from .llm import LLMClient

logger = logging.getLogger(__name__)

SAMPLE_PREFIXES = 80        # path prefixes shown to the model, the most populated first
MAX_PATTERNS = 12
MIN_PAGES_TO_PLAN = 50      # under this a crawl scores every sitemap page anyway: no call

_SYSTEM = """You rank the sections of a website for a document-finding goal.
You get the goal (in any language) and the site's own path prefixes with how many pages each holds.
Return ONLY a JSON object, no prose: {"prefer": [...], "skip": [...]}.
`prefer`: path substrings (as given, lowercase) whose pages are where the goal's documents or their listings
live. `skip`: path substrings whose pages cannot hold them (news, careers, contact, products of another kind).
Up to 12 each; only substrings that appear in the list; leave a list empty when unsure."""


def path_sample(urls: list[str], limit: int = SAMPLE_PREFIXES) -> list[tuple[str, int]]:
    """The site's path prefixes with page counts, the most populated first: every first segment, and a second
    segment when it is a section of its own (two pages or more), not one page's slug under /products/."""
    sections: collections.Counter[str] = collections.Counter()
    subsections: collections.Counter[str] = collections.Counter()
    for url in urls:
        segments = [s for s in unquote(urlparse(url).path).lower().split('/') if s]
        if segments:
            sections[f'/{segments[0]}/'] += 1
        if len(segments) > 1:
            subsections[f'/{segments[0]}/{segments[1]}/'] += 1
    counts = sections + collections.Counter({k: n for k, n in subsections.items() if n > 1})
    return counts.most_common(limit)


def plan_paths(llm: LLMClient, goal: str, urls: list[str]) -> dict[str, list[str]]:
    """{prefer, skip}: path substrings the model picked for `goal` from the site's own prefixes. Empty lists when
    the site is small, the model fails, or it names nothing that is on the site."""
    sample = path_sample(urls)
    if len(urls) < MIN_PAGES_TO_PLAN or not sample:
        return {'prefer': [], 'skip': []}
    listing = '\n'.join(f'{count:6}  {prefix}' for prefix, count in sample)
    try:
        answer = llm.complete_json(_SYSTEM, f'Goal: {goal}\n\nPath prefixes (pages):\n{listing}')
    except Exception as exc:  # noqa: BLE001 - a plan is a convenience; the crawl runs without one
        logger.warning('[paths] no path plan: %s', exc)
        return {'prefer': [], 'skip': []}
    known = {prefix for prefix, _ in sample}

    def clean(key: str) -> list[str]:
        values = answer.get(key) if isinstance(answer, dict) else None
        picked = [v.strip().lower() for v in (values or []) if isinstance(v, str) and v.strip()]
        return [v for v in picked if any(v in prefix for prefix in known)][:MAX_PATTERNS]
    prefer = clean('prefer')
    skip = [v for v in clean('skip') if v not in prefer]
    return {'prefer': prefer, 'skip': skip}


def is_planned(url: str, prefer: list[str]) -> bool:
    """Whether `url` sits under a path the plan prefers."""
    path = unquote(urlparse(url).path).lower()
    return any(p in path for p in prefer)


def rank_by_paths(urls: list[str], prefer: list[str], skip: list[str], years: frozenset[str] = frozenset()) -> list[str]:
    """`urls` with the preferred paths first - in the order the plan named them, the first being the model's
    best guess - then those naming a goal year, then the rest, the skipped ones last; within a rank, the
    sitemap's order. Nothing is dropped."""
    def rank(url: str) -> tuple[int, int]:
        path = unquote(urlparse(url).path).lower()
        hits = [i for i, p in enumerate(prefer) if p in path]
        if hits:
            return 0, hits[0]
        if any(y in path for y in years):
            return 0, len(prefer)
        return (2, 0) if any(s in path for s in skip) else (1, 0)
    return sorted(urls, key=rank)       # sorted is stable: ties keep the sitemap's order
