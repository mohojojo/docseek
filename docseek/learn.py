from __future__ import annotations

import json
import logging

from .llm import LLMClient
from .models import AgentStep, CrawlPlan
from .patterns import DomainPatterns, GateSequence, GateStep, PatternStore

logger = logging.getLogger(__name__)

_GATE_EXTRACT_SYSTEM = """\
You are a structured data extractor. Given a sequence of web interaction steps, extract:
1. Gate sequences (investor-type gates, country selectors, language pickers, consent modals)
2. A navigation hint describing how documents are found on this site

Respond with JSON only:
{
  "gate_sequences": [
    {
      "description": "short description of what this gate does",
      "steps": [
        {"tool": "select_option", "hint": "selector hint or label", "value": "the value selected"},
        {"tool": "click", "hint": "button label or hint"}
      ]
    }
  ],
  "navigation_hints": "1-2 sentences about navigation strategy and where documents are found."
}

If no gates were encountered, return {"gate_sequences": [], "navigation_hints": "..."}."""

_GATE_TOOLS = {'select_option', 'fill_input', 'click'}


def extract_patterns(
    domain: str,
    steps: list[AgentStep],
    final_memory: dict[str, str],
    crawl_plan: CrawlPlan,
    llm: LLMClient,
    goal: str = '',
) -> DomainPatterns:
    """Build a DomainPatterns from a completed crawl run."""
    visited_urls = [s.source_url for s in steps if s.source_url]
    confirmed_prefer = [p for p in crawl_plan.url_patterns_prefer if any(p in u for u in visited_urls)]

    gate_sequences: list[GateSequence] = []
    navigation_hints = ''
    gate_steps = [s for s in steps if s.tool in _GATE_TOOLS]

    if gate_steps:
        step_text = '\n'.join(
            f'{i + 1}. tool={s.tool} args={json.dumps(s.args)} reason={s.reason!r}'
            for i, s in enumerate(gate_steps)
        )
        try:
            data = llm.complete_json(_GATE_EXTRACT_SYSTEM, f'Interaction steps:\n{step_text}')
            if data:
                gate_sequences = [
                    GateSequence(
                        description=g.get('description', ''),
                        steps=[GateStep(**s) for s in g.get('steps', [])],
                    )
                    for g in data.get('gate_sequences', [])
                ]
                navigation_hints = data.get('navigation_hints', '')
        except Exception as exc:
            logger.warning('Gate extraction LLM call failed: %s', exc)

    return DomainPatterns(
        domain=domain,
        last_goal=goal,
        url_patterns_prefer=confirmed_prefer,
        url_patterns_skip=list(crawl_plan.url_patterns_skip),
        gate_sequences=gate_sequences,
        memory_snapshot=dict(final_memory),
        navigation_hints=navigation_hints,
    )


def extract_and_save(
    domain: str,
    steps: list[AgentStep],
    final_memory: dict[str, str],
    crawl_plan: CrawlPlan,
    store: PatternStore,
    llm: LLMClient,
    goal: str = '',
) -> None:
    """Extract patterns from a completed crawl and persist them. Swallows all exceptions."""
    print(f'[patterns] extracting patterns for {domain}...', flush=True)
    try:
        patterns = extract_patterns(domain, steps, final_memory, crawl_plan, llm, goal=goal)
        store.merge_and_save(domain, patterns)
        print(f'[patterns] saved → {domain}.json  (crawl #{patterns.successful_crawl_count})', flush=True)
    except Exception as exc:
        print(f'[patterns] extraction failed for {domain}: {exc}', flush=True)
