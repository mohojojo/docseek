from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread

from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import anthropic as _anthropic

from .agent import agentic_crawl, search_sites
from .judge import JudgeUnavailable, RelevanceJudge, make_judge
from .llm import LLMClient, make_llm
from .profile import UnknownProfile, available_profiles, load_profile
from .jev_crawl import jev_crawl
from .models import SearchSitesResponse
from .patterns import PatternStore
from .scraper import _DEFAULT_USER_AGENT

PATTERNS_DIR = os.environ.get('PATTERNS_DIR')

_WEB_SEARCH_COMPATIBLE_MODELS: frozenset[str] = frozenset({
    'claude-haiku-4-5-20251001',
    'claude-sonnet-4-5',
    'claude-opus-4-7',
})

app = FastAPI(title='crawler', version='0.2.0')


class DiscoverRequest(BaseModel):
    url: str
    goal: str = Field(
        default='Find and download relevant documents',
        min_length=1,
        max_length=2000,
    )
    max_pages: int = Field(default=10, ge=1, le=50)
    max_depth: int = Field(default=3, ge=0, le=10)
    same_domain_only: bool = True
    js_wait_ms: int = Field(default=2000, ge=0, le=15000)
    click_wait_ms: int = Field(default=2000, ge=0, le=15000)
    max_click_rounds: int = Field(default=3, ge=0, le=20, description='Max tool steps per page before auto-done')
    max_concurrent: int = Field(default=1, ge=1, le=10, description='Max pages visited in parallel')
    headless: bool = Field(
        default_factory=lambda: os.environ.get('CRAWLER_HEADLESS', 'true').lower() != 'false',
        description='Run browser headless; set false to watch visually. Default overridden by CRAWLER_HEADLESS env var.',
    )
    pre_interactions: list[dict] = Field(
        default_factory=list,
        description=(
            'Steps to run on every page after cookie/modal dismissal, before the agent loop. '
            'Use to fill listing-page filter forms. '
            'Each step: {"action": "select"|"click"|"wait", "selector": "...", "value": "...", "label": "...", "ms": N}'
        ),
    )
    include_screenshot_on_load: bool = Field(
        default=False,
        description='Attach a viewport screenshot to the initial page context for each page visit.',
    )
    model: str | None = Field(
        default=None,
        description='The agent model (escalations and the agent path). Default: LLM_MODEL, or claude-haiku-4-5 '
                    'on Anthropic.',
    )
    user_agent: str = _DEFAULT_USER_AGENT
    enable_learning: bool = True
    decision_layer: Literal['agent', 'jev'] | None = Field(
        default=None,
        description=(
            "How the crawl decides. 'jev' is the judge-driven crawl: a Relevance judge scores every Candidate, "
            "ranks the frontier and triggers agent escalations ('judge' says which one). 'agent' is the Claude "
            'tool loop. Omit it to use the judge-driven crawl whenever a judge can run, the agent path otherwise.'
        ),
    )
    judge: Literal['jev', 'llm'] | None = Field(
        default=None,
        description=(
            "The Relevance judge of the judge-driven crawl. 'jev' is TypeSafe Jev (needs TYPESAFE_API_KEY), with "
            "the LLM judge taking over if Jev becomes unavailable mid-crawl; 'llm' is any model configured by "
            'LLM_PROVIDER / LLM_MODEL / LLM_BASE_URL / LLM_API_KEY. Omit it for Jev when its key is set and the '
            'LLM judge otherwise. A judge asked for by name that cannot run is an error, not a silent downgrade.'
        ),
    )
    profile: str = Field(
        default='generic',
        description=(
            "The domain profile the judge words its questions with: 'generic' for any goal, or a bundled "
            "domain profile such as 'fund-reports' (periodic fund documents)."
        ),
    )
    allowed_hosts: list[str] = Field(
        default_factory=list,
        description=(
            'Hosts the jev layer may crawl besides the seed host, even without a link from it. '
            'Ignored when same_domain_only is true.'
        ),
    )
    include_rejected: bool = Field(
        default=False,
        description=(
            'Return rejected Candidates as rows as well as in rejected_count, each with its relevance '
            'score. Use it to see what the relevance model threw away and why.'
        ),
    )
    frontier_policy: Literal['tier', 'rescue'] = Field(
        default='tier',
        description=(
            "How the jev layer orders the pages it visits. 'tier' ranks by page kind. 'rescue' does the same "
            'until seven pages in a row accept nothing, then picks pages whose sibling links have paid, and '
            'keeps going while an unvisited fund page or listing is queued. Use it for a site where the '
            "crawl stops early with 'no_progress': it tends to find more documents there, at the cost of "
            'more pages, and can miss a document on sites the default order already covers.'
        ),
    )
    max_seconds: float = Field(default=180.0, ge=10, le=3600,
                               description='Wall-clock budget for the jev decision layer, sitemap scoring included.')


def _require_api_key(x_api_key: str | None) -> None:
    configured = os.environ.get('CRAWLER_API_KEY')
    if not configured:
        return
    if not x_api_key or x_api_key != configured:
        raise HTTPException(status_code=401, detail='Unauthorized')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/v1/profiles')
def profiles(x_api_key: str | None = Header(default=None)) -> dict:
    """The bundled domain profiles a request's `profile` may name."""
    _require_api_key(x_api_key)
    return {'profiles': [{'name': name, 'description': load_profile(name).description}
                         for name in available_profiles()]}


def _resolve_layer(payload: DiscoverRequest) -> tuple[str, RelevanceJudge | None]:
    """Pick the decision layer: the judge-driven crawl by default, the agent path when no judge can run.

    A judge-driven crawl costs a fraction of the agent path, so it is the default. A caller that names the
    judge-driven layer or a judge explicitly gets an error when it cannot run rather than a quiet, much more
    expensive downgrade.
    """
    if payload.decision_layer == 'agent':
        return 'agent', None
    try:
        profile = load_profile(payload.profile)
    except UnknownProfile as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    wanted = payload.judge or ('jev' if payload.decision_layer == 'jev' else None)
    try:
        return 'jev', make_judge(wanted, profile)
    except JudgeUnavailable as exc:
        if payload.decision_layer == 'jev' or payload.judge:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.warning('[crawler] no Relevance judge can run (%s) - falling back to the agent path', exc)
        return 'agent', None


def _agent_llm(payload: DiscoverRequest) -> LLMClient:
    """The LLM the agent runs on (docseek.llm), or a 503 when none is configured."""
    try:
        llm = make_llm(model=payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if llm is None:
        raise HTTPException(status_code=503, detail='No LLM configured: set LLM_PROVIDER / LLM_MODEL / LLM_API_KEY '
                                                    '(or ANTHROPIC_API_KEY)')
    return llm


def _run_crawl(payload: DiscoverRequest, llm: LLMClient, on_event=None):
    """Dispatch to the decision layer the request asked for, or the default."""
    layer, judge = _resolve_layer(payload)
    if layer == 'jev':
        return jev_crawl(
            payload.url, payload.goal, judge=judge, agent=llm,
            same_domain_only=payload.same_domain_only, allowed_hosts=payload.allowed_hosts,
            include_rejected=payload.include_rejected,
            max_pages=payload.max_pages, max_seconds=payload.max_seconds, max_depth=payload.max_depth,
            parallel_pages=payload.max_concurrent, frontier_policy=payload.frontier_policy,
            recipes_dir=PATTERNS_DIR if payload.enable_learning else None,
            user_agent=payload.user_agent, headless=payload.headless, on_event=on_event,
        )
    return agentic_crawl(
        payload.url,
        payload.goal,
        llm=llm,
        max_pages=payload.max_pages,
        max_depth=payload.max_depth,
        same_domain_only=payload.same_domain_only,
        user_agent=payload.user_agent,
        js_wait_ms=payload.js_wait_ms,
        click_wait_ms=payload.click_wait_ms,
        max_tool_steps=payload.max_click_rounds * 30 if payload.max_click_rounds else 100,
        max_concurrent=payload.max_concurrent,
        headless=payload.headless,
        pre_interactions=payload.pre_interactions or None,
        include_screenshot_on_load=payload.include_screenshot_on_load,
        on_event=on_event,
        patterns_dir=PATTERNS_DIR if payload.enable_learning else None,
        enable_learning=payload.enable_learning,
    )


@app.post('/v1/discover')
def discover(payload: DiscoverRequest, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    result = _run_crawl(payload, _agent_llm(payload))
    return result.model_dump()


class SearchSitesRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=20)
    auto_select: bool = False
    model: str = 'claude-haiku-4-5-20251001'


@app.post('/v1/search-sites')
def search_sites_endpoint(
    payload: SearchSitesRequest,
    x_api_key: str | None = Header(default=None),
) -> dict:
    _require_api_key(x_api_key)
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise HTTPException(status_code=503, detail='ANTHROPIC_API_KEY is missing')
    if payload.model not in _WEB_SEARCH_COMPATIBLE_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f'Model "{payload.model}" does not support web search. '
                f'Use one of: {sorted(_WEB_SEARCH_COMPATIBLE_MODELS)}'
            ),
        )
    try:
        client = _anthropic.Anthropic(api_key=api_key)
        results = search_sites(client, payload.model, payload.goal, payload.max_results)
    except _anthropic.APIError as exc:
        raise HTTPException(status_code=503, detail=f'Anthropic API error: {exc}') from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=f'Response parsing failed: {exc}') from exc

    if payload.auto_select:
        results = results[:1]

    return SearchSitesResponse(results=results).model_dump()


@app.post('/v1/discover-stream')
def discover_stream(
    payload: DiscoverRequest, x_api_key: str | None = Header(default=None)
) -> StreamingResponse:
    _require_api_key(x_api_key)
    llm = _agent_llm(payload)

    events: Queue[dict] = Queue()
    done_sentinel = object()

    def run_job() -> None:
        try:
            config_event = {
                'type': 'config',
                'url': payload.url,
                'goal': payload.goal,
                'max_pages': payload.max_pages,
                'max_depth': payload.max_depth,
                'model': llm.model,
                'decision_layer': payload.decision_layer or 'default (jev when configured)',
            }
            logger.debug('[docseek] discover-stream config: %s', config_event)
            events.put(config_event)

            result = _run_crawl(
                payload, llm,
                on_event=lambda ev: None if ev.get('type') == 'page_elements' else events.put(ev),
            )
            events.put({'type': 'result', 'payload': result.model_dump()})
        except Exception as exc:  # noqa: BLE001
            events.put({'type': 'error', 'message': str(exc)})
        finally:
            events.put(done_sentinel)  # type: ignore[arg-type]

    Thread(target=run_job, daemon=True).start()

    def event_stream() -> Generator[str, None, None]:
        while True:
            try:
                item = events.get(timeout=1)
            except Empty:
                heartbeat = {'type': 'heartbeat', 'ts': int(time.time())}
                yield f'event: message\ndata: {json.dumps(heartbeat)}\n\n'
                continue
            if item is done_sentinel:
                yield 'event: done\ndata: {}\n\n'
                break
            payload_text = json.dumps(item, ensure_ascii=False)
            yield f'event: message\ndata: {payload_text}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


def _require_patterns_dir() -> PatternStore:
    if not PATTERNS_DIR:
        raise HTTPException(status_code=404, detail='Pattern learning not configured (PATTERNS_DIR not set)')
    return PatternStore(PATTERNS_DIR)


@app.get('/v1/patterns')
def list_patterns(x_api_key: str | None = Header(default=None)) -> list[dict]:
    _require_api_key(x_api_key)
    store = _require_patterns_dir()
    return [p.model_dump() for p in store.list_all()]


@app.get('/v1/patterns/{domain}')
def get_patterns(domain: str, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    store = _require_patterns_dir()
    patterns = store.load(domain)
    if not patterns:
        raise HTTPException(status_code=404, detail=f'No patterns found for domain: {domain}')
    return patterns.model_dump()


@app.delete('/v1/patterns/{domain}')
def delete_patterns(domain: str, x_api_key: str | None = Header(default=None)) -> dict:
    _require_api_key(x_api_key)
    store = _require_patterns_dir()
    deleted = store.delete(domain)
    if not deleted:
        raise HTTPException(status_code=404, detail=f'No patterns found for domain: {domain}')
    return {'deleted': True, 'domain': domain}
