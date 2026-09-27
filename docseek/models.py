from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ANode(BaseModel):
    role: str
    name: str | None = None
    attributes: dict[str, str] | None = None
    children: list[ANode] = Field(default_factory=list)


ANode.model_rebuild()


class FullElement(BaseModel):
    """Enriched element with HTML attributes, built from the JS extractor output."""
    ml_id: str
    role: str
    name: str
    html_tag: str
    attributes: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    action: Literal['navigate', 'download', 'click'] | None = None
    is_pdf: bool = False


# Registry mapping ml_id → FullElement, built from the scraped page
ElementRegistry = dict[str, FullElement]


class AgentAction(BaseModel):
    """Normalized action for agent tools (kept for scraper layer compatibility)."""
    field: str
    action: Literal['navigate', 'download', 'click']
    url: str | None = None
    name: str
    ml_id: str
    role: str
    is_pdf: bool = False


class AgentStep(BaseModel):
    """One tool call made by the agentic crawl loop."""
    tool: str
    args: dict
    reason: str
    source_url: str | None = None


class AgenticDownload(BaseModel):
    """A Candidate document discovered by a crawl, with its Verdict when one was judged.

    relevance/verdict/source are filled by the Jev decision layer. On the agent path
    they stay at their defaults: no Relevance was judged, so the Verdict is 'unscored'.
    """
    url: str
    name: str
    reason: str
    source_page: str
    relevance: float | None = None
    verdict: Literal['accepted', 'unsure', 'rejected', 'unscored'] = 'unscored'
    source: Literal['page', 'sitemap', 'api', 'agent', 'program'] = 'agent'
    period: str | None = None
    year: str | None = None      # the year of the document's dated line, read by code, a Facet
    series: str | None = None    # the document's identity with its period taken out (docseek.series), a Facet
    latest_in_series: bool | None = None   # newest of its series; None: no period, or rejected


class CrawlPlan(BaseModel):
    """Structured extraction of a crawl goal - built once before the BFS loop."""
    doc_types: list[str] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)
    url_patterns_prefer: list[str] = Field(default_factory=list)
    url_patterns_skip: list[str] = Field(default_factory=list)


class AgenticCrawlResult(BaseModel):
    """Final result of an agentic_crawl() run."""
    start_url: str
    goal: str
    downloads: list[AgenticDownload] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    final_memory: dict[str, str] = Field(default_factory=dict)
    pages_visited: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # the judge's model on the judge-driven crawl; the agent's LLM on the agent path
    decision_model: str = ''
    relevance_model: str | None = None
    rejected_count: int = 0
    superseded_count: int = 0    # with `latest`: older documents of a series left out
    jev_requests: int = 0
    jev_cost_usd: float = 0.0
    escalations: dict[str, int] = Field(default_factory=dict)
    escalations_skipped: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    # URLs refused and page text flagged while crawling (never acted on, only counted)
    guard_counts: dict[str, int] = Field(default_factory=dict)
    # with generated programs on: which path answered and why (docseek.codegen.programs)
    program: dict | None = None


class SearchSiteResult(BaseModel):
    """A candidate website returned by the search-sites endpoint."""
    title: str
    url: str
    snippet: str
    snippet_is_synthesized: bool = True


class SearchSitesResponse(BaseModel):
    """Response from POST /v1/search-sites."""
    results: list[SearchSiteResult] = Field(default_factory=list)
