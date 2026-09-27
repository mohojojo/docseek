from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import unquote, urljoin, urlparse

import anthropic

from .learn import extract_and_save
from .reach import _robots_cache, store_robots  # noqa: F401 - _robots_cache re-exported for callers and tests
from .reach import is_safe_url as _is_safe_url, robots_allows as _url_allowed_by_robots
from .llm import DEFAULT_ANTHROPIC_MODEL, AnthropicLLM, LLMClient, make_llm
from .models import AgentStep, AgenticCrawlResult, AgenticDownload, CrawlPlan, FullElement, SearchSiteResult
from .patterns import DomainPatterns, PatternStore
from .query import _BINARY_EXTENSIONS
from .scraper import _DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)


# File extensions the fast-harvest path recognises as downloadable documents.
_STATIC_DOC_EXTENSIONS = frozenset({
    '.pdf', '.xlsx', '.xls', '.docx', '.doc', '.csv', '.pptx', '.ppt', '.zip',
})

_SYSTEM_PROMPT = """\
You are an autonomous web crawler. Discover and record downloadable documents that match the user's goal.

You interact with pages using tools. Each turn shows you the current page URL, depth, queue size, \
downloads found so far, and the page elements (links, buttons) with their ml_ids.

Tool usage:
- navigate(url, reason): Follow a single link immediately. Current page closes; you visit that URL next.
- queue_urls(urls, reason): Add many URLs to the visit queue. Use on listing pages with many relevant links. \
  You stay on the current page after calling this.
- click(ml_id, reason): Click a button or link in the live browser (for JS-driven content, PDF modals, etc.). \
  The page is re-extracted after clicking.
- record_download(url, name, reason): Record a single document URL matching the goal.
- record_downloads(items): Record many document URLs at once. Use this instead of calling \
  record_download() in a loop — one call records all of them immediately.
- done(reason): Signal you are finished with this page. The next queued URL is visited.
- remember(key, value, reason): Store a short observation to recall on later pages. \
  Example: remember("detail_path", "/products/"). Values should be brief (under 100 chars).
- recall(key, reason): Retrieve an observation stored with remember(). Returns the stored value or nothing.
- search_site(query, reason): Type a search query into the site's search box and submit. \
  Use when direct navigation hasn't found the target documents and a search form is visible.
- select_option(ml_id, value, reason): Select an option in a <select>/combobox element by \
  visible text or value attribute. Use to fill country/region/investor-type dropdowns in \
  gate forms before clicking the proceed button. The available options are shown in the \
  element listing as [options: ...].
- scroll_to_load(reason): Scroll to the bottom of the page to trigger lazy-loaded content. \
  Use when the page seems incomplete or more links are expected below the visible area.
- take_screenshot(reason): Take a viewport screenshot to visually inspect the page. \
  Use on detail pages when the element list shows no relevant links but the page should contain \
  documents — the screenshot may reveal tabs, accordions, document sections, or visual content \
  not captured in the element listing. Call ONCE per impasse before giving up with done(). \
  Each call costs extra tokens; do not use on listing pages or when elements are already clear. \
  On CANVAS pages a screenshot is already included in your first message — do NOT call \
  take_screenshot immediately; act on the screenshot you already have.
- fill_input(ml_id, value, reason): Type a value into a text input, date field, or other \
  fillable element. Use for filter forms requiring typed text (investor ID, date range, \
  registration number). Page is re-extracted after filling. If the field triggers an AJAX \
  reload, also call click() on the submit button.
- hover(ml_id, reason): Move the mouse over an element to reveal hover-triggered content \
  (dropdown menus, hidden sub-navigation). Page is re-extracted after hover. Use when a nav \
  element clearly has sub-links but none appear in the element listing.
- extract_text(ml_id, reason): Return the full text content of a page region by ml_id. \
  Cheaper than take_screenshot when you only need to read section text. Capped at 2 000 chars.
- click_at(coordinates, reason): Click at pixel coordinates on a canvas/Flutter page. \
  coordinates is a SINGLE string "x,y" — e.g. "408,36". Do NOT use separate x and y \
  fields. A screenshot of the updated state is returned automatically.
- collect_table_links(reason): Expand every clickable table row on the current page and \
  return all newly revealed links as page elements. Use when the element listing shows \
  [button] rows (product names, entity rows, data-grid rows) that expand inline on click — \
  call this ONCE instead of clicking each row individually. After the call the revealed \
  links appear in PAGE ELEMENTS and you can queue_urls() them all at once.

IMPORTANT — cookie/GDPR banners are auto-dismissed before you see the page. \
Treat any element whose name or URL contains "cookie", "süti", "consent", or "GDPR" as \
invisible — never click them; they will fail. \
Investor/country/language gate forms and listing-page filter forms are different — handle them: \
1. Modal gates (region/country/investor-type dropdowns with Accept/Weiter/Continue button): \
   call select_option() for any unfilled dropdowns, then click() the proceed button. \
2. Listing-page REQUIRED gate forms (dropdowns like "Please select a country" / "Please select an \
   investor type" with NO detail-page or document links visible yet): use select_option() and then \
   click() the submit button. ONLY do this when the PAGE ELEMENTS listing contains zero relevant \
   detail-page or document links — i.e. the form is a hard gate blocking all content. \
   If any detail-page or document links are already visible in PAGE ELEMENTS, skip the filter form \
   entirely and go straight to queue_urls() or record_download(). \
   Never interact with optional search/filter widgets (product finders, category filters, date \
   pickers) when content links are already present — those are UI refinements, not gates.

Strategy:
- ALWAYS record every matching document visible on the current page BEFORE navigating or calling \
  done(). Never skip a visible matching download just because you plan to navigate elsewhere.
- The DOWNLOADS FOUND list shows documents collected from ALL previously visited pages — NOT from \
  the current page. A download recorded on a previous page does NOT mean the current page has been \
  examined. Each detail page (one product, company, meeting, case...) may have its own separate document. Always inspect the current \
  page's elements before calling done().
- Listing/index pages with many links that share a URL pattern → call queue_urls() with ALL candidate \
  URLs in one call, then done(). Never call navigate() on listing pages — it visits only one link and \
  wastes the rest. IMPORTANT: queue ALL candidate pages regardless of whether they appear individually \
  relevant to the goal. The goal controls what you RECORD on each page — not which pages to visit. \
  Every detail page could contain a matching document.
- When you see 5 or more links whose URLs share the same base path (e.g., /reports/item-a, \
  /reports/item-b, /reports/item-c …), that IS a listing page. Queue ALL of them, then done(). \
  Do NOT pre-filter — you cannot know which pages hold the target document without visiting them.
- Do NOT navigate to a parent/category URL that is a prefix of links already visible on the page. \
  If you can already see /products/a, /products/b, /products/c — queue_urls() those directly. \
  Do not navigate to /products/ hoping it shows more links; parent category pages are often empty.
- When you see menuitem or link elements that clearly represent the detail pages of the subjects the goal is about \
  (e.g. paths like /products/..., /companies/..., /meetings/..., /cases/...), use \
  queue_urls() to queue ALL of them immediately — even on the homepage. Do NOT navigate \
  to general archive sections such as "Publications", "Documents", \
  "Announcements", or "Archive" (in any language) when detail page links are already \
  visible - the documents are on the detail pages, not in generic archives.
- Detail page with a single document → record_download() or click() to reveal it.
- Page with MANY visible document links → call record_downloads() ONCE with ALL matching URLs, \
  then done(). Do not loop record_download() one by one. Do not stop at 3 when there are 30.
- Call navigate() only when there is exactly one link worth following immediately.
- Always call done() when there is nothing left to do on the current page.
- Intercepted PDF downloads are recorded automatically — no need to call record_download() for those.
- Be concise in your reasons.
- Use remember() to record navigation patterns (e.g. "detail pages are under /products/") — recall them \
  on later pages to skip irrelevant sections faster.

Recognising document links — CRITICAL RULE:
A link does NOT need to end in .pdf to be a downloadable document. If the link's label, \
anchor text, or surrounding context strongly suggests it is a file (report, factsheet, announcement, \
data sheet, etc.) — words in any language like: download, PDF, report, factsheet, document, prospectus, Herunterladen, \
Bericht, télécharger, rapport, descargar, informe, letöltés, jelentés — use record_download() with that URL directly. \
Do NOT call navigate() for such links. navigate() is for web pages you need to browse, not for \
document files.

On detail pages where no direct download link is visible:
- Look for buttons (role=button) that suggest a download action — click() them to reveal the file.
- If the element list looks sparse or irrelevant on a page that should have documents, call \
  take_screenshot() ONCE to visually inspect — tabs, accordions, or document grids often appear \
  visually but are absent from the element listing. Act on what you see in the screenshot.
- If the page clearly shows a report but with no downloadable link at all, call done().
- Never call done() on a detail page until you have either recorded a download OR clicked every \
  plausible download button OR taken a screenshot and confirmed nothing is there. When in doubt, \
  take_screenshot() before giving up."""

# Cached system prompt block — constant across all API calls, cached by Anthropic for 5 min.
_SYSTEM_BLOCKS: list[dict] = [
    {
        'type': 'text',
        'text': _SYSTEM_PROMPT,
        'cache_control': {'type': 'ephemeral'},
    }
]

_TOOL_DEFINITIONS: list[dict] = [
    {
        'name': 'navigate',
        'description': 'Navigate to a URL immediately, closing the current page. Use for a single promising link.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'url': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['url', 'reason'],
        },
    },
    {
        'name': 'queue_urls',
        'description': 'Add multiple URLs to the crawl queue. Use on listing pages. You stay on the current page.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'urls': {'type': 'array', 'items': {'type': 'string'}},
                'reason': {'type': 'string'},
            },
            'required': ['urls', 'reason'],
        },
    },
    {
        'name': 'click',
        'description': 'Click a page element by ml_id. Use for buttons that reveal downloads via JavaScript.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'ml_id': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['ml_id', 'reason'],
        },
    },
    {
        'name': 'record_download',
        'description': 'Record a single document URL matching the goal.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'url': {'type': 'string'},
                'name': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['url', 'name', 'reason'],
        },
    },
    {
        'name': 'record_downloads',
        'description': (
            'Record multiple document URLs at once. Use when the page has many visible download links '
            'matching the goal — call this ONCE with all of them instead of calling record_download '
            'repeatedly. Each item must have url, name, and reason.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'url': {'type': 'string'},
                            'name': {'type': 'string'},
                            'reason': {'type': 'string'},
                        },
                        'required': ['url', 'name', 'reason'],
                    },
                },
            },
            'required': ['items'],
        },
    },
    {
        'name': 'done',
        'description': 'Signal done with this page. Advances to the next queued URL.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'reason': {'type': 'string'},
            },
            'required': ['reason'],
        },
    },
    {
        'name': 'remember',
        'description': (
            'Store a short observation to recall on later pages. '
            'Use to track patterns like "detail pages are under /products/" or "search works well here". '
            'Keep values under 100 characters.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'key': {'type': 'string'},
                'value': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['key', 'value', 'reason'],
        },
    },
    {
        'name': 'recall',
        'description': 'Retrieve an observation stored with remember(). Returns the stored value, or nothing if the key was not found.',
        'input_schema': {
            'type': 'object',
            'properties': {
                'key': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['key', 'reason'],
        },
    },
    {
        'name': 'search_site',
        'description': (
            "Type a search query into the site's search box and submit. "
            'Use when direct navigation has not found the target documents and a search form is available.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'query': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['query', 'reason'],
        },
    },
    {
        'name': 'select_option',
        'description': (
            'Select an option in a dropdown by its visible text. Works both on plain <select> elements '
            'and on custom comboboxes (a button that opens a list). '
            'Use for gate forms and for listing-page filters such as year or document type, then click '
            'the form\'s search/submit button. Available options are shown as [options: ...] when known; '
            'for a custom combobox they appear only after it is opened, and the tool reports them.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'ml_id': {'type': 'string'},
                'value': {'type': 'string', 'description': 'Visible option text or option value attribute'},
                'reason': {'type': 'string'},
            },
            'required': ['ml_id', 'value', 'reason'],
        },
    },
    {
        'name': 'scroll_to_load',
        'description': (
            'Scroll to the bottom of the page to trigger lazy-loaded content. '
            'Use when the page seems incomplete or more links are expected below the visible area.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'reason': {'type': 'string'},
            },
            'required': ['reason'],
        },
    },
    {
        'name': 'take_screenshot',
        'description': (
            'Capture a viewport screenshot to visually inspect the page. '
            'Use on detail pages when the element list shows no relevant links but the page should '
            'contain documents — reveals tabs, accordions, document grids not captured in the element '
            'listing. Call ONCE per impasse before giving up with done(). '
            'Each call costs extra tokens; do not use on listing pages or when elements are clear.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'reason': {'type': 'string'},
            },
            'required': ['reason'],
        },
    },
    {
        'name': 'fill_input',
        'description': (
            'Type a value into a text input, date field, or other fillable element. '
            'Use for filter forms requiring typed text (investor ID, date range, registration number). '
            'The page is re-extracted after filling. If the field triggers an AJAX reload, '
            'also call click() on the submit button.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'ml_id': {'type': 'string'},
                'value': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['ml_id', 'value', 'reason'],
        },
    },
    {
        'name': 'hover',
        'description': (
            'Move the mouse over an element to reveal hover-triggered content '
            '(dropdown menus, hidden sub-navigation links). Page is re-extracted after hover. '
            'Use when a nav element clearly has sub-links but none appear in the element listing.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'ml_id': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['ml_id', 'reason'],
        },
    },
    {
        'name': 'collect_table_links',
        'description': (
            'Expand every clickable table row on the current page (rows with cursor:pointer / '
            'tabindex) and collect all newly revealed <a> links. '
            'Use on listing tables where rows expand inline rather than navigating. '
            'Call ONCE per page instead of clicking each row individually. '
            'Returns a summary and adds the discovered links to the page element listing.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'reason': {'type': 'string'},
            },
            'required': ['reason'],
        },
    },
    {
        'name': 'click_at',
        'description': (
            'Click at pixel coordinates on a canvas/Flutter page. '
            'Pass coordinates as a single "x,y" string, e.g. "408,36". '
            'Use ONLY when no DOM elements are available. '
            'A screenshot of the updated state is returned automatically.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'coordinates': {
                    'type': 'string',
                    'description': 'Click position as "x,y" integers, e.g. "408,36". x=pixels from left, y=pixels from top.',
                },
                'reason': {'type': 'string'},
            },
            'required': ['coordinates', 'reason'],
        },
    },
    {
        # cache_control on the last tool caches all tool definitions in this list.
        'name': 'extract_text',
        'description': (
            'Return the full text content of a page region by its ml_id. '
            'Use when the element listing is too sparse to understand a section and a screenshot '
            'would be too expensive. Prefer over take_screenshot for text-heavy content. '
            'Output is capped at 2 000 characters.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'ml_id': {'type': 'string'},
                'reason': {'type': 'string'},
            },
            'required': ['ml_id', 'reason'],
        },
        'cache_control': {'type': 'ephemeral'},
    },
]

_DECOMPOSE_SYSTEM = """\
Extract a structured crawl plan from the user's goal.
Return ONLY a JSON object — no prose, no markdown fences — with these fields:
{
  "doc_types": ["file types or document categories to find, e.g. PDF, Excel, annual report, factsheet"],
  "key_terms": ["keywords in URLs or link text for target docs; include multilingual variants"],
  "url_patterns_prefer": ["URL path substrings that likely lead to target content, e.g. /products/, /reports/, /documents/"],
  "url_patterns_skip": ["URL path substrings to avoid, e.g. /contact, /about, /impressum, /kapcsolat"]
}
Keep each list short (3–8 items). Focus on what the goal implies about file types and site structure."""

_SEARCH_SYSTEM_PROMPT = """\
You are a web research assistant. Use the web_search tool to find official websites that best match the user's goal.
After searching, output ONLY a JSON array — no prose, no markdown fences — with this structure:
[
  {"title": "Site Name", "url": "https://example.com/specific-page", "snippet": "One sentence describing what this site contains and why it matches the goal."}
]
Rules:
- Prefer specific entry pages (investor relations, report archive, product listing) over homepages when available.
- Each snippet must be exactly one sentence synthesizing what the search revealed about that site.
- Include only sites that genuinely match the goal — do not pad with tangential results.
- Output must be parseable JSON with no surrounding text."""

_SHOW_ROLES = frozenset({'link', 'button', 'textbox', 'combobox', 'menuitem', 'checkbox', 'radio'})
_MAX_NAME_LEN = 120
_MAX_ELEMENTS = 500
_DOC_EXTENSIONS = frozenset({'.pdf', '.xlsx', '.xls', '.docx', '.doc', '.csv', '.pptx', '.zip'})


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_elements(registry: dict) -> str:
    lines: list[str] = []
    seen_urls: set[str] = set()
    for el in registry.values():
        if el.role not in _SHOW_ROLES:
            continue
        if el.url:
            if el.url in seen_urls:
                continue
            seen_urls.add(el.url)
        name = el.name if len(el.name) <= _MAX_NAME_LEN else el.name[:_MAX_NAME_LEN] + '…'
        url_part = f' — {el.url}' if el.url else ''
        opts = el.attributes.get('select_options', '') if el.attributes else ''
        opts_part = f' [options: {opts}]' if opts else ''
        lines.append(f'[{el.role} ml_id={el.ml_id}] {name}{url_part}{opts_part}')
        if len(lines) >= _MAX_ELEMENTS:
            lines.append(f'(capped at {_MAX_ELEMENTS} elements)')
            break
    return '\n'.join(lines) if lines else '(no interactive elements found)'


def _format_page_context(
    url: str,
    depth: int,
    queue_size: int,
    downloads: list[AgenticDownload],
    registry: dict,
    *,
    crawl_plan: CrawlPlan | None = None,
    memory: dict[str, str] | None = None,
    canvas_mode: str | None = None,
) -> str:
    downloads_text = (
        '\n'.join(f'  • {d.name}  [from: {d.source_page}]' for d in downloads)
        if downloads else '  (none yet)'
    )
    parts = [
        f'CURRENT PAGE: {url}',
        f'DEPTH: {depth}  QUEUE: {queue_size} remaining  DOWNLOADS: {len(downloads)}',
        f'\nDOWNLOADS FOUND:\n{downloads_text}',
    ]
    if crawl_plan and (crawl_plan.doc_types or crawl_plan.key_terms):
        plan_lines = []
        if crawl_plan.doc_types:
            plan_lines.append(f'  Doc types: {", ".join(crawl_plan.doc_types)}')
        if crawl_plan.key_terms:
            plan_lines.append(f'  Key terms: {", ".join(crawl_plan.key_terms)}')
        if crawl_plan.url_patterns_prefer:
            plan_lines.append(f'  Prefer URL patterns: {", ".join(crawl_plan.url_patterns_prefer)}')
        if crawl_plan.url_patterns_skip:
            plan_lines.append(f'  Skip URL patterns: {", ".join(crawl_plan.url_patterns_skip)}')
        parts.append('\nCRAWL PLAN:\n' + '\n'.join(plan_lines))
    if memory:
        mem_lines = '\n'.join(f'  {k} = {v}' for k, v in memory.items())
        parts.append(f'\nMEMORY:\n{mem_lines}')
    if canvas_mode:
        parts.insert(0, (
            f'CANVAS PAGE ({canvas_mode.upper()}): This page renders on a canvas — '
            'no DOM elements are available. A screenshot is already included. '
            'Use click_at(coordinates="x,y") to interact, e.g. click_at(coordinates="408,36").'
        ))
    parts.append(f'\nPAGE ELEMENTS:\n{_format_elements(registry)}')
    return '\n'.join(parts)


# Tools that re-extract the page; after anything else the element listing is unchanged and
# re-sending it only pays for tokens the model has already seen.
_PAGE_CHANGING_TOOLS = frozenset({
    'click', 'select_option', 'scroll_to_load', 'search_site', 'collect_table_links', 'fill_input',
    'hover', 'take_screenshot', 'click_at',
})


def _as_blocks(content: str | list) -> list:
    return [{'type': 'text', 'text': content}] if isinstance(content, str) else list(content)


def _set_cache_breakpoint(message: dict, on: bool) -> None:
    """Mark (or clear) a prompt-cache breakpoint on a message's last text block.

    The conversation only ever grows at the end, so the prefix is identical between steps. Without a
    breakpoint every step re-pays for the whole page listing, and input tokens dwarf output tokens.
    """
    blocks = _as_blocks(message['content'])
    for block in blocks:
        if isinstance(block, dict) and 'cache_control' in block:
            del block['cache_control']
    if on:
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get('type') == 'text':
                block['cache_control'] = {'type': 'ephemeral'}
                break
    message['content'] = blocks


def _compress_history(goal: str, steps: list[AgentStep], current_context: str) -> list[dict]:
    """Replace full history with a short summary to stay within context limits."""
    recent = steps[-10:] if len(steps) > 10 else steps
    tool_lines = '\n'.join(
        f'  {s.tool}({list(s.args.values())[0] if s.args else ""}) — {s.reason}'
        for s in recent
    )
    content = (
        f'[History compressed — {len(steps)} steps taken so far]\n'
        f'Goal: {goal}\n'
        f'Recent actions:\n{tool_lines}\n\n'
        f'{current_context}'
    )
    return [{'role': 'user', 'content': content}]


# ---------------------------------------------------------------------------
# URL / content helpers
# ---------------------------------------------------------------------------

def _strip_fragment(url: str) -> str:
    """Remove the #fragment portion from a URL - fragments are client-side anchors
    and the server returns the same page regardless of the fragment value."""
    i = url.find('#')
    return url[:i] if i != -1 else url


def _is_same_domain(url: str, seed_host: str) -> bool:
    """www.-, case- and port-insensitive host match.

    Sites redirect www.example.com <-> example.com; a strict netloc match against the seed
    then rejects every link on the site (a www-redirecting site would lose all its pages).
    """
    try:
        return _extract_domain(url) == _extract_domain(f'//{seed_host}')
    except Exception:
        return False


def _is_binary(url: str) -> bool:
    lower = url.lower().split('?')[0]
    return any(lower.endswith(ext) for ext in _BINARY_EXTENSIONS)


def _capture_pdf_response(resp, intercepted: list[tuple[str, str]]) -> None:
    if resp.url.startswith('blob:') or resp.url.startswith('data:'):
        return
    try:
        ct = resp.headers.get('content-type', '')
        cd = resp.headers.get('content-disposition', '')
    except Exception:
        ct = cd = ''
    if 'pdf' in ct.lower() or resp.url.lower().split('?')[0].endswith('.pdf'):
        filename = ''
        if 'filename' in cd:
            m = (
                re.search(r"filename\*\s*=\s*UTF-8''([^\s;]+)", cd, re.IGNORECASE)
                or re.search(r'filename\s*=\s*"([^"]+)"', cd, re.IGNORECASE)
                or re.search(r'filename\s*=\s*([^\s;]+)', cd, re.IGNORECASE)
            )
            if m:
                filename = unquote(m.group(1)).strip()
        intercepted.append((resp.url, filename))


def _flush_intercepted(
    intercepted: list[tuple[str, str]],
    downloads: list[AgenticDownload],
    source_page: str,
    reason: str,
    on_event: Callable[[dict], None] | None,
) -> None:
    """Drain intercepted PDF responses into downloads, deduplicating against existing entries."""
    seen = {d.url for d in downloads}
    for dl_url, dl_name in intercepted:
        if not dl_url or dl_url.startswith('blob:') or dl_url in seen:
            continue
        seen.add(dl_url)
        name = dl_name or dl_url.split('/')[-1].split('?')[0]
        dl = AgenticDownload(url=dl_url, name=name, reason=reason, source_page=source_page)
        downloads.append(dl)
        if on_event:
            on_event({'type': 'agent_download', 'url': dl_url, 'name': name})


# ---------------------------------------------------------------------------
# Goal decomposition (Unit 2)
# ---------------------------------------------------------------------------

def agent_llm(model: str | None = None, api_key: str | None = None) -> LLMClient:
    """The model the agent runs on: the LLM_* configuration (docseek.llm), else Anthropic with `api_key`."""
    return make_llm(model=model) or AnthropicLLM(model or DEFAULT_ANTHROPIC_MODEL, api_key=api_key)


def decompose_goal(llm: LLMClient, goal: str) -> CrawlPlan:
    """One LLM call to extract a structured CrawlPlan from a natural-language goal."""
    try:
        data = llm.complete_json(_DECOMPOSE_SYSTEM, f'Goal: {goal}')
        return CrawlPlan(
            doc_types=data.get('doc_types', []),
            key_terms=data.get('key_terms', []),
            url_patterns_prefer=data.get('url_patterns_prefer', []),
            url_patterns_skip=data.get('url_patterns_skip', []),
        )
    except Exception as exc:
        logger.warning('Goal decomposition failed, proceeding without plan: %s', exc)
    return CrawlPlan()


# ---------------------------------------------------------------------------
# Web search site discovery
# ---------------------------------------------------------------------------

def search_sites(
    client: anthropic.Anthropic,
    model: str,
    goal: str,
    max_results: int = 5,
) -> list[SearchSiteResult]:
    """Use Anthropic web search to find candidate websites matching a goal.

    Raises:
        anthropic.APIError: propagated so callers can return HTTP 503.
        ValueError / json.JSONDecodeError: on unparseable model output (HTTP 500).
    """
    max_uses = max(2, (max_results + 4) // 5)
    max_tokens = max(2048, 512 + max_results * 100)
    messages: list[dict] = [{'role': 'user', 'content': f'Goal: {goal}'}]
    tool_def = [{'type': 'web_search_20250305', 'name': 'web_search', 'max_uses': max_uses}]

    # anthropic.APIError propagates to let the endpoint return HTTP 503.
    response = client.messages.create(
        model=model,
        system=_SEARCH_SYSTEM_PROMPT,
        tools=tool_def,  # type: ignore[arg-type]
        messages=messages,
        max_tokens=max_tokens,
    )

    # Handle pause_turn: server hit its internal loop cap; continue up to 5 times.
    continuations = 0
    while response.stop_reason == 'pause_turn' and continuations < 5:
        messages = messages + [{'role': 'assistant', 'content': response.content}]
        response = client.messages.create(
            model=model,
            system=_SEARCH_SYSTEM_PROMPT,
            tools=tool_def,  # type: ignore[arg-type]
            messages=messages,
            max_tokens=max_tokens,
        )
        continuations += 1
    if continuations == 5 and response.stop_reason == 'pause_turn':
        logger.warning('search_sites: pause_turn not resolved after 5 continuations')

    # Pass 1: collect (title, url) from web_search_tool_result blocks.
    url_meta: dict[str, dict] = {}
    for block in response.content:
        if getattr(block, 'type', None) != 'web_search_tool_result':
            continue
        content = getattr(block, 'content', None)
        if not isinstance(content, list):
            continue
        for item in content:
            if getattr(item, 'type', None) == 'web_search_result':
                item_url = getattr(item, 'url', '')
                item_title = getattr(item, 'title', '')
                if item_url and item_url not in url_meta:
                    url_meta[item_url] = {'title': item_title, 'snippet': ''}

    # Pass 2: parse model's JSON text block for per-site snippets.
    # ValueError/JSONDecodeError propagates to let the endpoint return HTTP 500.
    text = next(
        (b.text for b in response.content if getattr(b, 'type', None) == 'text'),
        '',
    )
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group())
            for entry in items:
                entry_url = entry.get('url', '')
                if entry_url in url_meta:
                    url_meta[entry_url]['snippet'] = entry.get('snippet', '')
                elif entry_url:
                    url_meta[entry_url] = {
                        'title': entry.get('title', ''),
                        'snippet': entry.get('snippet', ''),
                    }
        except (json.JSONDecodeError, TypeError):
            if not url_meta:
                raise  # re-raise when we have nothing else to return

    # Filter, deduplicate, and cap.
    results: list[SearchSiteResult] = []
    seen_normalized: set[str] = set()
    for url, meta in url_meta.items():
        if not _is_safe_url(url):
            continue
        parsed = urlparse(url)
        norm_host = (parsed.hostname or '').lower().removeprefix('www.')
        norm_key = f'{parsed.scheme}://{norm_host}{parsed.path}'
        if norm_key in seen_normalized:
            continue
        seen_normalized.add(norm_key)
        results.append(SearchSiteResult(
            title=meta['title'],
            url=url,
            snippet=meta['snippet'],
        ))
        if len(results) >= max_results:
            break

    return results


# ---------------------------------------------------------------------------
# Sitemap auto-discovery (Unit 3)
# ---------------------------------------------------------------------------

def _fetch_llms_txt(base: str, timeout: int) -> list[str]:
    """Fetch /llms.txt and extract any URLs it lists (Markdown links or bare URLs)."""
    try:
        req = urllib.request.Request(f'{base}/llms.txt', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = r.read().decode('utf-8', errors='replace')
        urls: list[str] = []
        for m in re.finditer(r'\[.*?\]\((https?://[^)\s]+)\)', text):
            urls.append(m.group(1))
        for line in text.splitlines():
            line = line.strip()
            if line.startswith('http://') or line.startswith('https://'):
                urls.append(line)
        return list(dict.fromkeys(urls))
    except Exception:
        return []



@dataclass
class _FastHarvestResult:
    downloads: list[AgenticDownload]
    queue_entries: list[tuple[str, int]]
    needs_playwright: bool


def _fast_harvest(
    url: str,
    user_agent: str,
    seed_host: str,
    same_domain_only: bool,
    crawl_plan: CrawlPlan | None,
    min_url_score: float,
    depth: int,
    visited_snapshot: frozenset[str],
) -> _FastHarvestResult:
    """Plain HTTP fetch to pre-harvest document links from static HTML.

    Returns needs_playwright=True when the page requires the Playwright agent loop
    (blocked, JS-rendered shell, or no document links found in static HTML).
    Returns needs_playwright=False only for static pages where all downloadable
    content is directly visible - safe to skip browser startup entirely.
    """
    try:
        import httpx
        resp = httpx.get(url, headers={'User-Agent': user_agent}, follow_redirects=True, timeout=10.0)
    except Exception:
        return _FastHarvestResult([], [], True)

    if resp.status_code in (403, 429, 503):
        return _FastHarvestResult([], [], True)

    ct = resp.headers.get('content-type', '')
    if 'html' not in ct.lower():
        return _FastHarvestResult([], [], True)

    html = resp.text
    final_url = str(resp.url)

    # JS-shell detection: compare visible text length against embedded script data.
    body_text = re.sub(r'<[^>]+>', ' ', html)
    body_text = re.sub(r'\s+', ' ', body_text).strip()
    script_content = ''.join(re.findall(r'<script[^>]*>(.*?)</script>', html, re.IGNORECASE | re.DOTALL))
    js_shell = (
        len(body_text) < 300
        or (len(script_content) > len(body_text) * 3 and len(body_text) < 1000)
    )
    if js_shell:
        return _FastHarvestResult([], [], True)

    # Detect JS-framework signals - content may be behind a render cycle.
    _JS_SIGNALS = ('__NEXT_DATA__', 'data-reactroot', 'ng-app', 'data-v-app', 'window.__nuxt__')
    has_js_signals = any(s in html for s in _JS_SIGNALS)

    downloads: list[AgenticDownload] = []
    new_queue: list[tuple[str, int]] = []
    seen_dl: set[str] = set()
    seen_q: set[str] = set()

    for match in re.finditer(r'<a\b[^>]*\bhref=["\']([^"\']+)["\']', html, re.IGNORECASE):
        raw_href = match.group(1).strip()
        if not raw_href or raw_href.startswith('javascript:') or raw_href.startswith('#'):
            continue
        resolved = _strip_fragment(urljoin(final_url, raw_href))
        if not resolved.startswith('http'):
            continue

        tag_close = html.find('>', match.end())
        link_end = html.find('</a>', match.start())
        link_inner = html[tag_close + 1:link_end] if tag_close >= 0 and link_end > tag_close else ''
        name = re.sub(r'<[^>]+>', '', link_inner).strip() or resolved.split('/')[-1].split('?')[0]

        parsed_href = urlparse(resolved)
        if same_domain_only and not _is_same_domain(resolved, seed_host):
            continue
        path_lower = parsed_href.path.lower()
        last_seg = path_lower.rsplit('/', 1)[-1]
        ext = ('.' + last_seg.rsplit('.', 1)[-1]) if '.' in last_seg else ''

        if ext in _STATIC_DOC_EXTENSIONS:
            if resolved not in seen_dl:
                seen_dl.add(resolved)
                downloads.append(AgenticDownload(url=resolved, name=name, reason='fast-harvest', source_page=url))
        elif resolved not in visited_snapshot and resolved not in seen_q:
            if _url_passes_screen(resolved, depth + 1, crawl_plan, min_url_score):
                seen_q.add(resolved)
                new_queue.append((resolved, depth + 1))

    # Skip Playwright only for leaf document pages: has docs, no further nav links
    # to explore, and no JS signals. A page with both docs AND nav links is a hub
    # page - Playwright still needs to run for full agent-driven navigation.
    needs_playwright = len(downloads) == 0 or has_js_signals or bool(new_queue)

    return _FastHarvestResult(downloads=downloads, queue_entries=new_queue, needs_playwright=needs_playwright)


def _parse_sitemap_xml(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse sitemap XML. Returns (page_urls, child_sitemap_urls)."""
    page_locs: list[str] = []
    child_locs: list[str] = []
    try:
        root = ET.fromstring(xml_text)
        root_tag = root.tag.split('}')[-1] if '}' in root.tag else root.tag
        locs = [
            el.text.strip()
            for el in root.iter()
            if el.tag.split('}')[-1] == 'loc' and el.text and el.text.strip()
        ]
        if root_tag == 'sitemapindex':
            child_locs = locs
        else:
            page_locs = locs
    except ET.ParseError as exc:
        logger.debug('Sitemap XML parse error: %s', exc)
    return page_locs, child_locs


def _fetch_and_parse_sitemap(url: str, timeout: int) -> tuple[list[str], list[str]]:
    """Fetch a single sitemap URL. Returns (page_urls, child_sitemap_urls)."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            xml_text = r.read().decode('utf-8', errors='replace')
        return _parse_sitemap_xml(xml_text)
    except Exception as exc:
        logger.debug('Sitemap fetch failed for %s: %s', url, exc)
        return [], []


def fetch_sitemap(seed_url: str, timeout: int = 5) -> list[str]:
    """Return all page URLs from the site's sitemap (via robots.txt or /sitemap.xml).

    Side-effect: stores the seed's robots.txt in docseek.reach, so the first robots check needs no second fetch.
    Also probes /llms.txt for LLM-curated URL listings.
    """
    parsed = urlparse(seed_url)
    base = f'{parsed.scheme}://{parsed.netloc}'
    candidates: list[str] = []

    try:
        req = urllib.request.Request(f'{base}/robots.txt', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            robots_lines = r.read().decode('utf-8', errors='replace').splitlines()
        for line in robots_lines:
            if line.lower().startswith('sitemap:'):
                candidates.append(line.split(':', 1)[1].strip())
        store_robots(parsed.netloc, robots_lines)
    except Exception:
        pass

    if not candidates:
        candidates.append(f'{base}/sitemap.xml')

    all_locs: list[str] = []
    child_sitemaps: list[str] = []
    for sm_url in candidates:
        locs, children = _fetch_and_parse_sitemap(sm_url, timeout)
        all_locs.extend(locs)
        child_sitemaps.extend(children)

    for sm_url in child_sitemaps:
        locs, _ = _fetch_and_parse_sitemap(sm_url, timeout)
        all_locs.extend(locs)

    all_locs.extend(_fetch_llms_txt(base, timeout))
    return list(dict.fromkeys(all_locs))


# ---------------------------------------------------------------------------
# URL relevance pre-screener (Unit 4)
# ---------------------------------------------------------------------------

def _url_score(url: str, depth: int, crawl_plan: CrawlPlan | None) -> float:
    """Score a URL for relevance to the crawl goal. Higher = more relevant."""
    if crawl_plan is None:
        return 0.0
    try:
        path = urlparse(url).path.lower()
    except Exception:
        return 0.0

    score = 0.0

    if crawl_plan.key_terms:
        matches = sum(1 for term in crawl_plan.key_terms if term.lower() in path)
        score += 0.4 * matches / max(len(crawl_plan.key_terms), 1)

    if any(path.endswith(ext) for ext in _DOC_EXTENSIONS):
        score += 0.5

    for pattern in crawl_plan.url_patterns_prefer:
        if pattern.lower() in path:
            score += 0.3
            break

    for pattern in crawl_plan.url_patterns_skip:
        if pattern.lower() in path:
            score -= 1.0
            break

    score -= 0.05 * depth
    return score


def _url_passes_screen(
    url: str,
    depth: int,
    crawl_plan: CrawlPlan | None,
    min_score: float,
) -> bool:
    """Return True if the URL passes robots.txt and scores at or above min_score."""
    if not _url_allowed_by_robots(url):
        return False
    if crawl_plan is None or min_score <= 0.0:
        return True
    return _url_score(url, depth, crawl_plan) >= min_score


# ---------------------------------------------------------------------------
# Per-page visit result (Unit 7)
# ---------------------------------------------------------------------------

@dataclass
class _PageResult:
    """Results from visiting a single page. Merged into shared state by the coordinator."""
    navigate_url: tuple[str, int] | None = None  # (url, depth) when Claude calls navigate
    queued_urls: list[tuple[str, int]] = field(default_factory=list)
    downloads: list[AgenticDownload] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    memory: dict[str, str] = field(default_factory=dict)  # updated memory snapshot
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    steps_this_page: int = 0
    step_elements: list[dict | None] = field(default_factory=list)   # what each step addressed (for recipes)
    step_outcomes: list[str] = field(default_factory=list)            # ok | failed | recorded, per step
    storage_state: dict | None = None


# ---------------------------------------------------------------------------
# Per-page visitor (Units 1-6 logic, thread-safe)
# ---------------------------------------------------------------------------

def _visit_page(
    url: str,
    depth: int,
    *,
    llm: LLMClient,
    goal: str,
    system_blocks: list[dict],
    seed_host: str,
    same_domain_only: bool,
    js_wait_ms: int,
    click_wait_ms: int,
    max_tool_steps: int,
    max_tokens_budget: int | None = None,
    deadline: float | None = None,
    crawl_plan: CrawlPlan | None,
    min_url_score: float,
    memory_snapshot: dict[str, str],
    open_kwargs: dict,
    queue_size_hint: int,
    pre_interactions: list[dict],
    on_event: Callable[[dict], None] | None,
    visited_snapshot: frozenset[str],
    include_screenshot_on_load: bool = False,
) -> _PageResult:
    """Visit a single page and return all results. Each call has its own browser process.

    Thread-safe: uses no shared mutable state. Returns a _PageResult to be merged
    by the coordinator (agentic_crawl).
    """
    from .scraper import (
        _COLLECT_LINKS_JS,
        _try_accept_cookies,
        _try_dismiss_form_disclaimer,
        apply_pre_interactions,
        detect_canvas_page,
        extract_tree_from_page,
        open_page,
        scroll_page_to_load,
        search_in_page,
    )

    result = _PageResult(memory=dict(memory_snapshot))
    intercepted_pdfs: list[tuple[str, str]] = []

    if on_event:
        on_event({'type': 'crawl_page_start', 'url': url, 'depth': depth})

    try:
        with open_page(url, **open_kwargs) as page:
            page.on('download', lambda d: intercepted_pdfs.append((d.url, '')) if d.url else None)
            page.on('popup', lambda pop: (
                intercepted_pdfs.append((pop.url, ''))
                if pop.url.lower().split('?')[0].endswith('.pdf') else None
            ))
            page.on('response', lambda resp: _capture_pdf_response(resp, intercepted_pdfs))

            _try_dismiss_form_disclaimer(page)
            _try_accept_cookies(page, wait_ms=js_wait_ms)
            try:
                page.wait_for_load_state('networkidle', timeout=js_wait_ms * 3)
            except Exception:
                pass
            if pre_interactions:
                apply_pre_interactions(page, pre_interactions, wait_ms=js_wait_ms)

            canvas_mode = detect_canvas_page(page)
            if canvas_mode:
                # Extra wait: Flutter/canvas apps render asynchronously after networkidle
                page.wait_for_timeout(js_wait_ms)
                logger.info('[canvas] detected %s page: %s', canvas_mode, url)
                if on_event:
                    on_event({'type': 'canvas_page_detected', 'url': url, 'canvas_type': canvas_mode})
                registry = {}
            else:
                _, registry, _ = extract_tree_from_page(page)

                # Supplement accessibility tree with raw <a href> links
                try:
                    raw_links: list[dict] = page.evaluate(_COLLECT_LINKS_JS)
                    seen_urls = {el.url for el in registry.values() if el.url}
                    next_id = max((int(k) for k in registry), default=0) + 1
                    added = 0
                    for lnk in raw_links:
                        href = lnk.get('href', '')
                        if not href or href in seen_urls or href.startswith('javascript:'):
                            continue
                        if href.startswith('mailto:') or href.startswith('tel:'):
                            continue
                        seen_urls.add(href)
                        ml_id = str(next_id)
                        next_id += 1
                        registry[ml_id] = FullElement(
                            ml_id=ml_id,
                            role='link',
                            name=lnk.get('name') or href,
                            html_tag='a',
                            attributes={'href': href, 'resolved_href': href},
                            url=href,
                        )
                        added += 1
                    logger.debug('[raw-links] %s: %d total, %d added', url, len(raw_links), added)
                except Exception as exc:
                    logger.warning('Raw link collection failed on %s: %s', url, exc)

                if on_event:
                    element_lines = _format_elements(registry).splitlines()
                    on_event({
                        'type': 'page_elements',
                        'url': url,
                        'count': len(element_lines),
                        'lines': element_lines,
                    })

            context_str = _format_page_context(
                url, depth, queue_size_hint, result.downloads, registry,
                crawl_plan=crawl_plan, memory=result.memory,
                canvas_mode=canvas_mode,
            )
            initial_content: str | list
            if canvas_mode or include_screenshot_on_load:
                from .scraper import capture_screenshot
                b64 = capture_screenshot(page, max_width=None if canvas_mode else 800)
                if b64:
                    initial_content = [
                        {'type': 'text', 'text': context_str},
                        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                    ]
                else:
                    initial_content = context_str
            else:
                initial_content = context_str

            history: list[dict] = [
                {
                    'role': 'user',
                    'content': initial_content,
                }
            ]
            # The opening page context is the biggest block and never changes while this page is open.
            _set_cache_breakpoint(history[0], True)
            last_sent_elements = _format_elements(registry)

            while result.steps_this_page < max_tool_steps:
                if max_tokens_budget is not None and result.total_tokens >= max_tokens_budget:
                    # A step cap does not bound tokens: a few steps on a large page can spend a lot.
                    logger.warning('Token budget %d spent on %s - stopping this page',
                                   max_tokens_budget, url)
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    logger.warning('Crawl time budget spent on %s - stopping this page', url)
                    break
                reply = llm.chat(system_blocks, history, _TOOL_DEFINITIONS, force_tool=True, max_tokens=8192)

                result.total_tokens += reply.input_tokens + reply.output_tokens
                result.prompt_tokens += reply.input_tokens
                result.completion_tokens += reply.output_tokens
                result.cache_read_tokens += reply.cache_read_tokens
                result.cache_creation_tokens += reply.cache_write_tokens

                if reply.stop_reason == 'max_tokens':
                    logger.warning('Response truncated on %s - treating as done', url)
                    break

                tool_block = reply.tool_call
                if tool_block is None:
                    logger.warning('No tool call on %s - treating as done', url)
                    break

                tool_name = tool_block['name']
                tool_input = tool_block['input']
                reason = tool_input.get('reason', '')

                result.steps.append(AgentStep(
                    tool=tool_name, args=tool_input, reason=reason, source_url=url
                ))
                touched = registry.get(str(tool_input.get('ml_id', ''))) if isinstance(tool_input, dict) else None
                result.step_elements.append(touched.model_dump() if touched else None)
                if on_event:
                    on_event({
                        'type': 'agent_tool_call',
                        'url': url,
                        'tool': tool_name,
                        'args': tool_input,
                    })

                tool_result_text = ''
                tool_result_content = None
                should_break = False

                if tool_name == 'navigate':
                    nav_url = _strip_fragment(tool_input.get('url', ''))
                    if not nav_url:
                        tool_result_text = 'No URL provided.'
                    elif nav_url in visited_snapshot:
                        tool_result_text = (
                            f'Already visited: {nav_url}. '
                            f'Choose a different link or call done().'
                        )
                    elif same_domain_only and not _is_same_domain(nav_url, seed_host):
                        tool_result_text = (
                            f'Off-domain URL skipped: {nav_url}. '
                            f'Choose a link within {seed_host} or call done().'
                        )
                    elif _is_binary(nav_url):
                        tool_result_text = f'Binary URL skipped: {nav_url}. Use record_download() instead.'
                    else:
                        result.navigate_url = (nav_url, depth + 1)
                        tool_result_text = f'Navigating to: {nav_url}'
                        should_break = True

                elif tool_name == 'queue_urls':
                    urls: list[str] = tool_input.get('urls', [])
                    added = 0
                    for u in urls:
                        u = _strip_fragment(u)
                        if u in visited_snapshot:
                            continue
                        if same_domain_only and not _is_same_domain(u, seed_host):
                            continue
                        if _is_binary(u):
                            continue
                        if not _url_passes_screen(u, depth + 1, crawl_plan, min_url_score):
                            continue
                        if all(u != q[0] for q in result.queued_urls):
                            result.queued_urls.append((u, depth + 1))
                            added += 1
                            if on_event:
                                on_event({'type': 'queue_add', 'url': u, 'depth': depth + 1})
                    tool_result_text = f'Queued {added} of {len(urls)} URLs'

                elif tool_name == 'click':
                    ml_id = tool_input.get('ml_id', '')
                    try:
                        loc = page.locator(f'[data-ml-id="{ml_id}"]')
                        if loc.count() > 0:
                            el_info = registry.get(ml_id)
                            if el_info and el_info.role == 'checkbox':
                                try:
                                    loc.check(timeout=5000, force=True)
                                except Exception:
                                    loc.dispatch_event('click')
                            else:
                                loc.scroll_into_view_if_needed(timeout=5000)
                                loc.dispatch_event('click')
                            _, registry, _ = extract_tree_from_page(page, js_wait_ms=click_wait_ms)
                            _flush_intercepted(
                                intercepted_pdfs, result.downloads, url,
                                f'Intercepted after click ml_id={ml_id}', on_event,
                            )
                            tool_result_text = f'Clicked ml_id={ml_id}; page re-extracted'
                        else:
                            tool_result_text = f'Element ml_id={ml_id} not found'
                    except Exception as exc:
                        tool_result_text = f'Click failed: {exc}'
                        logger.warning('Click on ml_id=%s failed on %s: %s', ml_id, url, exc)

                elif tool_name == 'record_download':
                    dl_url = tool_input.get('url', '')
                    dl_name = tool_input.get('name', '') or dl_url.split('/')[-1].split('?')[0]
                    seen = {d.url for d in result.downloads}
                    if dl_url and not _is_safe_url(dl_url):
                        # the page (and so the model) chose this URL: it is never handed on unchecked
                        tool_result_text = f'Refused: {dl_url} is not a public http(s) address'
                    elif dl_url and dl_url not in seen:
                        dl = AgenticDownload(
                            url=dl_url, name=dl_name, reason=reason, source_page=url
                        )
                        result.downloads.append(dl)
                        if on_event:
                            on_event({'type': 'agent_download', 'url': dl_url, 'name': dl_name})
                        tool_result_text = f'Recorded: {dl_name}'
                    else:
                        tool_result_text = f'Already recorded or empty: {dl_url}'

                elif tool_name == 'record_downloads':
                    items: list[dict] = tool_input.get('items', [])
                    seen = {d.url for d in result.downloads}
                    added = 0
                    for item in items:
                        dl_url = item.get('url', '')
                        dl_name = item.get('name', '') or dl_url.split('/')[-1].split('?')[0]
                        dl_reason = item.get('reason', reason)
                        if dl_url and dl_url not in seen and _is_safe_url(dl_url):
                            seen.add(dl_url)
                            dl = AgenticDownload(
                                url=dl_url, name=dl_name, reason=dl_reason, source_page=url
                            )
                            result.downloads.append(dl)
                            if on_event:
                                on_event({'type': 'agent_download', 'url': dl_url, 'name': dl_name})
                            added += 1
                    tool_result_text = f'Recorded {added} of {len(items)} downloads'

                elif tool_name == 'done':
                    tool_result_text = f'Done: {reason}'
                    should_break = True

                elif tool_name == 'remember':
                    key = tool_input.get('key', '').strip()
                    value = tool_input.get('value', '').strip()
                    if key:
                        result.memory[key] = value
                        tool_result_text = f'Remembered: {key} = {value}'
                    else:
                        tool_result_text = 'No key provided.'

                elif tool_name == 'recall':
                    key = tool_input.get('key', '').strip()
                    value = result.memory.get(key)
                    tool_result_text = f'{key} = {value}' if value is not None else f'Nothing stored for key: {key}'

                elif tool_name == 'search_site':
                    query_text = tool_input.get('query', '')
                    new_registry, tool_result_text = search_in_page(page, query_text, js_wait_ms)
                    if new_registry:
                        registry = new_registry
                        _flush_intercepted(
                            intercepted_pdfs, result.downloads, url,
                            'Intercepted after site search', on_event,
                        )

                elif tool_name == 'scroll_to_load':
                    new_registry, delta, tool_result_text = scroll_page_to_load(page, js_wait_ms)
                    if new_registry and delta > 0:
                        registry = new_registry

                elif tool_name == 'take_screenshot':
                    from .scraper import capture_screenshot
                    b64 = capture_screenshot(page, max_width=None if canvas_mode else 800)
                    if b64:
                        tool_result_content = [
                            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}}
                        ]
                        tool_result_text = 'Screenshot captured.'
                    else:
                        tool_result_content = None  # falls back to tool_result_text
                        tool_result_text = 'Screenshot failed: browser returned no image.'
                    if on_event:
                        on_event({'type': 'agent_screenshot', 'url': url})

                elif tool_name == 'select_option':
                    ml_id = tool_input.get('ml_id', '')
                    value = tool_input.get('value', '')
                    try:
                        from .scraper import select_option_anywhere
                        outcome = select_option_anywhere(page, ml_id, value, wait_ms=click_wait_ms)
                        _, registry, _ = extract_tree_from_page(page, js_wait_ms=click_wait_ms)
                        _flush_intercepted(
                            intercepted_pdfs, result.downloads, url,
                            f'Intercepted after select_option ml_id={ml_id}', on_event,
                        )
                        tool_result_text = f'{outcome}; page re-extracted'
                    except Exception as exc:
                        tool_result_text = f'select_option failed: {exc}'
                        logger.warning('select_option ml_id=%s value=%r failed: %s', ml_id, value, exc)

                elif tool_name == 'fill_input':
                    ml_id = tool_input.get('ml_id', '')
                    value = tool_input.get('value', '')
                    try:
                        loc = page.locator(f'[data-ml-id="{ml_id}"]')
                        if loc.count() > 0:
                            loc.fill(str(value), timeout=5000)
                            _, registry, _ = extract_tree_from_page(page, js_wait_ms=click_wait_ms)
                            _flush_intercepted(
                                intercepted_pdfs, result.downloads, url,
                                f'Intercepted after fill_input ml_id={ml_id}', on_event,
                            )
                            tool_result_text = f'Filled ml_id={ml_id} with "{value}"; page re-extracted'
                        else:
                            tool_result_text = f'Element ml_id={ml_id} not found'
                    except Exception as exc:
                        tool_result_text = f'fill_input failed: {exc}'
                        logger.warning('fill_input on ml_id=%s failed on %s: %s', ml_id, url, exc)

                elif tool_name == 'hover':
                    ml_id = tool_input.get('ml_id', '')
                    try:
                        loc = page.locator(f'[data-ml-id="{ml_id}"]')
                        if loc.count() > 0:
                            loc.hover(timeout=5000)
                            page.wait_for_timeout(click_wait_ms)
                            _, registry, _ = extract_tree_from_page(page)
                            tool_result_text = f'Hovered ml_id={ml_id}; page re-extracted with {len(registry)} elements'
                        else:
                            tool_result_text = f'Element ml_id={ml_id} not found'
                    except Exception as exc:
                        tool_result_text = f'hover failed: {exc}'
                        logger.warning('hover on ml_id=%s failed on %s: %s', ml_id, url, exc)

                elif tool_name == 'collect_table_links':
                    try:
                        from .scraper import _COLLECT_LINKS_JS
                        initial_hrefs: set[str] = {
                            l['href'] for l in page.evaluate(_COLLECT_LINKS_JS)
                            if l.get('href')
                        }
                        rows = page.locator('tr[tabindex]')
                        row_count = rows.count()
                        clicked = 0
                        new_links: list[dict] = []
                        seen_new: set[str] = set()
                        for i in range(row_count):
                            try:
                                row = rows.nth(i)
                                cursor = row.evaluate('el => window.getComputedStyle(el).cursor')
                                if cursor != 'pointer':
                                    continue
                                row.scroll_into_view_if_needed(timeout=3000)
                                row.click(timeout=3000)
                                page.wait_for_timeout(max(500, click_wait_ms // 3))
                                clicked += 1
                                for lnk in page.evaluate(_COLLECT_LINKS_JS):
                                    href = lnk.get('href', '')
                                    if href and href not in initial_hrefs and href not in seen_new:
                                        seen_new.add(href)
                                        new_links.append(lnk)
                            except Exception:
                                continue
                        next_id = max((int(k) for k in registry), default=0) + 1
                        for lnk in new_links:
                            href = lnk['href']
                            ml_id_new = str(next_id)
                            next_id += 1
                            registry[ml_id_new] = FullElement(
                                ml_id=ml_id_new,
                                role='link',
                                name=lnk.get('name') or href,
                                html_tag='a',
                                attributes={'href': href, 'resolved_href': href},
                                url=href,
                            )
                        _flush_intercepted(
                            intercepted_pdfs, result.downloads, url,
                            'Intercepted after collect_table_links', on_event,
                        )
                        tool_result_text = (
                            f'Clicked {clicked} rows, found {len(new_links)} new links. '
                            f'They are now in PAGE ELEMENTS — use queue_urls() to enqueue them.'
                        )
                    except Exception as exc:
                        tool_result_text = f'collect_table_links failed: {exc}'
                        logger.warning('collect_table_links failed on %s: %s', url, exc)

                elif tool_name == 'extract_text':
                    ml_id = tool_input.get('ml_id', '')
                    try:
                        loc = page.locator(f'[data-ml-id="{ml_id}"]')
                        if loc.count() > 0:
                            text = loc.inner_text(timeout=5000)
                            tool_result_text = f'Text content of ml_id={ml_id}:\n{text[:2000]}'
                        else:
                            tool_result_text = f'Element ml_id={ml_id} not found'
                    except Exception as exc:
                        tool_result_text = f'extract_text failed: {exc}'
                        logger.warning('extract_text on ml_id=%s failed on %s: %s', ml_id, url, exc)

                elif tool_name == 'click_at':
                    coords_raw = str(tool_input.get('coordinates', '0,0')).strip()
                    try:
                        parts = [p.strip() for p in coords_raw.split(',') if p.strip()]
                        x, y = float(parts[0]), float(parts[1])
                    except (IndexError, ValueError) as parse_exc:
                        tool_result_text = f'click_at: invalid coordinates {coords_raw!r}: {parse_exc}'
                        logger.warning('click_at bad coords %r on %s: %s', coords_raw, url, parse_exc)
                        x = y = None  # type: ignore[assignment]
                    if x is not None:
                        try:
                            page.mouse.click(x, y)
                            try:
                                page.wait_for_load_state('networkidle', timeout=click_wait_ms * 5)
                            except Exception:
                                page.wait_for_timeout(click_wait_ms)
                            from .scraper import capture_screenshot
                            b64 = capture_screenshot(page, max_width=None)
                            if b64:
                                tool_result_content = [
                                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64}},
                                ]
                                tool_result_text = f'Clicked at ({x:.0f}, {y:.0f}); screenshot of updated page:'
                            else:
                                tool_result_text = f'Clicked at ({x:.0f}, {y:.0f}); screenshot unavailable'
                        except Exception as exc:
                            tool_result_text = f'click_at ({x:.0f}, {y:.0f}) failed: {exc}'
                            logger.warning('click_at (%.0f, %.0f) failed on %s: %s', x, y, url, exc)

                else:
                    tool_result_text = f'Unknown tool: {tool_name}'
                    logger.error('Unexpected tool call: %s', tool_name)

                history.append({'role': 'assistant', 'content': reply.content})
                if should_break:
                    follow_up = tool_result_text
                else:
                    current_elements = _format_elements(registry)
                    if tool_name in _PAGE_CHANGING_TOOLS and current_elements != last_sent_elements:
                        follow_up = _format_page_context(
                            url, depth, queue_size_hint, result.downloads, registry,
                            crawl_plan=crawl_plan, memory=result.memory,
                            canvas_mode=canvas_mode,
                        )
                        last_sent_elements = current_elements
                    else:
                        # The page did not change: recall the state without re-sending the listing.
                        follow_up = (
                            f'PAGE UNCHANGED: {url}  DOWNLOADS: {len(result.downloads)}  '
                            f'QUEUE: {queue_size_hint}. The PAGE ELEMENTS listing above still applies.'
                        )
                for message in history:
                    if message['role'] == 'user':
                        _set_cache_breakpoint(message, False)
                _set_cache_breakpoint(history[0], True)
                # what the step came to, for the recipe: a step the page refused is not worth replaying
                result.step_outcomes.append(
                    'failed' if re.search(r'failed|not found', tool_result_text or '') else
                    'recorded' if tool_name.startswith('record_download') and result.downloads else 'ok')
                history.append({
                    'role': 'user',
                    'content': [
                        {'type': 'tool_result', 'tool_use_id': tool_block['id'], 'content': tool_result_content if tool_result_content is not None else tool_result_text},
                        {'type': 'text', 'text': follow_up},
                    ],
                })
                # Roll the second breakpoint onto the newest message so each step caches the prefix.
                _set_cache_breakpoint(history[-1], True)
                result.steps_this_page += 1

                if result.steps_this_page % 20 == 0 and not should_break:
                    history = _compress_history(goal, result.steps, follow_up)

                if should_break:
                    break

            _flush_intercepted(
                intercepted_pdfs, result.downloads, url,
                'Auto-recorded on page close', on_event,
            )

            try:
                result.storage_state = page.context.storage_state()
            except Exception as exc:
                logger.debug('storage_state capture failed: %s', exc)

    except Exception as exc:
        logger.error('Page visit failed for %s: %s', url, exc)

    if on_event:
        on_event({
            'type': 'crawl_page_complete',
            'url': url,
            'depth': depth,
            'steps_this_page': result.steps_this_page,
            'downloads_this_page': len(result.downloads),
            'total_tokens': result.total_tokens,
        })

    return result


# ---------------------------------------------------------------------------
# Pattern learning helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Return the normalized domain key for a URL (no www., no port, lowercase)."""
    netloc = urlparse(url).netloc.lower()
    if ':' in netloc:
        netloc = netloc.split(':')[0]
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    return netloc


def _build_domain_knowledge_block(p: DomainPatterns) -> dict:
    lines = ['DOMAIN KNOWLEDGE (hints from prior crawls — treat as suggestions, not rules):']
    if p.last_goal:
        lines.append(f'  Learned while looking for: {p.last_goal!r}')
        lines.append('  URL patterns and memory below are goal-specific — apply judgment if your goal differs.')
    if p.url_patterns_prefer:
        lines.append(f'  Prefer URLs containing: {p.url_patterns_prefer}')
    if p.url_patterns_skip:
        lines.append(f'  Skip URLs containing: {p.url_patterns_skip}')
    if p.gate_sequences:
        lines.append('  Gate sequences seen before:')
        for g in p.gate_sequences:
            lines.append(f'    - {g.description}')
            for s in g.steps:
                suffix = f' = {s.value}' if s.value else ''
                lines.append(f'      {s.tool}: {s.hint}{suffix}')
    if p.memory_snapshot:
        lines.append(f'  Prior memory: {p.memory_snapshot}')
    if p.navigation_hints:
        lines.append(f'  Navigation: {p.navigation_hints}')
    return {'type': 'text', 'text': '\n'.join(lines)}


# ---------------------------------------------------------------------------
# Main crawl coordinator
# ---------------------------------------------------------------------------

def agentic_crawl(
    start_url: str,
    goal: str,
    *,
    api_key: str | None = None,
    llm: LLMClient | None = None,
    max_pages: int = 10,
    max_depth: int = 3,
    same_domain_only: bool = True,
    model: str | None = None,
    user_agent: str | None = None,
    js_wait_ms: int = 2000,
    click_wait_ms: int = 2000,
    max_tool_steps: int = 100,
    min_url_score: float = 0.0,
    max_concurrent: int = 1,
    headless: bool = True,
    pre_interactions: list[dict] | None = None,
    include_screenshot_on_load: bool = False,
    on_event: Callable[[dict], None] | None = None,
    patterns_dir: str | None = None,
    enable_learning: bool = True,
) -> AgenticCrawlResult:
    """Crawl a website with a tool-using LLM agent reasoning about page interactions. `llm` is any
    docseek.llm client; without one, agent_llm(model, api_key) picks it from the environment.

    Set max_concurrent > 1 to visit multiple pages in parallel (each in its own
    browser process via ThreadPoolExecutor). Shared state is merged after each batch.
    """
    llm = llm or agent_llm(model, api_key)
    seed_host = urlparse(start_url).netloc

    # Pre-crawl: decompose goal (Unit 2)
    if on_event:
        on_event({'type': 'decompose_start', 'goal': goal})
    crawl_plan = decompose_goal(llm, goal)
    if on_event:
        on_event({
            'type': 'decompose_complete',
            'doc_types': crawl_plan.doc_types,
            'key_terms': crawl_plan.key_terms,
            'url_patterns_prefer': crawl_plan.url_patterns_prefer,
            'url_patterns_skip': crawl_plan.url_patterns_skip,
        })
    logger.debug('[plan] doc_types=%s key_terms=%s', crawl_plan.doc_types, crawl_plan.key_terms)

    # Pre-crawl: sitemap (Unit 3)
    sitemap_urls = fetch_sitemap(start_url)
    if sitemap_urls:
        logger.info('[sitemap] found %d URLs', len(sitemap_urls))

    queue: list[tuple[str, int]] = [(start_url, 0)]
    visited: set[str] = set()
    downloads: list[AgenticDownload] = []
    all_steps: list[AgentStep] = []
    memory: dict[str, str] = {}
    total_tokens = prompt_tokens = completion_tokens = 0
    cache_read_tokens = cache_creation_tokens = 0
    # max_pages bounds LLM page-visits only; deterministic harvest (static pages) is cheap and
    # must not be starved by it. _total_visit_cap is the runaway guard on total pages processed.
    llm_pages = 0
    _total_visit_cap = max(max_pages * 20, 100)

    if sitemap_urls and on_event:
        on_event({
            'type': 'sitemap_loaded',
            'url_count': len(sitemap_urls),
            'queued_count': 0,
        })

    # Pre-crawl: deterministic API mining. Canvas/SPA sites (e.g. Flutter+Firebase) render their
    # document list from a backend the DOM/vision path can't see; mine the JS bundle for the API
    # and read the docs directly. Browserless and precise - no LLM budget consumed.
    try:
        from .api_mining import mine_documents
        mined = mine_documents(
            start_url,
            user_agent=user_agent or _DEFAULT_USER_AGENT,
            crawl_plan=crawl_plan,
            source_page=start_url,
        )
        for dl in mined:
            if all(dl.url != d.url for d in downloads):
                downloads.append(dl)
                if on_event:
                    on_event({'type': 'agent_download', 'url': dl.url, 'name': dl.name})
        if mined:
            logger.info('[api_mining] recorded %d document(s) from backend API', len(mined))
    except Exception as exc:
        logger.warning('[api_mining] pre-crawl mining failed: %s', exc)

    open_kwargs: dict = {'js_wait_ms': js_wait_ms, 'accept_downloads': True, 'headless': headless}
    if user_agent:
        open_kwargs['user_agent'] = user_agent

    # Load domain patterns and inject as hints
    _store: PatternStore | None = PatternStore(patterns_dir) if patterns_dir else None
    _domain = _extract_domain(start_url)
    _domain_patterns: DomainPatterns | None = _store.load(_domain) if _store else None
    domain_knowledge_blocks: list[dict] = []
    if _domain_patterns:
        domain_knowledge_blocks = [_build_domain_knowledge_block(_domain_patterns)]
        memory.update(_domain_patterns.memory_snapshot)
        if on_event:
            on_event({
                'type': 'patterns_loaded',
                'domain': _domain,
                'successful_crawl_count': _domain_patterns.successful_crawl_count,
                'url_patterns_prefer': _domain_patterns.url_patterns_prefer,
                'url_patterns_skip': _domain_patterns.url_patterns_skip,
                'gate_sequences': len(_domain_patterns.gate_sequences),
                'memory_keys': list(_domain_patterns.memory_snapshot.keys()),
            })
        # Augment crawl_plan URL patterns with learned ones (after decompose_goal)
        combined_prefer = list(dict.fromkeys(
            crawl_plan.url_patterns_prefer + _domain_patterns.url_patterns_prefer
        ))
        combined_skip = list(dict.fromkeys(
            crawl_plan.url_patterns_skip + _domain_patterns.url_patterns_skip
        ))
        crawl_plan = crawl_plan.model_copy(update={
            'url_patterns_prefer': combined_prefer,
            'url_patterns_skip': combined_skip,
        })

    goal_block = {'type': 'text', 'text': f'Goal: {goal}'}
    system_blocks = _SYSTEM_BLOCKS + domain_knowledge_blocks + [goal_block]

    _pre_interactions: list[dict] = pre_interactions or []

    def _build_page_kwargs(url: str, depth: int, queue_size: int) -> dict:
        return dict(
            llm=llm, goal=goal, system_blocks=system_blocks,
            seed_host=seed_host, same_domain_only=same_domain_only,
            js_wait_ms=js_wait_ms, click_wait_ms=click_wait_ms,
            max_tool_steps=max_tool_steps, crawl_plan=crawl_plan,
            min_url_score=min_url_score, memory_snapshot=dict(memory),
            open_kwargs=open_kwargs, queue_size_hint=queue_size,
            pre_interactions=_pre_interactions,
            include_screenshot_on_load=include_screenshot_on_load,
            on_event=on_event, visited_snapshot=frozenset(visited),
        )

    def _merge_result(result: _PageResult) -> None:
        nonlocal total_tokens, prompt_tokens, completion_tokens
        nonlocal cache_read_tokens, cache_creation_tokens
        downloads.extend(result.downloads)
        all_steps.extend(result.steps)
        memory.update(result.memory)
        total_tokens += result.total_tokens
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        cache_read_tokens += result.cache_read_tokens
        cache_creation_tokens += result.cache_creation_tokens
        if result.storage_state:
            open_kwargs['storage_state'] = result.storage_state

    def _pop_next() -> tuple[str, int] | None:
        """Pop the next eligible URL from the queue. Returns None if nothing left."""
        while queue:
            url, depth = queue.pop(0)
            url = _strip_fragment(url)
            if url in visited or depth > max_depth or _is_binary(url):
                continue
            visited.add(url)
            return url, depth
        return None

    _harvest_user_agent = open_kwargs.get('user_agent', _DEFAULT_USER_AGENT)

    def _apply_fast_harvest(current_url: str, current_depth: int) -> bool:
        """Run fast HTTP harvest for current_url. Returns True when Playwright is still needed."""
        harvest = _fast_harvest(
            current_url, _harvest_user_agent,
            seed_host, same_domain_only, crawl_plan, min_url_score,
            current_depth, frozenset(visited),
        )
        # Always pre-populate the queue - static link discovery helps even when Playwright runs.
        seen_q = {u for u, _ in queue}
        for fq_url, fq_depth in harvest.queue_entries:
            if fq_url not in visited and fq_url not in seen_q:
                seen_q.add(fq_url)
                queue.append((fq_url, fq_depth))
        if not harvest.needs_playwright:
            # Playwright is skipped - record downloads now (agent won't run to do it).
            seen_dl = {d.url for d in downloads}
            for dl in harvest.downloads:
                if dl.url not in seen_dl:
                    seen_dl.add(dl.url)
                    downloads.append(dl)
                    if on_event:
                        on_event({'type': 'agent_download', 'url': dl.url, 'name': dl.name})
            if on_event:
                on_event({
                    'type': 'fast_harvest',
                    'url': current_url,
                    'downloads': len(harvest.downloads),
                    'queued': len(harvest.queue_entries),
                })
        return harvest.needs_playwright

    if max_concurrent <= 1:
        # Sequential path - straightforward loop
        while queue and llm_pages < max_pages and len(visited) < _total_visit_cap:
            entry = _pop_next()
            if entry is None:
                break
            current_url, depth = entry

            if not _apply_fast_harvest(current_url, depth):
                # Static page fully handled by deterministic harvest - no browser, no LLM,
                # so it does not consume the max_pages (LLM) budget.
                continue

            result = _visit_page(
                current_url, depth,
                **_build_page_kwargs(current_url, depth, len(queue)),
            )
            llm_pages += 1
            _merge_result(result)

            # Handle navigate (insert at front with priority)
            if result.navigate_url:
                nav_url, nav_depth = result.navigate_url
                nav_url = _strip_fragment(nav_url)
                if nav_url not in visited:
                    queue.insert(0, (nav_url, nav_depth))

            # Merge queued URLs at front so they run before anything else in queue.
            new_urls = []
            for url, dep in result.queued_urls:
                url = _strip_fragment(url)
                if url in visited:
                    continue
                queue[:] = [(u, d) for u, d in queue if u != url]
                new_urls.append((url, dep))
            queue[0:0] = new_urls

    else:
        # Parallel path - batch dispatch via ThreadPoolExecutor
        # on_event uses queue.Queue.put() in server.py (thread-safe); debug on_events use print (GIL-safe).
        while queue and llm_pages < max_pages and len(visited) < _total_visit_cap:
            batch: list[tuple[str, int]] = []
            while queue and len(batch) < max_concurrent and len(visited) < _total_visit_cap:
                entry = _pop_next()
                if entry is None:
                    break
                batch.append(entry)

            if not batch:
                break

            # Fast-harvest each batch entry before committing to browser dispatch.
            playwright_batch: list[tuple[str, int]] = []
            for b_url, b_depth in batch:
                if not _apply_fast_harvest(b_url, b_depth):
                    continue  # static page handled deterministically - no browser, no LLM budget
                playwright_batch.append((b_url, b_depth))

            if not playwright_batch:
                continue
            llm_pages += len(playwright_batch)

            mem_snapshot = dict(memory)
            visited_snap = frozenset(visited)
            queue_size_hint = len(queue)

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(playwright_batch)) as pool:
                future_map = {
                    pool.submit(
                        _visit_page, url, depth,
                        llm=llm, goal=goal, system_blocks=system_blocks,
                        seed_host=seed_host, same_domain_only=same_domain_only,
                        js_wait_ms=js_wait_ms, click_wait_ms=click_wait_ms,
                        max_tool_steps=max_tool_steps, crawl_plan=crawl_plan,
                        min_url_score=min_url_score, memory_snapshot=mem_snapshot,
                        open_kwargs=open_kwargs, queue_size_hint=queue_size_hint,
                        pre_interactions=_pre_interactions,
                        include_screenshot_on_load=include_screenshot_on_load,
                        on_event=on_event, visited_snapshot=visited_snap,
                    ): (url, depth)
                    for url, depth in playwright_batch
                }

                navigate_urls: list[tuple[str, int]] = []
                for future in concurrent.futures.as_completed(future_map.keys()):
                    try:
                        result = future.result()
                        _merge_result(result)
                        if result.navigate_url:
                            navigate_urls.append(result.navigate_url)
                        new_batch_urls = []
                        for u, dep in result.queued_urls:
                            if u in visited:
                                continue
                            queue[:] = [(qu, qd) for qu, qd in queue if qu != u]
                            new_batch_urls.append((u, dep))
                        queue[0:0] = new_batch_urls
                    except Exception as exc:
                        src_url, _ = future_map[future]
                        logger.error('Page visit thread failed for %s: %s', src_url, exc)

            # Navigate URLs get priority (insert at front in reverse order to preserve order)
            for nav_url, nav_depth in reversed(navigate_urls):
                if nav_url not in visited:
                    queue.insert(0, (nav_url, nav_depth))

    if on_event:
        on_event({'type': 'crawl_complete', 'pages_crawled': len(visited)})

    result = AgenticCrawlResult(
        start_url=start_url,
        goal=goal,
        downloads=downloads,
        steps=all_steps,
        final_memory=dict(memory),
        pages_visited=len(visited),
        total_tokens=total_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        decision_model=llm.model,
    )

    if enable_learning and _store and len(result.downloads) >= 1:
        threading.Thread(
            target=extract_and_save,
            args=(_domain, all_steps, dict(memory), crawl_plan, _store, llm),
            kwargs={'goal': goal},
            daemon=False,
        ).start()

    return result
