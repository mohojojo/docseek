"""The Jev decision layer: a crawl whose per-page judgements are calibrated questions, not tool calls.

Shape:
  - one browser per worker, reused across pages; assets, trackers and third-party XHR blocked; a page is
    "loaded" when no first-party XHR is in flight and the link count has settled
  - code runs the reveal-only interactions (gate dismissal, table-row expansion, scroll-to-load) and owns
    every URL, target, domain rule and stop rule
  - Jev judges: Relevance for every Candidate, Page kind for the Frontier, and whether
    a page still hides documents
  - Escalation hands one page to the agent loop; the agent nominates, Jev gives the Verdict
  - the circuit breaker falls back to the agent path for the rest of the crawl
  - off-domain: Candidates may come from any host; navigation may cross to ONE host outside the
    seed and then crawl within it, never chaining to a third host

The agent path (`agent.agentic_crawl`) stays the default and the fallback.
"""
from __future__ import annotations

import collections
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from playwright.sync_api import sync_playwright

from .agent import (
    _SYSTEM_BLOCKS, agent_llm, _is_binary, _is_safe_url, _strip_fragment, _url_allowed_by_robots, _visit_page,
    agentic_crawl, fetch_sitemap,
)
from .api_mining import mine_documents
from .document_probe import DocumentProbe
from .frontier import Frontier, frontier_key  # noqa: F401 - frontier_key is re-exported
from .json_mining import MAX_JSON_BYTES, candidates_from_json
from .judge import RelevanceJudge, make_judge, verdict_for
from .llm import LLMClient
from .models import AgenticCrawlResult, AgenticDownload
from .reach import OffDomainPolicy, bare_host, is_crawlable  # noqa: F401 - re-exported
from .recipes import RecipeStore, recipe_from_steps, replay as replay_recipe
from .series import document_period, period_of  # noqa: F401 - re-exported: the period Facet is read by docseek.series
from .scraper import (
    _DEFAULT_USER_AGENT, _TRACKING_SCRIPT_HOSTS, _try_accept_cookies, _try_dismiss_form_disclaimer,
    detect_canvas_page, select_option_anywhere,
)

logger = logging.getLogger(__name__)

AGENT_STEPS_PER_ESCALATION = 20
AGENT_TOKEN_CAP = 300_000          # per crawl
MAX_FILTERED_LISTING_ESCALATIONS = 3   # enough for a filtered listing; more added no recall
THIN_LISTING_DOCUMENTS = 5             # a filter that hides items leaves few visible, where an
                                       # unfiltered listing shows many
FILTERED_LISTING_AT = 0.7          # a listing behind a filter shows its newest item and hides the rest,
                                   # which Jev scores right around the 0.8 line
ESCALATION_TOKEN_CAP = 50_000      # a page that has never paid gets a small budget
ESCALATION_TOKEN_CAP_RICH = 150_000  # ...a filtered listing, canvas or form gets enough to finish      # per escalated page: a step cap alone did not bound tokens
ESCALATION_MIN_SECONDS = 30        # an escalation takes tens of seconds; with less of the crawl's time left it
                                   # pays for the opening page context and is cut before it records anything
STALE_PAGES_STOP = 12              # pages with no new candidate before giving up
STALE_MIN_BUDGET_FRACTION = 0.5    # ...and only once half the page/time budget is spent
SITEMAP_CAP = 10_000               # sitemap URLs read
SITEMAP_PAGES_SCORED = 1_000
MAX_VARIANTS_PER_PATH = 3          # pagination variants of one path, beyond which it is all repetition
# A query parameter that pages, sorts or resets a listing rather than naming a different page. A
# URL whose query differs from a queued one only in these is another page of the same listing and counts
# against MAX_VARIANTS_PER_PATH; a URL that differs in any other parameter (`?id=`, `?nif=`, `?__ksinr=`)
# is a different page and is not capped. A council meeting portal and a regulator's fund register each
# list their pages under one path, and counting those as variants cut them to the cap.
_PAGING_PARAM = re.compile(r'(^|_)(cur|delta|page|pagina|seite|oldal|start|offset|limit|sort|order|orderby|'
                           r'resetcur|redirect|p_p_\w+|__c\w+|__canz|__cselect)$', re.IGNORECASE)
# A parameter that narrows a listing to a facet of itself (an issuer, a category, a date range, a search
# term) rather than naming a different page. One disclosure portal offers dozens of issuer facets and a
# date form from every listing page; each is "a page about one company that lists its documents" to the
# page-kind question, and each visit produces more. A facet of a listing is the listing.
_FACET_PARAM = re.compile(r'issuer|filter|facet|categor|kategor|type|typ\b|datefrom|dateto|date_from|date_to|'
                          r'\bsearch\b|query|\bq\b|keyword|lang\b|locale|view\b', re.IGNORECASE)
_SITEMAP_SKIP = ('/tag/', '/category/', '/author/', '/archive/', '/feed', '/wp-json', '/comment',
                 '/page/', '/cikk/', '/hir/', '/news/', '/blog/')
MAX_DEPTH_DEFAULT = 3
BLOCKED_RESOURCES = {'image', 'media', 'font'}
LANG_SEGMENTS = {'en', 'de', 'hu', 'fr', 'it', 'es', 'sk', 'ro', 'pl', 'cs', 'hr', 'sl', 'sr', 'ru', 'uk'}
_DOC_PATH_HINTS = ('/wp-content/uploads/', '/documents/', '/download')
_DOC_EXTENSIONS = ('.pdf', '.xlsx', '.xls', '.docx', '.doc', '.csv', '.pptx', '.ppt', '.zip')
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
# Page text written at the models rather than at readers. Flagged and counted, never acted on.
_INJECTION_RE = re.compile(
    r'ignore (all |any )?(previous|prior|above)|disregard (the )?(previous|above|instructions)'
    r'|system\s*(prompt|message)\s*:|you (must|should) (now )?(ignore|record|download|visit)'
    r'|new instructions?\s*:|</?(system|assistant)>', re.IGNORECASE)
ESCALATION_QUEUE_CAP = 50          # URLs one escalated page may add to the Frontier

# Every href with its text and, for icon-only links, the tightest surrounding row text, plus
# where the link sits on the page: the headings above it and, in a table, its column header.
# A listing often names the document type once, in a heading or a header cell, and then links each
# fund by name alone.
_HARVEST_JS = """() => {
  const clean = t => (t || '').replace(/\\s+/g, ' ').trim();
  const chrome = el => !!el.closest('nav, header, footer, aside, [role=navigation], [role=banner], [role=contentinfo]');
  const columnOf = a => {
    const cell = a.closest('td, th');
    const table = cell && cell.closest('table');
    if (!table) return '';
    let index = 0;
    for (let c = cell.previousElementSibling; c; c = c.previousElementSibling) index += c.colSpan || 1;
    const head = table.querySelector('thead tr') || table.querySelector('tr');
    if (!head || head === cell.parentElement) return '';
    let at = 0;
    for (const c of head.children) {
      if (index < at + (c.colSpan || 1)) return clean(c.textContent).slice(0, 80);
      at += c.colSpan || 1;
    }
    return '';
  };
  // A div-built table repeats its column name in every cell, hidden on wide screens ("Reports",
  // "Transcript", "Slides" beside three links that all read "PDF"). The cell's own label is the column.
  const cellLabelOf = a => {
    let el = a.parentElement;
    for (let depth = 0; el && depth < 4; depth++, el = el.parentElement) {
      if (el.matches('td, th, tr, table, [role=row], li')) return '';
      const prev = el.previousElementSibling;
      // the label cell is the value cell's twin (same class): an ordinary paragraph before a link is not a column
      if (!prev || prev.querySelector('a') || !el.className || prev.className !== el.className) continue;
      const t = clean(prev.textContent);
      if (t && t.length <= 30 && !/\\d/.test(t)) return t;
    }
    return '';
  };
  // A link inside a tab panel belongs to that tab: a site's tenders and its requests for
  // quotation can be two tables in one page, told apart only by the "Tenders" / "Request for Quotations"
  // buttons that show them. The tab is found by ARIA, by Bootstrap's target, or by the panel's own id.
  const panels = new Map();
  const tabFor = el => {
    const id = el.id;
    const lab = el.getAttribute('aria-labelledby');
    let tab = lab ? document.getElementById(lab.split(/\\s+/)[0]) : null;
    if (!tab && id) {
      const e = CSS.escape(id);
      tab = document.querySelector(`[aria-controls="${e}"], [data-bs-target="#${e}"], [data-target="#${e}"], a[href="#${e}"]`);
      const stem = id.replace(/[-_]?(table|panel|pane|tab|content|list|section|wrapper)$/i, '');
      if (!tab && stem && stem !== id) {
        tab = document.getElementById('tab-' + stem) || document.getElementById(stem + '-tab')
          || [...document.querySelectorAll('button, [role=tab]')].find(b => (b.getAttribute('onclick') || '').includes(`'${stem}'`));
      }
    }
    const t = tab ? clean(tab.textContent) : '';
    return t.length <= 60 ? t : '';
  };
  const panelOf = a => {
    for (let el = a.parentElement; el && el !== document.body; el = el.parentElement) {
      if (!el.id && el.getAttribute('role') !== 'tabpanel') continue;
      if (!panels.has(el)) panels.set(el, tabFor(el));
      if (panels.get(el)) return panels.get(el);
    }
    return '';
  };
  // Links that share a tag path are siblings: one table column, one menu, one card grid.
  const pathOf = a => {
    const parts = [];
    for (let el = a; el && el.tagName && parts.length < 8; el = el.parentElement) {
      const cls = (el.getAttribute('class') || '').trim().split(/\\s+/)[0];
      parts.push(el.tagName.toLowerCase() + (cls ? '.' + cls : ''));
    }
    return parts.reverse().join('/');
  };
  const out = [];
  const trail = [];          // headings above the current node, outermost first
  for (const el of document.querySelectorAll('h1, h2, h3, h4, h5, h6, summary, caption, a[href]')) {
    if (el.tagName !== 'A') {
      const text = clean(el.textContent).slice(0, 80);
      if (!text || chrome(el)) continue;
      const level = /^H[1-6]$/.test(el.tagName) ? +el.tagName[1] : 7;
      while (trail.length && trail[trail.length - 1].level >= level) trail.pop();
      trail.push({level, text});
      continue;
    }
    const a = el;
    const href = a.href;
    if (!href || href.startsWith('javascript:') || href.startsWith('mailto:') || href.startsWith('tel:')) continue;
    const name = clean(a.textContent) || clean(a.getAttribute('aria-label')) || clean(a.getAttribute('title'));
    let ctx = '';
    for (let el = a.parentElement, i = 0; el && i < 6 && !ctx; el = el.parentElement, i++) {
      const t = clean(el.innerText || el.textContent);
      if (t.length > name.length + 10) ctx = t.slice(0, 160);
    }
    const section = chrome(a) ? '' : [...trail.slice(-2).map(h => h.text), panelOf(a)].filter(Boolean).join(' > ');
    // the row's dated line, wherever it sits in the row: the year facet reads it, not the model
    let dated = '';
    for (let el = a.parentElement, i = 0; el && i < 4 && !dated; el = el.parentElement, i++) {
      const m = clean(el.innerText || el.textContent).match(/\\b20\\d\\d\\. ?(?:\\d{1,2}\\.|[a-záéíóöőúüű]+ \\d{1,2}\\.)|\\b\\d{1,2}[./]\\d{1,2}[./]20\\d\\d\\b|\\b20\\d\\d-\\d\\d-\\d\\d\\b/);
      if (m) dated = m[0];
    }
    out.push({href, name, context: ctx, section, column: columnOf(a) || cellLabelOf(a), path: pathOf(a), dated});
  }
  return out;
}"""

# One "load more" control, clicked. A listing that shows its first items and a button is the same case as
# scroll-to-load: the button only reveals. It must not navigate (no real href) and must not submit (not in
# a form); it is recognised by its own or its parent's class, id or test id, or by its label.
_LOAD_MORE_JS = """() => {
  const label = /^\\s*(load|show|view|see|read)\\s+more|more results|tov[aá]bbi|tov[aá]bb|t[oö]bb|bet[oö]lt|mehr\\s+(laden|anzeigen|erfahren)|weitere|cargar m[aá]s|ver m[aá]s|mostrar m[aá]s|voir plus|afficher plus/i;
  const hint = /load.?more|show.?more|more.?results|loadmore/i;
  for (const el of document.querySelectorAll('button, [role=button], a')) {
    const href = el.getAttribute('href');
    if (href && href !== '#' && !href.startsWith('javascript:')) continue;
    if (el.closest('form, nav, header, footer') || el.disabled) continue;
    const box = el.getBoundingClientRect();
    if (!box.width || !box.height || getComputedStyle(el).visibility === 'hidden') continue;
    const text = (el.textContent || '').replace(/\\s+/g, ' ').trim();
    const own = [el.className, el.id, el.getAttribute('data-testid'), el.parentElement && el.parentElement.className]
      .filter(v => typeof v === 'string').join(' ');
    if (hint.test(own) || (text.length < 40 && label.test(text))) { el.click(); return text || own; }
  }
  return null;
}"""
MAX_LOAD_MORE_CLICKS = 12
# A download control with no href: a <button> whose label says download. Clicking one is
# reveal-only - it opens a file, it does not navigate the page - and the download or popup event it
# raises carries the URL the site's script built, e.g. the /download prefix one fund manager's site puts in
# front of every path in its JSON. A few per page teach the rest.
_DOWNLOAD_BUTTONS_JS = """() => [...document.querySelectorAll('button, [role=button]')]
  .filter(el => !el.closest('a[href], form, nav, header, footer') && !el.disabled)
  .filter(el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
  .filter(el => /let[oö]lt|download|herunterladen|t[eé]l[eé]charger|descargar|\\bpdf\\b/i.test(
    (el.textContent || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')))
  .slice(0, %d).map((el, i) => { el.setAttribute('data-jev-dl', i); return (el.textContent || '').trim().slice(0, 40); })"""
MAX_DOWNLOAD_CLICKS = 3
EMPTY_PAGE_EXTRA_WAIT_MS = 8_000   # an app shell with no links yet: its data arrives in a chain of XHRs with
                                   # quiet gaps between them, so settle()'s 450 ms of quiet is too short
# Only where the button truncates a listing. Clicked on every page it cost one site a large share of its
# pages (each fund page has one, revealing a few links nobody asked for) and another many rounds on a
# news category.
LOAD_MORE_KINDS = ('seed', 'document_listing')

# Visible controls + whether the page has a text field that is not a site search, for the hidden-docs question.
_CONTROLS_JS = """() => {
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const clean = t => (t || '').replace(/\\s+/g, ' ').trim();
  const controls = [];
  for (const el of document.querySelectorAll('button, [role=button], [role=tab], summary, select, input')) {
    if (!vis(el) || controls.length >= 60) continue;
    if (el.tagName === 'SELECT') controls.push('dropdown: ' + clean(el.getAttribute('aria-label') || el.name) +
        ' [' + [...el.options].slice(0, 12).map(o => clean(o.textContent)).join(' | ') + ']');
    else if (el.tagName === 'INPUT') { if (['text', 'search', 'date', 'number', ''].includes(el.type))
        controls.push('text field: ' + clean(el.placeholder || el.name || el.getAttribute('aria-label'))); }
    else controls.push((el.getAttribute('role') || el.tagName.toLowerCase()) + ': ' + clean(el.textContent).slice(0, 60));
  }
  // a filter narrows what a listing shows: the newest item is visible and the rest are hidden
  let hasFilter = false;
  for (const el of document.querySelectorAll('select, [role=listbox], [aria-haspopup=listbox], [role=combobox]')) {
    if (vis(el) || el.getAttribute('aria-haspopup')) { hasFilter = true; break; }
  }
  let typedForm = false;
  for (const inp of document.querySelectorAll('input[type=text], input:not([type]), input[type=number], textarea')) {
    if (!vis(inp) || inp.closest('header, nav, [role=search], form[role=search]')) continue;
    const n = ((inp.name || '') + ' ' + (inp.placeholder || '') + ' ' + (inp.id || '')).toLowerCase();
    if (/search|keres|suche|query|^s$|\\bq\\b/.test(n)) continue;
    typedForm = true; break;
  }
  return {controls, typedForm, has_filter: hasFilter, text: clean(document.body ? document.body.innerText : '').slice(0, 1500)};
}"""


# The filters of a listing: native <select>s, and ARIA selects - a trigger plus the listbox it
# opens, tied by aria-controls, a shared label, or a shared parent. One bank's year filter is the second
# kind, and its values never appear in _CONTROLS_JS, which lists only the (empty) trigger button.
# Each trigger is tagged data-ml-id="jf<i>" so select_option_anywhere can drive it.
_FILTERS_JS = r"""() => {
  const clean = t => (t || '').replace(/\s+/g, ' ').trim();
  const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const byIds = ids => clean((ids || '').split(/\s+/).map(i => document.getElementById(i)).filter(Boolean)
    .map(e => e.textContent).join(' '));
  const labelOf = el => clean(el.getAttribute('aria-label')) || byIds(el.getAttribute('aria-labelledby'))
    || (el.id && clean((document.querySelector(`label[for="${CSS.escape(el.id)}"]`) || {}).textContent)) || clean(el.name);
  const out = [];
  document.querySelectorAll('select').forEach(el => {
    if (!vis(el) || el.options.length < 2) return;
    out.push({el, label: labelOf(el), current: clean(el.selectedOptions[0] && el.selectedOptions[0].textContent),
              options: [...el.options].map(o => clean(o.textContent)).filter(Boolean)});
  });
  document.querySelectorAll('[aria-haspopup=listbox], button[aria-expanded], [role=combobox]').forEach(el => {
    if (!vis(el)) return;
    let box = el.getAttribute('aria-controls') && document.getElementById(el.getAttribute('aria-controls'));
    const lab = el.getAttribute('aria-labelledby');
    if (!box && lab) box = [...document.querySelectorAll('[role=listbox]')].find(b => b !== el && b.getAttribute('aria-labelledby') === lab);
    if (!box && el.parentElement) box = el.parentElement.querySelector('[role=listbox]');
    if (!box) return;
    const options = [...box.querySelectorAll('[role=option], li')].map(o => clean(o.textContent)).filter(Boolean);
    if (options.length < 2) return;
    out.push({el, label: labelOf(el), current: clean(el.textContent), options});
  });
  return out.slice(0, 8).map((f, i) => { f.el.setAttribute('data-ml-id', 'jf' + i);
    return {id: 'jf' + i, label: f.label.slice(0, 80), current: f.current.slice(0, 80),
            options: f.options.slice(0, 60).map(o => o.slice(0, 80))}; });
}"""

# The filters' own apply button, if they have one: some sites set nothing until "Keresés" is pressed.
# Searched outward from the smallest element holding every filter, so a site-wide search box loses to it.
_FILTER_SUBMIT_JS = r"""() => {
  const fs = [...document.querySelectorAll('[data-ml-id^=jf]')];
  if (!fs.length) return null;
  let box = fs[0].parentElement;
  while (box && !fs.every(f => box.contains(f))) box = box.parentElement;
  const words = /^(keres|search|szűr|filter|apply|alkalmaz|suche|suchen|anwenden|filtern|ok$|go$|mehet|mutat|show|buscar|filtrar)/i;
  for (let el = box; el; el = el.parentElement) {
    const btn = [...el.querySelectorAll('button, input[type=submit], [role=button]')].find(b =>
      !b.hasAttribute('data-ml-id') && b.getAttribute('aria-haspopup') !== 'listbox' &&
      (b.type === 'submit' || words.test((b.textContent || b.value || '').trim())));
    if (btn) { btn.setAttribute('data-ml-id', 'jf-submit'); return (btn.textContent || btn.value || '').trim(); }
    if (el === document.body) break;
  }
  return null;
}"""

def canonical(url: str) -> str:
    return _strip_fragment(url).rstrip('/').lower()


def clean_text(text: str) -> str:
    """Strip control characters from page-supplied text before it reaches a model."""
    return _CONTROL_CHARS.sub(' ', text or '')


def looks_like_instructions(text: str) -> bool:
    """True when page text reads like an instruction aimed at a model, not at a human."""
    return bool(_INJECTION_RE.search(text or ''))


def looks_like_document(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_DOC_EXTENSIONS) or any(h in path for h in _DOC_PATH_HINTS)


# Letters that belong to one language among LANG_SEGMENTS, and each language's common short words.
_LANG_LETTERS = {'hu': 'őű', 'de': 'ß', 'es': 'ñ¿¡', 'pl': 'łąęśźż', 'cs': 'ěř', 'ro': 'ășțşţ', 'sk': 'ľĺŕ',
                 'uk': 'іїєґ', 'ru': 'ыэъё'}
_LANG_WORDS = {
    'en': {'the', 'of', 'and', 'all', 'for', 'from', 'find', 'every', 'with', 'in'},
    'de': {'der', 'die', 'das', 'und', 'alle', 'vom', 'von', 'für', 'zum', 'zur', 'mit', 'den', 'dem', 'des'},
    'hu': {'az', 'és', 'egy', 'meg', 'le', 'minden', 'összes', 'hogy', 'szóló', 'éves', 'havi'},
    'fr': {'les', 'des', 'du', 'et', 'tous', 'toutes', 'pour', 'la', 'trouver'},
    'es': {'el', 'los', 'las', 'del', 'y', 'todos', 'todas', 'para', 'encuentra'},
    'it': {'il', 'gli', 'dei', 'delle', 'della', 'tutti', 'tutte', 'per', 'trova'},
}


def goal_language(goal: str) -> str | None:
    """The goal's language among LANG_SEGMENTS, or None when the goal does not say (navigation only: pages under
    another language's path segment are visited later)."""
    text = goal.lower()
    scores = {lang: 3 * sum(text.count(c) for c in letters) for lang, letters in _LANG_LETTERS.items()}
    words = re.findall(r'\w+', text)
    for lang, common in _LANG_WORDS.items():
        scores[lang] = scores.get(lang, 0) + sum(w in common for w in words)
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    return best if ranked[0] > 0 and ranked[0] > ranked[1] else None


def url_language(url: str) -> str | None:
    """A leading language path segment, e.g. /en/investment-funds -> 'en'. None means the site default."""
    first = urlparse(url).path.strip('/').split('/')[0].lower()
    return first if first in LANG_SEGMENTS else None


_DATE_LINE_RE = re.compile(r'\b(20\d\d)\. ?(?:\d{1,2}\.|január|február|március|április|május|június|július|'
                           r'augusztus|szeptember|október|november|december)|\b\d{1,2}[./]\d{1,2}[./](20\d\d)\b'
                           r'|\b(20\d\d)-\d\d-\d\d\b')


def year_of(text: str) -> str | None:
    """The year of a dated line ('2022. május 12.', '12.05.2022', '2022-05-12') read by code and reported as a
    Facet. A goal that names a year is scoped by the consumer on this, never by the relevance question:
    a decision's row also carries case numbers from other years, and evaluation showed the model should not judge
    periods. Falls back to the year of a period_of match."""
    m = _DATE_LINE_RE.search(text or '')
    if m:
        return next(g for g in m.groups() if g)
    period = period_of(text)
    return period[:4] if period else None






def _escalation_trigger(judge, goal, url, title, kind, docs, new_pages, accepted, controls, record,
                        crawl_accepted: int = 0):
    """Escalate where escalation has been observed to pay.

    In evaluation, `empty_page` escalations have never returned a Candidate, and neither have
    hidden-document escalations on pages that already yielded one. The single case that paid was a
    document listing behind a filter, where the page shows the newest item and hides the rest.

    Re-measured on a wider set of sites: still only `filtered_listing` paid. `typed_form` escalations on
    a fund manager's report listing (slow and token-heavy, after the crawl had found every document) and
    `empty_page` escalations on a council meeting portal returned nothing. So: a page with a text field only
    escalates while the crawl has found nothing at all, and an empty page only when it is also a listing
    or fund page - the kinds where documents could be hidden - not a news item or a legal page.
    """
    filtered_listing = kind == 'document_listing' and controls.get('has_filter')
    if controls['typedForm'] and not docs and not crawl_accepted:
        return 'typed_form'
    if not filtered_listing and not (not docs and new_pages < 5) and accepted:
        return None                      # a page that produced documents and hides no filter
    hidden = judge.hides_documents(goal, {
        'page_url': url, 'page_title': title, 'documents_already_found': len(docs),
        'controls': [clean_text(c) for c in controls['controls']],
        'visible_text_excerpt': clean_text(controls['text'])})
    record['hides_documents'] = hidden
    if hidden is None:
        return None
    if filtered_listing and hidden >= FILTERED_LISTING_AT and len(docs) <= THIN_LISTING_DOCUMENTS:
        return 'filtered_listing'
    # a page with nothing harvested is worth one cheap look, but only when Jev is confident and the
    # page is of a kind that lists documents
    if not docs and hidden >= 0.9 and kind in ('document_listing', 'fund_or_product') and not crawl_accepted:
        return 'empty_page'
    return None


def escalation_skip_reason(now: float, crawl_deadline: float, tokens_spent: int, token_cap: int) -> str | None:
    """Why an escalation must not start, or None.

    max_seconds is checked between batches, so an escalation started late ran the crawl past it by a
    wide margin. The escalation itself also stops at the deadline.
    """
    if now > crawl_deadline - ESCALATION_MIN_SECONDS:
        return 'time'
    if tokens_spent >= token_cap:
        return 'tokens'
    return None


def should_stop_for_no_progress(stale_pages: int, pages_done: int, max_pages: int,
                                elapsed: float, max_seconds: float) -> bool:
    """Give up only after a run of empty pages AND a fair share of the budget.

    Stopping on the run alone ended crawls with most of their page budget unspent.
    """
    spent = max(pages_done / max_pages if max_pages else 1.0,
                elapsed / max_seconds if max_seconds else 1.0)
    return stale_pages >= STALE_PAGES_STOP and spent >= STALE_MIN_BUDGET_FRACTION


def sitemap_page_urls(urls: list[str]) -> list[str]:
    """Sitemap page URLs worth a page-kind question: content paths only, capped."""
    keep = [u for u in urls
            if not looks_like_document(u) and not _is_binary(u)
            and not any(skip in urlparse(u).path.lower() for skip in _SITEMAP_SKIP)]
    return keep[:SITEMAP_PAGES_SCORED]


def paging_identity(url: str, keep: frozenset[tuple[str, str]] = frozenset()) -> str:
    """What a URL is, once its paging and facet parameters are dropped: the key the variant cap counts on.

    `keep` is the seed's own facets. A caller who seeds a filtered listing means that listing, so a page
    that shares the seed's filter (its next page) keeps a distinct identity from the unfiltered one.
    """
    parsed = urlparse(url)
    stable = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                    if (k, v) in keep or (not _PAGING_PARAM.search(k) and not _FACET_PARAM.search(k)))
    return parsed.path.rstrip('/').lower() + ('?' + urlencode(stable) if stable else '')


def jev_crawl(
    start_url: str,
    goal: str,
    *,
    api_key: str | None = None,
    judge: RelevanceJudge | None = None,
    agent: LLMClient | None = None,
    same_domain_only: bool = True,
    allowed_hosts: list[str] | None = None,
    include_rejected: bool = False,
    max_pages: int = 40,
    max_seconds: float = 180.0,
    max_depth: int = MAX_DEPTH_DEFAULT,
    parallel_pages: int = 2,
    frontier_policy: str = 'tier',
    agent_token_cap: int = AGENT_TOKEN_CAP,
    user_agent: str = _DEFAULT_USER_AGENT,
    headless: bool = True,
    on_event: Callable[[dict], None] | None = None,
    on_trace: Callable[[dict], None] | None = None,
    recipes_dir: str | None = None,
) -> AgenticCrawlResult:
    """Crawl `start_url` for documents matching `goal`, with a Relevance judge (crawler.judge: Jev or any LLM)
    making the per-page judgements. Without one, make_judge() picks from the environment.

    same_domain_only=True (the default) keeps every page and Candidate on the seed host. With it False,
    the off-domain policy applies: Candidates may come from any host, and navigation may cross to
    one host outside the seed (hosts in `allowed_hosts` qualify from the start), then crawl within that
    host without ever chaining to a third one.

    `on_trace` is for the eval harness only: it receives each page's HTML and harvested links, and
    every page-kind answer, so a ranking change can be replayed offline. It is called from the page
    threads and changes nothing about the crawl.
    """
    started = time.perf_counter()
    crawl_deadline = started + max_seconds
    judge = judge or make_judge()
    emit = on_event or (lambda ev: None)
    agent = agent or agent_llm(None, api_key)     # the LLM an Escalation (and the agent fallback) runs on
    language = goal_language(goal)
    lock = threading.Lock()

    try:
        final_url = str(httpx.get(start_url, headers={'User-Agent': user_agent}, follow_redirects=True,
                                  timeout=15).url)
    except httpx.HTTPError:
        final_url = start_url
    seed = {'host': urlparse(final_url).netloc, 'resolved': False}
    policy = OffDomainPolicy(seed['host'], same_domain_only=same_domain_only, allowed_hosts=allowed_hosts)
    same_site = policy.is_seed
    may_visit, may_return = policy.may_visit, policy.may_return

    # 'tier' is the default order. 'rescue' switches to what has paid once the crawl runs dry: it
    # rescues a stalled crawl on some sites but cost a document on others, so a request opts in.
    frontier = Frontier(frontier_policy)
    probe = DocumentProbe(user_agent, lambda u: is_crawlable(u) and may_return(u))
    queued_from: dict[str, tuple[dict, str | None]] = {}   # a queued link and the page it was found on
    json_prefixes: set[str] = set()      # what this site puts in front of the relative paths in its JSON
    recipes = RecipeStore(recipes_dir) if recipes_dir else None   # replay paid escalations
    classified: set[str] = set()
    paths_queued: collections.Counter[str] = collections.Counter()
    visited: set[str] = set()
    candidates: dict[str, AgenticDownload | None] = {}
    tokens = {'prompt': 0, 'completion': 0, 'cache_read': 0, 'cache_creation': 0}
    escalations: dict[str, int] = {}
    skipped: list[str] = []
    guard = {'unsafe_urls': 0, 'robots_blocked': 0, 'instruction_like_text': 0}
    pages: list[dict] = []
    progress = {'stale': 0, 'count': 0}

    def add_candidates(links: list[dict], source: str, page_url: str, page_ctx: str,
                       neighbours: list[dict] = ()) -> tuple[int, int]:
        """Judge Relevance for links not seen yet. Returns (accepted, kept) where kept excludes rejected."""
        with lock:
            safe = []
            for c in links:
                if not _is_safe_url(c['url']):
                    guard['unsafe_urls'] += 1        # never judged, never returned
                    continue
                if not may_return(c['url']):
                    continue                          # off-domain Candidate while same_domain_only
                safe.append(c)
            fresh = [c for c in safe if c['url'] not in candidates]
            for c in fresh:
                candidates[c['url']] = None
        if not fresh:
            return 0, 0
        scores = judge.relevance(goal, page_ctx, fresh, neighbours)
        accepted = kept = 0
        for c, score in zip(fresh, scores):
            verdict = verdict_for(score)
            download = AgenticDownload(
                url=c['url'], name=c.get('name') or c['url'].rsplit('/', 1)[-1],
                reason=c.get('reason', f'{source} candidate, verdict {verdict}'), source_page=page_url,
                relevance=score, verdict=verdict, source=source,
                period=document_period(c.get('name', ''), c['url'])[0],
                year=year_of(f"{c.get('dated', '')} {c.get('name', '')} {c.get('context', '')} {c['url']}"),
            )
            with lock:
                candidates[c['url']] = download
            if verdict != 'rejected':
                kept += 1
                emit({'type': 'agent_download', 'url': download.url, 'name': download.name,
                      'verdict': verdict, 'relevance': score, 'source': source})
            accepted += verdict == 'accepted'
        return accepted, kept

    def add_pages(links: list[dict], depth: int, parent: str | None = None) -> int:
        """Classify unseen page links and put them in the Frontier."""
        with lock:
            fresh, seen = [], set()
            for c in links:
                key = canonical(c['url'])
                if key in classified or key in visited or key in seen or '[' in c['url']:
                    continue
                if not _is_safe_url(c['url']):
                    guard['unsafe_urls'] += 1
                    continue
                if not _url_allowed_by_robots(c['url']):
                    guard['robots_blocked'] += 1     # the site asked crawlers not to
                    continue
                seen.add(key)
                fresh.append(c)
            classified.update(canonical(c['url']) for c in fresh)
        if not fresh:
            return 0
        kinds = judge.page_kinds(goal, fresh)
        if on_trace:
            on_trace({'type': 'page_kinds', 'depth': depth, 'links': [
                {'url': c['url'], 'kind': kind, 'probability': probability}
                for c, (kind, probability) in zip(fresh, kinds)]})
        queued = 0
        with lock:
            for c, (kind, probability) in zip(fresh, kinds):
                path = paging_identity(c['url'], seed_facets)
                variants = paths_queued[path]
                if variants >= MAX_VARIANTS_PER_PATH:
                    continue                      # the rest is the same page, paginated
                paths_queued[path] += 1
                other = language is not None and url_language(c['url']) not in (None, language)
                queued_from[c['url']] = (c, parent)
                frontier.add(c['url'], kind=kind, probability=probability, depth=depth, other_language=other,
                             path_seen=variants > 0, group=c.get('path', ''), parent=parent)
                queued += 1
        return queued

    def split_links(raw: list[dict], page_url: str) -> tuple[list[dict], list[dict]]:
        """Merge duplicate hrefs (icon + text + row link), then split documents from pages."""
        merged: dict[str, dict] = {}
        for link in raw:
            url = _strip_fragment(link.get('href', ''))
            entry = merged.setdefault(url, {'url': url, 'name': '', 'context': '', 'section': '', 'column': '',
                                            'path': link.get('path') or '', 'dated': link.get('dated') or ''})
            name = clean_text(link.get('name') or '')
            context = clean_text(link.get('context') or '')
            if looks_like_instructions(name) or looks_like_instructions(context):
                with lock:
                    guard['instruction_like_text'] += 1
                logger.warning('[jev] instruction-like link text on %s: %r', url, (name or context)[:120])
            if len(name) > len(entry['name']):
                entry['name'] = name
            if context and (not entry['context'] or len(context) < len(entry['context'])):
                entry['context'] = context
            entry['section'] = entry['section'] or clean_text(link.get('section') or '')
            entry['column'] = entry['column'] or clean_text(link.get('column') or '')
        docs, page_links = [], []
        for entry in merged.values():
            if not entry['url'].startswith('http'):
                continue
            if looks_like_document(entry['url']):
                if may_return(entry['url']):
                    docs.append(entry)     # documents may live on any host
            elif may_visit(entry['url'], page_url) and not _is_binary(entry['url']):
                page_links.append(entry)
        return docs, page_links

    # --- pre-crawl: the site's own backend and its sitemap are Sources too ----------
    api_candidates = 0
    try:
        mined = mine_documents(final_url, user_agent=user_agent, crawl_plan=None, source_page=final_url)
        api_candidates = len(mined)
        add_candidates([{'url': d.url, 'name': d.name} for d in mined], 'api', final_url, 'site backend catalogue')
    except Exception as exc:  # noqa: BLE001 - mining is best-effort
        logger.warning('[jev] api mining failed: %s', exc)
    try:
        sitemap_urls = [u for u in fetch_sitemap(final_url) if same_site(u)][:SITEMAP_CAP]  # seed host only
    except Exception as exc:  # noqa: BLE001
        logger.warning('[jev] sitemap fetch failed: %s', exc)
        sitemap_urls = []
    sitemap_docs = [{'url': u, 'name': ''} for u in sitemap_urls if looks_like_document(u)]
    add_candidates(sitemap_docs, 'sitemap', final_url, 'sitemap')
    # the seed's own facets are the caller's intent (a date range, a document type): pages that share
    # them are the listing the caller asked for, not variants of the unfiltered one
    seed_facets = frozenset((k, v) for k, v in parse_qsl(urlparse(final_url).query, keep_blank_values=True)
                            if _FACET_PARAM.search(k) and not _PAGING_PARAM.search(k))
    frontier.add_seed(final_url)
    # Every sitemap document is judged, but only the first N page URLs are worth a question: scoring a large
    # sitemap's page URLs once took a big share of the time budget and found nothing.
    sitemap_pages = sitemap_page_urls(sitemap_urls)
    add_pages([{'url': u, 'name': ''} for u in sitemap_pages], 0)
    emit({'type': 'jev_pre_crawl', 'api_candidates': api_candidates, 'sitemap_urls': len(sitemap_urls),
          'sitemap_documents': len(sitemap_docs), 'sitemap_pages_scored': len(sitemap_pages),
          'frontier': len(frontier)})

    # --- browser sessions ----------------------------------------------------------------
    local = threading.local()

    def session():
        if not hasattr(local, 'page'):
            local.playwright = sync_playwright().start()
            local.browser = local.playwright.chromium.launch(
                headless=headless, args=['--no-sandbox', '--disable-setuid-sandbox'] if headless else [])
            local.context = local.browser.new_context(user_agent=user_agent, accept_downloads=True)
            local.inflight = set()
            local.json_bodies = []        # first-party JSON the current page fetched for itself

            def on_response(response):
                request = response.request
                if request.resource_type not in ('xhr', 'fetch'):
                    return
                if 'json' not in (response.headers.get('content-type') or '').lower():
                    return
                if bare_host(urlparse(response.url).netloc) not in policy.first_party_hosts:
                    return
                try:
                    if int(response.headers.get('content-length') or 0) > MAX_JSON_BYTES:
                        return
                    local.json_bodies.append((response.url, response.text()))
                except Exception:  # noqa: BLE001 - a body that is gone by now is not worth a page
                    pass

            def route(route_obj, request):
                kind = request.resource_type
                third_party = bare_host(urlparse(request.url).netloc) not in policy.first_party_hosts
                if kind in BLOCKED_RESOURCES \
                        or (third_party and kind in ('xhr', 'fetch', 'ping', 'beacon', 'eventsource')) \
                        or (kind == 'script' and urlparse(request.url).netloc in _TRACKING_SCRIPT_HOSTS):
                    route_obj.abort()
                else:
                    route_obj.continue_()

            local.context.route('**/*', route)
            local.page = local.context.new_page()
            for event in ('request', 'requestfinished', 'requestfailed'):
                local.page.on(event, (lambda event: lambda req: (
                    req.resource_type in ('xhr', 'fetch')
                    and (local.inflight.add(req) if event == 'request' else local.inflight.discard(req))))(event))
            local.page.on('response', on_response)
            local.cookies_done = False
        return local

    def settle(sess, max_ms: int = 15_000) -> int:
        """Loaded = no first-party XHR in flight and the link count held for ~450 ms."""
        start, last, stable = time.perf_counter(), -1, 0
        while (time.perf_counter() - start) * 1000 < max_ms:
            count = sess.page.evaluate('document.querySelectorAll("a[href]").length')
            stable = stable + 1 if (count == last and not sess.inflight) else 0
            if stable >= 3:
                return count
            last = count
            sess.page.wait_for_timeout(150)
        return last

    def filter_reveal(sess, url: str, title: str, harvested: list[str], shown: set[str]) -> tuple[list[dict], dict]:
        """Set the filter values Jev picks, in code, and harvest again. Returns the documents that
        appeared and what was done; nothing appears when there is no filter or Jev keeps them all."""
        filters = sess.page.evaluate(_FILTERS_JS)
        info: dict = {'filters': len(filters)}
        if not filters:
            return [], info
        picks = judge.filter_values(goal, {'page_url': url, 'page_title': title,
                                         'documents_listed_now': harvested[:15]}, filters)
        label = {f['id']: f['label'] or f['id'] for f in filters}
        info['set'] = {label[fid]: {'value': value, 'p': round(p, 2)} for fid, (value, p) in picks.items()}
        if not picks:
            return [], info
        for fid, (value, _) in picks.items():
            sess.page.wait_for_timeout(150)
            select_option_anywhere(sess.page, fid, value, 1500)
        info['submit'] = sess.page.evaluate(_FILTER_SUBMIT_JS)
        if info['submit']:
            sess.page.locator('[data-ml-id="jf-submit"]').first.click(timeout=5000)
        settle(sess, 8000)
        again, _ = split_links(sess.page.evaluate(_HARVEST_JS), url)
        return [d for d in again if d['url'] not in shown], info

    def click_downloads(sess) -> list[str]:
        """URLs that a few href-less download buttons open when clicked. Nothing is saved."""
        labels = sess.page.evaluate(_DOWNLOAD_BUTTONS_JS % MAX_DOWNLOAD_CLICKS)
        if not labels:
            return []
        urls: list[str] = []
        opened: list = []
        on_download = lambda download: (urls.append(download.url), download.cancel())   # noqa: E731
        on_popup = lambda popup: opened.append(popup)                                  # noqa: E731
        sess.page.on('download', on_download)
        sess.context.on('page', on_popup)
        try:
            for i in range(len(labels)):
                sess.page.evaluate(f'document.querySelector("[data-jev-dl=\'{i}\']")?.click()')
                sess.page.wait_for_timeout(1200)
            for popup in opened:
                try:
                    popup.wait_for_load_state('commit', timeout=3000)
                    if popup.url.startswith('http'):
                        urls.append(popup.url)
                finally:
                    popup.close()
        except Exception as exc:  # noqa: BLE001 - a click that throws ends the reveal, not the page
            logger.debug('[jev] download click failed: %s', exc)
        finally:
            sess.page.remove_listener('download', on_download)
            sess.context.remove_listener('page', on_popup)
        return [u for u in urls if u.startswith('http')]

    def reveal(sess, kind: str) -> dict:
        """Reveal-only interactions. Never types, never submits anything but a modal gate."""
        done = {'gate': bool(_try_dismiss_form_disclaimer(sess.page, wait_ms=300)), 'rows': 0, 'scrolls': 0,
                'more': 0}
        if done['gate']:
            settle(sess, 5000)
        try:
            rows = sess.page.locator('tr[tabindex]')
            for i in range(min(rows.count(), 30)):
                row = rows.nth(i)
                if row.evaluate('el => getComputedStyle(el).cursor') == 'pointer':
                    row.dispatch_event('click')
                    done['rows'] += 1
            if done['rows']:
                settle(sess, 5000)
        except Exception as exc:  # noqa: BLE001 - row expansion is best-effort
            logger.debug('[jev] row expansion failed: %s', exc)
        count = sess.page.evaluate('document.querySelectorAll("a[href]").length')
        for _ in range(3):
            sess.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            after = settle(sess, 3000)
            if after <= count:
                break
            count, done['scrolls'] = after, done['scrolls'] + 1
        try:
            for _ in range(MAX_LOAD_MORE_CLICKS if kind in LOAD_MORE_KINDS else 0):
                if not sess.page.evaluate(_LOAD_MORE_JS):
                    break
                after = settle(sess, 5000)
                if after <= count:
                    break                      # the click revealed nothing: stop, whatever the button was
                count, done['more'] = after, done['more'] + 1
        except Exception as exc:  # noqa: BLE001 - a click that navigates away or throws ends the reveal
            logger.debug('[jev] load-more failed: %s', exc)
        return done

    escalation_pool = ThreadPoolExecutor(1)   # the agent starts its own browser: keep it off page threads
    escalation_slot = threading.Lock()

    def escalation_budget(trigger: str) -> int:
        """What one escalated page may spend. Driving a filter needs a large budget; an empty page has never
        paid at all, so it gets a small budget."""
        rich = {'filtered_listing', 'canvas', 'typed_form'}
        return ESCALATION_TOKEN_CAP_RICH if trigger in rich else ESCALATION_TOKEN_CAP

    def escalate(url: str, depth: int, trigger: str, harvested: list[str]) -> dict:
        with escalation_slot:
            reason = escalation_skip_reason(time.perf_counter(), crawl_deadline, sum(tokens.values()),
                                            agent_token_cap)
            if reason:
                with lock:
                    skipped.append(url)
                emit({'type': 'jev_escalation_skipped', 'url': url, 'trigger': trigger, 'reason': reason})
                return {'skipped': True, 'reason': reason}
            emit({'type': 'jev_escalation', 'url': url, 'trigger': trigger})
            brief = {'type': 'text', 'text': (
                f'ESCALATION ({trigger}). Code already harvested {len(harvested)} document links from this page, '
                f'for example: {"; ".join(harvested[:15]) or "none"}. Do not re-record those. Your job is to reveal '
                'documents the goal asks for that are hidden behind page controls, record them, then call done().')}
            began = time.perf_counter()
            budget_left = min(escalation_budget(trigger), agent_token_cap - sum(tokens.values()))
            result = escalation_pool.submit(
                _visit_page, url, depth, llm=agent, goal=goal,
                system_blocks=_SYSTEM_BLOCKS + [{'type': 'text', 'text': f'Goal: {goal}'}, brief],
                seed_host=urlparse(url).netloc, same_domain_only=True, js_wait_ms=1000, click_wait_ms=1000,
                max_tool_steps=AGENT_STEPS_PER_ESCALATION, max_tokens_budget=budget_left, deadline=crawl_deadline,
                crawl_plan=None, min_url_score=0.0,
                memory_snapshot={}, open_kwargs={'js_wait_ms': 1000, 'accept_downloads': True,
                                                 'headless': headless},
                queue_size_hint=0, pre_interactions=[], on_event=None, visited_snapshot=frozenset(visited),
            ).result()
            with lock:
                tokens['prompt'] += result.prompt_tokens
                tokens['completion'] += result.completion_tokens
                tokens['cache_read'] += result.cache_read_tokens
                tokens['cache_creation'] += result.cache_creation_tokens
                escalations[trigger] = escalations.get(trigger, 0) + 1
            accepted, kept = add_candidates(
                [{'url': d.url, 'name': d.name} for d in result.downloads], 'agent', url, url)
            if recipes and kept:
                recipe = recipe_from_steps(url, goal, result.steps, result.step_elements, kept, result.step_outcomes)
                if recipe:
                    recipes.save(recipe)
                    emit({'type': 'jev_recipe_recorded', 'url': url, 'steps': len(recipe.steps)})
            # An injected page could otherwise flood the Frontier with pages we then pay to classify.
            nominated = (result.queued_urls + ([result.navigate_url] if result.navigate_url else []))
            add_pages([{'url': u, 'name': ''} for u, _ in nominated[:ESCALATION_QUEUE_CAP]
                       if may_visit(u, url)], depth + 1, url)
            return {'accepted': accepted, 'kept': kept, 'nominated': len(result.downloads),
                    'ms': round((time.perf_counter() - began) * 1000)}

    def visit(url: str, depth: int, kind: str) -> dict:
        record = {'url': url, 'depth': depth, 'kind': kind, 'accepted': 0, 'kept': 0}
        began = time.perf_counter()
        emit({'type': 'crawl_page_start', 'url': url, 'depth': depth})
        sess = session()
        trigger = None
        harvested: list[str] = []
        try:
            sess.json_bodies.clear()
            sess.page.goto(url, wait_until='domcontentloaded', timeout=30_000)
            if not sess.cookies_done:
                sess.cookies_done = True
                _try_accept_cookies(sess.page, wait_ms=500)
            settle(sess)
            if url == final_url and not seed['resolved']:
                # Follow the seed's own redirect (www -> bare) exactly once. Sitemap pages also enter
                # the Frontier at depth 0, and resolving on those would redefine the seed host mid-crawl.
                seed['host'] = urlparse(sess.page.url).netloc
                seed['resolved'] = True
                policy.seed_host = bare_host(seed['host'])
            title = sess.page.title()
            if detect_canvas_page(sess.page):
                record['canvas'] = True
                trigger = 'canvas' if api_candidates == 0 else None
            else:
                record['reveal'] = reveal(sess, kind)
                docs, page_links = split_links(sess.page.evaluate(_HARVEST_JS), url)
                if not docs and not page_links:
                    # no links at all: a single-page app still fetching its data, which arrives in a chain
                    # of XHRs with quiet gaps between them. Give it a bounded chance.
                    deadline = time.perf_counter() + EMPTY_PAGE_EXTRA_WAIT_MS / 1000
                    seen = len(sess.json_bodies)
                    while time.perf_counter() < deadline:
                        sess.page.wait_for_timeout(250)
                        if len(sess.json_bodies) > seen and not sess.inflight:
                            seen = len(sess.json_bodies)
                            deadline = min(deadline, time.perf_counter() + 2.5)   # one more quiet gap, then go
                    settle(sess, 5000)
                    docs, page_links = split_links(sess.page.evaluate(_HARVEST_JS), url)
                    record['waited_for_app'] = True
                if on_trace:
                    on_trace({'type': 'harvest', 'url': url, 'depth': depth, 'html': sess.page.content(),
                              'documents': docs, 'page_links': page_links})
                # links the URL did not give away: ask the server, one sibling group at a time
                served, page_links = probe.sort(page_links)
                docs += served
                record['probed_documents'] = len(served)
                # documents the page named in the JSON it fetched for itself, whether or not it rendered
                # them as links. Relative paths need a prefix the page's own links reveal; absolute
                # URLs are taken as given. Anything not already a document by URL is asked like a page link.
                known = {d['url'] for d in docs}
                mined = []
                clicked = click_downloads(sess) if sess.json_bodies and kind in LOAD_MORE_KINDS else []
                for u in clicked:
                    if u not in known and may_return(u) and not known.add(u):
                        docs.append({'url': u, 'name': '', 'context': '', 'section': '', 'column': '', 'path': 'click'})
                record['download_clicks'] = len(clicked)
                for _, body in sess.json_bodies:
                    mined += [c for c in candidates_from_json(body, url, [d['url'] for d in docs + page_links],
                                                              json_prefixes)
                              if c['url'] not in known and may_return(c['url']) and not known.add(c['url'])]
                if mined:
                    by_url = [c for c in mined if looks_like_document(c['url'])]
                    by_probe, _ = probe.sort_all([c for c in mined if not looks_like_document(c['url'])])
                    docs += by_url + by_probe
                record['json'] = {'responses': len(sess.json_bodies), 'named': len(mined),
                                  'documents': len(mined) and len(by_url) + len(by_probe)}
                sess.json_bodies.clear()
                harvested = [d['name'] or d['url'].rsplit('/', 1)[-1] for d in docs]
                accepted, kept = add_candidates(docs, 'page', url, f'{title} ({url})')
                record['accepted'], record['kept'] = accepted, kept
                new_pages = add_pages(page_links, depth + 1, url) if depth < max_depth else 0
                record['harvest'] = {'documents': len(docs), 'new_pages': new_pages}
                recipe = recipes.load(url) if recipes else None
                if recipe and recipe.steps:
                    # the agent has driven this page before: do what it did, in code, and harvest again
                    before = {d['url'] for d in docs}
                    steps_done = replay_recipe(sess.page, recipe, lambda: settle(sess, 5000))
                    again, more_pages = split_links(sess.page.evaluate(_HARVEST_JS), url)
                    revealed = [d for d in again if d['url'] not in before]
                    if revealed:
                        acc2, kept2 = add_candidates(revealed, 'page', url, f'{title} ({url})')
                        accepted, kept = accepted + acc2, kept + kept2
                        record['accepted'], record['kept'] = accepted, kept
                        docs += revealed
                    record['recipe'] = {'steps': steps_done, 'of': len(recipe.steps), 'revealed': len(revealed)}
                    emit({'type': 'jev_recipe_replayed', 'url': url, **record['recipe']})
                    recipe.replays += 1
                    recipe.last_revealed = len(revealed)
                    if revealed:
                        recipes.save(recipe)
                    else:
                        recipes.forget(recipe)         # the layout moved on; let the agent look again
                controls = sess.page.evaluate(_CONTROLS_JS)
                if looks_like_instructions(clean_text(controls['text'])):
                    with lock:
                        guard['instruction_like_text'] += 1
                    logger.warning('[jev] instruction-like page text on %s', url)
                with lock:
                    crawl_accepted = sum(1 for c in candidates.values() if c and c.verdict == 'accepted')
                trigger = _escalation_trigger(judge, goal, url, title, kind, docs, new_pages,
                                              accepted, controls, record, crawl_accepted)
                if trigger == 'filtered_listing':
                    # The agent is paid many tokens to set a filter; Jev picking the value costs
                    # a fraction of that and well under a second. The agent still gets the page when this reveals nothing.
                    try:
                        revealed, record['filter_reveal'] = filter_reveal(
                            sess, url, title, harvested, {d['url'] for d in docs})
                    except Exception as exc:  # noqa: BLE001 - a filter that will not drive is the agent's job
                        revealed, record['filter_reveal'] = [], {'error': str(exc)[:200]}
                    if revealed:
                        acc2, kept2 = add_candidates(revealed, 'page', url, f'{title} ({url})', docs)
                        accepted, kept = accepted + acc2, kept + kept2
                        record['accepted'], record['kept'] = accepted, kept
                        record['filter_reveal']['revealed'] = len(revealed)
                        if kept2:
                            trigger = None
                    emit({'type': 'jev_filter_reveal', 'url': url, **record['filter_reveal']})
                if (trigger == 'filtered_listing'
                        and escalations.get('filtered_listing', 0) >= MAX_FILTERED_LISTING_ESCALATIONS):
                    record['trigger_skipped'] = trigger    # the budget is better spent on more pages
                    trigger = None
        except Exception as exc:  # noqa: BLE001 - one bad page must not end the crawl
            if 'Download is starting' in str(exc):
                # The "page" is a file the probe did not catch. It cost a visit; do not also lose it.
                link, parent = queued_from.get(url, ({'url': url, 'name': ''}, None))
                accepted, kept = add_candidates([link], 'page', parent or url, parent or url)
                record.update({'accepted': accepted, 'kept': kept, 'was_document': True})
            else:
                record['error'] = str(exc)[:200]
                logger.warning('[jev] page failed %s: %s', url, exc)
        if trigger:
            record['trigger'] = trigger
            record['escalation'] = escalate(url, depth, trigger, harvested)
            record['accepted'] += record['escalation'].get('accepted', 0)
            record['kept'] += record['escalation'].get('kept', 0)
        record['ms'] = round((time.perf_counter() - began) * 1000)
        return record

    pool = ThreadPoolExecutor(parallel_pages)
    stop_reason = 'frontier_empty'
    try:
        while True:
            if len(pages) >= max_pages:
                stop_reason = 'max_pages'
                break
            if time.perf_counter() - started >= max_seconds:
                stop_reason = 'max_seconds'
                break
            # ...but not while a tier-1 group nobody has visited is still queued: one crawl gave up
            # with much of its budget unspent and fund pages still waiting
            if should_stop_for_no_progress(progress['stale'], len(pages), max_pages,
                                           time.perf_counter() - started, max_seconds) \
                    and not frontier.has_untried_tier1():
                stop_reason = 'no_progress'
                break
            if judge.open:
                stop_reason = 'jev_unavailable'
                break
            with lock:
                batch = []
                while len(batch) < min(parallel_pages, max_pages - len(pages)):
                    item = frontier.pop()
                    if item is None:
                        break
                    if canonical(item[0]) not in visited:
                        visited.add(canonical(item[0]))
                        batch.append(item)
            if not batch:
                break
            for record in pool.map(lambda item: visit(*item), batch):
                with lock:
                    progress['count'] += 1
                    record['page_no'] = progress['count']
                    # progress = any Candidate the contract would return, not accepted alone
                    progress['stale'] = 0 if record['kept'] else progress['stale'] + 1
                    frontier.record(record['url'], record['accepted'])
                    if frontier.rescued_at is not None and not progress.get('rescued'):
                        progress['rescued'] = True
                        emit({'type': 'jev_frontier_rescue', 'after_pages': frontier.rescued_at})
                pages.append(record)
                emit({'type': 'jev_page', **record})
    finally:
        barrier = threading.Barrier(parallel_pages)

        def close(_):
            barrier.wait(timeout=30)
            if hasattr(local, 'page'):
                local.context.close()
                local.browser.close()
                local.playwright.stop()

        list(pool.map(close, range(parallel_pages)))
        pool.shutdown()
        escalation_pool.shutdown()

    decision_model = judge.model
    if stop_reason == 'jev_unavailable':
        # No judge is left (Jev and its LLM fallback both unavailable): the rest of the crawl runs on
        # the agent path; its Candidates stay unscored.
        reason = judge.unavailable_reason or 'failures'
        emit({'type': 'jev_fallback', 'after_pages': len(pages), 'reason': reason})
        logger.warning('[jev] falling back to the agent path (%s)', reason)
        # The agent path costs far more tokens than a Jev page, so the fallback inherits what is
        # left of the crawl's token cap.
        fallback_pages = max(1, min(max_pages - len(pages),
                                    max(1, (agent_token_cap - sum(tokens.values())) // 120_000)))
        fallback = agentic_crawl(final_url, goal, llm=agent,
                                 max_pages=fallback_pages, same_domain_only=True,
                                 user_agent=user_agent, headless=headless, enable_learning=False,
                                 on_event=on_event)
        for download in fallback.downloads:
            candidates.setdefault(download.url, download)
        tokens['prompt'] += fallback.prompt_tokens
        tokens['completion'] += fallback.completion_tokens
        tokens['cache_read'] += fallback.cache_read_tokens
        tokens['cache_creation'] += fallback.cache_creation_tokens
        decision_model = f'{agent.model} (no judge available: {reason})'

    found = [c for c in candidates.values() if c is not None]
    # Rejected Candidates are a count, not rows - unless the caller asks to see them, which is
    # how you check what the relevance model threw away.
    returned = found if include_rejected else [c for c in found if c.verdict != 'rejected']
    emit({'type': 'crawl_complete', 'pages_crawled': len(pages)})
    return AgenticCrawlResult(
        start_url=start_url, goal=goal, downloads=returned, pages_visited=len(pages),
        total_tokens=tokens['prompt'] + tokens['completion'],
        prompt_tokens=tokens['prompt'], completion_tokens=tokens['completion'],
        cache_read_tokens=tokens['cache_read'], cache_creation_tokens=tokens['cache_creation'],
        decision_model=decision_model, relevance_model=judge.model if judge.requests else None,
        rejected_count=sum(1 for c in found if c.verdict == 'rejected'),
        jev_requests=judge.requests, jev_cost_usd=judge.cost_usd,
        escalations=escalations, escalations_skipped=skipped, stop_reason=stop_reason,
        guard_counts=guard,
    )
