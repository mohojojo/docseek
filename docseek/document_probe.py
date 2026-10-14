"""Tell a document from a page by what the server returns, not by how the URL is spelled.

`looks_like_document` knows a file extension and three path hints. `getfile.asp?id=105185`,
`/files/nuxeo/dl/<uuid>` and `/verdocumento/ver?e=<token>` are pages to it: they get a page-kind question,
a slot in the Frontier, and when visited the browser aborts with "Download is starting" and the document
is lost. On sites that serve documents this way, that alone can mean no document is found.

Asking the server about every link would cost a request per link, so links are probed by sibling group:
links that share a DOM tag path on one page are one table column or one list, and are almost always the
same kind of thing. The first and the last of a group are asked; if both are documents, so is the group.
Only the response headers are read. Site chrome (nav, header, footer) is never probed.

A site whose bot protection refuses the plain HTTP client (Cloudflare judges the client itself, and lets the
crawl's browser through) answers the probes 403. Once one is refused, same-site probes are asked from inside
the crawl's page instead (window.fetch), which leaves from the browser the site has let in.
"""
from __future__ import annotations

import logging
import re
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from urllib.parse import urlparse

import httpx

from .proxy import clearance_hooks, http_proxy

logger = logging.getLogger(__name__)

MAX_PROBES_PER_PAGE = 8
MAX_PROBES_PER_CRAWL = 120
PROBE_TIMEOUT_S = 8.0
_REFUSED = (403, 429, 503)
# fetch() resolves at the headers; aborting then leaves the body unread
_PAGE_PROBE_JS = """async ([url, timeoutMs]) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, {credentials: 'include', signal: controller.signal});
        const headers = {status: response.status, type: response.headers.get('content-type') || '',
                         disposition: response.headers.get('content-disposition') || ''};
        controller.abort();
        return headers;
    } catch (error) {
        return null;
    } finally {
        clearTimeout(timer);
    }
}"""
_CHROME = re.compile(r'(^|/)(nav|header|footer|aside)([./]|$)')
_DOCUMENT_TYPES = ('application/pdf', 'application/msword', 'application/vnd.', 'application/zip',
                   'application/octet-stream', 'application/rtf', 'text/csv')


def is_document_response(content_type: str, content_disposition: str = '') -> bool:
    """True when the headers say a file is coming rather than a page to read."""
    content_type = (content_type or '').lower().strip()
    if 'attachment' in (content_disposition or '').lower():
        return True
    # a few servers send a bare "pdf" or "zip" as the whole Content-Type
    return content_type.startswith(_DOCUMENT_TYPES) or content_type in ('pdf', 'zip')


class DocumentProbe:
    def __init__(self, user_agent: str, may_probe: Callable[[str], bool]):
        self._client = httpx.Client(headers={'User-Agent': user_agent}, follow_redirects=True,
                                    timeout=PROBE_TIMEOUT_S, proxy=http_proxy(), event_hooks=clearance_hooks())
        self._may_probe = may_probe
        self._lock = threading.Lock()
        self.probes = 0
        self.page_probes = 0
        self.page_answers: Counter[str] = Counter()     # document / page / refused / error / none
        self.documents_found = 0
        self.http_refused = False       # the site's bot protection refused the HTTP client: ask from the page

    def _admit(self, url: str) -> bool:
        if not self._may_probe(url):
            return False
        with self._lock:
            if self.probes >= MAX_PROBES_PER_CRAWL:
                return False
            self.probes += 1
        return True

    def is_document(self, url: str) -> bool | None:
        """Ask the server. None when it could not be asked or did not answer."""
        if not self._admit(url):
            return None
        try:
            # GET and stop at the headers: several document endpoints answer HEAD with 405 or with HTML
            with self._client.stream('GET', url) as response:
                if response.status_code in _REFUSED and not self.http_refused:
                    self.http_refused = True
                    logger.info('[probe] %s answered %d: asking from the page from now on', url,
                                response.status_code)
                if response.status_code >= 400:
                    return None
                return is_document_response(response.headers.get('content-type', ''),
                                            response.headers.get('content-disposition', ''))
        except httpx.HTTPError as exc:
            logger.debug('[probe] %s: %s', url, exc)
            return None

    def is_document_from_page(self, page, url: str) -> bool | None:
        """Ask the server from inside `page`, on the thread that drives it. Same origin only: a cross-origin
        response's headers are not the page's to read."""
        if not _same_origin(url, page.url) or not self._admit(url):
            return None
        try:
            answer = page.evaluate(_PAGE_PROBE_JS, [url, PROBE_TIMEOUT_S * 1000])
        except Exception as exc:  # noqa: BLE001 - a page that navigated away or closed has no answer
            logger.debug('[probe] from the page %s: %s', url, exc)
            answer = None
        if not answer:
            kind, result = 'none', None
        elif answer['status'] in _REFUSED:
            kind, result = 'refused', None
        elif answer['status'] >= 400:
            kind, result = 'error', None
        else:
            result = is_document_response(answer['type'], answer['disposition'])
            kind = 'document' if result else 'page'
        with self._lock:
            self.page_probes += 1
            self.page_answers[kind] += 1
        return result

    def _answer(self, urls: list[str], page) -> list[bool | None]:
        """Each URL asked over HTTP in parallel, or from `page` (one by one, on this thread) once the site has
        refused the HTTP client; a probe refused just now is asked again from the page."""
        from_page = lambda url: page is not None and self.http_refused and _same_origin(url, page.url)  # noqa: E731
        answers: list[bool | None] = [None] * len(urls)
        over_http = [i for i, url in enumerate(urls) if not from_page(url)]
        with ThreadPoolExecutor(4) as pool:
            for i, answer in zip(over_http, pool.map(self.is_document, [urls[i] for i in over_http])):
                answers[i] = answer
        for i, url in enumerate(urls):
            if answers[i] is None and from_page(url):
                answers[i] = self.is_document_from_page(page, url)
        return answers

    def sort_all(self, links: list[dict], page=None) -> tuple[list[dict], list[dict]]:
        """Links with no sibling structure (mined from JSON): group by URL up to the last segment and probe
        one per group, since a JSON list is one column by construction."""
        groups: dict[str, list[dict]] = {}
        for link in links:
            segments = urlparse(link['url']).path.rsplit('/', 2)
            groups.setdefault(urlparse(link['url']).netloc + segments[0], []).append(link)
        samples = [(key, links[0]) for key, links in groups.items()][:MAX_PROBES_PER_PAGE]
        if not samples:
            return [], links
        answers = self._answer([link['url'] for _, link in samples], page)
        documents = {link['url'] for (key, _), answer in zip(samples, answers) if answer
                     for link in groups[key]}
        with self._lock:
            self.documents_found += len(documents)
        return ([link for link in links if link['url'] in documents],
                [link for link in links if link['url'] not in documents])

    def sort(self, page_links: list[dict], page=None) -> tuple[list[dict], list[dict]]:
        """Split `page_links` into (documents, pages) by probing each sibling group's first and last link.
        `page` is the crawl's page the links came from, to ask from when the site refuses the HTTP client."""
        groups: dict[str, list[dict]] = {}
        for link in page_links:
            path = link.get('path') or ''
            if path and not _CHROME.search(path):
                groups.setdefault(path, []).append(link)
        samples = [(path, link) for path, links in groups.items() if len(links) > 1
                   for link in (links[0], links[-1])][:MAX_PROBES_PER_PAGE]
        if not samples:
            return [], page_links
        answers = self._answer([link['url'] for _, link in samples], page)
        by_group: dict[str, list[tuple[dict, bool | None]]] = {}
        for (path, link), answer in zip(samples, answers):
            by_group.setdefault(path, []).append((link, answer))
        documents: set[str] = set()
        for path, probed in by_group.items():
            if all(answer is True for _, answer in probed):
                documents.update(link['url'] for link in groups[path])       # the whole column is documents
            else:
                documents.update(link['url'] for link, answer in probed if answer is True)
        with self._lock:
            self.documents_found += len(documents)
        return ([link for link in page_links if link['url'] in documents],
                [link for link in page_links if link['url'] not in documents])


def _same_origin(url: str, page_url: str) -> bool:
    a, b = urlparse(url), urlparse(page_url)
    return (a.scheme, a.netloc) == (b.scheme, b.netloc)
