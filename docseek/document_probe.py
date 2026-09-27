"""Tell a document from a page by what the server returns, not by how the URL is spelled.

`looks_like_document` knows a file extension and three path hints. `getfile.asp?id=105185`,
`/files/nuxeo/dl/<uuid>` and `/verdocumento/ver?e=<token>` are pages to it: they get a page-kind question,
a slot in the Frontier, and when visited the browser aborts with "Download is starting" and the document
is lost. On sites that serve documents this way, that alone can mean no document is found.

Asking the server about every link would cost a request per link, so links are probed by sibling group:
links that share a DOM tag path on one page are one table column or one list, and are almost always the
same kind of thing. The first and the last of a group are asked; if both are documents, so is the group.
Only the response headers are read. Site chrome (nav, header, footer) is never probed.
"""
from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

MAX_PROBES_PER_PAGE = 8
MAX_PROBES_PER_CRAWL = 120
PROBE_TIMEOUT_S = 8.0
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
                                    timeout=PROBE_TIMEOUT_S)
        self._may_probe = may_probe
        self._lock = threading.Lock()
        self.probes = 0
        self.documents_found = 0

    def is_document(self, url: str) -> bool | None:
        """Ask the server. None when it could not be asked or did not answer."""
        if not self._may_probe(url):
            return None
        with self._lock:
            if self.probes >= MAX_PROBES_PER_CRAWL:
                return None
            self.probes += 1
        try:
            # GET and stop at the headers: several document endpoints answer HEAD with 405 or with HTML
            with self._client.stream('GET', url) as response:
                if response.status_code >= 400:
                    return None
                return is_document_response(response.headers.get('content-type', ''),
                                            response.headers.get('content-disposition', ''))
        except httpx.HTTPError as exc:
            logger.debug('[probe] %s: %s', url, exc)
            return None

    def sort_all(self, links: list[dict]) -> tuple[list[dict], list[dict]]:
        """Links with no sibling structure (mined from JSON): group by URL up to the last segment and probe
        one per group, since a JSON list is one column by construction."""
        groups: dict[str, list[dict]] = {}
        for link in links:
            segments = urlparse(link['url']).path.rsplit('/', 2)
            groups.setdefault(urlparse(link['url']).netloc + segments[0], []).append(link)
        samples = [(key, links[0]) for key, links in groups.items()][:MAX_PROBES_PER_PAGE]
        if not samples:
            return [], links
        with ThreadPoolExecutor(4) as pool:
            answers = list(pool.map(lambda sample: self.is_document(sample[1]['url']), samples))
        documents = {link['url'] for (key, _), answer in zip(samples, answers) if answer
                     for link in groups[key]}
        with self._lock:
            self.documents_found += len(documents)
        return ([link for link in links if link['url'] in documents],
                [link for link in links if link['url'] not in documents])

    def sort(self, page_links: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split `page_links` into (documents, pages) by probing each sibling group's first and last link."""
        groups: dict[str, list[dict]] = {}
        for link in page_links:
            path = link.get('path') or ''
            if path and not _CHROME.search(path):
                groups.setdefault(path, []).append(link)
        samples = [(path, link) for path, links in groups.items() if len(links) > 1
                   for link in (links[0], links[-1])][:MAX_PROBES_PER_PAGE]
        if not samples:
            return [], page_links
        with ThreadPoolExecutor(4) as pool:
            answers = list(pool.map(lambda sample: self.is_document(sample[1]['url']), samples))
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
