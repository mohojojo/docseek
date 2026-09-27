"""Documents named in the JSON a page fetches for itself.

A single-page app renders its document list from a JSON response and often without an `<a href>` at
all: one fund manager's literature grid is a set of download *buttons* over an empty <app-root>, so the
harvest sees nothing. The JSON that filled the grid was on the wire, though, with a title next to each
path. This module reads first-party JSON responses during a page's load and turns their records into
Candidates - which then go through relevance like any other link.

A record is an object holding one URL-ish string (a key containing href, url, path, file, link, download
or document) plus, if present, a title-ish string next to it. Absolute URLs are kept as they come.
Relative paths are the hard part: the site's own script decides what to put in front of them
(one site prepends /download), so a relative path is joined to the page and, when the page's
rendered links show the same path under a prefix, to that prefix as well.
"""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urlparse

MAX_JSON_BYTES = 4_000_000
MAX_RECORDS_PER_PAGE = 2_000
_URL_KEY = re.compile(r'href|url|path|file|link|download|document', re.IGNORECASE)
_TITLE_KEY = re.compile(r'title|name|label|caption|description', re.IGNORECASE)
_SKIP_KEY = re.compile(r'thumb|image|icon|logo|avatar|preview|css|script', re.IGNORECASE)
_NOT_DOCUMENT_EXT = re.compile(r'\.(jpe?g|png|gif|svg|webp|ico|css|js|json|xml|html?|woff2?|ttf|mp4|mp3)(\?|$)', re.IGNORECASE)


def _walk(node, out: list[dict]) -> None:
    if len(out) >= MAX_RECORDS_PER_PAGE:
        return
    if isinstance(node, list):
        for item in node:
            _walk(item, out)
    elif isinstance(node, dict):
        urls = [(k, v) for k, v in node.items()
                if isinstance(v, str) and _URL_KEY.search(k) and not _SKIP_KEY.search(k) and _looks_like_path(v)]
        if urls:
            titled = [(k, v) for k, v in node.items()
                      if isinstance(v, str) and _TITLE_KEY.search(k) and not _URL_KEY.search(k) and v.strip()]
            # a "title" beats a "name" beats a "description"; the longest of equals carries the most
            titled.sort(key=lambda kv: (0 if 'title' in kv[0].lower() else 1 if 'name' in kv[0].lower() else 2,
                                        -len(kv[1])))
            title = titled[0][1] if titled else ''
            for _, value in urls:
                out.append({'path': value.strip(), 'name': title.strip()[:200]})
        for value in node.values():
            if isinstance(value, (dict, list)):
                _walk(value, out)


def _looks_like_path(value: str) -> bool:
    value = value.strip()
    if len(value) < 8 or len(value) > 2000 or ' ' in value[:8] or _NOT_DOCUMENT_EXT.search(value):
        return False
    return value.startswith(('http://', 'https://', '/'))


def records_from_json(body: str) -> list[dict]:
    """Every {path, name} pair in a JSON document, in document order."""
    if len(body) > MAX_JSON_BYTES:
        return []
    try:
        data = json.loads(body)
    except ValueError:
        return []
    out: list[dict] = []
    _walk(data, out)
    return out


def learn_prefixes(rendered_hrefs: list[str], relative_paths: list[str]) -> set[str]:
    """Prefixes the page's own links put in front of paths the JSON gives relatively.

    A site's JSON says /en/factsheet/<id>/x.pdf; its fund pages link
    /download/en/factsheet/<id>/x.pdf. Seeing one such pair teaches the prefix for all.
    """
    prefixes: set[str] = set()
    paths = {p for p in relative_paths if p.startswith('/')}
    for href in rendered_hrefs:
        parsed = urlparse(href)
        for path in paths:
            if parsed.path.endswith(path) and parsed.path != path:
                prefixes.add(parsed.path[:-len(path)])
    return prefixes


def candidates_from_json(body: str, page_url: str, rendered_hrefs: list[str],
                         prefixes: set[str] | None = None) -> list[dict]:
    """Document-looking Candidates named in one JSON response: absolute URLs as given, relative paths
    joined to the page and to any prefix the page's links reveal.

    `prefixes` is the crawl's memory: a prefix learned on one page (a fund page that links
    /download/...) applies to the JSON of another (the listing that renders only buttons).
    """
    records = records_from_json(body)
    relative = [r['path'] for r in records if r['path'].startswith('/')]
    if prefixes is None:
        prefixes = set()
    prefixes |= learn_prefixes(rendered_hrefs, relative)
    seen: set[str] = set()
    out: list[dict] = []
    for record in records:
        path = record['path']
        # once the site has shown what it prepends, the bare path is the app shell, not the file
        if path.startswith('http'):
            urls = [path]
        elif prefixes:
            urls = [urljoin(page_url, prefix + path) for prefix in sorted(prefixes)]
        else:
            urls = [urljoin(page_url, path)]
        for url in urls:
            if url not in seen:
                seen.add(url)
                out.append({'url': url, 'name': record['name'], 'context': '', 'section': '', 'column': '',
                            'path': 'json'})
    return out
