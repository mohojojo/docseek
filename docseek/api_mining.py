"""Deterministic API-mining harvest for canvas / SPA sites.

Some sites (Flutter/React/Vue apps) render document lists from a backend API rather than
serving them as HTML links, so the agent's DOM/vision path finds nothing. This module mines
the site's own JS bundle for a backend config and reads the document list directly from the
API - no browser, no LLM. First supported backend: Firebase (Firestore + Storage), which is
common among Flutter web apps (e.g. a Flutter app backed by Firebase that renders its document
list on a canvas).
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote, urljoin

import httpx

from .models import AgenticDownload, CrawlPlan
from .reach import is_safe_url

logger = logging.getLogger(__name__)

# Collection names to try when the bundle does not reveal one. Kept short and generic.
_DEFAULT_COLLECTIONS = ('Files', 'files', 'Documents', 'documents', 'Reports', 'reports')
_PDF_NAME_RE = re.compile(r'\.pdf$', re.IGNORECASE)


def extract_firebase_config(text: str) -> dict | None:
    """Pull a Firebase web config (apiKey / projectId / storageBucket) out of a JS bundle."""
    cfg: dict[str, str] = {}
    for field in ('apiKey', 'projectId', 'storageBucket'):
        m = re.search(rf'{field}["\']?\s*[:=]\s*["\']([^"\']+)["\']', text)
        if m:
            cfg[field] = m.group(1)
    if 'apiKey' not in cfg:
        m = re.search(r'AIza[0-9A-Za-z_\-]{35}', text)
        if m:
            cfg['apiKey'] = m.group(0)
    if 'storageBucket' not in cfg:
        m = re.search(r'([a-z0-9-]+\.appspot\.com)', text)
        if m:
            cfg['storageBucket'] = m.group(1)
    if 'projectId' not in cfg and cfg.get('storageBucket'):
        cfg['projectId'] = cfg['storageBucket'].split('.')[0]
    if cfg.get('apiKey') and cfg.get('projectId'):
        cfg.setdefault('storageBucket', f'{cfg["projectId"]}.appspot.com')
        return cfg
    return None


def _discover_collections(bundle: str) -> list[str]:
    """Best-effort collection-name discovery from bundle strings, plus defaults."""
    found = re.findall(r'collection\(["\']([A-Za-z0-9_]+)["\']\)', bundle)
    ordered = list(dict.fromkeys(found + list(_DEFAULT_COLLECTIONS)))
    return ordered[:8]


def _firestore_list(client: httpx.Client, project: str, api_key: str, collection: str) -> list[dict]:
    base = (f'https://firestore.googleapis.com/v1/projects/{project}'
            f'/databases/(default)/documents/{collection}')
    docs: list[dict] = []
    token: str | None = None
    for _ in range(50):  # hard page cap
        params = {'key': api_key, 'pageSize': 300}
        if token:
            params['pageToken'] = token
        try:
            r = client.get(base, params=params, timeout=20.0)
        except Exception:
            break
        if r.status_code != 200:
            break
        data = r.json()
        for d in data.get('documents', []):
            f = d.get('fields', {})
            fp = f.get('filePath', {}).get('stringValue')
            nm = (f.get('name', {}).get('stringValue')
                  or (fp.split('/')[-1] if fp else None))
            if fp:
                docs.append({'name': nm, 'filePath': fp})
        token = data.get('nextPageToken')
        if not token:
            break
    return docs


def _storage_url(bucket: str, file_path: str) -> str:
    return (f'https://firebasestorage.googleapis.com/v0/b/{bucket}/o/'
            f'{quote(file_path, safe="")}?alt=media')


def _name_matches_plan(name: str, crawl_plan: CrawlPlan | None) -> bool:
    """Keep a document when its name matches the goal's key terms (or when there is no plan)."""
    if not name or not _PDF_NAME_RE.search(name):
        # Only keep obvious documents; a filePath without a doc extension is likely not one.
        return bool(name)
    if crawl_plan is None or not crawl_plan.key_terms:
        return True
    low = name.lower()
    return any(term.lower() in low for term in crawl_plan.key_terms)


def _fetch_bundle_texts(client: httpx.Client, page_url: str) -> list[str]:
    """Return the text of candidate JS bundles for a page (main.dart.js + <script src> files)."""
    texts: list[str] = []
    # Flutter convention first - cheap and covers the common case.
    if not is_safe_url(page_url):
        return texts
    for candidate in ('main.dart.js', 'flutter_bootstrap.js'):
        try:
            r = client.get(urljoin(page_url, candidate), timeout=30.0)
            if r.status_code == 200 and r.text:
                texts.append(r.text)
        except Exception:
            pass
    try:
        page = client.get(page_url, timeout=20.0)
        for m in re.finditer(r'<script[^>]+src=["\']([^"\']+)["\']', page.text, re.IGNORECASE):
            src = urljoin(page_url, m.group(1))
            if src.endswith('.js') and is_safe_url(src):     # the page names the script host: no SSRF targets
                try:
                    r = client.get(src, timeout=30.0)
                    if r.status_code == 200 and 'AIza' in r.text:
                        texts.append(r.text)
                except Exception:
                    pass
    except Exception:
        pass
    return texts


def mine_documents(
    page_url: str,
    *,
    user_agent: str,
    crawl_plan: CrawlPlan | None = None,
    source_page: str | None = None,
) -> list[AgenticDownload]:
    """Mine a page's JS bundle for a backend API and return document downloads.

    Deterministic and browserless. Returns [] when no supported backend is found.
    """
    headers = {'User-Agent': user_agent}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        bundles = _fetch_bundle_texts(client, page_url)
        cfg = None
        bundle_for_collections = ''
        for text in bundles:
            cfg = extract_firebase_config(text)
            if cfg:
                bundle_for_collections = text
                break
        if not cfg:
            return []

        logger.info('[api_mining] firebase project=%s at %s', cfg.get('projectId'), page_url)
        records: list[dict] = []
        for coll in _discover_collections(bundle_for_collections):
            records = _firestore_list(client, cfg['projectId'], cfg['apiKey'], coll)
            if records:
                logger.info('[api_mining] collection %r → %d records', coll, len(records))
                break

    bucket = cfg['storageBucket']
    downloads: list[AgenticDownload] = []
    seen: set[str] = set()
    for rec in records:
        if not _name_matches_plan(rec['name'], crawl_plan):
            continue
        url = _storage_url(bucket, rec['filePath'])
        if url in seen:
            continue
        seen.add(url)
        downloads.append(AgenticDownload(
            url=url,
            name=rec['name'] or rec['filePath'].split('/')[-1],
            reason='api-mining (firebase)',
            source_page=source_page or page_url,
        ))
    return downloads
