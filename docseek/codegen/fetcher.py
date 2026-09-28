"""The only way a generated program, or the agent that writes it, reaches the web.

Every request goes through the crawler's reach rules (docseek.reach): the site's host or a subdomain of it, no
private or metadata addresses, robots.txt. On top of those: one request at a time with a pause, a size cap and
a request budget.

Two things are allowed beyond the site itself, and only because the site's own pages did them first (learned
while rendering, saved with the program):
  data_hosts      hosts the pages load JSON from (a single-page app's backend), which a program may GET
  post_endpoints  host + path the pages POSTed to for JSON (a "load more" or search API), which a program may POST
"""
from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import httpx

from ..jev_crawl import _HARVEST_JS, BLOCKED_RESOURCES
from ..proxy import http_proxy, launch_browser
from ..reach import bare_host, is_safe_url, robots_allows
from ..scraper import _DEFAULT_USER_AGENT, _try_accept_cookies
from .sandbox import FetchRefused

MAX_BYTES = 4_000_000
PAUSE_S = 0.4
RENDER_WAIT_MS = 2500
_TEXT_TYPES = ('text/', 'application/json', 'application/xml', 'application/xhtml', 'application/ld+json',
               'application/javascript', 'application/x-javascript')


def site_of(host: str) -> str:
    """The registrable part a program may stay within: example.com for www.example.com, and the last three labels
    under a two-letter country code with a short second level (example.co.uk)."""
    parts = bare_host(host).split('.')
    if len(parts) > 2 and len(parts[-1]) == 2 and len(parts[-2]) <= 3:
        return '.'.join(parts[-3:])
    return '.'.join(parts[-2:])


def _post_data(request) -> str:
    """A request's POST body as text, for the agent to read; a binary (compressed) body is not text."""
    try:
        return (request.post_data or '')[:300]
    except UnicodeDecodeError:
        return '(binary body)'


def endpoint_of(url: str) -> str:
    u = urlparse(url)
    return f'{bare_host(u.netloc)}{u.path}'


class Fetcher:
    def __init__(self, start_url: str, *, max_requests: int = 400, max_renders: int = 40,
                 data_hosts: set[str] | None = None, post_endpoints: set[str] | None = None,
                 user_agent: str = _DEFAULT_USER_AGENT):
        self.site = site_of(urlparse(start_url).netloc)
        self.data_hosts: set[str] = set(data_hosts or ())
        self.post_endpoints: set[str] = set(post_endpoints or ())
        self.max_requests, self.max_renders = max_requests, max_renders
        self.requests = self.renders = 0
        self.user_agent = user_agent
        self._client = httpx.Client(headers={'User-Agent': user_agent}, follow_redirects=True, timeout=30,
                                    proxy=http_proxy())
        self._last = 0.0
        self._lock = threading.Lock()
        self._pw = self._browser = None
        self.cache: dict[str, dict] = {}

    def _on_site(self, host: str) -> bool:
        return host == self.site or host.endswith('.' + self.site)

    def check(self, url: str) -> None:
        """Raise FetchRefused, with the reason, unless the URL may be requested."""
        host = bare_host(urlparse(url).netloc)
        if not (self._on_site(host) or host in self.data_hosts):
            raise FetchRefused(f'off-site: {host} is not {self.site} or a data host its pages call')
        if not is_safe_url(url):
            raise FetchRefused('unsafe address')
        if not robots_allows(url):
            raise FetchRefused('disallowed by robots.txt')

    def _spend_request(self) -> None:
        if self.requests >= self.max_requests:
            raise FetchRefused(f'request budget of {self.max_requests} spent')
        self.requests += 1
        time.sleep(max(0.0, self._last + PAUSE_S - time.monotonic()))

    def fetch(self, url: str) -> dict:
        """{url, status, content_type, text} of a plain HTTP GET. Binary bodies come back as ''."""
        if url in self.cache:
            return self.cache[url]
        self.check(url)
        with self._lock:
            self._spend_request()
            try:
                resp = self._client.get(url)
            finally:
                self._last = time.monotonic()
        self.check(str(resp.url))                      # a redirect must stay on the site too
        ctype = resp.headers.get('content-type', '').split(';')[0].strip()
        text = resp.text[:MAX_BYTES] if ctype.startswith(_TEXT_TYPES) else ''
        out = {'url': str(resp.url), 'status': resp.status_code, 'content_type': ctype, 'text': text}
        self.cache[url] = out
        return out

    def post(self, url: str, body: dict) -> dict:
        """POST a JSON body - only to an endpoint the site's own page POSTed to for JSON."""
        self.check(url)
        if endpoint_of(url) not in self.post_endpoints:
            raise FetchRefused(f"POST to {endpoint_of(url)} is only allowed where the site's own page POSTs for data "
                               f'(seen: {sorted(self.post_endpoints) or "none yet - render or interact first"})')
        with self._lock:
            self._spend_request()
            try:
                resp = self._client.post(url, json=body)
            finally:
                self._last = time.monotonic()
        ctype = resp.headers.get('content-type', '').split(';')[0].strip()
        return {'url': str(resp.url), 'status': resp.status_code, 'content_type': ctype, 'text': resp.text[:MAX_BYTES]}

    def learn(self, calls: list[dict]) -> None:
        """Remember the data hosts and POST endpoints the site's own pages used (see the module docstring)."""
        for c in calls:
            if not c['content_type'].startswith(('application/json', 'text/json')) or not is_safe_url(c['url']):
                continue
            if c['method'] == 'POST':
                self.post_endpoints.add(endpoint_of(c['url']))
            host = bare_host(urlparse(c['url']).netloc)
            if not self._on_site(host):
                self.data_hosts.add(host)

    def _new_page(self):
        if self.renders >= self.max_renders:
            raise FetchRefused(f'render budget of {self.max_renders} spent')
        self.renders += 1
        if self._browser is None:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            self._browser = launch_browser(self._pw)
        page = self._browser.new_page(user_agent=self.user_agent)
        # images, media and fonts carry no links or data; skipping them is most of a proxy's per-GB bill
        page.route('**/*', lambda route, request: route.abort() if request.resource_type in BLOCKED_RESOURCES
                   else route.continue_())
        return page

    def render(self, url: str) -> dict:
        """{url, status, html, links, requests}: the page after its scripts ran, the crawler's harvest of its links,
        and the data calls (XHR/fetch) it made."""
        self.check(url)
        with self._lock:
            page = self._new_page()
            calls: list[dict] = []
            page.on('response', lambda r: calls.append({
                'url': r.url, 'method': r.request.method, 'status': r.status,
                'content_type': r.headers.get('content-type', '').split(';')[0]})
                if r.request.resource_type in ('xhr', 'fetch') and len(calls) < 80 else None)
            try:
                resp = page.goto(url, wait_until='domcontentloaded', timeout=45_000)
                page.wait_for_timeout(RENDER_WAIT_MS)
                _try_accept_cookies(page)
                self.check(page.url)
                self.learn(calls)
                return {'url': page.url, 'status': resp.status if resp else 0, 'html': page.content()[:MAX_BYTES],
                        'links': page.evaluate(_HARVEST_JS), 'requests': calls}
            finally:
                page.close()

    def interact(self, url: str, actions: list[dict]) -> dict:
        """Load a page, perform up to 8 actions and report what happened after them: the data calls the page made
        (with POST bodies) and its links. Actions: {"click": text or css}, {"select": [control, option]}."""
        self.check(url)
        with self._lock:
            page = self._new_page()
            calls: list[dict] = []
            page.on('response', lambda r: calls.append({
                'url': r.url, 'method': r.request.method, 'status': r.status,
                'post_data': _post_data(r.request),
                'content_type': r.headers.get('content-type', '').split(';')[0]})
                if r.request.resource_type in ('xhr', 'fetch') and len(calls) < 120 else None)
            done = []
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=45_000)
                page.wait_for_timeout(RENDER_WAIT_MS)
                _try_accept_cookies(page)
                self.check(page.url)
                before = len(calls)
                for action in actions[:8]:
                    try:
                        self._act(page, action)
                        page.wait_for_timeout(1500)
                        done.append(f'ok {action}')
                    except Exception as exc:  # noqa: BLE001 - report and go on
                        done.append(f'failed {action}: {type(exc).__name__}: {str(exc)[:120]}')
                page.wait_for_timeout(RENDER_WAIT_MS)
                # the site's own calls and any JSON; analytics beacons and tag managers are noise
                after = [c for c in calls[before:] if c['content_type'].startswith(('application/json', 'text/json'))
                         or self._on_site(bare_host(urlparse(c['url']).netloc))]
                self.learn(after)
                return {'url': page.url, 'actions': done, 'requests': after, 'links': page.evaluate(_HARVEST_JS)}
            finally:
                page.close()

    def _act(self, page, action: dict) -> None:
        if 'click' in action:
            self._locate(page, action['click']).click(timeout=5000)
            return
        control, option = action['select']
        target = self._locate(page, control)
        if (target.evaluate('e => e.tagName') or '').upper() == 'SELECT':
            target.select_option(label=option, timeout=5000)
            return
        target.click(timeout=5000)
        page.wait_for_timeout(500)
        page.get_by_role('option', name=option).or_(page.get_by_text(option, exact=True)).first.click(timeout=5000)

    @staticmethod
    def _locate(page, target: str):
        if target.startswith('#') and target[1:2].isdigit():
            target = f'[id="{target[1:]}"]'             # an id starting with a digit is not a valid CSS #id
        if target.startswith(('#', '.', '[')) or '>' in target or target.startswith(('button', 'a[', 'select', 'div', 'li')):
            return page.locator(target).first
        return page.get_by_role('button', name=target).or_(page.get_by_text(target, exact=False)).first

    def close(self) -> None:
        self._client.close()
        if self._browser:
            self._browser.close()
            self._pw.stop()
