"""Route the crawl's traffic to the site through a proxy, or run its browser remotely.

Only requests to the crawled site go through it: the browser, the plain HTTP fetches, the probes, robots.txt and
sitemaps. Calls to the model providers (Anthropic, TypeSafe, ...) do not. That is why the proxy has its own
variables and not HTTPS_PROXY, which httpx, urllib and the provider SDKs would all pick up.

    PROXY_SERVER     http://host:port (any provider: Oxylabs, Bright Data, a corporate proxy, ...)
    PROXY_USERNAME   optional
    PROXY_PASSWORD   optional
    BROWSER_CDP_URL  wss://... of a remote browser (Oxylabs Headless Browser, Browserless, Browserbase, ...)
                     used instead of a local Chromium. It brings its own network, so the proxy is not applied
                     to it; the plain HTTP fetches still use the proxy.

Read on every call, so a server picks up a change on its next crawl.
"""
from __future__ import annotations

import os
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit


def _settings() -> tuple[str, str, str] | None:
    server = (os.getenv('PROXY_SERVER') or '').strip()
    if not server:
        return None
    if '://' not in server:
        server = f'http://{server}'
    return server, os.getenv('PROXY_USERNAME') or '', os.getenv('PROXY_PASSWORD') or ''


def browser_proxy() -> dict | None:
    """The `proxy` argument for Playwright's chromium.launch(), or None."""
    settings = _settings()
    if not settings:
        return None
    server, username, password = settings
    proxy = {'server': server}
    if username:
        proxy.update(username=username, password=password)
    return proxy


def http_proxy() -> str | None:
    """The proxy URL, credentials included, for httpx (`proxy=`) and urllib, or None."""
    settings = _settings()
    if not settings:
        return None
    server, username, password = settings
    if not username:
        return server
    parts = urlsplit(server)
    netloc = f"{quote(username, safe='')}:{quote(password, safe='')}@{parts.netloc}"
    return urlunsplit(parts._replace(netloc=netloc))


def urlopen(request: urllib.request.Request, timeout: float):
    """urllib.request.urlopen through the proxy when one is set."""
    proxy = http_proxy()
    if not proxy:
        return urllib.request.urlopen(request, timeout=timeout)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    return opener.open(request, timeout=timeout)


def launch_browser(playwright, headless: bool = True):
    """The crawl's browser: the remote one at BROWSER_CDP_URL, or a local Chromium through the proxy."""
    cdp_url = (os.getenv('BROWSER_CDP_URL') or '').strip()
    if cdp_url:
        return playwright.chromium.connect_over_cdp(cdp_url)
    # --no-sandbox / --disable-setuid-sandbox are Linux/Docker flags; skip in headed mode.
    args = ['--no-sandbox', '--disable-setuid-sandbox'] if headless else []
    return playwright.chromium.launch(headless=headless, args=args, proxy=browser_proxy())
