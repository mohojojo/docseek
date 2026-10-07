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
    BROWSER_STEALTH  1: the local browser is Google Chrome driven by Patchright (Playwright with the automation
                     tells patched out), which gets past Cloudflare's challenge page from the host's own IP or
                     the proxy's. Needs the `stealth` extra and an installed Chrome. BROWSER_CDP_URL wins.

Read on every call, so a server picks up a change on its next crawl.
"""
from __future__ import annotations

import logging
import contextvars
import os
import sys
import time
import urllib.request
from urllib.parse import quote, urlsplit, urlunsplit

logger = logging.getLogger(__name__)


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


# --- which browser: the environment's strongest, or a plain local one until a site shows it needs more ---------
# BROWSER_CDP_URL and BROWSER_STEALTH name the strongest browser available. By default every crawl uses it. With
# BROWSER_DEFAULT=local, a crawl starts on the plain local Chromium (cheap, fast) and moves to the strong one only
# for a site that has shown Cloudflare's challenge: once seen, the site is recorded in the patterns dir and later
# crawls of it start strong. The choice is per thread (contextvars), so parallel crawls do not share it.

_forced: contextvars.ContextVar[str | None] = contextvars.ContextVar('browser_strategy', default=None)


def strongest() -> str:
    """The strongest browser the environment offers: 'remote', 'stealth' or 'local'."""
    if (os.getenv('BROWSER_CDP_URL') or '').strip():
        return 'remote'
    if (os.getenv('BROWSER_STEALTH') or '').strip().lower() in ('1', 'true', 'yes'):
        return 'stealth'
    return 'local'


def use_browser(strategy: str | None) -> None:
    """Force this thread's browser: 'local' for the plain Chromium whatever the environment offers, None for the
    environment's strongest."""
    _forced.set(strategy)


def browser_strategy() -> str:
    """This thread's browser: what use_browser forced, else the environment's strongest."""
    return _forced.get() or strongest()


def starting_browser(host: str, patterns_dir: str | None) -> str | None:
    """What a crawl of `host` starts on: None (the strongest) unless BROWSER_DEFAULT=local and the host has not
    been seen behind a challenge, in which case 'local'. Nothing to choose when only a local browser exists."""
    if (os.getenv('BROWSER_DEFAULT') or '').strip().lower() != 'local' or strongest() == 'local':
        return None
    if patterns_dir:
        from .patterns import PatternStore
        known = PatternStore(patterns_dir).load(host)
        if known and known.challenge_seen:
            return None
    return 'local'


def record_challenge(host: str, patterns_dir: str | None) -> None:
    """Remember that `host` showed a challenge, so later crawls of it start on the strong browser."""
    if patterns_dir:
        from .patterns import PatternStore
        PatternStore(patterns_dir).record_challenge(host, browser_strategy())


def _cdp_url() -> str:
    if _forced.get() == 'local':
        return ''
    return (os.getenv('BROWSER_CDP_URL') or '').strip()


def _stealth() -> bool:
    if _forced.get() == 'local':
        return False
    return (os.getenv('BROWSER_STEALTH') or '').strip().lower() in ('1', 'true', 'yes') and not _cdp_url()


def sync_playwright():
    """Playwright's entry point, or Patchright's under BROWSER_STEALTH: the same API."""
    if not _stealth():
        from playwright.sync_api import sync_playwright as start
        return start()
    try:
        from patchright.sync_api import sync_playwright as start
    except ImportError as exc:
        raise RuntimeError('BROWSER_STEALTH needs Patchright and Google Chrome: '
                           'pip install "docseek[stealth]" && patchright install chrome') from exc
    return start()


_stealth_user_agent: str | None = None


def _headless_user_agent(playwright) -> str:
    """Chrome's own user agent without the 'Headless' that headless mode puts in it, which alone fails a challenge.
    Asked of the browser once per process, so that it always names the version that is installed."""
    global _stealth_user_agent
    if _stealth_user_agent is None:
        browser = playwright.chromium.launch(headless=True, channel='chrome')
        try:
            user_agent = browser.new_page().evaluate('() => navigator.userAgent')
        finally:
            browser.close()
        _stealth_user_agent = user_agent.replace('HeadlessChrome', 'Chrome')
    return _stealth_user_agent


def launch_browser(playwright, headless: bool = True):
    """The crawl's browser: the remote one at BROWSER_CDP_URL, or a local one through the proxy - Chromium, or
    Google Chrome under BROWSER_STEALTH (Patchright's Chromium does not pass a challenge, Chrome does)."""
    cdp_url = _cdp_url()
    if cdp_url:
        return playwright.chromium.connect_over_cdp(cdp_url)
    if _stealth():
        args = [f'--user-agent={_headless_user_agent(playwright)}'] if headless else []
        if headless and sys.platform.startswith('linux'):
            args += ['--no-sandbox', '--disable-setuid-sandbox']
        return playwright.chromium.launch(headless=headless, channel='chrome', args=args, proxy=browser_proxy())
    # --no-sandbox / --disable-setuid-sandbox are Linux/Docker flags; skip in headed mode.
    args = ['--no-sandbox', '--disable-setuid-sandbox'] if headless else []
    return playwright.chromium.launch(headless=headless, args=args, proxy=browser_proxy())


# Remote browser services apply their fingerprint, stealth and CAPTCHA solving to the browser's default context,
# and ask that the user agent be left to them (Browserbase, Steel, Hyperbrowser, Browserless). A new context with
# our own user agent would crawl without what the service is paid for.

def browser_context(browser, **options):
    """A context to crawl in: a new one with `options` locally, the remote browser's default context otherwise.
    There only storage_state's cookies are carried over; downloads are accepted by default either way."""
    if _stealth():
        options.pop('user_agent', None)     # Chrome's own: a claimed user agent the browser does not match is a tell
    if not _cdp_url():
        return browser.new_context(**options)
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    cookies = (options.get('storage_state') or {}).get('cookies')
    if cookies:
        context.add_cookies(cookies)
    return context


def new_page(browser, user_agent: str):
    """A page with our user agent locally (Chrome's own under BROWSER_STEALTH); on a remote browser, a page in
    its default context."""
    if _stealth():
        return browser.new_page()
    if not _cdp_url():
        return browser.new_page(user_agent=user_agent)
    return browser_context(browser).new_page()


def close_context(context) -> None:
    """Close a context we made. A remote browser's default context is left to the service, which ends the session
    when we disconnect; closing it could take another crawl's pages on a shared session with it."""
    if not _cdp_url():
        context.close()


# A remote browser's service solves Cloudflare's challenge page by itself, and Chrome under BROWSER_STEALTH passes
# it, but either takes seconds to tens of seconds, and a crawl that reads the page at once reads 'Just a moment...'.
# The challenge's own requests are third-party ones, so a crawl that blocks those has to let CHALLENGE_HOSTS through.
CHALLENGE_WAIT_S = 60
CHALLENGE_HOSTS = frozenset({'challenges.cloudflare.com'})
CHECKBOX_AFTER_S = 4        # most challenges clear unasked within this; one that has not is showing its checkbox
CHECKBOX_CLICKS = 3
# The challenge's settings sit in an inline script. Read from the DOM: Patchright evaluates in a world of its own,
# where the page's window._cf_chl_opt does not exist.
_ON_CHALLENGE_JS = "() => [...document.scripts].some(script => script.textContent.includes('_cf_chl_opt'))"


def _on_challenge(page) -> bool | None:
    """Whether the page is Cloudflare's challenge; None while it is navigating, as it does when one clears."""
    try:
        return bool(page.evaluate(_ON_CHALLENGE_JS)) or page.title().lower().startswith('just a moment')
    except Exception:  # noqa: BLE001
        return None


def _click_checkbox(page) -> bool:
    """Click the challenge's 'Verify you are human' checkbox, which sits at the left of its iframe."""
    for frame in page.frames:
        if urlsplit(frame.url).netloc not in CHALLENGE_HOSTS:
            continue
        try:
            box = frame.frame_element().bounding_box()
            if box and box['width'] > 100:              # the widget, not one of the challenge's hidden frames
                page.mouse.click(box['x'] + 30, box['y'] + box['height'] / 2)
                return True
        except Exception:  # noqa: BLE001 - the frame went away: the challenge is moving on
            pass
    return False


def wait_out_challenge(page, timeout_s: float = CHALLENGE_WAIT_S) -> bool:
    """Call after a navigation: waits while the page is Cloudflare's challenge, and under BROWSER_STEALTH clicks
    its checkbox when it shows one. False when the page is still the challenge: at once on a plain local Chromium,
    which does not pass one, after timeout_s otherwise."""
    if not _on_challenge(page):
        return True
    if _cdp_url() or _stealth():
        began = time.monotonic()
        clicks = 0
        while time.monotonic() - began < timeout_s:
            page.wait_for_timeout(1000)
            if _stealth() and clicks < CHECKBOX_CLICKS and time.monotonic() - began >= CHECKBOX_AFTER_S * (clicks + 1) \
                    and _on_challenge(page) and _click_checkbox(page):
                clicks += 1
                continue
            if _on_challenge(page) is False:
                try:
                    page.wait_for_load_state('domcontentloaded', timeout=10_000)
                except Exception:  # noqa: BLE001 - the page behind the challenge is there; let the caller read it
                    pass
                return True
    logger.warning("[challenge] not past Cloudflare's challenge: %s", page.url)
    return False
