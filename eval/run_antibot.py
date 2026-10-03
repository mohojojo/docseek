"""Anti-bot reach: which of the crawl's ways of fetching a page get past Cloudflare, in the network mode the
environment sets.

    .venv/bin/python -m eval.run_antibot --label local
    PROXY_SERVER=http://... .venv/bin/python -m eval.run_antibot --label residential
    BROWSER_CDP_URL=wss://... .venv/bin/python -m eval.run_antibot --label browserless --urls https://site.example/

One run measures one mode (direct, PROXY_SERVER or BROWSER_CDP_URL); compare the reports of several runs.
Every URL is fetched three ways, each browser one in a fresh browser, so that a cf_clearance cookie won by one
does not help the next:

    http          a plain httpx GET: the probes, API mining and the codegen fetcher
    browser       the crawl's browser with nothing blocked: the agent's pages
    browser+jev   the same under the Jev crawl's request blocking (no images, media or fonts, no third-party
                  XHR), which is what a challenge has to solve itself under there

Outcomes: `pass` (no challenge), `solved` (a challenge that cleared within --wait), `challenge` (still on the
challenge page), `blocked` (an error status without a challenge), `error`.

Only Cloudflare's interstitial challenge is recognised. A Turnstile widget embedded in an ordinary page loads as
`pass`, and so does another vendor's block page that answers 200.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

from docseek.jev_crawl import BLOCKED_RESOURCES
from docseek.proxy import CHALLENGE_HOSTS, browser_context, close_context, http_proxy, launch_browser
from docseek.reach import bare_host
from docseek.scraper import _DEFAULT_USER_AGENT, _TRACKING_SCRIPT_HOSTS

_REPORTS = Path(__file__).resolve().parent / 'reports'
WAYS = ('http', 'browser', 'browser+jev')
DEFAULT_URLS = (
    'https://example.com/',                                        # control: not on Cloudflare
    'https://www.cloudflare.com/',                                 # control: on Cloudflare, no challenge
    'https://www.scrapingcourse.com/cloudflare-challenge',         # managed challenge, built to be scraped
    'https://www.scrapingcourse.com/antibot-challenge',
    'https://nopecha.com/demo/cloudflare',
    'https://2captcha.com/demo/cloudflare-turnstile-challenge',
)
CHALLENGE_TITLE = 'just a moment'           # 'Attention Required!' is a block page: nothing there to solve


def mode() -> str:
    """The network mode this run measures, without the addresses: they carry credentials."""
    parts = ['remote browser'] if os.getenv('BROWSER_CDP_URL') else ['local browser']
    parts.append('proxy' if os.getenv('PROXY_SERVER') else 'direct')
    return ', '.join(parts)


def on_cloudflare(headers) -> bool:
    return (headers.get('server') or '').lower() == 'cloudflare' or bool(headers.get('cf-ray'))


def outcome(challenged_first: bool, challenged_last: bool, status: int) -> str:
    if challenged_last:
        return 'challenge'
    if status >= 400:
        return 'blocked'
    return 'solved' if challenged_first else 'pass'


def check_http(url: str, user_agent: str) -> dict:
    with httpx.Client(headers={'User-Agent': user_agent}, follow_redirects=True, timeout=30,
                      proxy=http_proxy()) as client:
        response = client.get(url)
    challenged = response.headers.get('cf-mitigated') == 'challenge' or '_cf_chl_opt' in response.text
    return {'outcome': outcome(challenged, challenged, response.status_code), 'status': response.status_code,
            'cloudflare': on_cloudflare(response.headers)}


def _challenged(page) -> bool:
    try:
        return page.evaluate("() => typeof window._cf_chl_opt !== 'undefined'") \
            or page.title().lower().startswith(CHALLENGE_TITLE)
    except Exception:  # noqa: BLE001 - the page is navigating, as it does when a challenge clears: look again
        return True


def _jev_route(url: str):
    """The Jev crawl's request blocking (jev_crawl.crawl's `route`), with the URL's host as the only first party."""
    first_party = bare_host(urlparse(url).netloc)

    def route(route_obj, request):
        kind = request.resource_type
        host = urlparse(request.url).netloc
        third_party = bare_host(host) != first_party and host not in CHALLENGE_HOSTS
        if kind in BLOCKED_RESOURCES \
                or (third_party and kind in ('xhr', 'fetch', 'ping', 'beacon', 'eventsource')) \
                or (kind == 'script' and urlparse(request.url).netloc in _TRACKING_SCRIPT_HOSTS):
            route_obj.abort()
        else:
            route_obj.continue_()
    return route


def check_browser(url: str, user_agent: str, *, jev_routes: bool, wait_s: float, headless: bool) -> dict:
    with sync_playwright() as p:
        browser = launch_browser(p, headless)
        try:
            context = browser_context(browser, user_agent=user_agent, accept_downloads=True)
            page = context.new_page()
            if jev_routes:
                page.route('**/*', _jev_route(url))
            documents = []                 # the main frame's documents: the challenge, then the page behind it
            page.on('response', lambda r: documents.append(r)
                    if r.request.is_navigation_request() and r.frame == page.main_frame else None)
            started = time.monotonic()
            page.goto(url, wait_until='domcontentloaded', timeout=60_000)
            first = last = _challenged(page)
            while last and time.monotonic() - started < wait_s:
                page.wait_for_timeout(1000)
                last = _challenged(page)
            result = {
                'outcome': outcome(first, last, documents[-1].status),
                'status': documents[0].status,
                'final_status': documents[-1].status,
                'cloudflare': any(on_cloudflare(document.headers) for document in documents),
                'seconds': round(time.monotonic() - started, 1),
            }
            try:
                result.update(title=page.title()[:80], webdriver=page.evaluate('() => navigator.webdriver'),
                              user_agent=page.evaluate('() => navigator.userAgent'))
            except Exception:  # noqa: BLE001 - a challenge page that reloads itself mid-read still has an outcome
                pass
            close_context(context)
            return result
        finally:
            browser.close()


def check(url: str, way: str, args) -> dict:
    try:
        if way == 'http':
            return check_http(url, args.user_agent)
        return check_browser(url, args.user_agent, jev_routes=way == 'browser+jev', wait_s=args.wait,
                             headless=not args.headed)
    except Exception as exc:  # noqa: BLE001 - one unreachable target is a row in the report, not the end of the run
        return {'outcome': 'error', 'error': f'{type(exc).__name__}: {exc}'[:200]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--urls', nargs='+', default=list(DEFAULT_URLS))
    parser.add_argument('--ways', nargs='+', choices=WAYS, default=list(WAYS))
    parser.add_argument('--wait', type=float, default=25, help='seconds a challenge gets to clear')
    parser.add_argument('--headed', action='store_true', help='a visible local browser')
    parser.add_argument('--user-agent', default=_DEFAULT_USER_AGENT)
    parser.add_argument('--label', default='run')
    args = parser.parse_args()

    report = {'mode': mode(), 'headed': args.headed, 'wait': args.wait, 'results': {}}
    print(f"mode: {report['mode']}{', headed' if args.headed else ''}\n")
    print(f"{'url':<58}" + ''.join(f'{way:<14}' for way in args.ways))
    for url in args.urls:
        row = report['results'][url] = {way: check(url, way, args) for way in args.ways}
        print(f'{url[:56]:<58}' + ''.join(f"{row[way]['outcome']:<14}" for way in args.ways), flush=True)

    _REPORTS.mkdir(exist_ok=True)
    path = _REPORTS / f"antibot_{time.strftime('%Y%m%d_%H%M%S')}_{args.label}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f'\n{path}')


if __name__ == '__main__':
    main()
