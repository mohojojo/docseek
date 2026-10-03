"""Proxy (docseek.proxy): the crawl's traffic to the site goes through PROXY_SERVER, the model providers' does not."""
import urllib.request

import httpx
import pytest

import docseek.proxy as proxy


def test_no_proxy_by_default():
    assert proxy.browser_proxy() is None
    assert proxy.http_proxy() is None


def test_a_server_without_credentials(monkeypatch):
    monkeypatch.setenv('PROXY_SERVER', 'proxy.example:8080')     # the scheme is optional
    assert proxy.browser_proxy() == {'server': 'http://proxy.example:8080'}
    assert proxy.http_proxy() == 'http://proxy.example:8080'


def test_credentials_go_to_playwright_apart_and_into_the_url_escaped(monkeypatch):
    monkeypatch.setenv('PROXY_SERVER', 'http://pr.example:7777')
    monkeypatch.setenv('PROXY_USERNAME', 'customer-me-cc-de')
    monkeypatch.setenv('PROXY_PASSWORD', 'p@ss:w/rd')
    assert proxy.browser_proxy() == {'server': 'http://pr.example:7777', 'username': 'customer-me-cc-de',
                                     'password': 'p@ss:w/rd'}
    assert proxy.http_proxy() == 'http://customer-me-cc-de:p%40ss%3Aw%2Frd@pr.example:7777'
    httpx.Client(proxy=proxy.http_proxy()).close()               # httpx accepts it


def test_urlopen_goes_through_the_proxy_only_when_one_is_set(monkeypatch):
    direct = []
    monkeypatch.setattr(urllib.request, 'urlopen', lambda request, timeout: direct.append(request) or 'direct')
    request = urllib.request.Request('https://example.com/robots.txt')
    assert proxy.urlopen(request, timeout=1) == 'direct'

    monkeypatch.setenv('PROXY_SERVER', 'http://pr.example:7777')
    handlers = []

    class Opener:
        def open(self, request, timeout):
            return 'proxied'

    def build_opener(*given):
        handlers.extend(given)
        return Opener()
    monkeypatch.setattr(urllib.request, 'build_opener', build_opener)
    assert proxy.urlopen(request, timeout=1) == 'proxied'
    assert handlers[0].proxies == {'http': 'http://pr.example:7777', 'https': 'http://pr.example:7777'}
    assert len(direct) == 1


@pytest.mark.parametrize('name', ['HTTPS_PROXY', 'HTTP_PROXY'])
def test_the_generic_proxy_variables_are_not_ours(monkeypatch, name):
    # they would also route the model providers' calls; the crawl reads only PROXY_SERVER
    monkeypatch.setenv(name, 'http://elsewhere:3128')
    assert proxy.http_proxy() is None


class FakeChromium:
    def __init__(self):
        self.calls = []

    def launch(self, **kwargs):
        self.calls.append(('launch', kwargs))
        return 'local'

    def connect_over_cdp(self, url):
        self.calls.append(('connect_over_cdp', url))
        return 'remote'


class FakePlaywright:
    def __init__(self):
        self.chromium = FakeChromium()


def test_a_local_browser_goes_through_the_proxy(monkeypatch):
    monkeypatch.setenv('PROXY_SERVER', 'http://pr.example:7777')
    pw = FakePlaywright()
    assert proxy.launch_browser(pw) == 'local'
    kind, kwargs = pw.chromium.calls[0]
    assert kind == 'launch' and kwargs['proxy'] == {'server': 'http://pr.example:7777'} and kwargs['headless'] is True


def test_browser_cdp_url_connects_to_the_remote_browser_instead(monkeypatch):
    # the remote browser brings its own network: no local launch, no proxy
    monkeypatch.setenv('BROWSER_CDP_URL', 'wss://user:pass@hb.example?p_cc=DE')
    monkeypatch.setenv('PROXY_SERVER', 'http://pr.example:7777')
    pw = FakePlaywright()
    assert proxy.launch_browser(pw, headless=False) == 'remote'
    assert pw.chromium.calls == [('connect_over_cdp', 'wss://user:pass@hb.example?p_cc=DE')]


class FakeContext:
    def __init__(self, **options):
        self.options, self.cookies, self.closed = options, [], False

    def add_cookies(self, cookies):
        self.cookies += cookies

    def new_page(self):
        return ('page in', self)

    def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self, contexts=()):
        self.contexts = list(contexts)
        self.made = []

    def new_context(self, **options):
        context = FakeContext(**options)
        self.made.append(context)
        return context

    def new_page(self, **options):
        return ('own page', options)


def test_locally_a_new_context_with_our_options():
    browser = FakeBrowser()
    context = proxy.browser_context(browser, user_agent='docseek', accept_downloads=True)
    assert context.options == {'user_agent': 'docseek', 'accept_downloads': True}
    assert proxy.new_page(browser, 'docseek') == ('own page', {'user_agent': 'docseek'})
    proxy.close_context(context)
    assert context.closed


def test_a_remote_browser_crawls_in_its_default_context_with_its_own_user_agent(monkeypatch):
    # the service's fingerprint, stealth and CAPTCHA solving live in the default context
    monkeypatch.setenv('BROWSER_CDP_URL', 'wss://remote.example')
    default = FakeContext()
    browser = FakeBrowser([default])
    cookies = [{'name': 'consent', 'value': '1', 'url': 'https://site.example'}]
    context = proxy.browser_context(browser, user_agent='docseek', storage_state={'cookies': cookies})
    assert context is default and not browser.made
    assert default.cookies == cookies                       # a remembered cookie banner stays dismissed
    assert proxy.new_page(browser, 'docseek') == ('page in', default)
    proxy.close_context(context)
    assert not default.closed                                # the service ends the session on disconnect


def test_a_remote_browser_without_a_context_gets_a_plain_one(monkeypatch):
    monkeypatch.setenv('BROWSER_CDP_URL', 'wss://remote.example')
    browser = FakeBrowser()
    assert proxy.browser_context(browser, user_agent='docseek').options == {}


class ChallengePage:
    """A page that is Cloudflare's challenge for its first `challenged` looks."""
    url = 'https://site.example/'

    def __init__(self, challenged):
        self.challenged, self.waited = challenged, 0

    def evaluate(self, script):
        self.challenged -= 1
        return self.challenged >= 0

    def title(self):
        return 'Site'

    def wait_for_timeout(self, ms):
        self.waited += ms

    def wait_for_load_state(self, state, timeout):
        pass


def test_an_ordinary_page_is_not_waited_on(monkeypatch):
    monkeypatch.setenv('BROWSER_CDP_URL', 'wss://remote.example')
    page = ChallengePage(challenged=0)
    assert proxy.wait_out_challenge(page) and page.waited == 0


def test_a_remote_browser_is_given_time_to_solve_the_challenge(monkeypatch):
    monkeypatch.setenv('BROWSER_CDP_URL', 'wss://remote.example')
    page = ChallengePage(challenged=3)
    assert proxy.wait_out_challenge(page) and page.waited == 3000
    assert not proxy.wait_out_challenge(ChallengePage(challenged=99), timeout_s=0)     # it never cleared


def test_a_local_browser_does_not_wait_on_a_challenge_it_cannot_solve():
    page = ChallengePage(challenged=99)
    assert not proxy.wait_out_challenge(page) and page.waited == 0
