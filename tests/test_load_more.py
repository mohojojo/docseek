"""The reveal step's "load more" click, against real markup in a real browser, offline."""
import pytest

from docseek.jev_crawl import _LOAD_MORE_JS

playwright = pytest.importorskip('playwright.sync_api')

PAGE = """<html><body>
<nav><button class="load-more">Menu more</button></nav>
<form><button class="show-more">Search more</button></form>
<a href="/hirek/cikk">Tovább olvasok</a>
<main><ul id="list"><li><a href="/d/1">1</a></li></ul>%s</main>
<script>window.n = 1; window.more = function () { n++; document.querySelector('#list').insertAdjacentHTML('beforeend',
  '<li><a href="/d/' + n + '">' + n + '</a></li>'); if (n >= 4) document.querySelector('#more').remove(); };</script>
</body></html>"""


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f'chromium not available: {exc}')
        yield browser.new_page()
        browser.close()


@pytest.mark.parametrize('button', [
    '<div class="results__load-more"><button id="more" onclick="more()">Tovább olvasok</button></div>',   # a fund manager's site
    '<button id="more" onclick="more()">Load more</button>',
    '<a id="more" href="#" onclick="more(); return false">Mehr anzeigen</a>',
])
def test_it_clicks_until_the_button_is_gone(page, button):
    page.set_content(PAGE % button)
    clicks = 0
    while page.evaluate(_LOAD_MORE_JS) and clicks < 10:
        clicks += 1
    assert clicks == 3 and page.locator('#list a').count() == 4


def test_it_never_navigates_submits_or_touches_site_chrome(page):
    page.set_content(PAGE % '')          # only the nav button, the form button and the article link are left
    assert page.evaluate(_LOAD_MORE_JS) is None


def test_only_a_listing_is_worth_the_clicks():
    # one site has a load-more on every fund page, revealing nothing the goal asks for: clicked there it
    # cost a third of the crawl's pages
    from docseek.jev_crawl import LOAD_MORE_KINDS
    assert 'document_listing' in LOAD_MORE_KINDS and 'seed' in LOAD_MORE_KINDS
    assert 'subject_page' not in LOAD_MORE_KINDS and 'news_or_article' not in LOAD_MORE_KINDS
