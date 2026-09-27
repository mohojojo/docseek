"""A listing's filters, found and driven by code, with Jev choosing the values. Offline, real browser."""
from unittest.mock import patch

import pytest

from docseek.jev import JevClient
from docseek.jev_crawl import _FILTER_SUBMIT_JS, _FILTERS_JS
from docseek.scraper import select_option_anywhere

playwright = pytest.importorskip('playwright.sync_api')

# A bank's reports page, reduced: a site-wide search in the header, then two component-library
# selects (a button plus a hidden listbox, tied by a shared label) and the filter's own "Keresés" button.
PAGE = """<html><body>
<header><form role="search"><input placeholder="Keresés..."><button type="submit">Keresés</button></form></header>
<main>
  <div class="sf-select">
    <span id="type-label">Dokumentum típus</span>
    <button class="sf-select__button" aria-labelledby="type-label" aria-expanded="false"
            onclick="toggle('type')">Havi jelentés</button>
    <ul id="type" role="listbox" aria-labelledby="type-label" style="display:none">
      <li role="option" onclick="pick('type', this)">Éves jelentések</li>
      <li role="option" onclick="pick('type', this)">Havi jelentés</li>
    </ul>
  </div>
  <div class="sf-select">
    <span id="year-label">Év</span>
    <button class="sf-select__button" aria-labelledby="year-label" aria-expanded="false"
            onclick="toggle('year')"> </button>
    <ul id="year" role="listbox" aria-labelledby="year-label" style="display:none">
      <li role="option" onclick="pick('year', this)">2026</li>
      <li role="option" onclick="pick('year', this)">2025</li>
    </ul>
  </div>
  <label for="fund">Alap</label>
  <select id="fund"><option>Összes alap</option><option>Példa Kötvény</option></select>
  <button id="apply" onclick="apply()">Keresés</button>
  <div id="results"><a href="/static/havi_202608.pdf">2026. augusztus</a></div>
</main>
<script>
  window.chosen = {};
  function toggle(id) { const ul = document.getElementById(id); ul.style.display = ul.style.display ? '' : 'none'; }
  function pick(id, li) { window.chosen[id] = li.textContent; toggle(id);
    document.querySelector('[aria-labelledby="' + id + '-label"].sf-select__button').textContent = li.textContent; }
  function apply() { if (window.chosen.year === '2026') document.getElementById('results').innerHTML =
    ['01', '02', '03'].map(m => '<a href="/static/havi_2026' + m + '.pdf">2026/' + m + '</a>').join(''); }
</script>
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


def test_it_finds_native_and_aria_filters_with_their_values(page):
    page.set_content(PAGE)
    filters = {f['label']: f for f in page.evaluate(_FILTERS_JS)}
    assert filters['Év']['options'] == ['2026', '2025']
    assert filters['Dokumentum típus']['current'] == 'Havi jelentés'
    assert filters['Alap']['options'] == ['Összes alap', 'Példa Kötvény']


def test_setting_the_chosen_value_and_pressing_the_filters_own_button_reveals_the_documents(page):
    page.set_content(PAGE)
    year = next(f for f in page.evaluate(_FILTERS_JS) if f['label'] == 'Év')
    select_option_anywhere(page, year['id'], '2026', 200)
    assert page.evaluate(_FILTER_SUBMIT_JS) == 'Keresés'
    page.locator('[data-ml-id="jf-submit"]').first.click()
    assert page.locator('#results a').count() == 3


def test_the_site_wide_search_is_not_the_filters_button(page):
    page.set_content(PAGE)
    page.evaluate(_FILTERS_JS)
    page.evaluate(_FILTER_SUBMIT_JS)
    assert page.locator('[data-ml-id="jf-submit"]').get_attribute('id') == 'apply'


def test_a_page_without_filters_offers_nothing(page):
    page.set_content('<html><body><a href="/a.pdf">a</a><button>Menu</button></body></html>')
    assert page.evaluate(_FILTERS_JS) == []
    assert page.evaluate(_FILTER_SUBMIT_JS) is None


class TestFilterValues:
    FILTERS = [{'id': 'jf0', 'label': 'Dokumentum típus', 'current': 'Havi jelentés',
                'options': ['Éves jelentések', 'Havi jelentés']},
               {'id': 'jf1', 'label': 'Év', 'current': '', 'options': ['2026', '2025']}]

    def test_filters_jev_keeps_are_left_alone(self):
        answers = {'jf0': {'choice': 'keep', 'probabilities': {'o0': 0.1, 'o1': 0.2, 'keep': 0.7}},
                   'jf1': {'choice': 'o0', 'probabilities': {'o0': 0.91, 'o1': 0.0, 'keep': 0.09}}}
        with patch.object(JevClient, 'ask', return_value=answers) as ask:
            picks = JevClient(api_key='k').filter_values('2026-os jelentések', {'page_url': 'u'}, self.FILTERS)
        assert picks == {'jf1': ('2026', 0.91)}
        questions = ask.call_args.args[1]
        assert set(questions['jf1']['criteria']) == {'o0', 'o1', 'keep'}

    def test_nothing_is_set_when_jev_is_unavailable(self):
        with patch.object(JevClient, 'ask', return_value=None):
            assert JevClient(api_key='k').filter_values('goal', {}, self.FILTERS) == {}


class TestNeighbours:
    """Revealed links are judged next to the page's other documents, as a normal harvest judges them."""

    def test_neighbours_are_asked_about_but_only_candidates_get_a_score(self):
        revealed = [{'url': f'https://x.dev/havi_2026{m}.pdf', 'name': 'Letöltés'} for m in ('01', '02')]
        page_docs = [{'url': 'https://x.dev/havi_202608.pdf', 'name': '2026. augusztusi jelentés (Alapfigyelő)'}]

        def ask(state, questions):
            assert [link['text'] for link in state['links']][-1] == page_docs[0]['name']
            return {q: {'noul': 0.9 if q != 'q3' else 0.1} for q in questions}

        with patch.object(JevClient, 'ask', side_effect=ask) as mocked:
            scores = JevClient(api_key='k').relevance('2026-os jelentések', 'page', revealed, page_docs)
        assert scores == [0.9, 0.9]
        assert len(mocked.call_args.args[1]) == 3
