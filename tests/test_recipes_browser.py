"""Replaying a recipe on real markup."""
import pytest

from docseek.recipes import Recipe, RecipeStep, replay

playwright = pytest.importorskip('playwright.sync_api')

PAGE = """<html><body>
<label for="yr">Év</label><select id="yr" onchange="document.querySelector('#y').textContent=this.value"></select>
<button id="go">Keresés</button>
<span id="y"></span><div id="out"></div>
<script>for (const y of ['2024','2025','2026']) document.querySelector('#yr').add(new Option(y, y));
document.querySelector('#go').onclick = () => { if (document.querySelector('#y').textContent === '2026')
  document.querySelector('#out').innerHTML = '<a href="/havi_2026.pdf">2026</a>'; };</script>
</body></html>"""


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f'chromium not available: {exc}')
        yield browser.new_page()
        browser.close()


def test_a_select_then_click_recipe_reveals_what_the_agent_revealed(page):
    page.set_content(PAGE)
    recipe = Recipe(host='site-hu.example', path='/p', goal='g', revealed=1, steps=[
        RecipeStep(action='select', role='combobox', name='Év', attributes={'id': 'yr'}, value='2026'),
        RecipeStep(action='click', role='button', name='Keresés')])
    assert replay(page, recipe, settle=lambda: page.wait_for_timeout(100)) == 2
    assert page.locator('#out a').count() == 1


def test_a_missing_element_stops_the_replay(page):
    page.set_content(PAGE)
    recipe = Recipe(host='site-hu.example', path='/p', goal='g', revealed=1, steps=[
        RecipeStep(action='click', role='button', name='Nincs ilyen'), RecipeStep(action='click', role='button', name='Keresés')])
    assert replay(page, recipe, settle=lambda: None) == 0
