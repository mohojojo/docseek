"""Where a link sits names its document type when its own text does not. Offline, real browser."""
import pytest

from docseek.jev_crawl import _HARVEST_JS

playwright = pytest.importorskip('playwright.sync_api')

# A company results page, reduced: a table built from divs, every cell repeating its column name for narrow screens,
# and three links in a row that all read "PDF".
DIV_TABLE = """<html><body><main><h2>Full Year Results 2022/23</h2>
<div class="table_row">
  <div class="table_small"><div class="table_cell">Date</div><div class="table_cell">15 Jun</div></div>
  <div class="table_small"><div class="table_cell">Reports</div>
    <div class="table_cell"><div class="inner"><a href="/r/report.pdf">PDF</a></div></div></div>
  <div class="table_small"><div class="table_cell">Transcript</div>
    <div class="table_cell"><div class="inner"><a href="/r/summary-with-qa.pdf">PDF</a></div></div></div>
  <div class="table_small"><div class="table_cell">Slides</div>
    <div class="table_cell"><div class="inner"><a href="/r/presentation.pdf">PDF</a></div></div></div>
</div></main></body></html>"""

# A college procurement page, reduced: two tables switched by buttons, tied to them only by their ids.
TABS = """<html><body><main><h2>Procurement</h2>
<button id="tab-rfqs" onclick="show('rfqs')">Request for Quotations</button>
<button id="tab-tenders" onclick="show('tenders')">Bid Documents</button>
<div id="tenders-table" style="display:none"><table><tr><th>Title</th><th>Actions</th></tr>
  <tr><td>Supply of desktops</td><td><a href="/d/bid.pdf">View</a></td></tr></table></div>
<div id="rfqs-table"><table><tr><th>Title</th><th>Actions</th></tr>
  <tr><td>Graduation decor</td><td><a href="/d/rfq.pdf">View</a></td></tr></table></div>
<div role="tablist"><button role="tab" aria-controls="p-annual">Annual reports</button></div>
<div id="p-annual" role="tabpanel"><a href="/d/annual.pdf">Download</a></div>
</main></body></html>"""


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f'chromium not available: {exc}')
        yield browser.new_page()
        browser.close()


def harvest(page, html: str) -> dict[str, dict]:
    page.set_content(html)
    return {link['href'].rsplit('/', 1)[-1]: link for link in page.evaluate(_HARVEST_JS)}


def test_a_div_table_cell_names_its_column(page):
    links = harvest(page, DIV_TABLE)
    assert links['report.pdf']['column'] == 'Reports'
    assert links['summary-with-qa.pdf']['column'] == 'Transcript'
    assert links['presentation.pdf']['column'] == 'Slides'


def test_a_link_in_a_tab_panel_carries_the_tab(page):
    links = harvest(page, TABS)
    assert links['bid.pdf']['section'].endswith('Bid Documents')
    assert links['rfq.pdf']['section'].endswith('Request for Quotations')
    assert links['annual.pdf']['section'].endswith('Annual reports')


def test_a_real_table_keeps_its_header_cell(page):
    links = harvest(page, TABS)
    assert links['bid.pdf']['column'] == 'Actions'


def test_an_ordinary_link_gets_no_invented_label(page):
    links = harvest(page, '<html><body><main><h2>Reports</h2><p>Latest:</p>'
                          '<p><a href="/x/annual-2025.pdf">Annual report 2025</a></p></main></body></html>')
    assert links['annual-2025.pdf']['section'] == 'Reports'
    assert links['annual-2025.pdf']['column'] == ''
