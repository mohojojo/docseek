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


# A regulator's filing page, reduced: the files are "Download" links beside file names that carry the period end in
# their name, and the filing's own date sits in a labelled row further up.
FILING = """<html><body><main><h1>Audited Annual Report for 2025</h1>
<table><tr><td>Date</td><td>2026-03-19 09:56:26</td></tr>
<tr><td>Files</td><td><div class="files">
  <div class="row"><span>Annual_report_2025.pdf (14569 kB)</span> <a href="/?task=download&f_id=1">Download</a></div>
  <div class="row"><span>bank-2025-12-31-en.zip (9566 kB)</span> <a href="/?task=download&f_id=2">Download</a></div>
</div></td></tr></table>
<section><div><div><p>Published 12.03.2026 <a href="/?task=download&f_id=3">Download</a></p></div></div></section>
<section><div><div><p>Közzétéve: 2026.09.22. <a href="/?task=download&f_id=4">Letöltés</a></p></div></div></section>
<section><div><div><p>2026. május 12-én kelt <a href="/?task=download&f_id=5">Letöltés</a></p></div></div></section>
</main></body></html>"""


def test_a_date_glued_into_a_file_name_is_not_the_rows_dated_line(page):
    page.set_content(FILING)
    dated = {link['href'].rsplit('=', 1)[-1]: link['dated'] for link in page.evaluate(_HARVEST_JS)}
    assert dated['1'] == ''                      # no date in the row, and the filing's date is further up
    assert dated['2'] == ''                      # 2025-12-31 is part of the file name
    assert dated['3'] == '12.03.2026'            # a date written beside the link still counts
    assert dated['4'] == '2026.09.22.'           # the Hungarian numeric form, day included
    assert dated['5'] == ''                      # no dated line: "12-én" is a day with a suffix, as before


# A regulator's decisions listing, reduced: the header menu, the listing in <main>, its paginator in a <nav>, and a
# sidebar <nav> of the site's sections inside the content.
MENUS = """<html><body>
<header><nav><a href="/decisions">Decisions</a><a href="/about">About</a></nav></header>
<main>
  <table><tr><td><a href="/decision/1">NAIH-1-2022</a></td></tr></table>
  <nav class="pagination"><a href="/decisions?start=50">2</a><a href="/decisions?start=100">3</a><a href="/decisions?start=50">Next</a></nav>
  <nav aria-label="related"><a href="/decisions/by-tag">By tag</a></nav>
</main>
<footer><a href="/privacy">Privacy</a></footer>
</body></html>"""


def test_menus_are_chrome_a_sidebar_too_but_a_paginator_is_not(page):
    page.set_content(MENUS)
    keys = ('decisions?start=50', 'decisions?start=100', 'decisions/by-tag', 'decision/1', 'decisions', 'about', 'privacy')
    chrome = {next(k for k in keys if link['href'].endswith(k)): link['chrome'] for link in page.evaluate(_HARVEST_JS)}
    assert chrome['decisions'] and chrome['about'] and chrome['privacy']
    assert not chrome['decision/1'] and not chrome['decisions?start=50'] and not chrome['decisions?start=100']
    assert chrome['decisions/by-tag']
