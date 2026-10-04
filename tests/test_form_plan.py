"""A page's form, listed by code, planned by one model call, set by code. Offline, real browser."""
import datetime

import pytest

from docseek.form_plan import apply_plan, date_like, plan_form, read_form, validated
from docseek.llm import LLMUnavailable

playwright = pytest.importorskip('playwright.sync_api')

# A regulator's filing search, reduced: document types as checkboxes sharing one name, a date range typed
# into two fields whose picker accepts only dd.mm.yyyy and writes today's date over anything else, an icon
# button beside the dropdown, and "Search" / "Clear". Around it: the site search, a login and a newsletter.
PAGE = """<html><body>
<header><form role="search"><input name="q" placeholder="Search the site"><button>Go</button></form></header>
<form id="login"><input name="user"><input type="password" name="pw"><button>Log in</button></form>
<form name="filter" onsubmit="event.preventDefault(); apply(event.submitter)">
  <select name="doc_language"><option value="">All languages</option><option>LV</option><option>EN</option></select>
  <button class="icon-select">&nbsp;</button>
  <input type="text" name="doc_datefrom" placeholder="From:" onblur="picker(this)">
  <input type="text" name="doc_dateto" placeholder="To:" onblur="picker(this)">
  <input type="text" name="doc_keywords" placeholder="Keywords">
  <input type="submit" name="SEARCH" id="search" value="Search">
  <input type="submit" name="RESET" value="Clear">
  <div id="select_doc_type">
    <label for="t101"><input id="t101" type="checkbox" name="doc_types[]" value="101"> 1. Periodic information</label>
    <label for="t111"><input id="t111" type="checkbox" name="doc_types[]" value="111"> 1.1 Annual financial reports</label>
    <label for="t112"><input id="t112" type="checkbox" name="doc_types[]" value="112"> 1.3 Half-yearly financial report</label>
    <label for="t100"><input id="t100" type="checkbox" name="doc_types[]" value="100" disabled> Till 2017.03.01</label>
  </div>
</form>
<form id="newsletter"><input type="email" name="mail"><input name="first_name"><button>Subscribe</button></form>
<div id="results"><a href="/?view=details&id=9">Notice of a meeting</a></div>
<script>
  function two(n) { return String(n).padStart(2, '0'); }   // a function: set_content keeps the window
  function picker(el) { if (el.value && !/^\\d\\d\\.\\d\\d\\.\\d{4}$/.test(el.value)) { const t = new Date();
    el.value = two(t.getDate()) + '.' + two(t.getMonth() + 1) + '.' + t.getFullYear(); } }
  function apply(by) { const f = document.forms.filter;
    if (by && by.id === 'search' && document.getElementById('t111').checked)
      document.getElementById('results').innerHTML = '<a href="/?view=details&id=1">Annual report 2025 ('
        + f.doc_datefrom.value + ' - ' + f.doc_dateto.value + ')</a>'; }
</script>
</body></html>"""

FORM = {'controls': [{'id': 'fp0', 'kind': 'select', 'label': 'doc_language', 'current': 'All languages',
                      'options': ['All languages', 'LV', 'EN']},
                     {'id': 'fp1', 'kind': 'checkboxes', 'label': 'doc_types[]', 'current': '',
                      'options': ['1. Periodic information', '1.1 Annual financial reports']},
                     {'id': 'fp2', 'kind': 'text', 'label': 'From:', 'current': '', 'hint': 'From:'}],
        'buttons': [{'id': 'fpb0', 'label': 'Search'}, {'id': 'fpb1', 'label': 'Clear'}]}


@pytest.fixture(scope='module')
def page():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - no browser on this machine
            pytest.skip(f'chromium not available: {exc}')
        yield browser.new_page()
        browser.close()


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def complete_json(self, system, user):
        self.asked.append(user)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_it_lists_every_control_of_the_listings_form_and_none_of_the_sites_chrome(page):
    page.set_content(PAGE)
    form = read_form(page)
    by_label = {c['label']: c for c in form['controls']}
    assert set(by_label) == {'doc_language', 'doc_types[]', 'From:', 'To:', 'Keywords'}   # no search, login, newsletter
    assert by_label['doc_types[]']['kind'] == 'checkboxes'
    assert by_label['doc_types[]']['options'] == ['1. Periodic information', '1.1 Annual financial reports',
                                                 '1.3 Half-yearly financial report']       # not the disabled one
    assert by_label['doc_language']['options'] == ['All languages', 'LV', 'EN']
    assert [b['label'] for b in form['buttons']] == ['Search', 'Clear']                     # not the icon button


def test_a_plan_is_applied_and_a_date_is_typed_the_way_the_picker_writes_one(page):
    page.set_content(PAGE)
    form = read_form(page)
    ids = {c['label']: c['id'] for c in form['controls']}
    plan = validated({'set': [{'id': ids['doc_types[]'], 'value': '1.1 Annual financial reports'},
                              {'id': ids['From:'], 'value': '2026-01-01'}, {'id': ids['To:'], 'value': '2026-12-31'}],
                      'press': form['buttons'][0]['id']}, form)
    apply_plan(page, plan)
    assert page.locator('#t111').is_checked() and not page.locator('#t112').is_checked()
    assert page.locator('#results a').inner_text() == 'Annual report 2025 (01.01.2026 - 31.12.2026)'


def test_a_date_the_picker_will_not_take_leaves_the_field_empty(page):
    page.set_content(PAGE.replace("el.value = two(t.getDate())", "el.value = 'invalid'; return; el.value = two(t.getDate())"))
    form = read_form(page)
    ids = {c['label']: c['id'] for c in form['controls']}
    apply_plan(page, validated({'set': [{'id': ids['From:'], 'value': '2026-01-01'}], 'press': None}, form))
    assert page.locator('[name=doc_datefrom]').input_value() in ('', 'invalid')   # never a date nobody asked for
    assert page.locator('#results a').inner_text() == 'Notice of a meeting'       # and nothing was submitted


class TestValidated:
    def test_only_controls_and_options_the_page_offers_survive(self):
        plan = validated({'set': [{'id': 'fp9', 'value': 'x'},                                # no such control
                                  {'id': 'fp0', 'value': 'DE'},                               # no such option
                                  {'id': 'fp1', 'value': ['1.1 annual financial reports', 'Other']},
                                  {'id': 'fp2', 'value': '2026-01-01'}], 'press': 'fpb7'}, FORM)
        assert [(s['id'], s['values']) for s in plan['set']] == [('fp1', ['1.1 Annual financial reports']),
                                                                 ('fp2', ['2026-01-01'])]
        assert plan['press'] is None

    def test_a_value_the_control_already_has_is_not_a_step(self):
        assert validated({'set': [{'id': 'fp0', 'value': 'All languages'}], 'press': 'fpb0'}, FORM)['set'] == []

    def test_nothing_to_set_is_an_empty_plan_not_a_failure(self):
        assert validated({'set': [], 'press': None}, FORM) == {'set': [], 'press': None, 'press_label': ''}
        assert validated({}, FORM)['set'] == []

    def test_free_text_is_bounded(self):
        assert validated({'set': [{'id': 'fp2', 'value': 'x' * 200}]}, FORM)['set'] == []
        assert validated({'set': [{'id': 'fp2', 'value': {'a': 1}}]}, FORM)['set'] == []


class TestPlanForm:
    def test_the_model_sees_the_goal_and_the_controls_and_its_answer_is_validated(self):
        llm = FakeLLM({'set': [{'id': 'fp1', 'value': '1.1 Annual financial reports'}], 'press': 'fpb0'})
        plan = plan_form(llm, 'Annual reports published in 2026', 'https://site-lv.example/', 'Documents', [], FORM)
        assert plan['press_label'] == 'Search' and plan['set'][0]['kind'] == 'checkboxes'
        assert 'Annual reports published in 2026' in llm.asked[0] and '1.1 Annual financial reports' in llm.asked[0]

    @pytest.mark.parametrize('failure', [LLMUnavailable('no key'), ValueError('not JSON')])
    def test_a_model_that_cannot_answer_gives_no_plan(self, failure):
        assert plan_form(FakeLLM(failure), 'goal', 'https://site-lv.example/', 'Documents', [], FORM) is None


class TestDateLike:
    TODAY = datetime.date(2026, 10, 4)

    @pytest.mark.parametrize('sample, written', [
        ('04.10.2026', '01.03.2026'), ('10/04/2026', '03/01/2026'), ('4.10.2026', '1.3.2026'),
        ('2026.10.04.', '2026.03.01.'), ('04-10-26', '01-03-26')])
    def test_the_date_is_written_the_way_the_picker_wrote_today(self, sample, written):
        assert date_like(sample, '2026-03-01', self.TODAY) == written

    def test_a_field_that_does_not_show_today_gives_no_format(self):
        assert date_like('01.01.1970', '2026-03-01', self.TODAY) is None
        assert date_like('invalid', '2026-03-01', self.TODAY) is None
        assert date_like('04.10.2026', 'March 2026', self.TODAY) is None
