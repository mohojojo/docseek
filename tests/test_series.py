"""docseek.series: periods, series keys, and the newest document of each series."""
from __future__ import annotations

import pytest

from docseek.jev_crawl import year_of
from docseek.models import AgenticCrawlResult, AgenticDownload
from docseek.series import apply_latest, mark_latest, period_end, period_of, series_key


class TestPeriod:
    @pytest.mark.parametrize('text, period', [
        ('pelda_Egyensuly_A_202603.pdf', '2026-03'),
        ('Havi portfóliójelentés 2026-04', '2026-04'),
        ('factsheet_20260315.pdf', '2026-03'),
        ('Beschluss vom 15.03.2026', '2026-03'),
        ('Monatsbericht 03/2026', '2026-03'),
        ('2026. március havi jelentés', '2026-03'),
        ('Havi jelentés - 2026. augusztusi', '2026-08'),
        ('Factsheet March 2026', '2026-03'),
        ('Rapport mensuel juillet 2025', '2025-07'),
        ('Informe mensual diciembre 2025', '2025-12'),
        ('Relazione ottobre 2025', '2025-10'),
        ('Monatsbericht_Oktober_K501672_2021.pdf', '2021-10'),
        ('Q1 2026 results', '2026-Q1'),
        ('results_2026_q3.pdf', '2026-Q3'),
        ('1. Quartal 2026', '2026-Q1'),
        ('III. negyedév 2025', '2025-Q3'),
        ('H1 2025 presentation', '2025-H1'),
        ('2. Halbjahr 2025', '2025-H2'),
        ('Half-year report 2025', '2025-H1'),
        ('2025. féléves jelentés', '2025-H1'),
        ('Allegro B 2026-1 hu.pdf', '2026-01'),
        ('Annual report 2025', '2025'),
        ('prospektus.pdf', None),
    ])
    def test_periods_in_many_forms(self, text, period):
        assert period_of(text) == period

    @pytest.mark.parametrize('text', [
        'HAT-450-7/2026. (HAT-15402/2025.)',        # case numbers
        'https://site.example/uploads/2016/05/prospectus.pdf',   # an upload folder
        'Market outlook',                          # 'mar' inside a word is not March
        'Report no. 20261',                        # a longer number
    ])
    def test_things_that_are_not_periods(self, text):
        assert period_of(text) is None

    def test_a_finer_reading_wins(self):
        assert period_of('Annual report 2025 - March 2026 update') == '2026-03'

    def test_periods_compare_by_their_last_month(self):
        assert period_end('2026-Q1') == period_end('2026-03')
        assert period_end('2025') == period_end('2025-H2') == period_end('2025-12')
        assert period_end('2026-01') > period_end('2025')

    def test_the_year_facet_still_ignores_case_numbers(self):
        assert year_of('HAT-450-7/2026. (HAT-15402/2025.)') is None
        assert year_of('Annual report 2025') == '2025'


class TestSeriesKey:
    def test_the_period_is_taken_out_of_the_file_name(self):
        a = series_key('https://site.example/f/fund-a-factsheet-2026-03.pdf')
        assert a == series_key('https://site.example/f/fund-a-factsheet-2026-02.pdf') == 'fund-a-factsheet.pdf'
        assert a != series_key('https://site.example/f/fund-b-factsheet-2026-03.pdf')

    def test_full_dates_and_month_words_go_too(self):
        assert series_key('https://s.example/report_20260315.pdf') == series_key('https://s.example/report_20260412.pdf')
        assert series_key('https://s.example/Havi_jelentes_2026_marcius.pdf') == \
            series_key('https://s.example/Havi_jelentes_2026_aprilis.pdf')

    def test_an_id_after_the_file_name_is_skipped(self):
        assert series_key('https://s.example/documents/10/fund-a-2026-03.pdf/79dc3842-ed58-f28c-423a-d91a891cd8d9') == \
            series_key('https://s.example/documents/10/fund-a-2026-04.pdf/ae72503e-09eb-af91-e8e3-c6593df74ff5') == \
            'fund-a.pdf'

    def test_a_different_format_is_a_different_series(self):
        assert series_key('https://s.example/r-2026-03.pdf') != series_key('https://s.example/r-2026-03.xlsx')

    @pytest.mark.parametrize('url', ['https://s.example/getfile.aspx?id=12', 'https://s.example/download/4411',
                                     'https://s.example/files/9f86d081884c7d659a2feaa0c55ad015.pdf'])
    def test_an_opaque_file_name_falls_back_to_the_document_name(self, url):
        assert series_key(url, 'Fund A monthly report March 2026') == \
            series_key(url.replace('12', '13'), 'Fund A monthly report April 2026')
        assert series_key(url, 'Fund A monthly report March 2026') != series_key(url, 'Fund B monthly report March 2026')


def doc(url, verdict='accepted', name=''):
    return AgenticDownload(url=url, name=name or url.rsplit('/', 1)[-1], reason='', source_page='https://s.example/',
                           verdict=verdict)


class TestLatest:
    def test_each_series_keeps_its_newest_however_old(self):
        docs = [doc('https://s.example/fund-a-2026-02.pdf'), doc('https://s.example/fund-a-2026-03.pdf'),
                doc('https://s.example/closed-fund-2023-11.pdf'), doc('https://s.example/prospectus.pdf')]
        superseded = mark_latest(docs)
        assert [d.url for d in superseded] == ['https://s.example/fund-a-2026-02.pdf']
        assert [d.latest_in_series for d in docs] == [False, True, True, None]

    def test_a_quarter_and_its_last_month_tie_and_both_stay(self):
        docs = [doc('https://s.example/r-2026-Q1.pdf'), doc('https://s.example/r-2026-03.pdf')]
        assert series_key(docs[0].url) == series_key(docs[1].url) and mark_latest(docs) == []

    def test_a_rejected_document_supersedes_nothing(self):
        docs = [doc('https://s.example/fund-a-2026-02.pdf'), doc('https://s.example/fund-a-2026-03.pdf', 'rejected')]
        assert mark_latest(docs) == [] and [d.latest_in_series for d in docs] == [True, None]

    def test_apply_latest_filters_only_when_asked(self):
        def result():
            return AgenticCrawlResult(start_url='https://s.example/', goal='g', downloads=[
                doc('https://s.example/fund-a-2026-02.pdf'), doc('https://s.example/fund-a-2026-03.pdf')])
        marked = apply_latest(result(), latest=False)
        assert len(marked.downloads) == 2 and marked.superseded_count == 0
        assert marked.downloads[0].series == 'fund-a.pdf'
        filtered = apply_latest(result(), latest=True)
        assert [d.url for d in filtered.downloads] == ['https://s.example/fund-a-2026-03.pdf']
        assert filtered.superseded_count == 1
        everything = apply_latest(result(), latest=True, include_rejected=True)
        assert len(everything.downloads) == 2 and everything.superseded_count == 1


def test_the_api_marks_every_response_and_filters_on_request(monkeypatch):
    from fastapi.testclient import TestClient

    import docseek.server as server
    monkeypatch.delenv('CRAWLER_API_KEY', raising=False)
    monkeypatch.setattr(server, '_agent_llm', lambda payload: object())
    monkeypatch.setattr(server, '_run_crawl', lambda *a, **k: AgenticCrawlResult(
        start_url='https://s.example/', goal='g', downloads=[doc('https://s.example/fund-a-2026-02.pdf'),
                                                             doc('https://s.example/fund-a-2026-03.pdf')]))
    client = TestClient(server.app)
    plain = client.post('/v1/discover', json={'url': 'https://s.example/', 'goal': 'g'}).json()
    assert [d['latest_in_series'] for d in plain['downloads']] == [False, True]
    latest = client.post('/v1/discover', json={'url': 'https://s.example/', 'goal': 'g', 'latest': True}).json()
    assert [d['period'] for d in latest['downloads']] == ['2026-03'] and latest['superseded_count'] == 1


class TestNumberedIssues:
    @pytest.mark.parametrize('name, url, expected', [
        ('Economic Bulletin Issue 8, 2025', 'https://s.example/pub/eb202508.en.pdf', ('2025', 8)),
        ('SAN-2025-12', 'https://s.example/files/2025-12/san-2025-12.pdf', ('2025', 12)),
        ('Monatsbericht Heft 3', 'https://s.example/2026/heft3.pdf', (None, 3)),     # a year in a folder is not read
        ('Factsheet No. 3 March 2026', 'https://s.example/f.pdf', ('2026-03', 3)),
        ('Factsheet March 2026', 'https://s.example/f-2026-03.pdf', ('2026-03', None)),
    ])
    def test_a_numbered_issue_is_its_year_and_its_number(self, name, url, expected):
        from docseek.series import document_period
        assert document_period(name, url) == expected

    def test_the_latest_issue_is_the_highest_number_of_the_newest_year(self):
        docs = [doc(f'https://s.example/pub/eb2025{n:02d}.en.pdf', name=f'Economic Bulletin Issue {n}, 2025')
                for n in (1, 2, 8)]
        docs.append(doc('https://s.example/pub/eb202408.en.pdf', name='Economic Bulletin Issue 8, 2024'))
        mark_latest(docs)
        assert [d.latest_in_series for d in docs] == [False, False, True, False]
        assert {d.period for d in docs} == {'2025', '2024'}
