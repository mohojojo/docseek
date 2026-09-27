"""docseek.codegen.programs: the program store, hybrid discovery and the drift check."""
from __future__ import annotations

import pytest

from docseek.codegen.programs import (
    Program, ProgramStore, check_drift, drift_status, generate_program, health, hybrid_discover, program_key,
)
from docseek.models import AgenticCrawlResult, AgenticDownload

START, GOAL = 'https://www.site.example/reports', 'Find the annual reports'
KEY = program_key(START, GOAL)


class FakeJudge:
    model = 'fake-judge'

    def __init__(self, scores=None):
        self.scores = scores or {}

    def relevance(self, goal, page, candidates, neighbours=()):
        return [self.scores.get(c['url'], 0.9) for c in candidates]


def runner_returning(urls, error=None):
    def runner(code, fetcher):
        return {'documents': [{'url': u, 'name': u.rsplit('/', 1)[-1], 'context': ''} for u in urls],
                'error': error, 'fetch_failures': 0, 'requests': 3, 'renders': 0, 'seconds': 0.2}
    return runner


def store_with_program(tmp_path, **meta) -> ProgramStore:
    store = ProgramStore(tmp_path)
    store.save(Program(KEY, 'def discover(fetch, render):\n    return []', {'goal': GOAL, 'start_url': START, **meta}))
    return store


def crawled():
    calls = []

    def crawl():
        calls.append(1)
        return AgenticCrawlResult(start_url=START, goal=GOAL, decision_model='jev', downloads=[
            AgenticDownload(url='https://site.example/c.pdf', name='c', reason='', source_page=START, source='page')])
    return crawl, calls


DOCS = [f'https://site.example/{n}.pdf' for n in range(4)]


class TestHealth:
    def test_health_rules(self):
        assert health('Traceback', 5, 5) == 'error'
        assert health(None, 0, 4) == 'empty'
        assert health(None, 0, 0) == 'healthy'            # a site with nothing to find stays healthy when empty
        assert health(None, 1, 4) == 'dropped'
        assert health(None, 2, 4) == 'healthy'
        assert health(None, 3, None) == 'healthy'


class TestHybrid:
    def test_a_healthy_program_answers_without_a_crawl(self, tmp_path):
        store = store_with_program(tmp_path, last_kept=3)
        crawl, calls = crawled()
        judge = FakeJudge({DOCS[3]: 0.1})                  # one of the four is rejected
        result = hybrid_discover(START, GOAL, store=store, judge=judge, crawl=crawl, runner=runner_returning(DOCS))
        assert not calls
        assert result.program['path'] == 'program' and result.program['reason'] == 'healthy'
        assert [d.url for d in result.downloads] == DOCS[:3] and result.rejected_count == 1
        assert {d.source for d in result.downloads} == {'program'} and result.decision_model == 'program'
        assert store.load(KEY).meta['last_kept'] == 3

    def test_no_program_means_a_crawl(self, tmp_path):
        crawl, calls = crawled()
        result = hybrid_discover(START, GOAL, store=ProgramStore(tmp_path), judge=FakeJudge(), crawl=crawl)
        assert calls and result.program == {'key': KEY, 'path': 'crawl', 'reason': 'no_program'}

    @pytest.mark.parametrize('urls,error,reason', [(DOCS, 'Traceback: boom', 'error'), ([], None, 'empty'),
                                                   (DOCS[:1], None, 'dropped')])
    def test_an_unhealthy_program_is_marked_stale_and_the_crawl_answers(self, tmp_path, urls, error, reason):
        store = store_with_program(tmp_path, last_kept=4)
        crawl, calls = crawled()
        result = hybrid_discover(START, GOAL, store=store, judge=FakeJudge(), crawl=crawl,
                                 runner=runner_returning(urls, error))
        assert calls and result.program['reason'] == reason and result.downloads[0].url.endswith('c.pdf')
        meta = store.load(KEY).meta
        assert meta['stale'] and meta['failures'] == 1 and meta['last_reason'] == reason

    def test_a_stale_program_is_not_run_again(self, tmp_path):
        store = store_with_program(tmp_path, stale=True)
        crawl, calls = crawled()

        def must_not_run(code, fetcher):
            raise AssertionError('a stale program ran')
        result = hybrid_discover(START, GOAL, store=store, judge=FakeJudge(), crawl=crawl, runner=must_not_run)
        assert calls and result.program['reason'] == 'stale'


class TestGenerate:
    def test_a_generated_program_is_stored_fresh_with_what_it_kept(self, tmp_path):
        store = store_with_program(tmp_path, stale=True, failures=2)

        class FakeExplorer:
            def __init__(self, start_url, goal, llm, judge):
                pass

            def explore(self):
                return {'code': 'def discover(fetch, render):\n    return [1]', 'notes': 'n', 'submitted': True,
                        'data_hosts': ['api.backend.example'], 'post_endpoints': [], 'generated_kept': 7}
        program, report = generate_program(store, START, GOAL, llm=None, judge=FakeJudge(), explorer_cls=FakeExplorer)
        assert report['notes'] == 'n' and 'code' not in report
        loaded = store.load(KEY)
        assert program.key == KEY and loaded.code.endswith('return [1]\n')
        assert loaded.meta['stale'] is False and loaded.meta['last_kept'] == 7 and 'failures' not in loaded.meta
        assert loaded.fetcher().data_hosts == {'api.backend.example'}

    def test_no_code_stores_nothing(self, tmp_path):
        class Empty:
            def __init__(self, *a, **k):
                pass

            def explore(self):
                return {'code': ''}
        store = ProgramStore(tmp_path)
        assert generate_program(store, START, GOAL, llm=None, judge=FakeJudge(), explorer_cls=Empty) == (None, {})
        assert store.keys() == []


class TestDrift:
    def test_statuses(self):
        then = {'urls': DOCS}
        assert drift_status({'urls': DOCS, 'error': None}, then) == ('ok', 1.0)
        assert drift_status({'urls': DOCS + ['https://site.example/new.pdf'], 'error': None}, then)[0] == 'grew'
        assert drift_status({'urls': DOCS[:2], 'error': None}, then) == ('shrank', 0.5)
        assert drift_status({'urls': [], 'error': None}, then)[0] == 'broken'
        assert drift_status({'urls': DOCS, 'error': 'boom'}, then)[0] == 'broken'
        assert drift_status({'urls': [], 'error': None}, {'urls': []})[0] == 'ok'

    def test_the_first_check_takes_a_snapshot_and_the_next_compares(self, tmp_path):
        store = store_with_program(tmp_path)
        assert check_drift(store, runner=runner_returning(DOCS))[0]['status'] == 'snapshot'
        row = check_drift(store, runner=runner_returning(DOCS[:1]))[0]
        assert (row['status'], row['then'], row['now']) == ('shrank', 4, 1)


class TestStore:
    def test_keys_are_validated_so_a_request_cannot_reach_other_files(self, tmp_path):
        store = ProgramStore(tmp_path)
        with pytest.raises(KeyError):
            store.load('../../etc/passwd')
        assert program_key('https://WWW.Site.Example:8080/x', 'g') == program_key('https://site.example:8080/y', 'g')

    def test_delete_removes_the_program_and_its_snapshot(self, tmp_path):
        store = store_with_program(tmp_path)
        store.save_snapshot(KEY, {'date': 'd', 'urls': []})
        assert store.delete(KEY) and store.keys() == [] and not list(tmp_path.iterdir())


class TestVerifiedAndLog:
    def test_an_unverified_program_never_answers(self, tmp_path):
        store = store_with_program(tmp_path, verified=False, last_kept=0)
        crawl, calls = crawled()

        def must_not_run(code, fetcher):
            raise AssertionError('an unverified program ran')
        result = hybrid_discover(START, GOAL, store=store, judge=FakeJudge(), crawl=crawl, runner=must_not_run)
        assert calls and result.program['reason'] == 'unverified'

    def test_a_failed_generation_still_leaves_its_log(self, tmp_path):
        class Failed:
            def __init__(self, *a, **k):
                pass

            def explore(self):
                return {'code': '', 'notes': 'not submitted', 'turns': 42,
                        'log': [{'turn': 1, 'tool': 'fetch_page', 'result': '403 Access Denied'}]}
        store = ProgramStore(tmp_path)
        program, report = generate_program(store, START, GOAL, llm=None, judge=FakeJudge(), explorer_cls=Failed)
        log = store.load_log(KEY)
        assert program is None and report['turns'] == 42 and 'log' not in report
        assert log['log'][0]['result'] == '403 Access Denied' and store.last_attempt(KEY) is not None
        assert store.keys() == []                          # a log alone is not a program
