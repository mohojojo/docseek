"""The server's generated-program surface: `programs` on /v1/discover and the /v1/programs endpoints."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import docseek.codegen.programs as programs
import docseek.server as server
from docseek.codegen.programs import Program, ProgramStore, program_key
from docseek.models import AgenticCrawlResult, AgenticDownload

URL, GOAL = 'https://www.site.example/', 'Find the annual reports'
KEY = program_key(URL, GOAL)


class FakeJudge:
    model = 'fake-judge'

    def relevance(self, goal, page, candidates, neighbours=()):
        return [0.9 for _ in candidates]


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv('CRAWLER_API_KEY', raising=False)
    monkeypatch.setattr(server, 'PROGRAMS_DIR', str(tmp_path))
    monkeypatch.setattr(server, '_generating', set())
    monkeypatch.setattr(server, '_agent_llm', lambda payload: object())
    monkeypatch.setattr(server, 'make_judge', lambda judge=None, profile=None, llm=None: FakeJudge())
    return TestClient(server.app)


def save_program(tmp_path, **meta):
    ProgramStore(tmp_path).save(Program(KEY, 'def discover(fetch, render):\n    return []',
                                        {'goal': GOAL, 'start_url': URL, 'generated': '2026-09-27', **meta}))


def crawl_result(*args, **kwargs):
    return AgenticCrawlResult(start_url=URL, goal=GOAL, decision_model='jev', downloads=[
        AgenticDownload(url='https://site.example/crawled.pdf', name='c', reason='', source_page=URL, source='page')])


def test_programs_need_a_programs_dir(client, monkeypatch):
    monkeypatch.setattr(server, 'PROGRAMS_DIR', None)
    response = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': True})
    assert response.status_code == 404 and 'PROGRAMS_DIR' in response.json()['detail']


def test_programs_are_the_default_once_a_programs_dir_is_set(client, monkeypatch):
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL}).json()
    assert body['program'] == {'key': KEY, 'path': 'crawl', 'reason': 'no_program', 'generation': 'no_model'}


def test_without_a_programs_dir_the_default_is_a_plain_crawl(client, monkeypatch):
    monkeypatch.setattr(server, 'PROGRAMS_DIR', None)
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL}).json()
    assert body['program'] is None and body['downloads'][0]['source'] == 'page'


def test_a_healthy_program_answers_and_nothing_is_crawled(client, monkeypatch, tmp_path):
    save_program(tmp_path, last_kept=1)
    monkeypatch.setattr(server, '_run_crawl', lambda *a, **k: pytest.fail('crawled despite a healthy program'))
    monkeypatch.setattr(programs, 'run_saved', lambda program, judge, runner=None: (
        {'error': None, 'requests': 2, 'renders': 0, 'fetch_failures': 0},
        [AgenticDownload(url='https://site.example/a.pdf', name='a', reason='', source_page=URL, relevance=0.9,
                         verdict='accepted', source='program')]))
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': True}).json()
    assert body['program']['path'] == 'program' and body['downloads'][0]['source'] == 'program'


def test_without_a_program_the_crawl_answers_and_a_program_is_written_in_the_background(client, monkeypatch, tmp_path):
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    monkeypatch.setattr(server, 'codegen_llm', lambda: object())
    written = []
    monkeypatch.setattr(server, 'generate_program', lambda store, url, goal, llm, judge: (written.append((url, goal)), {}))
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': True}).json()
    assert body['program'] == {'key': KEY, 'path': 'crawl', 'reason': 'no_program', 'generation': 'started'}
    assert body['downloads'][0]['url'].endswith('crawled.pdf')
    for _ in range(50):
        if written and not server._generating:
            break
        time.sleep(0.05)
    assert written == [(URL, GOAL)] and not server._generating


def test_without_a_coding_model_the_crawl_still_answers(client, monkeypatch):
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    monkeypatch.setattr(server, 'codegen_llm', lambda: None)
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': True}).json()
    assert body['program']['generation'] == 'no_model' and body['downloads']


def test_programs_false_is_a_plain_crawl(client, monkeypatch):
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': False}).json()
    assert body['program'] is None


def test_list_get_and_delete(client, tmp_path):
    save_program(tmp_path, last_kept=3)
    listed = client.get('/v1/programs').json()
    assert listed[0]['key'] == KEY and listed[0]['last_kept'] == 3
    got = client.get(f'/v1/programs/{KEY}').json()
    assert got['code'].startswith('def discover') and got['meta']['goal'] == GOAL
    assert client.delete(f'/v1/programs/{KEY}').json() == {'deleted': True, 'key': KEY}
    assert client.get(f'/v1/programs/{KEY}').status_code == 404
    assert client.get('/v1/programs/not..a..key').status_code == 404


def test_generating_needs_a_coding_model(client, monkeypatch):
    monkeypatch.setattr(server, 'codegen_llm', lambda: None)
    response = client.post('/v1/programs', json={'url': URL, 'goal': GOAL})
    assert response.status_code == 503 and 'CODEGEN_MODEL' in response.json()['detail']


def test_generating_starts_in_the_background(client, monkeypatch):
    monkeypatch.setattr(server, 'codegen_llm', lambda: object())
    monkeypatch.setattr(server, 'generate_program', lambda *a, **k: (time.sleep(0.2), {}))
    response = client.post('/v1/programs', json={'url': URL, 'goal': GOAL})
    assert response.status_code == 202 and response.json() == {'key': KEY, 'generation': 'started'}
    again = client.post('/v1/programs', json={'url': URL, 'goal': GOAL}).json()
    assert again['generation'] == 'already_running'


def test_the_drift_check_endpoint(client, monkeypatch):
    monkeypatch.setattr(server, 'check_drift', lambda store, keys: [{'key': KEY, 'status': 'ok', 'keys': keys}])
    assert client.post('/v1/programs/check', json={}).json() == [{'key': KEY, 'status': 'ok', 'keys': None}]


def test_a_recent_attempt_holds_back_automatic_regeneration_but_not_an_explicit_one(client, monkeypatch, tmp_path):
    ProgramStore(tmp_path).save_log(KEY, {'log': []})
    monkeypatch.setattr(server, '_run_crawl', crawl_result)
    monkeypatch.setattr(server, 'codegen_llm', lambda: object())
    monkeypatch.setattr(server, 'generate_program', lambda *a, **k: (None, {}))
    body = client.post('/v1/discover', json={'url': URL, 'goal': GOAL, 'programs': True}).json()
    assert body['program']['generation'] == 'recently_tried'
    explicit = client.post('/v1/programs', json={'url': URL, 'goal': GOAL}).json()
    assert explicit['generation'] == 'started'
