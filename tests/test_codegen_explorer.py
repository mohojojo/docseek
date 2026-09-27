"""docseek.codegen.explorer: the agent loop that writes a discovery program, on any docseek.llm client."""
from __future__ import annotations

import json

from docseek.codegen.explorer import Explorer, _move_cache_breakpoint
from docseek.llm import ChatReply

GOOD = 'def discover(fetch, render):\n    return [{"url": "https://site.example/a.pdf", "name": "A", "context": "x"}]\n'
WEAK = 'def discover(fetch, render):\n    return []\n'


class ScriptedLLM:
    """Answers each turn with the next scripted tool call and remembers what it was shown."""

    provider, model = 'scripted', 'scripted-coder'

    def __init__(self, calls, input_tokens=100):
        self.calls, self.input_tokens, self.seen = list(calls), input_tokens, []

    def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192):
        self.seen.append(json.loads(json.dumps(messages)))
        name, arguments = self.calls.pop(0)
        return ChatReply(content=[{'type': 'tool_use', 'id': f't{len(self.seen)}', 'name': name, 'input': arguments}],
                         stop_reason='tool_use', input_tokens=self.input_tokens, output_tokens=20,
                         cache_read_tokens=50)


class FakeFetcher:
    def __init__(self):
        self.data_hosts, self.post_endpoints, self.closed = set(), set(), False
        self.requests = self.renders = 0

    def fetch(self, url):
        return {'url': url, 'status': 200, 'content_type': 'text/html',
                'text': '<h2>Reports</h2><a href="/a.pdf">Annual report</a>'}

    def close(self):
        self.closed = True


class FakeJudge:
    def relevance(self, goal, page, candidates, neighbours=()):
        return [0.95 for _ in candidates]


def fake_runner(code, fetcher):
    docs = [{'url': 'https://site.example/a.pdf', 'name': 'A', 'context': 'x'}] if code == GOOD else []
    return {'documents': docs, 'error': None, 'fetch_failures': 0, 'requests': 1, 'renders': 0, 'seconds': 0.1}


def explorer(llm, **kwargs):
    return Explorer('https://site.example/', 'Find the annual reports', llm=llm, judge=FakeJudge(),
                    fetcher=FakeFetcher(), runner=fake_runner, **kwargs)


def test_the_agent_explores_runs_a_draft_and_submits_it():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('run_program', {'code': GOOD}),
                       ('submit_program', {'code': GOOD, 'notes': 'reports are linked from the home page'})])
    ex = explorer(llm)
    result = ex.explore()
    assert result['submitted'] and result['code'] == GOOD and result['turns'] == 3
    assert result['generated_kept'] == 1 and result['model'] == 'scripted-coder'
    assert ex.fetcher.closed
    # what the agent saw: the page's links with their heading, then the judge's verdicts on its draft
    tool_results = [b['content'] for m in llm.seen[-1] if m['role'] == 'user'
                    for b in m['content'] if b.get('type') == 'tool_result']
    assert 'Annual report | https://site.example/a.pdf | Reports' in tool_results[0]
    assert "verdicts {'accepted': 1" in tool_results[1]
    assert result['usage'] == {'input': 300, 'output': 60, 'cache_read': 150, 'cache_write': 0}


def test_without_a_submission_the_best_draft_is_kept():
    llm = ScriptedLLM([('run_program', {'code': WEAK}), ('run_program', {'code': GOOD}), ('run_program', {'code': WEAK})])
    result = explorer(llm, max_turns=3).explore()
    assert not result['submitted'] and result['code'] == GOOD and 'best draft (1 accepted)' in result['notes']


def test_the_input_token_budget_stops_the_agent():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'text'})] * 10, input_tokens=600)
    result = explorer(llm, max_input_tokens=1000).explore()
    assert result['turns'] == 2 and len(llm.seen) == 2 and result['code'] == ''


def test_a_refused_page_is_reported_to_the_agent_not_raised():
    from docseek.codegen.sandbox import FetchRefused

    class Refusing(FakeFetcher):
        def fetch(self, url):
            raise FetchRefused('off-site: other.example is not site.example')
    llm = ScriptedLLM([('fetch_page', {'url': 'https://other.example/', 'view': 'html'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    ex = Explorer('https://site.example/', 'goal', llm=llm, judge=FakeJudge(), fetcher=Refusing(), runner=fake_runner)
    ex.explore()
    assert llm.seen[1][-1]['content'][0]['content'] == 'refused: off-site: other.example is not site.example'


def test_one_cache_breakpoint_moves_to_the_newest_message():
    messages = [{'role': 'user', 'content': 'start'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'x', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'r'}]}]
    _move_cache_breakpoint(messages[:1])
    _move_cache_breakpoint(messages)
    marked = [b for m in messages for b in m['content'] if isinstance(b, dict) and 'cache_control' in b]
    assert marked == [messages[2]['content'][0]]
