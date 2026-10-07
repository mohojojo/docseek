"""docseek.codegen.explorer: the agent loop that writes a discovery program, on any docseek.llm client."""
from __future__ import annotations

import json

from docseek.codegen.explorer import Explorer, _move_cache_breakpoint
from docseek.llm import ChatReply

GOOD = 'def discover(fetch, render):\n    return [{"url": "https://site.example/a.pdf", "name": "A", "context": "x"}]\n'
WEAK = 'def discover(fetch, render):\n    return []\n'
BROKEN = 'def discover(fetch, render):\n    raise ValueError("x")\n'


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
    return {'documents': docs, 'error': 'ValueError: x' if code == BROKEN else None, 'fetch_failures': 0,
            'requests': 1, 'renders': 0, 'seconds': 0.1}


def tool_results(llm):
    """Every tool result the model was shown, in order, as of its last call."""
    return [b['content'] for m in llm.seen[-1] if m['role'] == 'user' and isinstance(m['content'], list)
            for b in m['content'] if b.get('type') == 'tool_result']


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
    assert llm.seen[1][-1]['content'][0]['content'].startswith('refused: off-site: other.example is not site.example')


def test_one_cache_breakpoint_moves_to_the_newest_message():
    messages = [{'role': 'user', 'content': 'start'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'x', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'r'}]}]
    _move_cache_breakpoint(messages[:1])
    _move_cache_breakpoint(messages)
    marked = [b for m in messages for b in m['content'] if isinstance(b, dict) and 'cache_control' in b]
    assert marked == [messages[2]['content'][0]]


def test_several_tool_calls_in_one_turn_are_all_answered():
    class TwoAtOnce(ScriptedLLM):
        def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192):
            assert force_tool is False                      # the model may call several tools per turn
            self.seen.append(json.loads(json.dumps(messages)))
            if len(self.seen) == 1:
                content = [{'type': 'tool_use', 'id': 'a', 'name': 'fetch_page',
                            'input': {'url': 'https://site.example/', 'view': 'links'}},
                           {'type': 'tool_use', 'id': 'b', 'name': 'run_program', 'input': {'code': GOOD}}]
            else:
                content = [{'type': 'tool_use', 'id': 'c', 'name': 'submit_program', 'input': {'code': GOOD, 'notes': ''}}]
            return ChatReply(content=content, stop_reason='tool_use')
    llm = TwoAtOnce([])
    result = explorer(llm).explore()
    answered = [b['tool_use_id'] for b in llm.seen[1][-1]['content'] if b.get('type') == 'tool_result']
    assert answered == ['a', 'b'] and result['submitted'] and result['turns'] == 2


def test_a_failing_model_keeps_the_best_draft(monkeypatch):
    import docseek.codegen.explorer as explorer_module
    monkeypatch.setattr(explorer_module.time, 'sleep', lambda s: None)

    class Flaky(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if len(self.seen) >= 1:
                raise ConnectionError('ssl alert bad record mac')
            return super().chat(*args, **kwargs)
    result = explorer(Flaky([('run_program', {'code': GOOD})])).explore()
    assert result['code'] == GOOD and 'ConnectionError' in result['stopped'] and result['turns'] == 1


def test_submitted_code_that_never_ran_is_run_once_for_its_baseline():
    llm = ScriptedLLM([('submit_program', {'code': GOOD, 'notes': ''})])
    ex = explorer(llm)
    result = ex.explore()
    assert result['generated_kept'] == 1 and len(ex.runs) == 1


def test_a_program_that_only_met_refusals_is_not_verified():
    def blocked_runner(code, fetcher):
        return {'documents': [], 'error': None, 'fetch_failures': 3, 'requests': 3, 'renders': 0, 'seconds': 0.1}
    llm = ScriptedLLM([('submit_program', {'code': WEAK, 'notes': 'the site answered 403 to everything'})])
    ex = Explorer('https://site.example/', 'goal', llm=llm, judge=FakeJudge(), fetcher=FakeFetcher(), runner=blocked_runner)
    result = ex.explore()
    assert result['submitted'] and result['verified'] is False and result['generated_kept'] == 0


def test_a_program_that_found_documents_is_verified_and_the_log_shows_the_way():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    result = explorer(llm).explore()
    assert result['verified'] is True
    assert [e['tool'] for e in result['log'] if 'tool' in e] == ['fetch_page', 'submit_program']
    assert result['log'][0]['result'].startswith('200 text/html')


def test_every_tool_result_ends_with_the_budget_and_the_wrap_up_comes_at_three_quarters():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('run_program', {'code': GOOD}),
                       ('fetch_page', {'url': 'https://site.example/x', 'view': 'text'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    explorer(llm, max_turns=4).explore()
    shown = tool_results(llm)
    assert shown[0].endswith('[budget: turn 1 of 4, 0k of 3000k input tokens, 0 runs, no run yet]')
    assert shown[1].endswith('[budget: turn 2 of 4, 0k of 3000k input tokens, 1 runs, best run 1 accepted]')
    assert 'WRAP UP' in shown[2] and 'turn 3 of 4' in shown[2]


def test_the_token_budget_wraps_up_too_and_no_run_by_half_is_said():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('fetch_page', {'url': 'https://site.example/x', 'view': 'text'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})], input_tokens=450)
    explorer(llm, max_turns=40, max_input_tokens=1000).explore()     # 500 tokens a turn: half, then all
    shown = tool_results(llm)
    assert 'no program has run' in shown[0] and 'WRAP UP' not in shown[0]
    assert 'WRAP UP' in shown[1]


def test_a_submission_that_errors_is_handed_back():
    llm = ScriptedLLM([('submit_program', {'code': BROKEN, 'notes': ''}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    result = explorer(llm).explore()
    assert 'NOT submitted' in tool_results(llm)[0] and 'ValueError' in tool_results(llm)[0]
    assert result['submitted'] and result['code'] == GOOD and result['turns'] == 2


def test_a_repeated_tool_call_is_not_run_again():
    llm = ScriptedLLM([('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('fetch_page', {'url': 'https://site.example/', 'view': 'links'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    ex = explorer(llm)
    ex.explore()
    assert 'already called with the same arguments at turn 1' in tool_results(llm)[1]


def test_two_identical_runs_tell_the_agent_to_change_approach():
    llm = ScriptedLLM([('run_program', {'code': WEAK}), ('run_program', {'code': WEAK + '# v2\n'}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    explorer(llm).explore()
    shown = tool_results(llm)
    assert 'NO PROGRESS' not in shown[0] and 'NO PROGRESS' in shown[1]


def test_a_narrower_run_than_the_best_is_noted():
    llm = ScriptedLLM([('run_program', {'code': GOOD}), ('run_program', {'code': WEAK}),
                       ('submit_program', {'code': GOOD, 'notes': ''})])
    explorer(llm).explore()
    assert '0 accepted, down from 1' in tool_results(llm)[1]
