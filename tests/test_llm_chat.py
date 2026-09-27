"""Tool-using chat through crawler.llm: the OpenAI-compatible translation, and the agent on a non-Anthropic model."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx

from docseek.llm import ChatReply, OpenAICompatibleLLM, Usage, to_openai_messages
from tests.test_agent import _patch_visit_page_scraper

PNG = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'iVBOR'}}
TOOLS = [{'name': 'done', 'description': 'Finish.', 'input_schema': {'type': 'object', 'properties': {
    'reason': {'type': 'string'}}, 'required': ['reason']}, 'cache_control': {'type': 'ephemeral'}}]


def test_canonical_messages_become_chat_completions():
    history = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'PAGE ELEMENTS ...', 'cache_control': {'type': 'ephemeral'}}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Looking.'},
                                          {'type': 'tool_use', 'id': 'c1', 'name': 'take_screenshot', 'input': {'reason': 'see'}}]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c1', 'content': [PNG]},
                                     {'type': 'text', 'text': 'PAGE UNCHANGED'}]},
    ]
    out = to_openai_messages([{'type': 'text', 'text': 'You are a crawler.'}], history)
    assert out[0] == {'role': 'system', 'content': 'You are a crawler.'}
    assert out[1] == {'role': 'user', 'content': [{'type': 'text', 'text': 'PAGE ELEMENTS ...'}]}
    assert out[2]['tool_calls'][0]['function'] == {'name': 'take_screenshot', 'arguments': '{"reason": "see"}'}
    assert out[3] == {'role': 'tool', 'tool_call_id': 'c1', 'content': '(see the image)'}
    # a tool message carries text only: the screenshot rides in the user message that follows it
    assert out[4]['content'][0] == {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBOR'}}
    assert out[4]['content'][1] == {'type': 'text', 'text': 'PAGE UNCHANGED'}


def _llm(handler) -> OpenAICompatibleLLM:
    llm = OpenAICompatibleLLM('gpt-x', base_url='http://llm.local/v1', api_key='k')
    llm._client = httpx.Client(transport=httpx.MockTransport(handler))
    return llm


def _tool_call_reply(name='done', arguments='{"reason": "finished"}', finish='tool_calls'):
    return httpx.Response(200, json={'choices': [{'finish_reason': finish, 'message': {
        'content': None, 'tool_calls': [{'id': 'call_1', 'type': 'function',
                                         'function': {'name': name, 'arguments': arguments}}]}}],
        'usage': {'prompt_tokens': 120, 'completion_tokens': 9, 'prompt_tokens_details': {'cached_tokens': 100}}})


def test_chat_forces_one_tool_call_and_reads_it_back():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return _tool_call_reply()
    reply = _llm(handler).chat('system', [{'role': 'user', 'content': 'go'}], TOOLS)
    assert seen['tool_choice'] == 'required' and seen['parallel_tool_calls'] is False
    assert seen['tools'][0]['function']['name'] == 'done' and 'cache_control' not in json.dumps(seen['tools'])
    assert reply.tool_call == {'type': 'tool_use', 'id': 'call_1', 'name': 'done', 'input': {'reason': 'finished'}}
    assert (reply.stop_reason, reply.input_tokens, reply.cache_read_tokens) == ('tool_use', 120, 100)


def test_a_server_without_parallel_tool_calls_is_asked_again_without_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if 'parallel_tool_calls' in body:
            return httpx.Response(400, text='unknown field: parallel_tool_calls')
        return _tool_call_reply()
    assert _llm(handler).chat('s', [{'role': 'user', 'content': 'go'}], TOOLS).tool_call['name'] == 'done'
    assert ['parallel_tool_calls' in b for b in bodies] == [True, False]


def test_a_truncated_reply_is_max_tokens_and_bad_arguments_are_empty():
    reply = _llm(lambda r: _tool_call_reply(arguments='{"reason": "fini', finish='length')).chat('s', [], TOOLS)
    assert reply.stop_reason == 'max_tokens' and reply.tool_call['input'] == {}


class ScriptedLLM:
    """A non-Anthropic model that answers with a scripted list of tool calls."""

    provider, model = 'scripted', 'scripted-1'

    def __init__(self, calls):
        self.calls, self.usage, self.histories = list(calls), Usage(), []

    def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192):
        self.histories.append(json.loads(json.dumps(messages, default=str)))
        name, arguments = self.calls.pop(0)
        return ChatReply(content=[{'type': 'tool_use', 'id': f'id{len(self.histories)}', 'name': name, 'input': arguments}],
                         stop_reason='tool_use', input_tokens=10, output_tokens=2)

    def complete_json(self, system, user):
        return {}


def test_the_agent_completes_a_page_on_a_non_anthropic_model():
    from docseek.agent import _visit_page
    page = MagicMock()
    page.url = 'https://example.com/'
    llm = ScriptedLLM([('record_download', {'url': 'https://example.com/report.pdf', 'name': 'Annual report',
                                            'reason': 'the goal asks for it'}),
                       ('done', {'reason': 'finished'})])
    with _patch_visit_page_scraper(page, None):
        result = _visit_page(
            'https://example.com/', depth=0, llm=llm, goal='Find the annual report', system_blocks=[],
            seed_host='example.com', same_domain_only=True, js_wait_ms=0, click_wait_ms=0, max_tool_steps=5,
            crawl_plan=None, min_url_score=0.0, memory_snapshot={}, open_kwargs={}, queue_size_hint=0,
            pre_interactions=[], on_event=None, visited_snapshot=frozenset())
    assert [d.url for d in result.downloads] == ['https://example.com/report.pdf']
    assert [s.tool for s in result.steps] == ['record_download', 'done']
    # the second turn carries the first call and its result back, in the canonical format
    second = llm.histories[1]
    assert second[1]['content'][0]['name'] == 'record_download'
    assert second[2]['content'][0]['type'] == 'tool_result' and second[2]['content'][0]['tool_use_id'] == 'id1'


def test_a_reasoning_model_gets_max_completion_tokens_and_no_temperature():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if 'temperature' in body:
            return httpx.Response(400, text="Unsupported value: 'temperature' does not support 0")
        if 'max_tokens' in body:
            return httpx.Response(400, text="Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"ok": true}'}}], 'usage': {}})
    llm = _llm(handler)
    assert llm.complete_json('s', 'u') == {'ok': True}                     # temperature dropped
    llm.chat('s', [{'role': 'user', 'content': 'go'}], TOOLS)               # max_tokens renamed
    last = bodies[-1]
    assert 'temperature' not in last and 'max_tokens' not in last and last['max_completion_tokens'] == 8192
    bodies.clear()
    llm.complete_json('s', 'u')
    assert len(bodies) == 1                                                 # remembered: no 400s the second time


def test_an_empty_credit_balance_is_permanent_not_retried():
    import anthropic
    import pytest

    from docseek.llm import AnthropicLLM, LLMUnavailable

    class Client:
        class messages:
            @staticmethod
            def create(**kwargs):
                response = httpx.Response(400, request=httpx.Request('POST', 'https://api.anthropic.com/v1/messages'))
                raise anthropic.BadRequestError('Your credit balance is too low to access the Anthropic API.',
                                                response=response, body=None)
    with pytest.raises(LLMUnavailable):
        AnthropicLLM('claude-test', client=Client()).chat('s', [{'role': 'user', 'content': 'go'}], TOOLS)


def test_an_exhausted_openai_quota_is_permanent():
    import pytest

    from docseek.llm import LLMUnavailable
    llm = _llm(lambda r: httpx.Response(429, json={'error': {'code': 'insufficient_quota',
                                                             'message': 'You exceeded your current quota'}}))
    with pytest.raises(LLMUnavailable):
        llm.complete_json('s', 'u')
