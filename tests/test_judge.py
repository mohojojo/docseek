"""The Relevance judge seam (crawler.judge) and the model-independent LLM client (crawler.llm)."""
from __future__ import annotations

import json

import httpx
import pytest

from docseek.judge import (
    ACCEPTED_AT, LEVELS, REJECTED_BELOW, FallbackJudge, JudgeUnavailable, LLMJudge, make_judge, verdict_for,
)
from docseek.llm import LLMUnavailable, OpenAICompatibleLLM, Usage, make_llm, parse_json_object
from docseek.profile import load_profile


class FakeLLM:
    """Answers from a function of the user message; counts calls like a real client."""

    provider, model = 'fake', 'fake-model'

    def __init__(self, answer):
        self.answer = answer
        self.usage = Usage()
        self.prompts: list[str] = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        self.usage.add(10, 2)
        result = self.answer(user)
        if isinstance(result, Exception):
            raise result
        return result


# --- the LLM client ----------------------------------------------------------------------------------
@pytest.mark.parametrize('text', ['{"L1": "clearly_yes"}', '```json\n{"L1": "clearly_yes"}\n```',
                                  'Sure! Here it is: {"L1": "clearly_yes"} Hope that helps.'])
def test_parse_json_object_tolerates_fences_and_prose(text):
    assert parse_json_object(text) == {'L1': 'clearly_yes'}


def test_parse_json_object_refuses_an_answer_without_one():
    with pytest.raises(ValueError):
        parse_json_object('I cannot judge these links.')


def _openai(handler) -> OpenAICompatibleLLM:
    llm = OpenAICompatibleLLM('local-model', base_url='http://llm.local/v1', api_key='k')
    llm._client = httpx.Client(transport=httpx.MockTransport(handler), headers={'Authorization': 'Bearer k'})
    return llm


def _completion(content: str, prompt_tokens=7, completion_tokens=3) -> httpx.Response:
    return httpx.Response(200, json={'choices': [{'message': {'content': content}}],
                                     'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens}})


def test_the_openai_compatible_adapter_speaks_chat_completions():
    seen = {}

    def handler(request):
        seen['url'], seen['body'] = str(request.url), json.loads(request.content)
        seen['auth'] = request.headers.get('authorization')
        return _completion('{"answer": "probably_no"}')
    llm = _openai(handler)
    assert llm.complete_json('system', 'user') == {'answer': 'probably_no'}
    assert seen['url'] == 'http://llm.local/v1/chat/completions' and seen['auth'] == 'Bearer k'
    assert seen['body']['model'] == 'local-model' and seen['body']['response_format'] == {'type': 'json_object'}
    assert [m['role'] for m in seen['body']['messages']] == ['system', 'user']
    assert (llm.usage.requests, llm.usage.input_tokens, llm.usage.output_tokens) == (1, 7, 3)


def test_a_server_without_json_mode_is_asked_again_without_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if 'response_format' in body:
            return httpx.Response(400, text='unsupported parameter: response_format')
        return _completion('{"answer": "unsure"}')
    llm = _openai(handler)
    assert llm.complete_json('s', 'u') == {'answer': 'unsure'}
    assert llm.complete_json('s', 'u') == {'answer': 'unsure'}      # remembered: no second 400
    assert ['response_format' in b for b in bodies] == [True, False, False]


@pytest.mark.parametrize('status', [401, 402, 403])
def test_an_auth_or_billing_refusal_is_permanent(status):
    with pytest.raises(LLMUnavailable):
        _openai(lambda r: httpx.Response(status, text='no')).complete_json('s', 'u')


def test_make_llm_reads_the_environment(monkeypatch):
    for var in ('LLM_PROVIDER', 'LLM_MODEL', 'LLM_BASE_URL', 'LLM_API_KEY', 'ANTHROPIC_API_KEY', 'OPENAI_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    assert make_llm() is None                                   # nothing configured
    monkeypatch.setenv('LLM_PROVIDER', 'openai-compatible')
    monkeypatch.setenv('LLM_MODEL', 'llama3.1')
    monkeypatch.setenv('LLM_BASE_URL', 'http://localhost:11434/v1')
    llm = make_llm()
    assert (llm.provider, llm.model, llm.base_url) == ('openai-compatible', 'llama3.1', 'http://localhost:11434/v1')
    monkeypatch.setenv('LLM_PROVIDER', 'nonsense')
    with pytest.raises(ValueError):
        make_llm()


# --- the LLM judge -------------------------------------------------------------------------------------
def test_every_level_lands_in_the_verdict_band_it_names():
    assert verdict_for(LEVELS['clearly_yes']) == verdict_for(LEVELS['probably_yes']) == 'accepted'
    assert verdict_for(LEVELS['unsure']) == 'unsure'
    assert verdict_for(LEVELS['probably_no']) == verdict_for(LEVELS['clearly_no']) == 'rejected'
    assert LEVELS['probably_yes'] >= ACCEPTED_AT > LEVELS['unsure'] >= REJECTED_BELOW > LEVELS['probably_no']


def test_the_llm_judge_scores_links_with_the_profile_wording():
    llm = FakeLLM(lambda user: {'L1': 'clearly_yes', 'L2': 'probably_no', 'L3': 'banana'})
    judge = LLMJudge(llm, load_profile('fund-reports'))
    links = [{'url': f'https://site.example/{i}.pdf', 'name': n} for i, n in enumerate(['Havi jelentés 2026-08', 'KID', 'x'])]
    assert judge.relevance('goal', 'page', links) == [LEVELS['clearly_yes'], LEVELS['probably_no'], None]
    assert load_profile('fund-reports').relevance['false'] in llm.prompts[0]


def test_neighbours_are_context_and_not_judged():
    llm = FakeLLM(lambda user: {'L1': 'probably_yes'})
    judge = LLMJudge(llm)
    judge.relevance('goal', 'page', [{'url': 'https://site.example/new.pdf', 'name': 'Letöltés'}],
                    neighbours=[{'url': 'https://site.example/aug.pdf', 'name': '2026. augusztusi jelentés'}])
    assert 'N1' in llm.prompts[0] and '2026. augusztusi jelentés' in llm.prompts[0]


def test_candidates_are_asked_in_batches():
    llm = FakeLLM(lambda user: {f'L{i}': 'unsure' for i in range(1, 21)})
    scores = LLMJudge(llm).relevance('g', 'p', [{'url': f'https://site.example/{i}.pdf'} for i in range(45)])
    assert len(scores) == 45 and llm.usage.requests == 3


def test_page_kinds_outside_the_profile_fall_back_to_other():
    llm = FakeLLM(lambda user: {'L1': {'kind': 'document_listing', 'sure': 'sure'}, 'L2': {'kind': 'shop'}})
    kinds = LLMJudge(llm).page_kinds('g', [{'url': 'https://site.example/docs'}, {'url': 'https://site.example/shop'}])
    assert kinds == [('document_listing', 0.9), ('other', 0.0)]


def test_a_filter_value_the_page_does_not_offer_is_never_set():
    llm = FakeLLM(lambda user: {'jf0': '2026', 'jf1': '1999', 'jf2': 'keep'})
    filters = [{'id': f'jf{i}', 'label': 'Év', 'current': '', 'options': ['2026', '2025']} for i in range(3)]
    assert LLMJudge(llm).filter_values('g', {}, filters) == {'jf0': ('2026', LEVELS['probably_yes'])}


def test_the_breaker_opens_after_repeated_failures_and_on_a_refusal():
    flaky = LLMJudge(FakeLLM(lambda user: ValueError('not json')))
    for _ in range(3):
        assert flaky.hides_documents('g', {}) is None
    assert flaky.open and flaky.unavailable_reason == 'failures'
    refused = LLMJudge(FakeLLM(lambda user: LLMUnavailable('401')))
    refused.hides_documents('g', {})
    assert refused.open and refused.unavailable_reason == 'auth'


# --- Jev first, the LLM judge when Jev breaks ------------------------------------------------------------
class BreakingJudge:
    """A primary judge that goes unavailable during its first call."""

    name, model, requests, cost_usd = 'jev', 'jev-x', 1, 0.01

    def __init__(self):
        self.open, self.unavailable_reason = False, None

    def relevance(self, goal, page, candidates, neighbours=()):
        self.open, self.unavailable_reason = True, 'no_credits'
        return [None] * len(candidates)


def test_the_fallback_answers_the_call_during_which_jev_broke():
    secondary = LLMJudge(FakeLLM(lambda user: {'L1': 'clearly_yes'}))
    judge = FallbackJudge(BreakingJudge(), secondary)
    assert judge.relevance('g', 'p', [{'url': 'https://site.example/a.pdf'}]) == [LEVELS['clearly_yes']]
    assert not judge.open and judge.model == 'jev-x, then fake-model (no_credits)'


def test_without_a_fallback_the_judge_opens_with_jev():
    judge = FallbackJudge(BreakingJudge(), None)
    judge.relevance('g', 'p', [{'url': 'https://site.example/a.pdf'}])
    assert judge.open and judge.unavailable_reason == 'no_credits'


def test_make_judge_picks_by_configuration(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    llm = FakeLLM(lambda user: {})
    assert isinstance(make_judge(llm=llm), LLMJudge)                       # no Jev key: the LLM judge
    with pytest.raises(JudgeUnavailable, match='TYPESAFE_API_KEY'):
        make_judge('jev', llm=llm)
    monkeypatch.setenv('TYPESAFE_API_KEY', 'k')
    assert isinstance(make_judge(llm=llm), FallbackJudge)                  # Jev key: Jev, LLM behind it
    assert isinstance(make_judge('llm', llm=llm), LLMJudge)
    with pytest.raises(JudgeUnavailable):
        make_judge('oracle', llm=llm)


# --- Laya: Jev's questions on a self-hosted server ------------------------------------------------------
def test_make_judge_runs_laya_only_by_name_and_only_with_its_url(monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    monkeypatch.delenv('LAYA_URL', raising=False)
    llm = FakeLLM(lambda user: {})
    with pytest.raises(JudgeUnavailable, match='LAYA_URL'):
        make_judge('laya', llm=llm)
    monkeypatch.setenv('LAYA_URL', 'http://laya.example:8000')
    assert isinstance(make_judge(llm=llm), LLMJudge)                       # a URL alone does not make it the default
    judge = make_judge('laya', llm=llm)
    assert isinstance(judge, FallbackJudge) and judge.name == 'laya' and not judge.open


def test_laya_posts_to_its_own_server_without_the_typesafe_key(monkeypatch):
    from unittest.mock import MagicMock

    from docseek.jev import LAYA_MAX_LEN, LayaClient

    monkeypatch.setenv('TYPESAFE_API_KEY', 'typesafe-secret')
    monkeypatch.setenv('LAYA_URL', 'http://laya.example:8000/')
    monkeypatch.setenv('LAYA_MODEL', 'multilingual')
    monkeypatch.delenv('LAYA_API_KEY', raising=False)
    laya = LayaClient()
    laya._client = MagicMock()
    laya._client.post.return_value = MagicMock(status_code=200, json=lambda: {
        'answers': {'hidden': {'noul': 0.9}}, 'usage': {'input_tokens': 7, 'output_tokens': 0}})
    assert laya.hides_documents('g', {'text': 'page'}) == 0.9
    (url,), sent = laya._client.post.call_args
    assert url == 'http://laya.example:8000/v1/systemone' and sent['headers'] == {}
    assert sent['json']['model'] == 'multilingual' and sent['json']['max_len'] == LAYA_MAX_LEN
    assert laya.model == 'laya-multilingual' and laya.cost_usd == 0.0
