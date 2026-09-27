"""A model-independent LLM client: one question in, one JSON object out.

Two adapters sit behind it:
  anthropic          the Anthropic Messages API (Claude models)
  openai-compatible  any /chat/completions endpoint: OpenAI, Azure OpenAI, OpenRouter, Gemini's OpenAI endpoint,
                     Ollama, vLLM, LM Studio

Two operations:
  complete_json(system, user)                -> one JSON object (judges, planning, pattern learning)
  chat(system, messages, tools, force_tool)  -> ChatReply, for tool-using agents

Messages use one canonical block format - the Anthropic Messages shape, which the Anthropic adapter passes
through unchanged (so prompt-cache breakpoints keep working) and the OpenAI-compatible adapter translates:
  {'type': 'text', 'text': ..., 'cache_control': {...}?}
  {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': ...}}
  {'type': 'tool_use', 'id': ..., 'name': ..., 'input': {...}}                 (assistant)
  {'type': 'tool_result', 'tool_use_id': ..., 'content': str | [text/image]}  (user)
Tools use the Anthropic schema: {'name', 'description', 'input_schema'}.

Configured from the environment unless given explicitly:
  LLM_PROVIDER   anthropic | openai-compatible     (default: anthropic when ANTHROPIC_API_KEY is set)
  LLM_MODEL      the model id                      (default for anthropic: claude-haiku-4-5)
  LLM_BASE_URL   base URL for openai-compatible    (default: https://api.openai.com/v1)
  LLM_API_KEY    the key (falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY; a local server may need none)
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Protocol

import httpx

DEFAULT_ANTHROPIC_MODEL = 'claude-haiku-4-5'
DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'
MAX_TOKENS = 4096


_BILLING_WORDS = ('credit balance', 'billing', 'insufficient_quota', 'quota exceeded', 'exceeded your current quota')


def _is_billing_error(text: str) -> bool:
    """Out of credit or quota: permanent until someone pays, so retrying only wastes time."""
    return any(word in (text or '').lower() for word in _BILLING_WORDS)


class LLMUnavailable(Exception):
    """The provider refused the request for good (no key, bad key, no credit): retrying cannot help."""


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += input_tokens or 0
            self.output_tokens += output_tokens or 0


@dataclass
class ChatReply:
    content: list[dict]                 # the assistant's blocks, canonical format: append them to the history
    stop_reason: str                    # 'tool_use' | 'end_turn' | 'max_tokens'
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def tool_call(self) -> dict | None:
        """The first tool call: {'type': 'tool_use', 'id', 'name', 'input'}, or None."""
        return next((b for b in self.content if b.get('type') == 'tool_use'), None)


class LLMClient(Protocol):
    provider: str
    model: str
    usage: Usage

    def complete_json(self, system: str, user: str) -> dict:
        """The model's answer parsed as one JSON object. Raises LLMUnavailable for a permanent refusal and
        ValueError when the answer is not JSON."""

    def chat(self, system: str | list[dict], messages: list[dict], tools: list[dict], *,
             force_tool: bool = True, max_tokens: int = 8192) -> ChatReply:
        """One turn of a tool-using conversation. force_tool: the model must call exactly one tool."""


def parse_json_object(text: str) -> dict:
    """The first JSON object in a model's answer, tolerating code fences and surrounding prose."""
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start < 0 or end <= start:
            raise ValueError(f'no JSON object in the answer: {text[:200]!r}') from None
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError('the answer is JSON but not an object')
    return value


class AnthropicLLM:
    provider = 'anthropic'

    def __init__(self, model: str = DEFAULT_ANTHROPIC_MODEL, api_key: str | None = None, timeout: float = 120.0,
                 client=None):
        import anthropic
        self._anthropic = anthropic
        self.model = model
        self.usage = Usage()
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def _create(self, **kwargs):
        try:
            response = self._client.messages.create(model=self.model, **kwargs)
        except (self._anthropic.AuthenticationError, self._anthropic.PermissionDeniedError) as exc:
            raise LLMUnavailable(f'anthropic: {exc}') from exc
        except self._anthropic.APIStatusError as exc:
            if _is_billing_error(str(exc)):
                raise LLMUnavailable(f'anthropic: {exc}') from exc
            raise
        self.usage.add(response.usage.input_tokens, response.usage.output_tokens)
        return response

    def complete_json(self, system: str, user: str) -> dict:
        response = self._create(max_tokens=MAX_TOKENS, system=system, messages=[{'role': 'user', 'content': user}])
        return parse_json_object(''.join(b.text for b in response.content if isinstance(getattr(b, 'text', None), str)))

    def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192) -> ChatReply:
        response = self._create(
            system=system, messages=messages, tools=tools, max_tokens=max_tokens,
            tool_choice={'type': 'any', 'disable_parallel_tool_use': True} if force_tool else {'type': 'auto'})
        usage = response.usage
        return ChatReply(
            content=[_anthropic_block(b) for b in response.content], stop_reason=response.stop_reason,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
            cache_write_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0)


def _anthropic_block(block) -> dict:
    """An SDK content block in the canonical format; anything else (thinking) passes through as the SDK dumps it."""
    if block.type == 'tool_use':
        return {'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.input}
    if block.type == 'text':
        return {'type': 'text', 'text': block.text}
    return block.model_dump(exclude_none=True)


class OpenAICompatibleLLM:
    provider = 'openai-compatible'

    def __init__(self, model: str, base_url: str = DEFAULT_OPENAI_BASE_URL, api_key: str | None = None,
                 timeout: float = 60.0):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.usage = Usage()
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)
        self._json_mode = True          # response_format json_object; dropped for servers that reject it
        self._dropped: set[str] = set()
        self._renamed: dict[str, str] = {}

    def _adapt(self, body: dict) -> dict:
        body = {k: v for k, v in body.items() if k not in self._dropped}
        for old, new in self._renamed.items():
            if old in body:
                body[new] = body.pop(old)
        return body

    def _post(self, body: dict) -> dict:
        """POST /chat/completions, adapting to what this server accepts: an optional parameter it rejects is
        dropped (response_format, parallel_tool_calls, tool_choice, temperature), and max_tokens becomes
        max_completion_tokens where the server asks for it (OpenAI's reasoning models). Remembered per client."""
        body = self._adapt(body)
        for _ in range(len(_OPTIONAL_PARAMS) + 2):
            resp = self._client.post(f'{self.base_url}/chat/completions', json=body)
            if resp.status_code != 400:
                break
            if 'max_tokens' in body and 'max_completion_tokens' in resp.text:
                self._renamed['max_tokens'] = 'max_completion_tokens'
            else:
                unsupported = next((p for p in _OPTIONAL_PARAMS if p in body and p in resp.text), None)
                if not unsupported:
                    break
                self._dropped.add(unsupported)
                if unsupported == 'response_format':
                    self._json_mode = False
            body = self._adapt(body)
        if resp.status_code in (401, 402, 403, 404) or (resp.status_code == 429 and _is_billing_error(resp.text)):
            raise LLMUnavailable(f'{self.provider} {resp.status_code}: {resp.text[:200]}')
        resp.raise_for_status()
        data = resp.json()
        usage = data.get('usage') or {}
        self.usage.add(usage.get('prompt_tokens', 0), usage.get('completion_tokens', 0))
        return data

    def complete_json(self, system: str, user: str) -> dict:
        body = {'model': self.model, 'temperature': 0,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if self._json_mode:
            body['response_format'] = {'type': 'json_object'}
        data = self._post(body)
        return parse_json_object(data['choices'][0]['message']['content'] or '')

    def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192) -> ChatReply:
        body = {'model': self.model, 'max_tokens': max_tokens, 'messages': to_openai_messages(system, messages),
                'tools': [{'type': 'function', 'function': {'name': t['name'], 'description': t.get('description', ''),
                                                            'parameters': t['input_schema']}} for t in tools]}
        if force_tool:
            body.update(tool_choice='required', parallel_tool_calls=False)
        data = self._post(body)
        choice = data['choices'][0]
        message = choice['message']
        content = [{'type': 'text', 'text': message['content']}] if message.get('content') else []
        for call in message.get('tool_calls') or []:
            try:
                arguments = json.loads(call['function'].get('arguments') or '{}')
            except json.JSONDecodeError:
                arguments = {}
            content.append({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'], 'input': arguments})
        usage = data.get('usage') or {}
        finish = choice.get('finish_reason')
        return ChatReply(
            content=content,
            stop_reason='max_tokens' if finish == 'length' else 'tool_use' if message.get('tool_calls') else 'end_turn',
            input_tokens=usage.get('prompt_tokens', 0), output_tokens=usage.get('completion_tokens', 0),
            cache_read_tokens=(usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0)


_OPTIONAL_PARAMS = ('response_format', 'parallel_tool_calls', 'tool_choice', 'temperature')


def _image_url(block: dict) -> dict:
    source = block['source']
    return {'type': 'image_url', 'image_url': {'url': f"data:{source['media_type']};base64,{source['data']}"}}


def to_openai_messages(system: str | list[dict], messages: list[dict]) -> list[dict]:
    """Canonical messages in the Chat Completions shape. A tool result's screenshot goes into the user message
    that follows the tool messages, since a `tool` message carries text only."""
    system_text = system if isinstance(system, str) else '\n\n'.join(
        b['text'] for b in system if b.get('type') == 'text')
    out: list[dict] = [{'role': 'system', 'content': system_text}]
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            out.append({'role': message['role'], 'content': content})
            continue
        if message['role'] == 'assistant':
            text = ''.join(b['text'] for b in content if b.get('type') == 'text')
            calls = [{'id': b['id'], 'type': 'function',
                      'function': {'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)}}
                     for b in content if b.get('type') == 'tool_use']
            out.append({'role': 'assistant', 'content': text or None, **({'tool_calls': calls} if calls else {})})
            continue
        parts, images = [], []
        for block in content:
            kind = block.get('type')
            if kind == 'tool_result':
                result = block['content']
                if isinstance(result, str):
                    text = result
                else:
                    text = ''.join(x.get('text', '') for x in result if x.get('type') == 'text')
                    images += [x for x in result if x.get('type') == 'image']
                out.append({'role': 'tool', 'tool_call_id': block['tool_use_id'], 'content': text or '(see the image)'})
            elif kind == 'text':
                parts.append({'type': 'text', 'text': block['text']})
            elif kind == 'image':
                parts.append(_image_url(block))
        parts = [_image_url(image) for image in images] + parts
        if parts:
            out.append({'role': 'user', 'content': parts})
    return out


def make_llm(provider: str | None = None, model: str | None = None, base_url: str | None = None,
             api_key: str | None = None) -> LLMClient | None:
    """The configured LLM client, or None when nothing is configured."""
    provider = provider or os.environ.get('LLM_PROVIDER') or ('anthropic' if os.environ.get('ANTHROPIC_API_KEY') else None)
    model = model or os.environ.get('LLM_MODEL')
    if provider == 'anthropic':
        key = api_key or os.environ.get('LLM_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
        return AnthropicLLM(model or DEFAULT_ANTHROPIC_MODEL, api_key=key) if key else None
    if provider == 'openai-compatible':
        if not model:
            raise ValueError('LLM_MODEL is required for the openai-compatible provider')
        key = api_key or os.environ.get('LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
        return OpenAICompatibleLLM(model, base_url or os.environ.get('LLM_BASE_URL') or DEFAULT_OPENAI_BASE_URL, key)
    if provider:
        raise ValueError(f'unknown LLM_PROVIDER {provider!r}: use anthropic or openai-compatible')
    return None
