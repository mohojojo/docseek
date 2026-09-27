from __future__ import annotations

import copy
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from docseek.llm import AnthropicLLM




# ---------------------------------------------------------------------------
# Helpers for _visit_page dispatch tests
# ---------------------------------------------------------------------------

def _make_tool_use_block(tool_name: str, tool_input: dict, block_id: str = 'tu_1'):
    """Return a minimal ToolUseBlock-like MagicMock for a given tool call."""
    block = MagicMock()
    block.type = 'tool_use'
    block.name = tool_name
    block.input = tool_input
    block.id = block_id
    return block


def _make_claude_response(tool_block):
    """Return a minimal Claude messages.create response containing one tool call."""
    resp = MagicMock()
    resp.stop_reason = 'tool_use'
    resp.content = [tool_block]
    resp.usage = MagicMock(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return resp


def _make_done_response():
    """Return a Claude response with a 'done' tool call to exit the loop."""
    done_block = _make_tool_use_block('done', {'reason': 'finished'}, block_id='tu_done')
    return _make_claude_response(done_block)


@contextmanager
def _patch_visit_page_scraper(page_mock, capture_screenshot_return):
    """Patch all scraper imports used by _visit_page, yielding a controlled page mock."""
    @contextmanager
    def _fake_open_page(url, **kwargs):
        yield page_mock

    fake_extract = MagicMock(return_value=('', {}, []))

    with (
        patch('docseek.scraper.open_page', side_effect=_fake_open_page),
        patch('docseek.scraper._try_dismiss_form_disclaimer'),
        patch('docseek.scraper._try_accept_cookies'),
        patch('docseek.scraper.apply_pre_interactions'),
        patch('docseek.scraper.extract_tree_from_page', fake_extract),
        patch('docseek.scraper.scroll_page_to_load', return_value=(None, 0, '')),
        patch('docseek.scraper.search_in_page', return_value=(None, '')),
        patch('docseek.scraper._COLLECT_LINKS_JS', ''),
        patch('docseek.scraper.capture_screenshot', create=True, return_value=capture_screenshot_return),
    ):
        yield


# ---------------------------------------------------------------------------
# _visit_page dispatch tests: take_screenshot
# ---------------------------------------------------------------------------

def test_visit_page_take_screenshot_happy_path():
    """Happy path: capture_screenshot returns b64 string → history content is an image list."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.viewport_size = {'width': 800, 'height': 600}
    page_mock.evaluate.return_value = []

    screenshot_block = _make_tool_use_block('take_screenshot', {'reason': 'inspect'}, block_id='tu_ss')
    screenshot_response = _make_claude_response(screenshot_block)
    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.side_effect = [screenshot_response, done_response]

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return='abc123'):
        _visit_page(
            'https://example.com/',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=None,
            visited_snapshot=frozenset(),
        )

    # history is a mutable list - search the final state for the screenshot tool_result by id
    all_calls = client.messages.create.call_args_list
    all_messages = all_calls[0][1]['messages']  # same object as all subsequent calls
    screenshot_tr_block = None
    for msg in all_messages:
        if msg['role'] != 'user' or not isinstance(msg.get('content'), list):
            continue
        for blk in msg['content']:
            if blk.get('type') == 'tool_result' and blk.get('tool_use_id') == 'tu_ss':
                screenshot_tr_block = blk
                break
    assert screenshot_tr_block is not None, 'screenshot tool_result not found in history'
    # content must be the image list, not a plain string
    assert isinstance(screenshot_tr_block['content'], list)
    image_block = screenshot_tr_block['content'][0]
    assert image_block['type'] == 'image'
    assert image_block['source']['data'] == 'abc123'
    assert image_block['source']['media_type'] == 'image/png'


def test_visit_page_take_screenshot_failure_path():
    """Failure path: capture_screenshot returns None → history content is a 'Screenshot failed' string."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.viewport_size = {'width': 800, 'height': 600}
    page_mock.evaluate.return_value = []

    screenshot_block = _make_tool_use_block('take_screenshot', {'reason': 'inspect'}, block_id='tu_ss')
    screenshot_response = _make_claude_response(screenshot_block)
    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.side_effect = [screenshot_response, done_response]

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return=None):
        _visit_page(
            'https://example.com/',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=None,
            visited_snapshot=frozenset(),
        )

    # history is a mutable list - search the final state for the screenshot tool_result by id
    all_calls = client.messages.create.call_args_list
    all_messages = all_calls[0][1]['messages']
    screenshot_tr_block = None
    for msg in all_messages:
        if msg['role'] != 'user' or not isinstance(msg.get('content'), list):
            continue
        for blk in msg['content']:
            if blk.get('type') == 'tool_result' and blk.get('tool_use_id') == 'tu_ss':
                screenshot_tr_block = blk
                break
    assert screenshot_tr_block is not None, 'screenshot tool_result not found in history'
    # content must be a plain string containing 'Screenshot failed'
    assert isinstance(screenshot_tr_block['content'], str)
    assert 'Screenshot failed' in screenshot_tr_block['content']


def test_visit_page_take_screenshot_emits_on_event():
    """on_event receives {'type': 'agent_screenshot', 'url': url} when screenshot is taken."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.viewport_size = {'width': 800, 'height': 600}
    page_mock.evaluate.return_value = []

    screenshot_block = _make_tool_use_block('take_screenshot', {'reason': 'inspect'}, block_id='tu_ss')
    screenshot_response = _make_claude_response(screenshot_block)
    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.side_effect = [screenshot_response, done_response]

    emitted_events = []

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return='abc123'):
        _visit_page(
            'https://example.com/page',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=emitted_events.append,
            visited_snapshot=frozenset(),
        )

    screenshot_events = [e for e in emitted_events if e.get('type') == 'agent_screenshot']
    assert len(screenshot_events) == 1
    assert screenshot_events[0] == {'type': 'agent_screenshot', 'url': 'https://example.com/page'}


# ---------------------------------------------------------------------------
# _visit_page: include_screenshot_on_load tests (Unit 4)
# ---------------------------------------------------------------------------

def test_visit_page_include_screenshot_on_load_success():
    """include_screenshot_on_load=True, capture succeeds: history[0]['content'] is a list with text+image."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.evaluate.return_value = []

    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.return_value = done_response

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return='img_b64'):
        _visit_page(
            'https://example.com/',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=None,
            visited_snapshot=frozenset(),
            include_screenshot_on_load=True,
        )

    all_messages = client.messages.create.call_args_list[0][1]['messages']
    initial_content = all_messages[0]['content']
    assert isinstance(initial_content, list), 'Expected a list when screenshot succeeds'
    assert len(initial_content) == 2
    text_block = initial_content[0]
    image_block = initial_content[1]
    assert text_block['type'] == 'text'
    assert isinstance(text_block['text'], str)
    assert image_block['type'] == 'image'
    assert image_block['source']['type'] == 'base64'
    assert image_block['source']['media_type'] == 'image/png'
    assert image_block['source']['data'] == 'img_b64'


def test_visit_page_include_screenshot_on_load_capture_fails():
    """include_screenshot_on_load=True, capture returns None: no image block is sent."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.evaluate.return_value = []

    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.return_value = done_response

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return=None):
        _visit_page(
            'https://example.com/',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=None,
            visited_snapshot=frozenset(),
            include_screenshot_on_load=True,
        )

    all_messages = client.messages.create.call_args_list[0][1]['messages']
    initial_content = all_messages[0]['content']
    # Content is a block list so the page context can carry a cache breakpoint; no image is attached.
    assert all(block.get('type') != 'image' for block in initial_content)
    assert any(block.get('type') == 'text' for block in initial_content)


def test_visit_page_include_screenshot_on_load_false():
    """include_screenshot_on_load=False (default): text only, no image block."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.evaluate.return_value = []

    done_response = _make_done_response()

    client = MagicMock()
    client.messages.create.return_value = done_response

    with _patch_visit_page_scraper(page_mock, capture_screenshot_return='img_b64'):
        _visit_page(
            'https://example.com/',
            depth=0,
            llm=AnthropicLLM('claude-test', client=client),
            goal='Find documents',
            system_blocks=[],
            seed_host='example.com',
            same_domain_only=True,
            js_wait_ms=100,
            click_wait_ms=100,
            max_tool_steps=10,
            crawl_plan=None,
            min_url_score=0.0,
            memory_snapshot={},
            open_kwargs={},
            queue_size_hint=0,
            pre_interactions=[],
            on_event=None,
            visited_snapshot=frozenset(),
            include_screenshot_on_load=False,
        )

    all_messages = client.messages.create.call_args_list[0][1]['messages']
    initial_content = all_messages[0]['content']
    assert all(block.get('type') != 'image' for block in initial_content)
    assert any(block.get('type') == 'text' for block in initial_content)


# ---------------------------------------------------------------------------
# Prompt cost: cache breakpoints and not re-sending an unchanged page
# ---------------------------------------------------------------------------

def _blocks(content):
    return [content] if isinstance(content, str) else content


def _cache_marked(message):
    return [b for b in _blocks(message['content'])
            if isinstance(b, dict) and 'cache_control' in b]


def _run_two_steps(first_tool, first_args):
    """Drive _visit_page through one tool call and then done(), returning the recorded requests."""
    from docseek.agent import _visit_page

    page_mock = MagicMock()
    page_mock.evaluate.return_value = []
    page_mock.locator.return_value.count.return_value = 1

    client = MagicMock()
    responses = [
        _make_claude_response(_make_tool_use_block(first_tool, first_args)),
        _make_done_response(),
    ]
    sent: list[list] = []

    def _create(**kwargs):
        # _visit_page passes its history list by reference and keeps appending to it, so snapshot
        # each request as it is made.
        sent.append(copy.deepcopy(kwargs['messages']))
        return responses[len(sent) - 1]

    client.messages.create.side_effect = _create
    with _patch_visit_page_scraper(page_mock, capture_screenshot_return=None):
        _visit_page(
            'https://example.com/', depth=0, llm=AnthropicLLM('claude-test', client=client), goal='Find documents',
            system_blocks=[], seed_host='example.com', same_domain_only=True, js_wait_ms=10,
            click_wait_ms=10, max_tool_steps=10, crawl_plan=None, min_url_score=0.0, memory_snapshot={},
            open_kwargs={}, queue_size_hint=0, pre_interactions=[], on_event=None,
            visited_snapshot=frozenset(),
        )
    return sent


def test_page_context_carries_a_cache_breakpoint():
    """The page listing is the biggest part of every request and never changes while the page is open."""
    messages = _run_two_steps('record_download',
                              {'url': 'https://example.com/a.pdf', 'name': 'A', 'reason': 'match'})
    assert _cache_marked(messages[0][0]), 'opening page context should be cacheable'


def test_at_most_two_breakpoints_are_live_at_once():
    """Anthropic allows four; the system prompt and tool definitions already use two."""
    messages = _run_two_steps('record_download',
                              {'url': 'https://example.com/a.pdf', 'name': 'A', 'reason': 'match'})
    last_request = messages[-1]
    marked = [m for m in last_request if m['role'] == 'user' and _cache_marked(m)]
    assert len(marked) <= 2


def test_an_unchanged_page_is_not_re_sent():
    """record_download does not re-extract, so the element listing is already in the history."""
    messages = _run_two_steps('record_download',
                              {'url': 'https://example.com/a.pdf', 'name': 'A', 'reason': 'match'})
    follow_up = [b for b in _blocks(messages[-1][-1]['content'])
                 if isinstance(b, dict) and b.get('type') == 'text'][-1]['text']
    assert 'PAGE UNCHANGED' in follow_up
    assert 'PAGE ELEMENTS:' not in follow_up


def test_a_click_that_reveals_elements_re_sends_the_listing():
    """When a click reveals new elements, the model must see them."""
    import copy as _copy

    from docseek.agent import _visit_page
    from docseek.models import FullElement

    revealed = {'9': FullElement(ml_id='9', role='link', name='Havi jelentés 2026. március',
                                 html_tag='a', attributes={}, url='https://example.com/r.pdf')}
    page_mock = MagicMock()
    page_mock.evaluate.return_value = []
    page_mock.locator.return_value.count.return_value = 1

    client = MagicMock()
    responses = [_make_claude_response(_make_tool_use_block('click', {'ml_id': '3', 'reason': 'reveal'})),
                 _make_done_response()]
    sent: list[list] = []

    def _create(**kwargs):
        sent.append(_copy.deepcopy(kwargs['messages']))
        return responses[len(sent) - 1]

    client.messages.create.side_effect = _create
    extract = MagicMock(side_effect=[('', {}, []), ('', revealed, [])])
    with _patch_visit_page_scraper(page_mock, capture_screenshot_return=None), \
            patch('docseek.scraper.extract_tree_from_page', extract):
        _visit_page(
            'https://example.com/', depth=0, llm=AnthropicLLM('claude-test', client=client), goal='Find documents',
            system_blocks=[], seed_host='example.com', same_domain_only=True, js_wait_ms=10,
            click_wait_ms=10, max_tool_steps=10, crawl_plan=None, min_url_score=0.0, memory_snapshot={},
            open_kwargs={}, queue_size_hint=0, pre_interactions=[], on_event=None,
            visited_snapshot=frozenset(),
        )

    follow_up = [b for b in _blocks(sent[-1][-1]['content'])
                 if isinstance(b, dict) and b.get('type') == 'text'][-1]['text']
    assert 'PAGE ELEMENTS:' in follow_up
    assert 'Havi jelentés' in follow_up


def test_a_click_that_changes_nothing_is_not_re_sent():
    """Re-extracting an identical page and re-sending it would pay twice for the same listing."""
    messages = _run_two_steps('click', {'ml_id': '3', 'reason': 'reveal documents'})
    follow_up = [b for b in _blocks(messages[-1][-1]['content'])
                 if isinstance(b, dict) and b.get('type') == 'text'][-1]['text']
    assert 'PAGE UNCHANGED' in follow_up
