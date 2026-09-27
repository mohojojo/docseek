"""Tests for POST /v1/search-sites and supporting functions."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from docseek.agent import _is_safe_url, search_sites
from docseek.models import SearchSiteResult
from docseek.server import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_item(url: str, title: str) -> MagicMock:
    item = MagicMock()
    item.type = 'web_search_result'
    item.url = url
    item.title = title
    return item


def _make_search_block(results: list[dict]) -> MagicMock:
    block = MagicMock()
    block.type = 'web_search_tool_result'
    block.content = [_make_search_item(r['url'], r['title']) for r in results]
    return block


def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = 'text'
    block.text = text
    return block


def _make_llm_response(
    search_results: list[dict],
    snippets: list[dict] | None = None,
    stop_reason: str = 'end_turn',
) -> MagicMock:
    """Build a mock Anthropic messages.create response with web search results."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    json_text = json.dumps(snippets or [
        {'title': r['title'], 'url': r['url'], 'snippet': r.get('snippet', 'A relevant site.')}
        for r in search_results
    ])
    resp.content = [_make_search_block(search_results), _make_text_block(json_text)]
    return resp


# ---------------------------------------------------------------------------
# _is_safe_url
# ---------------------------------------------------------------------------

class TestIsSafeUrl:
    def test_public_https(self):
        assert _is_safe_url('https://www.example.com/page') is True

    def test_public_http(self):
        assert _is_safe_url('http://example.com') is True

    def test_ftp_scheme(self):
        assert _is_safe_url('ftp://example.com') is False

    def test_no_scheme(self):
        assert _is_safe_url('example.com') is False

    def test_rfc1918_10(self):
        assert _is_safe_url('http://10.0.0.1/') is False

    def test_rfc1918_172(self):
        assert _is_safe_url('http://172.16.0.1/') is False

    def test_rfc1918_192_168(self):
        assert _is_safe_url('http://192.168.1.1/') is False

    def test_loopback_ip(self):
        assert _is_safe_url('http://127.0.0.1/') is False

    def test_loopback_hostname(self):
        assert _is_safe_url('http://localhost/') is False

    def test_link_local_metadata(self):
        assert _is_safe_url('http://169.254.169.254/latest/meta-data/') is False

    def test_cgnat(self):
        assert _is_safe_url('http://100.64.0.1/') is False

    def test_dot_local(self):
        assert _is_safe_url('http://printer.local/') is False

    def test_dot_internal(self):
        assert _is_safe_url('http://service.internal/') is False

    def test_ipv6_loopback(self):
        assert _is_safe_url('http://[::1]/') is False

    def test_ipv6_6to4(self):
        assert _is_safe_url('http://[2002::1]/') is False

    def test_public_ipv6(self):
        # 2001:db8:: is a documentation range - passes static IP check
        assert _is_safe_url('http://[2001:db8::1]/') is True

    def test_empty_string(self):
        assert _is_safe_url('') is False

    def test_malformed_url(self):
        assert _is_safe_url('not-a-url') is False


# ---------------------------------------------------------------------------
# search_sites()
# ---------------------------------------------------------------------------

class TestSearchSites:
    def _client(self, response: MagicMock) -> MagicMock:
        client = MagicMock()
        client.messages.create.return_value = response
        return client

    def test_happy_path_returns_results(self):
        results_data = [
            {'url': 'https://example.com', 'title': 'Example', 'snippet': 'Good site.'},
            {'url': 'https://other.com/reports', 'title': 'Other', 'snippet': 'Another site.'},
        ]
        client = self._client(_make_llm_response(results_data))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'find reports', 5)
        assert len(results) == 2
        assert all(isinstance(r, SearchSiteResult) for r in results)
        assert all(r.snippet_is_synthesized is True for r in results)
        urls = [r.url for r in results]
        assert 'https://example.com' in urls
        assert 'https://other.com/reports' in urls

    def test_max_results_1_caps_output_and_scales_tokens(self):
        results_data = [
            {'url': f'https://site{i}.com', 'title': f'Site {i}', 'snippet': 'Desc.'}
            for i in range(5)
        ]
        client = self._client(_make_llm_response(results_data))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 1)
        assert len(results) == 1
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs['max_tokens'] == 2048  # max(2048, 512 + 1*100)
        tool = call_kwargs['tools'][0]
        assert tool['max_uses'] == 2  # max(2, (1+4)//5)

    def test_max_results_20_scales_tokens_and_max_uses(self):
        results_data = [
            {'url': f'https://site{i}.com', 'title': f'Site {i}', 'snippet': 'Desc.'}
            for i in range(20)
        ]
        client = self._client(_make_llm_response(results_data))
        search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 20)
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs['max_tokens'] == 2512  # max(2048, 512 + 20*100)
        tool = call_kwargs['tools'][0]
        assert tool['max_uses'] == 4  # max(2, (20+4)//5) = max(2, 4) = 4

    def test_empty_search_results(self):
        client = self._client(_make_llm_response([], snippets=[]))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert results == []

    def test_all_urls_fail_safety(self):
        results_data = [
            {'url': 'http://10.0.0.1/', 'title': 'Internal', 'snippet': 'Bad.'},
            {'url': 'http://192.168.1.1/', 'title': 'LAN', 'snippet': 'Bad.'},
        ]
        client = self._client(_make_llm_response(results_data))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert results == []

    def test_www_normalization_deduplication(self):
        results_data = [
            {'url': 'https://www.example.com/page', 'title': 'WWW', 'snippet': 'First.'},
            {'url': 'https://example.com/page', 'title': 'No-WWW', 'snippet': 'Dup.'},
        ]
        client = self._client(_make_llm_response(results_data))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert len(results) == 1

    def test_different_paths_same_domain_kept(self):
        results_data = [
            {'url': 'https://example.com/reports', 'title': 'Reports', 'snippet': 'First.'},
            {'url': 'https://example.com/funds', 'title': 'Funds', 'snippet': 'Second.'},
        ]
        client = self._client(_make_llm_response(results_data))
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert len(results) == 2

    def test_a_host_that_starts_with_w_is_not_mistaken_for_www(self):
        # lstrip('www.') stripped characters, so web.dev became eb.dev and merged with it
        results_data = [
            {'url': 'https://web.dev/reports', 'title': 'A', 'snippet': 'First.'},
            {'url': 'https://eb.dev/reports', 'title': 'B', 'snippet': 'Second.'},
        ]
        client = self._client(_make_llm_response(results_data))
        assert len(search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)) == 2

    def test_error_block_content_is_skipped(self):
        """web_search_tool_result with non-list content (error) should be skipped."""
        error_block = MagicMock()
        error_block.type = 'web_search_tool_result'
        error_block.content = 'error_object'  # not a list

        good_results = [{'url': 'https://good.com', 'title': 'Good', 'snippet': 'Fine.'}]
        good_block = _make_search_block(good_results)
        text_block = _make_text_block(json.dumps([
            {'title': 'Good', 'url': 'https://good.com', 'snippet': 'Fine.'}
        ]))

        resp = MagicMock()
        resp.stop_reason = 'end_turn'
        resp.content = [error_block, good_block, text_block]

        client = self._client(resp)
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert len(results) == 1
        assert results[0].url == 'https://good.com'

    def test_json_parse_failure_returns_empty(self):
        resp = MagicMock()
        resp.stop_reason = 'end_turn'
        resp.content = [_make_text_block('not json at all')]
        client = self._client(resp)
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert results == []

    def test_api_exception_propagates(self):
        import anthropic as _a
        import httpx
        client = MagicMock()
        client.messages.create.side_effect = _a.APIError(
            'rate limit',
            request=httpx.Request('POST', 'https://api.anthropic.com'),
            body={},
        )
        with pytest.raises(_a.APIError):
            search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)

    def test_pause_turn_triggers_continuation(self):
        results_data = [{'url': 'https://example.com', 'title': 'Ex', 'snippet': 'Desc.'}]
        first_resp = _make_llm_response([], snippets=[], stop_reason='pause_turn')
        second_resp = _make_llm_response(results_data)

        client = MagicMock()
        client.messages.create.side_effect = [first_resp, second_resp]
        results = search_sites(client, 'claude-haiku-4-5-20251001', 'goal', 5)
        assert client.messages.create.call_count == 2
        assert len(results) == 1


# ---------------------------------------------------------------------------
# /v1/search-sites endpoint
# ---------------------------------------------------------------------------

client = TestClient(app)


class TestSearchSitesEndpoint:
    @pytest.fixture(autouse=True)
    def _api_key(self, monkeypatch):
        # The endpoint reads the key per request. Without this the tests only pass on a machine whose
        # services/crawler/.env is loaded, and fail in CI or a worktree.
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')

    def _mock_results(self, n: int = 2) -> list[SearchSiteResult]:
        return [
            SearchSiteResult(
                title=f'Site {i}',
                url=f'https://site{i}.com',
                snippet='A relevant site.',
            )
            for i in range(n)
        ]

    def test_happy_path(self):
        with patch('docseek.server.search_sites', return_value=self._mock_results()):
            resp = client.post('/v1/search-sites', json={'goal': 'Hungarian investment funds'})
        assert resp.status_code == 200
        body = resp.json()
        assert 'results' in body
        assert len(body['results']) == 2
        assert body['results'][0]['snippet_is_synthesized'] is True

    def test_auto_select_returns_one(self):
        with patch('docseek.server.search_sites', return_value=self._mock_results(3)):
            resp = client.post('/v1/search-sites', json={
                'goal': 'annual reports',
                'auto_select': True,
            })
        assert resp.status_code == 200
        assert len(resp.json()['results']) == 1

    def test_empty_results_200(self):
        with patch('docseek.server.search_sites', return_value=[]):
            resp = client.post('/v1/search-sites', json={'goal': 'niche topic'})
        assert resp.status_code == 200
        assert resp.json() == {'results': []}

    def test_missing_api_key_503(self, monkeypatch):
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        resp = client.post('/v1/search-sites', json={'goal': 'test'})
        assert resp.status_code == 503

    def test_invalid_model_422(self):
        resp = client.post('/v1/search-sites', json={
            'goal': 'test',
            'model': 'claude-3-haiku-20240307',
        })
        assert resp.status_code == 422
        assert 'does not support web search' in resp.json()['detail']

    def test_valid_model_in_allowlist(self):
        with patch('docseek.server.search_sites', return_value=[]):
            resp = client.post('/v1/search-sites', json={
                'goal': 'test',
                'model': 'claude-haiku-4-5-20251001',
            })
        assert resp.status_code == 200

    def test_anthropic_api_error_503(self, monkeypatch):
        import anthropic as _a
        import httpx
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        err = _a.APIError(
            'rate limit',
            request=httpx.Request('POST', 'https://api.anthropic.com'),
            body={},
        )
        with patch('docseek.server.search_sites', side_effect=err):
            resp = client.post('/v1/search-sites', json={'goal': 'test'})
        assert resp.status_code == 503

    def test_parse_error_500(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        with patch('docseek.server.search_sites', side_effect=ValueError('bad json')):
            resp = client.post('/v1/search-sites', json={'goal': 'test'})
        assert resp.status_code == 500

    def test_goal_empty_422(self):
        resp = client.post('/v1/search-sites', json={'goal': ''})
        assert resp.status_code == 422

    def test_max_results_zero_422(self):
        resp = client.post('/v1/search-sites', json={'goal': 'test', 'max_results': 0})
        assert resp.status_code == 422

    def test_max_results_over_limit_422(self):
        resp = client.post('/v1/search-sites', json={'goal': 'test', 'max_results': 21})
        assert resp.status_code == 422

    def test_auth_missing_key_401(self, monkeypatch):
        monkeypatch.setenv('CRAWLER_API_KEY', 'secret')
        resp = client.post('/v1/search-sites', json={'goal': 'test'})
        assert resp.status_code == 401

    def test_auth_correct_key_passes(self, monkeypatch):
        monkeypatch.setenv('CRAWLER_API_KEY', 'secret')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        with patch('docseek.server.search_sites', return_value=[]):
            resp = client.post(
                '/v1/search-sites',
                json={'goal': 'test'},
                headers={'x-api-key': 'secret'},
            )
        assert resp.status_code == 200

    def test_auth_wrong_key_401(self, monkeypatch):
        monkeypatch.setenv('CRAWLER_API_KEY', 'secret')
        resp = client.post(
            '/v1/search-sites',
            json={'goal': 'test'},
            headers={'x-api-key': 'wrong'},
        )
        assert resp.status_code == 401

    # Regression: existing endpoints unaffected
    def test_health_still_works(self):
        resp = client.get('/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 'ok'}

    def test_discover_still_requires_url(self):
        resp = client.post('/v1/discover', json={'goal': 'test'})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Integration: full path from HTTP request to mocked SDK boundary
# ---------------------------------------------------------------------------

class TestSearchSitesIntegration:
    """Drives the full path without mocking internal functions - only the SDK boundary."""

    def test_full_path_returns_results(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        results_data = [
            {'url': 'https://funds.example.org/alapok', 'title': 'Fund Association', 'snippet': 'Hungarian fund data.'},
            {'url': 'https://news.example.net/alapok', 'title': 'Fund News', 'snippet': 'Fund news.'},
        ]
        mock_resp = _make_llm_response(results_data)

        with patch('anthropic.Anthropic') as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post('/v1/search-sites', json={
                'goal': 'Hungarian investment fund annual reports',
                'max_results': 5,
            })

        assert resp.status_code == 200
        body = resp.json()
        assert len(body['results']) == 2
        urls = [r['url'] for r in body['results']]
        assert 'https://funds.example.org/alapok' in urls

    def test_auto_select_full_path(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        results_data = [
            {'url': f'https://site{i}.com', 'title': f'Site {i}', 'snippet': 'Desc.'}
            for i in range(3)
        ]
        mock_resp = _make_llm_response(results_data)

        with patch('anthropic.Anthropic') as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post('/v1/search-sites', json={
                'goal': 'test',
                'auto_select': True,
            })

        assert resp.status_code == 200
        assert len(resp.json()['results']) == 1

    def test_internal_ip_filtered_from_response(self, monkeypatch):
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
        results_data = [
            {'url': 'http://10.0.0.1/', 'title': 'Internal', 'snippet': 'Bad.'},
            {'url': 'https://safe.com', 'title': 'Safe', 'snippet': 'Good.'},
        ]
        mock_resp = _make_llm_response(results_data)

        with patch('anthropic.Anthropic') as MockClient:
            MockClient.return_value.messages.create.return_value = mock_resp
            resp = client.post('/v1/search-sites', json={'goal': 'test'})

        assert resp.status_code == 200
        urls = [r['url'] for r in resp.json()['results']]
        assert 'http://10.0.0.1/' not in urls
        assert 'https://safe.com' in urls
