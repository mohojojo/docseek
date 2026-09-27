"""Tests for the three pi-scraper-inspired improvements:
  - _fetch_llms_txt
  - _url_allowed_by_robots / fetch_sitemap robots caching
  - _fast_harvest escalation logic
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from docseek.agent import (
    _fast_harvest,
    _fetch_llms_txt,
    _is_same_domain,
    _robots_cache,
    _url_allowed_by_robots,
    _url_passes_screen,
)
from docseek.models import CrawlPlan


# ---------------------------------------------------------------------------
# llms.txt extraction
# ---------------------------------------------------------------------------

class TestFetchLlmsTxt:
    def _make_response(self, body: str, status: int = 200):
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        m.read.return_value = body.encode()
        return m

    def test_extracts_markdown_links(self):
        body = "# Site\n\n- [Page A](https://example.com/a)\n- [Page B](https://example.com/b)\n"
        with patch('urllib.request.urlopen', return_value=self._make_response(body)):
            urls = _fetch_llms_txt('https://example.com', timeout=5)
        assert 'https://example.com/a' in urls
        assert 'https://example.com/b' in urls

    def test_extracts_bare_urls(self):
        body = "https://example.com/page1\nhttps://example.com/page2\n"
        with patch('urllib.request.urlopen', return_value=self._make_response(body)):
            urls = _fetch_llms_txt('https://example.com', timeout=5)
        assert 'https://example.com/page1' in urls
        assert 'https://example.com/page2' in urls

    def test_deduplicates(self):
        body = "https://example.com/x\nhttps://example.com/x\n"
        with patch('urllib.request.urlopen', return_value=self._make_response(body)):
            urls = _fetch_llms_txt('https://example.com', timeout=5)
        assert urls.count('https://example.com/x') == 1

    def test_returns_empty_on_network_error(self):
        with patch('urllib.request.urlopen', side_effect=OSError('timeout')):
            urls = _fetch_llms_txt('https://example.com', timeout=5)
        assert urls == []


# ---------------------------------------------------------------------------
# robots.txt Disallow compliance
# ---------------------------------------------------------------------------

class TestRobotsCompliance:
    def setup_method(self):
        _robots_cache.clear()

    def _populate_cache(self, domain: str, robots_text: str) -> None:
        import urllib.robotparser
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(robots_text.splitlines())
        _robots_cache[domain] = rp

    def test_allows_permitted_url(self):
        self._populate_cache('example.com', 'User-agent: *\nDisallow: /private/\n')
        assert _url_allowed_by_robots('https://example.com/public/page') is True

    def test_blocks_disallowed_url(self):
        self._populate_cache('example.com', 'User-agent: *\nDisallow: /private/\n')
        assert _url_allowed_by_robots('https://example.com/private/secret') is False

    def test_an_uncached_host_has_its_robots_fetched_first(self, monkeypatch):
        # Not cached: the rules are loaded on this first check (crawler.reach), not assumed to allow
        import docseek.reach as reach
        loaded = []
        monkeypatch.setattr(reach, '_load_robots', lambda scheme, netloc: loaded.append(netloc))
        assert _url_allowed_by_robots('https://uncached.example.com/page') is True   # none readable: allowed
        assert loaded == ['uncached.example.com']

    def test_url_passes_screen_respects_robots(self):
        self._populate_cache('example.com', 'User-agent: *\nDisallow: /blocked/\n')
        plan = CrawlPlan()
        assert _url_passes_screen('https://example.com/blocked/page', 1, plan, 0.0) is False
        assert _url_passes_screen('https://example.com/allowed/page', 1, plan, 0.0) is True


# ---------------------------------------------------------------------------
# fast harvest escalation
# ---------------------------------------------------------------------------

def _make_httpx_response(html: str, status: int = 200, content_type: str = 'text/html; charset=utf-8', final_url: str = 'https://example.com/page'):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {'content-type': content_type}
    resp.text = html
    resp.url = final_url
    return resp


class TestFastHarvest:
    def _harvest(self, html: str, **kwargs):
        defaults = dict(
            url='https://example.com/page',
            user_agent='Mozilla/5.0',
            seed_host='example.com',
            same_domain_only=True,
            crawl_plan=None,
            min_url_score=0.0,
            depth=1,
            visited_snapshot=frozenset(),
        )
        defaults.update(kwargs)
        import httpx
        mock_resp = _make_httpx_response(html, final_url=defaults['url'])
        with patch.object(httpx, 'get', return_value=mock_resp):
            return _fast_harvest(**defaults)

    def test_needs_playwright_on_network_error(self):
        import httpx
        with patch.object(httpx, 'get', side_effect=OSError('timeout')):
            result = _fast_harvest(
                url='https://example.com/', user_agent='Mozilla/5.0',
                seed_host='example.com', same_domain_only=True,
                crawl_plan=None, min_url_score=0.0, depth=0,
                visited_snapshot=frozenset(),
            )
        assert result.needs_playwright is True
        assert result.downloads == []

    def test_needs_playwright_on_403(self):
        import httpx
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {'content-type': 'text/html'}
        with patch.object(httpx, 'get', return_value=mock_resp):
            result = _fast_harvest(
                url='https://example.com/', user_agent='Mozilla/5.0',
                seed_host='example.com', same_domain_only=True,
                crawl_plan=None, min_url_score=0.0, depth=0,
                visited_snapshot=frozenset(),
            )
        assert result.needs_playwright is True

    def test_needs_playwright_for_js_shell(self):
        js_shell = '<html><body><div id="app"></div><script>window.__NEXT_DATA__={};</script></body></html>'
        result = self._harvest(js_shell)
        assert result.needs_playwright is True

    def test_needs_playwright_when_no_doc_links(self):
        html = '<html><body>' + 'word ' * 200 + '<a href="/about">About</a></body></html>'
        result = self._harvest(html)
        assert result.needs_playwright is True
        assert result.downloads == []

    def test_harvests_pdf_links_skips_playwright(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="/reports/annual.pdf">Annual Report</a>'
            '</body></html>'
        )
        result = self._harvest(html)
        assert result.needs_playwright is False
        assert len(result.downloads) == 1
        assert result.downloads[0].url == 'https://example.com/reports/annual.pdf'
        assert result.downloads[0].name == 'Annual Report'

    def test_harvests_xlsx_links(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="/data.xlsx">Data</a>'
            '</body></html>'
        )
        result = self._harvest(html)
        assert any(d.url.endswith('.xlsx') for d in result.downloads)

    def test_still_needs_playwright_with_js_signals_even_if_docs_found(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<script>window.__NEXT_DATA__={"props":{}};</script>'
            '<a href="/doc.pdf">Doc</a>'
            '</body></html>'
        )
        result = self._harvest(html)
        assert result.needs_playwright is True
        # But docs are still pre-recorded
        assert len(result.downloads) == 1

    def test_queues_html_page_links(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="/funds/fund-a">Fund A</a>'
            '<a href="/funds/fund-b">Fund B</a>'
            '</body></html>'
        )
        result = self._harvest(html)
        queued_urls = [u for u, _ in result.queue_entries]
        assert 'https://example.com/funds/fund-a' in queued_urls
        assert 'https://example.com/funds/fund-b' in queued_urls

    def test_skips_off_domain_links_when_same_domain_only(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="https://other.com/report.pdf">External</a>'
            '</body></html>'
        )
        result = self._harvest(html, same_domain_only=True)
        assert all('other.com' not in d.url for d in result.downloads)

    def test_keeps_links_on_bare_host_when_seed_is_www(self):
        # A www. seed that redirects to its bare host; a strict host match dropped every link.
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="https://example.com/report.pdf">Report</a>'
            '<a href="https://example.com/funds/fund-a">Fund A</a>'
            '</body></html>'
        )
        result = self._harvest(html, url='https://example.com/', seed_host='www.example.com')
        assert [d.url for d in result.downloads] == ['https://example.com/report.pdf']
        assert 'https://example.com/funds/fund-a' in [u for u, _ in result.queue_entries]

    def test_skips_already_visited_urls(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="/already">Already visited</a>'
            '</body></html>'
        )
        result = self._harvest(html, visited_snapshot=frozenset({'https://example.com/already'}))
        queued_urls = [u for u, _ in result.queue_entries]
        assert 'https://example.com/already' not in queued_urls

    def test_resolves_relative_urls(self):
        html = (
            '<html><body>' + 'word ' * 200 +
            '<a href="../other/doc.pdf">Doc</a>'
            '</body></html>'
        )
        result = self._harvest(html, url='https://example.com/section/page')
        assert len(result.downloads) == 1
        assert result.downloads[0].url == 'https://example.com/other/doc.pdf'


class TestIsSameDomain:
    @pytest.mark.parametrize('url, seed_host', [
        ('https://example.com/a', 'example.com'),
        ('https://example.com/a', 'www.example.com'),
        ('https://www.example.com/a', 'example.com'),
        ('https://WWW.Example.com/a', 'www.example.com'),
    ])
    def test_same_site_ignoring_www_and_case(self, url, seed_host):
        # Arrange / Act / Assert
        assert _is_same_domain(url, seed_host) is True

    @pytest.mark.parametrize('url, seed_host', [
        ('https://other.com/a', 'example.com'),
        ('https://sub.example.com/a', 'example.com'),
        ('https://example.com.evil.io/a', 'example.com'),
    ])
    def test_other_hosts_are_off_domain(self, url, seed_host):
        assert _is_same_domain(url, seed_host) is False
