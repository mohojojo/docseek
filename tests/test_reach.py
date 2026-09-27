"""Reach (crawler.reach): where a crawl may go - SSRF with DNS resolution, robots.txt for every host."""
import urllib.robotparser

import pytest

import docseek.reach as reach
from docseek.reach import is_crawlable, is_safe_url, robots_allows


def resolving_to(*addresses):
    return lambda host: tuple(addresses)


@pytest.mark.parametrize('addresses', [
    ('127.0.0.1',), ('169.254.169.254',), ('10.1.2.3',), ('192.168.0.10',), ('::1',), ('fe80::1',),
    ('93.184.216.34', '10.0.0.5'),                      # one private answer is enough to refuse
    (),                                                  # does not resolve
])
def test_a_hostname_is_judged_by_where_it_points(monkeypatch, addresses):
    monkeypatch.setattr(reach, '_resolve', resolving_to(*addresses))
    assert is_safe_url('https://innocent-looking.example/report.pdf') is False


def test_a_public_hostname_is_safe(monkeypatch):
    monkeypatch.setattr(reach, '_resolve', resolving_to('93.184.216.34', '2606:2800:220:1::1'))
    assert is_safe_url('https://example.com/report.pdf') is True


@pytest.mark.parametrize('url', ['http://[::ffff:127.0.0.1]/', 'http://127.0.0.1:8010/', 'file:///etc/passwd',
                                 'https://localhost/', 'https://printer.local/', 'ftp://example.com/a.pdf'])
def test_literal_and_local_targets_are_refused_without_resolving(monkeypatch, url):
    monkeypatch.setattr(reach, '_resolve', lambda host: pytest.fail(f'resolved {host}'))
    assert is_safe_url(url) is False


def _rules(*lines) -> urllib.robotparser.RobotFileParser:
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(list(lines))
    return parser


def test_robots_are_loaded_once_for_any_host_on_its_first_check(monkeypatch):
    loads = []

    def load(scheme, netloc):
        loads.append(netloc)
        return _rules('User-agent: *', 'Disallow: /private/')
    monkeypatch.setattr(reach, '_load_robots', load)
    assert robots_allows('https://other-host.example/private/doc.pdf') is False
    assert robots_allows('https://other-host.example/public/doc.pdf') is True
    assert loads == ['other-host.example']                 # not only the seed a sitemap was read from


def test_a_host_without_readable_robots_is_allowed(monkeypatch):
    monkeypatch.setattr(reach, '_load_robots', lambda scheme, netloc: None)
    assert robots_allows('https://no-robots.example/anything') is True


def test_crawlable_needs_both(monkeypatch):
    monkeypatch.setattr(reach, '_load_robots', lambda scheme, netloc: _rules('User-agent: *', 'Disallow: /'))
    assert is_crawlable('https://example.com/page') is False
    monkeypatch.setattr(reach, '_resolve', resolving_to('127.0.0.1'))
    reach._robots_cache.clear()
    monkeypatch.setattr(reach, '_load_robots', lambda scheme, netloc: None)
    assert is_crawlable('https://example.com/page') is False


def test_the_agent_does_not_record_a_download_that_points_inside(monkeypatch):
    from unittest.mock import MagicMock

    from docseek.agent import _visit_page
    from docseek.llm import ChatReply, Usage
    from tests.test_agent import _patch_visit_page_scraper

    monkeypatch.setattr(reach, '_resolve', lambda host: ('169.254.169.254',) if host == 'metadata.evil' else ('93.184.216.34',))

    class Scripted:
        provider, model, usage = 's', 's', Usage()

        def __init__(self):
            self.calls = [('record_download', {'url': 'http://metadata.evil/latest/meta-data/', 'name': 'x', 'reason': 'r'}),
                          ('record_downloads', {'items': [{'url': 'http://metadata.evil/a.pdf'},
                                                          {'url': 'https://example.com/a.pdf'}], 'reason': 'r'}),
                          ('done', {'reason': 'd'})]

        def chat(self, system, messages, tools, *, force_tool=True, max_tokens=8192):
            name, args = self.calls.pop(0)
            return ChatReply(content=[{'type': 'tool_use', 'id': name, 'name': name, 'input': args}], stop_reason='tool_use')

    page = MagicMock()
    page.url = 'https://example.com/'
    with _patch_visit_page_scraper(page, None):
        result = _visit_page('https://example.com/', depth=0, llm=Scripted(), goal='g', system_blocks=[],
                             seed_host='example.com', same_domain_only=True, js_wait_ms=0, click_wait_ms=0,
                             max_tool_steps=5, crawl_plan=None, min_url_score=0.0, memory_snapshot={}, open_kwargs={},
                             queue_size_hint=0, pre_interactions=[], on_event=None, visited_snapshot=frozenset())
    assert [d.url for d in result.downloads] == ['https://example.com/a.pdf']
