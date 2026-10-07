"""The sitemap chain: robots lines or the usual names, indexes to any depth, gzip, the parent domain, dates."""
from __future__ import annotations

import gzip
from unittest.mock import MagicMock, patch

import pytest

import docseek.agent as agent
from docseek.agent import fetch_sitemap, fetch_sitemap_entries

URLSET = '''<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://site.example/a.pdf</loc><lastmod>2026-03-05</lastmod></url>
<url><loc>https://site.example/page</loc></url></urlset>'''
INDEX = '''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://site.example/sm-1.xml</loc></sitemap>
<sitemap><loc>https://site.example/deeper.xml</loc></sitemap></sitemapindex>'''
DEEPER = '''<sitemapindex><sitemap><loc>https://site.example/sm-2.xml.gz</loc></sitemap></sitemapindex>'''
SM2 = '<urlset><url><loc>https://site.example/b.pdf</loc></url></urlset>'


def serve(routes: dict[str, bytes | str]):
    """urlopen that answers from `routes` (a str is utf-8, bytes as they are) and refuses the rest."""
    calls = []

    def urlopen(req, timeout=None):
        url = req.full_url
        calls.append(url)
        if url not in routes:
            raise OSError(f'404 {url}')
        body = routes[url]
        m = MagicMock()
        m.__enter__ = lambda s: s
        m.__exit__ = MagicMock(return_value=False)
        m.read.return_value = body.encode() if isinstance(body, str) else body
        return m
    return urlopen, calls


@pytest.fixture(autouse=True)
def no_robots_cache(monkeypatch):
    monkeypatch.setattr(agent, 'store_robots', lambda host, lines: None)


def test_robots_sitemap_indexes_to_any_depth_and_gzip():
    urlopen, calls = serve({
        'https://site.example/robots.txt': 'User-agent: *\nSitemap: https://site.example/index.xml\n',
        'https://site.example/index.xml': INDEX,
        'https://site.example/sm-1.xml': URLSET,
        'https://site.example/deeper.xml': DEEPER,
        'https://site.example/sm-2.xml.gz': gzip.compress(SM2.encode()),
    })
    with patch('urllib.request.urlopen', urlopen):
        entries = fetch_sitemap_entries('https://site.example/start')
    assert entries == [{'url': 'https://site.example/a.pdf', 'lastmod': '2026-03-05'},
                       {'url': 'https://site.example/page', 'lastmod': None},
                       {'url': 'https://site.example/b.pdf', 'lastmod': None}]
    assert 'https://site.example/sitemap.xml' not in calls          # robots named the sitemap: no guessing


def test_without_robots_the_usual_names_are_tried_in_turn():
    urlopen, calls = serve({'https://site.example/sitemap_index.xml': URLSET})
    with patch('urllib.request.urlopen', urlopen):
        assert fetch_sitemap('https://site.example/') == ['https://site.example/a.pdf', 'https://site.example/page']
    assert calls[:3] == ['https://site.example/robots.txt', 'https://site.example/sitemap.xml',
                         'https://site.example/sitemap_index.xml']
    assert 'https://site.example/sitemap-index.xml' not in calls      # the first that answers is used


def test_a_subdomain_without_a_sitemap_reads_its_parent_domain():
    urlopen, calls = serve({'https://parent.example/sitemap.xml': URLSET})
    with patch('urllib.request.urlopen', urlopen):
        urls = fetch_sitemap('https://funds.parent.example/')
    assert urls == ['https://site.example/a.pdf', 'https://site.example/page']   # the caller keeps its own host
    assert all(c.startswith('https://funds.parent.example/') for c in calls[:6])  # own names first


def test_the_sitemap_count_is_capped(monkeypatch):
    monkeypatch.setattr(agent, 'SITEMAP_FILES', 3)
    children = ''.join(f'<sitemap><loc>https://site.example/sm-{i}.xml</loc></sitemap>' for i in range(10))
    routes = {'https://site.example/sitemap.xml': f'<sitemapindex>{children}</sitemapindex>'}
    routes.update({f'https://site.example/sm-{i}.xml': f'<urlset><url><loc>https://site.example/p{i}</loc></url></urlset>'
                   for i in range(10)})
    urlopen, calls = serve(routes)
    with patch('urllib.request.urlopen', urlopen):
        urls = fetch_sitemap('https://site.example/')
    assert urls == ['https://site.example/p0', 'https://site.example/p1']    # the index plus two sitemaps


def test_nothing_anywhere_is_an_empty_list():
    urlopen, _ = serve({})
    with patch('urllib.request.urlopen', urlopen):
        assert fetch_sitemap_entries('https://site.example/') == []
