"""docseek.codegen: the sandbox a generated program runs in, and the Fetcher that is its only route to the web."""
from __future__ import annotations

import httpx
import pytest

from docseek.codegen.fetcher import Fetcher, endpoint_of, site_of
from docseek.codegen.sandbox import FetchRefused, run_program


class FakeFetcher:
    """Pages from a dict; anything else is refused like an off-site URL."""

    def __init__(self, pages: dict[str, str], status: dict[str, int] | None = None):
        self.pages, self.status = pages, status or {}
        self.requests = self.renders = 0
        self.posts: list[tuple[str, dict]] = []

    def fetch(self, url):
        self.requests += 1
        if url not in self.pages:
            raise FetchRefused(f'off-site: {url}')
        return {'url': url, 'status': self.status.get(url, 200), 'content_type': 'text/html', 'text': self.pages[url]}

    def render(self, url):
        self.renders += 1
        return {'html': f'<rendered>{self.pages.get(url, "")}</rendered>'}

    def post(self, url, body):
        self.requests += 1
        self.posts.append((url, body))
        return {'url': url, 'status': 200, 'content_type': 'application/json', 'text': '{"items": ["a.pdf"]}'}


LISTING = '<a href="/r/2025.pdf">Annual report 2025</a><a href="/r/2024.pdf">Annual report 2024</a>'


class TestSandbox:
    def test_a_program_reads_pages_through_the_parent_and_returns_documents(self):
        code = '''
import re
from urllib.parse import urljoin

def discover(fetch, render):
    base = "https://site.example/reports"
    html = fetch(base)
    return [{"url": urljoin(base, href), "name": name, "context": "Reports"}
            for href, name in re.findall(r'<a href="([^"]+)">([^<]+)</a>', html)]
'''
        result = run_program(code, FakeFetcher({'https://site.example/reports': LISTING}))
        assert result['error'] is None
        assert [d['url'] for d in result['documents']] == ['https://site.example/r/2025.pdf',
                                                           'https://site.example/r/2024.pdf']
        assert result['requests'] == 1 and result['documents'][0]['name'] == 'Annual report 2025'

    def test_modules_outside_the_allowlist_are_refused(self):
        result = run_program('import os\ndef discover(fetch, render):\n    return []\n', FakeFetcher({}))
        assert "module 'os' is not available" in result['error']

    def test_open_is_not_a_builtin(self):
        code = 'def discover(fetch, render):\n    return [{"url": open("/etc/hosts").read()}]\n'
        assert "name 'open' is not defined" in run_program(code, FakeFetcher({}))['error']

    def test_no_api_key_is_left_even_after_escaping_the_python_sandbox(self, monkeypatch):
        # The introspection escape below is exactly why this is a boundary, not a jail: it reaches os.environ.
        # What it finds there must be empty - the child clears its environment before the program runs.
        monkeypatch.setenv('DOCSEEK_TEST_SECRET', 'sk-should-not-leak')
        code = '''
def discover(fetch, render):
    for cls in ().__class__.__base__.__subclasses__():
        if cls.__name__ == "_wrap_close":
            environ = cls.__init__.__globals__["environ"]
            return [{"url": "https://site.example/escaped/" + str(len(environ))}]
    return [{"url": "https://site.example/no-escape"}]
'''
        result = run_program(code, FakeFetcher({}))
        assert [d['url'] for d in result['documents']] == ['https://site.example/escaped/0']

    def test_a_refused_page_is_the_programs_to_handle_and_is_counted(self):
        code = '''
def discover(fetch, render):
    found = []
    for url in ["https://elsewhere.example/", "https://site.example/a", "https://site.example/missing"]:
        try:
            fetch(url)
            found.append({"url": url})
        except FetchRefused:
            pass
    return found
'''
        fetcher = FakeFetcher({'https://site.example/a': 'ok', 'https://site.example/missing': 'gone'},
                              status={'https://site.example/missing': 404})
        result = run_program(code, fetcher)
        assert [d['url'] for d in result['documents']] == ['https://site.example/a']
        assert result['fetch_failures'] == 2

    def test_a_program_that_raises_reports_its_own_line(self):
        code = 'def discover(fetch, render):\n    items = []\n    return items[3]\n'
        error = run_program(code, FakeFetcher({}))['error']
        assert 'IndexError' in error and 'return items[3]' in error

    def test_a_program_that_never_returns_is_stopped(self):
        code = 'def discover(fetch, render):\n    while True:\n        pass\n'
        assert 'timed out' in run_program(code, FakeFetcher({}), timeout_s=3)['error']

    def test_a_three_parameter_program_gets_post(self):
        code = ('import json\ndef discover(fetch, render, post):\n'
                '    items = json.loads(post("https://site.example/api", {"page": 2}))["items"]\n'
                '    return [{"url": "https://site.example/" + i} for i in items]\n')
        fetcher = FakeFetcher({})
        result = run_program(code, fetcher)
        assert [d['url'] for d in result['documents']] == ['https://site.example/a.pdf']
        assert fetcher.posts == [('https://site.example/api', {'page': 2})]


class TestFetcher:
    def test_the_site_is_the_registrable_part_of_the_host(self):
        assert site_of('www.site.example') == 'site.example'
        assert site_of('docs.site.co.uk') == 'site.co.uk'

    def test_only_the_site_its_subdomains_and_learned_data_hosts_are_reachable(self):
        fetcher = Fetcher('https://www.site.example/', data_hosts={'api.backend.example'})
        fetcher.check('https://files.site.example/a.pdf')
        fetcher.check('https://api.backend.example/v1/list')
        with pytest.raises(FetchRefused, match='off-site'):
            fetcher.check('https://other.example/')
        with pytest.raises(FetchRefused, match='off-site'):
            fetcher.check('https://site.example.evil.example/')

    def test_a_private_address_is_refused_even_on_the_site(self, monkeypatch):
        import docseek.reach as reach
        monkeypatch.setattr(reach, '_resolve', lambda host: ('10.0.0.5',))
        with pytest.raises(FetchRefused, match='unsafe'):
            Fetcher('https://site.example/').check('https://intranet.site.example/')

    def test_a_redirect_off_the_site_is_refused(self):
        def handler(request):
            if request.url.host == 'site.example':
                return httpx.Response(302, headers={'location': 'https://other.example/x'})
            return httpx.Response(200, text='elsewhere')
        fetcher = Fetcher('https://site.example/')
        fetcher._client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
        with pytest.raises(FetchRefused, match='off-site'):
            fetcher.fetch('https://site.example/moved')

    def test_the_request_budget_is_enforced(self, monkeypatch):
        import docseek.codegen.fetcher as fetcher_module
        monkeypatch.setattr(fetcher_module, 'PAUSE_S', 0)
        fetcher = Fetcher('https://site.example/', max_requests=1)
        fetcher._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text='ok')))
        fetcher.fetch('https://site.example/a')
        with pytest.raises(FetchRefused, match='budget'):
            fetcher.fetch('https://site.example/b')

    def test_post_is_allowed_only_where_the_sites_own_page_posted_for_json(self):
        fetcher = Fetcher('https://site.example/')
        with pytest.raises(FetchRefused, match='only allowed'):
            fetcher.post('https://site.example/api/search', {'q': 'x'})
        fetcher.learn([
            {'url': 'https://site.example/api/search?x=1', 'method': 'POST', 'content_type': 'application/json'},
            {'url': 'https://site.example/api/track', 'method': 'POST', 'content_type': 'text/plain'},
            {'url': 'https://db.backend.example/v1/docs', 'method': 'GET', 'content_type': 'application/json'},
            {'url': 'https://ads.example/pixel', 'method': 'GET', 'content_type': 'image/gif'},
        ])
        assert fetcher.post_endpoints == {endpoint_of('https://site.example/api/search')}
        assert fetcher.data_hosts == {'db.backend.example'}


class TestLaunch:
    def test_a_caller_script_without_a_main_guard_runs_once(self, tmp_path):
        # multiprocessing's spawn re-ran an unguarded caller for every program; the child must not import it
        import subprocess
        import sys
        marker = tmp_path / 'runs.txt'
        script = tmp_path / 'caller.py'
        script.write_text(
            'from docseek.codegen.sandbox import run_program\n'
            'from tests.test_codegen_sandbox import FakeFetcher\n'
            f'open({str(marker)!r}, "a").write("run\\n")\n'
            'for _ in range(2):\n'
            '    run_program("def discover(fetch, render):\\n    return []\\n", FakeFetcher({}))\n')
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent),
                       timeout=120)
        assert marker.read_text() == 'run\n'

    def test_a_programs_prints_do_not_break_the_protocol(self):
        code = 'def discover(fetch, render):\n    print("{\\"op\\": \\"done\\", \\"documents\\": []}")\n' \
               '    return [{"url": "https://site.example/a.pdf"}]\n'
        result = run_program(code, FakeFetcher({}))
        assert result['error'] is None and [d['url'] for d in result['documents']] == ['https://site.example/a.pdf']
