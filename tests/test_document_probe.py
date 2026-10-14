"""A document is what the server says it is. No network: the probe's answer is stubbed."""
import httpx
import pytest

from docseek.document_probe import MAX_PROBES_PER_PAGE, DocumentProbe, is_document_response


@pytest.mark.parametrize('content_type, disposition, expected', [
    ('application/pdf; charset=utf-8', '', True),                      # a council system's getfile.asp
    ('application/pdf', 'inline; filename="x.pdf"', True),
    ('text/html; charset=utf-8', '', False),
    ('text/html', 'attachment; filename=report.pdf', True),            # the disposition wins
    ('application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', '', True),
    ('application/octet-stream', '', True),
    ('pdf', '', True),                                                 # one registry server sends this
    ('application/json', '', False),
    ('', '', False),
])
def test_headers_decide(content_type, disposition, expected):
    assert is_document_response(content_type, disposition) is expected


def probe_answering(documents: set[str]) -> DocumentProbe:
    probe = DocumentProbe('test-agent', lambda url: True)
    probe.asked = []

    def answer(url):
        probe.asked.append(url)
        return url in documents
    probe.is_document = answer
    return probe


def links(prefix: str, n: int, path: str) -> list[dict]:
    return [{'url': f'https://site-de.example/{prefix}?id={i}', 'name': f'{prefix} {i}', 'path': path} for i in range(n)]


class TestSort:
    def test_a_column_of_documents_costs_two_requests(self):
        files = links('getfile.asp', 30, 'table/tr/td/a')
        probe = probe_answering({link['url'] for link in files})
        documents, pages = probe.sort(files + links('si0057.asp', 5, 'div/ul/li/a'))
        assert len(documents) == 30 and len(pages) == 5
        assert len(probe.asked) == 4                                   # first and last of each group

    def test_a_mixed_group_only_gives_up_what_was_seen_to_be_a_document(self):
        group = links('item', 6, 'div/a')
        documents, pages = probe_answering({group[0]['url']}).sort(group)
        assert [d['url'] for d in documents] == [group[0]['url']]
        assert len(pages) == 5                     # the rest stay pages; a visit that starts a download recovers them

    def test_site_chrome_and_lone_links_are_never_probed(self):
        probe = probe_answering(set())
        menu = links('menu', 9, 'header/nav.main/ul/li/a') + links('legal', 4, 'footer.site/ul/li/a')
        documents, pages = probe.sort(menu + links('single', 1, 'main/p/a') + [{'url': 'https://site-de.example/sitemap-page'}])
        assert documents == [] and len(pages) == 15 and probe.asked == []

    def test_a_page_cannot_spend_more_than_its_share(self):
        many = [link for g in range(10) for link in links(f'g{g}', 3, f'section.s{g}/a')]
        probe = probe_answering(set())
        probe.sort(many)
        assert len(probe.asked) == MAX_PROBES_PER_PAGE

    def test_an_unanswered_probe_leaves_the_link_a_page(self):
        probe = DocumentProbe('test-agent', lambda url: False)        # robots or the SSRF guard said no
        group = links('getfile.asp', 4, 'table/tr/td/a')
        assert probe.sort(group) == ([], group) and probe.probes == 0


# --- a site that refuses the HTTP client is asked from the crawl's page ------------------------------------------

class SitePage:
    """The crawl's page on the site: its fetch() gets the headers the HTTP client was refused."""
    url = 'https://site-de.example/sitzungen'

    def __init__(self, documents: set[str]):
        self.documents, self.fetched = documents, []

    def evaluate(self, script, args):
        url, _ = args
        self.fetched.append(url)
        return {'status': 200, 'type': 'application/pdf' if url in self.documents else 'text/html',
                'disposition': ''}


def refusing_probe(status: int = 403) -> DocumentProbe:
    probe = DocumentProbe('test-agent', lambda url: True)
    probe._client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status)))
    return probe


class TestRefusedHttp:
    def test_a_refused_probe_is_asked_again_from_the_page(self):
        files = links('getfile.asp', 6, 'table/tr/td/a')
        page = SitePage({link['url'] for link in files})
        probe = refusing_probe()
        documents, pages = probe.sort(files, page)
        assert documents == files and pages == []
        assert probe.http_refused and probe.page_probes == 2

    def test_once_refused_the_page_is_asked_first(self):
        probe = refusing_probe()
        probe.http_refused = True
        probe._client = None                                     # the HTTP client is not touched again
        page = SitePage(set())
        probe.sort(links('getfile.asp', 3, 'table/tr/td/a'), page)
        assert len(page.fetched) == 2

    def test_another_origin_is_not_asked_from_the_page(self):
        # its headers are not the page's to read
        probe = refusing_probe()
        probe.http_refused = True
        page = SitePage(set())
        cdn = [{'url': f'https://cdn.example/f/{i}', 'path': 'table/tr/td/a'} for i in range(3)]
        assert probe.sort(cdn, page) == ([], cdn) and page.fetched == []

    def test_without_a_page_a_refusal_leaves_the_links_pages(self):
        files = links('getfile.asp', 4, 'table/tr/td/a')
        assert refusing_probe().sort(files) == ([], files)

    def test_a_not_found_is_not_a_refusal(self):
        probe = refusing_probe(404)
        page = SitePage(set())
        probe.sort(links('getfile.asp', 4, 'table/tr/td/a'), page)
        assert not probe.http_refused and page.fetched == []

    def test_what_the_page_answered_is_counted(self):
        files = links('getfile.asp', 4, 'table/tr/td/a')
        page = RefusingAfterOnePage({files[0]['url']})
        probe = refusing_probe()
        probe.sort(files + links('si0057.asp', 3, 'div/ul/li/a'), page)
        assert probe.page_answers == {'document': 1, 'refused': 1, 'page': 2}


class RefusingAfterOnePage(SitePage):
    """Its second fetch is refused, as a site's bot protection might once the page asks too often."""
    def evaluate(self, script, args):
        answer = super().evaluate(script, args)
        return {**answer, 'status': 403} if len(self.fetched) == 2 else answer
