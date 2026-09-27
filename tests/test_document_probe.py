"""A document is what the server says it is. No network: the probe's answer is stubbed."""
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
