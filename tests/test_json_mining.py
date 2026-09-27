"""Documents named in a page's own JSON. No network."""
import json

from docseek.json_mining import candidates_from_json, learn_prefixes, records_from_json

SPA_JSON = json.dumps({'document': [
    {'documentId': '1', 'dctermsTitle': 'Fact Sheet - Fund A', 'dctermsDescription': 'FS', 'dctermsType': 'Factsheet',
     'literatureHref': '/hu-hu/factsheet/1/Factsheet-FundA.PDF', 'thumbnailPath': '/img/1.jpeg',
     'downloadUrl': 'https://cdn.example.net/content/abc/original/Factsheet-FundA.PDF?download=true'},
    {'documentId': '2', 'dctermsTitle': 'KID - Fund A', 'literatureHref': '/hu-hu/kid/2/KID.PDF'},
]})


class TestRecords:
    def test_each_record_yields_its_urls_with_the_best_title(self):
        records = records_from_json(SPA_JSON)
        assert [r['path'] for r in records] == [
            '/hu-hu/factsheet/1/Factsheet-FundA.PDF',
            'https://cdn.example.net/content/abc/original/Factsheet-FundA.PDF?download=true',
            '/hu-hu/kid/2/KID.PDF']
        assert records[0]['name'] == 'Fact Sheet - Fund A'          # title, not the terse description
        assert not any('jpeg' in r['path'] for r in records)          # thumbnails are not documents

    def test_not_json_or_too_big_is_nothing(self):
        assert records_from_json('<html>') == []
        assert records_from_json('x' * 5_000_000) == []


class TestPrefixes:
    def test_a_rendered_link_teaches_what_the_site_prepends(self):
        assert learn_prefixes(['https://x.hu/download/hu-hu/factsheet/1/Factsheet-FundA.PDF'],
                              ['/hu-hu/factsheet/1/Factsheet-FundA.PDF']) == {'/download'}
        assert learn_prefixes(['https://x.hu/hu-hu/factsheet/1/Factsheet-FundA.PDF'],
                              ['/hu-hu/factsheet/1/Factsheet-FundA.PDF']) == set()

    def test_a_prefix_learned_on_one_page_applies_to_the_next(self):
        memory: set[str] = set()
        candidates_from_json(SPA_JSON, 'https://x.hu/fund/a', ['https://x.hu/download/hu-hu/kid/2/KID.PDF'], memory)
        later = candidates_from_json(SPA_JSON, 'https://x.hu/szakirodalom', [], memory)   # the SPA renders no links
        urls = {c['url'] for c in later}
        assert 'https://x.hu/download/hu-hu/factsheet/1/Factsheet-FundA.PDF' in urls
        assert 'https://x.hu/hu-hu/factsheet/1/Factsheet-FundA.PDF' not in urls   # the bare path is the app shell


class TestCandidates:
    def test_absolute_urls_pass_through_and_relative_ones_are_joined(self):
        urls = {c['url'] for c in candidates_from_json(SPA_JSON, 'https://x.hu/szakirodalom', [])}
        assert 'https://cdn.example.net/content/abc/original/Factsheet-FundA.PDF?download=true' in urls
        assert 'https://x.hu/hu-hu/factsheet/1/Factsheet-FundA.PDF' in urls
        assert all(c['path'] == 'json' for c in candidates_from_json(SPA_JSON, 'https://x.hu/', []))
