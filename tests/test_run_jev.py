"""The Jev eval runner's scoring. No network, no browser."""
from docseek.models import AgenticDownload
from eval.run_jev import score_run, summarise

EXPECTED = {'https://shop.example/a.pdf', 'https://shop.example/b.pdf', 'https://shop.example/c.pdf'}
PAGES = {'https://shop.example/early': 5, 'https://shop.example/late': 25}


def candidate(name: str, verdict: str, source_page: str = 'https://shop.example/early', source: str = 'page'):
    return AgenticDownload(url=f'https://shop.example/{name}.pdf', name=name, reason='', source_page=source_page,
                           verdict=verdict, source=source)


class TestScoreRun:
    def test_accepted_and_returned_are_scored_separately(self):
        run = score_run([candidate('a', 'accepted'), candidate('x', 'unsure'), candidate('y', 'rejected')],
                        PAGES, EXPECTED)
        assert run['accepted']['precision'] == 1.0
        assert run['returned']['found_count'] == 2          # rejected is never returned
        assert run['returned']['precision'] == 0.5

    def test_recall_at_counts_only_candidates_found_by_that_page(self):
        run = score_run([candidate('a', 'accepted'), candidate('b', 'accepted', 'https://shop.example/late')],
                        PAGES, EXPECTED)
        assert run['recall_at'] == {'10': 0.333, '20': 0.333, '40': 0.667}
        assert run['pages_to_first_hit'] == 5

    def test_sitemap_and_api_candidates_exist_before_the_first_page(self):
        run = score_run([candidate('a', 'accepted', 'https://shop.example/', 'sitemap')], PAGES, EXPECTED)
        assert run['pages_to_first_hit'] == 0
        assert run['recall_at']['10'] == 0.333

    def test_no_hit_has_no_first_page(self):
        assert score_run([candidate('x', 'accepted')], PAGES, EXPECTED)['pages_to_first_hit'] is None


class TestSummarise:
    def run(self, recall: float, discarded: bool = False) -> dict:
        band = {'recall': recall, 'precision': 1.0}
        return {'discarded': discarded, 'accepted': band, 'returned': band,
                'recall_at': {'10': recall, '20': recall, '40': recall},
                'pages': 10, 'elapsed_s': 1.0, 'agent_tokens': 0, 'jev_usd': 0.0}

    def test_a_jev_unavailable_run_is_left_out(self):
        summary = summarise([self.run(0.8), self.run(0.0, discarded=True), {'error': 'boom'}])
        assert summary['runs_kept'] == 1 and summary['runs_discarded'] == 2
        assert summary['accepted_recall']['mean'] == 0.8

    def test_no_usable_run_has_no_summary(self):
        assert summarise([self.run(0.0, discarded=True)]) is None


class TestWiderEval:
    def test_a_site_with_nothing_to_find_scores_false_accepts_not_a_zero(self):
        clean = score_run([candidate('x', 'rejected'), candidate('y', 'unsure')], PAGES, set())
        assert clean['accepted']['recall'] == 1.0 and clean['accepted']['precision'] == 1.0
        assert clean['returned']['false_accepts'] == 1              # unsure is still returned to the caller
        wrong = score_run([candidate('x', 'accepted')], PAGES, set())
        assert wrong['accepted']['precision'] == 0.0 and wrong['accepted']['false_accepts'] == 1

    def test_the_query_string_can_be_a_documents_identity(self):
        docs = [AgenticDownload(url=f'https://shop.example/getfile.aspx?id={i}', name=str(i), reason='',
                                source_page='https://shop.example/early', verdict='accepted', source='page')
                for i in (1, 2)]
        expected = {'https://shop.example/getfile.aspx?id=1', 'https://shop.example/getfile.aspx?id=2'}
        assert score_run(docs, PAGES, expected, keep_query=True)['accepted']['true_positives'] == 2
        assert score_run(docs, PAGES, {'https://shop.example/getfile.aspx'})['accepted']['found_count'] == 1
