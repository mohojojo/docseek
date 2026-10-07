"""docseek.paths: a model's path plan for a goal, grounded in the site's own prefixes, orders the sitemap pages."""
from docseek.paths import MIN_PAGES_TO_PLAN, is_planned, path_sample, plan_paths, rank_by_paths

URLS = [f'https://site.example/products/{i}' for i in range(60)] + \
       [f'https://site.example/investors/reports/{y}' for y in (2024, 2025)] + \
       ['https://site.example/news/story', 'https://site.example/about']


class ScriptedLLM:
    model = 'scripted'

    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def complete_json(self, system, user):
        self.prompts.append(user)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_the_sample_is_the_sections_with_counts_most_populated_first_and_no_single_page_slugs():
    sample = path_sample(URLS)
    assert sample[0] == ('/products/', 60) and ('/investors/reports/', 2) in sample
    assert not any(prefix.startswith('/products/') and prefix != '/products/' for prefix, _ in sample)


def test_the_plan_keeps_only_substrings_that_are_on_the_site():
    llm = ScriptedLLM({'prefer': ['/investors/', '/reports/', '/imaginary/'], 'skip': ['/news/', 7, '/investors/']})
    plan = plan_paths(llm, 'Find the annual reports', URLS)
    assert plan == {'prefer': ['/investors/', '/reports/'], 'skip': ['/news/']}
    assert '    60  /products/' in llm.prompts[0] and 'Find the annual reports' in llm.prompts[0]


def test_a_small_site_or_a_failing_model_means_no_plan():
    llm = ScriptedLLM({'prefer': ['/investors/'], 'skip': []})
    assert plan_paths(llm, 'g', URLS[:MIN_PAGES_TO_PLAN - 1]) == {'prefer': [], 'skip': []} and not llm.prompts
    assert plan_paths(ScriptedLLM(RuntimeError('down')), 'g', URLS) == {'prefer': [], 'skip': []}


def test_ranking_puts_preferred_paths_first_in_the_plans_order_then_goal_years_then_skipped_last():
    ranked = rank_by_paths(URLS, ['/investors/'], ['/news/'], years=frozenset({'2025'}))
    assert ranked[:2] == ['https://site.example/investors/reports/2024', 'https://site.example/investors/reports/2025']
    assert ranked[-1] == 'https://site.example/news/story'
    assert ranked[2] == 'https://site.example/products/0' and sorted(ranked) == sorted(URLS)
    # the plan's first pattern is the model's best guess: its pages come before a broader pattern's
    ranked = rank_by_paths(URLS, ['/investors/reports/', '/products/'], [])
    assert ranked[0].startswith('https://site.example/investors/') and ranked[2] == 'https://site.example/products/0'
    ranked = rank_by_paths(URLS, ['/products/', '/investors/reports/'], [])
    assert ranked[0] == 'https://site.example/products/0'
    # a goal year without a preferred path still comes before the rest
    ranked = rank_by_paths(URLS, [], [], years=frozenset({'2025'}))
    assert ranked[0] == 'https://site.example/investors/reports/2025' and ranked[1] == 'https://site.example/products/0'


def test_is_planned_reads_the_decoded_path():
    assert is_planned('https://site.example/Produkte/Holz%C3%B6le/x', ['/produkte/holzöle/'])
    assert not is_planned('https://site.example/news/x', ['/produkte/'])
