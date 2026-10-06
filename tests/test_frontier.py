"""The Frontier's two policies. Pure code: no network, no browser."""
from docseek.frontier import Frontier
from docseek.jev_crawl import frontier_key


def fill(frontier: Frontier, links: list[tuple]) -> None:
    for url, kind, probability, group in links:
        frontier.add(url, kind=kind, probability=probability, depth=0, group=group, parent='seed')


def drain(frontier: Frontier, pays: dict[str, int], n: int) -> list[str]:
    order = []
    for _ in range(n):
        url, _, _ = frontier.pop()
        frontier.record(url, pays.get(url, 0))
        order.append(url)
    return order


class TestTierPolicy:
    def test_it_orders_exactly_like_the_shipped_key(self):
        links = [('a', 'news_or_article', 0.9, 'g'), ('b', 'fund_or_product', 0.8, 'g'),
                 ('c', 'fund_or_product', 0.99, 'g'), ('d', 'category_or_overview', 1.0, 'g')]
        frontier = Frontier('tier')
        fill(frontier, links)
        expected = sorted(links, key=lambda l: frontier_key(l[1], l[2], 0, False, links.index(l)))
        assert drain(frontier, {}, 4) == [l[0] for l in expected]


class TestBanditPolicy:
    # a confident menu of empty fund overviews, and a report table that pays
    MENU = [(f'menu{i}', 'fund_or_product', 0.99, 'nav/ul/li/a') for i in range(6)]
    TABLE = [(f'table{i}', 'fund_or_product', 0.88, 'table/tr/td/a') for i in range(6)]
    PAYS = {f'table{i}': 30 for i in range(6)}

    def test_a_paying_group_is_reached_and_then_kept(self):
        frontier = Frontier('bandit')
        fill(frontier, self.MENU + self.TABLE)
        order = drain(frontier, self.PAYS, 8)
        assert sum(url.startswith('table') for url in order) >= 6      # tier order finds 2 in 8

    def test_the_shipped_order_does_not(self):
        frontier = Frontier('tier')
        fill(frontier, self.MENU + self.TABLE)
        assert sum(url.startswith('table') for url in drain(frontier, self.PAYS, 8)) == 2

    def test_one_empty_page_does_not_bury_its_siblings(self):
        # the "demote" case: two empty pages sank the one that mattered
        listings = [(f'doc{i}', 'document_listing', 0.9, 'menu/a') for i in range(4)]
        others = [(f'other{i}', 'category_or_overview', 0.9, 'x/a') for i in range(4)]
        frontier = Frontier('bandit')
        fill(frontier, listings + others)
        assert drain(frontier, {'doc3': 6}, 4) == ['doc0', 'doc1', 'doc2', 'doc3']   # tier 1 before tier 2

    def test_a_page_that_opens_new_groups_is_credited(self):
        frontier = Frontier('bandit')
        fill(frontier, [('hub', 'category_or_overview', 0.9, 'cards/a')])
        hub, _, _ = frontier.pop()
        frontier.add('fund', kind='fund_or_product', probability=0.9, depth=0, group='grid/a', parent=hub)
        frontier.add('legal', kind='company_or_legal', probability=0.9, depth=0, group='footer/a', parent=hub)
        frontier.record(hub, 0)
        assert frontier.groups['cards/a'].reward > 0          # the fund grid counts, the footer does not
        assert frontier._opened_by == {}

    def test_links_without_a_tag_path_group_by_url_template(self):
        frontier = Frontier('bandit')
        frontier.add('https://site-hu.example/alapok/a', kind='other', probability=0.5, depth=0)
        frontier.add('https://site-hu.example/alapok/b', kind='other', probability=0.5, depth=0)
        assert list(frontier.groups) == ['url:/alapok/*']


class TestRescuePolicy:
    MENU = [(f'menu{i}', 'fund_or_product', 0.99, 'nav/ul/li/a') for i in range(12)]
    TABLE = [(f'table{i}', 'fund_or_product', 0.88, 'table/tr/td/a') for i in range(6)]

    def test_a_crawl_that_pays_never_leaves_the_shipped_order(self):
        links = self.MENU + self.TABLE
        shipped, rescue = Frontier('tier'), Frontier('rescue')
        fill(shipped, links)
        fill(rescue, links)
        pays = {url: 1 for url, *_ in links}
        assert drain(rescue, pays, 18) == drain(shipped, pays, 18)
        assert rescue.rescued_at is None

    def test_a_dry_run_hands_over_to_the_bandit(self):
        frontier = Frontier('rescue', dry_pages=3)
        fill(frontier, self.MENU + self.TABLE)
        order = drain(frontier, {f'table{i}': 5 for i in range(6)}, 10)
        assert frontier.rescued_at == 3
        assert sum(url.startswith('table') for url in order) == 6      # the shipped order finds none in 10

    def test_an_unvisited_tier1_group_keeps_the_crawl_alive(self):
        frontier = Frontier('rescue')
        fill(frontier, self.MENU[:2] + self.TABLE[:1])
        drain(frontier, {}, 1)
        assert frontier.has_untried_tier1()
        drain(frontier, {}, 2)
        assert not frontier.has_untried_tier1()
        assert not Frontier('tier').has_untried_tier1()


class TestDefault:
    def test_the_crawl_and_the_service_default_to_the_measured_order(self):
        # rescue is a recall trade: a request has to ask for it
        import inspect

        from docseek.jev_crawl import jev_crawl
        from docseek.server import DiscoverRequest
        assert inspect.signature(jev_crawl).parameters['frontier_policy'].default == 'tier'
        assert DiscoverRequest(url='https://site-hu.example', goal='g').frontier_policy == 'tier'


class TestGoalYear:
    def test_the_years_a_goal_names_including_ranges(self):
        from docseek.jev_crawl import goal_years
        assert goal_years('Finde alle Ausgaben des Amtsblatts aus dem Jahr 2025') == {'2025'}
        assert goal_years('Find the annual reports from 2021 to 2025') == {'2021', '2022', '2023', '2024', '2025'}
        assert goal_years('Voranschläge der Jahre 2021 bis 2023') == {'2021', '2022', '2023'}
        assert goal_years('Informes Anuales de 2020 a 2022') == {'2020', '2021', '2022'}
        assert goal_years('Find every annual report') == frozenset()

    def test_a_link_names_the_year_in_its_text_or_url_not_inside_a_longer_number(self):
        from docseek.jev_crawl import names_goal_year
        years = frozenset({'2025'})
        assert names_goal_year({'name': 'Amtsblatt 2025', 'url': 'https://s.example/a'}, years)
        assert names_goal_year({'name': 'Archiv', 'url': 'https://s.example/amtsblatt-2025/'}, years)
        assert not names_goal_year({'name': 'Amtsblatt', 'url': 'https://s.example/id/202512345'}, years)
        assert not names_goal_year({'name': 'Amtsblatt 2025', 'url': 'https://s.example/a'}, frozenset())

    def test_the_goal_years_archive_goes_before_the_current_listing(self):
        current = frontier_key('document_listing', 0.95, 1, False, 0)
        archive = frontier_key('category_or_overview', 0.7, 1, False, 1, goal_year=True)
        news = frontier_key('news_or_article', 0.9, 1, False, 2, goal_year=True)
        assert sorted([current, news, archive]) == [archive, current, news]   # news is never promoted

    def test_the_frontier_knows_a_goal_year_page_is_still_waiting(self):
        f = Frontier()
        f.add('https://s.example/amtsblatt-2026/', kind='document_listing', probability=0.9, depth=1)
        assert not f.has_goal_year_page()
        f.add('https://s.example/amtsblatt-2025/', kind='category_or_overview', probability=0.6, depth=1,
              goal_year=True)
        assert f.has_goal_year_page()
        assert f.pop()[0] == 'https://s.example/amtsblatt-2025/'
        assert not f.has_goal_year_page()


def test_a_news_page_naming_the_year_does_not_hold_the_crawl_open():
    f = Frontier()
    f.add('https://s.example/news/2025-results', kind='news_or_article', probability=0.9, depth=1, goal_year=True)
    assert not f.has_goal_year_page()


class TestRevealedPages:
    def test_pages_a_filter_revealed_come_before_the_unfiltered_listings_own(self):
        frontier = Frontier('tier')
        frontier.add('https://site-lv.example/?id=1', kind='fund_or_product', probability=0.9, depth=1, group='g')
        frontier.add('https://site-lv.example/?id=2', kind='fund_or_product', probability=0.6, depth=1, group='g',
                     revealed=True)
        assert frontier.pop()[0] == 'https://site-lv.example/?id=2'

    def test_they_come_before_a_page_that_merely_names_the_goals_year_whatever_they_look_like(self):
        frontier = Frontier('tier')
        frontier.add('https://site-lv.example/?id=1', kind='fund_or_product', probability=0.9, depth=1, group='g',
                     goal_year=True)
        frontier.add('https://site-lv.example/?id=2', kind='news_or_article', probability=0.6, depth=1, group='g',
                     revealed=True)
        assert frontier.pop()[0] == 'https://site-lv.example/?id=2'


class TestContentFirst:
    def test_a_pages_own_links_come_before_its_menu_links_in_the_same_tier(self):
        frontier = Frontier('tier')
        frontier.add('https://site-hu.example/menu-fund', kind='fund_or_product', probability=1.0, depth=1, group='g',
                     chrome=True)
        frontier.add('https://site-hu.example/table-archive', kind='fund_or_product', probability=0.5, depth=1, group='g')
        assert frontier.pop()[0] == 'https://site-hu.example/table-archive'

    def test_but_not_before_a_better_tier(self):
        frontier = Frontier('tier')
        frontier.add('https://site-hu.example/menu-reports', kind='document_listing', probability=0.9, depth=1, group='g',
                     chrome=True)
        frontier.add('https://site-hu.example/news-item', kind='news_or_article', probability=0.9, depth=1, group='g')
        assert frontier.pop()[0] == 'https://site-hu.example/menu-reports'

    def test_the_old_order_is_still_there_for_the_replay(self):
        frontier = Frontier('tier', content_first=False)
        frontier.add('https://site-hu.example/menu-fund', kind='fund_or_product', probability=1.0, depth=1, group='g',
                     chrome=True)
        frontier.add('https://site-hu.example/table-archive', kind='fund_or_product', probability=0.5, depth=1, group='g')
        assert frontier.pop()[0] == 'https://site-hu.example/menu-fund'
