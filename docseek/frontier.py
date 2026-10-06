"""The Jev crawl's Frontier: which page to visit next.

Two policies behind one interface, so the eval can replay either over a recorded site:

  tier    the default order: goal language, tier, unseen path, then P(kind), depth, discovery order.
  bandit  the same outer order, but inside it links are grouped by where they sat on the page and the
          next group is chosen by what its siblings have paid so far.
  rescue  tier until the crawl has gone `dry_pages` pages without an accepted document, bandit from
          then on. Run from the first page, the bandit wasted pages and lost documents on sites the
          tier order already handled well.

`tier` is the default. `rescue` is opt-in per request: on a site where the tier order stalled it found
many more accepted documents and stopped the crawl quitting early, but on other sites it occasionally
lost a ground-truth document, or crawled more pages for the same documents.

Why a bandit. P(kind) is the model's certainty that a link is what it looks like, not evidence that the
page holds anything: on one site confident-but-empty pages outranked most of the report pages
that each held a full history. Feeding yield back by URL prefix was tried and failed both ways:
promoting what paid locked the crawl into the first corner that paid, demoting what did not buried
another site's one gateway page. Both are the textbook failures of a bandit without optimism, so:

  - an untried group starts from an optimistic prior and is never demoted; one empty page halves its
    mean, it does not send it to the back
  - an exploration bonus keeps every group with unvisited links alive (a "sleeping" bandit: Kleinberg
    et al. 2010, used for this task by Gauquier, Manolescu and Senellart, EDBT 2026)
  - the group is the link's DOM tag path, the key that paper found to predict similar content. On
    a report table it puts all the report links in one group; a URL template splits them several ways
  - P(kind) stays, as the tiebreak inside a group
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .jev import KIND_TIER

PRIOR_MEAN = 0.7          # log1p(1): an untried group is assumed to pay one document. Set above what a
                          # paying group really pays per page, the crawl explores and never exploits
PRIOR_PULLS = 1
EXPLORE = 0.3             # UCB weight. The literature's 2*sqrt(2) would spend a 40-page budget exploring
NEW_GROUP_CREDIT = 0.5    # a page that opens new tier-1 groups paid too, even with no document on it
DRY_PAGES = 7             # replayed on recorded sites: a shorter dry run already cost documents


def frontier_key(kind: str, probability: float, depth: int, other_language: bool, order: int,
                 path_seen: bool = False, goal_year: bool = False, revealed: bool = False,
                 chrome: bool = False, listing_first: bool = False) -> tuple:
    """Tier ordering: goal language, then tier, then the goal's year, then unseen paths, then P(kind), depth,
    discovery order.

    `path_seen` demotes another query-variant of a path already queued. A Liferay listing offers its
    own pagination as `?…_cur=1…10` and `?…_delta=8…60`: such variants once filled most of a
    crawl's page budget, all yielding nothing, while the fund pages waited.

    `goal_year`: the link names a year the goal names. A site lists the current year and links each earlier year's
    archive, and a crawl asked for 2025 spent its budget on the 2026 listing while the `…-2025` archive waited. Such
    a link goes first within its tier, and a category page for that year counts as a listing; a news or legal page
    naming the year is not promoted.

    `revealed`: the link appeared when the crawl set a listing's filter for the goal. Those pages are the listing the
    goal asked for, whatever kind each looks like (a filing titled like a news item is one), and go before everything
    else: the unfiltered listing's newest items name this year too, and ranked level with them a 10-page crawl never
    reached one.

    `chrome`: the link sits in the site's menus, header, footer or a sidebar and nowhere in the page's content (a
    paginator is content). Within a tier the content's links come first (`content_first`, the default): a hub page's
    table links each subject's report archive through an icon, while the header menu names every fund in words, and
    the named menu pages took the whole budget. Measured live 2026-10-06: that site went from 1 to 9 of 17 reports in
    10 pages (two runs) and from 3 to 17 in 40; a company whose reports page is reached only through its menu was
    unchanged (7 of 8 in 10 pages either way); a regulator whose menu leads to a notices listing holding a fifth of
    its hits scored 0.30, 0.55 and 0.51 against 0.51, 0.53 and 0.53, so `has_menu_listing` holds the crawl open
    while such a listing waits. Six replayed recordings moved nothing.

    `listing_first`: within the first tier, document listings before single-subject pages. Replayed only: it took one
    results archive from 2 to 11 accepted in 10 pages but cost the hub site above every report, so it is not the
    default. Both are kept for eval.frontier_replay to weigh on new recordings.
    """
    tier = KIND_TIER.get(kind, 2)
    if revealed or (goal_year and tier == 2):
        tier = 1
    listing = 0 if not listing_first or kind == 'document_listing' else 1
    # a menu names every subject; a content link to one subject's page is the page's own pointer. A menu's link to a
    # document listing keeps its rank: that is where sites put their document sections, and one regulator's
    # notices listing, reached only from the menu, held a fifth of its decisions.
    return (1 if other_language else 0, tier, 0 if revealed else 1 if goal_year else 2, listing, 1 if chrome else 0,
            1 if path_seen else 0, -probability, depth, order)


def url_template(url: str) -> str:
    """Group key for a link with no tag path (sitemap, escalation): its path with the last segment open."""
    segments = [s for s in urlparse(url).path.split('/') if s]
    return 'url:/' + '/'.join(segments[:-1] + ['*'])


@dataclass
class _Item:
    url: str
    depth: int
    kind: str
    outer: tuple          # (other language, tier, goal's year, path already queued): never crossed by the bandit
    inner: tuple          # (-P(kind), depth, discovery order)
    group: str


@dataclass
class _Group:
    pulls: int = 0
    reward: float = 0.0
    picked: int = 0       # pops, including pages still loading: two are in flight at a time


@dataclass
class Frontier:
    policy: str = 'tier'
    prior_mean: float = PRIOR_MEAN
    explore: float = EXPLORE
    new_group_credit: float = NEW_GROUP_CREDIT
    dry_pages: int = DRY_PAGES
    items: dict[str, _Item] = field(default_factory=dict)
    groups: dict[str, _Group] = field(default_factory=dict)
    _group_of: dict[str, str] = field(default_factory=dict)
    _opened_by: dict[str, int] = field(default_factory=dict)
    _order: int = 0
    _pulls: int = 0
    _dry: int = 0
    rescued_at: int | None = None     # pages visited when `rescue` left the shipped order
    content_first: bool = True        # a page's content links before its menu links (see frontier_key)
    listing_first: bool = False       # within the first tier, document listings before single-subject pages

    def __len__(self) -> int:
        return len(self.items)

    def add(self, url: str, *, kind: str, probability: float, depth: int, other_language: bool = False,
            path_seen: bool = False, group: str = '', parent: str | None = None, goal_year: bool = False,
            revealed: bool = False, chrome: bool = False) -> None:
        group = group or url_template(url)
        key = frontier_key(kind, probability, depth, other_language, self._order + 1, path_seen, goal_year, revealed,
                           chrome and self.content_first, self.listing_first)
        tier = key[1]
        if group not in self.groups:
            self.groups[group] = _Group()
            if parent and tier == 1:
                self._opened_by[parent] = self._opened_by.get(parent, 0) + 1
        self._order += 1
        self.items[url] = _Item(url, depth, kind, key[:6], key[6:], group)
        self._group_of[url] = group

    def add_seed(self, url: str) -> None:
        self.groups.setdefault('seed', _Group())
        self.items[url] = _Item(url, 0, 'seed', (0, 0, 0, 0, 0, 0), (-1.0, 0, 0), 'seed')
        self._group_of[url] = 'seed'

    def _score(self, group: str) -> float:
        g = self.groups[group]
        mean = (self.prior_mean * PRIOR_PULLS + g.reward) / (PRIOR_PULLS + g.pulls)
        return mean + self.explore * math.sqrt(math.log(self._pulls + 2) / (PRIOR_PULLS + g.pulls))

    def pop(self) -> tuple[str, int, str] | None:
        if not self.items:
            return None
        if self.policy == 'rescue' and self.rescued_at is None and self._dry >= self.dry_pages:
            self.rescued_at = self._pulls
        if self.policy == 'tier' or (self.policy == 'rescue' and self.rescued_at is None):
            best = min(self.items.values(), key=lambda i: (i.outer, i.inner))
        else:
            outer = min(i.outer for i in self.items.values())
            awake = [i for i in self.items.values() if i.outer == outer]
            first = {}                                   # each awake group's best link
            for item in sorted(awake, key=lambda i: i.inner, reverse=True):
                first[item.group] = item
            best = max(first.values(), key=lambda i: (self._score(i.group), [-x for x in i.inner]))
        self.groups[best.group].picked += 1
        del self.items[best.url]
        return best.url, best.depth, best.kind

    def has_untried_tier1(self) -> bool:
        """True while a tier-1 group nobody has visited is still queued: the crawl has not looked everywhere."""
        return self.policy != 'tier' and any(
            i.outer[1] <= 1 and not self.groups[i.group].picked for i in self.items.values())

    def has_goal_year_page(self) -> bool:
        """True while a listing naming the goal's year, or a page a filter set for the goal revealed, is still
        queued: what the goal asks for is unvisited. A news or legal page that merely names the year does not
        hold the crawl open."""
        return any(i.outer[2] <= 1 and i.outer[1] <= 1 for i in self.items.values())

    def has_menu_listing(self) -> bool:
        """True while a document listing the site's menu links to is still queued. The content's links come first,
        so a crawl that found nothing in them has not yet looked where the site files its documents."""
        return any(i.kind == 'document_listing' and i.outer[4] == 1 for i in self.items.values())

    def record(self, url: str, accepted: int) -> None:
        """What visiting `url` paid: documents accepted for the first time, plus new tier-1 groups."""
        group = self.groups[self._group_of.get(url, 'seed')]
        group.pulls += 1
        group.reward += math.log1p(accepted + self.new_group_credit * self._opened_by.pop(url, 0))
        self._pulls += 1
        self._dry = 0 if accepted else self._dry + 1
