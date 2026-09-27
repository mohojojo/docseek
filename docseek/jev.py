"""TypeSafe Jev: the decision layer's calibrated questions.

Jev answers typed questions (Choice / Noul) instead of generating text, so the crawler asks it the
judgements and keeps every action, URL and target in code.

Design choices:
  - Verdict bands (see judge.py): accepted, unsure, rejected, and unscored when Jev is unavailable.
  - The relevance question names sibling document types in its false criterion, and sends the row text
    of weakly named links (e.g. a bare "Download document").
  - The period is a Facet, not a filter: the question judges document type, and code reads the period
    off the link. Filtering on period would reject sites that overwrite one file per fund each month.
  - Page kind ranks the Frontier.
  - "Does this page still hide documents?" triggers an Escalation; BREAKER_FAILURES consecutive failures
    open the circuit breaker and the crawl falls back to the agent path.
  - A filtered listing gets one Choice per filter (its values, or keep) before any Escalation.

The cutoffs are calibrated for JEV_MODEL. Re-validate them before changing the pinned version.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .judge import (  # noqa: F401 - re-exported: the shared Verdict bands and link state live in docseek.judge
    ACCEPTED_AT, C_HIDDEN, C_OLDER, HIDDEN_DOCS_AT, MAX_FILTER_OPTIONS, Q_HIDDEN, Q_OLDER, REJECTED_BELOW,
    document_states, is_weakly_named, link_state, verdict_for,
)
from .profile import Profile, load_profile

logger = logging.getLogger(__name__)

JEV_URL = 'https://api.typesafe.ai/v1/systemone'
JEV_MODEL = 'jev-1.13.0'          # pinned: the cutoffs below are only valid for this version
# your TypeSafe price per million input tokens, for the reported cost; 0 reports no cost
JEV_PRICE_PER_MTOK = float(os.getenv('JEV_PRICE_PER_MTOK') or 0)

BATCH = 20                        # a batch of candidates in one request costs about the same as one
MAX_NEIGHBOURS = 10               # already-judged links re-scored alongside a batch, answers dropped
BREAKER_FAILURES = 3
RATE_LIMIT_ATTEMPTS = 6            # 429 is back-pressure, not an outage: wait it out
_PARALLEL_REQUESTS = 6

# --- relevance --------------------------------------------------------------------------------------
# The criteria come from the Profile (docseek/profiles/*.json): what counts, and which look-alikes do not.
Q_RELEVANCE = 'Is link {cid} a document that the goal asks for?'

# --- page kind: ranks the Frontier; the ids are fixed, the Profile describes them ---------------
KIND_TIER = {'fund_or_product': 1, 'document_listing': 1, 'category_or_overview': 2, 'other': 2,
             'news_or_article': 3, 'company_or_legal': 3}

# --- filter values ----------------------------------------------------------------------------
# A listing behind a filter shows its newest item; leaving the pick to the agent is expensive. Code finds
# the filters and their values, Jev picks one per filter, code sets it (the model chooses from a menu
# code built, it never invents an action).
Q_FILTER = ('Which value of filter {fid} ("{label}", now "{current}") makes the listing show the documents '
            'the goal asks for?')
C_FILTER_KEEP = ('Leave this filter as it is: none of its values narrows the listing to what the goal asks for, '
                 'or it already does.')


class JevUnavailable(Exception):
    """The circuit breaker is open: the crawl continues on the agent path."""


class JevClient:
    """Thread-safe TypeSafe client with a circuit breaker.

    `open` means no further requests are attempted for this crawl: callers treat every Candidate as
    `unscored` and the crawl falls back to the agent path.
    """

    name = 'jev'
    #: why the breaker is open, for the result and the UI ('no_credits', 'failures', 'no_key')
    unavailable_reason: str | None

    def __init__(self, api_key: str | None = None, *, profile: Profile | None = None, model: str = JEV_MODEL,
                 timeout: float = 30.0):
        self.api_key = api_key if api_key is not None else os.environ.get('TYPESAFE_API_KEY')
        self.profile = profile or load_profile()
        self.model = model
        self._client = httpx.Client(timeout=timeout, limits=httpx.Limits(max_connections=24))
        self._lock = threading.Lock()
        self.requests = self.input_tokens = self.output_tokens = self.failures = 0
        self._consecutive_failures = 0
        self.latencies_ms: list[float] = []
        self.open = not self.api_key
        self.unavailable_reason = None if self.api_key else 'no_key'

    @property
    def available(self) -> bool:
        return not self.open

    @property
    def cost_usd(self) -> float:
        return round(self.input_tokens * JEV_PRICE_PER_MTOK / 1e6, 6)

    def ask(self, state: dict, questions: dict) -> dict | None:
        """Return the answers, or None when the request failed (the caller records `unscored`)."""
        if self.open:
            return None
        attempt = rate_limited = 0
        while attempt < 3 and rate_limited < RATE_LIMIT_ATTEMPTS:
            started = time.perf_counter()
            try:
                resp = self._client.post(
                    JEV_URL, headers={'Authorization': f'Bearer {self.api_key}'},
                    json={'model': self.model, 'state': state, 'questions': questions},
                )
                if resp.status_code == 200:
                    body = resp.json()
                    with self._lock:
                        self.requests += 1
                        self.input_tokens += body['usage']['input_tokens']
                        self.output_tokens += body['usage']['output_tokens']
                        self.latencies_ms.append((time.perf_counter() - started) * 1000)
                        self._consecutive_failures = 0
                    return body['answers']
                if resp.status_code in (402, 401, 403):
                    # Billing or auth: retrying cannot help, and the agent fallback costs far more in
                    # tokens, so open at once and say why.
                    with self._lock:
                        self.open = True
                        self.unavailable_reason = 'no_credits' if resp.status_code == 402 else 'auth'
                        self.failures += 1
                    logger.error('[jev] %s - decision layer unavailable: %s', resp.status_code, resp.text[:200])
                    return None
                if resp.status_code == 429:
                    # Rate limited. Waiting is the right answer; counting it as a failure would open
                    # the breaker and send the rest of the crawl down the agent path.
                    rate_limited += 1
                    delay = float(resp.headers.get('retry-after') or min(30.0, 2.0 ** rate_limited))
                    logger.info('[jev] rate limited, waiting %.1fs (attempt %d)', delay, rate_limited)
                    time.sleep(delay)
                    continue
                if resp.status_code < 500:
                    logger.warning('[jev] %s %s', resp.status_code, resp.text[:200])
                    break
            except httpx.HTTPError as exc:
                logger.warning('[jev] request failed: %s', exc)
            time.sleep(1.0 * 2 ** attempt)
            attempt += 1
        with self._lock:
            self.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= BREAKER_FAILURES:
                self.open = True
                self.unavailable_reason = self.unavailable_reason or 'failures'
                logger.warning('[jev] circuit breaker open after %d consecutive failures', self._consecutive_failures)
        return None

    # --- the three questions ------------------------------------------------------------------
    def _batched(self, items: list, build) -> list:
        chunks = [items[i:i + BATCH] for i in range(0, len(items), BATCH)]

        def run(chunk):
            state, questions, read = build(chunk)
            return read(self.ask(state, questions))

        with ThreadPoolExecutor(_PARALLEL_REQUESTS) as pool:
            return [item for part in pool.map(run, chunks) for item in part]

    def relevance(self, goal: str, page: str, candidates: list[dict],
                  neighbours: list[dict] = ()) -> list[float | None]:
        """Relevance for every Candidate, whatever its Source. Order matches `candidates`.

        `neighbours` are links already judged on the same page, scored again alongside and their answers
        dropped. Links a filter revealed are often all named "Download": alone they score low, while next
        to the page's named report of the same kind they score clearly higher. Sent only as context,
        without a question of their own, they land unpredictably on either side of the cutoff.
        """
        def build(chunk):
            asked = list(chunk) + list(neighbours[:MAX_NEIGHBOURS])
            links = [link_state(f'L{i + 1}', c.get('name', ''), c['url'], c.get('context', ''),
                                c.get('section', ''), c.get('column', ''))
                     for i, c in enumerate(asked)]
            criteria = self.profile.relevance
            questions = {
                f'q{i + 1}': {'type': 'noul', 'instructions': Q_RELEVANCE.format(cid=f'L{i + 1}'),
                              'criteria': criteria}
                for i in range(len(asked))
            }
            return ({'goal': goal, 'page': page, 'links': links}, questions,
                    lambda answers: [answers[f'q{i + 1}']['noul'] if answers else None for i in range(len(chunk))])
        return self._batched(candidates, build)

    def page_kinds(self, goal: str, links: list[dict]) -> list[tuple[str, float]]:
        """Page kind + its probability for each link. Falls back to ('other', 0.0) when unavailable."""
        def build(chunk):
            # Icon-only links carry no text of their own, so send the row they sit in - the same rule
            # relevance uses. Without it such a link is just a URL: a report table that links each subject
            # through an icon would have none of those pages classified as worth visiting.
            states = [link_state(f'L{i + 1}', c.get('name', ''), c['url'], c.get('context', ''))
                      for i, c in enumerate(chunk)]
            kinds = self.profile.page_kinds
            questions = {
                f'q{i + 1}': {'type': 'choice', 'instructions': f'What kind of page does link L{i + 1} lead to?',
                              'criteria': kinds}
                for i in range(len(chunk))
            }

            def read(answers):
                if not answers:
                    return [('other', 0.0)] * len(chunk)
                out = []
                for i in range(len(chunk)):
                    a = answers[f'q{i + 1}']
                    out.append((a['choice'], a['probabilities'][a['choice']]))
                return out
            return ({'goal': goal, 'links': states}, questions, read)
        return self._batched(links, build)

    def filter_values(self, goal: str, page_state: dict, filters: list[dict]) -> dict[str, tuple[str, float]]:
        """The value to set per filter id, with its probability; filters Jev would keep are left out.
        Empty when Jev is unavailable, which sends the page on to the Escalation as before."""
        questions = {}
        for f in filters:
            criteria = {f'o{i}': f'Set the filter to "{o}"' for i, o in enumerate(f['options'][:MAX_FILTER_OPTIONS])}
            criteria['keep'] = C_FILTER_KEEP
            questions[f['id']] = {'type': 'choice', 'criteria': criteria, 'instructions': Q_FILTER.format(
                fid=f['id'], label=f.get('label') or 'unlabelled', current=f.get('current', ''))}
        answers = self.ask({'goal': goal, **page_state, 'filters': filters}, questions) if questions else None
        if not answers:
            return {}
        picks = {}
        for f in filters:
            answer = answers[f['id']]
            if answer['choice'] != 'keep':
                picks[f['id']] = (f['options'][int(answer['choice'][1:])], answer['probabilities'][answer['choice']])
        return picks

    def older_editions(self, goal: str, documents: list[dict]) -> list[float | None]:
        """For each document, the probability that a newer edition of it is also in the list. Every question sees
        the whole list: an edition is older only next to its successor."""
        states = document_states(documents)

        def build(indices):
            questions = {f'q{i + 1}': {'type': 'noul', 'instructions': Q_OLDER.format(did=f'D{i + 1}'), 'criteria': C_OLDER}
                         for i in indices}
            return ({'goal': goal, 'documents': states}, questions,
                    lambda answers: [answers[f'q{i + 1}']['noul'] if answers else None for i in indices])
        return self._batched(list(range(len(documents))), build)

    def hides_documents(self, goal: str, page_state: dict) -> float | None:
        answers = self.ask({'goal': goal, **page_state},
                           {'hidden': {'type': 'noul', 'instructions': Q_HIDDEN, 'criteria': C_HIDDEN}})
        return answers['hidden']['noul'] if answers else None
