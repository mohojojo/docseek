"""Periods and series: which document is the newest of its kind, read by code, never by a model.

  period_of(text)      '2026-03' | '2026-Q1' | '2026-H1' | '2026' | None, from a document's name and URL
  series_key(url, name) what stays of a document's identity once its period is taken out
  order_of(url, name)  what orders the documents of one series, with no interpretation of what the numbers mean
  mark_latest(docs)    sets each document's `series` and `latest_in_series` facets; returns the superseded ones

The relevance judge scores a document's type, never its period (judging the period rejected every target on
sites that overwrite one file per month), so "the latest report of each fund" is answered here, and every series
keeps its newest - however old it is.

Reading dates out of names will never be right for every site ('2211', 'eb202508', 'SAN-2025-12', 'Heft 3' each
mean something different), so the two jobs are kept apart. `period` is a best-effort label. The latest filter does
not use it: the documents of one series share a naming pattern, so they are ordered by the numbers in it without
deciding what those numbers mean, and a series whose documents cannot be ordered cleanly keeps all of them.

Periods are read in the languages docseek detects goals in (en, de, hu, fr, es, it). A year next to a slash is
not read: '450-7/2026' is a case number and '/2016/05/' an upload folder, not a period.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

MONTHS = {
    1: ('january', 'januar', 'jänner', 'január', 'janvier', 'enero', 'gennaio', 'jan', 'janv', 'ene', 'gen'),
    2: ('february', 'februar', 'február', 'février', 'fevrier', 'febrero', 'febbraio', 'feb', 'febr', 'févr', 'fevr'),
    3: ('march', 'märz', 'maerz', 'március', 'marcius', 'mars', 'marzo', 'mar', 'mär', 'márc', 'marc'),
    4: ('april', 'április', 'aprilis', 'avril', 'abril', 'aprile', 'apr', 'ápr', 'avr', 'abr'),
    5: ('may', 'mai', 'május', 'majus', 'mayo', 'maggio', 'máj', 'maj', 'mag'),
    6: ('june', 'juni', 'június', 'junius', 'juin', 'junio', 'giugno', 'jun', 'jún', 'giu'),
    7: ('july', 'juli', 'július', 'julius', 'juillet', 'julio', 'luglio', 'jul', 'júl', 'juil', 'lug'),
    8: ('august', 'augusztus', 'août', 'aout', 'agosto', 'aug', 'ago'),
    9: ('september', 'szeptember', 'septembre', 'septiembre', 'setiembre', 'settembre', 'sep', 'sept', 'szept', 'set'),
    10: ('october', 'oktober', 'október', 'octobre', 'octubre', 'ottobre', 'oct', 'okt', 'ott'),
    11: ('november', 'novembre', 'noviembre', 'nov'),
    12: ('december', 'dezember', 'décembre', 'decembre', 'diciembre', 'dicembre', 'dec', 'dez', 'déc', 'dic'),
}
_MONTH_WORD = {w: m for m, words in MONTHS.items() for w in words}
_WORDS = '|'.join(sorted(map(re.escape, _MONTH_WORD), key=len, reverse=True))
_L = 'a-zà-öø-ÿőű'                                   # letters, so a month word is a whole word or a word's start
_Y = r'(20\d{2})'
_NO_YEAR_EDGE_BEFORE, _NO_YEAR_EDGE_AFTER = r'(?<![\d/])', r'(?![\d/])'
_QUARTER_WORDS = r'(?:quarter|quartal|negyedév|trimestre)'
_HALF_WORDS = r'(?:half|halbjahr|félév|semestre|semester)'
_ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}


def _n(value: str) -> int:
    return _ROMAN.get(value) or int(value)


# groups of (pattern, period builder) in priority order: the most specific reading wins; within a group the
# tightest match does ('2025 - March 2026': March belongs to 2026)
_GROUPS = [
    # 20260315, 2026-03, 2026_03, 202603
    (re.compile(rf'(?<!\d){_Y}(0[1-9]|1[0-2])(?:[0-2]\d|3[01])(?!\d)'), lambda m: f'{m[1]}-{m[2]}'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[-_. ]?(0[1-9]|1[0-2])(?!\d)'), lambda m: f'{m[1]}-{m[2]}'),
    # 15.03.2026, 03/2026, 3.2026 - a day before it, or nothing that could make it a case number
    (re.compile(rf'(?:^|(?<=[\s_(])|(?<=\d[./]))(0?[1-9]|1[0-2])[./]{_Y}{_NO_YEAR_EDGE_AFTER}'),
     lambda m: f'{m[2]}-{int(m[1]):02d}'),
    # 2026. március, 2026-march, March 2026, Oktober_K501672_2021
    [(re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[\W_]{{0,3}}({_WORDS})[{_L}]*', re.IGNORECASE),
      lambda m: f'{m[1]}-{_MONTH_WORD[m[2].lower()]:02d}'),
     (re.compile(rf'(?<![{_L}])({_WORDS})[{_L}]*[\W_]{{1,3}}(?:[^\W_]{{1,20}}[\W_]{{1,3}})?{_Y}{_NO_YEAR_EDGE_AFTER}',
                 re.IGNORECASE), lambda m: f'{m[2]}-{_MONTH_WORD[m[1].lower()]:02d}')],
    # Q1 2026, 2026-Q1, 1. Quartal 2026, I. negyedév 2026, 2026 1er trimestre
    (re.compile(rf'(?<![a-z0-9])q([1-4])[\W_]{{0,3}}{_Y}{_NO_YEAR_EDGE_AFTER}', re.IGNORECASE),
     lambda m: f'{m[2]}-Q{m[1]}'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[\W_]{{0,3}}q([1-4])(?![a-z0-9])', re.IGNORECASE),
     lambda m: f'{m[1]}-Q{m[2]}'),
    (re.compile(rf'(?<![a-z0-9])([1-4]|iv|i{{1,3}})\.?[ ]?(?:st|nd|rd|th|er|e)?[\W_]{{0,2}}{_QUARTER_WORDS}[{_L}]*'
                rf'[\W_]{{0,3}}{_Y}{_NO_YEAR_EDGE_AFTER}', re.IGNORECASE),
     lambda m: f'{m[2]}-Q{_n(m[1].lower())}'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[\W_]{{0,3}}([1-4]|iv|i{{1,3}})\.?[ ]?(?:st|nd|rd|th|er|e)?[\W_]{{0,2}}'
                rf'{_QUARTER_WORDS}', re.IGNORECASE), lambda m: f'{m[1]}-Q{_n(m[2].lower())}'),
    # H1 2025, 2025-H2, 1. Halbjahr 2025, 2025 1st half, half-year report 2025, 2025. féléves
    (re.compile(rf'(?<![a-z0-9])h([12])[\W_]{{0,3}}{_Y}{_NO_YEAR_EDGE_AFTER}', re.IGNORECASE),
     lambda m: f'{m[2]}-H{m[1]}'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[\W_]{{0,3}}h([12])(?![a-z0-9])', re.IGNORECASE),
     lambda m: f'{m[1]}-H{m[2]}'),
    (re.compile(rf'(?<![a-z0-9])([12])\.?[ ]?(?:st|nd|er|e)?[\W_]{{0,2}}{_HALF_WORDS}[{_L}]*[\W_]{{0,3}}{_Y}'
                rf'{_NO_YEAR_EDGE_AFTER}', re.IGNORECASE), lambda m: f'{m[2]}-H{m[1]}'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[\W_]{{0,3}}([12])\.?[ ]?(?:st|nd|er|e)?[\W_]{{0,2}}{_HALF_WORDS}',
                re.IGNORECASE), lambda m: f'{m[1]}-H{m[2]}'),
    (re.compile(rf'(?:half[- ]?year|halbjahres|félév)[{_L}]*[\W_]{{0,3}}(?:[{_L}]{{1,12}}[\W_]{{1,3}})?{_Y}'
                rf'{_NO_YEAR_EDGE_AFTER}', re.IGNORECASE), lambda m: f'{m[1]}-H1'),
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}\.?[\W_]{{0,3}}(?:half[- ]?year|halbjahres|félév)', re.IGNORECASE),
     lambda m: f'{m[1]}-H1'),
    # 2026-1, 2026.8 (a one-digit month, only with - or .)
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}[-.]([1-9])(?![\w.])'), lambda m: f'{m[1]}-{int(m[2]):02d}'),
    # a bare year
    (re.compile(rf'{_NO_YEAR_EDGE_BEFORE}{_Y}{_NO_YEAR_EDGE_AFTER}'), lambda m: m[1]),
]


_READERS = [reader for group in _GROUPS for reader in (group if isinstance(group, list) else [group])]


def period_of(text: str) -> str | None:
    """The period a document covers, read from its name and URL: '2026-03', '2026-Q1', '2026-H1' or '2026'."""
    text = unquote(text or '')
    for group in _GROUPS:
        found = [(m, build) for pattern, build in (group if isinstance(group, list) else [group])
                 if (m := pattern.search(text))]
        if found:
            m, build = min(found, key=lambda f: len(f[0][0]))
            return build(m)
    return None


_GENERIC_NAMES = {'download', 'downloads', 'file', 'files', 'document', 'documents', 'doc', 'view', 'get', 'getfile',
                  'attachment', 'index', 'default', 'pdf', 'show', 'open', 'dl', 'fetch', 'content', 'media', 'asset'}
_SCRIPT_EXTENSIONS = ('aspx', 'php', 'asp', 'jsp', 'ashx', 'cgi')
# what varies between the documents of one series: month words and numbers (a q/h prefix marks a quarter/half)
_TOKEN = re.compile(rf'(?<![{_L}])({_WORDS})[{_L}]*|(?:(?<![a-z0-9])([qh]))?(\d+)', re.IGNORECASE)


def _strip_periods(text: str) -> str:
    for pattern, _ in _READERS:
        text = pattern.sub(' ', text)
    return text


def _key(text: str) -> str:
    return re.sub(r'[\W_]+', '-', text.lower()).strip('-')


def _identity(url: str, name: str) -> tuple[str, str]:
    """(text, extension) that identifies a document: its file name, or its name when the file name says nothing
    (getfile.aspx?id=12, a hash, 'download'). An id after the file name (/documents/1/report.pdf/<uuid>) is skipped."""
    segments = [unquote(seg) for seg in urlparse(url).path.split('/') if seg]
    filename = next((seg for seg in reversed(segments) if re.search(r'\.[A-Za-z0-9]{2,5}$', seg)),
                    segments[-1] if segments else '')
    stem, dot, ext = filename.rpartition('.')
    stem, ext = (stem, ext.lower()) if dot and len(ext) <= 5 else (filename, '')
    key = _key(_strip_periods(stem.lower()))
    letters = re.sub(r'[^a-zà-ÿőű]', '', key)
    opaque = (len(letters) < 3 or key in _GENERIC_NAMES or ext in _SCRIPT_EXTENSIONS
              or bool(re.fullmatch(r'[0-9a-f-]{16,}', key)))
    text = name if opaque and name else stem
    return text.lower(), '' if ext in _SCRIPT_EXTENSIONS else ext


def series_key(url: str, name: str = '') -> str:
    """The document's identity with its period taken out: one key for every month of one fund's factsheet."""
    text, ext = _identity(url, name)
    key = _key(_strip_periods(text))
    return f'{key}.{ext}' if ext else key


def order_of(url: str, name: str = '') -> tuple[tuple, tuple]:
    """(shape, key) that orders the documents of one series: the numbers and month words of its identity, four-digit
    years first. Nothing is interpreted - whether '08' is August or issue 8 does not matter, as long as every
    document of the series writes it the same way, which `shape` checks."""
    text, _ = _identity(url, name)
    years, rest = [], []
    for m in _TOKEN.finditer(text):
        if m[1]:
            rest.append(('m', _MONTH_WORD[m[1].lower()]))
            continue
        value = int(m[3])
        kind = m[2] or ('y' if len(m[3]) == 4 and 1900 <= value <= 2099 else 'n')
        (years if kind == 'y' else rest).append((kind, value))
    tokens = years + rest
    return tuple(kind for kind, _ in tokens), tuple(value for _, value in tokens)


def mark_latest(documents: list) -> list:
    """Set `series`, `latest_in_series` (and a best-effort `period`) on each AgenticDownload; return the superseded.

    Within a series, documents are ordered by order_of. When in doubt, nothing is dropped: latest_in_series is None
    - and the document kept - when its series mixes formats (2026-Q1 beside 2026-03), when it has nothing to order
    by, or when it was rejected. Only accepted, unsure or unscored documents can supersede another."""
    series: dict[str, list] = {}
    for doc in documents:
        doc.period = doc.period or period_of(f'{doc.name} {doc.url}')
        doc.series = series_key(doc.url, doc.name)
        doc.latest_in_series = None
        if doc.verdict != 'rejected':
            series.setdefault(doc.series, []).append((doc, *order_of(doc.url, doc.name)))
    superseded = []
    for members in series.values():
        shapes = {shape for _, shape, _ in members}
        if len(shapes) != 1 or not next(iter(shapes)):
            continue                                       # mixed formats or nothing to order by: keep them all
        newest = max(key for _, _, key in members)
        for doc, _, key in members:
            doc.latest_in_series = key == newest
            if not doc.latest_in_series:
                superseded.append(doc)
    return superseded


MAX_JUDGED_GROUPS = 25        # doubtful groups asked about per result; past it, the rest is kept
MAX_GROUP_SIZE = 40           # a larger group is kept whole: too many editions to show in one question


def _skeleton(doc) -> str:
    """The identity with every number and month word taken out: 'etalon-2211' and 'etalon-2212' meet, and so do
    'cm4' and 'cm5' - which is why a group formed this way is only a question, never an answer."""
    text, ext = _identity(doc.url, doc.name)
    key = _key(_TOKEN.sub(' ', text))
    return f'{key}.{ext}' if ext else key


def refine_latest(documents: list, goal: str, judge, superseded_at: float | None = None) -> list:
    """Ask the relevance judge about the groups mark_latest could not settle, and return the documents it is
    sure are older editions. A group is doubtful when its documents look alike once their numbers are taken out
    but code did not order them all as one series ('ETALON-2211' beside 'ETALON-2212', '2026-Q1' beside
    '2026-03', 'cm4' beside 'cm5'). Only a probability of at least SUPERSEDED_AT drops a document; anything less,
    or no answer, keeps it."""
    from .judge import SUPERSEDED_AT
    threshold = SUPERSEDED_AT if superseded_at is None else superseded_at
    groups: dict[str, list] = {}
    for doc in documents:
        if doc.verdict != 'rejected' and doc.latest_in_series is not False:
            groups.setdefault(_skeleton(doc), []).append(doc)
    doubtful = [members for members in groups.values() if 2 <= len(members) <= MAX_GROUP_SIZE
                and not (len({d.series for d in members}) == 1 and all(d.latest_in_series for d in members))]
    superseded = []
    for members in doubtful[:MAX_JUDGED_GROUPS]:
        shown = [{'name': d.name, 'url': d.url} for d in members]
        # First the group as a whole: when the judge is sure it is one document's editions and sure which is the
        # newest, the rest go. Otherwise (several documents, or unsure) each is asked about on its own.
        same, newest, sure = judge.edition_group(goal, shown)
        if same is not None and same >= threshold and newest is not None and sure is not None and sure >= threshold:
            older = [doc for i, doc in enumerate(members) if i != newest]
        else:
            older = [doc for doc, p in zip(members, judge.older_editions(goal, shown))
                     if p is not None and p >= threshold]
        for doc in older:
            doc.latest_in_series = False
            superseded.append(doc)
    return superseded


def apply_latest(result, latest: bool, include_rejected: bool = False, judge=None):
    """Mark every document of an AgenticCrawlResult; with `latest`, drop the superseded ones from `downloads` and
    count them in `superseded_count`. A caller that asked for rejected rows keeps the superseded ones too, marked.

    `judge` - a RelevanceJudge, or a callable returning one (or None) - settles the groups code could not order;
    it is consulted only with `latest`. Without one, those groups are kept whole."""
    superseded = mark_latest(result.downloads)
    if latest and judge is not None:
        judge = judge() if callable(judge) and not hasattr(judge, 'older_editions') else judge
        if judge is not None:
            superseded += refine_latest(result.downloads, result.goal, judge)
    if latest:
        result.superseded_count = len(superseded)
        if not include_rejected:
            gone = {id(d) for d in superseded}
            result.downloads = [d for d in result.downloads if id(d) not in gone]
    return result
