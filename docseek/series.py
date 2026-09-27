"""Periods and series: which document is the newest of its kind, read by code, never by a model.

  period_of(text)      '2026-03' | '2026-Q1' | '2026-H1' | '2026' | None, from a document's name and URL
  period_end(period)   a comparable month index: a quarter, half or year counts as its last month
  series_key(url, name) what stays of a document's identity once its period is taken out
  mark_latest(docs)    sets each document's `series` and `latest_in_series` facets; returns the superseded ones

The relevance judge scores a document's type, never its period (judging the period rejected every target on
sites that overwrite one file per month), so "the latest report of each fund" is answered here: the documents
of one series are compared by period, and every series keeps its newest - however old it is.

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


def period_end(period: str) -> int:
    """Months since year 0 at the period's end: 2026-Q1 and 2026-03 compare equal, 2026 is its December."""
    year, _, part = period.partition('-')
    if not part:
        month = 12
    elif part[0] == 'Q':
        month = 3 * int(part[1])
    elif part[0] == 'H':
        month = 6 * int(part[1])
    else:
        month = int(part)
    return int(year) * 12 + month


_GENERIC_NAMES = {'download', 'downloads', 'file', 'files', 'document', 'documents', 'doc', 'view', 'get', 'getfile',
                  'attachment', 'index', 'default', 'pdf', 'show', 'open', 'dl', 'fetch', 'content', 'media', 'asset'}


def _strip_periods(text: str) -> str:
    for pattern, _ in _READERS:
        text = pattern.sub(' ', text)
    return text


def _key(text: str) -> str:
    return re.sub(r'[\W_]+', '-', text.lower()).strip('-')


def series_key(url: str, name: str = '') -> str:
    """The document's file name with its period taken out: one key for every month of one fund's factsheet. When
    the file name says nothing about the document (getfile.aspx?id=12, a hash, 'download'), its name is used."""
    segments = [unquote(seg) for seg in urlparse(url).path.split('/') if seg]
    # the last segment that looks like a file name: some sites put an id after it (/documents/1/report.pdf/<uuid>)
    filename = next((seg for seg in reversed(segments) if re.search(r'\.[A-Za-z0-9]{2,5}$', seg)),
                    segments[-1] if segments else '')
    stem, dot, ext = filename.rpartition('.')
    stem, ext = (stem, ext.lower()) if dot and len(ext) <= 5 else (filename, '')
    key = _key(_strip_periods(stem.lower()))
    letters = re.sub(r'[^a-zà-ÿőű]', '', key)
    opaque = (len(letters) < 3 or key in _GENERIC_NAMES or ext in ('aspx', 'php', 'asp', 'jsp', 'ashx', 'cgi')
              or bool(re.fullmatch(r'[0-9a-f-]{16,}', key)))
    if opaque and name:
        key = _key(_strip_periods(name.lower()))
    return f'{key}.{ext}' if ext and ext not in ('aspx', 'php', 'asp', 'jsp', 'ashx', 'cgi') else key


def mark_latest(documents: list) -> list:
    """Set `series` and `latest_in_series` on each AgenticDownload and return the superseded ones.

    latest_in_series is True for the newest of a series (ties included), False for an older one, None when the
    document has no period (nothing to compare it by - a file a site overwrites each month often has none) or
    was rejected. Only accepted, unsure or unscored documents can supersede another."""
    newest: dict[str, int] = {}
    for doc in documents:
        doc.period = doc.period or period_of(f'{doc.name} {doc.url}')
        doc.series = series_key(doc.url, doc.name)
        if doc.period and doc.verdict != 'rejected':
            newest[doc.series] = max(newest.get(doc.series, 0), period_end(doc.period))
    superseded = []
    for doc in documents:
        if not doc.period or doc.verdict == 'rejected':
            doc.latest_in_series = None
            continue
        doc.latest_in_series = period_end(doc.period) >= newest[doc.series]
        if not doc.latest_in_series:
            superseded.append(doc)
    return superseded


def apply_latest(result, latest: bool, include_rejected: bool = False):
    """Mark every document of an AgenticCrawlResult; with `latest`, drop the superseded ones from `downloads` and
    count them in `superseded_count`. A caller that asked for rejected rows keeps the superseded ones too, marked."""
    superseded = mark_latest(result.downloads)
    if latest:
        result.superseded_count = len(superseded)
        if not include_rejected:
            gone = {id(d) for d in superseded}
            result.downloads = [d for d in result.downloads if id(d) not in gone]
    return result
