"""Domain profiles: the wording a Relevance judge uses for one kind of document hunt.

A profile holds the relevance criteria (what counts as a document the goal asks for, and which look-alikes do
not) and the page kinds that rank the Frontier. It is data, chosen explicitly by the caller: `generic` works for
any goal; others (`fund-reports`) carry wording tuned for one domain. The page-kind ids are fixed - the Frontier
tiers and the Escalation trigger are keyed on them - only their descriptions vary.

    load_profile()                      # generic
    load_profile('fund-reports')
    load_profile('/path/to/my-profile.json')
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROFILES_DIR = Path(__file__).resolve().parent / 'profiles'
DEFAULT_PROFILE = 'generic'
# The ids are part of what the judge reads, not just keys: Jev ranks pages by the label as well as its description,
# so 'fund_or_product' covers any subject page (see the profiles' wording). Renaming an id changes the crawl -
# re-run the eval before you do.
PAGE_KIND_IDS =('fund_or_product', 'document_listing', 'category_or_overview', 'news_or_article',
                 'company_or_legal', 'other')


class UnknownProfile(ValueError):
    pass


@dataclass(frozen=True)
class Profile:
    name: str
    relevance: dict[str, str]          # {'true': criterion, 'false': criterion}
    page_kinds: dict[str, str]         # page-kind id -> description
    description: str = ''


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob('*.json'))


def load_profile(name_or_path: str | None = None) -> Profile:
    """A bundled profile by name, or a profile file by path."""
    name_or_path = name_or_path or DEFAULT_PROFILE
    path = Path(name_or_path)
    if path.suffix != '.json':
        path = PROFILES_DIR / f'{name_or_path}.json'
    if not path.exists():
        raise UnknownProfile(f'unknown profile {name_or_path!r}; bundled: {", ".join(available_profiles())}')
    data = json.loads(path.read_text())
    relevance, kinds = data.get('relevance', {}), data.get('page_kinds', {})
    if set(relevance) != {'true', 'false'}:
        raise UnknownProfile(f'{path.name}: relevance needs exactly a "true" and a "false" criterion')
    if set(kinds) != set(PAGE_KIND_IDS):
        raise UnknownProfile(f'{path.name}: page_kinds must describe exactly {", ".join(PAGE_KIND_IDS)}')
    return Profile(name=path.stem, relevance=relevance, page_kinds=kinds, description=data.get('description', ''))
