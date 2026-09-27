"""Pay the agent once per page layout, then let code replay what it did.

An escalation hands a page to the Claude tool loop, which drives a filter or a control and records the
documents it reveals. That costs a large number of agent tokens and tens of seconds, and on a site with a
filtered listing it is paid on every crawl for the same page - and the agent's answer varies (an
escalation that usually pays can return nothing). Stagehand, Skyvern and browser-use all cache a successful action sequence and replay it as
code so the model is paid once per site layout; this is that, for escalations.

A recipe is the reveal-only steps of an escalation that produced Candidates: clicks and selections,
each addressed by the element's role and accessible name (what the agent saw) rather than the per-load
`ml_id` it used. Typing is never recorded. Recipes are keyed by host and page path and stored next to
the domain patterns; a replay that reveals nothing new is forgotten so a changed layout does not keep
costing a failed attempt.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REPLAYABLE = {'click', 'select_option'}
MAX_STEPS = 12
_KEY_RE = re.compile(r'[^a-z0-9._-]+')
_GENERATED_ID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-|\d{6,}|^(ember|react|radix|mui|headlessui|:r)')   # per-load ids


class RecipeStep(BaseModel):
    action: str                     # click | select
    role: str = ''
    name: str = ''
    attributes: dict[str, str] = Field(default_factory=dict)    # id, name, aria-label: fallbacks
    value: str | None = None        # for select


class Recipe(BaseModel):
    host: str
    path: str
    goal: str
    steps: list[RecipeStep]
    revealed: int                   # Candidates the escalation produced when recorded
    recorded_at: float = Field(default_factory=time.time)
    replays: int = 0
    last_revealed: int | None = None


def recipe_from_steps(url: str, goal: str, steps: list, registry_by_step: list[dict | None],
                      revealed: int, outcomes: list[str] | None = None) -> Recipe | None:
    """The replayable part of an escalation's tool calls, or None if there is none worth keeping.

    Kept: the reveal steps that worked, up to the last one before the agent's last recording. A step the
    page refused (the agent's own tool result said so) is dropped, and so is anything after the last
    recording - on one site that was a 'next page' click that never became visible.
    """
    outcomes = outcomes or ['ok'] * len(steps)
    last_recording = max((i for i, o in enumerate(outcomes) if o == 'recorded'), default=len(steps) - 1)
    out: list[RecipeStep] = []
    for i, (step, element) in enumerate(zip(steps, registry_by_step)):
        if i > last_recording or step.tool not in REPLAYABLE or element is None or outcomes[i] == 'failed':
            continue
        attrs = {k: v for k, v in (element.get('attributes') or {}).items()
                 if k in ('id', 'name', 'aria-label', 'data-testid', 'title', 'class') and v}
        if _GENERATED_ID.search(attrs.get('id', '')):
            del attrs['id']                                   # a uuid minted on each load identifies nothing
        attrs['tag'] = element.get('html_tag') or ''
        out.append(RecipeStep(action='select' if step.tool == 'select_option' else 'click',
                              role=element.get('role', ''), name=element.get('name', ''), attributes=attrs,
                              value=step.args.get('value') if step.tool == 'select_option' else None))
    # the agent opens a dropdown with a click before it selects in it; the select does the opening on
    # replay, and a second click would close it again. A click on a control a select later addresses,
    # or on any other dropdown button, is not a step.
    selected = {(s.role, s.name) for s in out if s.action == 'select'}
    out = [s for s in out if not (s.action == 'click' and ((s.role, s.name) in selected
                                                           or 'select' in s.attributes.get('class', '')))]
    if not out or revealed <= 0:
        return None
    parsed = urlparse(url)
    return Recipe(host=parsed.netloc.lower().removeprefix('www.'), path=parsed.path.rstrip('/') or '/',
                  goal=goal, steps=out[:MAX_STEPS], revealed=revealed)


def _same_kind(page, loc, step: RecipeStep):
    """Of several elements with the recorded role and name, the one with the recorded tag and class."""
    tag, cls = step.attributes.get('tag', '').lower(), step.attributes.get('class', '')
    for i in range(min(loc.count(), 10)):
        candidate = loc.nth(i)
        try:
            info = candidate.evaluate('el => [el.tagName.toLowerCase(), el.className || ""]')
        except Exception:  # noqa: BLE001
            continue
        if (not tag or info[0] == tag) and (not cls or info[1] == cls):
            return candidate
    return loc.first


def _locator(page, step: RecipeStep):
    """The element the step meant, by what does not change between loads: an authored id or test id,
    then the accessible role and name the agent saw (tag and class break ties), then attributes."""
    for attr in ('id', 'data-testid'):
        if step.attributes.get(attr):
            loc = page.locator(f'[{attr}="{step.attributes[attr]}"]')
            if loc.count():
                return loc.first
    if step.role and step.name:
        for exact in (True, False):
            try:
                loc = page.get_by_role(step.role, name=step.name, exact=exact)
                if loc.count():
                    return _same_kind(page, loc, step)
            except Exception:  # noqa: BLE001 - an unknown role string
                break
    if step.name:
        loc = page.get_by_text(step.name, exact=True)
        if loc.count():
            return _same_kind(page, loc, step)
    for attr in ('name', 'aria-label', 'title'):
        if step.attributes.get(attr):
            loc = page.locator(f'[{attr}="{step.attributes[attr]}"]')
            if loc.count():
                return loc.first
    return None


def replay(page, recipe: Recipe, settle, wait_ms: int = 1000) -> int:
    """Run the recipe's steps on `page`. Returns how many steps found their element."""
    from .scraper import select_option_anywhere
    done = 0
    for step in recipe.steps:
        loc = _locator(page, step)
        if loc is None:
            logger.info('[recipe] %s%s: no element for %s %r', recipe.host, recipe.path, step.action, step.name)
            break
        try:
            if step.action == 'select' and step.value is not None:
                ml_id = loc.evaluate('el => { el.setAttribute("data-ml-id", "recipe"); return "recipe"; }')
                select_option_anywhere(page, ml_id, step.value, wait_ms=wait_ms)
            else:
                loc.scroll_into_view_if_needed(timeout=3000)
                loc.dispatch_event('click')
            done += 1
            settle()
        except Exception as exc:  # noqa: BLE001 - a step that fails ends the replay, not the page
            logger.info('[recipe] step failed on %s%s: %s', recipe.host, recipe.path, exc)
            break
    return done


class RecipeStore:
    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._lock = threading.Lock()

    def _path(self, host: str, path: str) -> Path:
        key = _KEY_RE.sub('_', f'{host}{path}'.lower()).strip('_')[:150]
        return self._dir / 'recipes' / f'{key}.json'

    def load(self, url: str) -> Recipe | None:
        parsed = urlparse(url)
        file = self._path(parsed.netloc.lower().removeprefix('www.'), parsed.path.rstrip('/') or '/')
        try:
            return Recipe.model_validate_json(file.read_text()) if file.exists() else None
        except (OSError, ValueError):
            return None

    def save(self, recipe: Recipe) -> None:
        file = self._path(recipe.host, recipe.path)
        with self._lock:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(recipe.model_dump(), ensure_ascii=False, indent=1))

    def forget(self, recipe: Recipe) -> None:
        with self._lock:
            self._path(recipe.host, recipe.path).unlink(missing_ok=True)
