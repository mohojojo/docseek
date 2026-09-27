"""Escalation recipes: record what the agent did, replay it as code. No network."""
from pathlib import Path

from docseek.models import AgentStep
from docseek.recipes import Recipe, RecipeStep, RecipeStore, recipe_from_steps

YEAR = {'ml_id': '7', 'role': 'combobox', 'name': 'Év', 'html_tag': 'button', 'attributes': {'id': 'year-filter'}}
SEARCH = {'ml_id': '9', 'role': 'button', 'name': 'Keresés', 'html_tag': 'button', 'attributes': {}}


def steps(*calls):
    return [AgentStep(tool=t, args=a, reason='', source_url='u') for t, a in calls]


class TestRecording:
    def test_only_reveal_steps_are_kept_addressed_by_what_the_agent_saw(self):
        recipe = recipe_from_steps('https://www.fund.example/hu/dokumentumok/jelentesek/', 'g', steps(
            ('select_option', {'ml_id': '7', 'value': '2026'}), ('fill_input', {'ml_id': '3', 'value': 'x'}),
            ('click', {'ml_id': '9'}), ('record_download', {'url': 'https://x/a.pdf'}), ('done', {})),
            [YEAR, None, SEARCH, None, None], revealed=6)
        assert recipe.host == 'fund.example' and recipe.path == '/hu/dokumentumok/jelentesek'
        assert [(s.action, s.name, s.value) for s in recipe.steps] == [('select', 'Év', '2026'), ('click', 'Keresés', None)]
        assert recipe.steps[0].attributes == {'id': 'year-filter', 'tag': 'button'}   # typing is never recorded

    def test_failed_steps_and_anything_after_the_last_recording_are_dropped(self):
        # as the agent really did it on a bank site: select year, search, record, scroll, a next-page click the
        # page refused, done. Only the select and the search are the recipe.
        NEXT = {'ml_id': '120', 'role': 'button', 'name': 'Következő oldal', 'html_tag': 'button', 'attributes': {}}
        recipe = recipe_from_steps('https://site-hu.example/p', 'g', steps(
            ('select_option', {'ml_id': '94', 'value': '2026'}), ('click', {'ml_id': '96'}),
            ('record_downloads', {'items': []}), ('scroll_to_load', {}), ('click', {'ml_id': '120'}), ('done', {})),
            [YEAR, SEARCH, None, None, NEXT, None], revealed=9,
            outcomes=['ok', 'ok', 'recorded', 'ok', 'failed', 'ok'])
        assert [(s.action, s.name) for s in recipe.steps] == [('select', 'Év'), ('click', 'Keresés')]

    def test_opening_a_dropdown_is_not_a_step(self):
        # the agent clicked the type filter open (and chose nothing), clicked the year filter open, then
        # selected in it: on replay the select opens the dropdown itself and the clicks would toggle it shut
        TYPE = {'ml_id': '92', 'role': 'button', 'name': 'Dokumentum típus', 'html_tag': 'button',
                'attributes': {'class': 'sf-select__button'}}
        recipe = recipe_from_steps('https://site-hu.example/p', 'g', steps(
            ('click', {'ml_id': '92'}), ('click', {'ml_id': '7'}), ('select_option', {'ml_id': '7', 'value': '2026'}),
            ('click', {'ml_id': '9'})), [TYPE, YEAR, YEAR, SEARCH], revealed=9)
        assert [(s.action, s.name) for s in recipe.steps] == [('select', 'Év'), ('click', 'Keresés')]

    def test_a_per_load_id_is_not_kept(self):
        el = {'ml_id': '94', 'role': 'button', 'name': 'Év', 'html_tag': 'button',
              'attributes': {'id': 'f4d77604-f739-4f13-85ba-8bafe4333925-btn', 'class': 'sf-select__button'}}
        recipe = recipe_from_steps('https://site-hu.example/p', 'g', steps(('select_option', {'ml_id': '94', 'value': '2026'})), [el], 9)
        assert recipe.steps[0].attributes == {'class': 'sf-select__button', 'tag': 'button'}

    def test_an_escalation_that_revealed_nothing_leaves_no_recipe(self):
        assert recipe_from_steps('https://site-hu.example/p', 'g', steps(('click', {'ml_id': '9'})), [SEARCH], revealed=0) is None
        assert recipe_from_steps('https://site-hu.example/p', 'g', steps(('done', {})), [None], revealed=3) is None


class TestStore:
    def test_round_trip_and_forget(self, tmp_path: Path):
        store = RecipeStore(tmp_path)
        recipe = Recipe(host='fund.example', path='/hu/dokumentumok/jelentesek', goal='g',
                        steps=[RecipeStep(action='click', role='button', name='Keresés')], revealed=6)
        store.save(recipe)
        assert store.load('https://www.fund.example/hu/dokumentumok/jelentesek?x=1').steps[0].name == 'Keresés'
        assert store.load('https://www.fund.example/hu/dokumentumok/beszamolok') is None
        store.forget(recipe)
        assert store.load('https://www.fund.example/hu/dokumentumok/jelentesek') is None
