from __future__ import annotations

import pytest

from docseek.models import AgentStep, CrawlPlan
from docseek.patterns import DomainPatterns, GateSequence, PatternStore


# ---------------------------------------------------------------------------
# PatternStore - domain normalization and validation
# ---------------------------------------------------------------------------

def test_normalize_domain(tmp_path):
    store = PatternStore(str(tmp_path))
    assert store._normalize_domain('WWW.Example.COM') == 'example.com'
    assert store._normalize_domain('example.com:443') == 'example.com'
    assert store._normalize_domain('www.sub.example.com') == 'sub.example.com'


def test_validate_domain_key_rejects_traversal(tmp_path):
    store = PatternStore(str(tmp_path))
    with pytest.raises(ValueError):
        store._validate_domain_key('../etc/passwd')
    with pytest.raises(ValueError):
        store._validate_domain_key('foo/bar')
    with pytest.raises(ValueError):
        store._validate_domain_key('foo\\bar')
    with pytest.raises(ValueError):
        store._validate_domain_key('foo bar')


def test_validate_domain_key_accepts_valid(tmp_path):
    store = PatternStore(str(tmp_path))
    store._validate_domain_key('example.com')
    store._validate_domain_key('sub.example.co.uk')
    store._validate_domain_key('example-fund.com')


# ---------------------------------------------------------------------------
# PatternStore - save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    store = PatternStore(str(tmp_path))
    p = DomainPatterns(
        domain='example.com',
        url_patterns_prefer=['/products/'],
        url_patterns_skip=['/contact'],
        memory_snapshot={'detail_path': '/products/'},
        navigation_hints='Documents are on detail pages.',
    )
    store.save(p)
    loaded = store.load('example.com')
    assert loaded is not None
    assert loaded.domain == 'example.com'
    assert loaded.url_patterns_prefer == ['/products/']
    assert loaded.memory_snapshot == {'detail_path': '/products/'}


def test_load_missing_returns_none(tmp_path):
    store = PatternStore(str(tmp_path))
    assert store.load('missing.com') is None


# ---------------------------------------------------------------------------
# PatternStore - merge semantics
# ---------------------------------------------------------------------------

def test_merge_url_patterns_capped(tmp_path):
    store = PatternStore(str(tmp_path))
    existing = DomainPatterns(
        domain='example.com',
        url_patterns_prefer=[f'/path{i}/' for i in range(18)],
    )
    store.save(existing)
    new = DomainPatterns(
        domain='example.com',
        url_patterns_prefer=['/pathA/', '/pathB/', '/pathC/', '/pathD/'],
    )
    store.merge_and_save('example.com', new)
    result = store.load('example.com')
    assert result is not None
    assert len(result.url_patterns_prefer) == 20


def test_merge_url_patterns_dedup(tmp_path):
    store = PatternStore(str(tmp_path))
    existing = DomainPatterns(domain='example.com', url_patterns_prefer=['/products/', '/reports/'])
    store.save(existing)
    new = DomainPatterns(domain='example.com', url_patterns_prefer=['/products/', '/new/'])
    store.merge_and_save('example.com', new)
    result = store.load('example.com')
    assert result is not None
    assert result.url_patterns_prefer.count('/products/') == 1
    assert '/new/' in result.url_patterns_prefer


def test_merge_gate_sequences_dedup(tmp_path):
    store = PatternStore(str(tmp_path))
    existing = DomainPatterns(
        domain='example.com',
        gate_sequences=[GateSequence(description='Investor gate', steps=[])],
    )
    store.save(existing)
    # Same description - should not duplicate
    new = DomainPatterns(
        domain='example.com',
        gate_sequences=[
            GateSequence(description='Investor gate', steps=[]),
            GateSequence(description='Language gate', steps=[]),
        ],
    )
    store.merge_and_save('example.com', new)
    result = store.load('example.com')
    assert result is not None
    descriptions = [g.description for g in result.gate_sequences]
    assert descriptions.count('Investor gate') == 1
    assert 'Language gate' in descriptions


def test_merge_gate_sequences_capped(tmp_path):
    store = PatternStore(str(tmp_path))
    existing = DomainPatterns(
        domain='example.com',
        gate_sequences=[GateSequence(description=f'Gate {i}', steps=[]) for i in range(4)],
    )
    store.save(existing)
    new = DomainPatterns(
        domain='example.com',
        gate_sequences=[GateSequence(description='New gate', steps=[]), GateSequence(description='Another gate', steps=[])],
    )
    store.merge_and_save('example.com', new)
    result = store.load('example.com')
    assert result is not None
    assert len(result.gate_sequences) == 5


def test_merge_memory_last_write_wins(tmp_path):
    store = PatternStore(str(tmp_path))
    existing = DomainPatterns(domain='example.com', memory_snapshot={'key': 'old_value', 'other': 'keep'})
    store.save(existing)
    new = DomainPatterns(domain='example.com', memory_snapshot={'key': 'new_value'})
    store.merge_and_save('example.com', new)
    result = store.load('example.com')
    assert result is not None
    assert result.memory_snapshot['key'] == 'new_value'
    assert result.memory_snapshot['other'] == 'keep'


def test_merge_increments_crawl_count(tmp_path):
    store = PatternStore(str(tmp_path))
    p = DomainPatterns(domain='example.com', successful_crawl_count=2)
    store.save(p)
    store.merge_and_save('example.com', DomainPatterns(domain='example.com'))
    result = store.load('example.com')
    assert result is not None
    assert result.successful_crawl_count == 3


# ---------------------------------------------------------------------------
# PatternStore - delete and list
# ---------------------------------------------------------------------------

def test_delete_existing(tmp_path):
    store = PatternStore(str(tmp_path))
    store.save(DomainPatterns(domain='example.com'))
    assert store.delete('example.com') is True
    assert store.load('example.com') is None


def test_delete_missing(tmp_path):
    store = PatternStore(str(tmp_path))
    assert store.delete('missing.com') is False


def test_list_all(tmp_path):
    store = PatternStore(str(tmp_path))
    store.save(DomainPatterns(domain='alpha.com'))
    store.save(DomainPatterns(domain='beta.com'))
    all_patterns = store.list_all()
    domains = {p.domain for p in all_patterns}
    assert domains == {'alpha.com', 'beta.com'}


# ---------------------------------------------------------------------------
# extract_patterns - deterministic part (no LLM call needed)
# ---------------------------------------------------------------------------

def test_extract_patterns_url_patterns(tmp_path):
    from docseek.learn import extract_patterns
    from unittest.mock import MagicMock

    steps = [
        AgentStep(tool='navigate', args={}, reason='', source_url='https://example.com/products/a'),
        AgentStep(tool='navigate', args={}, reason='', source_url='https://example.com/products/b'),
        AgentStep(tool='done', args={}, reason='', source_url='https://example.com/contact'),
    ]
    plan = CrawlPlan(url_patterns_prefer=['/products/'], url_patterns_skip=['/contact'])
    client = MagicMock()
    # No gate steps → no LLM call needed
    result = extract_patterns('example.com', steps, {'k': 'v'}, plan, client)
    assert '/products/' in result.url_patterns_prefer
    assert result.memory_snapshot == {'k': 'v'}
    assert result.url_patterns_skip == ['/contact']


def test_extract_and_save_swallows_exceptions(tmp_path):
    from docseek.learn import extract_and_save
    from unittest.mock import MagicMock

    store = PatternStore(str(tmp_path))
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError('network error')

    steps = [AgentStep(tool='select_option', args={}, reason='', source_url=None)]
    plan = CrawlPlan()
    # Must not raise
    extract_and_save('example.com', steps, {}, plan, store, client)
