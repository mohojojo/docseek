"""Tests never reach the network for reach checks: every host resolves to a public address and has no
robots.txt, unless a test patches crawler.reach itself."""
import pytest

import docseek.reach as reach


@pytest.fixture(autouse=True)
def _offline_reach(monkeypatch):
    monkeypatch.setattr(reach, '_resolve', lambda host: ('93.184.216.34',))
    monkeypatch.setattr(reach, '_load_robots', lambda scheme, netloc: None)
    reach._robots_cache.clear()
    yield
    reach._robots_cache.clear()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    """A developer's proxy or remote browser settings must not send test traffic anywhere."""
    for name in ('PROXY_SERVER', 'PROXY_USERNAME', 'PROXY_PASSWORD', 'BROWSER_CDP_URL'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_model_keys(monkeypatch):
    """A developer's .env must not reach the tests: with a key in it, the server starts writing a real program
    in the background, against a paid API, where a test expects no model."""
    for name in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'LLM_API_KEY', 'LLM_PROVIDER', 'LLM_MODEL', 'LLM_BASE_URL',
                 'CODEGEN_MODEL', 'TYPESAFE_API_KEY', 'LAYA_API_KEY', 'LAYA_MODEL'):
        monkeypatch.delenv(name, raising=False)
