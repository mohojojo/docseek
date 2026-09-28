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
