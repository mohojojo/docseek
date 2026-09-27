"""Reach: where a crawl may go and what it may hand back.

One module owns the rules every decision layer applies before touching a URL:
  is_safe_url(url)      no SSRF target: http(s) only, and every address the host resolves to is public
  robots_allows(url)    the site's robots.txt permits the fetch (loaded lazily, once per host)
  is_crawlable(url)     both of the above
  bare_host(host)       host identity: case and a leading 'www.' do not make a different site
  OffDomainPolicy       which hosts a crawl may visit and return documents from, given its seed

Resolution and robots fetches are cached per host for the life of the process.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
import threading
import urllib.request
import urllib.robotparser
from functools import lru_cache
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

ROBOTS_TIMEOUT_S = 5
_ROBOTS_USER_AGENT = 'Mozilla/5.0 (compatible; document-crawler)'

# --- SSRF -----------------------------------------------------------------------------------------
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network('0.0.0.0/8'),
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('100.64.0.0/10'),   # CGNAT
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('169.254.0.0/16'),  # link-local
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fc00::/7'),        # ULA
    ipaddress.ip_network('fe80::/10'),       # IPv6 link-local
    ipaddress.ip_network('ff00::/8'),        # multicast
    ipaddress.ip_network('2002::/16'),       # 6to4 (embeds IPv4 in addr)
]

_BLOCKED_HOSTNAMES = frozenset({'localhost'})
_BLOCKED_SUFFIXES = ('.local', '.internal', '.localhost')


def _is_public(address: str) -> bool:
    addr = ipaddress.ip_address(address.split('%')[0])          # drop an IPv6 zone id
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped                                   # ::ffff:127.0.0.1 is 127.0.0.1
    return not any(addr in net for net in _SSRF_BLOCKED_NETWORKS)


@lru_cache(maxsize=4096)
def _resolve(host: str) -> tuple[str, ...]:
    """Every address the host resolves to; () when it does not resolve."""
    try:
        return tuple(sorted({info[4][0] for info in socket.getaddrinfo(host, None)}))
    except (socket.gaierror, UnicodeError, OSError):
        return ()


def is_safe_url(url: str) -> bool:
    """True only if the URL is safe to fetch or hand to callers: http(s), a named or literal host that is not
    local, and every address it resolves to public. A hostname pointing at 127.0.0.1 or a metadata address is
    refused like the literal address; a host that does not resolve is refused too."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return False
        host = (parsed.hostname or '').lower()
        if not host or host in _BLOCKED_HOSTNAMES or any(host.endswith(s) for s in _BLOCKED_SUFFIXES):
            return False
        try:
            return _is_public(host)                               # a literal address
        except ValueError:
            pass
        addresses = _resolve(host)
        return bool(addresses) and all(_is_public(a) for a in addresses)
    except Exception:
        return False


# --- robots.txt -------------------------------------------------------------------------------------
# host (netloc) -> parsed rules, or None when the site has none we could read (everything allowed)
_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}
_robots_lock = threading.Lock()


def store_robots(netloc: str, lines: list[str]) -> urllib.robotparser.RobotFileParser:
    """Remember a host's robots.txt (fetch_sitemap reads it anyway)."""
    parser = urllib.robotparser.RobotFileParser()
    parser.parse(lines)
    _robots_cache[netloc] = parser
    return parser


def _load_robots(scheme: str, netloc: str) -> urllib.robotparser.RobotFileParser | None:
    url = f'{scheme}://{netloc}/robots.txt'
    if not is_safe_url(url):
        return None
    try:
        request = urllib.request.Request(url, headers={'User-Agent': _ROBOTS_USER_AGENT})
        with urllib.request.urlopen(request, timeout=ROBOTS_TIMEOUT_S) as response:
            lines = response.read().decode('utf-8', errors='replace').splitlines()
    except Exception as exc:  # noqa: BLE001 - no readable robots.txt means no rules
        logger.debug('robots.txt unavailable for %s: %s', netloc, exc)
        return None
    return store_robots(netloc, lines)


def robots_allows(url: str) -> bool:
    """False when the host's robots.txt disallows the URL. The rules are fetched on the first check for a host,
    whichever layer asks - not only for the seed a sitemap was read from."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc
        if netloc not in _robots_cache:
            with _robots_lock:
                if netloc not in _robots_cache:
                    _robots_cache[netloc] = _load_robots(parsed.scheme or 'https', netloc)
        rules = _robots_cache[netloc]
        return True if rules is None else rules.can_fetch('*', url)
    except Exception:
        return True


def is_crawlable(url: str) -> bool:
    """A URL this crawler may fetch: no SSRF targets, and robots.txt permits it."""
    return is_safe_url(url) and robots_allows(url)


# --- host identity and the off-domain rule ------------------------------------------------------------
def bare_host(host: str) -> str:
    return host.lower().removeprefix('www.')


class OffDomainPolicy:
    """Where a crawl may go and what it may return, once the seed host is known.

    Documents may come from any host (the SSRF block still applies). Navigation may cross to one host
    outside the seed, taken from the seed itself, and then crawl within that host - never chaining on to
    a third. Hosts in `allowed_hosts` qualify without a link. `same_domain_only` is a hard lock.
    """

    def __init__(self, seed_host: str, *, same_domain_only: bool = True,
                 allowed_hosts: list[str] | None = None):
        self.seed_host = bare_host(seed_host)
        self.same_domain_only = same_domain_only
        self.allowed = {bare_host(h) for h in (allowed_hosts or [])}
        self.crossed_to: set[str] = set()
        self._lock = threading.Lock()

    def is_seed(self, url: str) -> bool:
        return bare_host(urlparse(url).netloc) == self.seed_host

    def may_visit(self, url: str, from_url: str) -> bool:
        host = bare_host(urlparse(url).netloc)
        if host == self.seed_host:
            return True
        if self.same_domain_only:
            return False
        if host in self.allowed or host in self.crossed_to:
            return True
        if self.is_seed(from_url):
            with self._lock:
                self.crossed_to.add(host)
            return True
        return False

    def may_return(self, url: str) -> bool:
        return self.is_seed(url) or not self.same_domain_only

    @property
    def first_party_hosts(self) -> set[str]:
        return {self.seed_host} | self.allowed | self.crossed_to
