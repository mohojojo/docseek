from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

_DOMAIN_KEY_RE = re.compile(r'^[a-zA-Z0-9.-]+$')
_MAX_URL_PATTERNS = 20
_MAX_GATE_SEQUENCES = 5


class GateStep(BaseModel):
    tool: str
    hint: str
    value: str | None = None


class GateSequence(BaseModel):
    description: str
    steps: list[GateStep] = Field(default_factory=list)


class DomainPatterns(BaseModel):
    domain: str
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    successful_crawl_count: int = 0
    last_goal: str = ''
    url_patterns_prefer: list[str] = Field(default_factory=list)
    url_patterns_skip: list[str] = Field(default_factory=list)
    gate_sequences: list[GateSequence] = Field(default_factory=list)
    memory_snapshot: dict[str, str] = Field(default_factory=dict)
    navigation_hints: str = ''


class PatternStore:
    def __init__(self, patterns_dir: str) -> None:
        self._dir = Path(patterns_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _normalize_domain(self, domain: str) -> str:
        domain = domain.lower()
        if ':' in domain:
            domain = domain.split(':')[0]
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain

    def _validate_domain_key(self, key: str) -> None:
        if '..' in key or '/' in key or '\\' in key:
            raise ValueError(f'Potential path traversal in domain key: {key!r}')
        if not _DOMAIN_KEY_RE.match(key):
            raise ValueError(f'Invalid domain key: {key!r}')

    def _path_for(self, domain: str) -> Path:
        key = self._normalize_domain(domain)
        self._validate_domain_key(key)
        return self._dir / f'{key}.json'

    def load(self, domain: str) -> DomainPatterns | None:
        try:
            path = self._path_for(domain)
            if not path.exists():
                return None
            return DomainPatterns.model_validate_json(path.read_text())
        except Exception:
            return None

    def save(self, patterns: DomainPatterns) -> None:
        path = self._path_for(patterns.domain)
        tmp = path.with_suffix('.tmp')
        tmp.write_text(patterns.model_dump_json(indent=2))
        os.replace(tmp, path)

    def merge_and_save(self, domain: str, new: DomainPatterns) -> None:
        existing = self.load(domain)
        if existing is None:
            merged = new.model_copy(update={
                'domain': self._normalize_domain(domain),
                'updated_at': datetime.now(timezone.utc),
                'successful_crawl_count': 1,
            })
        else:
            prefer = list(dict.fromkeys(existing.url_patterns_prefer + new.url_patterns_prefer))[:_MAX_URL_PATTERNS]
            skip = list(dict.fromkeys(existing.url_patterns_skip + new.url_patterns_skip))[:_MAX_URL_PATTERNS]

            seen_descs = {g.description for g in existing.gate_sequences}
            new_gates = [g for g in new.gate_sequences if g.description not in seen_descs]
            gate_sequences = (existing.gate_sequences + new_gates)[:_MAX_GATE_SEQUENCES]

            merged_memory = {**existing.memory_snapshot, **new.memory_snapshot}

            merged = DomainPatterns(
                domain=existing.domain,
                updated_at=datetime.now(timezone.utc),
                successful_crawl_count=existing.successful_crawl_count + 1,
                last_goal=new.last_goal or existing.last_goal,
                url_patterns_prefer=prefer,
                url_patterns_skip=skip,
                gate_sequences=gate_sequences,
                memory_snapshot=merged_memory,
                navigation_hints=new.navigation_hints or existing.navigation_hints,
            )
        self.save(merged)

    def delete(self, domain: str) -> bool:
        try:
            path = self._path_for(domain)
            if not path.exists():
                return False
            path.unlink()
            return True
        except Exception:
            return False

    def list_all(self) -> list[DomainPatterns]:
        results = []
        for path in sorted(self._dir.glob('*.json')):
            if path.stem.startswith('_'):
                continue
            try:
                results.append(DomainPatterns.model_validate_json(path.read_text()))
            except Exception:
                continue
        return results
