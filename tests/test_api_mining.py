from __future__ import annotations

import httpx

from docseek.api_mining import _firestore_list


def test_firestore_list_calls_the_firestore_rest_documents_path():
    # Firestore's REST path is /v1/projects/{project}/databases/...; a repo-wide route
    # rename once rewrote it to /v1/extractions/, silently breaking all Firebase mining.
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json={'documents': [
            {'fields': {'name': {'stringValue': 'Fund A 2026-1 hu.pdf'},
                        'filePath': {'stringValue': 'Funds/A/Docs/Fund A 2026-1 hu.pdf'}}},
        ]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        docs = _firestore_list(client, 'demo-project', 'KEY', 'Funds')

    assert seen[0].path == '/v1/projects/demo-project/databases/(default)/documents/Funds'
    assert docs == [{'name': 'Fund A 2026-1 hu.pdf', 'filePath': 'Funds/A/Docs/Fund A 2026-1 hu.pdf'}]
