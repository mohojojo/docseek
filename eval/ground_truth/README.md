# Ground truth

The eval scripts score a crawl against a hand-checked list of the documents a goal should find on
one site. No real ground truth ships with this repository: you supply your own, one JSON file per
site/goal in this directory.

`example.json` shows the format. It describes a fictional site, so it cannot be run. The loader
skips it unless you name its site explicitly (`--sites reports.example`).

Files in this directory other than `example.json` are ignored by git (see `.gitignore`), so your
ground truth stays local unless you choose to commit it.

## File format

```json
{
  "site": "example.com",
  "goal": "Find the 2025 annual reports (PDF)",
  "start_url": "https://example.com/investors/",
  "expected_documents": [
    {"url": "https://example.com/files/annual-report-2025.pdf", "name": "Annual report 2025"}
  ],
  "profile": "generic",
  "notes": "Captured 2026-01-15 from the investors page."
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `site` | yes | The site key. `--sites` selects on it, and it names the site in reports. |
| `goal` | yes | The natural-language goal passed to the crawler. |
| `expected_documents` | yes | List of `{url, name}`. Only `url` is scored; `name` is for people. An empty list is allowed: it means the site holds nothing the goal asks for, and every document returned there counts against precision. |
| `start_url` | no | Where `run_jev` seeds the crawl. Default: `https://www.<site>/` (or `https://<site>/` when `site` already starts with `www.`). |
| `off_domain` | no | `true` lets `run_jev` follow links off the seed host. Use it when the documents live on another host. |
| `match_query` | no | `true` keeps the query string in a document's identity, for sites that tell documents apart only by query (`getfile.aspx?id=123`). |
| `identity_re` | no | A regex with one capture group that identifies a document. Use it when one document is served under several URLs (`?download=12:report`, `?download=12:report&start=50`, `/file/12-report`). Matching URLs are compared as `<host>#<group>`; non-matching URLs fall back to the normal rule. |
| `goal_year` | no | For a goal that asks for one year. `run_jev` then scores only candidates dated that year or undated, as a consumer filtering on the year would. |
| `latest` | no | `true` for a goal that asks for each series' newest document. The runners then score what `"latest": true` returns: older documents of a series are left out. |
| `profile` | no | The domain profile the relevance judge words its questions with: a bundled name (`generic`, `fund-reports`) or a path to a profile JSON. Default `generic`. |
| `shape` | no | Free text describing what the site tests (for example "paginated archive"). Not used in scoring. |
| `notes` | no | Free text. Not used in scoring. |

## Adding your own site

1. Pick a site and a goal a person could check by hand.
2. Capture the expected documents from the site's own pages, by hand or with a small script
   that reads the listing pages. Record every document the goal asks for and nothing else.
   Do not build the list from the crawler's own output, or the eval only measures agreement
   with itself.
3. Save it as `eval/ground_truth/<site>.json`, with the capture date in `notes`.
4. Run it once with `--runs 1` and read the `missed` and `extra` lists in the report. A miss
   or extra that is really a ground-truth mistake gets fixed in the file.
5. Sites change. Re-capture the ground truth when documents are added, moved or renamed,
   otherwise recall drops for reasons that have nothing to do with the crawler.

## Scoring

A document's identity is its URL with the scheme and host lowercased and the query, fragment
and trailing slash removed. `match_query` keeps the query; `identity_re` replaces the rule for
URLs it matches. The same rule is applied to expected and found URLs, so it is a comparison key,
not the URL that gets fetched.

Per run:

- recall = matched / expected (1.0 when `expected_documents` is empty)
- precision = matched / found (0.0 when nothing was found but something was expected; 1.0 when
  both are empty)
- `missed` and `extra` list the unmatched identities on each side.

`run_jev` scores every run in two bands, using the judge's relevance score:

- `accepted`: candidates scored 0.75 or higher. This is the band precision is guarded on.
- `returned`: accepted plus `unsure` (0.4 up to 0.75) plus `unscored` (no score obtainable).
  This is what an API consumer actually receives.

It also reports `recall_at` (accepted recall after 10, 20 and 40 pages, which shows budget
efficiency) and `pages_to_first_hit`. Sitemap and API candidates count as found on page 0.
A run that stops with `jev_unavailable` measures the Jev service, not the crawler: it is kept
in the report, marked `discarded`, and left out of every summary.

## Running

Reports are written to `eval/reports/`, snapshots to `eval/snapshots/`. Both are ignored by git.
Keys are read from the environment or from `.env` at the repository root.

### run_jev: the judge-driven crawl

Needs what its judge needs: `TYPESAFE_API_KEY` for `--judge jev` (the default when it is set), `LAYA_URL` for
`--judge laya`, or an
LLM (`LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`, or `ANTHROPIC_API_KEY`) for `--judge llm`. Escalations to the
browsing agent run on the same LLM; without one, pass `--no-escalation`.

```sh
.venv/bin/python -m eval.run_jev --runs 2                       # every ground-truth file
.venv/bin/python -m eval.run_jev --runs 1 --sites example.com --no-snapshots
.venv/bin/python -m eval.run_jev --runs 1 --sites example.com --start-url https://docs.example.com/reports
```

Useful options: `--max-pages` (default 40), `--max-seconds` (default 180), `--off-domain`,
`--judge jev|laya|llm`, `--profile` (overrides every file's `profile`), `--frontier tier|rescue|bandit`,
`--recipes <dir>`, `--no-escalation`, `--label <tag>`. `--goal` overrides the file's goal, after
which the scores mean little.

Snapshots (each page's HTML, harvested links and page-kind answers) let `eval.frontier_replay`
replay a frontier policy over a recorded site offline:

```sh
.venv/bin/python -m eval.run_jev --runs 1 --max-pages 120 --max-seconds 900 --no-escalation \
    --label rec --sites example.com
.venv/bin/python -m eval.frontier_replay eval/snapshots/<stamp>/example.com/run1
```

The replay reports recall only when the recording's goal matches the ground-truth goal.

### run_baseline: the agentic crawler

Needs an LLM: `LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_KEY`, or `ANTHROPIC_API_KEY`.

```sh
.venv/bin/python -m eval.run_baseline                           # every file, 3 runs each
.venv/bin/python -m eval.run_baseline --runs 1 --sites example.com
.venv/bin/python -m eval.run_baseline --max-pages 15 --model claude-haiku-4-5
```

It seeds, leaves the seed host and matches documents the way `run_jev` does (`start_url`, `off_domain`,
`match_query`, `identity_re`). The agent has no relevance judge, so `profile` does not apply, and it scores
every document it records, with no accepted/returned split. Use at least 3 runs to see run-to-run variance.

### judge_compare: a relevance judge on a frozen set

Scores a judge offline on labelled candidates, with no crawling and no browser.

```sh
.venv/bin/python -m eval.judge_compare my-set.json --judge jev
.venv/bin/python -m eval.judge_compare my-set.json --judge llm --profile fund-reports --model claude-haiku-4-5
```

`--judge jev` needs `TYPESAFE_API_KEY`; `--judge laya` needs `LAYA_URL`; `--judge llm` needs an LLM configured as above.
`--profile` takes a bundled profile name or a profile path (default `generic`) and applies to
every goal in the set.

The set is a JSON list of candidates:

```json
[
  {
    "site": "example.com",
    "goal": "Find the 2025 annual reports (PDF)",
    "batch": "https://example.com/investors/",
    "context": "Investors - Reports",
    "url": "https://example.com/files/annual-report-2025.pdf",
    "text": "Annual report 2025",
    "row_text": "Annual report 2025 | PDF | 2.1 MB",
    "section": "Annual reports",
    "column": "Download",
    "label": 1
  }
]
```

| Field | Required | Meaning |
| --- | --- | --- |
| `site` | yes | Groups results per site. |
| `goal` | yes | The goal the candidate is judged against. |
| `batch` | yes | The page the candidate was found on. Candidates with the same site, goal, batch and context are judged together, as in a crawl. |
| `url` | yes | The candidate link. |
| `label` | yes | `1` if it is a document the goal asks for, `0` if not. |
| `context` | no | Page context passed to the judge (for example the page title). |
| `text` | no | The link text. |
| `row_text` | no | Text around the link, such as its table row. |
| `section`, `column` | no | The section heading and table column the link sits under. |

Build a set from `run_jev` snapshots: `run.json` lists each candidate's `url`, `name` (use it as
`text`) and `source_page` (use it as `batch`), and the `harvest_*.json` files hold the page
context. Label each candidate by hand. The script prints precision and recall of the `accepted` band per site and pooled, and
writes every prediction to the report.

### run_codegen: generated programs

```sh
.venv/bin/python -m eval.run_codegen --sites example.com                  # generate if missing, 2 replays
.venv/bin/python -m eval.run_codegen --sites example.com --regenerate --replays 1
```

Generation needs a coding model (`CODEGEN_MODEL`, or `LLM_MODEL`); replays need only the judge. Programs are
kept in `eval/programs/` (ignored by git) and reused until `--regenerate`. Scores mean the same as `run_jev`'s.
