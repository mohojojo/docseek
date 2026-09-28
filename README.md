# docseek

Goal-driven document discovery. Give it a website and a goal in plain language, in any language:

> *"Find the English datasheet of every battery charger"* · *"Töltsd le a 2026-os havi alap jelentéseket"* · *"Finde alle Sitzungsunterlagen vom März"*

and it crawls the site with a real browser and returns the documents the goal asks for, each with a
**relevance verdict** (`accepted`, `unsure`, `rejected`) and facets such as the period and year it covers.

It is built for sites where the documents are not one link away: behind year filters and tabs, in
"load more" listings, in single-page apps that load them from a JSON API, one or two clicks below a
listing, or spread over hundreds of pages of look-alike documents.

## How it decides

The crawl is driven by a **relevance judge** that answers four questions, and code does everything else:

1. **Is this link a document the goal asks for?** Every candidate gets a relevance score and a verdict.
2. **What kind of page does this link lead to?** Pages are visited in order of how likely they list documents.
3. **Does this page still hide documents?** If so, the page is **escalated** to a browsing agent that clicks,
   fills in and scrolls until it finds them.
4. **Which filter value shows the documents?** Code finds the page's filters, the judge picks the value, code sets it.

Two judges are built in:

| Judge | What it is |
| --- | --- |
| `llm` | Any LLM: Anthropic, or any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, OpenRouter, Gemini, Ollama, vLLM, LM Studio). Answers on five levels mapped into the verdict bands. |
| `jev` | [TypeSafe Jev](https://typesafe.ai), a decision model that returns calibrated probabilities. Needs a TypeSafe key. If it becomes unavailable mid-crawl, the LLM judge takes over. |

The judge's wording comes from a **domain profile** (`profile`): `generic` works for any goal; `fund-reports`
is an example of a profile tuned for one domain (periodic fund documents). A profile is a small JSON file -
see [`docseek/profiles/`](docseek/profiles) - so you can write one for your own domain.

The browsing agent (escalations, and the `agent` decision layer that crawls with the agent alone) runs on the
same LLM configuration.

## Quick start

```bash
pip install docseek
playwright install chromium
```

or from source:

```bash
git clone https://github.com/mohojojo/docseek && cd docseek
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e ".[dev]"
playwright install chromium
```

Then start the API:

```bash
export ANTHROPIC_API_KEY=...              # or configure any other model, see below
uvicorn docseek.server:app --port 8010
```

```bash
curl -s localhost:8010/v1/discover -H 'Content-Type: application/json' -d '{
  "url": "https://www.example.com/",
  "goal": "Find the 2025 annual reports (PDF)",
  "profile": "generic",
  "max_pages": 30
}'
```

Or with Docker: `docker compose up --build`.

Or from the command line (the same crawl, printed as JSON):

```bash
docseek https://www.example.com/ "Find the 2025 annual reports (PDF)" --max-pages 30
```

### Any model

```bash
# OpenAI
export LLM_PROVIDER=openai-compatible LLM_MODEL=gpt-4.1-mini LLM_API_KEY=sk-...

# a local model through Ollama
export LLM_PROVIDER=openai-compatible LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1

# Anthropic (the default when ANTHROPIC_API_KEY is set)
export LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-4-5
```

The client adapts to what a server accepts (for example OpenAI's reasoning models, which take
`max_completion_tokens` and no `temperature`). The browsing agent needs a model with tool calling;
screenshots need a vision-capable model.

## The latest of each series

The relevance judge scores what a document *is*, never which period it covers, so a goal like "each fund's
latest factsheet" finds the whole archive. Code answers the "latest" part. Every document carries a `series` facet -
its file name with the date taken out, or its title when the file name says nothing - and `latest_in_series` marks
the newest of each series.

Set `"latest": true` (CLI `--latest`) to keep only those: older documents of a series are counted in
`superseded_count` instead of returned. Every series keeps its newest document, however old.

Reading dates out of names will never be right for every site - `2211`, `eb202508`, `SAN-2025-12` and `Heft 3` each
mean something different - so the filter does not try to. The documents of one series share a naming pattern, and
they are ordered by the numbers in it (years first) without deciding whether `08` is August or issue 8. **When in
doubt, nothing is dropped:** a series that mixes formats, or a document with nothing to order by, is kept whole. A
wrong keep costs a caller one extra document; a wrong drop would lose one.

What code cannot settle goes to the relevance judge, and only with `latest`. Documents that look alike once their
numbers are taken out (`ETALON-2211` beside `ETALON-2212`, `2026-Q1` beside `2026-03`, but also `cm4` beside `cm5`)
form a doubtful group, and the judge is asked about it. First about the group as a whole: are these all editions of
one document, and which is the newest? When it is sure of both (probability 0.8 or more with Jev's calibrated
answers), the rest of the group goes. Otherwise each document is asked about on its own - is a newer edition of it
listed? - and dropped only when the judge is sure. Anything less is kept. Grouping by look is only a question: the
judge tells two products apart as readily as two editions.

Where a site shows a date beside a document - in its row, its dated line - that date is the `published` facet
(`YYYY-MM-DD`). Only forms with one meaning are read (`2026-03-12`, `12.03.2026`, `12 March 2026`,
`2026. március 12.`), never a slash date like `03/12/2026` and never a file name. It orders a series whose names
cannot (one file name a site overwrites each month), and the judge sees it when it compares editions.

The `period` facet (`2026-03`, `2026-Q1`, `2026-H1` or `2026`, read in English, German, Hungarian, French, Spanish
and Italian) is a best-effort label for people and callers, not what the filter decides by.

## Generated programs

For a site you query again and again, docseek can write a **discovery program**: a coding agent explores the site
once - raw HTML, rendered pages, the JSON calls a page makes - and writes a small Python function that finds the
documents the goal asks for. Later requests run the program with no model at all, in seconds, and the relevance
judge scores what it returns exactly as it scores a crawl.

```bash
export PROGRAMS_DIR=./programs CODEGEN_MODEL=claude-sonnet-5
docseek generate https://www.example.com/ "Find the 2025 annual reports (PDF)"   # a few minutes, once
docseek https://www.example.com/ "Find the 2025 annual reports (PDF)" --programs  # seconds, no model
docseek check                                                                      # has any site changed?
```

With `"programs": true`, `/v1/discover` answers from the site's program when it is **healthy**, and crawls
otherwise: when the program errors, returns nothing where it used to find documents, or keeps fewer than half of
what it kept last time, it is marked stale, the crawl answers, and a new program is written in the background.
`docseek check` (or `POST /v1/programs/check`) replays every program against a snapshot of what it returned
before - no model, no judge - and reports `ok`, `grew`, `shrank` or `broken`.

What a program can do is fenced in: it runs in a separate process with an empty environment, CPU and memory
limits, an allowlist of standard-library modules, and no network except the parent's fetcher - the site and its
subdomains, the data hosts and POST endpoints the site's own pages used while the program was written,
robots.txt, and the same public-address rule as the crawl. **This is a boundary, not a hardened jail**: the code
was written by a model that read untrusted page text, so the feature is off unless `PROGRAMS_DIR` is set, and a
docseek that strangers can reach belongs in a container.

A program is **verified** when its final code ran and either returned documents or had every fetch succeed; one
written against a site that refused every request is not, and is never used to answer. Every generation leaves a
log (`<key>.log.json`, also in `GET /v1/programs/{key}`) of what the agent said and did, including generations that
produced nothing. After a crawl, a program is regenerated automatically at most once every `CODEGEN_RETRY_HOURS`,
so a site that blocks bots does not cost a generation per request; `POST /v1/programs` always runs.

Health cannot tell when a program confidently returns the wrong slice of a site; a crawl or a person can.

## Configuration

| Variable | Description |
| --- | --- |
| `LLM_PROVIDER` | `anthropic` or `openai-compatible`. Default: `anthropic` when `ANTHROPIC_API_KEY` is set. |
| `LLM_MODEL` | Model for the agent and the LLM judge. Default on `anthropic`: `claude-haiku-4-5`; required for `openai-compatible`. |
| `LLM_BASE_URL` | Base URL of an OpenAI-compatible endpoint (default `https://api.openai.com/v1`). |
| `LLM_API_KEY` | Key for the LLM (falls back to `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; a local server may need none). |
| `ANTHROPIC_API_KEY` | Enough on its own to run everything on Claude. `/v1/search-sites` uses Anthropic's web search and always needs it. |
| `TYPESAFE_API_KEY` | Enables the `jev` judge (the default judge when set). |
| `JEV_PRICE_PER_MTOK` | Your TypeSafe price per million input tokens, used only for the cost the result reports (`jev_cost_usd`). Default 0. |
| `CRAWLER_API_KEY` | When set, every request must send it as `X-API-Key`. Unset, the API is open - set it before exposing the service. |
| `PROGRAMS_DIR` | Directory for generated discovery programs. Unset: the feature is off. |
| `CODEGEN_MODEL` | Model that writes programs, on the configured provider (default: `LLM_MODEL`). Use a strong coding model. |
| `CODEGEN_RETRY_HOURS` | Least time between automatic regenerations of one site's program (default 24). |
| `CODEGEN_MAX_TURNS`, `CODEGEN_MAX_INPUT_TOKENS` | Budget for writing one program (default 45 turns, 3M input tokens including cached reads). |
| `PATTERNS_DIR` | Directory where learned site knowledge (gate sequences, replayable escalation steps) is kept. Unset: nothing is learned. |

## HTTP API

| Endpoint | |
| --- | --- |
| `POST /v1/discover` | Crawl and return the result as JSON. |
| `POST /v1/discover-stream` | The same crawl as a server-sent event stream (every page, verdict and escalation as it happens). |
| `GET /v1/profiles` | The bundled domain profiles. |
| `POST /v1/search-sites` | Suggest websites for a goal (Anthropic web search). |
| `GET/DELETE /v1/patterns[/{domain}]` | Learned site knowledge. |
| `POST /v1/programs` | Write (or rewrite) a site's discovery program in the background. |
| `GET/DELETE /v1/programs[/{key}]` | Generated programs: code, notes, health. |
| `POST /v1/programs/check` | Replay programs against their snapshots (drift check). |
| `GET /health` | Liveness. |

Main request fields for `/v1/discover`:

| Field | Default | |
| --- | --- | --- |
| `url`, `goal` | | Where to start and what to find. |
| `decision_layer` | judge-driven | `jev` for the judge-driven crawl (despite the name, with any judge), `agent` for the browsing agent alone. |
| `judge` | `jev` if its key is set, else `llm` | Which relevance judge. |
| `profile` | `generic` | Domain profile name, or a path to a profile file. |
| `max_pages`, `max_seconds`, `max_depth` | 10, 180, 3 | Crawl budget. |
| `same_domain_only`, `allowed_hosts` | `true`, `[]` | Off-domain policy: with `same_domain_only: false` the crawl may cross to one host linked from the start site. |
| `latest` | `false` | Keep only the newest document of each series (see above). |
| `programs` | `false` | Answer from the site's generated program when it is healthy (needs `PROGRAMS_DIR`). |
| `include_rejected` | `false` | Also return rejected candidates, to see what the judge threw away. |
| `model` | `LLM_MODEL` | Agent model override. |

The result lists the documents with `relevance`, `verdict`, `source` (page, sitemap, api, agent, program),
`period`, `year`, `published`, `series` and `latest_in_series`, plus `stop_reason`, token counts and which model decided.

## Crawling responsibly

- **robots.txt** is honoured for every host the crawl touches.
- **SSRF protection:** only `http(s)` URLs whose host resolves exclusively to public addresses are fetched or
  returned - a hostname pointing at `127.0.0.1` or a cloud metadata address is refused. The same rule applies
  to URLs a page or a model hands the crawler.
- Pages are visited one at a time by default (`max_concurrent`); set `max_pages` and `max_seconds` to what the
  site can take.
- Page text reaches the models as data. Instruction-like text in a page is flagged and never acted on.
- Crawl only what you are allowed to, and respect each site's terms.

## Evaluating

`eval/` measures recall and precision against hand-checked ground truth: see
[`eval/ground_truth/README.md`](eval/ground_truth/README.md) for the format and the scripts
(`run_jev.py` for the judge-driven crawl, `run_baseline.py` for the agent path, `run_codegen.py` for generated
programs, `judge_compare.py` for scoring a judge offline on a frozen, labelled candidate set).

## Development

```bash
uv pip install -e ".[dev]"
playwright install chromium
pytest
```

The tests run offline: DNS resolution and robots.txt are faked (`tests/conftest.py`) and every model call is
mocked. Browser tests use a local Chromium and are skipped when it is missing.

Contributions are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues privately - see
[SECURITY.md](SECURITY.md).

## Licence

Apache-2.0 - see [LICENSE](LICENSE). TypeSafe Jev is a third-party service with its own terms; its API client
is included, its model is not.
