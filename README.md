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
git clone <this repo> && cd docseek
uv venv && source .venv/bin/activate      # or python -m venv .venv
uv pip install -e ".[dev]"
playwright install chromium

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
| `PATTERNS_DIR` | Directory where learned site knowledge (gate sequences, replayable escalation steps) is kept. Unset: nothing is learned. |

## HTTP API

| Endpoint | |
| --- | --- |
| `POST /v1/discover` | Crawl and return the result as JSON. |
| `POST /v1/discover-stream` | The same crawl as a server-sent event stream (every page, verdict and escalation as it happens). |
| `GET /v1/profiles` | The bundled domain profiles. |
| `POST /v1/search-sites` | Suggest websites for a goal (Anthropic web search). |
| `GET/DELETE /v1/patterns[/{domain}]` | Learned site knowledge. |
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
| `include_rejected` | `false` | Also return rejected candidates, to see what the judge threw away. |
| `model` | `LLM_MODEL` | Agent model override. |

The result lists the documents with `relevance`, `verdict`, `source` (page, sitemap, api, agent), `period` and
`year`, plus `stop_reason`, token counts and which model decided.

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
(`run_jev.py` for the judge-driven crawl, `run_baseline.py` for the agent path, `judge_compare.py` for
scoring a judge offline on a frozen, labelled candidate set).

## Development

```bash
uv pip install -e ".[dev]"
playwright install chromium
pytest
```

The tests run offline: DNS resolution and robots.txt are faked (`tests/conftest.py`) and every model call is
mocked. Browser tests use a local Chromium and are skipped when it is missing.

## Licence

Apache-2.0 - see [LICENSE](LICENSE). TypeSafe Jev is a third-party service with its own terms; its API client
is included, its model is not.
