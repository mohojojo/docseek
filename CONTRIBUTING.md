# Contributing

Thank you for helping. Issues and pull requests are welcome; for anything larger than a fix, open an issue first
so we can agree on the shape before you write it.

## Setup

```bash
git clone https://github.com/mohojojo/docseek && cd docseek
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
playwright install chromium
```

## Before a pull request

```bash
ruff check .
pytest
```

The tests run offline: DNS resolution and robots.txt are faked (`tests/conftest.py`), every model call is mocked,
and browser tests use a local Chromium. A pull request must keep them offline - no test may need an API key or
reach the internet. CI runs both commands.

## When the change affects what a crawl finds

Unit tests cannot tell whether a change finds more documents or fewer. Measure it on sites with known answers:
write a ground-truth file for a site you know (see [`eval/ground_truth/README.md`](eval/ground_truth/README.md)),
run `eval/run_jev.py` before and after the change, and say in the pull request what moved. Ground-truth files are
yours - they are ignored by git, and you need not share them.

Some things look cosmetic and are not: the page-kind ids in `docseek/profile.py`, and the wording of the questions
in `docseek/judge.py`, are part of what the judge reads. Change them only with a before/after eval.

## How the code decides

- **Code decides what it can decide reliably; a judge decides the rest.** Deterministic rules handle the clear
  cases. When a rule would have to guess, the question goes to the relevance judge instead of into another special
  case.
- **When in doubt, keep.** A wrong keep costs a caller one extra document; a wrong drop loses one silently.
- **Any model.** New model calls go through `docseek.llm` (`LLMClient`), never a provider SDK directly, so they work
  with Anthropic and every OpenAI-compatible endpoint.
- **Reach rules everywhere.** Every URL that is fetched or returned passes `docseek.reach` (public addresses only,
  robots.txt).

## Security issues

Please do not open a public issue: see [SECURITY.md](SECURITY.md).
