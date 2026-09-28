# Security

## Reporting a vulnerability

Please report vulnerabilities privately, through GitHub: **Security → Report a vulnerability** on this repository
(https://github.com/mohojojo/docseek/security/advisories/new). Do not open a public issue for them.

Include what you did, what happened and what you expected, and the docseek version (`pip show docseek`). Fixes
ship as a new release with a note in its release notes.

Only the latest release is supported.

## What counts

docseek fetches pages it is pointed at, hands their text to language models, and - with generated programs on -
runs code a model wrote. These are its security boundaries, and a way around any of them is a vulnerability:

- **Fetching a private address.** Every URL a crawl, an agent or a program requests must resolve only to public
  addresses (`docseek.reach`). Reaching `127.0.0.1`, a cloud metadata address or an internal host by any route -
  redirects, DNS tricks, a URL a page or a model supplied - is in scope.
- **robots.txt.** A crawl fetching what a site's robots.txt disallows.
- **A generated program reaching past its fetcher.** A program may reach the web only through the parent's fetcher
  (the site, its subdomains, and the data hosts and POST endpoints the site's own pages used). A program that opens
  a socket, starts a process, reads or writes a file, or reads an API key is in scope.
- **Page text taking control.** Instruction-like text in a page that makes an agent act on it - navigate off the
  allowed hosts, submit a form, exfiltrate data.
- **The API.** Bypassing `CRAWLER_API_KEY`, or reaching files outside `PROGRAMS_DIR` or `PATTERNS_DIR` through a
  crafted program key or domain.

## Known limits

- **The program sandbox is a boundary, not a hardened jail.** It runs each program in a fresh interpreter with an
  empty environment, OS resource limits, an import allowlist and an audit hook. A determined escape from CPython
  (for example through object introspection) is possible and is documented: an escape that stays inside the child
  process - with no network, no secrets and no files - is not in scope; one that reaches past it is. Generated
  programs are off unless `PROGRAMS_DIR` is set, and a docseek that strangers can reach belongs in a container.
- **Relevance is a model's judgement.** A page that talks a judge into accepting an irrelevant document is a
  quality problem, not a security one - unless it leads to one of the actions above.
