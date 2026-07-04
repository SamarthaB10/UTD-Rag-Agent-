# Controlled UT Dallas RAG Agent

This agent searches the local Student Handbook collection first, applies a
deterministic evidence gate, and only then falls back to fetched content from
allowlisted official domains. Search snippets are never used as evidence.

## Configuration

Create `.env` with:

```dotenv
OPENAI_API_KEY=...
UTD_HANDBOOK_PATH=/absolute/path/to/StudentHandbook.pdf

# Optional trusted-web fallback (Google Programmable Search JSON API)
GOOGLE_CSE_API_KEY=...
GOOGLE_CSE_ID=...

# Or use Tavily for discovery
TAVILY_API_KEY=...
```

Web search is optional. Google CSE is preferred when its two variables are
configured; otherwise Tavily is used when `TAVILY_API_KEY` exists. Without a
provider, the agent abstains when local evidence is insufficient. Search results
are discovery URLs only: the application fetches, extracts, and validates the
actual source before using it as evidence.

Run the CLI:

```bash
.venv/bin/python main.py
```

Run the offline test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The API-facing result has stable fields: `answer`, `evidence_source`,
`citations`, `next_step`, and `needs_official_confirmation`.
