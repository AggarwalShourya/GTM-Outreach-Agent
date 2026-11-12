# GTM Outreach Agent

Given a prospect — **name · role · company** — it verifies who the person is
from public reporting, retrieves and scores the live signals that matter, and
drafts a short, personalized outbound email that a human reviews before
sending. Six stages, each streamed with its own evidence.

## Layout

The package lives in `ps3/` (a real Python package); `app.py` sits at the
repo root and is the Streamlit entry point. Run from the repo root.

## Run it locally

```bash
git clone https://github.com/AggarwalShourya/GTM-Outreach-Agent.git
cd GTM-Outreach-Agent
cp env.example .env        # paste GOOGLE_API_KEY + SERPER_API_KEY

streamlit run app.py       # -> http://localhost:8501
# or the zero-dependency web server:
python -m ps3.server       # -> http://localhost:8377
```

`streamlit` is the only dependency; the pipeline itself is stdlib.

## Deploy on Streamlit Cloud

- **App main file:** `app.py` (from the repo root)
- **Keys:** add `GOOGLE_API_KEY` and `SERPER_API_KEY` under *App editor →
  Secrets* — `app.py` reads them via `st.secrets`; they are never committed.

No keys? Still runs — key-free fallbacks (Google News RSS, DuckDuckGo, local
embeddings); the draft stage just reports "not generated" instead of shipping
a canned email. Python 3.10+.

## The pipeline

```
1 verify identity (2 name-anchored queries, confidence from independent
   corroboration; < 0.5 -> stop & confirm)
2 hybrid RAG retrieval (live docs -> chunks -> BM25 + embeddings -> RRF;
   person profile with name-collision guard)
3 score & rank (5 axes, fixed weights: relevance .35, recency .20,
   specificity .20, attributability .15, actionability .10)
4 contradiction & staleness (newer story wins; "person left" demotes stale
   signals)
5 pick the hook (total >= 3.0 + fresh <= 90d -> trigger; else soft
   relationship-builder, still opens on the best available icebreaker)
6 LLM draft (Gemini; given up to 5 ranked icebreaker candidates and picks
   the one that is professional, specific, and hard to ignore; awkward
   beats like departures/layoffs are excluded; grounded only in retrieved
   facts)
```
## Configuration (`.env`)

| Var | For | Default |
|---|---|---|
| `GOOGLE_API_KEY` | embeddings + LLM draft | unset -> local vectors, no draft |
| `SERPER_API_KEY` | live search (first in provider chain) | unset -> RSS -> DuckDuckGo |
| `PS3_LLM_MODEL` | draft model | `gemini-flash-lite-latest` |
| `PS3_LLM_JUDGE` | LLM scores axes instead of the heuristic | `0` |

Full list: `env.example`.

## Honest limitations

- Confidence scales with **notoriety**, not correctness (many articles ≈ high
  score, even on thin identity evidence).
- LinkedIn is **snippet-only** — full profiles sit behind a login wall; we
  cite what search indexed, never invent.
- Name collisions are guarded in the profile layer, not every SERP hit.
- The freshest news (last ~48h) is often below the search index's
  resolution — the best hook the moment it happens is the hardest to catch.
