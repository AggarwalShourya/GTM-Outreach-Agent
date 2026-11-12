# PS-3 · End-to-End Flow (live-only) — how a query becomes a personalized draft

*No seeded/demo branching. Every prospect — your own customers, a C-level exec,
a student — goes through the same pipeline: live search → retrieval → judgment →
draft. This is the story to walk through before the demo.*

## The one-line version

You type **name · role · company**. The process verifies *who that is* from
public reporting (with an evidence-based confidence), pulls the real signals that
matter through a fault-tolerant search chain, hybrid-retrieves and scores them,
resolves contradictions and staleness, picks the one hook worth citing (or
honestly says there isn't one), and drafts a personalized message that a human
reviews before anything sends.

---

## The flow

```
   ┌──────────────────────────────────────────────────────────────┐
   │  QUERY IN:  first · last · role · company                    │
   │  e.g.  Shourya · Aggarwal · Student · IIT Roorkee            │
   └──────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 1 · ENTITY VERIFICATION (who are we writing to?) │
        │                                                        │
        │   Q1 = “<person>”  (name-anchored — survives typos in  │
        │   Q2 = “<person> <company>”      company)              │
        │                                                        │
        │   run both through the provider chain ↓                 │
        │                                                        │
        │   confidence = f(corroborating articles, full-name     │
        │   mentions)   0 art → .20 · 0-name → .35 · 1 → .55    │
        │   2+ → .80 · 3+ → .90                                  │
        └───────────────────────────┬────────────────────────────┘
                                    │
                       conf ≥ 0.5 ?
                    ┌──────────────┴──────────────┐
            no     │                              │  yes
   ┌────────────────┴─────────┐                    │
   │ STOP · confirm panel      │                    │
   │ “confirm, or proceed with │                    │
   │  a low-confidence caveat” │ (override carries a │
   │ (no draft is produced)    │  “NOT verified”     │
   └───────────────────────────┘  caveat)            │
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 2 · THE PROVIDER CHAIN (fault-tolerant fetch)    │
        │                                                        │
        │  serper/serpapi · tavily · bing   (keyed; 403 body     │
        │        │  captured → *why* it failed)                  │
        │        ▼                                               │
        │  Google News RSS  (key-free)                           │
        │        │                                                │
        │        ▼                                                │
        │  DuckDuckGo  (key-free, last resort)                   │
        │                                                        │
        │  Each failure falls through; only when the WHOLE chain │
        │  is down does the process say “unverified” (reported,  │
        │  never disguised as “no evidence found”).              │
        └───────────────────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 3 · GATHER + RETRIEVE (the RAG part)             │
        │                                                        │
        │  query = person + company + role + GTM triggers         │
        │  (“sales OR SDR OR go-to-market OR hiring OR …”)        │
        │  → live docs → chunk (400 chars / 100 overlap)          │
        │                                                        │
        │  HYBRID INDEX (one index, any source):                  │
        │     BM25 (exact lexes)  ⊕  dense embeddings             │
        │     (Google Gemini when key set, local vectors else)    │
        │     → reciprocal-rank fusion → top signals + evidence   │
        │                                                        │
        │  hygiene: junk .cdr/.pdf/.zip hits filtered; same      │
        │  story from two providers deduped (no fake corrob.)    │
        │                                                        │
        │  + PERSON PROFILE: “name linkedin”, “name company”     │
        │    → their search-indexed LinkedIn snippet + public    │
        │    mentions (real, cited, never invented)              │
        └───────────────────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 4 · JUDGE (why these, not others)                │
        │                                                        │
        │  score = Σ axis × weight   (code does the sum)         │
        │     relevance .35 · recency .20 · specificity .20      │
        │     attributability .15 · actionability .10            │
        │  · specificity is GATED on GTM relevance → naming the  │
        │    person ≠ a reason to call them (career retros /     │
        │    bios can't fake a trigger)                           │
        │  · negation-aware: “NOT hiring for sales” ≠ trigger    │
        │  · recency: live docs vs REAL today (≈30d = fresh)     │
        │                                                        │
        │  CONTRADICTIONS: opposing stories → newer wins, older  │
        │  penalized + caveat (“don't cite the stale one”)       │
        │  STALENESS: if a “person left” story post-dates a      │
        │  personal signal → that signal is demoted, not cited   │
        └───────────────────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 5 · PICK THE HOOK                                 │
        │                                                        │
        │  strong = total ≥ 3.5 AND fresh AND no penalty          │
        │     ├─ yes → TRIGGER mode (draft cites the signal)     │
        │     └─ no  → RELATIONSHIP-BUILDER (soft, cites nothing) │
        │        + “volume ≠ relevance” caveat when signals exist │
        │          but none crossed the bar                        │
        └───────────────────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  LAYER 6 · DRAFT                                         │
        │                                                        │
        │  LLM (gemini-flash-lite-latest — cheap, fast) writes    │
        │  the message grounded ONLY in: the chosen signal +      │
        │  the person's public profile + the value prop.          │
        │     └─ on 503/timeout/bad JSON → deterministic template  │
        │        fallback (the UI badge shows which engine ran)   │
        └───────────────────────────┬────────────────────────────┘
                                    │
        ┌───────────────────────────┴────────────────────────────┐
        │  OUTPUT + HUMAN IN THE LOOP                              │
        │  draft + chosen hook + full score math + caveats +     │
        │  confidence + “grounded in” source list                │
        │  → saved to dashboard (history) → a human reviews      │
        │     before anything is ever sent                        │
        └─────────────────────────────────────────────────────────┘
```

---

## The confidence rule (Layer 1, single source of truth)

| Evidence | Full-name mentions | Confidence | Verdict |
|---|---|---|---|
| whole chain down | — | 0.20 | `unverified` → stop, note names the provider errors |
| 0 corroborating articles | — | 0.20 | `no-evidence` → stop |
| ≥1 article, none names them | 0 | 0.35 | `weak` → stop (“company real, person unconfirmed”) |
| 1 | 1 | 0.55 | proceed + unconfirmed-band caveat |
| 2+ | 2+ | 0.80 | proceed |
| 3+ | 3+ | 0.90 | proceed, high band |

Gate: **< 0.5 → stop-and-confirm** (no draft). Override is allowed but attaches
a “NOT verified” caveat and lowers the draft's posture — the human-in-the-loop seam.

## Why each layer exists (the 30-second "why" for each)

- **Layer 1 (verification)** — the cost of being wrong is asymmetric: drafting
  on the wrong entity sends a message citing a story about a different person.
  So confidence is *evidence-proportional* and a stop is a *decision*, not a failure.
- **Layer 2 (chain)** — live systems fail; one dead provider must never kill
  the run. Order: your keyed provider → key-free RSS → key-free DuckDuckGo.
  Every failure is captured *with its reason* (403 body) so the UI says
  “serper: out of credits” instead of “no evidence found.”
- **Layer 3 (RAG)** — “finding the signal *is* the problem.” Hybrid =
  BM25 (exact terms like “$40M Series B”) + dense embeddings (semantics)
  fused by reciprocal-rank fusion; evidence chunks travel with every signal
  so the draft is literally grounded in retrieved text.
- **Layer 4 (judge)** — *model judges, code sums*: an LLM (or heuristic) scores
  five bounded axes with one-line reasons; the weighted sum is plain Python,
  so any “why did this hook win” is reproducible math.
- **Layer 5 (hook)** — strong vs. soft is a *named threshold*, not a vibe:
  ≥3.5 weighted, fresh ≤90d, no penalty. Otherwise relationship-builder + honest caveat.
- **Layer 6 (draft)** — the LLM may only use what was retrieved (signal +
  profile). Fallback keeps the run alive; the badge keeps it honest.
