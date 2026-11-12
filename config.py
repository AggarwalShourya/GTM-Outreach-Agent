"""Central, env-driven configuration.

All provider resolution + knobs live here so the interviewer can point at
ONE place and see the trade-offs. Overridable via env vars / a `.env` file
(the server loads it at startup):

  * No keys            -> corpus + local hashed vectors + key-free RSS. Fully
    offline and deterministic.
  * GOOGLE_API_KEY     -> real Google embeddings (text-embedding-004, 768-d).
  * PS3_VERTEX=1       -> Vertex AI embedding endpoint (project-scoped).
  * TAVILY/SERPER/BING -> live web search for unseeded companies;
    without any, Google News RSS is used (key-free).

Trimmed knobs (kept deliberately few): RRF_K (standard constant),
TOP_K, recency-as-tiebreak, optional LLM rerank. No MMR, no source-
weight multipliers — source quality is the judge's job (scoring).
"""

import os

def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def embedding_provider():
    if os.environ.get("PS3_VERTEX") in ("1", "true"):
        return "vertex"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("VERTEX"):
        return "google"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_EMBEDDING_KEY"):
        return "openai"
    if os.environ.get("COHERE_API_KEY"):
        return "cohere"
    return "local"

EMBEDDING_MODEL = os.environ.get("PS3_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.environ.get("PS3_EMBED_DIM", "512"))
VERTEX_PROJECT = os.environ.get("PS3_VERTEX_PROJECT", "")
VERTEX_LOCATION = os.environ.get("PS3_VERTEX_LOCATION", "us-central1")
VERTEX_SERVICE_ACCOUNT = os.environ.get("PS3_VERTEX_SA", "")

def _configured_providers():
    """Providers with a key set, in preferred order. Serper (serper.dev) and
    SerpApi (serpapi.com) are DIFFERENT services with different endpoints, so
    each gets its own provider id keyed to its own env var."""
    p = []
    if os.environ.get("TAVILY_API_KEY"):
        p.append("tavily")
    if os.environ.get("SERPER_API_KEY"):
        p.append("serper")
    if os.environ.get("SERPAPI_KEY"):
        p.append("serpapi")
    if os.environ.get("BING_SEARCH_KEY"):
        p.append("bing")
    return p

def search_provider():
    order = os.environ.get("PS3_SEARCH_PROVIDER", "auto").strip().lower()
    providers = _configured_providers()
    if order == "auto":
        providers += ["rss", "ddg"]  # key-free fallbacks, in order, always last
        return providers[0] if providers else "none"
    return order if (order in providers or order in ("rss", "ddg", "none")) else "none"

AUGMENT_SEEDED = _env_bool("PS3_AUGMENT_SEEDED", False)
LIVE_SEARCH = _env_bool("PS3_LIVE", True)
SEARCH_TOP_N = int(os.environ.get("PS3_SEARCH_TOP", "8"))
ENTITY_PROBE_TOP = int(os.environ.get("PS3_ENTITY_PROBE_TOP", "10"))

RRF_K = 60
TOP_K = 24
USE_RECENCY_TIEBREAK = True
RERANK_TOP_N = int(os.environ.get("PS3_RERANK_TOP", "6"))
RERANK_ENABLED = _env_bool("PS3_RERANK", False)
