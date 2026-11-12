"""PS-3 Personalised Outreach — package.

Modules:
  corpus             seeded demo prospects + value prop + reference date
  documents          raw article corpus (21 docs, 6 companies)
  normalize          text helpers
  scoring            axis definitions, judge (heuristic default), weighted sum
  retrieval          hybrid RAG: chunk -> BM25 + dense embeddings -> RRF
  bm25               Okapi BM25 index
  embeddings         deterministic hashed char-n-gram vectors (offline dense channel)
  embed_real         real embedding providers: Google (API key + Vertex), OpenAI, Cohere
  web_search         live web-search retrievers: Tavily / Serper / Bing + key-free RSS
  live_retrieval     key-free Google News RSS fetcher (the fallback live path)
  entity_resolution  live identity verification against public reporting
  pipeline           the 6-stage orchestrator
  data               SQLite run-history store (dashboard)
  llm_optional       LLM judge/draft/rerank plug-in point
  server             zero-dependency HTTP server: UI + SSE JSON API
  config             env-driven provider resolution + knobs

Run:  python -m ps3.server  ->  http://localhost:8377
"""

__all__ = ["corpus", "documents", "normalize", "scoring", "retrieval", "bm25",
           "embeddings", "embed_real", "web_search", "live_retrieval",
           "entity_resolution", "pipeline", "data", "llm_optional", "server", "config"]
