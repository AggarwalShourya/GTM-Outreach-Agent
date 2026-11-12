"""Retrieval layer. Real hybrid search over corpus *and* live web docs.

The capability comes from the models, not from hand-tuned thresholds:

  dense channel  -> hosted embeddings (Google real model when a key is set,
                    deterministic local vectors otherwise) via embed_real.
  lexical channel -> Okapi BM25 over the chunked corpus.
  fusion         -> reciprocal-rank fusion (the one standard knob we keep).
  recency        -> a *tie-break* (fresher wins at equal relevance), not a
                    multiplicative weight — one of the knobs we deliberately
                    trimmed.
  LLM rerank     -> optional: an LLM re-scores the top-N fused chunks. Off by
                    default so the run stays fast and key-free.

Document source:
  seeded company   -> the raw articles in documents.py (deterministic edge
                      cases stay reproducible mid-interview). Recency anchors
                      to the demo reference date.
  unseeded company -> LIVE web search (web_search.py): fresh, dated,
                      linkable docs, chunked + embedded into the same index.
                      Recency anchors to the REAL wall-clock today, so live
                      news is scored on its true age.
"""

import datetime as _dt
import math
import re

from . import corpus, config, normalize
from .documents import all_docs
from .embed_real import embed_batch, provider as embed_provider
from .bm25 import BM25Index
from .embeddings import cosine

# The few knobs we kept, deliberately.
RRF_K = config.RRF_K
TOP_K = config.TOP_K


# --------------------------------------------------------------------- chunking
def chunk_text(text, size=400, overlap=100):
    """Sliding-window chunker with overlap; breaks near sentence boundaries."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + size)
        if end < n:
            sp = text.rfind(" ", start + size - 60, end)
            if sp > start:
                end = sp
        ch = text[start:end].strip()
        if ch:
            chunks.append(ch)
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def real_today():
    """Real wall-clock date, ISO. Live docs anchor recency to this, not to the
    demo reference date."""
    return _dt.date.today().isoformat()


# --------------------------------------------------------------------- retriever
class Retriever:
    """Builds a chunked, embedded, BM25-indexed search engine over a doc list.

    `embed_batch` is used for the dense channel (real model or local fallback),
    so this class is source-agnostic: corpus docs and live-search docs go
    through the identical path.
    """

    def __init__(self, docs, reference_date=None):
        self.docs = docs
        self.reference_date = reference_date or corpus.REFERENCE_DATE
        self._chunks = self._build_chunks()
        self.bm25 = BM25Index(self._chunks)
        texts = [c["text"] for c in self._chunks]
        vecs, self._embed_used = embed_batch(texts)
        self._chunk_vec = {c["id"]: v for c, v in zip(self._chunks, vecs)}
        self._by_chunk = {c["id"]: c for c in self._chunks}

    def _build_chunks(self):
        out = []
        for d in self.docs:
            text = f"{d['title']}. {d['body']}"
            for i, ch in enumerate(chunk_text(text)):
                out.append({
                    "id": f"{d['id']}#c{i}", "doc_id": d["id"], "text": ch,
                    "company_key": d.get("company_key", "live"),
                    "company_name": d.get("company_name", d.get("id")),
                    "source": d.get("source", "press"),
                    "type": d.get("type", d.get("source", "press")),
                    "published": d.get("published", self.reference_date),
                    "url": d.get("url", ""),
                    "title": d.get("title", ""),
                    "doc_body": d.get("body", ""),
                    "retrieval": d.get("retrieval", "seeded-corpus"),
                    # recency reference: live docs -> real today; seeded -> demo date
                    "recency_ref": (real_today() if d.get("retrieval", "").startswith("live")
                                    else self.reference_date),
                })
        return out

    # ------------------------------------------------------------- hybrid
    def search(self, query, doc_ids=None, top=TOP_K):
        """Hybrid BM25+vector search, RRF-fused. `doc_ids` restricts the pool
        to a set of document ids (used to confine a search to one company)."""
        doc_pool = set(doc_ids) if doc_ids else None
        pool = [c for c in self._chunks
                if (doc_pool is None or c["doc_id"] in doc_pool)]
        if not pool:
            return []
        pool_ids = [c["id"] for c in pool]
        idset = set(pool_ids)
        pool_by_id = {c["id"]: c for c in pool}

        # --- BM25 (postings are global; restrict to the pool)
        bm25_pairs = [(cid, sc) for cid, sc in
                      self.bm25.score_all(query, top_k=len(pool) + 5)
                      if cid in idset]
        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_pairs)}
        bm25_score = dict(bm25_pairs)

        # --- vector (dense: real Google embeddings when a key is set, else local)
        qv = _embed_one(query)
        vec = []
        for cid in pool_ids:
            vec.append((cid, cosine(qv, self._chunk_vec[cid])))
        vec.sort(key=lambda x: -x[1])
        vec_rank = {cid: r for r, (cid, _) in enumerate(vec)}

        # --- reciprocal-rank fusion (the one kept constant)
        fused = {}
        for cid, _ in bm25_pairs:
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + bm25_rank[cid] + 1)
        for cid, _ in vec:
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + vec_rank[cid] + 1)
        fused_rank = sorted(fused.items(), key=lambda kv: -kv[1])

        out = []
        for rank, (cid, fval) in enumerate(fused_rank[:top]):
            c = pool_by_id[cid]
            c["_rrf"] = round(fval, 5)
            c["_mmr_rank"] = rank
            c["_evidence"] = c["text"][:300]
            out.append(c)
        return out

    def rerank(self, query, chunks, llm):
        """Optional LLM re-score of the top chunks. Returns re-ordered chunks."""
        if not (config.RERANK_ENABLED and llm and getattr(llm, "rerank", None)):
            return chunks
        head = chunks[: config.RERANK_TOP_N]
        try:
            order = llm.rerank(query, [c["_evidence"] for c in head])
            return [head[i] for i in order] + chunks[config.RERANK_TOP_N:]
        except Exception:
            return chunks

    # --------------------------------------------------- doc -> signal
    def to_signals(self, chunks, first, last, role):
        """Cluster to one signal per doc, attach evidence + text-derived flags."""
        name = f"{first} {last}".strip()
        by_doc = {}
        for c in chunks:
            cur = by_doc.get(c["doc_id"])
            if cur is None or c["_rrf"] > cur["_rrf"]:
                by_doc[c["doc_id"]] = c
        doc_chunks = sorted(by_doc.values(), key=lambda c: -c["_rrf"])

        resolved_docs = [self._find_doc(c["doc_id"]) for c in doc_chunks]
        move_doc = self._find_move_doc(name, [d for d in resolved_docs if d])
        move_date = move_doc["published"] if move_doc else None
        move_id = move_doc["id"] if move_doc else None
        doc_chunks = [c for c in doc_chunks if c["doc_id"] != move_id]

        out = []
        for c in doc_chunks:
            doc = self._find_doc(c["doc_id"])
            sig = self._to_signal(doc, c)
            text = doc["body"] + " " + c["text"]
            sig["person_related"] = _mentions_person(text, name, role)
            sig["relevance_base"] = _gtm_relevance(text, doc["source"])
            sig["role_context"] = _role_context(sig, move_doc, move_date)
            sig["role_note"] = (move_doc["title"] if (move_doc and sig.get("person_related")
                                                      and sig["published"] < move_date) else None)
            sig["recency_ref"] = c.get("recency_ref", self.reference_date)
            out.append(sig)
        # recency as a TIE-BREAK, not a multiplier
        out.sort(key=lambda s: (-s["_rrf"], s["published"]))
        return out

    def _find_doc(self, doc_id):
        return next((d for d in self.docs if d["id"] == doc_id), None)

    def _find_move_doc(self, name, docs):
        first = name.split()[0].lower() if name else ""
        for d in docs:
            if not d:
                continue
            body = d["body"].lower()
            if _mentions_move(body, name) and first and first in body:
                return d
        return None

    def _to_signal(self, doc, chunk):
        evidence = chunk["_evidence"]
        return {
            "id": doc["id"], "company_key": doc.get("company_key", "live"),
            "company_name": doc.get("company_name", doc["id"]),
            "source": doc["source"], "date": doc.get("published", ""),
            "published": doc.get("published", ""),
            "type": doc.get("type", doc["source"]),
            "headline": doc["title"], "summary": _first_meaningful_sentence(evidence),
            "url": doc.get("url", ""), "_rrf": chunk["_rrf"], "evidence": evidence,
            "retrieval": doc.get("retrieval", "seeded-corpus"),
        }


# ------------------------------------------------------------------ query build
def build_query(company_key, name, role):
    """Prospect-specific query: person + company + GTM trigger vocabulary."""
    company_human = company_key.replace("-", " ") if company_key not in ("unknown", "live") else ""
    parts = [name, company_human, role or "",
             "sales SDR go-to-market prospect research outbound pipeline"]
    return " ".join(p for p in parts if p)


# ------------------------------------------------------------------ text flags
GTM_TRIGGERS = ["sdr", "sales", "go-to-market", "gtm", "outbound", "prospect",
                "propecting", "research", "pipeline", "reply rate", "commercial team"]
NEGATION_PATTERNS = ["not hiring", "no sales", "does not discuss", "says nothing about",
                     "no customer-facing", "no outbound", "no go-to-market", "not about"]


def _first_meaningful_sentence(text):
    for cand in re.split(r"(?<=[.!?])\s+", text):
        if len(cand) >= 40:
            return cand[:280]
    return text[:280]


def _mentions_person(text, name, role):
    full = " ".join(p for p in name.lower().split() if p)
    return bool(full and len(full) > 4 and full in text.lower())


def _mentions_move(body, name):
    t = body.lower()
    first = (name.split()[0].lower() if name else "")
    return (("left" in t or "depart" in t or "joining" in t or "joined" in t) and first in t)


def _gtm_relevance(text, source):
    """Text-derived 1-5 base relevance. Negation-aware: a doc that says 'we are
    NOT hiring for sales' floors to 1 so volume never masquerades as a hook.
    This is the 'judge' — source quality lives here, not in a rerank weight."""
    t = text.lower()
    if any(p in t for p in NEGATION_PATTERNS):
        return 1
    hits = sum(1 for g in GTM_TRIGGERS if g in t)
    if hits >= 4:
        return 5
    if hits >= 3:
        return 4
    if hits >= 2:
        return 3
    if hits == 1:
        return 2
    return {"funding": 5, "exec_interview": 4, "blog": 4,
            "job_postings": 3, "product": 2, "social": 3, "news": 2, "press": 2}.get(source, 2)


def _role_context(sig, move_doc, move_date):
    if not sig.get("person_related"):
        return "company-level"
    if move_doc and sig["published"] < move_date:
        return "stale"
    return "aligns"


# ------------------------------------------------------- embedding helper
def _embed_one(text):
    """Embed a single query text. Uses the configured provider (Google real
    embeddings when a key is set; local deterministic vectors otherwise)."""
    from .embed_real import embed_batch as _eb
    vecs, _ = _eb([text])
    return vecs[0]


# ------------------------------------------------------------------ public API
_CORPUS_RT = None


def _corpus_doc_ids_for(company_key):
    return [d["id"] for d in all_docs() if d["company_key"] == company_key]


def _corpus_docs_for(company_key):
    return [d for d in all_docs() if d["company_key"] == company_key]


def get_corpus_retriever():
    """Cache one Retriever over the whole seeded corpus (real embeddings when
    a key is present; local vectors otherwise)."""
    global _CORPUS_RT
    if _CORPUS_RT is None:
        _CORPUS_RT = Retriever(all_docs())
    return _CORPUS_RT


def gather_signals(company_key, first, last, role, company_name=None, llm=None):
    """Public entry point the pipeline calls. Returns (signals, query, meta).

    meta carries what actually happened, so the run view can be honest:
    which embedding provider, whether live search fired, doc count, etc.
    """
    from . import web_search
    name = f"{first} {last}".strip()
    query = build_query(company_key, name, role)
    meta = {"query": query, "embed_provider": embed_provider(), "live_search": False}

    if company_key != "unknown":
        # Seeded company: deterministic corpus. (Optional live augmentation.)
        rt = get_corpus_retriever()
        doc_ids = _corpus_doc_ids_for(company_key)
        docs = _corpus_docs_for(company_key)
        if config.AUGMENT_SEEDED:
            live_docs = web_search.search(company_key, first, last, role)
            if live_docs:
                docs = docs + live_docs
                meta["live_search"] = True
                rt = Retriever(docs, reference_date=corpus.REFERENCE_DATE)
                doc_ids = None
        chunks = rt.search(query, doc_ids=doc_ids, top=TOP_K)
        chunks = rt.rerank(query, chunks, llm)
        signals = rt.to_signals(chunks, first, last, role)
    else:
        # Unseeded company: LIVE web search -> same index path. Recency anchors
        # to the real wall-clock today so live news is scored on its true age.
        live_docs = web_search.search(company_name or company_key, first, last, role)
        meta["live_search"] = bool(live_docs)
        if not live_docs:
            meta["note"] = "live search returned nothing; taking the quiet path"
            return [], query, meta
        rt = Retriever(live_docs, reference_date=real_today())
        chunks = rt.search(query, doc_ids=None, top=TOP_K)
        chunks = rt.rerank(query, chunks, llm)
        signals = rt.to_signals(chunks, first, last, role)

    meta["doc_count"] = len(rt._chunks) if rt else 0
    meta["signal_count"] = len(signals)
    meta["company_name"] = company_name or company_key
    return signals, query, meta
