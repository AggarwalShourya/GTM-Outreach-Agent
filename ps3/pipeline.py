"""The PS-3 pipeline: one call in, an explainable outreach draft out.

Stages (each emits its intermediate artifacts so the run view is transparent):
  1. entity_resolve -> 2. gather_signals -> 3. score -> 4. contradictions
  -> 5. select_hook -> 6. draft

Design choices, made concrete:
  * Model judges, code sums: per-signal axis scores (heuristic offline /
    LLM live) + fixed weighted sum in code.
  * Live entity resolution (the PIVOT): non-seeded prospects are verified
    against public reporting (entity_resolution.py) — confidence is
    evidence-based, not "was it in my corpus". A low-confidence stop is
    deliberate; the `confirmed` override lets a human push through to the
    live retrieval path with a low-confidence caveat.
  * Hybrid retrieval: chunked docs -> BM25 + dense embeddings -> RRF.
    Seeded companies use the deterministic corpus; unseeded companies pull
    FRESH docs from live web search (person + company queries) into the
    same index; failure degrades to the quiet path — never a crash.
"""

import datetime as _dt

from . import corpus, config
from .normalize import age_days, slugify
from .scoring import score_signal, AXES, WEIGHTS, WEIGHTS_SUM
from .entity_resolution import resolve as resolve_identity

# ---- thresholds (the deliberate judgment, kept in one place) --------------
ACT_THRESHOLD = 0.5      # below this we stop and ask for identity confirmation
LOW_THRESHOLD = 0.75     # 0.5-0.75 proceed but flag "unconfirmed" in caveats
STRONG_HOOK = 3.0        # relaxed bar: weighted total at/above which we call it trigger mode
FRESH_DAYS = 90          # a signal older than this isn't "fresh" on its own


def _age(s):
    """Age in days using the signal's OWN recency anchor: live web docs anchor
    to the real wall-clock today; seeded corpus docs anchor to the demo
    reference date. (This is the fix for real prospects' news looking 'ancient'.)"""
    ref = s.get("recency_ref") or corpus.REFERENCE_DATE
    return age_days(s.get("date") or s.get("published") or "", ref)


def _recompute(b):
    """Re-sum the weighted total after a penalty. Sum of weights is 1.0."""
    return sum(b[k]["score"] * b[k]["weight"] for k in WEIGHTS if k in b)


def run(first, last, role, company, llm=None, on_stage=None, confirmed=False):
    """Run the full pipeline. Returns (stages, result, status).
    `on_stage`, if given, is called after each stage is appended so a caller
    (the SSE server) can stream stages live as they execute.
    `confirmed=True` is the human-in-the-loop override: "I confirm this is the
    right person" — bypasses the stop-on-low-confidence branch so an
    unverified prospect still runs (low-confidence caveat attached).
    """
    stages = []

    def emit(stage):
        if on_stage:
            on_stage(stage)

    def _add(stage):
        stages.append(stage)
        emit(stage)

    # ------------------------------------------------------------------ stage 1
    # PIVOT: entity resolution is live, evidence-based — not a corpus check.
    #   seeded    -> deterministic corpus resolution (edge cases stay stable)
    #   unseeded  -> verify "person + company" against public reporting:
    #                 corroborating articles + full-name mentions -> confidence
    s = {"id": "entity_resolve", "name": "Entity resolution", "status": "ok"}
    prospect = corpus.find_prospect(first, last, company)
    res = resolve_identity(first, last, role, company, prospect=prospect)
    s["data"] = {
        "match": res["match"], "confidence": res["confidence"],
        "candidates": res["candidates"], "note": res["note"],
        "source": res.get("source", "seeded-corpus"), "live": res.get("live", False),
        "offline": res.get("offline", False),
        "queries": res.get("queries", []),
    }
    conf = res["confidence"]
    s["detail"] = (
        f"Live-verified against public reporting ({res.get('source', 'corpus')}): "
        f"{conf:.0%} confidence." if res.get("live")
        else f"Resolved against the demo corpus at {conf:.0%} confidence.")
    _add(s)

    result = {"prospect": {"first": first, "last": last, "role": role, "company": company,
                           "slug": slugify(f"{first} {last} {company}")}}
    result["entity"] = {"match": res["match"], "confidence": conf,
                        "candidates": res["candidates"], "source": res.get("source")}

    # ---- branch: stop on unconfirmed identity (unless human overrode) -----
    if conf < ACT_THRESHOLD and not confirmed:
        s["status"] = "stopped"
        result["confidence"] = "low"
        result["mode"] = None
        result["status"] = "needs_confirmation"
        result["caveats"] = [res["note"]]
        result["top_hook"] = None
        result["draft"] = None
        result["confirmation_message"] = (
            f"Couldn't confirm '{first} {last}' at '{company}' from public reporting "
            "before writing anything — could you confirm the identity so the outreach "
            "goes to the right person and doesn't cite a story about the wrong one? "
            "Or override: run anyway with a low-confidence caveat."
        )
        return stages, result, "needs_confirmation"
    if conf < ACT_THRESHOLD and confirmed:
        s["status"] = "proceeded"
        result["override"] = True

    company_key = prospect["company_key"] if prospect else "unknown"

    # ------------------------------------------------------------------ stage 2
    # Seeded -> hybrid search over the deterministic corpus.
    # Unseeded -> LIVE web search (person + company + GTM triggers) into the
    # same chunk/embed/index path; empty result degrades to the quiet path.
    from . import retrieval
    s2 = {"id": "gather_signals", "name": "Gather signals (hybrid retrieval)", "status": "ok"}
    raw, query, meta = retrieval.gather_signals(company_key, first, last, role,
                                                 company_name=company, llm=llm)
    embed_used = meta.get("embed_provider", "local")
    retrieval_mode = ("hybrid BM25 + " + embed_used + " embeddings, reciprocal-rank fusion"
                      + (" + live web search" if meta.get("live_search") else ""))
    s2["data"] = {"count": len(raw), "retrievers": _retrievers_used(raw), "query": query,
                  "retrieval": retrieval_mode,
                  "embed_provider": embed_used,
                  "live_search": meta.get("live_search", False),
                  "company_name": meta.get("company_name", company),
                  "items": [{"id": r["id"], "source": r.get("source", "seeded"),
                             "origin": r.get("retrieval", "seeded-corpus"),
                             "date": r.get("published", r.get("date", "")),
                             "date_known": r.get("date_known", True),
                             "headline": r["headline"], "url": r.get("url", ""),
                             "evidence": r.get("evidence", "")} for r in raw]}
    s2["detail"] = (f"{len(raw)} public artifacts retrieved & ranked "
                    f"(dense channel: {embed_used})." if raw
                    else "No public artifacts retrieved — quiet company or live search "
                         "returned nothing; using fallback.")
    _add(s2)

    # ---- person profile (live prospects): who they are, beyond one headline
    # LinkedIn is used via its search-indexed SERP snippet (full profiles are
    # login-walled). Seeded demo personas keep their deterministic behavior.
    profile = None
    if company_key == "unknown" and config.LIVE_SEARCH:
        from . import web_search as _ws
        try:
            profile = _ws.profile_person(first, last, role, company)
            if not profile.get("available"):
                profile = None
        except Exception:
            profile = None
    result["profile"] = profile

    # ------------------------------------------------------------------ stage 3
    s3 = {"id": "score", "name": "Score & rank signals", "status": "ok"}
    scored = [{"signal": r, "breakdown": score_signal(r, prospect, llm=llm)} for r in raw]
    scored.sort(key=lambda x: x["breakdown"]["total"], reverse=True)
    s3["data"] = {
        "axes": [{"key": k, "weight": w, "desc": d} for k, w, d in AXES],
        "ranked": (
            [{"id": x["signal"]["id"], "headline": x["signal"]["headline"],
              "total": x["breakdown"]["total"], "breakdown": x["breakdown"]}
             for x in scored[:3]]
            if scored else []
        ),
    }
    s3["detail"] = "Each axis judged 1-5 with a reason; weighted sum computed in code."
    _add(s3)

    # ------------------------------------------------------------------ stage 4
    s4 = {"id": "contradictions", "name": "Contradiction & staleness check", "status": "ok"}
    caveats = []
    conflicts = _detect_conflicts([x["signal"] for x in scored])
    stale = [x for x in scored if x["signal"].get("role_context") == "stale"]
    if confirmed and conf < ACT_THRESHOLD:
        caveats.append("Identity was NOT verified; the user overrode the stop-and-confirm. "
                       "Treat the draft as low-confidence and double-check the target before sending.")
    elif conf < LOW_THRESHOLD:
        caveats.append(f"Identity confirmed at {conf:.0%} (unconfirmed band) — verify the person before sending.")

    if conflicts:
        for c in conflicts:
            winner = _newer(c["a"], c["b"])
            loser = c["b"] if winner is c["a"] else c["a"]
            for x in scored:
                if x["signal"]["id"] == loser["id"]:
                    x["breakdown"]["relevance"]["score"] = max(1, x["breakdown"]["relevance"]["score"] // 2)
                    x["breakdown"]["relevance"]["reason"] = "Penalized: contradicted by a newer, opposing story."
                    x["breakdown"]["contradicted"] = True
                    x["breakdown"]["total"] = round(_recompute(x["breakdown"]), 3)
            caveats.append(c["caveat"])
        s4["data"] = {"conflicts": [
            {"a": c["a"]["id"], "b": c["b"]["id"], "note": c["caveat"]} for c in conflicts]}
        s4["detail"] = "Found a narrative conflict; resolved toward the more recent story and penalized the stale one."
    elif stale:
        for x in stale:
            x["breakdown"]["attributability"]["score"] = 1
            x["breakdown"]["attributability"]["reason"] = "Penalized: no longer tied to this person's role."
            x["breakdown"]["stale"] = True
            x["breakdown"]["total"] = round(_recompute(x["breakdown"]), 3)
        note = stale[0]["signal"].get("role_note", "The signal is about a role that has changed.")
        caveats.append("Strongest personal signal is stale — the person no longer holds that role. "
                       "Downgraded to a company-level hook; verify the current role before sending.")
        s4["data"] = {"stale": [{"id": x["signal"]["id"], "note": note} for x in stale]}
        s4["detail"] = "A top signal no longer applies to this person — downgraded attributability."
    else:
        s4["data"] = {"conflicts": [], "stale": []}
        s4["detail"] = "No contradictions or staleness detected."
    _add(s4)
    result["caveats"] = caveats

    # Re-rank after any penalties.
    scored.sort(key=lambda x: x["breakdown"]["total"], reverse=True)

    # ------------------------------------------------------------------ stage 5
    s5 = {"id": "select_hook", "name": "Select the hook", "status": "ok"}
    best_sig = None
    total = None
    if scored:
        best, best_sig = scored[0], scored[0]["signal"]
        total = best["breakdown"]["total"]
        fresh = _age(best_sig) is not None and _age(best_sig) <= FRESH_DAYS
        strong = (total >= STRONG_HOOK and fresh
                  and not best["breakdown"].get("contradicted") and not best["breakdown"].get("stale"))
        mode = "trigger" if strong else "relationship-builder"
        s5["data"] = {
            "chosen": {"id": best_sig["id"], "source": best_sig["source"], "date": best_sig["date"],
                       "headline": best_sig["headline"], "total": total, "mode": mode},
            "runners_up": [
                {"id": x["signal"]["id"], "headline": x["signal"]["headline"],
                 "total": x["breakdown"]["total"], "why_not": _why_not(x, best_sig, mode)}
                for x in scored[1:3]
            ],
            "weights": WEIGHTS, "weights_sum": WEIGHTS_SUM,
        }
        s5["detail"] = f"Chose '{best_sig['headline'][:45]}' ({total:.2f}) in {mode} mode."
        result["top_hook"] = {
            "id": best_sig["id"], "source": best_sig["source"], "date": best_sig["date"],
            "headline": best_sig["headline"], "summary": best_sig.get("summary", ""),
            "url": best_sig.get("url", ""), "score_breakdown": best["breakdown"],
            "weighted_total": total,
            "origin": best_sig.get("retrieval", "seeded-corpus"),
        }
        result["runners_up"] = s5["data"]["runners_up"]
    else:
        strong, mode = False, "relationship-builder"
        s5["data"] = {"chosen": None, "runners_up": [], "weights": WEIGHTS, "weights_sum": WEIGHTS_SUM}
        s5["detail"] = "No usable hook found — falling back to a relationship-builder with no cited news."
        result["top_hook"] = None
        result["runners_up"] = []

    # Volume != relevance: signals exist but none crossed the trigger bar.
    if not strong and scored:
        caveats.append(
            f"Retrieved {len(scored)} public signals; the strongest "
            f"('{best_sig['headline'][:40]}' at {total:.2f}) is below the "
            f"{STRONG_HOOK} trigger threshold - not a clean pitch angle, so the "
            "relationship-builder cites it lightly (a 'saw you say…' opener) "
            "rather than forcing a trigger. Keep the tone soft.")
        s5["data"]["volume_not_relevance"] = True

    s5["status"] = "ok" if strong else "fallback"
    _add(s5)
    result["mode"] = mode
    result["confidence"] = _final_confidence(strong=strong, identity_conf=conf, caveats=caveats)

    # ------------------------------------------------------------------ stage 6
    # The LLM is the only drafting engine; if it's off or the call fails we
    # surface that honestly instead of shipping a canned template email.
    s6 = {"id": "draft", "name": "Draft the message", "status": "ok"}
    draft = _draft(best_sig, scored, mode, first, last, role, company, caveats,
                   llm=llm, profile=profile)
    if draft:
        engine = draft.get("engine", "llm")
        s6["data"] = {"mode": mode, "subject": draft["subject"], "body": draft["body"],
                      "engine": engine,
                      "grounded_on": draft.get("grounded_on", []),
                      "grounding": "LLM draft built around the chosen hook; grounded only in retrieved facts."}
        s6["detail"] = (f"Drafted a {mode} message via {engine}"
                        + (", built around the chosen hook." if best_sig else
                           " - no strong hook, so kept personal and soft."))
    else:
        s6["status"] = "skipped"
        s6["data"] = {"mode": mode, "subject": None, "body": None,
                      "engine": None,
                      "note": "Draft not generated - the LLM drafter is off "
                               "(set GOOGLE_API_KEY + PS3_LLM_DRAFT=1) or its "
                               "call failed. The hook and caveats above are "
                               "still valid; no canned email is shipped."}
        s6["detail"] = "Draft engine unavailable - skipping the draft, not shipping a template."
    _add(s6)
    result["draft"] = draft
    result["status"] = "complete"
    return _finish(stages, result, "complete")


def _retrievers_used(raw):
    seen = []
    for r in raw:
        src = r.get("retrieval", r.get("source"))
        if src not in seen:
            seen.append(src)
    return seen or ["none"]


def _detect_conflicts(signals):
    """Find opposing narratives (scale-up vs. cost-cutting). Deterministic on
    type/headline text offline; an LLM would do this more robustly live."""
    out = []

    def kind(sig):
        t = (sig.get("type", "") + " " + sig.get("headline", "") + " " + sig.get("summary", "")).lower()
        if "layoff" in t or "restructur" in t or "leaner" in t or "headcount trimmed" in t:
            return "down"
        if "series c" in t or "scale fast" in t or "aggressive" in t or "scaling" in t:
            return "up"
        return None

    ups = [s for s in signals if kind(s) == "up"]
    downs = [s for s in signals if kind(s) == "down"]
    for u in ups:
        for d in downs:
            winner = _newer(u, d)
            out.append({
                "a": u, "b": d,
                "caveat": (f"Conflicting narratives: '{u.get('type', u['id'])}' ({u['date']}) vs "
                           f"'{d.get('type', d['id'])}' ({d['date']}). The more recent story "
                           f"({winner.get('type', winner['id']).lower()}) is the current reality; "
                           "don't cite the older one as if it still holds."),
            })
    return out


def _newer(a, b):
    aa, bb = _age(a), _age(b)
    if aa is None:
        return b
    if bb is None:
        return a
    return b if bb < aa else a


def _why_not(x, best, mode):
    s = x["signal"]
    if s.get("role_context") == "stale":
        return "About a role the person has left — no longer personal."
    if x["breakdown"].get("contradicted"):
        return "Contradicted by a more recent story."
    if s.get("source") == "press" and not s.get("person_related"):
        return "Company-level only; weak as a personal hook."
    return "Lower weighted total than the chosen hook."


def _final_confidence(strong, identity_conf, caveats):
    """high/medium/low from hook strength, identity confidence, and caveats.
    Made explicit so the rule is easy to point at in the interview."""
    caveat_text = " ".join(c.lower() for c in caveats)
    has_issue = any(k in caveat_text for k in ("stale", "conflict", "no fresh", "unconfirmed",
                                               "not verified", "volume"))
    if strong and identity_conf >= LOW_THRESHOLD and not has_issue:
        return "high"
    if strong or identity_conf >= LOW_THRESHOLD:
        return "medium"
    return "low"


def _draft(sig, scored, mode, first, last, role, company, caveats, llm=None, profile=None):
    """LLM-only drafting. The deterministic template was removed - it read
    generic and undercut the personalization the retrieval exists for. If the
    LLM is unavailable or its call fails, returns None so the stage shows
    'draft not generated' instead of shipping a canned email. The LLM draft
    is built around the CHOSEN SIGNAL, with the person's profile used only as
    the personal anchor for the opener."""
    if llm is None or not callable(getattr(llm, "draft", None)):
        return None
    out = llm.draft(sig, scored, mode, first, last, role, company, caveats,
                    profile=profile)
    if out:
        out["engine"] = f"llm ({getattr(llm, '_model', 'model')})"
    return out


def _finish(stages, result, status):
    result["finished_at"] = _dt.datetime.utcnow().isoformat() + "Z"
    return stages, result, status
