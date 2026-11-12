"""Live entity resolution — verify a prospect against public reporting.

THE PIVOT: Stage 1 no longer means "is this in my corpus?" For any
non-seeded prospect we run a robust live verification (web_search.
verify_person: two queries, person-anchored, so a misspelled company
can't zero the search — Google News ANDs every term) and compute an
*evidence-based* confidence.

Confidence rule table (single place, deliberately simple):

    corroborating articles   full-name mentions   -> confidence  verdict
    source unreachable       —                    ->  0.20      unverified (stop; note names the provider error)
    0                        —                    ->  0.20      no-evidence (stop)
    ≥1, 0 full-name          —                    ->  0.35      weak (stop: company real, person unconfirmed)
    1                        —                    ->  0.55      proceed, unconfirmed-band caveat
    2+                       —                    ->  0.80      proceed
    3+                       —                    ->  0.90      proceed, high band

Seeded demo prospects keep their deterministic corpus resolution (the four
designed edge cases stay reproducible mid-interview). stdlib only; a
transport failure becomes a *labeled* "unverified" result, never a crash
and never a fake "confirmed."
"""

from . import config, web_search


def _confidence_rule(evidence_count, full_name_count):
    if evidence_count == 0:
        return 0.20
    if full_name_count == 0:
        return 0.35
    if full_name_count == 1:
        return 0.55
    if full_name_count >= 2:
        return 0.80
    if full_name_count >= 3:
        return 0.90
    return 0.65


def resolve(first, last, role, company, prospect=None):
    """Resolve a prospect's identity. Returns:
    {match, confidence, candidates, note, source, live, offline, queries}
    """
    name = " ".join(p for p in [first, last] if p).strip()

    # ---- seeded: deterministic corpus resolution (edge cases stay stable)
    if prospect is not None and prospect.get("resolution"):
        res = prospect["resolution"]
        return {
            "match": res["match"], "confidence": res["confidence"],
            "candidates": res["candidates"], "note": res["note"],
            "source": "seeded-corpus", "live": False, "offline": False,
            "queries": [],
        }

    # ---- live: verify against public reporting (person-anchored, chain-aware)
    provider = web_search.search_provider()
    res = web_search.verify_person(first, last, role, company,
                                   limit=config.ENTITY_PROBE_TOP, strict=False)
    items, queries_used = res["items"], res["queries"]
    err, warn = res["error"], res["warning"]
    used = res["used_providers"] or [provider]

    lname = name.lower()
    evidence = [it for it in items if it.get("title") or it.get("body")]
    full_name_count = sum(
        1 for it in evidence
        if lname in (it.get("title", "") + " " + it.get("body", "")).lower()
    )

    def cand_conf(it):
        text = (it.get("title", "") + " " + it.get("body", "")).lower()
        return 0.9 if lname in text else 0.6

    candidates = [
        {
            "company": it.get("company_name", company),
            "role": role,
            "confidence": cand_conf(it),
            "source": f"{it.get('retrieval', provider)} · {it.get('published', '')}",
            "title": it.get("title", ""),
            "url": it.get("url", ""),
        }
        for it in evidence[:3]
    ]

    if err:
        # Whole chain unreachable -> unverified, never fake "confirmed".
        from . import net
        conf = 0.20
        match = "unverified"
        note = (f"Could not reach any live source: {net.short_error(Exception(err))}. "
                "Identity treated as unverified — confirm manually before "
                "relying on the draft, or retry.")
    elif not evidence:
        conf = 0.20
        match = "no-evidence"
        note = (f"No corroborating public reporting found for "
                f"'{name}' at '{company}' via {provider}. "
                "Proceeding risks attaching a hook to the wrong person.")
    elif not full_name_count:
        conf = 0.35
        match = "weak"
        note = (f"Found {len(evidence)} public stories about {company}, but none names "
                f"'{name}'. The company is real; the person is unconfirmed.")
    else:
        conf = _confidence_rule(len(evidence), full_name_count)
        match = "live-verified"
        note = (f"{len(evidence)} public articles corroborate this engagement; "
                f"{full_name_count} name '{name}' by full name. "
                + ("Company and person both confirmed in reporting."
                   if conf >= 0.8 else "Proceeding with an unconfirmed-band caveat."))

    if warn:
        # A configured provider failed but the fallback delivered evidence:
        # the verdict stands, with a degradation note.
        note = f"{note} (degraded: {warn})"

    return {
        "match": match, "confidence": conf, "candidates": candidates,
        "note": note, "source": " + ".join(used), "live": True,
        "offline": bool(err), "queries": queries_used,
    }
