"""Signal scoring: LLM judges each axis, code applies fixed weights.

Design choice (the one to say out loud in the video): the judge (an LLM live,
or a deterministic heuristic stand-in offline) produces a 1-5 score + a
one-line justification on each axis. The *weights* are fixed constants and
the final score is a plain weighted sum — so the math is reproducible and
the "why did this hook win" answer is always showable.

Recency anchors per-signal: seeded corpus docs anchor to the demo reference
date (deterministic demo), live web-searched docs anchor to the real
wall-clock today (passed down as `recency_ref` from retrieval). This is the
fix that lets a real prospect's 2025 news be scored on its true age instead
of looking "ancient" against the demo date.
"""

from .normalize import age_days
from . import corpus

AXES = [
    ("relevance", 0.35, "Ties to a problem we solve / a trigger that creates budget"),
    ("recency", 0.20, "Newer = stronger; stale signals decay"),
    ("specificity", 0.20, "A named pain beats a generic announcement"),
    ("attributability", 0.15, "Ties to the person/team, not just the company"),
    ("actionability", 0.10, "Opens a conversation rather than a lecture"),
]
WEIGHTS = {k: w for k, w, _ in AXES}
WEIGHTS_SUM = sum(WEIGHTS.values())  # 1.0


def heuristic_axes(signal, prospect):
    """Offline stand-in for the LLM judgment, derived only from fields we
    actually have (corpus-set or retrieval-derived). Deterministic, so the
    demo behaves the same every run.

    In live mode, replace this call with an LLM prompt that returns the same
    shape: {axis: {"score": 1-5, "reason": "one line"}}.
    """
    src = signal.get("source", "press")
    rel = signal.get("relevance_base")
    if rel is None:
        rel = {"funding": 5, "job_postings": 5, "exec_interview": 4, "blog": 5,
               "product": 3, "social": 3, "press": 2, "news": 2}.get(src, 2)

    recency = _recency_score(signal)
    # Specificity = a specific *trigger*, not merely a named person. Gate the
    # high mark on the signal actually being a GTM-relevant trigger
    # (relevance_base), so a career retrospective / bio about the person
    # doesn't inflate specificity. This is the tone-deaf-pitch guard.
    is_trigger = (rel or 0) >= 3
    if signal.get("person_related") and is_trigger:
        specificity = 5
    elif is_trigger:
        specificity = 4
    elif src in ("funding", "job_postings", "exec_interview"):
        specificity = 3
    else:
        specificity = 2
    attrib = 5 if signal.get("person_related") else 2
    action = 4 if src in ("exec_interview", "blog", "social", "funding", "job_postings") else 3

    return {
        "relevance": {"score": rel, "reason": _why_relevance(src)},
        "recency": {"score": recency, "reason": f"{signal.get('date') or signal.get('published', '?')} — " + _why_recency(recency)},
        "specificity": {"score": specificity, "reason": (
            "A concrete, GTM-relevant trigger tied to the org" if (signal.get("relevance_base") or 0) >= 3
            else "Named but not a sales-research trigger — career/bio/press content" if signal.get("person_related")
            else "Company-level; no named pain")},
        "attributability": {"score": attrib, "reason": "Directly tied to their team/role" if attrib >= 4 else "Company-level; person is only indirectly involved"},
        "actionability": {"score": action, "reason": "Gives a natural, specific opening line" if action >= 4 else "Usable but generic"},
    }


def _recency_score(signal):
    # Per-signal recency anchor: live docs use the real wall-clock today;
    # seeded docs use the deterministic demo reference date. This is what
    # keeps real 2025 news for a real prospect from being scored "ancient".
    ref = signal.get("recency_ref") or corpus.REFERENCE_DATE
    age = age_days(signal.get("date") or signal.get("published") or "", ref)
    if age is None:
        return 3
    if age < 0:
        # signal dated after the reference: it's *newer than "today"* — score it
        # as the freshest thing, not as missing.
        return 5
    if age <= 14:
        return 5
    if age <= 45:
        return 4
    if age <= 90:
        return 3
    if age <= 180:
        return 2
    return 1


def _why_relevance(src):
    from . import corpus as _c
    return _c.RELEVANT_TRIGGERS.get(src, "Generic news; only weakly related to what we sell.")


def _why_recency(score):
    return {5: "very fresh", 4: "recent", 3: "a few months old", 2: "fading", 1: "stale"}[score]


def score_signal(signal, prospect, llm=None):
    """Score one signal. `llm` may be a callable judge OR an LLM object with
    .judge(); either way, weights are ALWAYS applied here, in code. Falls
    back to the deterministic heuristic when the LLM declines (None)."""
    if llm is None:
        axes = heuristic_axes(signal, prospect)
        used = "heuristic"
    elif callable(llm):
        axes = llm(signal, prospect) or heuristic_axes(signal, prospect)
        used = "llm-callable"
    else:  # LLM object
        ax = getattr(llm, "judge", None)
        axes = ax(signal, prospect) if callable(ax) else None
        if axes is None:
            axes = heuristic_axes(signal, prospect)
            used = "heuristic (llm judge off/failed)"
        else:
            used = f"llm ({getattr(llm, '_model', '?')})"
    breakdown, total = {}, 0.0
    for key, weight, _desc in AXES:
        a = axes.get(key, {"score": 3, "reason": ""})
        breakdown[key] = {"score": int(a.get("score", 3)), "weight": weight, "reason": a.get("reason", "")}
        total += breakdown[key]["score"] * weight
    breakdown["total"] = round(total, 3)
    breakdown["_judge"] = used
    return breakdown
