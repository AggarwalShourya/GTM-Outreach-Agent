"""PS-3 GTM Outreach Agent — Streamlit front-end.

Runs the same six-stage pipeline as the stdlib web server, but with
Streamlit's live re-render instead of SSE. `pipeline.run()` already accepts
an `on_stage` callback, which maps 1:1 onto "render this stage as it
finishes".

Run locally:
    streamlit run app.py

Deploy on Streamlit Cloud:
    Set GOOGLE_API_KEY and SERPER_API_KEY under the app's "Secrets". They are
    read here via st.secrets and are never committed to the repo.
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------- path + env
# app.py sits at the REPO ROOT; the package is the `ps3/` subdirectory next
# to it. Put the repo root on sys.path so `import ps3` works no matter what
# the clone folder is called (a local dir, or Streamlit Cloud's
# /mount/src/<repo>). This is the fix for "No module named 'ps3'".
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


def _apply_secrets():
    """Fill unset keys from the most available source. Real environment
    variables always win; then Streamlit Cloud secrets; then a local .env."""
    try:
        for k in ("GOOGLE_API_KEY", "SERPER_API_KEY",
                  "OPENAI_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY",
                  "BING_SEARCH_KEY", "PS3_LLM_MODEL"):
            v = st.secrets.get(k)
            if v is not None and not os.environ.get(k):
                os.environ[k] = str(v)
    except Exception:
        pass  # st.secrets is empty/unavailable outside Streamlit Cloud


_apply_secrets()


def _load_env_file():
    for p in (APP_DIR / ".env", APP_DIR / "ps3" / ".env"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:]
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


_load_env_file()

from ps3 import pipeline, config, embed_real, data      # noqa: E402
from ps3.llm_optional import LLM                         # noqa: E402

# ---------------------------------------------------------------- resources
@st.cache_resource
def get_llm():
    return LLM()


def _probe_line():
    chips = [f"embed: {embed_real.provider()}", f"search: {config.search_provider()}"]
    if not config.LIVE_SEARCH:
        chips.append("live off (PS3_LIVE=0)")
    return " · ".join(chips)


# ---------------------------------------------------------------- stage render
def _render_stage(idx, stage):
    """One stage card. Called from the on_stage callback as stages finish,
    and again (idempotently) when rebuilding after a re-run."""
    s = stage
    status = s.get("status", "ok")
    head = (f"{'✅' if status == 'ok' else '⏸️' if status in ('stopped', 'unavailable', 'skipped') else '⚠️'} "
            f"Stage {idx} · {s.get('name', s.get('id'))} "
            f"<span style='opacity:.6;font-size:12px'>({status})</span>")

    d = s.get("data") or {}
    extra = []

    if s["id"] == "entity_resolve" and "confidence" in d:
        extra.append(f"confidence **{d.get('confidence')}** · match `{d.get('match')}` · "
                     f"source `{d.get('source')}`")
        if d.get("queries"):
            extra.append("verification queries: " + " · ".join(f"“{q}”" for q in d["queries"]))
        if d.get("candidates"):
            cands = "\n".join(
                f"- {c.get('title') or (c.get('company','') + ' — ' + str(c.get('role','')))} "
                f"({c.get('confidence')}) {('[' + c.get('source','') + ']') if c.get('source') else ''}"
                + (f"  <a href='{c['url']}' target='_blank'>link</a>" if c.get("url") else "")
                for c in d["candidates"][:3])
            extra.append(f"<details><summary>evidence</summary>{cands}</details>")
        if d.get("note"):
            extra.append(f"<em>{d['note']}</em>")

    elif s["id"] == "gather_signals" and "count" in d:
        extra.append(f"{d.get('count')} artifacts · retrieval: {d.get('retrieval')} · "
                     f"dense channel: <code>{d.get('embed_provider')}</code>"
                     + (" · **live web search**" if d.get("live_search") else ""))
        if d.get("items"):
            rows = "\n".join(
                f"- [💡 evidence]{i.get('evidence','')[:160]}"
                f"  <a href='{i.get('url','')}' target='_blank'>{i.get('headline','')[:70]}</a> "
                f"<span style='opacity:.6'>{i.get('date','')} · {i.get('origin',i.get('source',''))}</span>"
                for i in d["items"][:8])
            extra.append(f"<details><summary>{d.get('count')} retrieved artifacts</summary>{rows}</details>")

    elif s["id"] == "score" and "ranked" in d:
        if d.get("ranked"):
            lines = []
            for r in d["ranked"]:
                axes = r.get("breakdown", {})
                ax = " · ".join(f"{k} {axes[k]['score']}/5" for k in
                                ("relevance", "recency", "specificity", "attributability", "actionability")
                                if k in axes)
                mark = " ⚡" if r["total"] >= 3.5 else ""
                lines.append(f"- **{r['total']}**{mark} — {r['headline'][:60]}  <span style='opacity:.7'>{ax}</span>")
            extra.append("\n".join(lines))

    elif s["id"] == "contradictions":
        if d.get("conflicts"):
            extra.append("<br>".join(f"⚠️ {c.get('note','')}" for c in d["conflicts"]))
        if d.get("stale"):
            extra.append("<br>".join(f"🕓 stale: {x.get('note','')}" for x in d["stale"]))

    elif s["id"] == "select_hook":
        ch = d.get("chosen")
        if ch:
            extra.append(f"chosen: **{ch['headline'][:80]}** ({ch['total']}) → {ch['mode']}")
        if d.get("runners_up"):
            extra.append("runners-up: " + "; ".join(
                f"{r['headline'][:40]} ({r['total']}) — {r.get('why_not','')}" for r in d["runners_up"]))

    elif s["id"] == "draft":
        if d.get("body"):
            extra.append(f"<b>{d.get('subject')}</b> <span style='opacity:.6'>({d.get('engine')})</span>\n\n"
                         f"<pre style='white-space:pre-wrap'>{d['body']}</pre>")
        elif d.get("note"):
            extra.append(f"⚠️ {d['note']}")

    with st.container(border=True):
        st.markdown(head, unsafe_allow_html=True)
        if s.get("detail"):
            st.caption(s["detail"])
        for e in extra:
            st.markdown(e, unsafe_allow_html=True)


# ---------------------------------------------------------------- run
PACING = 0.35  # seconds of visible pause between stages (demo pacing; set 0 to go fast)


def run_pipeline(first, last, role, company, confirmed):
    llm = get_llm()

    def on_stage(stage):
        _render_stage(len(st.session_state.setdefault("_stages_seen", []) ) + 1, stage)
        st.session_state["_stages_seen"].append(stage["id"])
        if PACING:
            time.sleep(PACING)

    st.session_state["_stages_seen"] = []
    t0 = time.time()
    stages, result, status = pipeline.run(
        first, last, role, company, llm=llm, on_stage=on_stage, confirmed=confirmed)

    # persist for the web dashboard too (harmless; same SQLite the server uses)
    try:
        data.new_run(result, stages)
    except Exception:
        pass

    _render_outcome(result, status, time.time() - t0, confirmed)
    return result


def _render_outcome(result, status, secs, confirmed):
    st.divider()
    if status == "needs_confirmation":
        st.warning(
            f"⏸️ **Identity unconfirmed — no draft produced.**\n\n"
            f"{result.get('confirmation_message','')}",
        )
        st.caption("The process stopped deliberately rather than guess. A human can override.")
        c1, _ = st.columns([1, 3])
        if c1.button("Proceed with low confidence (live retrieval)", type="primary"):
            st.session_state["_force"] = True
            st.rerun()
    else:
        mode = result.get("mode")
        conf = result.get("confidence")
        st.write(
            f"### Outcome · {mode} · {conf} confidence · {secs:.1f}s end-to-end"
        )
        th = result.get("top_hook")
        if th:
            st.markdown(f"**Hook:** {th['headline'][:120]}"
                        + (f" · total **{th['weighted_total']}**" if th.get("weighted_total") is not None else "")
                        + (f" · <span style='color:#6d28d9'>live · {th['origin']}</span>"
                           if th.get("origin", "seeded-corpus") != "seeded-corpus" else ""))
        for c in result.get("caveats", []):
            st.markdown(f"⚠️ {c}")
        d = result.get("draft")
        if d and d.get("body"):
            st.markdown(f"**Draft** — <span style='color:#6d28d9'>{d.get('engine')}</span>\n")
            st.markdown(f"**{d.get('subject','')}**\n")
            st.code(d["body"])
            if d.get("grounded_on"):
                st.caption("Grounded in: " + " · ".join(d["grounded_on"]))
            st.caption("Human review required before sending. Draft is stored in the dashboard history.")
        else:
            st.info("Draft not generated (LLM off or call failed). The hook and caveats above are still valid.")
    st.session_state.pop("_force", None)


# ---------------------------------------------------------------- layout
st.set_page_config(page_title="PS-3 · GTM Outreach Agent", page_icon="📩", layout="wide")
st.title("PS-3 · GTM Outreach Agent")
st.caption(_probe_line() + "  ·  keys via Streamlit Secrets (or local `.env`)")

with st.sidebar:
    st.subheader("Prospect")
    st.text_input("First name", key="ps3_first", placeholder="Shourya")
    st.text_input("Last name",  key="ps3_last",  placeholder="Aggarwal")
    st.text_input("Role",       key="ps3_role",  placeholder="CSE Undergraduate")
    st.text_input("Company",    key="ps3_company", placeholder="IIT Roorkee")
    st.divider()
    st.caption("Override the stop-and-confirm gate even at low identity confidence "
               "(adds a 'NOT verified' caveat to the draft).")

first   = (st.session_state.get("ps3_first") or "")
last    = (st.session_state.get("ps3_last") or "")
role    = (st.session_state.get("ps3_role") or "")
company = (st.session_state.get("ps3_company") or "")

run_btn = st.button("▶  Run process", type="primary", use_container_width=True)

if (run_btn or st.session_state.get("_force")) and (first.strip() and last.strip() and company.strip()):
    confirmed = bool(st.session_state.get("_force"))
    run_pipeline(first.strip(), last.strip(), role.strip(), company.strip(), confirmed)
elif st.session_state.get("_force"):
    st.error("Fill the prospect fields to proceed.")
elif not (first.strip() and last.strip() and company.strip()):
    st.info("Enter a prospect, then press **Run process**.")
    st.markdown(
        "The six stages execute live: ① entity verification → ② hybrid RAG retrieval "
        "→ ③ axis scoring → ④ contradiction/staleness → ⑤ hook selection → ⑥ LLM draft. "
        "Every stage shows its own evidence."
    )
