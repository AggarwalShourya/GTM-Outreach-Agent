"""Live web-search retrievers — the 'pull FRESH documents' half.

Providers (first configured key wins; key-free RSS is the fallback):
  tavily   TAVILY_API_KEY        LLM-tuned web search
  serper   SERPER_API_KEY         SERP API
  bing     BING_SEARCH_KEY        classic web + news
  rss      (no key)              Google News RSS — key-free, offline-capable
  none     (forced)              corpus only

Two public entry points:
  * verify_person(...)   — robust IDENTITY verification for a prospect. Anchors
    on the DISTINCTIVE person name (the token that actually identifies them)
    and uses the company only as a corroborating filter, so a misspelled
    company can't zero the whole query (Google News ANDs every term — the v1
    bug where 'Micorosoft' killed a search that 'satya nadella' alone would
    have answered).
  * search_raw(...) / search(...) — signal retrieval for a company's GTM
    triggers; strict mode surfaces transport failures instead of [] so
    "source unreachable" != "source says nothing".

Every provider is stdlib urllib; non-strict callers degrade to [] on failure
so the pipeline falls through to the corpus / quiet path instead of crashing.
"""

import json
import re
import urllib.request
import urllib.parse
import urllib.error
import datetime as _dt

from . import config
from . import net
from .live_retrieval import rss_docs, _today

_UA = {"User-Agent": "Mozilla/5.0 (PS-3 outreach research)"}
_TIMEOUT = int(__import__("os").environ.get("PS3_SEARCH_TIMEOUT", "25"))


def _norm_item(doc_id, company_name, source, published, type, title, body, url, retrieval,
               date_known=True):
    return {
        "id": doc_id, "company_key": "live", "company_name": company_name,
        "source": source, "published": published, "type": type,
        "title": re.sub(r"\s+", " ", title or "").strip(),
        "body": re.sub(r"\s+", " ", body or "").strip(),
        "url": url, "retrieval": retrieval, "date_known": date_known,
    }


_JUNK_RE = re.compile(
    r"\.(cdr|dwg|pdf|docx?|xlsx?|pptx?|csv|zip|txt|png|jpe?g)$", re.I)


def _is_junk(item):
    """Drop file-like SERP noise (e.g. 'April-June 2023.cdr', stray .pdf/.zip
    hits) before they can reach the signal list."""
    t = (item.get("title") or "").lower()
    u = (item.get("url") or "").lower()
    if _JUNK_RE.search(u.split("?")[0].split("/").pop()) or _JUNK_RE.search(t):
        return True
    if len(t) < 25 and " " not in t:
        return True
    return False


def _api_date(raw):
    """Pull an ISO-ish date out of API fields (serper 'date', bing
    'datePublished', tavily 'published_date'); None when absent/unparseable."""
    raw = (raw or "").strip()
    if not raw:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
    if m:
        return m.group(1)
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(raw[:25], fmt).date().isoformat()
        except ValueError:
            pass
    return None


def _clamp_date(d):
    today = _today()
    try:
        dd = _dt.date.fromisoformat(d[:10])
        return today.isoformat() if dd > today else d[:10]
    except Exception:
        return today.isoformat()


# ---------------------------------------------------------------- providers
def _tavily_raw(query, limit):
    import os
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("tavily: TAVILY_API_KEY not set")
    payload = {"api_key": key, "query": query, "search_depth": "advanced",
               "include_raw_content": False, "max_results": limit}
    from . import net
    with net.https_open("https://api.tavily.com/search",
                        headers={**_UA, "Content-Type": "application/json"},
                        data=json.dumps(payload).encode(), method="POST",
                        timeout=_TIMEOUT) as r:
        data = json.loads(r.read().decode())
    out = []
    for i, it in enumerate(data.get("results", [])[:limit]):
        url = it.get("url", "")
        pub = _api_date(it.get("published_date"))
        out.append(_norm_item(f"tvl-{abs(hash(url or it.get('title'))) % 10**8}-{i}",
                              "", "press", pub or _today().isoformat(),
                              "Live web", it.get("title", ""), (it.get("content") or "")[:1000],
                              url, "live-tavily", date_known=bool(pub)))
    return out


def _serper_raw(query, limit):
    """Google Serper (google.serper.dev). SERPER_API_KEY belongs to THIS
    service — do NOT reuse a SerpApi key here; those use _serpapi_raw."""
    import os
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("serper: SERPER_API_KEY not set")
    payload = {"q": query, "num": limit}
    from . import net
    with net.https_open("https://google.serper.dev/search",
                        headers={**_UA, "X-API-KEY": key, "Content-Type": "application/json"},
                        data=json.dumps(payload).encode(), method="POST",
                        timeout=_TIMEOUT) as r:
        data = json.loads(r.read().decode())
    out = []
    for i, it in enumerate(data.get("organic", [])[:limit]):
        url = it.get("link", "")
        pub = _api_date(it.get("date"))
        out.append(_norm_item(f"srp-{abs(hash(url or it.get('title'))) % 10**8}-{i}",
                              "", "press", pub or _today().isoformat(),
                              "Live web", it.get("title", ""), (it.get("snippet") or "")[:900],
                              url, "live-serper", date_known=bool(pub)))
    return out


def _serpapi_raw(query, limit):
    """SerpApi (serpapi.com). Distinct service from Google Serper;
    SERPAPI_KEY belongs to THIS one. Same normalized output."""
    import os
    key = os.environ.get("SERPAPI_KEY")
    if not key:
        raise RuntimeError("serpapi: SERPAPI_KEY not set")
    from . import net
    q = urllib.parse.quote(f"{query}&num={limit}")
    url = f"https://serpapi.com/search.json?engine=google&q={q}&api_key={key}"
    with net.https_open(url, headers=_UA, timeout=_TIMEOUT) as r:
        data = json.loads(r.read().decode())
    out = []
    for i, it in enumerate(data.get("organic_results", [])[:limit]):
        u = it.get("link", "")
        out.append(_norm_item(f"sp-{abs(hash(u or it.get('title'))) % 10**8}-{i}",
                              "", "press", _today().isoformat(), "Live web",
                              it.get("title", ""), (it.get("snippet") or "")[:900],
                              u, "live-serpapi"))
    return out


def _bing_raw(query, limit):
    import os
    key = os.environ.get("BING_SEARCH_KEY")
    if not key:
        raise RuntimeError("bing: BING_SEARCH_KEY not set")
    from . import net
    q = urllib.parse.quote(query)
    url = f"https://api.bing.microsoft.com/v7.0/search?q={q}&count={limit}&freshness=Month"
    with net.https_open(url, headers={**_UA, "Ocp-Apim-Subscription-Key": key},
                        timeout=_TIMEOUT) as r:
        data = json.loads(r.read().decode())
    out = []
    for i, it in enumerate(data.get("webPages", {}).get("value", [])[:limit]):
        u = it.get("link", "")
        pub = _api_date(it.get("datePublished") or it.get("dateLastCrawled"))
        out.append(_norm_item(f"bing-{abs(hash(u or it.get('name'))) % 10**8}-{i}",
                              "", "press", pub or _today().isoformat(),
                              "Live web", it.get("name", ""), (it.get("snippet") or "")[:900],
                              u, "live-bing", date_known=bool(pub)))
    return out


def _rss_raw(query, limit, strict=False):
    return rss_docs(query, "", limit, strict=strict)


# ---------------------------------------------------------------- public api
def search_provider():
    return config.search_provider()


def _ddg_raw(query, limit):
    """Key-free DuckDuckGo fallback (not Google-restricted like RSS).

    Tries the lite endpoint first (lighter, less anomaly-gated), then the
    HTML endpoint. Returns best-effort results; [] when DDG blocks this IP
    (common on datacenter IPs, rare on residential)."""
    import urllib.parse as _up
    import re as _re

    def _unwrap(href):
        m = _re.search(r"(uddg|ku)=([^&]+)", href)
        return _up.unquote(m.group(2)) if m else href

    def _clean(h):
        return _re.sub(r"<[^>]+>", "", h).strip()

    for host, marker in (("lite.duckduckgo.com/lite/?q=", "lite"),
                         ("html.duckduckgo.com/html/?q=", "html")):
        url = host + _up.quote(query)
        try:
            with net.https_open(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                                timeout=_TIMEOUT) as r:
                html = r.read().decode("utf-8", "ignore")
        except Exception:
            continue
        if "anomaly" in html.lower() or "unusual traffic" in html.lower():
            continue
        out = []
        if marker == "lite":
            # lite: <a rel="nofollow" href="...">Title</a> followed by snippet <td>
            pairs = _re.findall(
                r'<a[^>]+href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>', html, _re.S)
            if not pairs:
                pairs = _re.findall(
                    r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, _re.S)
            snips = _re.findall(r'<td[^>]*class="[^"]*"[^>]*>(.*?)</td>', html, _re.S)
            for i, (href, title) in enumerate(pairs[:limit]):
                snip = _clean(snips[i]) if i < len(snips) else ""
                out.append(_norm_item(f"ddg-{abs(hash(href)) % 10**8}-{i}",
                                      "", "press", _today().isoformat(), "Live web",
                                      _clean(title), snip[:600], _unwrap(href), "live-ddg",
                                      date_known=False))
        else:
            blocks = _re.findall(
                r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, _re.S)
            snips = _re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html, _re.S)
            for i, (href, title) in enumerate(blocks[:limit]):
                snip = _clean(snips[i]) if i < len(snips) else ""
                out.append(_norm_item(f"ddg-{abs(hash(href)) % 10**8}-{i}",
                                      "", "press", _today().isoformat(), "Live web",
                                      _clean(title), snip[:600], _unwrap(href), "live-ddg",
                                      date_known=False))
        if out:
            return out
    return []


_PROVIDER_FN = {"tavily": _tavily_raw, "serper": _serper_raw, "serpapi": _serpapi_raw,
                "bing": _bing_raw, "rss": _rss_raw, "ddg": _ddg_raw}


def _try_provider(name, query, limit):
    """Run one provider; return items or raise RuntimeError(name: err).
    HTTP errors carry the response BODY so a 403 says *why* (bad key vs
    out-of-credits vs region), not just that it happened."""
    fn = _PROVIDER_FN.get(name)
    if fn is None:
        raise RuntimeError(f"{name}: unknown provider")
    try:
        if name == "rss":
            items = _rss_raw(query, limit, strict=True) or []
        else:
            items = fn(query, limit) or []
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "ignore")[:400]
        except Exception:
            body = ""
        hint = f"HTTP {e.code}"
        if e.code == 403:
            hint += (" (out of credits/quota)" if ("credit" in body.lower() or "quota" in body.lower() or "limit" in body.lower())
                    else " (invalid key, region, or permission)" if "key" in body.lower() or "invalid" in body.lower() or "permission" in body.lower()
                    else "")
        raise RuntimeError(f"{name}: {hint}{(' — ' + body) if body else ''}")
    except Exception as e:
        raise RuntimeError(f"{name}: {e}")
    return items


def search_raw(query, company_name="", limit=None, source_type="Live web", strict=False):
    """Run the configured live provider against a raw query. Returns normalized
    items. Non-strict: [] on any failure. Strict: raises RuntimeError with the
    provider + error so callers can distinguish 'unreachable' from 'nothing found'."""
    limit = limit or config.SEARCH_TOP_N
    name = config.search_provider()
    if name == "none":
        if strict:
            raise RuntimeError("none: live search disabled (PS3_SEARCH_PROVIDER=none)")
        return []
    if strict:
        items = _try_provider(name, query, limit)  # raises RuntimeError(name: err)
    else:
        try:
            items = _try_provider(name, query, limit)
        except Exception:
            items = []
    for it in items:
        it["company_name"] = company_name or it.get("company_name", "")
        if source_type and it.get("type") in ("Live news",):
            it["type"] = source_type
    seen, out = set(), []
    for it in items:
        if _is_junk(it):
            continue
        k = it.get("url") or it.get("title")
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def search_chain(query, company_name="", limit=None, source_type="Live web"):
    """Resilient multi-provider search: try each configured provider in order
    (a keyed one, then the key-free RSS fallback) and return the FIRST source
    that works. A keyed provider that errors (bad key, quota, geo-block,
    timeout) no longer poisons the run — RSS takes over. Returns
    (items, used_provider, errors)."""
    from . import config as _cfg
    chain = []
    keyed = _cfg.search_provider()
    if keyed != "none":
        chain.append(keyed)
    if _cfg.LIVE_SEARCH and "rss" not in chain:
        chain += ["rss", "ddg"]  # key-free fallbacks, in order
    if not chain:
        return [], "none", []
    limit = limit or config.SEARCH_TOP_N
    errors = []
    for name in chain:
        try:
            items = _try_provider(name, query, limit)
        except Exception as e:
            errors.append(str(e))
            continue
        for it in items:
            it["company_name"] = company_name or it.get("company_name", "")
            if source_type and it.get("type") in ("Live news",):
                it["type"] = source_type
        seen, out = set(), []
        for it in items:
            if _is_junk(it):
                continue
            k = it.get("url") or it.get("title")
            if k and k not in seen:
                seen.add(k)
                out.append(it)
        return out, name, errors if errors else None
    return [], chain[-1], errors


# ------------------------------------------------- robust identity verification
def verify_person(first, last, role, company, limit=None, strict=True):
    """Verify a person against public reporting, robustly.

    Strategy (fixes the AND-fragility): Google News ANDs every query term,
    so a single misspelled token kills the whole query. Instead we anchor on
    the DISTINCTIVE person name and treat the company as a corroborating
    filter, running two queries and merging:

      Q1  "<first> <last>"            — who is this person (distinctive name)
      Q2  "<first> <last> <company>"  — same person, at this company

    Evidence = union. A misspelled company only weakens Q2; Q1 alone still
    corroborates the person. A typo in the *person's* name is the only
    thing that should truly fail verification.

    Returns (items, queries_used, error_or_None).
    """
    name = " ".join(p for p in [first, last] if p).strip()
    q1 = f'"{name}"' if name else ""
    q2 = f'{name} {company}' if (name and company) else q1
    queries = [q for q in (q1, q2) if q]

    # Chain-aware error semantics: a provider failing is only FATAL when the
    # whole chain yields nothing. If a fallback (e.g. key-free RSS) delivered
    # evidence, the earlier failure is recorded as a `degraded` warning, not
    # an "unreachable" verdict.
    merged, seen, failures, used_providers = [], set(), [], set()
    for q in queries:
        if strict:
            items = search_raw(q, company_name=company, limit=limit, strict=True)
            used_providers.add(web_search_provider_for_strict())
        else:
            items, used, errs = search_chain(q, company_name=company, limit=limit)
            used_providers.add(used)
            if errs:
                failures.extend(errs if isinstance(errs, list) else [errs])
        for it in items:
            key = it.get("url") or it.get("title")
            if key and key not in seen:
                seen.add(key)
                it["verify_query"] = q
                merged.append(it)

    fatal = (not merged) and bool(failures)
    return {
        "items": merged,
        "queries": queries,
        "used_providers": sorted(used_providers),
        # fatal -> the WHOLE chain is unreachable: "unverified" verdict.
        "error": ("; ".join(failures) if fatal else None),
        # non-fatal -> fallback worked but a configured provider failed:
        "warning": ("; ".join(failures) if (failures and merged) else None),
    }


def web_search_provider_for_strict():
    return config.search_provider()


def profile_person(first, last, role, company, limit=None):
    """Build a compact 'who is this person' profile from live sources.

    Runs a few person-anchored queries (exact name, name+linkedin,
    name+company, name+role) through the resilient provider chain and
    distills:
      * linkedin   - the person's public profile SERP hit (url + headline +
                     snippet). Full LinkedIn profiles sit behind a login
                     wall, so we use exactly what search has indexed -
                     real, cited, never fabricated.
      * facts      - distinct public mentions of the person (news, posts,
                     pages), each with url for citation.
    This is what the LLM drafts around: a multi-faceted picture of the
    person, not a single news headline.
    """
    name = " ".join(p for p in [first, last] if p).strip()
    queries = [f'"{name}"', f'"{name}" linkedin']
    if company:
        queries.append(f'"{name}" {company}')
    if role and role.lower() not in " ".join(queries).lower():
        queries.append(f'"{name}" {role}')
    queries = list(dict.fromkeys(q for q in queries if q))

    items_all, providers_used, errors = [], set(), []
    for q in queries:
        items, used, errs = search_chain(q, company_name=company, limit=limit or 8)
        if errs:
            errors.extend(errs if isinstance(errs, list) else [errs])
        providers_used.add(used)
        for it in items:
            it["_q"] = q
            items_all.append(it)

    lname = name.lower()
    co_words = [w for w in re.sub(r"[^a-z0-9 ]", "", (company or "").lower()).split() if len(w) > 3]
    role_words = [w for w in re.sub(r"[^a-z0-9 ]", "", (role or "").lower()).split() if len(w) > 3]

    def matches_target(it):
        """Name-collision guard: a personal fact only counts if it's about
        the same person - the name must appear AND the company (or role) must
        corroborate. Without this, a common name pulls a DIFFERENT person's
        LinkedIn headline into the draft (e.g. 'Shourya Aggarwal, finance at
        Indiana University' instead of the IIT Roorkee CSE student)."""
        blob = (it.get("title", "") + " " + it.get("body", "")).lower()
        if lname not in blob:
            return False
        if co_words and any(w in blob for w in co_words):
            return True
        if role_words and any(w in blob for w in role_words):
            return True
        return False

    linkedin = None
    for it in items_all:
        u = (it.get("url") or "").lower()
        t = (it.get("title") or "").lower()
        if ("linkedin.com/in/" in u or "linkedin" in t) and matches_target(it):
            linkedin = {"url": it.get("url", ""), "title": it.get("title", ""),
                        "headline": (it.get("body") or "")[:300],
                        "provider": it.get("retrieval", "")}
            break
    # No company-matched LinkedIn hit? Better a null profile than the WRONG
    # person's headline - the draft will fall back to the signal + a warm,
    # non-specific opener instead of fabricating a personal fact.
    if linkedin and co_words and not any(w in (linkedin["title"] + linkedin["headline"]).lower() for w in co_words) \
       and not any(w in (linkedin["title"] + linkedin["headline"]).lower() for w in role_words):
        linkedin = None

    facts, seen = [], set()
    for it in items_all:
        text = (it.get("title") or "").strip()
        body = (it.get("body") or "").strip()
        if not text and not body:
            continue
        if lname and lname not in (text + " " + body).lower():
            continue  # company-level hit, not about the person
        if not matches_target(it):
            continue  # different person with the same name
        if linkedin and it.get("url") == linkedin["url"]:
            continue
        key = text[:60]
        if key in seen:
            continue
        seen.add(key)
        facts.append({"text": text[:160], "detail": body[:240],
                      "url": it.get("url", ""), "provider": it.get("retrieval", "")})
        if len(facts) >= 6:
            break

    return {
        "name": name,
        "linkedin": linkedin,
        "facts": facts,
        "queries": queries,
        "providers": sorted(providers_used),
        "errors": errors,
        "available": bool(facts or linkedin),
    }


def gtm_query(company, first=None, last=None, role=None):
    """Signal-hunting query for a company's GTM triggers. Includes the person
    (when given) so the retriever can also surface person-tied stories. This
    is *deliberately* an AND across the person+company — for signal hunting
    the company is the anchor; for identity verification it is not."""
    parts = []
    if first and last:
        parts.append(f"{first} {last}")
    if role:
        parts.append(role)
    parts.append(company)
    parts.append("sales OR SDR OR 'go-to-market' OR hiring OR prospect OR 'outbound' OR research")
    return " ".join(parts)


def search(company, first=None, last=None, role=None, limit=None):
    """Fetch fresh docs for a company (and optionally its named rep) via the
    resilient provider chain (keyed provider -> RSS fallback). Doc-shaped;
    [] only when the WHOLE chain yields nothing, so the pipeline degrades to
    the quiet path."""
    query = gtm_query(company, first, last, role)
    items, used, errs = search_chain(query, company_name=company, limit=limit)
    return items


if __name__ == "__main__":
    # CLI: python ps3/web_search.py "Satya Nadella Microsoft"
    # Tests the whole provider chain on THIS machine and prints what works.
    import sys, os
    if os.path.exists(os.path.join(os.path.dirname(__file__), ".env")):
        for line in open(os.path.join(os.path.dirname(__file__), ".env")):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if line.startswith("export "):
                k, _, v = line[7:].partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    q = " ".join(sys.argv[1:]) or '"Satya Nadella" Microsoft'
    print(f"Provider chain test for: {q!r}")
    items, used, errs = search_chain(q, limit=5)
    print(f"  provider used: {used}")
    print(f"  items: {len(items)}")
    for it in items[:3]:
        print(f"   - {it['title'][:70]}  ({it.get('published','?')})")
    if errs:
        print(f"  errors from earlier providers in chain:")
        for e in errs:
            print(f"   {e[:140]}")
    if not items:
        print("\n  ALL PROVIDERS FAILED ON THIS MACHINE. Live retrieval is down;")
        print("  the process will stop-and-confirm (or override -> quiet path).")
