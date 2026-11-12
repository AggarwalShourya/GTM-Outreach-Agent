"""Central HTTPS + CA-bundle handling.

The Windows Python problem this exists for: Windows Python does NOT use the
OS certificate store by default, and its default ssl CA bundle is frequently
missing/stale — so every HTTPS call dies with
`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` before
sending a single byte. The standard fix is `certifi`; this module makes the
process *pick the right CA source automatically* instead of assuming one.

CA source precedence:
  1. PS3_CA_BUNDLE env var -> explicit bundle path (corporate CAs, custom)
  2. certifi               -> pip install certifi (the Windows fix)
  3. truststore            -> system store (pip install truststore, py3.9+)
  4. python default        -> whatever the interpreter ships with

PS3_INSECURE_TLS=1 -> skip verification entirely. Last resort (corporate MITM
proxy without a shareable root, or a demo machine you control). A loud
warning is printed at startup when this is active.

Every live call in the package (RSS, web-search APIs, embedding APIs) goes
through `https_open` so the CA decision is made in exactly one place.
"""

import os
import ssl
import urllib.request


def build_ssl_context():
    """Return (context, source_label). Deterministic; cached after first build."""
    global _CTX_CACHE
    if _CTX_CACHE is not None:
        return _CTX_CACHE

    insecure = os.environ.get("PS3_INSECURE_TLS", "").strip().lower() in ("1", "true", "yes", "on")
    if insecure:
        ctx = ssl._create_unverified_context()
        ctx._ps3_insecure = True
        _CTX_CACHE = (ctx, "INSECURE (PS3_INSECURE_TLS=1 — verification disabled)")
        return _CTX_CACHE

    explicit = os.environ.get("PS3_CA_BUNDLE", "").strip()
    if explicit:
        ctx = ssl.create_default_context(cafile=explicit)
        _CTX_CACHE = (ctx, f"custom bundle: {explicit}")
        return _CTX_CACHE

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        _CTX_CACHE = (ctx, "certifi")
        return _CTX_CACHE
    except ImportError:
        pass

    try:
        import truststore
        ctx = ssl.create_default_context()
        truststore.inject_into(ctx)
        _CTX_CACHE = (ctx, "system store (truststore)")
        return _CTX_CACHE
    except ImportError:
        pass

    ctx = ssl.create_default_context()
    _CTX_CACHE = (ctx, "python default (no CA bundle found — pip install certifi)")
    return _CTX_CACHE


_CTX_CACHE = None


def https_open(url, headers=None, data=None, method="GET", timeout=25, user_agent=None):
    """urlopen with the process-wide CA context. This is the ONLY place in the
    package that opens HTTPS — every provider uses it."""
    global _CTX_CA_WARN
    ctx, source = build_ssl_context()
    if getattr(ctx, "_ps3_insecure", False) or "INSECURE" in source:
        _CTX_CA_WARN = True
    if user_agent and not (headers or {}).get("User-Agent"):
        headers = dict(headers or {})
        headers["User-Agent"] = user_agent
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


_CTX_CA_WARN = False


def ca_source():
    return build_ssl_context()[1]


def diagnose():
    """Run from CLI: `python ps3/net.py` (or `python -m ps3.net` style).
    Prints a full diagnostic: CA source, live-HTTPS probe, which providers
    are configured, and the concrete fix for whatever is broken."""
    print("PS-3 live-retrieval diagnostic")
    print("=" * 60)
    print(f"CA source now in use: {ca_source()}")
    if "INSECURE" in ca_source():
        print("!! PS3_INSECURE_TLS=1 — TLS verification is DISABLED.")
    try:
        from . import config as _c
        have_cfg = True
    except Exception:
        _c = None; have_cfg = False
    if have_cfg:
        print(f"search provider (auto): {_c.search_provider()}")
    tls = tls_doctor(timeout=10)
    print(f"live HTTPS probe: {'OK' if tls['ok'] else 'BLOCKED'}")
    print(f"  detail: {tls['detail']}")
    if tls.get("fix"):
        print(f"  fix:    {tls['fix']}")
    import urllib.request, ssl
    print("-" * 60)
    print("If 'BLOCKED' on a certificate error, run EXACTLY this with the")
    print("same python that runs the server:")
    print("    python -m pip install certifi")
    print("    python -c \"import urllib.request, ssl, certifi; ctx=ssl.create_default_context(cafile=certifi.where()); print('TLS OK' if urllib.request.urlopen('https://news.google.com/rss/articles?hl=en-US&gl=US&ceid=US:en', timeout=10, context=ctx).status == 200 else 'TLS ok but probe failed')\"")
    print("(Windows gotcha: `pip` and `python` may be different installs —")
    print(" always use `python -m pip`, never bare `pip`.)")
    if "TLS" in tls["detail"] or "certificate" in tls["detail"].lower():
        print("\nDIAGNOSIS: your Windows Python cannot verify TLS certs.")
        print(" 1. Is `python` the same install that has certifi?  python -c \"import certifi\"")
        print(" 2. Corporate proxy? If you have an HTTPS_PROXY set, cert must come from that proxy's CA (PS3_CA_BUNDLE).")
        print(" 3. Last resort on a machine you control: PS3_INSECURE_TLS=1 in ps3/.env (loud warning).")



def short_error(e, provider=None):
    """One-line, human-readable diagnosis + fix hint. Used by entity
    resolution and the UI so a broken TLS stack never dumps a traceback into
    a prospect-facing message. `provider` (or a "name: ..." prefix in the
    error string) tailors the advice: an RSS 403 is a geo-block, a keyed
    403 is a plan/permission problem."""
    s = str(e)
    # per-provider detail if the chain passed "prov1: x; prov2: y"
    prov = provider
    if prov is None and ":" in s:
        head = s.split(":")[0].strip()
        if head in ("rss", "serper", "serpapi", "tavily", "bing", "none"):
            prov = head
    if "CERTIFICATE_VERIFY_FAILED" in s or "certificate verify failed" in s:
        return ("TLS certificate error — this Windows Python has no usable CA "
                "store. Fix: `python -m pip install certifi` (same python that "
                "runs the server), then restart.")
    if "CERTIFICATE" in s and "verify" in s:
        return "TLS verification failed — check your CA bundle (PS3_CA_BUNDLE)."
    if "timed out" in s.lower():
        return "request timed out — check network/proxy, or raise PS3_SEARCH_TIMEOUT."
    if "HTTP Error 403" in s:
        if prov in ("rss", None) and "serper" not in s.lower() and "serpapi" not in s.lower() and "tavily" not in s.lower():
            return ("Google News RSS returned 403 — usually geo-restricted or anti-bot "
                    "from this network (no key is involved with RSS). Fix: set a "
                    "search API key (SERPER_API_KEY) in ps3/.env and restart.")
        return (f"HTTP 403 Forbidden from {prov or 'provider'} — key rejected "
                "(plan/region/permissions/quota); check the key or switch provider.")
    if "HTTP Error 401" in s:
        return "HTTP 401 Unauthorized — check the API key."
    if "HTTP Error 429" in s:
        return "HTTP 429 — provider rate limit; retry later or switch provider."
    if "Name or service not known" in s or "Temporary failure in name resolution" in s:
        return "DNS failure — no network access on this machine."
    if "unreachable" in s.lower() or "WinError 100" in s or "urlopen error" in s:
        return "network unreachable — check connectivity / proxy settings."
    return s[:120]


def tls_doctor(timeout=6):
    """Probe live HTTPS at startup. Returns a dict the server banner +
    /api/probe can surface, so a broken stack is caught *before* the demo.

    {ok, ca_source, detail, fix, tls_ok}
    tls_ok=True means TLS+network themselves are fine; the failure was at the
    HTTP layer (e.g. Google News geo-restricted -> 403)."""
    probe = "https://news.google.com/rss/articles?hl=en-US&gl=US&ceid=US:en"
    try:
        https_open(probe, headers={"User-Agent": "Mozilla/5.0 (PS-3 outreach research)"},
                   timeout=timeout)
        return {"ok": True, "tls_ok": True, "ca_source": ca_source(),
                "detail": "live HTTPS reachable", "fix": None}
    except Exception as e:
        import urllib.error
        detail = short_error(e)
        # An HTTP error means TLS + network SUCCEEDED (the request reached the
        # server and got a response). The failure is endpoint-level, not TLS.
        http_code = getattr(e, "code", None)
        if http_code is not None:
            return {"ok": False, "tls_ok": True, "ca_source": ca_source(),
                    "detail": f"TLS + network OK; Google News probe got HTTP {http_code} "
                              "(likely geo-restricted or blocked on this network).",
                    "fix": "Live retrieval still works via a search API key: "
                           "put your serper.dev key in ps3/.env as SERPER_API_KEY."}
        fix = None
        if "TLS certificate error" in detail or "certificate" in detail.lower():
            fix = ("`python -m pip install certifi` (with the SAME python that runs "
                   "the server) and restart — the server auto-uses certifi's CA bundle. "
                   "If still broken: corporate proxy? set PS3_INSECURE_TLS=1 in ps3/.env.")
        elif "network unreachable" in detail or "DNS failure" in detail:
            fix = "check connectivity / proxy; or run fully offline (corpus only)"
        return {"ok": False, "tls_ok": "certificate" not in detail.lower(),
                "ca_source": ca_source(), "detail": detail, "fix": fix}


if __name__ == "__main__":
    diagnose()
