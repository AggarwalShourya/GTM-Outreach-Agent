"""Key-free live ingestion: Google News RSS.

This is the no-key live path: for a company (or a person + company query) we
pull real, dated, linkable news from the Google News RSS endpoint. Items are
normalized to the same doc shape as `documents.py` so they flow through the
identical chunk/embed/search pipeline. stdlib only (urllib + xml.etree);
on any failure the caller degrades to the corpus / quiet path — never a crash.

Set PS3_LIVE=0 to force the fully-offline path.
"""

import os
import re
import datetime as _dt
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from . import corpus, config

_UA = {"User-Agent": "Mozilla/5.0 (PS-3 outreach research)"}
_TIMEOUT = int(os.environ.get("PS3_SEARCH_TIMEOUT", "20"))


def _enabled():
    return config.LIVE_SEARCH


def _today():
    """REAL wall-clock date (a date object). Live-fetched docs must carry
    their true date, not the demo's reference date (that's only for the
    seeded corpus). Callers wanting a string do _today().isoformat()."""
    return _dt.date.today()


def _parse_when(tag, default_date):
    raw = (tag or "").strip()
    if not raw:
        return default_date.isoformat()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(re.sub(r"\s+", " ", raw), fmt).date().isoformat()
        except ValueError:
            continue
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", raw)
    return m.group(1) if m else default_date.isoformat()


def _rss_items(root):
    NS = {"atom": "http://www.w3.org/2005/Atom"}
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = re.sub(r"<[^>]+>", " ", (item.findtext("description") or ""))
        desc = re.sub(r"\s+", " ", desc).strip()
        src_el = item.find("source")
        source = src_el.text.strip() if src_el is not None and src_el.text else "news"
        pub = _parse_when(item.findtext("pubDate") or item.findtext("updated"), _today())
        yield {"title": title, "summary": desc[:400], "link": link, "source": source, "published": pub}
    for e in root.findall("atom:entry", NS):
        title = (e.findtext("atom:title", namespaces=NS) or "").strip()
        link_el = e.find("atom:link", NS)
        link = link_el.get("href") if link_el is not None else ""
        summ = (e.findtext("atom:summary", namespaces=NS) or e.findtext("atom:content", namespaces=NS) or "")
        summ = re.sub(r"<[^>]+>", " ", summ).strip()[:400]
        upd = e.findtext("atom:updated", namespaces=NS) or e.findtext("atom:published", namespaces=NS)
        pub = _parse_when(upd, _today())
        yield {"title": title, "summary": summ, "link": link, "source": "news", "published": pub}


def _fetch_rss(query, limit):
    from . import net
    url = ("https://news.google.com/rss/search"
           "?q=" + urllib.parse.quote(query)
           + "&hl=en-US&gl=US&ceid=US:en")
    with net.https_open(url, headers=_UA, timeout=_TIMEOUT) as resp:
        data = resp.read()
    root = ET.fromstring(data)
    return list(_rss_items(root))[:limit]


def rss_docs(query, company_name, limit, source_type="Live news", strict=False):
    """Fetch and normalize real news items for `query`.

    Returns pipeline/doc-shaped dicts; [] on any failure. `query` may be a
    bare company name OR a person+company verification query — the caller
    decides. With `strict=True` provider failures RAISE (as RuntimeError)
    instead of degrading to [] — used by entity_resolution so "source
    unreachable" and "source says nothing" stay distinguishable."""
    if not _enabled():
        if strict:
            raise RuntimeError("rss: live retrieval disabled (PS3_LIVE=0)")
        return []
    if not query:
        if strict:
            raise RuntimeError("rss: empty query")
        return []
    try:
        items = _fetch_rss(query, limit)
    except Exception as e:
        if strict:
            raise RuntimeError(f"rss: {e.__class__.__name__}: {e}")
        return []

    out = []
    seen = set()
    today = _today()
    for i, it in enumerate(items):
        link = it["link"]
        key = link or it["title"]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "id": f"rss-{abs(hash(key)) % 10**8}-{i}",
            "company_key": "live",
            "company_name": company_name,
            "source": "news",
            "published": it["published"] if _dt.date.fromisoformat(it["published"]) <= today else today.isoformat(),
            "type": source_type,
            "title": re.sub(r"\s+", " ", it["title"])[:160],
            "body": re.sub(r"\s+", " ", it["summary"])[:400],
            "url": link,
            "retrieval": "live-google-news-rss",
        })
    return out
