"""Seeded demo corpus: the 6 designed prospects + the shared constants.

Seeded prospects carry a deterministic `resolution` (their four edge cases
must stay reproducible mid-interview). Everything else in this module is the
shared context the pipeline and retrieval build on. A non-seeded prospect
returns None from `find_prospect`, which is the trigger for the *live*
entity-resolution path (entity_resolution.py).
"""

import re

VALUE_PROP = (
    "We help B2B sales teams cut prospect research from ~20 minutes to ~2 "
    "minutes by auto-detecting the one real signal worth mentioning and "
    "drafting the message around it."
)

# Deterministic "today" for the SEEDED corpus so recency scoring of demo
# docs is stable. Live (web-search) signals use the real wall-clock today
# instead (see scoring / pipeline), so real news isn't scored against a
# fictional date.
REFERENCE_DATE = "2026-11-10"

RELEVANT_TRIGGERS = {
    "funding": "Fresh budget; usually means they're hiring and scaling fast.",
    "job_postings": "Active hiring = they're about to spend on the tool.",
    "exec_interview": "A public statement about a pain you directly solve.",
    "blog": "Self-reported pain about the exact work we automate.",
    "product": "New product/launch = a moment when teams need a go-to-market assist.",
    "social": "Personal, recent, and easy to reference naturally in a first line.",
    "press": "Company-level news; weaker hook unless it names a specific pain.",
}

PROSPECTS = [
    {"slug": "maya-chen-northwind", "first": "Maya", "last": "Chen",
     "role": "VP Marketing", "company": "Northwind Cloud", "company_key": "northwind",
     "resolution": {"match": "exact", "confidence": 0.97,
                    "candidates": [{"company": "Northwind Cloud", "role": "VP Marketing",
                                     "confidence": 0.97, "source": "company site + 2 corroborating sources"}],
                    "note": "Unambiguous single match."}},
    {"slug": "dan-okafor-smallco", "first": "Dan", "last": "Okafor",
     "role": "Head of Partnerships", "company": "Smallco Robotics", "company_key": "smallco",
     "resolution": {"match": "exact", "confidence": 0.9,
                    "candidates": [{"company": "Smallco Robotics", "role": "Head of Partnerships",
                                     "confidence": 0.9, "source": "single public profile"}],
                    "note": "Resolves cleanly; the company is quiet, so the test is retrieval + fallback, not identity."}},
    {"slug": "priya-nair-vantage", "first": "Priya", "last": "Nair",
     "role": "CTO", "company": "Vantage AI", "company_key": "vantage",
     "resolution": {"match": "exact", "confidence": 0.94,
                    "candidates": [{"company": "Vantage AI", "role": "CTO", "confidence": 0.94,
                                     "source": "company site + funding press"}],
                    "note": "Identity is clean; the signals themselves disagree."}},
    {"slug": "marco-rossi-ledgerline", "first": "Marco", "last": "Rossi",
     "role": "Head of Product", "company": "Ledgerline", "company_key": "ledgerline",
     "resolution": {"match": "exact", "confidence": 0.91,
                    "candidates": [{"company": "Ledgerline", "role": "Head of Product", "confidence": 0.91,
                                     "source": "company site"}],
                    "note": "Identity resolves, but the strongest signal no longer belongs to him."}},
    {"slug": "sarah-lee-acme", "first": "Sarah", "last": "Lee",
     "role": "Director of Sales", "company": "Acme", "company_key": "acme",
     "resolution": {"match": "ambiguous", "confidence": 0.4,
                    "candidates": [
                        {"company": "Acme Inc", "role": "Director of Sales", "confidence": 0.52,
                         "source": "consumer retail brand, ~1.2k results"},
                        {"company": "Acme Systems", "role": "Sales Director", "confidence": 0.44,
                         "source": "B2B infra company, ~900 results"}],
                    "note": "Common name + a company name that maps to two different businesses. Below the act threshold."}},
    {"slug": "nadia-rahman-helio", "first": "Nadia", "last": "Rahman",
     "role": "Head of Growth", "company": "Helio Analytics", "company_key": "helio",
     "resolution": {"match": "exact", "confidence": 0.9,
                    "candidates": [{"company": "Helio Analytics", "role": "Head of Growth",
                                     "confidence": 0.9, "source": "company site + single public profile"}],
                    "note": "Identity is clean and the company is active; the test is that plenty of signals exist but none is a trigger we can pitch against."}},
]


def list_demo_prospects():
    return [
        {"slug": p["slug"], "label": f"{p['first']} {p['last']} — {p['role']}, {p['company']}"}
        for p in PROSPECTS
    ]


def find_prospect(first, last, company):
    """Match typed input to a seeded prospect. None if not seeded."""
    def s(x):
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())
    f, l, c = s(first), s(last), s(company)
    for p in PROSPECTS:
        if f == s(p["first"]) and l == s(p["last"]) and c == s(p["company"]):
            return p
    return None


def find_by_slug(slug):
    for p in PROSPECTS:
        if p["slug"] == slug:
            return p
    return None
