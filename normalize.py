"""Small text normalization helpers shared across the pipeline."""
import re


def norm(s):
    """Lowercase + strip for comparisons."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")


def age_days(date_str, reference_date):
    """Whole days between a signal date and the reference date. None if unparseable.
    A negative age means the signal date is *after* the reference date."""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(date_str[:10])
        r = _dt.date.fromisoformat(reference_date[:10])
        return (r - d).days
    except Exception:
        return None
