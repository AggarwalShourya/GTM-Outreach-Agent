"""Run-history store backed by SQLite. Backs the dashboard: a history of every
run with status, confidence, top hook, and the draft. Persists across restarts.
"""

import json
import sqlite3
import uuid
from pathlib import Path

DB_PATH = Path(__file__).with_name("runs.db")
CONN = None


def _conn():
    global CONN
    if CONN is None:
        CONN = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        CONN.row_factory = sqlite3.Row
        _init(CONN)
    return CONN


def _init(c):
    c.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            ts TEXT,
            prospect TEXT,
            status TEXT,
            confidence TEXT,
            mode TEXT,
            top_hook TEXT,
            draft TEXT,
            result_json TEXT
        )"""
    )
    c.commit()


def new_run(result, stages):
    row = {
        "id": uuid.uuid4().hex[:12],
        "ts": result.get("finished_at"),
        "prospect": f"{result['prospect']['first']} {result['prospect']['last']} ({result['prospect']['company']})",
        "status": result.get("status"),
        "confidence": result.get("confidence"),
        "mode": result.get("mode") or "-",
        "top_hook": (result.get("top_hook") or {}).get("headline") or (result.get("confirmation_message") or "—")[:60],
        "draft": (result.get("draft") or {}).get("subject") or "—",
        "result_json": json.dumps(result),
    }
    c = _conn()
    c.execute(
        "INSERT INTO runs (id, ts, prospect, status, confidence, mode, top_hook, draft, result_json) "
        "VALUES (:id,:ts,:prospect,:status,:confidence,:mode,:top_hook,:draft,:result_json)",
        row,
    )
    c.commit()
    return row["id"]


def list_runs(limit=50):
    c = _conn()
    rows = c.execute(
        "SELECT id, ts, prospect, status, confidence, mode, top_hook, draft "
        "FROM runs ORDER BY rowid DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id):
    c = _conn()
    r = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["result"] = json.loads(d["result_json"])
    return d


def clear_runs():
    c = _conn()
    c.execute("DELETE FROM runs")
    c.commit()
