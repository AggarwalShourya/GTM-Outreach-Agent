"""Zero-dependency server: static UI + a streaming JSON API.

Run:  python -m ps3.server  ->  http://localhost:8377

On startup it loads a `.env` file next to the package (KEY=VALUE lines,
'export ' prefix tolerated) so the demo can run with real Google
embeddings / web-search keys without shell exports.

Endpoints:
  GET /                     index.html (run view + dashboard)
  GET /api/prospects        seeded demo prospect list + value prop + weights
  GET /api/probe            liveness of the live stack (RSS reachable? keys set?)
  GET /api/runs?first=&last=&role=&company=[&confirmed=1]   SSE stream
  GET /api/history?limit=50 dashboard rows
  GET /api/run/<id>         one past run's full result
  DELETE /api/history        clear the dashboard

The SSE stream is what makes the run view animate: each stage event is
pushed as the pipeline emits it, so the UI shows "each stage as it executes."
"""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import os
from . import pipeline, corpus, data, scoring, config, web_search, embed_real, net
from .llm_optional import LLM as _LLM
from .live_retrieval import rss_docs

ROOT = Path(__file__).resolve().parent
STAGE_PAUSE = 0.55  # seconds between stage events so the run view is watchable


def _load_env_file():
    """Minimal .env loader (KEY=VALUE, optional 'export ' prefix, comments).
    Real environment variables always win — .env only fills in what's unset,
    so the demo behaves identically to exporting in the shell."""
    p = ROOT / ".env"
    if not p.exists():
        return
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


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, path):
        p = ROOT / "static" / path
        if not p.exists():
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"not found: {path} — place the UI at ps3/static/{path}".encode())
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # keep the console quiet during the demo

    # ------------------------------------------------------------- GET routes
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            self._html("index.html")
        elif u.path == "/api/prospects":
            self._json({"demo": corpus.list_demo_prospects(), "value_prop": corpus.VALUE_PROP,
                        "reference_date": corpus.REFERENCE_DATE, "weights": scoring.WEIGHTS,
                        "thresholds": {"act": 0.5, "low_band": 0.75, "strong_hook": 3.5}})
        elif u.path == "/api/probe":
            self._json(self._probe())
        elif u.path == "/api/runs":
            self._stream(q)
        elif u.path == "/api/history":
            self._json({"runs": data.list_runs(int(q.get("limit", 50)))})
        elif u.path.startswith("/api/run/"):
            rid = u.path.rsplit("/", 1)[-1]
            row = data.get_run(rid)
            self._json(row or {"error": "not found"}, 200 if row else 404)
        else:
            self._json({"error": "unknown route", "path": u.path}, 404)

    def _probe(self):
        """Report what's live on this machine. No network calls — just reads
        configured providers + whether the key-free RSS provider is available."""
        import os
        prov = config.search_provider()
        keys = {
            "google_embeddings": bool(os.environ.get("GOOGLE_API_KEY")),
            "vertex": os.environ.get("PS3_VERTEX") in ("1", "true")
                      and bool(os.environ.get("PS3_VERTEX_PROJECT")),
            "openai": bool(os.environ.get("OPENAI_API_KEY")),
            "cohere": bool(os.environ.get("COHERE_API_KEY")),
            "tavily": bool(os.environ.get("TAVILY_API_KEY")),
            "serper": bool(os.environ.get("SERPER_API_KEY") or os.environ.get("SERPAPI_KEY")),
            "bing": bool(os.environ.get("BING_SEARCH_KEY")),
        }
        tls = net.tls_doctor()
        return {
            "embedding_provider": embed_real.provider(),
            "embed_model": config.EMBEDDING_MODEL,
            "search_provider": prov,
            "keys": keys,
            "live_search_enabled": config.LIVE_SEARCH,
            "ca_source": net.ca_source(),
            "tls": tls,
            "note": ("Key-free Google News RSS is available for live retrieval; "
                     "set a search API key for richer results."
                     if prov in ("rss", "none") else f"Live search: {prov}.")
                    + ("" if tls["ok"] else f" TLS: {tls['detail']}."
                       + (f" Fix: {tls['fix']}." if tls["fix"] else "")),
        }

    # ---------------------------------------------------------- DELETE routes
    def do_DELETE(self):
        if urlparse(self.path).path == "/api/history":
            data.clear_runs()
            self._json({"cleared": True})
        else:
            self._json({"error": "unknown route"}, 404)

    # --------------------------------------------------------- SSE streaming
    def _stream(self, q):
        first = q.get("first", "")
        last = q.get("last", "")
        role = q.get("role", "")
        company = q.get("company", "")
        confirmed = q.get("confirmed", "0") in ("1", "true", "yes")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        def push(event, payload):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        order = []

        def on_stage(stage):
            order.append(stage["id"])
            push("stage", {"index": len(order) - 1, "id": stage["id"], "name": stage["name"],
                           "status": stage["status"], "detail": stage["detail"], "data": stage["data"]})
            time.sleep(STAGE_PAUSE)  # make each stage visible before the next

        try:
            push("start", {"first": first, "last": last, "role": role, "company": company,
                           "confirmed": confirmed})
            llm = _LLM()  # active only if GOOGLE_API_KEY set and PS3_LLM != 0
            stages, result, status = pipeline.run(first, last, role, company,
                                                  on_stage=on_stage, confirmed=confirmed,
                                                  llm=llm)
            run_id = data.new_run(result, stages)
            push("done", {"run_id": run_id, "result": result, "status": status})
        except Exception as e:
            push("error", {"message": str(e)})
        finally:
            try:
                self.wfile.flush()
            except Exception:
                pass


def main():
    port = int(os.environ.get("PORT", "8377"))
    data._conn()  # init the DB
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"PS-3 outreach process running:  http://localhost:{port}")
    print(f"  embedding provider: {embed_real.provider()}")
    print(f"  search provider:    {config.search_provider()}")
    # First-run TLS doctor: catch a broken CA store BEFORE the demo, not during.
    tls = net.tls_doctor()
    if tls["ok"]:
        print(f"  live retrieval:     OK ({net.ca_source()})")
    else:
        print(f"  LIVE RETRIEVAL BLOCKED: {tls['detail']}")
        if tls["fix"]:
            print(f"  fix: {tls['fix']}")
    if "INSECURE" in net.ca_source():
        print("  WARNING: PS3_INSECURE_TLS=1 — TLS verification is DISABLED.")
    print("Seeded demo prospects are preloaded in the UI; any real company name runs the live path.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
