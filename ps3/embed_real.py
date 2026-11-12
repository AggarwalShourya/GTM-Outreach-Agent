"""Real embedding providers — the swap-in seam.

`embeddings.py` (hashed char-n-grams) is the deterministic, key-free fallback.
This module provides hosted models for the dense/semantic channel, with the
exact same `embed(text) -> vector` contract, so the fusion + rerank code
downstream is unchanged.

Provider resolution lives in `config.embedding_provider()`:
  * "google"   GOOGLE_API_KEY  -> Generative Language batchEmbedContents
  * "vertex"   PS3_VERTEX=1    -> Vertex AI (service account or ADC)
  * "openai"   OPENAI_API_KEY  -> /v1/embeddings
  * "cohere"   COHERE_API_KEY  -> /v1/embed
  * "local"    -> embeddings.embed (deterministic, always available, offline)

All HTTP is stdlib `urllib` — no new dependencies. On any failure a provider
returns None / raises and `embed_batch` falls back to local vectors, so the
run never dies.
"""

import json
import math
import urllib.request

from . import embeddings as _local
from . import config

_TIMEOUT = int(__import__("os").environ.get("PS3_EMBED_TIMEOUT", "25"))
_UA = {"User-Agent": "Mozilla/5.0 (PS-3 outreach research)"}


def _post(url, headers, payload):
    from . import net
    with net.https_open(url, headers={**_UA, **headers, "Content-Type": "application/json"},
                        data=json.dumps(payload).encode("utf-8"), method="POST",
                        timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _l2(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


# ---------------------------------------------------------------- providers
def embed_google(texts):
    """Google Generative Language embedding via a plain API key.

    Default model: gemini-embedding-001 (768-d). Uses the per-text
    `embedContent` endpoint with the parts content shape — the shape the
    plain API key accepts (`batchEmbedContents` is 404; text-embedding-004/005
    are Vertex-only). One call per text; at this corpus size that's fine.
    stdlib urllib; no google client required.
    """
    import os
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    model = config.EMBEDDING_MODEL
    out = []
    for t in texts:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
               f":embedContent?alt=json&key={key}")
        # NB: no "autoTruncate" here — that parameter is Vertex-only and the
        # plain generativelanguage endpoint 400s on it.
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": t}]},
            "outputDimensionality": 768,
        }
        data = _post(url, {}, payload)
        values = data.get("embedding", {}).get("values", [])
        out.append(_l2(values))
    return _pad(out)


# --------------------------------------------------------------- vertex AI
def _vertex_token():
    """Mint a bearer token for Vertex AI. Tries, in order:
    1. `google.auth` service-account file (PS3_VERTEX_SA) or ADC (installed)
    2. `gcloud auth print-access-token` (user's local gcloud auth)
    3. raises a clear, actionable error.
    """
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    try:
        if config.VERTEX_SERVICE_ACCOUNT:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                config.VERTEX_SERVICE_ACCOUNT, scopes=scopes)
        else:
            from google.auth import default as _default
            creds, _ = _default(scopes=scopes)
        import google.auth.transport.requests as _transport
        creds.refresh(_transport.Request())
        return creds.token
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError("Vertex auth via google-auth failed: " + str(e))
    import subprocess
    try:
        out = subprocess.run(
            ["gcloud", "auth", "print-access-token", "--project", config.VERTEX_PROJECT],
            capture_output=True, text=True, timeout=30, check=True)
        return out.stdout.strip()
    except Exception as e:
        raise RuntimeError(
            "No Vertex credentials. Either pip install google-auth (service "
            f"account: {config.VERTEX_SERVICE_ACCOUNT or 'ADC'}) or authenticate "
            f"gcloud with the right project. Details: {e}")


def embed_vertex(texts):
    """Google Vertex AI embeddings endpoint (project-scoped).

    POST {location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}
         /publishers/google/models/{model}:predict
    text-embedding-004 returns predictions[].values.
    """
    if not config.VERTEX_PROJECT:
        raise RuntimeError("PS3_VERTEX_PROJECT is not set for the Vertex provider")
    token = _vertex_token()
    model = config.EMBEDDING_MODEL
    location = config.VERTEX_LOCATION
    base = (f"https://{location}-aiplatform.googleapis.com/v1/projects/"
            f"{config.VERTEX_PROJECT}/locations/{location}/publishers/google")
    out = []
    for i in range(0, len(texts), 30):
        batch = texts[i:i + 30]
        payload = {
            "instances": [{"content": t} for t in batch],
            "parameters": {"outputDimensionality": 768, "autoTruncate": True},
        }
        data = _post(f"{base}/models/{model}:predict",
                     {"Authorization": "Bearer " + token}, payload)
        for p in data.get("predictions", []):
            out.append(_l2(p.get("values", [])))
    return _pad(out)


def embed_openai(texts):
    import os
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_EMBEDDING_KEY")
    if not key:
        return None
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    data = _post(
        base.rstrip("/") + "/v1/embeddings",
        {"Authorization": "Bearer " + key},
        {"model": config.EMBEDDING_MODEL, "input": texts, "encoding_format": "float"},
    )
    vecs = [_l2(d["embedding"]) for d in data["data"]]
    return _pad(vecs)


def embed_cohere(texts):
    import os
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        return None
    data = _post(
        "https://api.cohere.ai/v1/embed",
        {"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        {"model": "embed-english-v3.0", "texts": texts,
         "input_type": "search_document", "embedding_types": ["float"]},
    )
    return _pad([_l2(v) for v in data.get("embeddings", [])])


_PROVIDER_FN = {
    "openai": embed_openai,
    "cohere": embed_cohere,
    "google": embed_google,
    "vertex": embed_vertex,
    "local": lambda texts: [_local.embed(t) for t in texts],
}


def _pad(vecs):
    if not vecs:
        return []
    dim = max(len(v) for v in vecs)
    return [(v + [0.0] * dim)[:dim] for v in vecs]


# ---------------------------------------------------------------- public api
def provider():
    return config.embedding_provider()


def embed_batch(texts):
    """Embed a list of texts with the configured provider; fall back to the
    local deterministic vectors on any provider error so the run never dies.
    Returns (vectors, provider_name_used)."""
    name = provider()
    try:
        vecs = _PROVIDER_FN.get(name, _PROVIDER_FN["local"])(texts)
        if vecs is None:
            vecs = _PROVIDER_FN["local"](texts)
            name = "local (fallback)"
        return vecs, name
    except Exception:
        return _PROVIDER_FN["local"](texts), "local (fallback)"
