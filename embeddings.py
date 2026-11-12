"""Deterministic embeddings — the offline dense channel.

Hashed character n-gram vectors (512-d, no download, no key). The seam where
a real encoder (embed_real.py: Google / OpenAI / Cohere) plugs in; the rest
of the pipeline (cosine, RRF) is unchanged.
"""
import hashlib
import math

DIM = 512
_NGRAMS = (3, 4)


def _stable_token_hash(tok, salt):
    h = hashlib.md5(f"{salt}:{tok}".encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big")
    sign = 1 if (h[8] & 1) == 0 else -1
    return sign, x % DIM


def _char_ngrams(text, n):
    t = text.lower()
    for i in range(max(1, len(t) - n + 1)):
        yield t[i:i + n]


def _word_bigrams(text):
    words = [w for w in text.lower().split() if len(w) > 2]
    for i in range(len(words) - 1):
        yield words[i] + " " + words[i + 1]


def embed(text):
    vec = [0.0] * DIM
    for n in _NGRAMS:
        for gram in _char_ngrams(text, n):
            sign, b = _stable_token_hash(gram, salt=n)
            vec[b] += sign * 1.0
    for bigram in _word_bigrams(text):
        sign, b = _stable_token_hash(bigram, salt=99)
        vec[b] += sign * 2.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
