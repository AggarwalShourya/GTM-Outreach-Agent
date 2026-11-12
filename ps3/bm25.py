"""Okapi BM25 over the chunked index."""
import math
import re

K1 = 1.2
B = 0.75
_TOKEN = re.compile(r"[a-z0-9$%.\-]+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks):
        self.chunk_ids = [c["id"] for c in chunks]
        self.n = len(chunks)
        self.df = {}
        self.tf = {}
        self.len = {}
        total_len = 0
        for c in chunks:
            toks = tokenize(c["text"])
            freq = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            self.tf[c["id"]] = freq
            self.len[c["id"]] = len(toks)
            total_len += len(toks)
            for t in freq:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (total_len / self.n) if self.n else 0.0

    def score(self, query, chunk_id, tf_map):
        q = tokenize(query)
        if not q or self.n == 0:
            return 0.0
        ctf = tf_map.get(chunk_id, {})
        dl = self.len.get(chunk_id, 0)
        s = 0.0
        for t in q:
            if t not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            f = ctf.get(t, 0)
            s += idf * (f * (K1 + 1)) / (f + K1 * (1 - B + B * dl / self.avgdl))
        return s

    def score_all(self, query, top_k=50):
        q = tokenize(query)
        hits = {}
        postings = self.postings
        for t in q:
            if t in self.df:
                for cid in postings.get(t, []):
                    hits.setdefault(cid, 0)
        scored = []
        for cid in hits:
            sc = self.score(query, cid, self.tf)
            if sc > 0:
                scored.append((cid, sc))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def build_postings(self):
        self._postings = {}
        for cid, freq in self.tf.items():
            for t in freq:
                self._postings.setdefault(t, set()).add(cid)
        self._postings = {t: sorted(s) for t, s in self._postings.items()}

    @property
    def postings(self):
        if not hasattr(self, "_postings"):
            self.build_postings()
        return self._postings
