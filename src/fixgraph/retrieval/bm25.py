"""BM25 as Qdrant sparse vectors.

Documents store the BM25 term-frequency component tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl));
the IDF factor is applied by Qdrant (`Modifier.IDF`) at query time. Queries are unit weights
per unique term. Term ids are stable CRC32 hashes, so no vocabulary file is needed.

Used by: retrieval.index (document vectors), retrieval.hybrid (query vectors), retrieval.rerank
(tokenizer for the lexical fake), api/app.py. No fixgraph imports.
"""

import re
import zlib
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*")
STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
        "from", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
        "our", "so", "than", "that", "the", "their", "then", "there", "these", "this", "to",
        "up", "was", "what", "when", "where", "which", "while", "who", "why", "will", "with",
        "you", "your",
    )
)  # fmt: skip


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [t for t in toks if t not in STOPWORDS]


def term_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


class BM25Encoder:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 150.0) -> None:
        self.k1, self.b, self.avgdl = k1, b, avgdl

    def fit(self, docs: list[str]) -> "BM25Encoder":
        lengths = [len(tokenize(d)) for d in docs]
        self.avgdl = sum(lengths) / len(lengths) if lengths else 1.0
        return self

    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        dl = len(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        weights: dict[int, float] = {}
        for tok, tf in Counter(toks).items():
            tid = term_id(tok)
            weights[tid] = weights.get(tid, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        ids = sorted(weights)
        return ids, [weights[i] for i in ids]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        ids = sorted({term_id(t) for t in tokenize(text)})
        return ids, [1.0] * len(ids)
