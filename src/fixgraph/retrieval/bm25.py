"""BM25 as Qdrant sparse vectors.

Documents store the BM25 term-frequency component tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl));
the IDF factor is applied by Qdrant (`Modifier.IDF`) at query time. Queries are unit weights
per unique term. Term ids are stable CRC32 hashes, so no vocabulary file is needed.

Used by: retrieval.index (document vectors), retrieval.hybrid (query vectors), retrieval.rerank
(tokenizer for the lexical fake), api/app.py. No fixgraph imports.
"""

# Imports: regex tokenizer, crc32 for stable term ids, Counter for term frequencies.
import re
import zlib
from collections import Counter

# Tokenizer pattern keeps words and numbers, including joined ones like "wi-fi", "17.2", "don't".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'.][a-z0-9]+)*")
# Common words dropped before scoring, since they match almost every chunk.
STOPWORDS = frozenset(
    (
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for",
        "from", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
        "our", "so", "than", "that", "the", "their", "then", "there", "these", "this", "to",
        "up", "was", "what", "when", "where", "which", "while", "who", "why", "will", "with",
        "you", "your",
    )
)  # fmt: skip


# Lowercases text, splits it into tokens and drops stopwords; shared by doc and query encoding.
def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower().replace("’", "'"))
    return [t for t in toks if t not in STOPWORDS]


# Maps a token to a stable integer id (Qdrant sparse index), so no vocabulary file is needed.
def term_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & 0x7FFFFFFF


# Turns text into BM25 sparse vectors for Qdrant: documents get BM25 term weights, queries get 1s.
class BM25Encoder:
    # k1 and b are the usual BM25 knobs; avgdl is the average document length in tokens.
    def __init__(self, k1: float = 1.2, b: float = 0.75, avgdl: float = 150.0) -> None:
        self.k1, self.b, self.avgdl = k1, b, avgdl

    # Learns the average chunk length from the corpus at index build time.
    def fit(self, docs: list[str]) -> "BM25Encoder":
        lengths = [len(tokenize(d)) for d in docs]
        self.avgdl = sum(lengths) / len(lengths) if lengths else 1.0
        return self

    # Encodes one chunk: BM25 term-frequency weight per term (IDF is added later by Qdrant).
    def encode_doc(self, text: str) -> tuple[list[int], list[float]]:
        toks = tokenize(text)
        dl = len(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        weights: dict[int, float] = {}
        # Weight each unique term by its saturated term frequency, normalized by document length.
        for tok, tf in Counter(toks).items():
            tid = term_id(tok)
            weights[tid] = weights.get(tid, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        # Return sorted ids and matching weights, the format Qdrant's SparseVector expects.
        ids = sorted(weights)
        return ids, [weights[i] for i in ids]

    # Encodes a question: each unique term gets weight 1, so the score sums the IDF-weighted
    # document weights of the matching terms.
    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        ids = sorted({term_id(t) for t in tokenize(text)})
        return ids, [1.0] * len(ids)
