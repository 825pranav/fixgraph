"""Validate graph-form extractions against the chunk and the schema (spec §8.1).

Rejects: empty and duplicate entities, relations violating head/tail type constraints, bad
indexes, self-loops, evidence not found in the chunk (fuzzy < threshold), and relations whose
endpoints are not grounded in the chunk. Every rejection reason is counted.

`conf` of a kept relation = min(head grounding, tail grounding, evidence score): a
grounding-based confidence, since small models' self-reported confidences are uninformative.
"""

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from pydantic import BaseModel, Field

from fixgraph.kg.extraction import ChunkExtraction
from fixgraph.kg.schema import relation_allowed

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    return _WS.sub(" ", text).strip(" .,:;!?\"'")


def partial_ratio(needle: str, haystack: str) -> float:
    """Best similarity (0-1) between `needle` and any same-length window of `haystack`."""
    n, h = normalize(needle), normalize(haystack)
    if not n:
        return 0.0
    if n in h:
        return 1.0
    if len(n) >= len(h):
        return SequenceMatcher(None, n, h).ratio()
    best = 0.0
    matcher = SequenceMatcher(None, n, h, autojunk=False)
    for block in matcher.get_matching_blocks():
        start = min(max(0, block.b - block.a), len(h) - len(n))
        window = h[start : start + len(n)]
        best = max(best, SequenceMatcher(None, n, window).ratio())
        if best == 1.0:
            return best
    # Fallback: word windows of +-1 word around the needle's length, which tolerates small
    # insertions/deletions ("turn on the bluetooth" vs "turn on bluetooth").
    n_words, h_words = n.split(" "), h.split(" ")
    for size in range(max(1, len(n_words) - 1), len(n_words) + 2):
        for i in range(0, max(1, len(h_words) - size + 1)):
            window = " ".join(h_words[i : i + size])
            if abs(len(window) - len(n)) > max(8, len(n) // 3):
                continue
            best = max(best, SequenceMatcher(None, n, window).ratio())
    return best


class Span(BaseModel):
    start: int
    end: int


class ValidEntity(BaseModel):
    type: str
    text: str
    grounding: float  # partial_ratio of the text against the chunk
    span: Span | None = None  # exact location when found verbatim


class ValidRelation(BaseModel):
    head: int  # index into ValidatedExtraction.entities
    rel: str
    tail: int
    conf: float
    evidence: str
    evidence_score: float


class ValidatedExtraction(BaseModel):
    entities: list[ValidEntity] = Field(default_factory=lambda: list[ValidEntity]())
    relations: list[ValidRelation] = Field(default_factory=lambda: list[ValidRelation]())
    rejects: dict[str, int] = Field(default_factory=lambda: dict[str, int]())


def locate(text: str, chunk_text: str) -> Span | None:
    """Case-insensitive exact location of `text` in the chunk (quote-normalized)."""
    hay = chunk_text.translate(_QUOTES).lower()
    needle = text.translate(_QUOTES).lower().strip()
    idx = hay.find(needle) if needle else -1
    return None if idx < 0 else Span(start=idx, end=idx + len(needle))


def validate(
    raw: ChunkExtraction,
    chunk_text: str,
    context: str = "",
    evidence_threshold: float = 0.9,
    entity_threshold: float = 0.8,
) -> ValidatedExtraction:
    """Ground against everything the model saw (`context` = article title + section heading,
    plus the chunk); spans always refer to the chunk text."""
    seen_text = f"{context}\n{chunk_text}" if context else chunk_text
    rejects: Counter[str] = Counter()
    entities: list[ValidEntity] = []
    remap: dict[int, int] = {}
    seen: dict[tuple[str, str], int] = {}
    for i, e in enumerate(raw.entities):
        text = e.text.strip()
        if not normalize(text):
            rejects["entity_empty"] += 1
            continue
        key = (e.type, normalize(text))
        if key in seen:
            rejects["entity_duplicate"] += 1
            remap[i] = seen[key]  # relations to the duplicate point at the first mention
            continue
        seen[key] = remap[i] = len(entities)
        entities.append(
            ValidEntity(
                type=e.type,
                text=text,
                grounding=partial_ratio(text, seen_text),
                span=locate(text, chunk_text),
            )
        )

    relations: list[ValidRelation] = []
    seen_rel: set[tuple[int, str, int]] = set()
    for r in raw.relations:
        if r.head not in remap or r.tail not in remap:
            rejects["relation_bad_index"] += 1
            continue
        h, t = remap[r.head], remap[r.tail]
        if h == t:
            rejects["relation_self_loop"] += 1
            continue
        if not relation_allowed(r.rel, entities[h].type, entities[t].type):
            rejects["relation_type_violation"] += 1
            continue
        score = partial_ratio(r.evidence, seen_text)
        if score < evidence_threshold:
            rejects["relation_evidence_not_found"] += 1
            continue
        if min(entities[h].grounding, entities[t].grounding) < entity_threshold:
            rejects["relation_entity_not_grounded"] += 1
            continue
        if (h, r.rel, t) in seen_rel:
            rejects["relation_duplicate"] += 1
            continue
        seen_rel.add((h, r.rel, t))
        conf = min(entities[h].grounding, entities[t].grounding, score)
        relations.append(
            ValidRelation(
                head=h,
                rel=r.rel,
                tail=t,
                conf=round(conf, 3),
                evidence=r.evidence,
                evidence_score=round(score, 3),
            )
        )
    return ValidatedExtraction(entities=entities, relations=relations, rejects=dict(rejects))
