"""Entity resolution (spec §8.2).

1. Rule-based normalization for Product, OSVersion, ErrorCode, Component, Feature via the
   ontology.
2. Embedding clustering for Symptom, Cause, Fix: average-linkage agglomerative clustering on
   cosine distance with a per-type threshold; the medoid (most central member, ties broken by
   mention count) becomes the canonical text.
3. Optional LLM adjudication for borderline cluster pairs (similarity within a band below the
   threshold), cached.
4. Every merge decision is logged.

Used by: kg.build (canonicalization), kg/cli.py (`llm_adjudicator` for `kg build`).
Uses: core.ontology, embeddings, kg.validation.normalize, llm.structured.
"""

# Imports: scipy for hierarchical clustering, numpy for embeddings, LLM types for the adjudicator.
import hashlib
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, Field
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from fixgraph.core.ontology import Ontology, parse_os_version
from fixgraph.embeddings import Embedder, Matrix
from fixgraph.kg.validation import normalize
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

logger = logging.getLogger(__name__)

# Free-text types are merged by embedding clustering; all other types use ontology rules.
CLUSTERED_TYPES = ("Symptom", "Cause", "Fix")
# Cosine-distance thresholds (1 - similarity); tuned on a labeled pair sample (DECISIONS.md).
DEFAULT_THRESHOLDS: dict[str, float] = {"Symptom": 0.12, "Cause": 0.12, "Fix": 0.10}
MAX_AGGLOMERATIVE = 12_000  # above this, fall back to greedy leader clustering (memory bound)

# Regexes: error-code number, leading "the/your/a" words, and characters to replace in slugs.
_ERROR_CODE_RE = re.compile(r"(?:error\s*(?:code)?\s*)?(-?\d{1,6})", re.IGNORECASE)
_LEADING_DET_RE = re.compile(r"^(?:the|your|a|an|my)\s+", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


# The resolved form of a mention: final node label, canonical text and extra properties.
class Canonical(BaseModel):
    label: str  # node label (Product, ProductFamily, OSVersion, ...)
    text: str  # canonical text
    props: dict[str, str | int | list[str]] = Field(
        default_factory=lambda: dict[str, str | int | list[str]]()
    )


# One logged merge decision, so every collapsed name can be audited later.
class MergeRecord(BaseModel):
    type: str
    canonical: str
    members: list[str]
    method: str  # rule | cluster | llm


# Lowercase, dash-separated, length-capped version of a text for readable node ids.
def slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")[:60] or "x"


# Stable node id like "product:iphone:ab12cd" built from the label and canonical text.
def node_id(label: str, canonical: str) -> str:
    base = f"{label.lower()}:{slug(canonical)}"
    # Short hash of the exact canonical text keeps IDs unique when slugs collide.
    digest = hashlib.sha1(f"{label}|{canonical}".encode()).hexdigest()[:6]
    return f"{base}:{digest}"


# Drop a leading "the/your/a/an/my" and trailing dot so "your iPhone" matches "iPhone".
def strip_determiner(text: str) -> str:
    return _LEADING_DET_RE.sub("", text.strip().strip(".")).strip()


# ---------------------------------------------------------------------------
# 1. Rule-based canonicalization
# ---------------------------------------------------------------------------


# Rule-based canonical form for non-clustered types; called by build for each mention.
def canonical_rule(type_: str, text: str, ontology: Ontology) -> Canonical | None:
    """Canonical form for rule-typed entities; None means drop the mention."""
    # Products: match a known model, else a single known family, else keep the cleaned text.
    if type_ == "Product":
        products = ontology.products_in(text)
        if products:
            family, name = products[0]
            return Canonical(label="Product", text=name, props={"family": family})
        families = ontology.families_in(text)
        if len(families) == 1:
            return Canonical(label="Product", text=families[0], props={"family": families[0]})
        cleaned = strip_determiner(text)
        return Canonical(label="Product", text=cleaned, props={"family": ""}) if cleaned else None
    # OS versions: keep only real versions and store platform plus major/minor/patch numbers.
    if type_ == "OSVersion":
        versions = ontology.os_versions_in(text)
        if not versions:
            return None  # "macOS", "the latest iOS": not a version
        platform, major, minor, patch = parse_os_version(versions[0])
        return Canonical(
            label="OSVersion",
            text=versions[0],
            props={"platform": platform, "major": major, "minor": minor, "patch": patch},
        )
    # Error codes: normalize to "error <number>".
    if type_ == "ErrorCode":
        m = _ERROR_CODE_RE.search(text)
        if not m:
            return None
        return Canonical(label="ErrorCode", text=f"error {m.group(1)}", props={"code": m.group(1)})
    # Components/features: snap to the ontology vocabulary if possible, else lowercase cleaned text.
    if type_ in ("Component", "Feature"):
        vocab = ontology.components_in(text) if type_ == "Component" else ontology.features_in(text)
        if vocab:
            # Prefer the longest (most specific) vocabulary match, e.g. "iCloud Photos".
            return Canonical(
                label=type_, text=max(vocab, key=len), props={"name": max(vocab, key=len)}
            )
        cleaned = strip_determiner(text).lower()
        return Canonical(label=type_, text=cleaned, props={"name": cleaned}) if cleaned else None
    raise ValueError(f"{type_} is not rule-canonicalized")


# ---------------------------------------------------------------------------
# 2. Embedding clustering
# ---------------------------------------------------------------------------


# Pick the most central member of a cluster (highest summed similarity) as its display text.
def _medoid(idx: list[int], emb: Matrix, counts: list[int]) -> int:
    sub = emb[idx]
    centrality = (sub @ sub.T).sum(axis=1)
    return max(range(len(idx)), key=lambda i: (round(float(centrality[i]), 4), counts[idx[i]], -i))


# Group similar phrases: agglomerative clustering when small enough, greedy leaders when huge.
def cluster_texts(
    texts: list[str], counts: list[int], emb: Matrix, threshold: float
) -> list[list[int]]:
    """Group indices of `texts` whose embeddings are within `threshold` cosine distance
    (average linkage). Deterministic: input order is sorted by the caller."""
    n = len(texts)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    # Normal path: average-linkage clustering on cosine distance, cut at the type's threshold.
    if n <= MAX_AGGLOMERATIVE:
        z = linkage(pdist(emb.astype(np.float64), metric="cosine"), method="average")
        labels = fcluster(z, t=threshold, criterion="distance")
        groups: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            groups[int(lab)].append(i)
        return sorted(groups.values(), key=lambda g: g[0])
    # Greedy leader clustering: most frequent texts first become leaders.
    # Large path: each text joins the nearest leader if close enough, else becomes a new leader.
    order = sorted(range(n), key=lambda i: (-counts[i], texts[i]))
    leaders: list[int] = []
    members: dict[int, list[int]] = {}
    for i in order:
        if leaders:
            sims = emb[leaders] @ emb[i]
            j = int(np.argmax(sims))
            if 1.0 - float(sims[j]) <= threshold:
                members[leaders[j]].append(i)
                continue
        leaders.append(i)
        members[i] = [i]
    return sorted((sorted(m) for m in members.values()), key=lambda g: g[0])


# Signature of the optional LLM judge that decides if two phrases mean the same thing.
Adjudicator = Callable[[str, str, str], bool]  # (type, text_a, text_b) -> same meaning?


# Result of clustering one type: member text -> canonical text, plus the merge log.
class ClusterResult(BaseModel):
    canonical_of: dict[str, str]  # normalized member text -> canonical text
    merges: list[MergeRecord]


# Resolve one free-text type: embed every distinct text, cluster, then map each to its medoid.
def resolve_clustered(
    type_: str,
    mention_counts: Counter[str],
    embedder: Embedder,
    threshold: float,
    adjudicate: Adjudicator | None = None,
    band: float = 0.05,
    max_pairs: int = 500,
) -> ClusterResult:
    """Cluster the distinct texts of one type; map every text to its cluster's medoid."""
    texts = sorted(mention_counts)
    counts = [mention_counts[t] for t in texts]
    emb = embedder.encode(texts)
    groups = cluster_texts(texts, counts, emb, threshold)
    merges: list[MergeRecord] = []

    # 3. LLM adjudication: merge borderline cluster pairs the adjudicator says are the same.
    if adjudicate is not None and len(groups) > 1:
        centroids = np.stack([emb[g].mean(axis=0) for g in groups])
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
        parent = list(range(len(groups)))

        # Union-find lookup so merged clusters can be chained together.
        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # Only ask the LLM about cluster pairs whose similarity sits just below the merge threshold.
        sims = centroids @ centroids.T
        lo, hi = 1.0 - threshold - band, 1.0 - threshold
        pairs = [
            (i, j)
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
            if lo <= sims[i, j] < hi
        ]
        pairs = sorted(pairs, key=lambda ij: -float(sims[ij]))[:max_pairs]
        # Merge the pair when the LLM says yes, then rebuild the groups from the union-find roots.
        for i, j in pairs:
            a = texts[groups[i][_medoid(groups[i], emb, counts)]]
            b = texts[groups[j][_medoid(groups[j], emb, counts)]]
            if find(i) != find(j) and adjudicate(type_, a, b):
                parent[find(j)] = find(i)
                merges.append(MergeRecord(type=type_, canonical=a, members=[a, b], method="llm"))
        merged: dict[int, list[int]] = defaultdict(list)
        for gi, g in enumerate(groups):
            merged[find(gi)].extend(g)
        groups = sorted((sorted(g) for g in merged.values()), key=lambda g: g[0])

    # Pick each group's medoid as canonical text and log multi-member groups as merges.
    canonical_of: dict[str, str] = {}
    for g in groups:
        canon = texts[g[_medoid(g, emb, counts)]]
        for i in g:
            canonical_of[texts[i]] = canon
        if len(g) > 1:
            merges.append(
                MergeRecord(
                    type=type_, canonical=canon, members=[texts[i] for i in g], method="cluster"
                )
            )
    return ClusterResult(canonical_of=canonical_of, merges=merges)


# Key used before clustering so case, quotes and "the/your" differences do not split phrases.
def clustering_key(text: str) -> str:
    """Normalization applied before clustering (case, quotes, determiners, punctuation)."""
    return normalize(strip_determiner(text))


# ---------------------------------------------------------------------------
# 3. LLM adjudicator (borderline pairs only; responses are cached by the LLM client)
# ---------------------------------------------------------------------------


# Schema for the adjudicator's yes/no JSON reply.
class _SameMeaning(BaseModel):
    same: bool


# Build an adjudicator backed by the LLM; build_kg calls it on borderline cluster pairs.
def llm_adjudicator(client: LLMClient, model: str, num_ctx: int = 2048) -> Adjudicator:
    # Ask "same meaning?" in a fixed order (better cache hits); a bad reply counts as "no".
    def adjudicate(type_: str, a: str, b: str) -> bool:
        a, b = sorted((a, b))  # order-independent cache key
        prompt = (
            f"Do these two {type_.lower()} descriptions from Apple support articles mean the "
            f"same thing, so they should be one node in a knowledge graph?\n"
            f'A: {a}\nB: {b}\nAnswer with JSON {{"same": true|false}}.'
        )
        request = LLMRequest(
            model=model, messages=[ChatMessage(role="user", content=prompt)], num_ctx=num_ctx
        )
        try:
            return complete_structured(client, request, _SameMeaning).same
        except StructuredOutputError:
            return False

    return adjudicate
