"""Build the parquet KG from validated extractions (spec §7, §8.2).

Every edge carries provenance: source_chunk_ids, extraction_model, extraction_confidence (max
over supporting mentions), extracted_at (earliest), origin. Ontology-derived edges (IN_FAMILY
and seed DEPENDS_ON) are marked with extraction_model="ontology".

Used by: `fixgraph kg build` (kg/cli.py), which writes the result with kg.store.write_kg.
Uses: kg.run_extract (ExtractionRecord input), kg.resolve (canonicalization), kg.store
(schemas), core.ontology, embeddings.
"""

# Imports: polars for the output tables, resolve for merging duplicate names, store for the schemas.
import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import polars as pl

from fixgraph.core.models import Article
from fixgraph.core.ontology import Ontology
from fixgraph.embeddings import Embedder
from fixgraph.kg.resolve import (
    CLUSTERED_TYPES,
    DEFAULT_THRESHOLDS,
    Adjudicator,
    Canonical,
    MergeRecord,
    canonical_rule,
    clustering_key,
    node_id,
    resolve_clustered,
)
from fixgraph.kg.run_extract import ExtractionRecord
from fixgraph.kg.store import EDGE_SCHEMA, KG, MENTION_SCHEMA, NODE_SCHEMA

logger = logging.getLogger(__name__)

# Entities that fuzzy-match the chunk below this score are not trusted enough to become nodes.
MIN_ENTITY_GROUNDING = 0.8


# Running totals for one edge while merging: which chunks support it, best confidence, models.
@dataclass
class _EdgeAcc:
    chunks: set[str] = field(default_factory=lambda: set[str]())
    conf: float = 0.0
    extracted_at: str = "9999"
    models: set[str] = field(default_factory=lambda: set[str]())


# Summary returned with the graph: how many records failed, why mentions were dropped, merges made.
@dataclass
class BuildReport:
    n_records: int = 0
    n_failed_records: int = 0
    dropped_mentions: Counter[str] = field(default_factory=lambda: Counter[str]())
    merges: list[MergeRecord] = field(default_factory=lambda: list[MergeRecord]())


# Turn validated extraction records plus the article list into the KG (nodes, edges, mentions).
# Called by `fixgraph kg build`; the caller writes the result to parquet.
def build_kg(
    records: list[ExtractionRecord],
    articles: list[Article],
    ontology: Ontology,
    embedder: Embedder,
    thresholds: dict[str, float] | None = None,
    adjudicate: Adjudicator | None = None,
    canonicalize: bool = True,
) -> tuple[KG, BuildReport]:
    """`canonicalize=False` is the "without canonicalization" ablation (spec §11.5): every
    distinct normalized surface string becomes its own node."""
    # Use only records that succeeded and have a validated extraction.
    thresholds = thresholds or DEFAULT_THRESHOLDS
    report = BuildReport(n_records=len(records))
    ok = [r for r in records if r.ok and r.validated is not None]
    report.n_failed_records = len(records) - len(ok)

    # --- 1. rule canonicalization + collect clustered texts --------------------------------
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)  # key -> surface forms
    cluster_counts: dict[str, Counter[str]] = defaultdict(Counter)
    # Count every grounded surface form of the free-text types (fixes, causes, ...) for clustering.
    for r in ok:
        assert r.validated is not None
        for e in r.validated.entities:
            if e.type in CLUSTERED_TYPES and e.grounding >= MIN_ENTITY_GROUNDING:
                key = clustering_key(e.text)
                if key:
                    cluster_counts[e.type][key] += 1
                    surface_counts[f"{e.type}|{key}"][e.text] += 1

    # Cluster each free-text type by embedding so near-duplicate phrases share one canonical key;
    # the no-canonicalization ablation keeps every key as its own node.
    canonical_of: dict[str, dict[str, str]] = {}
    for type_ in CLUSTERED_TYPES:
        counts = cluster_counts.get(type_, Counter())
        if not canonicalize:
            canonical_of[type_] = {k: k for k in counts}
            continue
        res = resolve_clustered(type_, counts, embedder, thresholds[type_], adjudicate)
        canonical_of[type_] = res.canonical_of
        report.merges.extend(res.merges)
        logger.info(
            "%s: %d distinct -> %d canonical",
            type_,
            len(counts),
            len(set(res.canonical_of.values())),
        )

    # Show a cluster as its most common original spelling rather than the normalized key.
    def display(type_: str, key: str) -> str:
        forms = surface_counts.get(f"{type_}|{key}")
        return forms.most_common(1)[0][0] if forms else key

    # Map one mention to its canonical node: cluster lookup for free text, ontology rules otherwise.
    def canon(type_: str, text: str) -> Canonical | None:
        if type_ in CLUSTERED_TYPES:
            key = clustering_key(text)
            target = canonical_of[type_].get(key)
            if target is None:
                return None
            return Canonical(label=type_, text=display(type_, target))
        if not canonicalize:
            return Canonical(label=type_, text=text.strip())
        return canonical_rule(type_, text, ontology)

    # --- 2. nodes, mentions, edges ---------------------------------------------------------
    # Accumulators: nodes by id, mention triples, and edges keyed by (src, rel, dst).
    nodes: dict[str, tuple[str, str, dict[str, object]]] = {}
    mentions: set[tuple[str, str, str]] = set()
    edges: dict[tuple[str, str, str], _EdgeAcc] = defaultdict(_EdgeAcc)
    rule_merges: dict[tuple[str, str], set[str]] = defaultdict(set)

    # Create the node the first time its id is seen; node ids come from label + canonical text.
    def add_node(c: Canonical) -> str:
        nid = node_id(c.label, c.text)
        if nid not in nodes:
            nodes[nid] = (c.label, c.text, dict(c.props))
        return nid

    # For each chunk, map entity indexes to node ids, recording a mention per kept entity.
    for r in ok:
        assert r.validated is not None
        local: dict[int, str] = {}
        for i, e in enumerate(r.validated.entities):
            if e.grounding < MIN_ENTITY_GROUNDING:
                report.dropped_mentions[f"{e.type}:ungrounded"] += 1
                continue
            c = canon(e.type, e.text)
            if c is None:
                report.dropped_mentions[f"{e.type}:no_canonical"] += 1
                continue
            nid = local[i] = add_node(c)
            mentions.add((nid, r.chunk_id, e.text))
            if e.type not in CLUSTERED_TYPES and e.text.strip() != c.text:
                rule_merges[(c.label, c.text)].add(e.text.strip())
        # Then turn each relation into an edge, merging repeats across chunks with provenance.
        for rel in r.validated.relations:
            src, dst = local.get(rel.head), local.get(rel.tail)
            if src is None or dst is None or src == dst:
                continue
            acc = edges[(src, rel.rel, dst)]
            acc.chunks.add(r.chunk_id)
            acc.conf = max(acc.conf, rel.conf)
            acc.extracted_at = min(acc.extracted_at, r.extracted_at)
            acc.models.add(r.model)

    # Log rule-based renames (e.g. spelling variants) as merge records for the report.
    for (label, text), members in sorted(rule_merges.items()):
        report.merges.append(
            MergeRecord(type=label, canonical=text, members=sorted(members), method="rule")
        )

    # --- 3. ontology-derived structure: families, articles, seed dependencies --------------
    # Which chunks mention each node, so family edges can inherit provenance from the product.
    product_chunks: dict[str, set[str]] = defaultdict(set)
    for nid, chunk_id, _ in mentions:
        product_chunks[nid].add(chunk_id)
    # Link each product to its product family from the ontology with an IN_FAMILY edge.
    for nid, (label, _text, props) in list(nodes.items()):
        family = props.get("family") if label == "Product" else None
        if isinstance(family, str) and family:
            fid = add_node(Canonical(label="ProductFamily", text=family, props={"name": family}))
            acc = edges[(nid, "IN_FAMILY", fid)]
            acc.chunks |= product_chunks[nid]
            acc.conf = 1.0
            acc.models.add("ontology")
    # Add known product dependencies from the ontology when both products are already in the graph.
    generic = {text: nid for nid, (label, text, _) in nodes.items() if label == "Product"}
    for dep in ontology.dependencies:
        if dep.child in generic and dep.parent in generic:
            acc = edges[(generic[dep.child], "DEPENDS_ON", generic[dep.parent])]
            acc.conf = max(acc.conf, 1.0)
            acc.models.add("ontology")
    # Add one Article node per article so the graph can point back to its sources.
    for a in articles:
        add_node(
            Canonical(
                label="Article",
                text=a.title,
                props={
                    "article_id": a.article_id,
                    "url": a.url,
                    "last_updated": a.last_updated or "",
                },
            )
        )

    # --- 4. frames ---------------------------------------------------------------------------
    # Turn the dicts into rows in sorted order so the output is deterministic.
    node_rows = [
        {
            "node_id": nid,
            "label": lab,
            "canonical_text": text,
            "props_json": json.dumps(props, ensure_ascii=False),
        }
        for nid, (lab, text, props) in sorted(nodes.items())
    ]
    # Edge id is a short hash of (src, rel, dst), so an edge keeps the same id across rebuilds.
    edge_rows = []
    for (src, rel, dst), acc in sorted(edges.items()):
        edge_rows.append(
            {
                "edge_id": hashlib.sha1(f"{src}|{rel}|{dst}".encode()).hexdigest()[:16],
                "src": src,
                "rel": rel,
                "dst": dst,
                "source_chunk_ids": sorted(acc.chunks),
                "extraction_model": ",".join(sorted(acc.models)),
                "extraction_confidence": round(acc.conf, 3),
                "extracted_at": "" if acc.extracted_at == "9999" else acc.extracted_at,
                "origin": "extracted",
                "n_support": len(acc.chunks),
            }
        )
    mention_rows = [{"node_id": n, "chunk_id": c, "surface": s} for n, c, s in sorted(mentions)]
    # Wrap the three tables with the fixed schemas from kg.store.
    kg = KG(
        nodes=pl.DataFrame(node_rows, schema=NODE_SCHEMA),
        edges=pl.DataFrame(edge_rows, schema=EDGE_SCHEMA),
        mentions=pl.DataFrame(mention_rows, schema=MENTION_SCHEMA),
    )
    return kg, report
