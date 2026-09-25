"""Edge verification (DECISIONS.md D32): keep an extracted edge only if a source chunk states it.

Validation (kg.validation) only checks that each entity string occurs in the chunk, so an edge
between two co-occurring strings passes even when the chunk never relates them. Here every
extracted edge becomes a plain-language statement built from the surface forms the chunk itself
used, and the judge-class model decides, per source chunk, whether the passage states it and
quotes the sentence that does. Code then checks that the quote really occurs in the passage the
model saw (title, heading and chunk text; fuzzy partial ratio >= QUOTE_THRESHOLD), as in the
benchmark pre-screen (D29). An edge is kept if at least one source chunk supports it; its
provenance is narrowed to the supporting chunks.

Measurement (`edge_sample`, `evaluate_edges`): a stratified random sample of edges is labelled
blind to the verifier; the hallucinated-edge rate is estimated for all edges (before) and for
kept edges (after), with stratified bootstrap CIs, plus the verifier's precision/recall.

Used by: `fixgraph kg verify-edges | edge-sample | edge-eval` (kg/cli.py).
Uses: kg.store (KG), kg.validation (partial_ratio), llm.structured.
"""

import random
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, Field
from tqdm import tqdm

from fixgraph.kg.store import KG
from fixgraph.kg.validation import partial_ratio
from fixgraph.llm.base import ChatMessage, LLMClient, LLMRequest
from fixgraph.llm.structured import StructuredOutputError, complete_structured

QUOTE_THRESHOLD = 0.85
MAX_CLAIMS_PER_CALL = 12

# One statement per relation type; {h} / {t} are the chunk's own surface forms.
TEMPLATES: dict[str, str] = {
    "RESOLVED_BY": 'One way to fix the problem "{h}" is: "{t}".',
    "CAUSED_BY": 'The problem "{h}" can be caused by: "{t}".',
    "ADDRESSES": 'The fix "{h}" deals with the cause "{t}".',
    "EXHIBITS": '{h} can have the problem "{t}".',
    "INVOLVES": 'The problem "{h}" involves {t}.',
    "HAS_COMPONENT": "{h} has the part or component {t}.",
    "REQUIRES": 'The fix "{h}" requires {t}.',
    "APPLIES_TO": '"{h}" applies to {t}.',
    "RUNS": "{h} runs {t}.",
    "DEPENDS_ON": "{h} needs {t} in order to work.",
    "SIGNALS": '{h} indicates the problem "{t}".',
}

SYSTEM_VERIFY = """You check facts extracted from an Apple support passage.
For each numbered statement decide whether the passage itself states it (paraphrase is fine).
Mark it NOT supported when:
- the two parts are only mentioned near each other but the passage does not connect them,
- it needs outside knowledge or a guess,
- the passage says something different (another condition, device, version or problem).
If supported, copy the passage sentence that states it, exactly as written."""


class EdgeClaim(BaseModel):
    edge_id: str
    rel: str
    chunk_id: str
    statement: str


class EdgeVerdict(BaseModel):
    edge_id: str
    chunk_id: str
    llm_supported: bool
    quote: str = ""
    quote_score: float = 0.0
    supported: bool  # llm_supported and the quote occurs in the chunk
    error: str | None = None


class _Item(BaseModel):
    id: int
    supported: bool
    quote: str = ""


class _Reply(BaseModel):
    verdicts: list[_Item] = Field(default_factory=lambda: list[_Item]())


def verifiable_edges(kg: KG) -> pl.DataFrame:
    """LLM-extracted edges (ontology-derived IN_FAMILY / seed DEPENDS_ON are not text claims)."""
    return kg.extracted_edges().filter(pl.col("extraction_model") != "ontology")


def edge_claims(kg: KG) -> dict[str, list[EdgeClaim]]:
    """chunk_id -> statements for every edge that chunk is cited for. Each endpoint is phrased
    with the surface form that chunk used (falls back to the canonical text), so a merged node's
    medoid text from another article does not leak into the check."""
    text = dict(zip(kg.nodes["node_id"], kg.nodes["canonical_text"], strict=True))
    surface: dict[tuple[str, str], str] = {}
    for nid, cid, s in kg.mentions.select("node_id", "chunk_id", "surface").iter_rows():
        surface.setdefault((nid, cid), s)
    out: dict[str, list[EdgeClaim]] = defaultdict(list)
    for r in verifiable_edges(kg).iter_rows(named=True):
        for cid in r["source_chunk_ids"]:
            h = surface.get((r["src"], cid), text[r["src"]])
            t = surface.get((r["dst"], cid), text[r["dst"]])
            out[cid].append(
                EdgeClaim(
                    edge_id=r["edge_id"],
                    rel=r["rel"],
                    chunk_id=cid,
                    statement=TEMPLATES[r["rel"]].format(h=h, t=t),
                )
            )
    return dict(out)


def verify_request(passage: str, claims: list[EdgeClaim], model: str) -> LLMRequest:
    numbered = "\n".join(f"{i}. {c.statement}" for i, c in enumerate(claims))
    return LLMRequest(
        model=model,
        messages=[
            ChatMessage(role="system", content=SYSTEM_VERIFY),
            ChatMessage(
                role="user",
                content=f"Passage:\n{passage}\n\nStatements:\n{numbered}\n\n"
                f"Return one verdict per statement id (0 to {len(claims) - 1}).",
            ),
        ],
        temperature=0.0,
        num_ctx=4096,
        max_tokens=150 + 80 * len(claims),
        think=False,
    )


def verify_chunk(
    client: LLMClient, passage: str, claims: list[EdgeClaim], model: str
) -> list[EdgeVerdict]:
    """Verdicts for one chunk's claims, in batches. The quote is checked against the passage
    the model saw (title + heading + chunk text), as validation grounds entities against the
    same context. A failed call marks its claims unsupported with the error recorded (fail
    closed: an edge nobody could confirm is not kept)."""
    out: list[EdgeVerdict] = []
    for i in range(0, len(claims), MAX_CLAIMS_PER_CALL):
        batch = claims[i : i + MAX_CLAIMS_PER_CALL]
        try:
            reply = complete_structured(client, verify_request(passage, batch, model), _Reply)
            items = {v.id: v for v in reply.verdicts}
            error = None
        except (StructuredOutputError, RuntimeError) as exc:
            items, error = {}, str(exc)[:200]
        for j, c in enumerate(batch):
            v = items.get(j)
            llm_ok = bool(v and v.supported)
            quote = v.quote.strip() if v else ""
            score = partial_ratio(quote, passage) if quote else 0.0
            out.append(
                EdgeVerdict(
                    edge_id=c.edge_id,
                    chunk_id=c.chunk_id,
                    llm_supported=llm_ok,
                    quote=quote,
                    quote_score=round(score, 3),
                    supported=llm_ok and score >= QUOTE_THRESHOLD,
                    error=error if v is None else None,
                )
            )
    return out


def run_verification(
    client: LLMClient,
    claims: dict[str, list[EdgeClaim]],
    passages: dict[str, str],
    model: str,
    concurrency: int = 2,
) -> list[EdgeVerdict]:
    """All chunks, `concurrency` requests at a time; order of the output is deterministic."""
    order = sorted(claims)

    def one(cid: str) -> list[EdgeVerdict]:
        return verify_chunk(client, passages[cid], claims[cid], model)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(tqdm(pool.map(one, order), total=len(order), desc="verify"))
    return [v for chunk_verdicts in results for v in chunk_verdicts]


def supported_edges(verdicts: Iterable[EdgeVerdict]) -> dict[str, list[str]]:
    """edge_id -> sorted supporting chunk ids (edges with no support are absent)."""
    out: dict[str, set[str]] = defaultdict(set)
    for v in verdicts:
        if v.supported:
            out[v.edge_id].add(v.chunk_id)
    return {e: sorted(c) for e, c in out.items()}


def verified_kg(kg: KG, verdicts: Iterable[EdgeVerdict]) -> KG:
    """Keep supported extracted edges (provenance narrowed to supporting chunks) and the
    ontology IN_FAMILY edges; drop unsupported edges and the text-free seed DEPENDS_ON edges.
    Nodes and mentions are unchanged (a node without edges still maps to its chunks)."""
    keep = supported_edges(verdicts)
    edges = kg.edges
    ontology_family = edges.filter(
        (pl.col("extraction_model") == "ontology") & (pl.col("rel") == "IN_FAMILY")
    )
    support = pl.DataFrame(
        {"edge_id": list(keep), "supporting": list(keep.values())},
        schema={"edge_id": pl.String, "supporting": pl.List(pl.String)},
    )
    extracted = (
        verifiable_edges(kg)
        .join(support, on="edge_id", how="inner")
        .with_columns(
            pl.col("supporting").alias("source_chunk_ids"),
            pl.col("supporting").list.len().cast(pl.Int64).alias("n_support"),
        )
        .select(kg.edges.columns)
    )
    others = edges.filter(pl.col("origin") != "extracted")
    new_edges = pl.concat([ontology_family, extracted, others]).sort("edge_id")
    return KG(nodes=kg.nodes, edges=new_edges, mentions=kg.mentions)


def write_verdicts(verdicts: list[EdgeVerdict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(v.model_dump_json() + "\n" for v in verdicts), encoding="utf-8")


def read_verdicts(path: Path) -> list[EdgeVerdict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [EdgeVerdict.model_validate_json(x) for x in lines if x.strip()]


# ---------------------------------------------------------------------------
# Measurement: stratified sample, blind labels, before/after hallucination rate
# ---------------------------------------------------------------------------


class SampledEdge(BaseModel):
    edge_id: str
    rel: str
    chunk_ids: list[str]
    statements: list[str]  # one per chunk, same order


class EdgeLabel(BaseModel):
    edge_id: str
    supported: bool  # at least one of the edge's source chunks states it
    labeler: str


def allocate(sizes: dict[str, int], n: int, min_per: int) -> dict[str, int]:
    """Proportional allocation with a floor of `min_per` (capped at the stratum size)."""
    total = sum(sizes.values())
    alloc = {k: min(v, max(min_per, round(n * v / total))) for k, v in sizes.items()}
    return alloc


def edge_sample(kg: KG, n: int = 200, min_per: int = 5, seed: int = 13) -> list[SampledEdge]:
    claims = [c for cs in edge_claims(kg).values() for c in cs]
    by_edge: dict[str, list[EdgeClaim]] = defaultdict(list)
    for c in claims:
        by_edge[c.edge_id].append(c)
    strata: dict[str, list[str]] = defaultdict(list)
    for eid, cs in sorted(by_edge.items()):
        strata[cs[0].rel].append(eid)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n, min_per)
    rng = random.Random(seed)
    out: list[SampledEdge] = []
    for rel in sorted(strata):
        for eid in sorted(rng.sample(strata[rel], alloc[rel])):
            cs = sorted(by_edge[eid], key=lambda c: c.chunk_id)
            out.append(
                SampledEdge(
                    edge_id=eid,
                    rel=rel,
                    chunk_ids=[c.chunk_id for c in cs],
                    statements=[c.statement for c in cs],
                )
            )
    return out


def _rate(
    strata: dict[str, list[tuple[bool, bool]]], sizes: dict[str, int], kept_only: bool
) -> float:
    """Population-weighted share of unsupported edges (among kept edges if kept_only).
    Each item is (label_supported, verifier_kept)."""
    num = den = 0.0
    for rel, items in strata.items():
        if not items:
            continue
        pool = [x for x in items if x[1]] if kept_only else items
        weight = sizes[rel] / len(items)  # edges each sampled item stands for
        den += weight * len(pool)
        num += weight * sum(not lab for lab, _ in pool)
    return num / den if den else float("nan")


def evaluate_edges(
    sample: list[SampledEdge],
    labels: list[EdgeLabel],
    verdicts: list[EdgeVerdict],
    stratum_sizes: dict[str, int],
    n_kept_by_rel: dict[str, int],
    n_boot: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    lab = {x.edge_id: x.supported for x in labels}
    kept = set(supported_edges(verdicts))
    strata: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for s in sample:
        if s.edge_id in lab:
            strata[s.rel].append((lab[s.edge_id], s.edge_id in kept))

    def ci(kept_only: bool) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            resampled = {
                r: [items[i] for i in rng.integers(0, len(items), len(items))]
                for r, items in strata.items()
            }
            boots.append(_rate(resampled, stratum_sizes, kept_only))
        arr = np.asarray([b for b in boots if b == b])
        if arr.size == 0:  # nothing kept (or nothing labelled) in this subset
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
        lo, hi = np.quantile(arr, [0.025, 0.975])
        return {
            "mean": round(_rate(strata, stratum_sizes, kept_only), 3),
            "lo": round(float(lo), 3),
            "hi": round(float(hi), 3),
        }

    items = [x for v in strata.values() for x in v]
    tp = sum(lab_ and k for lab_, k in items)
    fp = sum((not lab_) and k for lab_, k in items)
    fn = sum(lab_ and not k for lab_, k in items)
    tn = sum((not lab_) and not k for lab_, k in items)
    by_rel = {
        rel: {
            "n_sampled": len(v),
            "labelled_unsupported": sum(not a for a, _ in v),
            "edges_total": stratum_sizes[rel],
            "edges_kept": n_kept_by_rel.get(rel, 0),
        }
        for rel, v in sorted(strata.items())
    }
    return {
        "n_labelled": len(items),
        "hallucinated_rate_before": ci(kept_only=False),
        "hallucinated_rate_after": ci(kept_only=True),
        "verifier_vs_labels": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision_keep": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall_keep": round(tp / (tp + fn), 3) if tp + fn else None,
            "reject_precision": round(tn / (tn + fn), 3) if tn + fn else None,
        },
        "edges_total": sum(stratum_sizes.values()),
        "edges_kept": sum(n_kept_by_rel.values()),
        "by_relation": by_rel,
        "note": "rates are population-weighted over relation strata (stratified sample, seed "
        f"{seed}); CIs are stratified bootstrap percentiles ({n_boot} resamples).",
    }


def write_jsonl(rows: Iterable[BaseModel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(x) for x in lines if x.strip()]


def stratum_sizes(kg: KG) -> dict[str, int]:
    counts = verifiable_edges(kg).group_by("rel").len()
    return dict(zip(counts["rel"], counts["len"], strict=True))
