"""Graph + hybrid combinations on a locked test split (DECISIONS.md D37, D38).

Every candidate is hybrid-first and returns exactly K chunks. The candidate pool is S1's top-30
cross-encoder pool plus the chunks of articles Apple-linked (either direction) to S1's top-d
hits; the edge source is Apple's human-authored link graph, not the LLM-extracted KG.

Pipeline: `split_questions` (seeded, stratified, bridge pairs kept together) ->
`build_cache` (one GPU pass per split: fused candidates, cross-encoder scores for the S1 pool
and the link pool, and timings) -> `rank` (pure arithmetic over the cache for any config) ->
`evaluate` (recall@8, answer-chunk hit@8, recovered/broken vs S1, paired tests, CIs, latency).

Used by: `fixgraph bench combo-split | combo-cache | combo-dev | combo-test` (bench/cli.py).
Uses: retrieval.hybrid (candidates, reranker), bench.stats, bench.schema.
"""

import random
import time
from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from fixgraph.bench.schema import Question, article_of
from fixgraph.bench.stats import bootstrap_ci, holm, paired_permutation_test
from fixgraph.retrieval.hybrid import HybridRetriever

K = 8
POOL = 30  # S1 reranks the top 30 fused candidates
FUSED = 50


def split_questions(
    questions: list[Question], test_frac: float = 0.4, seed: int = 13
) -> dict[str, str]:
    """qid -> "dev" | "test". Stratified by question type; bridge pairs (b###a / b###d) are one
    unit so a pair never straddles the split."""
    units: dict[str, list[str]] = defaultdict(list)  # unit id -> qids
    unit_type: dict[str, str] = {}
    for q in questions:
        unit = q.qid[:-1] if q.qtype in ("bridge", "bridge_direct") else q.qid
        units[unit].append(q.qid)
        unit_type[unit] = "bridge_pair" if q.qtype in ("bridge", "bridge_direct") else q.qtype
    rng = random.Random(seed)
    out: dict[str, str] = {}
    by_type: dict[str, list[str]] = defaultdict(list)
    for u in sorted(units):
        by_type[unit_type[u]].append(u)
    for t in sorted(by_type):
        pool = by_type[t]
        rng.shuffle(pool)
        n_test = round(len(pool) * test_frac)
        for i, u in enumerate(pool):
            for qid in units[u]:
                out[qid] = "test" if i < n_test else "dev"
    return out


class CacheRow(BaseModel):
    qid: str
    fused: list[str]  # top FUSED fused candidates, best first
    ce: dict[str, float]  # cross-encoder score for every chunk in any pool
    s1: list[str]  # S1's top K (CE order over fused[:POOL])
    link_pool: dict[str, list[str]]  # "1" / "3" -> link-reached chunks outside fused[:POOL]
    link_reached: dict[str, list[str]]  # "1" / "3" -> every link-reached chunk
    s1_seconds: float  # fused search + CE over the S1 pool
    extra_ce_seconds: dict[str, float]  # CE time for the link pool beyond S1's


def build_cache(
    questions: list[Question],
    hybrid: HybridRetriever,
    chunk_docs: dict[str, str],
    links: list[tuple[str, str]],
    chunk_ids: list[str],
) -> list[CacheRow]:
    chunks_of: dict[str, list[str]] = defaultdict(list)
    for c in sorted(chunk_ids):
        chunks_of[article_of(c)].append(c)
    nbrs: dict[str, set[str]] = defaultdict(set)
    for a, b in links:
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    assert hybrid.reranker is not None
    rows: list[CacheRow] = []
    for q in questions:
        t0 = time.perf_counter()
        fused = [c for c, _ in hybrid.candidates_for(q.question, FUSED)]
        head = fused[:POOL]
        scores = hybrid.reranker.score(q.question, [chunk_docs[c] for c in head])
        ce = dict(zip(head, scores, strict=True))
        s1 = sorted(head, key=lambda c: -ce[c])[:K]
        s1_seconds = time.perf_counter() - t0
        reached: dict[str, list[str]] = {}
        pool: dict[str, list[str]] = {}
        extra: dict[str, float] = {}
        for d in (1, 3):
            arts = {n for c in s1[:d] for n in nbrs.get(article_of(c), ())}
            reached[str(d)] = sorted({c for a in arts for c in chunks_of[a]})
            new = [c for c in reached[str(d)] if c not in ce]
            t1 = time.perf_counter()
            if new:
                ce.update(zip(new, hybrid.reranker.score(q.question, [chunk_docs[c] for c in new]),
                              strict=True))  # fmt: skip
            extra[str(d)] = time.perf_counter() - t1
            pool[str(d)] = [c for c in reached[str(d)] if c not in head]
        # d=3's extra time excludes chunks already scored for d=1; charge d=3 the full cost
        extra["3"] += extra["1"]
        rows.append(
            CacheRow(
                qid=q.qid,
                fused=fused,
                ce=ce,
                s1=s1,
                link_pool=pool,
                link_reached=reached,
                s1_seconds=s1_seconds,
                extra_ce_seconds=extra,
            )
        )
    return rows


class Config(BaseModel):
    """One candidate. kind: s1 | prior | protected | union."""

    kind: str
    d: int = 3
    lam: float = 0.0
    routing: bool = False

    @property
    def name(self) -> str:
        if self.kind == "s1":
            return "S1"
        route = "+route" if self.routing else ""
        lam = f"-l{self.lam:g}" if self.kind == "prior" else ""
        return f"{self.kind}-d{self.d}{lam}{route}"


def grid() -> list[Config]:
    """The D37 search grid: 12 prior + 2 protected + 2 union configs."""
    out = [Config(kind="prior", d=d, lam=lam, routing=r)
           for d in (1, 3) for lam in (0.05, 0.1, 0.2) for r in (False, True)]  # fmt: skip
    out += [Config(kind="protected", d=d) for d in (1, 3)]
    out += [Config(kind="union", d=d) for d in (1, 3)]
    return out


def rank(row: CacheRow, cfg: Config, route_threshold: float) -> list[str]:
    """Top K chunks for one question under `cfg` (pure; uses only cached scores)."""
    if cfg.kind == "s1":
        return row.s1
    top_score = row.ce[row.s1[0]] if row.s1 else float("-inf")
    if cfg.routing and top_score >= route_threshold:
        return row.s1  # confident: leave S1 alone
    head = row.fused[:POOL]
    reached = set(row.link_reached[str(cfg.d)])
    pool = list(dict.fromkeys(head + row.link_pool[str(cfg.d)]))
    if cfg.kind == "protected":
        keep = row.s1[:6]
        extra = sorted((c for c in reached if c not in keep), key=lambda c: -row.ce[c])
        fill = extra + [c for c in row.s1[6:] if c not in extra]
        return (keep + fill)[:K]
    lam = cfg.lam if cfg.kind == "prior" else 0.0
    return sorted(pool, key=lambda c: -(row.ce[c] + (lam if c in reached else 0.0)))[:K]


def _answer_chunk(q: Question, by_qid: dict[str, Question]) -> str | None:
    """Gold C chunk for bridge questions (the direct twin's only gold chunk)."""
    if q.qtype == "bridge_direct":
        return q.gold_chunk_ids[0]
    if q.qtype == "bridge":
        twin = by_qid.get(q.qid[:-1] + "d")
        return twin.gold_chunk_ids[0] if twin else None
    return None


def per_question(
    questions: list[Question], ranked: dict[str, list[str]]
) -> dict[str, dict[str, float]]:
    by_qid = {q.qid: q for q in questions}
    out: dict[str, dict[str, float]] = {}
    for q in questions:
        got = set(ranked[q.qid][:K])
        m = {"recall@8": sum(c in got for c in q.gold_chunk_ids) / len(q.gold_chunk_ids)}
        c = _answer_chunk(q, by_qid)
        if c is not None:
            m["answer_hit@8"] = float(c in got)
        out[q.qid] = m
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(
    questions: list[Question], per: dict[str, dict[str, float]], subset: str
) -> dict[str, Any]:
    qs = [q for q in questions if _in_subset(q, subset)]
    rec = [per[q.qid]["recall@8"] for q in qs]
    hits = [per[q.qid]["answer_hit@8"] for q in qs if "answer_hit@8" in per[q.qid]]
    return {"n": len(qs), "recall@8": _mean(rec), "answer_hit@8": _mean(hits)}


def _in_subset(q: Question, subset: str) -> bool:
    bridge = q.qtype in ("bridge", "bridge_direct")
    return {"main": not bridge, "bridge_all": bridge, "bridge": q.qtype == "bridge",
            "bridge_direct": q.qtype == "bridge_direct"}.get(subset, q.qtype == subset)  # fmt: skip


def recovered_broken(
    questions: list[Question], s1: dict[str, list[str]], combo: dict[str, list[str]]
) -> dict[str, dict[str, int]]:
    """Per question type, gold chunks S1 missed that the combo finds (recovered) and gold
    chunks S1 found that the combo drops (broken)."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"recovered": 0, "broken": 0})
    for q in questions:
        a, b = set(s1[q.qid][:K]), set(combo[q.qid][:K])
        for c in q.gold_chunk_ids:
            if c not in a and c in b:
                out[q.qtype]["recovered"] += 1
            if c in a and c not in b:
                out[q.qtype]["broken"] += 1
    return dict(out)


def test_report(
    questions: list[Question],
    rows: dict[str, CacheRow],
    finalists: list[Config],
    route_threshold: float,
) -> dict[str, Any]:
    """The single locked-test evaluation (D38): S1 vs each finalist."""
    s1_rank = {q.qid: rank(rows[q.qid], Config(kind="s1"), route_threshold) for q in questions}
    s1_per = per_question(questions, s1_rank)
    systems: dict[str, Any] = {}
    tests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cfg in finalists:
        r = {q.qid: rank(rows[q.qid], cfg, route_threshold) for q in questions}
        per = per_question(questions, r)
        extra = [rows[q.qid].extra_ce_seconds[str(cfg.d)] for q in questions]
        if cfg.routing:
            extra = [
                0.0 if rows[q.qid].ce[rows[q.qid].s1[0]] >= route_threshold else e
                for q, e in zip(questions, extra, strict=True)
            ]
        systems[cfg.name] = {
            "config": cfg.model_dump(),
            "per_question": per,
            "recovered_broken": recovered_broken(questions, s1_rank, r),
            "extra_latency_s": {"mean": _mean(extra), "p95": _p95(extra)},
        }
    subsets = ["main", *sorted({q.qtype for q in questions if not _in_subset(q, "bridge_all")}),
               "bridge", "bridge_direct"]  # fmt: skip
    table: dict[str, dict[str, Any]] = {}
    for sub in subsets:
        qs = [q for q in questions if _in_subset(q, sub)]
        if not qs:
            continue
        table[sub] = {"S1": _metric_ci(qs, s1_per)}
        for name, sysd in systems.items():
            table[sub][name] = _metric_ci(qs, sysd["per_question"])
        for metric in ("recall@8", "answer_hit@8"):
            keyed = [q.qid for q in qs if metric in s1_per[q.qid]]
            if len(keyed) < 3:
                continue
            rows_t = []
            for name, sysd in systems.items():
                a = [sysd["per_question"][k][metric] for k in keyed]
                b = [s1_per[k][metric] for k in keyed]
                rows_t.append({"subset": sub, "metric": metric, "system": name, "n": len(keyed),
                               "diff": round(sum(a) / len(a) - sum(b) / len(b), 4),
                               "p": round(paired_permutation_test(a, b), 4)})  # fmt: skip
            for row, adj in zip(rows_t, holm([r["p"] for r in rows_t]), strict=True):
                row["p_holm"] = round(adj, 4)
            tests[sub] += rows_t
    s1_lat = [rows[q.qid].s1_seconds for q in questions]
    return {
        "n_questions": len(questions),
        "route_threshold": route_threshold,
        "table": table,
        "tests_vs_S1": dict(tests),
        "recovered_broken": {n: s["recovered_broken"] for n, s in systems.items()},
        "latency_s": {
            "S1": {"mean": _mean(s1_lat), "p95": _p95(s1_lat)},
            **{n: {"extra_over_S1": s["extra_latency_s"]} for n, s in systems.items()},
        },
        "note": "edge source: Apple's human-authored article links, not the LLM-extracted KG; "
        "bridge rows are non-independent (pairs were built from those links, D37).",
    }


def _metric_ci(qs: list[Question], per: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(qs)}
    for metric in ("recall@8", "answer_hit@8"):
        vals = [per[q.qid][metric] for q in qs if metric in per[q.qid]]
        if vals:
            c = bootstrap_ci(vals)
            out[metric] = {"mean": round(c.mean, 4), "lo": round(c.lo, 4), "hi": round(c.hi, 4)}
    return out


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[max(0, int(len(s) * 0.95) - 1)], 4)
