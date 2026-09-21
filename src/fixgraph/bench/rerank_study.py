"""Reranking study on the locked dev/test split (DECISIONS.md D39).

The miss analysis (D37) found most of S1's misses already inside its top-30 cross-encoder pool,
so this study varies only the last stage: which cross-encoder scores the pool, how large the
pool is, and whether the reranker score is fused with a second reranker or with the first-stage
(dense + BM25 RRF) score.

Pipeline: `build_cache` (one GPU pass per scorer: the top-POOL_MAX fused candidates and every
scorer's score for each of them, with scoring time per pool slice) -> `rank` (pure arithmetic
over the cache for any `StudyConfig`) -> `evaluate` (recall@8 and answer-chunk hit@8 per
subset, recovered/broken vs S1, latency estimate).

Used by: `fixgraph bench rr-cache | rr-dev | rr-test` (bench/cli.py).
Uses: bench.combo (subset + per-question metrics), bench.stats, retrieval.hybrid.
"""

import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from fixgraph.bench.combo import K, per_question, recovered_broken
from fixgraph.bench.schema import Question
from fixgraph.retrieval.hybrid import zscore

POOL_MAX = 100
SLICES = (30, 50, 100)  # scoring time is recorded per slice so any pool in SLICES is priced
SUBSETS = ("all", "main", "bridge", "bridge_direct")


class StudyRow(BaseModel):
    qid: str
    fused: list[str]  # top POOL_MAX fused (dense + BM25 RRF) candidates, best first
    rrf: list[float]  # their fusion scores
    search_s: float  # embedding + both first-stage searches + fusion
    scores: dict[str, dict[str, float]]  # scorer -> chunk -> score
    score_s: dict[str, dict[str, float]]  # scorer -> str(slice end) -> seconds for that slice


def score_slices(
    question: str,
    fused: list[str],
    docs: dict[str, str],
    score: Callable[[str, list[str]], list[float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Score `fused` slice by slice (0-30, 30-50, 50-100); return scores and per-slice time."""
    out: dict[str, float] = {}
    secs: dict[str, float] = {}
    start = 0
    for end in SLICES:
        part = fused[start:end]
        t0 = time.perf_counter()
        vals = score(question, [docs[c] for c in part]) if part else []
        secs[str(end)] = time.perf_counter() - t0
        out.update(zip(part, vals, strict=True))
        start = end
    return out, secs


class StudyConfig(BaseModel):
    """scorer: primary reranker; second: optional second reranker blended with weight w2;
    alpha: weight of the first-stage RRF score. Scores are z-normalised within the pool."""

    scorer: str = "bge"
    pool: int = 30
    second: str | None = None
    w2: float = 0.0
    alpha: float = 0.0

    @property
    def name(self) -> str:
        n = f"{self.scorer}@{self.pool}"
        if self.second:
            n += f"+{self.second}x{self.w2:g}"
        if self.alpha:
            n += f"+rrf{self.alpha:g}"
        return n


def rank(row: StudyRow, cfg: StudyConfig) -> list[str]:
    """Top K chunks for one question (pure; cached scores only). Ties keep first-stage order."""
    pool = row.fused[: cfg.pool]
    total = zscore([row.scores[cfg.scorer][c] for c in pool])
    if cfg.second:
        z2 = zscore([row.scores[cfg.second][c] for c in pool])
        total = [(1 - cfg.w2) * a + cfg.w2 * b for a, b in zip(total, z2, strict=True)]
    if cfg.alpha:
        zr = zscore(row.rrf[: cfg.pool])
        total = [a + cfg.alpha * b for a, b in zip(total, zr, strict=True)]
    order = sorted(range(len(pool)), key=lambda i: (-total[i], i))
    return [pool[i] for i in order[:K]]


def seconds(row: StudyRow, cfg: StudyConfig) -> float:
    """Estimated retrieval latency: search + scoring time of each scorer over the pool."""
    total = row.search_s
    for s in {cfg.scorer, *([cfg.second] if cfg.second else [])}:
        total += sum(v for k, v in row.score_s[s].items() if int(k) <= cfg.pool)
    return total


def grid(scorers: Sequence[str]) -> list[StudyConfig]:
    """The D39 grid: every scorer at each pool size; for the two base families, fusion with the
    first-stage score; and bge + each other scorer blended."""
    out = [StudyConfig(scorer=s, pool=p) for s in scorers for p in SLICES]
    for s in scorers:
        out += [StudyConfig(scorer=s, pool=p, alpha=a) for p in (30, 50) for a in (0.1, 0.25, 0.5)]
    for s in scorers:
        if s == "bge":
            continue
        out += [StudyConfig(scorer=s, pool=p, second="bge", w2=w)
                for p in (30, 50) for w in (0.25, 0.5)]  # fmt: skip
    return out


def _in(q: Question, subset: str) -> bool:
    bridge = q.qtype in ("bridge", "bridge_direct")
    return {"all": True, "main": not bridge}.get(subset, q.qtype == subset)


def _summary(qs: list[Question], per: dict[str, dict[str, float]]) -> dict[str, Any]:
    rec = [per[q.qid]["recall@8"] for q in qs]
    hits = [per[q.qid]["answer_hit@8"] for q in qs if "answer_hit@8" in per[q.qid]]
    return {
        "n": len(qs),
        "recall@8": round(sum(rec) / len(rec), 4) if rec else None,
        "answer_hit@8": round(sum(hits) / len(hits), 4) if hits else None,
    }


def evaluate(
    questions: list[Question],
    rows: dict[str, StudyRow],
    configs: list[StudyConfig],
    baseline: StudyConfig,
) -> list[dict[str, Any]]:
    """One log entry per config: metrics per subset, recovered/broken vs the baseline, and the
    mean / p95 estimated latency."""
    base = {q.qid: rank(rows[q.qid], baseline) for q in questions}
    log: list[dict[str, Any]] = []
    for cfg in configs:
        r = {q.qid: rank(rows[q.qid], cfg) for q in questions}
        per = per_question(questions, r)
        rb = recovered_broken(questions, base, r)
        lat = sorted(seconds(rows[q.qid], cfg) for q in questions)
        entry: dict[str, Any] = {"config": cfg.name, "spec": cfg.model_dump()}
        for sub in SUBSETS:
            entry[sub] = _summary([q for q in questions if _in(q, sub)], per)
        entry["recovered"] = sum(v["recovered"] for v in rb.values())
        entry["broken"] = sum(v["broken"] for v in rb.values())
        entry["latency_s"] = {
            "mean": round(sum(lat) / len(lat), 4),
            "p95": round(lat[max(0, int(len(lat) * 0.95) - 1)], 4),
        }
        log.append(entry)
    return log


def select(log: list[dict[str, Any]], baseline_name: str) -> dict[str, Any]:
    """D39 selection: highest all-dev recall@8; ties -> lower mean latency. The baseline wins
    unless a config strictly beats it."""
    base = next(e for e in log if e["config"] == baseline_name)
    best = max(log, key=lambda e: (e["all"]["recall@8"], -e["latency_s"]["mean"]))
    return best if best["all"]["recall@8"] > base["all"]["recall@8"] else base


def select_finalists(log: list[dict[str, Any]], baseline_name: str) -> list[dict[str, Any]]:
    """D41: the D39 pick, plus the best configuration whose mean estimated latency is within 10%
    of the baseline's (if it differs from the pick and strictly beats the baseline on dev)."""
    base = next(e for e in log if e["config"] == baseline_name)
    picks = [select(log, baseline_name)]
    cheap = [e for e in log if e["latency_s"]["mean"] <= 1.1 * base["latency_s"]["mean"]]
    best_cheap = select(cheap, baseline_name)
    if best_cheap["config"] not in {p["config"] for p in picks}:
        picks.append(best_cheap)
    return [p for p in picks if p["config"] != baseline_name]


def test_summary(
    questions: list[Question],
    ranked: dict[str, dict[str, list[str]]],
    latency: dict[str, list[float]],
    baseline_name: str,
) -> dict[str, Any]:
    """Locked-test table: recall@8 / answer-chunk hit@8 with bootstrap CIs per subset, paired
    permutation tests vs the baseline (Holm across finalists per subset and metric), recovered /
    broken gold chunks, and live latency p50 / p95 / mean."""
    from fixgraph.bench.stats import bootstrap_ci, holm, paired_permutation_test

    per = {name: per_question(questions, r) for name, r in ranked.items()}
    table: dict[str, Any] = {}
    tests: list[dict[str, Any]] = []
    for sub in SUBSETS:
        qs = [q for q in questions if _in(q, sub)]
        if not qs:
            continue
        table[sub] = {}
        for name, p in per.items():
            row: dict[str, Any] = {"n": len(qs)}
            for metric in ("recall@8", "answer_hit@8"):
                vals = [p[q.qid][metric] for q in qs if metric in p[q.qid]]
                if vals:
                    c = bootstrap_ci(vals)
                    row[metric] = {"mean": round(c.mean, 4), "lo": round(c.lo, 4),
                                   "hi": round(c.hi, 4)}  # fmt: skip
            table[sub][name] = row
        for metric in ("recall@8", "answer_hit@8"):
            keyed = [q.qid for q in qs if metric in per[baseline_name][q.qid]]
            if len(keyed) < 3:
                continue
            rows_t = []
            for name, p in per.items():
                if name == baseline_name:
                    continue
                a = [p[k][metric] for k in keyed]
                b = [per[baseline_name][k][metric] for k in keyed]
                rows_t.append({"subset": sub, "metric": metric, "system": name, "n": len(keyed),
                               "diff": round(sum(a) / len(a) - sum(b) / len(b), 4),
                               "p": round(paired_permutation_test(a, b), 4)})  # fmt: skip
            for row_t, adj in zip(rows_t, holm([r["p"] for r in rows_t]), strict=True):
                row_t["p_holm"] = round(adj, 4)
            tests += rows_t
    lat: dict[str, Any] = {}
    for name, xs in latency.items():
        s = sorted(xs)
        lat[name] = {"mean": round(sum(s) / len(s), 4), "p50": round(s[len(s) // 2], 4),
                     "p95": round(s[max(0, int(len(s) * 0.95) - 1)], 4)}  # fmt: skip
    rb = {name: recovered_broken(questions, ranked[baseline_name], r)
          for name, r in ranked.items() if name != baseline_name}  # fmt: skip
    return {"n_questions": len(questions), "baseline": baseline_name, "table": table,
            "tests_vs_baseline": tests, "recovered_broken": rb, "latency_s": lat}  # fmt: skip
