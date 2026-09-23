"""Benchmark runner (spec §11). Stages run separately so only one heavy model is on the GPU at
a time (spec §5.4); each stage writes JSONL under data/results/<run>/ and can be re-run.

1. mentions  - LLM (answer model) extracts question mentions for entity linking
2. retrieve  - embedder + reranker + graph: every system retrieves for every question
3. answer    - LLM answer model writes grounded answers from each system's context
4. judge     - judge model: claim verifier + correctness rubric
5. report    - metrics, bootstrap CIs, paired permutation tests (Holm), per-type table, MLflow
"""

import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from fixgraph.answer.grounded import GroundedAnswer, answer_question
from fixgraph.answer.verifier import VerifiedAnswer, verify_answer
from fixgraph.bench import metrics as M
from fixgraph.bench.judge import JudgeOutput, judge_answer
from fixgraph.bench.schema import Question
from fixgraph.bench.stats import bootstrap_ci, holm, paired_effect_size, paired_permutation_test
from fixgraph.llm.base import LLMClient
from fixgraph.retrieval.base import RetrievalResult, Retriever
from fixgraph.retrieval.linking import Mentions

logger = logging.getLogger(__name__)
K = 8


def _write(path: Path, rows: Sequence[BaseModel] | Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write((r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r)) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- stages -----------------------------------------------------------------------------------


def stage_mentions(
    questions: list[Question], client: LLMClient, model: str, out: Path
) -> dict[str, Mentions]:
    from fixgraph.retrieval.linking import extract_mentions

    found = {
        q.question: extract_mentions(client, q.question, model)
        for q in tqdm(questions, desc="mentions")
    }
    _write(out, [{"question": k, "mentions": v.model_dump()} for k, v in found.items()])
    return found


def load_mentions(path: Path) -> dict[str, Mentions]:
    return {r["question"]: Mentions.model_validate(r["mentions"]) for r in _read(path)}


class RetrievalRow(BaseModel):
    qid: str
    system: str
    result: RetrievalResult
    seeds: list[str] = []


def stage_retrieve(
    questions: list[Question], systems: dict[str, Retriever], out: Path, k: int = K
) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    for name, retriever in systems.items():
        for q in tqdm(questions, desc=f"retrieve {name}"):
            res = retriever.retrieve(q.question, k)
            seeds = [s.node_id for s in getattr(retriever, "last_seeds", [])]
            rows.append(RetrievalRow(qid=q.qid, system=name, result=res, seeds=seeds))
    _write(out, rows)
    return rows


class AnswerRow(BaseModel):
    qid: str
    system: str
    answer: GroundedAnswer


def stage_answer(
    questions: list[Question],
    retrievals: list[RetrievalRow],
    systems: list[str],
    chunk_text: dict[str, str],
    client: LLMClient,
    model: str,
    out: Path,
) -> list[AnswerRow]:
    by_key = {(r.qid, r.system): r for r in retrievals}
    rows: list[AnswerRow] = []
    for name in systems:
        for q in tqdm(questions, desc=f"answer {name}"):
            if name == "S0":
                ans = answer_question(client, q.question, [], chunk_text, model, closed_book=True)
            else:
                ranked = by_key[(q.qid, name)].result.chunk_ids
                ans = answer_question(client, q.question, ranked, chunk_text, model)
            rows.append(AnswerRow(qid=q.qid, system=name, answer=ans))
    _write(out, rows)
    return rows


class JudgeRow(BaseModel):
    qid: str
    system: str
    verified: VerifiedAnswer
    judge: JudgeOutput


def stage_judge(
    questions: list[Question],
    answers: list[AnswerRow],
    chunk_text: dict[str, str],
    client: LLMClient,
    model: str,
    out: Path,
) -> list[JudgeRow]:
    qmap = {q.qid: q for q in questions}
    rows: list[JudgeRow] = []
    for a in tqdm(answers, desc="judge"):
        q = qmap[a.qid]
        ver = (
            VerifiedAnswer(verdicts=[], kept=[], abstained=a.answer.abstained)
            if a.system == "S0"
            else verify_answer(client, a.answer, chunk_text, model)
        )
        jd = judge_answer(client, q, a.answer.text, a.answer.abstained, model)
        rows.append(JudgeRow(qid=a.qid, system=a.system, verified=ver, judge=jd))
    _write(out, rows)
    return rows


# --- report -------------------------------------------------------------------------------------


def per_question_metrics(
    q: Question, retrieval: RetrievalRow | None, answer: AnswerRow, judged: JudgeRow
) -> dict[str, float]:
    chunks = retrieval.result.chunk_ids if retrieval else []
    cited = answer.answer.cited_chunk_ids
    facts = judged.judge.key_facts_covered
    m: dict[str, float] = {
        "correctness": judged.judge.score,
        "key_fact_recall": (sum(facts) / len(facts)) if facts else float("nan"),
        "abstained": float(answer.answer.abstained),
        "latency_answer_s": answer.answer.latency_s,
        "latency_retrieval_s": retrieval.result.timings.get("total_s", 0.0) if retrieval else 0.0,
    }
    if q.answerable and retrieval is not None:
        m["recall@8"] = M.recall_at_k(chunks, q.gold_chunk_ids, 8)
        m["recall@4"] = M.recall_at_k(chunks, q.gold_chunk_ids, 4)
        m["support_complete@8"] = M.support_complete(chunks, q.gold_chunk_ids, 8)
    if q.answerable and answer.system != "S0" and not answer.answer.abstained:
        m["citation_precision"] = M.citation_precision(cited, q.gold_chunk_ids)
        m["citation_recall"] = M.citation_recall(cited, q.gold_chunk_ids)
        m["unsupported_rate"] = M.unsupported_rate(judged.verified.verdicts)
    if retrieval is not None:
        m["graph_fallback"] = retrieval.result.timings.get("fallback", 0.0)
    return m


def build_report(
    questions: list[Question],
    retrievals: list[RetrievalRow],
    answers: list[AnswerRow],
    judged: list[JudgeRow],
    baseline: str = "S1",
) -> dict[str, Any]:
    qmap = {q.qid: q for q in questions}
    r_by = {(r.qid, r.system): r for r in retrievals}
    a_by = {(a.qid, a.system): a for a in answers}
    per: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)  # system -> qid -> metrics
    for j in judged:
        q = qmap[j.qid]
        per[j.system][j.qid] = per_question_metrics(
            q, r_by.get((j.qid, j.system)), a_by[(j.qid, j.system)], j
        )
    systems = sorted(per)
    metric_names = sorted({k for s in per.values() for m in s.values() for k in m})
    table: dict[str, dict[str, dict[str, float]]] = {}
    for s in systems:
        table[s] = {}
        for name in metric_names:
            vals = M.nan_drop([per[s][qid].get(name, float("nan")) for qid in per[s]])
            ci = bootstrap_ci(vals)
            table[s][name] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}
        lat = [
            per[s][qid]["latency_answer_s"] + per[s][qid]["latency_retrieval_s"] for qid in per[s]
        ]
        table[s]["latency_p50_s"] = {
            "mean": M.percentile(lat, 50),
            "lo": math.nan,
            "hi": math.nan,
            "n": len(lat),
        }
        table[s]["latency_p95_s"] = {
            "mean": M.percentile(lat, 95),
            "lo": math.nan,
            "hi": math.nan,
            "n": len(lat),
        }
        abst = M.abstention_prf(
            [bool(per[s][qid]["abstained"]) for qid in per[s]],
            [not qmap[qid].answerable for qid in per[s]],
        )
        table[s]["abstention_precision"] = {
            "mean": abst["precision"],
            "lo": math.nan,
            "hi": math.nan,
            "n": abst["tp"] + abst["fp"],
        }
        table[s]["abstention_recall"] = {
            "mean": abst["recall"],
            "lo": math.nan,
            "hi": math.nan,
            "n": abst["tp"] + abst["fn"],
        }

    # Paired significance: baseline vs each other system on shared questions, Holm-corrected.
    tests: list[dict[str, Any]] = []
    for metric in ("correctness", "recall@8", "support_complete@8", "unsupported_rate"):
        rows = []
        for s in systems:
            if s == baseline or baseline not in per:
                continue
            shared = [
                qid
                for qid in per[s]
                if qid in per[baseline]
                and not math.isnan(per[s][qid].get(metric, math.nan))
                and not math.isnan(per[baseline][qid].get(metric, math.nan))
            ]
            if len(shared) < 3:
                continue
            a = [per[s][qid][metric] for qid in shared]
            b = [per[baseline][qid][metric] for qid in shared]
            rows.append(
                {
                    "metric": metric,
                    "system": s,
                    "baseline": baseline,
                    "n": len(shared),
                    "diff": sum(a) / len(a) - sum(b) / len(b),
                    "p": paired_permutation_test(a, b),
                    "effect_dz": paired_effect_size(a, b),
                }
            )
        for row, adj in zip(rows, holm([r["p"] for r in rows]), strict=True):
            row["p_holm"] = adj
        tests.extend(rows)

    by_type: dict[str, dict[str, float]] = defaultdict(dict)
    for s in systems:
        groups: dict[str, list[float]] = defaultdict(list)
        for qid, m in per[s].items():
            groups[qmap[qid].qtype].append(m["correctness"])
        for qt, vals in groups.items():
            by_type[qt][s] = sum(vals) / len(vals)
    return {
        "n_questions": len(questions),
        "types": dict(Counter(q.qtype for q in questions)),
        "table": table,
        "tests": tests,
        "correctness_by_type": dict(by_type),
        "per_question": {s: per[s] for s in systems},
    }


def render_markdown(report: dict[str, Any]) -> str:
    table = report["table"]
    systems = sorted(table)
    cols = [
        "correctness", "key_fact_recall", "recall@8", "support_complete@8", "citation_precision",
        "citation_recall", "unsupported_rate", "abstention_precision", "abstention_recall",
        "latency_p50_s", "latency_p95_s",
    ]  # fmt: skip

    def cell(s: str, c: str) -> str:
        v = table[s].get(c)
        if not v or v["mean"] is None or (isinstance(v["mean"], float) and math.isnan(v["mean"])):
            return "—"
        if math.isnan(v["lo"]):
            return f"{v['mean']:.2f}"
        return f"{v['mean']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"

    lines = [f"Questions: {report['n_questions']} ({report['types']})", ""]
    lines.append("| metric | " + " | ".join(systems) + " |")
    lines.append("|---|" + "---|" * len(systems))
    for c in cols:
        lines.append(f"| {c} | " + " | ".join(cell(s, c) for s in systems) + " |")
    lines += ["", "Paired permutation tests vs S1 (Holm-corrected):", ""]
    lines.append("| metric | system | n | diff | p | p (Holm) | d_z |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in report["tests"]:
        lines.append(
            f"| {t['metric']} | {t['system']} | {t['n']} | {t['diff']:+.3f} | {t['p']:.3f} | "
            f"{t['p_holm']:.3f} | {t['effect_dz']:+.2f} |"
        )
    lines += ["", "Correctness by question type:", ""]
    lines.append("| type | " + " | ".join(systems) + " |")
    lines.append("|---|" + "---|" * len(systems))
    for qt, vals in sorted(report["correctness_by_type"].items()):
        lines.append(
            f"| {qt} | " + " | ".join(f"{vals.get(s, float('nan')):.2f}" for s in systems) + " |"
        )
    return "\n".join(lines) + "\n"
