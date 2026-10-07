"""Benchmark runner (spec §11). Stages run separately so only one heavy model is on the GPU at
a time (spec §5.4); each stage writes JSONL under data/results/<run>/ and can be re-run.

1. mentions  - LLM (answer model) extracts question mentions for entity linking
2. retrieve  - embedder + reranker + graph: every system retrieves for every question
3. answer    - LLM answer model writes grounded answers from each system's context
4. judge     - judge model: claim verifier + correctness rubric
5. report    - metrics, bootstrap CIs, paired permutation tests (Holm), per-type table, MLflow

Used by: `fixgraph bench run | report` (bench/cli.py calls the stage_* functions and
build_report / render_markdown; bench/cli.py builds the retrievers and logs to MLflow).
Uses: retrieval.linking (mentions), answer.grounded, answer.verifier, bench.judge,
bench.metrics, bench.stats.
"""

# Imports: answer generation, claim verifier, judge, metrics and stats used by the stages below.
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
from fixgraph.bench.judge import JudgeExample, JudgeOutput, JudgeVariant, judge_answer
from fixgraph.bench.schema import Question
from fixgraph.bench.stats import bootstrap_ci, holm, paired_effect_size, paired_permutation_test
from fixgraph.llm.base import LLMClient
from fixgraph.retrieval.base import RetrievalResult, Retriever
from fixgraph.retrieval.linking import Mentions

# K = how many chunks each retriever hands to the answer model.
logger = logging.getLogger(__name__)
K = 8


# Write a list of rows (pydantic models or dicts) as JSONL, one per line.
def _write(path: Path, rows: Sequence[BaseModel] | Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write((r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r)) + "\n")


# Read a JSONL file back into a list of dicts.
def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- stages -----------------------------------------------------------------------------------


# Stage 1: the LLM extracts entity mentions from each question; saved for graph seed linking.
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


# Reload saved mentions so later runs skip the LLM extraction step.
def load_mentions(path: Path) -> dict[str, Mentions]:
    return {r["question"]: Mentions.model_validate(r["mentions"]) for r in _read(path)}


# One retrieval result: which system, which question, ranked chunks and graph seed nodes.
class RetrievalRow(BaseModel):
    qid: str
    system: str
    result: RetrievalResult
    seeds: list[str] = []


# Stage 2: every given system retrieves top-k chunks for every question; saved to JSONL.
def stage_retrieve(
    questions: list[Question], systems: dict[str, Retriever], out: Path, k: int = K
) -> list[RetrievalRow]:
    rows: list[RetrievalRow] = []
    # Loop systems x questions; graph retrievers also expose the seed nodes they started from.
    for name, retriever in systems.items():
        for q in tqdm(questions, desc=f"retrieve {name}"):
            res = retriever.retrieve(q.question, k)
            seeds = [s.node_id for s in getattr(retriever, "last_seeds", [])]
            rows.append(RetrievalRow(qid=q.qid, system=name, result=res, seeds=seeds))
    _write(out, rows)
    return rows


# One generated answer for one (question, system) pair.
class AnswerRow(BaseModel):
    qid: str
    system: str
    answer: GroundedAnswer


# Stage 3: the answer model writes a grounded answer from each system's retrieved chunks.
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
    # S0 is the closed-book baseline (no context); other systems answer from their ranked chunks.
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


# One judged answer: claim-level verifier verdicts plus the correctness judge's score.
class JudgeRow(BaseModel):
    qid: str
    system: str
    verified: VerifiedAnswer
    judge: JudgeOutput


# Stage 4: verify claims against cited chunks and grade correctness with the judge model.
def stage_judge(
    questions: list[Question],
    answers: list[AnswerRow],
    chunk_text: dict[str, str],
    client: LLMClient,
    model: str,
    out: Path,
    variant: JudgeVariant = "v1",
    examples: Sequence[JudgeExample] = (),
) -> list[JudgeRow]:
    """Claim verifier + correctness judge. Writes `judge_meta.json` next to `out` so the report
    states which judge prompt produced the scores."""
    qmap = {q.qid: q for q in questions}
    rows: list[JudgeRow] = []
    # S0 has no citations to verify, so it gets an empty verdict list.
    for a in tqdm(answers, desc="judge"):
        q = qmap[a.qid]
        ver = (
            VerifiedAnswer(verdicts=[], kept=[], abstained=a.answer.abstained)
            if a.system == "S0"
            else verify_answer(client, a.answer, chunk_text, model)
        )
        jd = judge_answer(client, q, a.answer.text, a.answer.abstained, model, variant, examples)
        rows.append(JudgeRow(qid=a.qid, system=a.system, verified=ver, judge=jd))
    _write(out, rows)
    # Record which judge model and prompt produced these scores, for the report.
    meta = {"model": model, "variant": variant, "n_examples": len(examples)}
    (out.parent / "judge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return rows


# --- report -------------------------------------------------------------------------------------


# Turn one judged answer into a dict of metrics: correctness, recall, citations, latency.
def per_question_metrics(
    q: Question, retrieval: RetrievalRow | None, answer: AnswerRow, judged: JudgeRow
) -> dict[str, float]:
    chunks = retrieval.result.chunk_ids if retrieval else []
    cited = answer.answer.cited_chunk_ids
    facts = judged.judge.key_facts_covered
    # Metrics that apply to every question.
    m: dict[str, float] = {
        "correctness": judged.judge.score,
        "key_fact_recall": (sum(facts) / len(facts)) if facts else float("nan"),
        "abstained": float(answer.answer.abstained),
        "latency_answer_s": answer.answer.latency_s,
        "latency_retrieval_s": retrieval.result.timings.get("total_s", 0.0) if retrieval else 0.0,
    }
    # Retrieval metrics only for answerable questions that had a retrieval step.
    if q.answerable and retrieval is not None:
        m["recall@8"] = M.recall_at_k(chunks, q.gold_chunk_ids, 8)
        m["recall@4"] = M.recall_at_k(chunks, q.gold_chunk_ids, 4)
        m["support_complete@8"] = M.support_complete(chunks, q.gold_chunk_ids, 8)
    # Citation and hallucination metrics only when the system actually answered with context.
    if q.answerable and answer.system != "S0" and not answer.answer.abstained:
        m["citation_precision"] = M.citation_precision(cited, q.gold_chunk_ids)
        m["citation_recall"] = M.citation_recall(cited, q.gold_chunk_ids)
        m["unsupported_rate"] = M.unsupported_rate(judged.verified.verdicts)
    # Whether the graph retriever fell back to plain retrieval for this question.
    if retrieval is not None:
        m["graph_fallback"] = retrieval.result.timings.get("fallback", 0.0)
    return m


# Metrics we run significance tests on, comparing each system to the baseline.
TEST_METRICS = ("correctness", "recall@8", "support_complete@8", "unsupported_rate")


# Paired tests of each system vs baseline on the same questions, Holm-corrected per metric.
def _paired_tests(
    per: dict[str, dict[str, dict[str, float]]],
    systems: list[str],
    baseline: str,
    metrics: Sequence[str],
    qids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Paired permutation test of each system vs `baseline` on shared questions (optionally
    restricted to `qids`), Holm-corrected across systems within each metric."""
    tests: list[dict[str, Any]] = []
    if baseline not in per:
        return tests
    # For each metric, test every system on the questions both it and the baseline have values for.
    for metric in metrics:
        rows = []
        for s in systems:
            if s == baseline:
                continue
            shared = [
                qid
                for qid in per[s]
                if (qids is None or qid in qids)
                and qid in per[baseline]
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
        # Holm-correct within the metric across systems.
        for row, adj in zip(rows, holm([r["p"] for r in rows]), strict=True):
            row["p_holm"] = adj
        tests.extend(rows)
    return tests


# Split results into single-article vs multi-article questions: the key graph-vs-RAG comparison.
def _by_evidence_span(
    questions: list[Question], per: dict[str, dict[str, dict[str, float]]], systems: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean correctness and recall@8 per system, split by whether the gold evidence spans one
    article or several: the split the graph-vs-RAG research question is about."""
    # Label each question by how many articles its gold evidence spans.
    span = {
        q.qid: "multi-article" if len(q.article_ids) > 1 else "single-article" for q in questions
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for group in ("single-article", "multi-article"):
        out[group] = {}
        for s in systems:
            rows = [m for qid, m in per[s].items() if span.get(qid) == group]
            if not rows:
                continue
            # S0 is closed-book: it retrieves nothing, so recall is not applicable.
            recall = [] if s == "S0" else M.nan_drop([m.get("recall@8", math.nan) for m in rows])
            out[group][s] = {
                "n": len(rows),
                "correctness": sum(m["correctness"] for m in rows) / len(rows),
                "recall@8": sum(recall) / len(recall) if recall else math.nan,
            }
    return out


# Average one metric per question type per system, for the per-type tables.
def _mean_by_type(
    qmap: dict[str, Question],
    per: dict[str, dict[str, dict[str, float]]],
    systems: list[str],
    metric: str,
) -> dict[str, dict[str, float]]:
    """{qtype: {system: mean metric}}, skipping questions where the metric is undefined."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for s in systems:
        groups: dict[str, list[float]] = defaultdict(list)
        for qid, m in per[s].items():
            v = m.get(metric, math.nan)
            if not math.isnan(v):
                groups[qmap[qid].qtype].append(v)
        for qt, vals in groups.items():
            out[qt][s] = sum(vals) / len(vals)
    return dict(out)


# Count verified, auto-screened and total questions, so the report is honest about review.
def _provenance(questions: list[Question]) -> dict[str, Any]:
    """How much of the question set was verified (and by whom) vs only auto-screened."""
    by = Counter(q.verified_by or "unknown" for q in questions if q.verified)
    return {
        "verified": sum(q.verified for q in questions),
        "verified_by": dict(sorted(by.items())),
        "auto_screen_passed": sum(bool(q.screen and q.screen.passed) for q in questions),
        "total": len(questions),
    }


# Stage 5: combine the saved JSONL from all stages into one report dict with CIs and tests.
def build_report(
    questions: list[Question],
    retrievals: list[RetrievalRow],
    answers: list[AnswerRow],
    judged: list[JudgeRow],
    baseline: str = "S1",
) -> dict[str, Any]:
    # Index retrievals and answers by (qid, system) so judged rows can be joined to them.
    qmap = {q.qid: q for q in questions}
    r_by = {(r.qid, r.system): r for r in retrievals}
    a_by = {(a.qid, a.system): a for a in answers}
    per: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)  # system -> qid -> metrics
    # Compute per-question metrics for every judged row.
    for j in judged:
        if j.qid not in qmap:
            continue  # row for a question outside this report's set (e.g. rejected in review)
        q = qmap[j.qid]
        per[j.system][j.qid] = per_question_metrics(
            q, r_by.get((j.qid, j.system)), a_by[(j.qid, j.system)], j
        )
    # Build the main table: bootstrap CI of each metric per system.
    systems = sorted(per)
    metric_names = sorted({k for s in per.values() for m in s.values() for k in m})
    table: dict[str, dict[str, dict[str, float]]] = {}
    for s in systems:
        table[s] = {}
        for name in metric_names:
            vals = M.nan_drop([per[s][qid].get(name, float("nan")) for qid in per[s]])
            ci = bootstrap_ci(vals)
            table[s][name] = {"mean": ci.mean, "lo": ci.lo, "hi": ci.hi, "n": ci.n}
        # Latency percentiles over retrieval plus answer time (no CI for these).
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
        # Abstention precision and recall per system.
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

    # Significance tests vs the baseline, on all questions and on multi-article ones only.
    tests = _paired_tests(per, systems, baseline, TEST_METRICS)
    multi_qids = {q.qid for q in questions if len(q.article_ids) > 1}
    subset_tests = _paired_tests(per, systems, baseline, ("correctness", "recall@8"), multi_qids)

    # Everything the CLI writes to report.json and logs to MLflow.
    return {
        "n_questions": len(questions),
        "types": dict(Counter(q.qtype for q in questions)),
        "provenance": _provenance(questions),
        "table": table,
        "tests": tests,
        "multi_article_tests": subset_tests,
        "by_evidence_span": _by_evidence_span(questions, per, systems),
        "correctness_by_type": _mean_by_type(qmap, per, systems, "correctness"),
        "recall_by_type": _mean_by_type(qmap, per, [s for s in systems if s != "S0"], "recall@8"),
        "per_question": {s: per[s] for s in systems},
    }


# One table cell for the evidence-span table: correctness / recall (n).
def _span_cell(v: dict[str, float]) -> str:
    recall = "—" if math.isnan(v["recall@8"]) else f"{v['recall@8']:.2f}"
    return f"{v['correctness']:.2f} / {recall} (n={int(v['n'])})"


# Render the report dict as a Markdown summary with tables.
def render_markdown(report: dict[str, Any]) -> str:
    table = report["table"]
    systems = sorted(table)
    cols = [
        "correctness", "key_fact_recall", "recall@8", "support_complete@8", "citation_precision",
        "citation_recall", "unsupported_rate", "abstention_precision", "abstention_recall",
        "latency_p50_s", "latency_p95_s",
    ]  # fmt: skip

    # Format one cell as "mean [lo, hi]", or just the mean when there is no CI.
    def cell(s: str, c: str) -> str:
        v = table[s].get(c)
        if not v or v["mean"] is None or (isinstance(v["mean"], float) and math.isnan(v["mean"])):
            return "—"
        if math.isnan(v["lo"]):
            return f"{v['mean']:.2f}"
        return f"{v['mean']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"

    # Main metric table: one row per metric, one column per system.
    lines = [f"Questions: {report['n_questions']} ({report['types']})", ""]
    lines.append("| metric | " + " | ".join(systems) + " |")
    lines.append("|---|" + "---|" * len(systems))
    for c in cols:
        lines.append(f"| {c} | " + " | ".join(cell(s, c) for s in systems) + " |")
    lines += _tests_table("Paired permutation tests vs S1 (Holm-corrected):", report["tests"])
    # Add the provenance and judge lines near the top.
    prov = report.get("provenance")
    if prov:
        lines.insert(
            1,
            f"Verified: {prov['verified']}/{prov['total']}"
            + (f" (by {', '.join(prov['verified_by'])})" if prov.get("verified_by") else "")
            + "; "
            f"auto-screen passed: {prov['auto_screen_passed']}/{prov['total']}",
        )
    judge = report.get("judge")
    if judge:
        lines.insert(1, f"Judge: {judge['model']}, prompt {judge['variant']}")
    # Extra tables: by evidence span, multi-article tests and per-type breakdowns.
    spans = report.get("by_evidence_span", {})
    if any(spans.values()):
        lines += ["", "By evidence span (correctness / recall@8, n):", ""]
        lines.append("| evidence | " + " | ".join(systems) + " |")
        lines.append("|---|" + "---|" * len(systems))
        for group, vals in spans.items():
            if vals:
                cells = [_span_cell(vals[s]) if s in vals else "—" for s in systems]
                lines.append(f"| {group} | " + " | ".join(cells) + " |")
    if report.get("multi_article_tests"):
        lines += _tests_table(
            "Multi-article questions only, vs S1 (Holm-corrected):", report["multi_article_tests"]
        )
    for title, key in (("Correctness", "correctness_by_type"), ("Recall@8", "recall_by_type")):
        by_type = report.get(key, {})
        if by_type:
            lines += _by_type_table(f"{title} by question type:", by_type, systems)
    return "\n".join(lines) + "\n"


# Markdown table for a list of paired test results.
def _tests_table(title: str, tests: list[dict[str, Any]]) -> list[str]:
    lines = ["", title, "", "| metric | system | n | diff | p | p (Holm) | d_z |"]
    lines.append("|---|---|---|---|---|---|---|")
    for t in tests:
        lines.append(
            f"| {t['metric']} | {t['system']} | {t['n']} | {t['diff']:+.3f} | {t['p']:.3f} | "
            f"{t['p_holm']:.3f} | {t['effect_dz']:+.2f} |"
        )
    return lines


# Markdown table of a metric per question type per system.
def _by_type_table(
    title: str, by_type: dict[str, dict[str, float]], systems: list[str]
) -> list[str]:
    lines = ["", title, "", "| type | " + " | ".join(systems) + " |"]
    lines.append("|---|" + "---|" * len(systems))
    for qt, vals in sorted(by_type.items()):
        cells = [f"{vals[s]:.2f}" if s in vals else "—" for s in systems]
        lines.append(f"| {qt} | " + " | ".join(cells) + " |")
    return lines
