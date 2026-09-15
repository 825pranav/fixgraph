"""`fixgraph index build` and `fixgraph bench run | report`."""

import json
import logging
from pathlib import Path
from typing import cast, get_args

import typer
from tqdm import tqdm

from fixgraph.bench import run as R
from fixgraph.bench.schema import Question, QuestionFilter, read_questions, select_questions
from fixgraph.core.config import Settings, load_settings
from fixgraph.ingest.store import read_articles, read_chunks
from fixgraph.retrieval.index import chunk_document

logger = logging.getLogger(__name__)
index_app = typer.Typer(no_args_is_help=True, help="Search indexes (Qdrant local mode).")
app = typer.Typer(no_args_is_help=True, help="TroubleshootQA benchmark.")


def _chunk_docs(settings: Settings) -> tuple[dict[str, str], dict[str, str]]:
    """chunk_id -> indexed document (title + text), chunk_id -> text shown to the answer model."""
    paths = settings.paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    chunks = read_chunks(paths.chunks)
    docs = {c.chunk_id: chunk_document(c, titles.get(c.article_id, "")) for c in chunks}
    return docs, docs


@index_app.command("build")
def index_build() -> None:
    """Embed chunks (dense + BM25) and KG node names into Qdrant local mode."""
    import polars as pl

    from fixgraph.embeddings import SentenceTransformerEmbedder
    from fixgraph.kg.store import read_kg
    from fixgraph.retrieval.index import build_chunk_index, build_node_index, open_client

    settings = load_settings()
    paths = settings.paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    chunks = read_chunks(paths.chunks)
    embedder = SentenceTransformerEmbedder()
    client = open_client(paths.index)
    try:
        build_chunk_index(client, chunks, titles, embedder, paths.index)
        if (paths.kg / "nodes.parquet").exists():
            kg = read_kg(paths.kg)
            nodes = kg.nodes.filter(pl.col("label") != "Article")
            names = {
                (r["node_id"], r["label"], r["canonical_text"]) for r in nodes.iter_rows(named=True)
            }
            labels = dict(zip(nodes["node_id"], nodes["label"], strict=True))
            for nid, surface in kg.mentions.select("node_id", "surface").iter_rows():
                if nid in labels:
                    names.add((nid, labels[nid], surface))  # surface forms act as aliases
            build_node_index(client, sorted(names), embedder)
    finally:
        client.close()
        embedder.release()
    typer.echo(f"indexed {len(chunks)} chunks -> {paths.index}")


_WHICH_HELP = "all | screened (auto-screen passed or human-verified) | verified (human only)."


def _load_questions(questions_file: str, which: str, limit: int | None) -> list[Question]:
    if which not in get_args(QuestionFilter):
        raise typer.BadParameter(f"--questions must be one of {get_args(QuestionFilter)}")
    selected = select_questions(read_questions(Path(questions_file)), cast(QuestionFilter, which))
    return selected[: limit or None]


def _unload(settings: Settings, model: str) -> None:
    if settings.llm.backend == "ollama":
        from fixgraph.llm.ollama import OllamaClient

        c = OllamaClient(settings.llm.base_url)
        c.unload(model)
        c.close()


@app.command("run")
def bench_run(
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    run_name: str = typer.Option("dev"),
    systems: str = typer.Option("S0,S1,S2,S3"),
    stages: str = typer.Option("mentions,retrieve,answer,judge,report"),
    limit: int | None = typer.Option(None, help="Only the first N questions (smoke runs)."),
    type_weights: bool = typer.Option(True, help="--no-type-weights = PPR ablation."),
    kg_dir: str | None = typer.Option(None, help="Alternative KG (e.g. data/kg_nocanon)."),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
) -> None:
    """Run benchmark stages; one heavy model on the GPU at a time."""
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    paths = settings.paths
    questions = _load_questions(questions_file, which, limit)
    out = paths.results / run_name
    system_list = [s.strip() for s in systems.split(",") if s.strip()]
    stage_list = [s.strip() for s in stages.split(",")]
    _, chunk_text = _chunk_docs(settings)
    answer_model, judge_model = settings.llm.answer_model, settings.llm.judge_model

    if "mentions" in stage_list:
        client = build_llm_client(settings)
        try:
            R.stage_mentions(questions, client, settings.llm.linking_model, out / "mentions.jsonl")
        finally:
            client.close()
            _unload(settings, settings.llm.linking_model)

    if "retrieve" in stage_list:
        retrievals = _retrieve(settings, questions, system_list, out, type_weights, kg_dir)
        logger.info("retrieved %d rows", len(retrievals))

    if "answer" in stage_list:
        retrievals = [R.RetrievalRow.model_validate(r) for r in R._read(out / "retrieval.jsonl")]
        client = build_llm_client(settings)
        try:
            R.stage_answer(
                questions,
                retrievals,
                system_list,
                chunk_text,
                client,
                answer_model,
                out / "answers.jsonl",
            )
        finally:
            client.close()
            _unload(settings, answer_model)

    if "judge" in stage_list:
        answers = [R.AnswerRow.model_validate(r) for r in R._read(out / "answers.jsonl")]
        client = build_llm_client(settings)
        try:
            R.stage_judge(questions, answers, chunk_text, client, judge_model, out / "judged.jsonl")
        finally:
            client.close()
            _unload(settings, judge_model)

    if "report" in stage_list:
        report_(run_name=run_name, questions_file=questions_file, limit=limit, which=which)


def _retrieve(
    settings: Settings,
    questions: list[Question],
    system_list: list[str],
    out: Path,
    type_weights: bool,
    kg_dir: str | None,
) -> list[R.RetrievalRow]:
    from fixgraph.core.ontology import load_ontology
    from fixgraph.embeddings import SentenceTransformerEmbedder
    from fixgraph.kg.store import read_kg
    from fixgraph.retrieval.base import NoRetrieval, Retriever
    from fixgraph.retrieval.graph import build_graph_index
    from fixgraph.retrieval.graphrag import PathRetriever, PPRRetriever
    from fixgraph.retrieval.hybrid import HybridRetriever
    from fixgraph.retrieval.index import load_bm25, open_client
    from fixgraph.retrieval.linking import EntityLinker, merge_mentions, rule_mentions
    from fixgraph.retrieval.rerank import CrossEncoderReranker

    paths = settings.paths
    docs, _ = _chunk_docs(settings)
    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    client = open_client(paths.index)
    try:
        hybrid = HybridRetriever(client, embedder, load_bm25(paths.index), reranker, docs)
        systems: dict[str, Retriever] = {}
        if "S1" in system_list:
            systems["S1"] = hybrid
        if {"S2", "S3"} & set(system_list):
            kg = read_kg(Path(kg_dir) if kg_dir else paths.kg)
            graph = build_graph_index(kg, use_type_weights=type_weights)
            counts = {graph.node_ids[i]: len(c) for i, c in graph.node_chunks.items()}
            linker = EntityLinker(client, embedder, counts)
            onto = load_ontology()
            llm_m = R.load_mentions(out / "mentions.jsonl")
            mentions = {
                q.question: merge_mentions(
                    llm_m.get(q.question, rule_mentions("", onto)), rule_mentions(q.question, onto)
                )
                for q in questions
            }
            if "S2" in system_list:
                systems["S2"] = PPRRetriever(graph, linker, hybrid, mentions)
            if "S3" in system_list:
                systems["S3"] = PathRetriever(graph, linker, hybrid, mentions)
        if "S0" in system_list:
            systems["S0"] = NoRetrieval()
        return R.stage_retrieve(questions, systems, out / "retrieval.jsonl")
    finally:
        client.close()
        reranker.release()
        embedder.release()


@app.command("report")
def report_(
    run_name: str = typer.Option("dev"),
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    limit: int | None = typer.Option(None),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
) -> None:
    """Metrics with bootstrap CIs, significance tests, per-type table; logs to MLflow."""
    settings = load_settings()
    out = settings.paths.results / run_name
    questions = _load_questions(questions_file, which, limit)
    retrievals = [R.RetrievalRow.model_validate(r) for r in R._read(out / "retrieval.jsonl")]
    answers = [R.AnswerRow.model_validate(r) for r in R._read(out / "answers.jsonl")]
    judged = [R.JudgeRow.model_validate(r) for r in R._read(out / "judged.jsonl")]
    report = R.build_report(questions, retrievals, answers, judged)
    stem = "report" if which == "all" else f"report_{which}"  # one file per question filter
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = R.render_markdown(report)
    (out / f"{stem}.md").write_text(md, encoding="utf-8")
    try:
        import mlflow

        mlflow.set_tracking_uri((settings.paths.root.parent / "mlruns").as_uri())
        mlflow.set_experiment("troubleshootqa")
        with mlflow.start_run(run_name=f"{run_name}-{which}"):
            mlflow.log_params({"questions": questions_file, "filter": which, "n": len(questions)})
            for system, metrics in report["table"].items():
                for name, v in metrics.items():
                    if v["mean"] == v["mean"]:  # skip NaN
                        mlflow.log_metric(f"{system}.{name.replace('@', '_at_')}", float(v["mean"]))
            mlflow.log_artifact(str(out / f"{stem}.md"))
    except Exception as exc:  # MLflow is bookkeeping; never fail the report on it
        logger.warning("MLflow logging skipped: %s", exc)
    typer.echo(md)


# Default mix for the test set: multi-article types dominate; single_hop is the control group.
DEFAULT_MIX = (
    "single_hop=40,multi_constraint=80,version_conditional=30,cross_device=30,error_code=11"
)


def _parse_mix(mix: str) -> dict[str, int]:
    """Parse `single_hop=40,multi_constraint=80` into {"single_hop": 40, "multi_constraint": 80}."""
    out: dict[str, int] = {}
    for part in mix.split(","):
        name, _, n = part.partition("=")
        out[name.strip()] = int(n)
    return out


@app.command("generate")
def bench_generate(
    mix: str = typer.Option(DEFAULT_MIX, help="Paths per question type, e.g. single_hop=40,..."),
    out_file: str = typer.Option("data/bench/generated.jsonl"),
) -> None:
    """Generate questions from KG paths with the judge-class model (unverified until reviewed)."""
    from fixgraph.bench.generate import generate_question, sample_paths
    from fixgraph.bench.schema import write_questions
    from fixgraph.kg.store import read_kg
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    samples = sample_paths(read_kg(settings.paths.kg), _parse_mix(mix), seed=settings.seed)
    client = build_llm_client(settings)
    qs = []
    try:
        for i, s in enumerate(tqdm(samples, desc="generate")):
            q = generate_question(client, s, chunk_text, settings.llm.judge_model, f"t{i:03d}")
            if q is not None:
                qs.append(q)
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    write_questions(qs, Path(out_file))
    typer.echo(f"{len(qs)} questions from {len(samples)} paths -> {out_file}")


@app.command("screen")
def bench_screen(
    questions_file: str = typer.Option("data/bench/generated.jsonl"),
    rescreen: bool = typer.Option(False, help="Re-run questions that already have a verdict."),
) -> None:
    """Automatic pre-screen (answerable? supported? leaks answer? truly multi-article?).

    Advisory only: results are stored on each question and shown during `bench verify`."""
    from collections import Counter

    from fixgraph.bench.schema import write_questions
    from fixgraph.bench.screen import review_order, screen_question
    from fixgraph.kg.store import read_kg
    from fixgraph.llm.factory import build_llm_client

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    nodes = read_kg(settings.paths.kg).nodes
    node_text = dict(zip(nodes["node_id"], nodes["canonical_text"], strict=True))
    path = Path(questions_file)
    qs = read_questions(path)
    client = build_llm_client(settings)
    try:
        for i, q in enumerate(tqdm(qs, desc="screen")):
            if q.screen is None or rescreen:
                qs[i] = q.model_copy(
                    update={
                        "screen": screen_question(
                            client, q, chunk_text, settings.llm.judge_model, node_text
                        )
                    }
                )
                write_questions(qs, path)  # resumable: progress survives a crash
    finally:
        client.close()
        _unload(settings, settings.llm.judge_model)
    qs = review_order(qs)
    write_questions(qs, path)
    passed = Counter(q.qtype for q in qs if q.screen and q.screen.passed)
    multi = sum(bool(q.screen and q.screen.passed and q.screen.needs_multiple_articles) for q in qs)
    typer.echo(f"passed {sum(passed.values())}/{len(qs)} ({dict(passed)}); {multi} need >1 article")


@app.command("verify")
def bench_verify(
    questions_file: str = typer.Option("data/bench/generated.jsonl"),
    reviewer: str = typer.Option("developer"),
) -> None:
    """Human verification per question (keys: y = keep, n = reject, s = skip, q = quit).

    Screen-passed questions come first, so stopping early keeps the most likely keepers."""
    from fixgraph.bench.generate import verify_loop
    from fixgraph.bench.schema import write_questions
    from fixgraph.bench.screen import review_order

    settings = load_settings()
    _, chunk_text = _chunk_docs(settings)
    path = Path(questions_file)
    qs = review_order(read_questions(path))
    n = verify_loop(qs, chunk_text, lambda q: write_questions(q, path), reviewer)
    typer.echo(f"verified {n}; {sum(q.verified for q in qs)}/{len(qs)} verified in file")


@app.command("label")
def bench_label(
    run_name: str = typer.Option("dev"),
    questions_file: str = typer.Option("data/bench/dev_handwritten.jsonl"),
    n: int = typer.Option(80, help="Answers to label (spec: 80-100)."),
    labeler: str = typer.Option("developer"),
    which: str = typer.Option("all", "--questions", help=_WHICH_HELP),
) -> None:
    """Blind human labels for judge validation (keys: 1 = correct, 5 = partial, 0 = wrong)."""
    from fixgraph.bench.validate import (
        agreement,
        label_loop,
        load_run,
        read_labels,
        sample_for_labeling,
        write_labels,
    )

    settings = load_settings()
    run_dir = settings.paths.results / run_name
    questions = {q.qid: q for q in _load_questions(questions_file, which, None)}
    answers, judged = load_run(run_dir)
    labels_path = settings.paths.bench / f"judge_labels_{run_name}.jsonl"
    todo = sample_for_labeling(list(answers), questions, n)
    labels = label_loop(
        todo,
        questions,
        answers,
        read_labels(labels_path),
        lambda x: write_labels(x, labels_path),
        labeler,
    )
    if labels:
        typer.echo(agreement(labels, judged).model_dump_json(indent=2))


@app.command("kappa")
def bench_kappa(run_name: str = typer.Option("dev")) -> None:
    """Cohen's kappa between human labels and the LLM judge for a run."""
    from fixgraph.bench.validate import agreement, load_run, read_labels

    settings = load_settings()
    _, judged = load_run(settings.paths.results / run_name)
    labels = read_labels(settings.paths.bench / f"judge_labels_{run_name}.jsonl")
    typer.echo(agreement(labels, judged).model_dump_json(indent=2))
