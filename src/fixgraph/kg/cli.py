"""`fixgraph kg extract | resolve | build | eval | ...`."""

import json
import logging
from pathlib import Path

import typer

from fixgraph.core.config import load_settings
from fixgraph.core.ontology import load_ontology
from fixgraph.ingest.store import read_articles, read_chunks
from fixgraph.kg.build import build_kg
from fixgraph.kg.quality import GoldChunk, evaluate_extractions, graph_stats
from fixgraph.kg.resolve import llm_adjudicator
from fixgraph.kg.run_extract import output_path, read_records, run_extraction
from fixgraph.kg.store import read_kg, write_kg
from fixgraph.llm.factory import build_llm_client
from fixgraph.llm.ollama import OllamaClient

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, help="Knowledge-graph construction.")


@app.command()
def extract(
    model: str | None = typer.Option(None, help="Defaults to llm.extraction_model."),
    limit: int | None = typer.Option(None, help="Process at most N pending chunks."),
    concurrency: int = typer.Option(2, help="Concurrent requests (match OLLAMA_NUM_PARALLEL)."),
    chunk_ids_file: str | None = typer.Option(None, help="Only chunks listed (one id per line)."),
    dry_run: bool = typer.Option(False, help="Print what would run; no LLM calls."),
    unload: bool = typer.Option(True, help="Unload the model from VRAM when done."),
) -> None:
    """Extract typed entities/relations from chunks with the local LLM (cached, resumable)."""
    settings = load_settings()
    paths = settings.paths
    model = model or settings.llm.extraction_model
    chunks = read_chunks(paths.chunks)
    if chunk_ids_file:
        lines = Path(chunk_ids_file).read_text(encoding="utf-8").splitlines()
        wanted = {line.strip() for line in lines if line.strip()}
        chunks = [c for c in chunks if c.chunk_id in wanted]
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    out = output_path(paths.extractions, model)
    if dry_run:
        typer.echo(f"{len(chunks)} chunks in scope, model {model}, output {out}")
        return
    client = build_llm_client(settings)
    try:
        stats = run_extraction(
            client,
            chunks,
            titles,
            model,
            out,
            ontology=load_ontology(),
            num_ctx=settings.llm.num_ctx,
            concurrency=concurrency,
            limit=limit,
        )
    finally:
        client.close()
    if unload and settings.llm.backend == "ollama":
        ollama = OllamaClient(settings.llm.base_url)
        ollama.unload(model)
        ollama.close()
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def build(
    model: str | None = typer.Option(None, help="Extraction model whose output to use."),
    canonicalize: bool = typer.Option(True, help="--no-canonicalize = ablation graph."),
    adjudicate: bool = typer.Option(False, help="LLM adjudication of borderline merges."),
    out_dir: str | None = typer.Option(None, help="Defaults to data/kg (or data/kg_nocanon)."),
) -> None:
    """Resolve entities and write the parquet KG (nodes, edges with provenance, mentions)."""
    from fixgraph.embeddings import SentenceTransformerEmbedder

    settings = load_settings()
    paths = settings.paths
    model = model or settings.llm.extraction_model
    records = read_records(output_path(paths.extractions, model))
    if not records:
        raise typer.BadParameter(f"no extractions for {model}; run `fixgraph kg extract` first")
    target = Path(out_dir) if out_dir else (paths.kg if canonicalize else paths.root / "kg_nocanon")
    embedder = SentenceTransformerEmbedder()
    client = build_llm_client(settings) if adjudicate else None
    try:
        adj = llm_adjudicator(client, settings.llm.extraction_model) if client else None
        kg, report = build_kg(
            records,
            read_articles(paths.articles),
            load_ontology(),
            embedder,
            adjudicate=adj,
            canonicalize=canonicalize,
        )
    finally:
        embedder.release()
        if client:
            client.close()
    write_kg(kg, target)
    with (target / "merges.jsonl").open("w", encoding="utf-8") as f:
        for m in report.merges:
            f.write(m.model_dump_json() + "\n")
    stats = graph_stats(kg)
    stats["records"] = report.n_records
    stats["failed_records"] = report.n_failed_records
    stats["dropped_mentions"] = dict(report.dropped_mentions.most_common())
    stats["merge_groups"] = len(report.merges)
    (target / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def stats(kg_dir: str | None = typer.Option(None)) -> None:
    """Print graph statistics for a built KG."""
    paths = load_settings().paths
    typer.echo(json.dumps(graph_stats(read_kg(Path(kg_dir) if kg_dir else paths.kg)), indent=2))


@app.command("eval")
def eval_(
    models: list[str] | None = typer.Option(None, "--model", help="Repeatable; default 4B."),
    gold_file: str | None = typer.Option(None, help="Defaults to data/gold/extraction_gold.jsonl"),
) -> None:
    """Entity/relation P/R/F1 per type against the gold set, per extraction model."""
    paths = load_settings().paths
    gold_path = Path(gold_file) if gold_file else paths.gold / "extraction_gold.jsonl"
    gold = [
        GoldChunk.model_validate_json(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report: dict[str, object] = {"gold_chunks": len(gold)}
    for model in models or ["qwen3:4b"]:
        recs = {r.chunk_id: r for r in read_records(output_path(paths.extractions, model))}
        missing = [g.chunk_id for g in gold if g.chunk_id not in recs]
        preds = {cid: r.validated for cid, r in recs.items() if r.ok and r.validated}
        ent, rel = evaluate_extractions(preds, gold)
        secs = [recs[g.chunk_id].seconds for g in gold if g.chunk_id in recs]
        report[model] = {
            "missing_predictions": len(missing),
            "seconds_per_chunk": round(sum(secs) / len(secs), 2) if secs else None,
            "entities": [prf.row(k) for k, prf in sorted(ent.items())],
            "relations": [prf.row(k) for k, prf in sorted(rel.items())],
        }
    typer.echo(json.dumps(report, indent=2))


@app.command("gold-sample")
def gold_sample(n: int = typer.Option(50), seed: int = typer.Option(13)) -> None:
    """Pick N chunks for the gold set and write their ids to data/gold/gold_chunk_ids.txt."""
    from fixgraph.kg.annotate import sample_gold_chunks

    paths = load_settings().paths
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    picked = sample_gold_chunks(read_chunks(paths.chunks), titles, n=n, seed=seed)
    out = paths.gold / "gold_chunk_ids.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(c.chunk_id + "\n" for c in picked), encoding="utf-8")
    typer.echo(f"{len(picked)} chunk ids -> {out}")


@app.command()
def annotate(reviewer: str = typer.Option("developer")) -> None:
    """Review draft gold annotations: [a]ccept, [e]dit in Notepad, [s]kip, [q]uit."""
    from fixgraph.kg.annotate import read_gold, review_loop, write_gold

    paths = load_settings().paths
    gold_path = paths.gold / "extraction_gold.jsonl"
    gold = read_gold(gold_path)
    chunks = {c.chunk_id: c for c in read_chunks(paths.chunks)}
    titles = {a.article_id: a.title for a in read_articles(paths.articles)}
    done = review_loop(gold, chunks, titles, lambda g: write_gold(g, gold_path), reviewer)
    remaining = sum(g.status != "reviewed" for g in gold)
    typer.echo(f"reviewed {done} this session; {remaining} drafts remaining")
