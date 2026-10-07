"""Run extraction over chunks: cached, resumable, 1-2 concurrent requests (spec §8.1, §5.6).

Output: one JSON line per chunk in data/extractions/<model>_<prompt_version>.jsonl. Chunks already
extracted successfully are skipped, so a crash or sleep just resumes; failures are retried.

Used by: `fixgraph kg extract | build | eval` (kg/cli.py); kg.build consumes ExtractionRecord.
Uses: kg.extraction (prompt/schema), kg.validation.validate, llm.structured, core.ontology.
"""

# Imports: a thread pool for 1-2 parallel LLM calls, plus the prompt, validator and LLM client.
import json
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from tqdm import tqdm

from fixgraph.core.models import Chunk
from fixgraph.core.ontology import Ontology
from fixgraph.kg.extraction import (
    PROMPT_VERSION,
    ChunkExtraction,
    LLMExtraction,
    build_request,
    to_graph,
)
from fixgraph.kg.validation import ValidatedExtraction, validate
from fixgraph.llm.base import LLMClient
from fixgraph.llm.structured import StructuredOutputError, complete_structured

logger = logging.getLogger(__name__)


# One JSONL line per chunk: the raw model form, the typed graph, the validated graph and timing.
# ok=False rows keep the error so the chunk is retried next run.
class ExtractionRecord(BaseModel):
    chunk_id: str
    model: str
    prompt_version: str
    extracted_at: str
    ok: bool
    error: str | None = None
    raw: LLMExtraction | None = None
    graph: ChunkExtraction | None = None
    validated: ValidatedExtraction | None = None
    seconds: float = 0.0


# Make a model name safe for a file name, e.g. "qwen3:4b" -> "qwen3_4b".
def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


# Where the JSONL for this model and prompt version lives, e.g. qwen3_4b_v2.jsonl.
def output_path(extractions_dir: Path, model: str, prompt_version: str = PROMPT_VERSION) -> Path:
    return extractions_dir / f"{model_slug(model)}_{prompt_version}.jsonl"


# Read the extraction JSONL back; kg build and the resume check both use this.
def read_records(path: Path) -> list[ExtractionRecord]:
    """Latest record per chunk (a failed chunk retried later is superseded by its retry)."""
    if not path.exists():
        return []
    latest: dict[str, ExtractionRecord] = {}
    # Later lines overwrite earlier ones for the same chunk, so a retry replaces its failure.
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = ExtractionRecord.model_validate_json(line)
                latest[rec.chunk_id] = rec
    return list(latest.values())


# Extract one chunk: build the prompt, get schema-checked JSON from the LLM, convert and validate.
# Never raises on model failure; returns a record with ok=False instead.
def extract_one(
    client: LLMClient,
    chunk: Chunk,
    title: str,
    model: str,
    num_ctx: int,
    ontology: Ontology,
) -> ExtractionRecord:
    start = time.perf_counter()
    now = datetime.now(UTC).isoformat(timespec="seconds")
    # Ask the model for the nested form; a bad reply after the retry becomes a failed record.
    try:
        request = build_request(chunk, title, model, num_ctx)
        raw = complete_structured(client, request, LLMExtraction)
    except (StructuredOutputError, RuntimeError) as exc:
        logger.warning("extraction failed for %s: %s", chunk.chunk_id, str(exc)[:200])
        return ExtractionRecord(
            chunk_id=chunk.chunk_id,
            model=model,
            prompt_version=PROMPT_VERSION,
            extracted_at=now,
            ok=False,
            error=str(exc)[:500],
            seconds=time.perf_counter() - start,
        )
    # Turn the nested form into typed entities/relations, then ground them against the chunk text.
    graph = to_graph(raw, ontology)
    return ExtractionRecord(
        chunk_id=chunk.chunk_id,
        model=model,
        prompt_version=PROMPT_VERSION,
        extracted_at=now,
        ok=True,
        raw=raw,
        graph=graph,
        validated=validate(graph, chunk.text, context=f"{title}\n{chunk.heading}"),
        seconds=time.perf_counter() - start,
    )


# Main extraction loop used by `fixgraph kg extract`: chunks in, JSONL records appended out.
def run_extraction(
    client: LLMClient,
    chunks: Iterable[Chunk],
    titles: dict[str, str],
    model: str,
    out_path: Path,
    ontology: Ontology,
    num_ctx: int = 8192,
    concurrency: int = 2,
    limit: int | None = None,
) -> dict[str, float]:
    """Extract every not-yet-done chunk; append records to `out_path`. Returns timing stats."""
    # Resume: skip chunks that already have a successful record, optionally cap how many to run.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.chunk_id for r in read_records(out_path) if r.ok}  # failures are retried
    all_chunks = list(chunks)
    todo = [c for c in all_chunks if c.chunk_id not in done]
    batch = todo[:limit] if limit else todo
    lock = threading.Lock()
    wall_start = time.perf_counter()
    n_ok = 0
    # Run extract_one in a small thread pool and append each result as soon as it finishes.
    with (
        out_path.open("a", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool,
    ):
        futures = [
            pool.submit(
                extract_one, client, c, titles.get(c.article_id, ""), model, num_ctx, ontology
            )
            for c in batch
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"extract {model}"):
            rec = fut.result()
            n_ok += rec.ok
            # Written from the main thread; flush so a crash loses only in-flight chunks.
            with lock:
                out.write(rec.model_dump_json() + "\n")
                out.flush()
    # Timing summary, including a projection of how long the remaining chunks will take.
    wall = time.perf_counter() - wall_start
    per_chunk = wall / len(batch) if batch else 0.0
    remaining = len(todo) - len(batch)
    stats = {
        "processed": len(batch),
        "ok": n_ok,
        "already_done": len(done),
        "remaining": remaining,
        "wall_seconds": round(wall, 1),
        "seconds_per_chunk": round(per_chunk, 2),
        "projected_remaining_hours": round(per_chunk * remaining / 3600, 2),
    }
    logger.info("extraction stats: %s", json.dumps(stats))
    return stats
