"""Run extraction over chunks: cached, resumable, 1-2 concurrent requests (spec §8.1, §5.6).

Output: one JSON line per chunk in data/extractions/<model>_<prompt_version>.jsonl. Chunks already
extracted successfully are skipped, so a crash or sleep just resumes; failures are retried.
"""

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


def model_slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def output_path(extractions_dir: Path, model: str, prompt_version: str = PROMPT_VERSION) -> Path:
    return extractions_dir / f"{model_slug(model)}_{prompt_version}.jsonl"


def read_records(path: Path) -> list[ExtractionRecord]:
    """Latest record per chunk (a failed chunk retried later is superseded by its retry)."""
    if not path.exists():
        return []
    latest: dict[str, ExtractionRecord] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = ExtractionRecord.model_validate_json(line)
                latest[rec.chunk_id] = rec
    return list(latest.values())


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r.chunk_id for r in read_records(out_path) if r.ok}  # failures are retried
    all_chunks = list(chunks)
    todo = [c for c in all_chunks if c.chunk_id not in done]
    batch = todo[:limit] if limit else todo
    lock = threading.Lock()
    wall_start = time.perf_counter()
    n_ok = 0
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
            with lock:
                out.write(rec.model_dump_json() + "\n")
                out.flush()
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
