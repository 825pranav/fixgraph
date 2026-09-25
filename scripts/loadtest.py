"""Load-test the FixGraph API (DECISIONS.md D36): latency under concurrency + cache hit rate.

Usage: with the service running (`uv run fixgraph serve`),
    uv run python scripts/loadtest.py --base-url http://localhost:8000 \
        --out results/serving/loadtest.json

For each (endpoint, concurrency) cell the same pool of questions is sent `--requests` times by
`--concurrency` workers (async httpx). Questions come from the frozen benchmark files, so the
answer cells replay realistic LLM calls; /health is read before and after each cell, and the
difference in the server's LLM-cache counters gives the cell's hit rate. The first /answer cell
after a server start on an empty response cache is the cold measurement; repeats are warm.
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_QUESTIONS = ("data/bench/generated.jsonl", "data/bench/bridge.jsonl")


def load_questions(files: tuple[str, ...], limit: int) -> list[str]:
    out: list[str] = []
    for f in files:
        path = Path(f)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("verified", True):
                    out.append(row["question"])
    if not out:
        raise SystemExit(f"no questions found in {files}")
    return out[:limit]


async def run_cell(
    base_url: str, endpoint: str, system: str, questions: list[str], n: int, concurrency: int
) -> dict[str, Any]:
    latencies: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as client:
        before = (await client.get("/health")).json().get("llm_cache") or {}

        async def one(i: int) -> None:
            nonlocal errors
            body = {"question": questions[i % len(questions)], "system": system, "k": 8}
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(endpoint, json=body)
                    r.raise_for_status()
                    latencies.append(time.perf_counter() - t0)
                except httpx.HTTPError:
                    errors += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
        after = (await client.get("/health")).json().get("llm_cache") or {}

    lat = sorted(latencies)
    hits = (after.get("hits") or 0) - (before.get("hits") or 0)
    misses = (after.get("misses") or 0) - (before.get("misses") or 0)
    return {
        "endpoint": endpoint,
        "system": system,
        "concurrency": concurrency,
        "requests": n,
        "errors": errors,
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(lat) / wall, 3) if wall else None,
        "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
        "latency_p95_s": round(lat[max(0, int(len(lat) * 0.95) - 1)], 3) if lat else None,
        "llm_cache": {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / (hits + misses), 4) if hits + misses else None,
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--system", default="S1")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--questions-limit", type=int, default=60)
    ap.add_argument("--endpoints", nargs="+", default=["/retrieve", "/answer"])
    ap.add_argument("--label", default="", help="e.g. cold | warm; recorded with every cell.")
    ap.add_argument("--out", default="results/serving/loadtest.json")
    args = ap.parse_args()

    questions = load_questions(DEFAULT_QUESTIONS, args.questions_limit)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as client:
        health = (await client.get("/health")).json()
    cells = []
    for endpoint in args.endpoints:
        for c in args.concurrency:
            cell = await run_cell(args.base_url, endpoint, args.system, questions, args.requests, c)
            cell["label"] = args.label
            print(json.dumps(cell))
            cells.append(cell)

    out = Path(args.out)
    log = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    log.append(
        {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "server_mode": health.get("mode"),
            "n_questions": len(questions),
            "cells": cells,
        }
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2) + "\n", encoding="utf-8")
    print(f"appended {len(cells)} cells -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
