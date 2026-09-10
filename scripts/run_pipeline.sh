#!/usr/bin/env bash
# End-to-end run after extraction: KG -> quality eval -> GNN -> index -> benchmark.
# Each step logs to data/results/pipeline/<step>.log; stops on the first failure.
set -euo pipefail
export PYTHONUTF8=1
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."
LOG=data/results/pipeline
mkdir -p "$LOG"
run() { local name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOG/$name.log" 2>&1; }
run kg_build        uv run --no-sync fixgraph kg build
run kg_eval         uv run --no-sync fixgraph kg eval --model qwen3:4b
run gnn_train       uv run --no-sync fixgraph gnn train
run gnn_gaps        uv run --no-sync fixgraph gnn gaps
run index_build     uv run --no-sync fixgraph index build
run bench_dev       uv run --no-sync fixgraph bench run --questions-file data/bench/dev_handwritten.jsonl --run-name dev
echo "[$(date +%H:%M:%S)] done"
