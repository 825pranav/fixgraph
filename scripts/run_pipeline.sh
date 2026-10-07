#!/usr/bin/env bash
# End-to-end run after extraction: KG -> quality eval -> GNN -> index -> benchmark.
# Each step logs to data/results/pipeline/<step>.log; stops on the first failure.
# Strict mode: stop on any error, unset variable or failed pipe step.
set -euo pipefail
export PYTHONUTF8=1
export PATH="$HOME/.local/bin:$PATH"
# Run from the repo root no matter where the script is called from, and make the log folder.
cd "$(dirname "$0")/.."
LOG=data/results/pipeline
mkdir -p "$LOG"
# Helper: print a timestamped step name, then run the command with all output sent to its log file.
run() { local name=$1; shift; echo "[$(date +%H:%M:%S)] $name"; "$@" > "$LOG/$name.log" 2>&1; }
# The stages in order: build graph, score extraction, train GNN, find gaps, build index, benchmark.
run kg_build        uv run --no-sync fixgraph kg build
run kg_eval         uv run --no-sync fixgraph kg eval --model qwen3:4b
run gnn_train       uv run --no-sync fixgraph gnn train
run gnn_gaps        uv run --no-sync fixgraph gnn gaps
run index_build     uv run --no-sync fixgraph index build
run bench_dev       uv run --no-sync fixgraph bench run --questions-file data/bench/dev_handwritten.jsonl --run-name dev
echo "[$(date +%H:%M:%S)] done"
