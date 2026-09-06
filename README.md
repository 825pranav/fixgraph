# FixGraph

Troubleshooting knowledge graph, GNN link prediction and GraphRAG benchmark, running on free
local open-weight models. See [PROJECT_SPEC.md](PROJECT_SPEC.md). Results-first README comes in M7.

## Quick start (Windows 11, native)

```powershell
uv sync
uv run poe gpu            # must print CUDA available: True + your GPU
ollama pull qwen3:4b
copy .env.example .env
uv run poe check          # ruff + pyright + unit tests
uv run fixgraph llm smoke # one schema-constrained call to Ollama
```
