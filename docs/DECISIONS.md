# Design decisions

Newest milestone last. Each entry: decision, reason, alternatives considered.

## M0: Scaffold & machine check (2026-09-23)

**Plan / files touched:** `pyproject.toml` (deps, CUDA index, poe tasks, ruff/pyright/pytest config),
`scripts/check_gpu.py`, `src/fixgraph/core/{config,paths}.py`, `src/fixgraph/llm/*`,
`src/fixgraph/cli.py`, `configs/base.yaml`, `.env.example`, `.gitignore`, `tests/unit/*`,
`tests/integration/test_ollama_structured.py`, `.github/workflows/ci.yml`.

**Machine verified:** RTX 4050 Laptop GPU (6 GiB, CC 8.9), NVIDIA driver 610.62, Ollama 0.34.2,
uv 0.12.18, CPython 3.12.14.

### D1: PyTorch CUDA variant: `cu130`
torch 2.14.0 (latest stable) is published for cu126, cu130 and cu132 (RELEASE.md compatibility
matrix). cu130 is the middle, mainstream build; driver 610 supports it. The pytorch.org selector
page renders client-side, so the wheel index itself was checked. `uv run poe gpu` confirms
`2.14.0+cu130`, CUDA available: True.

### D2: Project location `C:\dev\fixgraph`
Per spec §5.2: short path, no spaces, not OneDrive-synced.

### D3: Default LLM backend is Ollama's native `/api/chat`, not `/v1`
Verified empirically: Ollama's OpenAI-compatible `/v1/chat/completions` **ignores `num_ctx`**
(`/api/ps` showed the model loaded with `context_length: 4096`). The native API honours
`options.num_ctx` (confirmed 8192), `think`, `keep_alive` and `format` (JSON Schema). Spec §5.5
requires explicit `num_ctx`, so `OllamaClient` (native) is the default. `OpenAICompatClient`
remains for vLLM / any hosted endpoint (`llm.backend: openai`). Both implement the same
`LLMClient` protocol. The `/v1` base URL from `.env` is accepted by the native client (the `/v1`
suffix is stripped).

### D4: Qwen3 thinking disabled via `think: false` (native API)
Ollama 0.34 supports the `think` toggle directly, which is cleaner than `/no_think` prompt
tags. On `/v1` the equivalent is `reasoning_effort: "none"` (also verified: no reasoning tokens).
Config: `llm.think: false`.

### D5: No OpenAI SDK dependency
Clients are ~60 lines of `httpx` each (httpx is already in §4). Avoids a dependency not in §4
(working rule 4) and keeps request bodies explicit and unit-testable with `httpx.MockTransport`.

### D6: Structured-output retry
`complete_structured` sends the Pydantic model's JSON Schema, validates, and on failure retries
once with the invalid output + validation error appended; a second failure raises
`StructuredOutputError` for callers to log-and-skip (spec §5.5).

### D7: Cache key
SHA-256 over the canonical JSON of the full `LLMRequest` (model, messages, temperature,
max_tokens, num_ctx, think, json_schema, seed). Any parameter change busts the cache.
`keep_alive` is a client setting, not part of the key.

### D8: Config precedence
init kwargs > environment > `.env` > `configs/base.yaml` > code defaults, via pydantic-settings.
Nested keys use `__` (e.g. `LLM__BACKEND=fake`), so `.env.example` uses `LLM__BASE_URL`
rather than the spec's `LLM_BASE_URL`.

### D9: CI installs the same CUDA torch wheel on Linux
Simplest option: one lockfile, one install path. The CUDA wheel runs fine on CPU-only runners;
the cost is a larger download, mitigated by setup-uv caching. Revisit only if CI time becomes a
problem (option: a `cpu` / `cu130` extra with uv conflicts). The Docker image (M7) will use CPU
torch as §5.9 specifies.

### Measured
qwen3:4b Q4_K_M, fully in VRAM (3.0 GiB), 8k context: warm structured call ≈ 1.1 s for
29 prompt / 26 output tokens (~23 tok/s generation); cold load ≈ 4 s.
