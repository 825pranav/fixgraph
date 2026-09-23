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

## M1: Corpus (2026-09-23)

**Plan / files touched:** `src/fixgraph/ingest/{scrape,parse,chunk,select,store,cli}.py`,
`src/fixgraph/core/{models,ontology}.py`, `configs/ontology.yaml`, `DATA.md`,
`tests/unit/test_{scrape,parse,chunk,ontology,corpus_store_select}.py`, a synthetic HTML fixture.

### D10: URL discovery from the sitemap, not crawling
`robots.txt` publishes a sitemap index; its `ac` sitemap lists ~1,900 numeric en-us articles.
Crawling links would hit disallowed search pages; the sitemap is the polite, complete source.

### D11: Scrape everything, then select ~1,000
Fetching all ~1,900 pages at 1 req/s takes ~35 min, and it makes corpus selection a
transparent, testable scoring function instead of a guess made at crawl time.

### D12: Parser keyed on Apple's `gb-*` classes inside `#content`
Headings are `h2/h3.gb-header`, paragraphs `p.gb-paragraph`, steps `ol.gb-list > li`, notes
`div.gb-note/gb-callout`. In-page tables of contents (lists whose links are all `#anchors`),
global navigation and a site-wide third-party disclaimer are dropped. An empty parent heading
is folded into its first child's heading ("Unpair and pair again › On your iPhone") so the
context survives chunking.

### D13: Token counting
`\w+|[^\w\s]` count as a deterministic, dependency-free estimate of BPE tokens for English.
Good enough for the 200–500 token chunk bounds; the LLM context budget uses Ollama's own
counts.

### D14: Windows Smart App Control
Smart App Control (enforcement mode) intermittently blocked unsigned compiled extensions in
the venv (`scipy/linalg/cython_lapack`, `sklearn/.../_radius_neighbors`). The developer turned
it off. Anyone reproducing on Windows 11 with SAC on will see `DLL load failed ... Application
Control policy`.

## M2: Knowledge graph (2026-09-23)

### D15: Prompt v1 (entity list + index-based relations) rejected
On 10 real chunks qwen3:4b produced **zero** valid relations: every one violated head/tail type
constraints (e.g. `Bluetooth HAS_COMPONENT iPhone`, `Symptom EXHIBITS Component`), and one
chunk ran to the 1,500-token cap. Small models are poor at wiring relations by index.

### D16: Prompt v2, a problem-centric nested schema
The model fills `products[] -> problems[] -> {symptom, products, error_codes, components,
features, os_versions, causes, fixes[] -> {action, addresses_cause, requires}}`. The relation
type is implied by where a string sits, so type violations are impossible by construction;
`kg.extraction.to_graph` converts deterministically to typed triples. All lists have
`maxItems` so constrained decoding cannot run away. This is the "small-model tactic" of
§8.1, chosen over two-pass extraction because it needs one call per chunk.
Sample of 16 real chunks: 16/16 valid JSON, 0 type violations, 78% of relations grounded.

### D17: Grounding-based confidence and context-aware validation
Every entity and relation evidence string is fuzzy-matched against what the model saw
(article title + section heading + chunk); threshold 0.9 for relation evidence (spec) and
0.8 for entity grounding. `extraction_confidence = min(head, tail, evidence)` grounding
scores; self-reported confidences from a 4B model are uninformative. Grounding against the
title/heading as well as the chunk raised kept relations from 61% to 78% on the sample (the
symptom is usually stated in the article title).

### D18: Throughput
qwen3:4b Q4_K_M, 8k ctx, prompt v2: 6.2 s/chunk at concurrency 1, 5.4 s/chunk at
concurrency 2 (GPU otherwise idle). With a game running on the GPU it was 3–4x slower
(VRAM contention forces partial CPU offload), so long runs need the GPU to themselves.

### D19: Entity resolution
Rule-based for Product (ontology model patterns, then families), OSVersion (regex + macOS
marketing names; unversioned mentions like "macOS" are dropped), ErrorCode, Component and
Feature (ontology aliases, else normalized text). Symptom/Cause/Fix: Qwen3-Embedding-0.6B +
average-linkage agglomerative clustering on cosine distance (scipy), per-type thresholds, medoid
as canonical; greedy leader clustering above 12k distinct strings (memory). Optional LLM
adjudication for borderline cluster pairs (similarity band just below the threshold), capped
at 500 pairs, cached. Every merge is logged to `data/kg/merges.jsonl`.
Sanity check: paraphrases score 0.91–0.98 cosine, different problems 0.4–0.7, so default
distance thresholds 0.10–0.12 sit in the gap; tuned on labeled pairs below.

### D20: Time-boxed working corpus (500 chunks)
The developer is time-constrained, so the working corpus is cut from 1,000 articles / 2,531
chunks to **220 whole articles / 500 chunks** (`fixgraph ingest subset --max-chunks 500`):
the 50 gold-set articles are always kept, the rest are the highest-relevance articles that fit.
Every system (RAG and GraphRAG) indexes the same 500 chunks, so the comparison stays fair; the
full corpus is kept in `articles_full.parquet` / `chunks_full.parquet` and the pipeline can be
rerun at full size. Cost: a smaller graph (fewer multi-hop paths, sparser GNN training data)
and wider confidence intervals; per-chunk extraction quality is unchanged.
