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

## M3–M7: Retrieval, benchmark, GNN, serving (2026-09-23)

### D21: Qdrant local mode supports the hybrid
Verified in local mode (`QdrantClient(path=...)`): named dense + sparse vectors, `Modifier.IDF`
on the sparse vector (so documents store only the BM25 TF component) and RRF fusion via
`prefetch`. No switch to LanceDB needed. BM25 term ids are CRC32 hashes (no vocabulary file).

### D22: Reranker
`BAAI/bge-reranker-v2-m3` loads cleanly with sentence-transformers' `CrossEncoder` (fp16 on
GPU). S1, S2 and S3 all rerank their top 30 candidates with it, so only candidate generation
differs between systems.

### D23: GraphRAG details
- Entity linking: LLM mentions (qwen3:4b, cached) merged with rule-based ontology mentions,
  matched against a dense index of node names *and* extracted surface forms (aliases); seed
  weight = similarity x HippoRAG node specificity 1/log(2 + #chunks).
- S2: PPR (damping 0.5, as in HippoRAG) on the undirected KG with relation-type weights
  (IN_FAMILY 0.2 ... RESOLVED_BY 1.0) x extraction confidence; chunk score = sum of PPR mass of
  the nodes each chunk supports. Implementation matches `networkx.pagerank` to 1e-6
  (property-tested on random graphs including dangling nodes).
- S3: beam search (width 300, <= 3 hops, IN_FAMILY excluded) for paths ending in
  Symptom/Cause/Fix; paths that touch several seeds are boosted; evidence = provenance chunks
  of path edges.
- Both fall back to S1 candidates when no seed links (reported as `graph_fallback`).
- Predicted (GNN) edges are routing-only: their chunks never become evidence.

### D24: GNN decoder skip term
First run: hetero-GraphSAGE with a pure dot-product decoder scored filtered MRR 0.03 on the
article-held-out split, far below the cosine baseline (0.22): with a few hundred training
pairs it overfits and loses the text signal. The decoder now adds a learnable weight on the
cosine of the input text features (a skip connection), so graph structure is learned as a
correction on top of text similarity. Baselines are still reported separately.

### D25: Benchmark scope under the time box
30 hand-written dev questions (8 single-hop, 6 cross-device, 5 version-conditional, 5
error-code, 3 multi-constraint, 3 unanswerable), gold chunks checked against the corpus text.
Drafted with an AI assistant from the corpus text and marked unverified until the developer reviews them.
Path-based generation (`fixgraph bench generate`) and the keystroke verification CLI
(`fixgraph bench verify`) are implemented and tested but were not run at scale; the ≥150
human-verified questions and the 80–100 judge-validation labels are future work that needs
the developer's time.

### D26: Metric definitions
Citation precision/recall are measured against the gold support chunks; unsupported-claim
rate comes from the qwen3:8b claim verifier; correctness is the qwen3:8b rubric (0/0.5/1);
unanswerable questions score 1 only when the system abstains. CIs: percentile bootstrap
(2,000 resamples). Tests: paired sign-flip permutation (10,000), S1 vs each system,
Holm-corrected per metric.

## Test benchmark: multi-article questions (2026-09-25)

### D27: Question types that genuinely need two articles
The original `cross_device` paths looked multi-article (755 of 768 candidates cite chunks from
two articles), but the second article only supplied the `DEPENDS_ON` fact ("AirPods depend on
iPhone"), which the answer does not need. Counting them as multi-hop would overstate what the
benchmark tests. New type `multi_constraint`: two symptoms, each resolved by the SAME fix node,
where each `RESOLVED_BY` edge is supported by a different article ("I have problem A and
problem B; what one thing could help with both?"). Confirming that the fix covers both needs
both articles. Fixes shared by more than 3 symptoms ("Restart your device") are excluded
because they are guessable closed-book. `version_conditional` paths whose fix and version
requirement come from different articles are sampled first. Default mix: 40 single_hop
(control), 80 multi_constraint, 30 version_conditional, 30 cross_device, 11 error_code (all
that exist).

### D28: Gold answers anchored to the graph
First smoke run: the generator padded gold answers with invented reasons ("by addressing
potential system conflicts") and phrased multi_constraint questions as yes/no questions that
named the fix. The prompt now passes the answer node's text as "the correct answer", requires
the reference answer to restate it using only source text, and forbids naming the answer in
the question.

### D29: Automatic pre-screen before human verification
`fixgraph bench screen` annotates every generated question; it never sets `verified`.
- Answer leak: deterministic. Share of the answer node's content words (crude stemming,
  device/app names and the question's own seed-node words removed) already in the question;
  >= 0.6 flags a leak. An LLM leak check approved obvious leaks in the smoke run.
- Support: qwen3:8b must copy the supporting evidence sentence verbatim; code checks the quote
  occurs in the evidence (fuzzy partial ratio >= 0.85). In the first smoke run an LLM-only
  screen passed 6/6 questions including a nonsense one; with the quote check it rejects
  unsupported answers.
- Multi-article check: each article's evidence is shown alone; if any single article fully
  answers the question, `needs_multiple_articles = False`.
`bench verify` shows the verdict and orders the queue (passed multi-article first), so a
time-boxed human review spends its time on likely keepers. Reports state how many questions are
human-verified vs only auto-screened; `bench run --questions screened|verified` selects them.

### D30: Reporting by evidence span
The report splits correctness and recall@8 by single- vs multi-article gold evidence and runs
the paired permutation tests (Holm) on the multi-article subset alone, because that subset is
where the research question lives. Judge-validation labels are sampled round-robin across
systems so kappa is not dominated by one retriever.
