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

### D31: Model review in place of human verification
The developer could not verify Apple troubleshooting questions with confidence, so the
verification and judge-labeling steps were run by a stronger model (Claude Opus 5.5) than the
generator and judge (qwen3), and recorded as such rather than as human work.
- Questions: four reviewers applied one written rubric (answer stated in the gold chunks; for
  multi_constraint the fix appears in each symptom's own article; for version_conditional the
  requirement comes from the same article; for cross_device the dependency is stated).
  Decisions and reasons go to `data/bench/generated_review.jsonl`; kept questions get
  `verified_by: claude-opus-5-5`. Rejected questions stay in the file unverified instead of
  being deleted, so the completed run still lines up with the question file; reports use
  `--questions verified`. 6/6 spot-checked decisions matched the evidence.
- Judge labels: two independent blind labelers each scored all 99 sampled answers with the
  judge's rubric; their agreement (κ = 0.91) bounds how well a judge can be expected to agree,
  and the 5 disagreements were adjudicated. Both raters' scores are kept.
- Reports print who verified the questions (`Verified: 93/93 (by claude-opus-5-5)`) instead of
  assuming a human did. A human pass remains the next step.

## Strengthening the evaluation (2026-09-25)

Protocols below were written before the runs they describe.

### D32: Edge verification against the source sentence
Validation (D13) only fuzzy-matches each entity string to the chunk, so an edge between two
strings that merely co-occur passes. `fixgraph kg verify-edges` turns every LLM-extracted edge
into a templated statement using the chunk's own surface forms and asks qwen3:8b, per source
chunk (<= 12 statements per call), whether the passage states it, with an exact supporting
quote; code requires the quote to occur in the passage the model saw (article title, section
heading and chunk text; partial ratio >= 0.85). A 3-chunk smoke run showed correct claims
quoted from the title failing a chunk-text-only check, so the check was widened before the
full run and before any labels existed. An edge is kept if
any source chunk supports it and its provenance is narrowed to those chunks. Failed calls fail
closed. Ontology seed DEPENDS_ON edges (no source text) are dropped; IN_FAMILY (rule-derived
from product names) is kept. The extraction prompt's `depends_on` few-shot example, a likely
source of invented dependencies, is left unchanged: editing it without re-extracting would make
the committed code disagree with the committed extractions. The verified graph is written to
`data/kg_verified/`; nodes and mentions are unchanged.

Measurement, fixed before running the verifier:
- `fixgraph kg edge-sample` draws edges (seed 13, target 200), stratified by relation with
  proportional allocation and a floor of 5 per relation; rounding and the floors give 209 edges
  (235 edge-chunk statements).
- Each sampled edge is labelled "at least one source chunk states it" by two independent blind
  labelers (Claude Opus 5.5, `labeler: claude-opus-5-5`; not human) who see the statements and
  full chunk text but not the verifier's output; disagreements are adjudicated and both raters'
  labels are kept.
- `fixgraph kg edge-eval` reports the hallucinated-edge rate (share of edges no source chunk
  states) over all edges (before) and over kept edges (after), weighted by relation stratum
  size, with stratified bootstrap 95% CIs, plus the verifier's precision/recall against the
  labels. Output: `results/kg/edge_verification.json`. No threshold or prompt is changed after
  seeing the labels.

### D33: Judge calibration with a held-out split
The qwen3:8b judge agreed with the blind reference labels at kappa = 0.38 (target 0.6), and was
one step harsh on fully correct answers. Protocol, fixed before any variant was run:
- Reference labels: the 99 blind labels from `data/bench/judge_labels_test.jsonl` (two Claude
  Opus 5.5 raters, kappa 0.91 with each other, adjudicated). They are model labels, not human;
  every claim says "agreement with model reference labels".
- `fixgraph bench label-split` splits them once (seed 13, stratified by system): 1/3 dev for
  designing prompts, 2/3 held out (`data/bench/judge_label_split_test.json`). The split refuses
  to be redrawn.
- Four variants declared in advance (bench/judge.py): v1 original; v2 tighter rubric (defines
  "main solution", lists what must not lower a score); v3 v2 wording with the score derived in
  code from main_solution yes/partly/no + a contradiction flag; v4 v2 + three worked examples,
  the first dev item at each score level (excluded from v4's own dev score).
- `fixgraph bench judge-calibrate --split dev` scores variants on dev; every run is appended to
  `results/judge/calibration.json`. The variant with the highest dev weighted kappa is chosen
  (ties -> the simpler variant, in the order v1 < v2 < v3 < v4). The chosen variant is scored on
  the held-out split exactly once (the command refuses a second held-out run).
- Known leak: the "one step harsh" pattern that motivated v2-v4 was seen on all 99 labels,
  including the held-out items, before the split. The held-out kappa is therefore optimistic;
  a second, untouched check is the fresh labels collected on the stage-4 benchmark (D35), which
  no prompt was designed against.
- If no variant reaches 0.6, qwen3:14b (partly CPU-offloaded, local) is tried with the chosen
  prompt; if that also misses, the miss is reported. No API judge (the project stays local).
- The chosen variant then re-judges the test run (`bench run --stages judge,report
  --judge-variant vN`); cached answers and verifier calls are reused. `judge_meta.json` records
  which prompt produced each run's scores.

### D34: Document link layer (S2L) and fusion retriever (S4)
Apple articles link to each other inside their instructions; `fixgraph ingest links` records
every link from a content block to another corpus article, with its sentence and source chunk
(`data/corpus/links.parquet`; 305 links, all inside working-set chunks). These links are written
by Apple, not extracted, so they are not subject to the hallucination measured in D32.
- S2L = S2 (PPR, same seeds, damping 0.5) over the entity graph plus a document layer: one node
  per article, joined with weight 0.2 to every entity its chunks mention and with weight 0.8 to
  every article it links to. An article node scores all its chunks. Weights are set a priori by
  analogy (IN_FAMILY 0.2, DEPENDS_ON 0.8) and are not tuned on any benchmark. S2 without the
  layer stays in every run as the ablation.
- S4 = reciprocal-rank fusion (k = 60, equal weights) of the S1 hybrid candidate list (50) and
  the S2L PPR chunk ranking (top 50), top 30 fused chunks reranked by the same cross-encoder;
  without linked seeds it reduces to S1. Fixed before any run; not tuned.
- Because the bridge benchmark (D35) is built from the same links, S2L and S4 see the bridge
  structure and S1 does not (as with hyperlink-based retrievers on HotpotQA). This is stated
  with every bridge result, and S2 (no links) shows how much of any gain the links carry.
- All graph systems (S2, S2L, S3, S4) use the verified graph `data/kg_verified` from here on.

### D35: Bridge benchmark: a fixed rule, frozen before any system runs
Rule (bench/bridge.py), fixed before generation:
1. Candidates: links whose sentence contains if / when / unless and that sit in a corpus chunk;
   one per (source, target) article pair; at most 2 per source article (seeded shuffle, seed 13).
2. qwen3:8b writes a matched pair per candidate from A's chunk and the first 3 chunks of C:
   a bridge question (A's situation and condition; must not name the link) and a direct
   question (asks about C's topic), one shared answer from C, and a verbatim quote from C.
3. Mechanical checks, no LLM: the quote must occur in one of C's chunks (partial ratio >= 0.85;
   that chunk is the gold C chunk); the bridge question may contain no content word of the
   anchor except generic link words and words of A's title; answer-word overlap < 0.6.
4. Review (Claude Opus 5.5, `verified_by: claude-opus-5-5`, not human) checks that the answer
   is stated in C, that the bridge question is a natural question in A's situation that C
   answers without naming the link, and that the direct question asks for the same thing.
   Pairs are kept or dropped as a unit. Nothing is filtered on any system's retrieval or
   answers, and the rule is not revised after generation. If fewer pairs survive than hoped,
   the smaller n is reported.
5. `fixgraph bench bridge-freeze` records the file's SHA-256 before the first retrieval run;
   `bridge-generate` refuses to run after freezing.
Analysis, fixed now:
- Systems S0, S1, S2, S2L, S3, S4 on the verified graph; judge = the variant chosen in D33.
- Primary retrieval metric: answer-chunk hit@8 (the gold C chunk in the top 8), reported for
  bridge and direct questions separately, with bootstrap CIs; paired permutation tests vs S1,
  Holm-corrected across the five comparisons, per question type.
- Hop cost per system = hit@8(direct) - hit@8(bridge) over pairs; the crossover test compares
  each system's hop cost with S1's (paired over pairs, Holm-corrected).
- Correctness (judge) reported the same way. A fresh blind sample of 60 bridge-run answers
  (balanced across systems, two Claude raters) gives the untouched judge-agreement check (D33).

### D36: Serving measurements
`CachedLLMClient` counts hits/misses; `/health` reports the hit rate. `scripts/loadtest.py`
sends the frozen benchmark questions to /retrieve and /answer (async httpx) at concurrency
1 / 4 / 16, 60 requests per cell, and records p50/p95 latency, throughput, errors and the
cell's cache hit rate (difference of the server's counters) to `results/serving/loadtest.json`.
The first /answer pass after a server start on an empty response cache is labelled cold;
repeats are warm. Numbers come from the same laptop that serves the model, so they measure the
single-box setup, not a scaled deployment.
- Amendment (before any question was generated): step 2 is done by Claude Opus 5.5 instead of
  qwen3:8b, from the same inputs (A's chunk, the link sentence and anchor, C's first 3 chunks)
  and the same instructions; questions carry `source: bridge-claude`. Reason: qwen3:8b wrote
  weaker questions in D27-D29 and ties up the GPU for about an hour. The mechanical checks of
  step 3 are unchanged and run in code (`fixgraph bench bridge-import`). The step-4 review is
  done by a separate Claude instance that did not write the questions. Claude does not judge
  answers or verify edges anywhere, so no Claude output is scored against other Claude output.
- Bug fix to the step-3 anchor check (before any system ran, and independent of any system's
  output): it counted function words ("but", "all", "after", "while", "two") and device/OS
  names ("macOS") as link concepts, contrary to the rule's "content word" wording, and rejected
  11 pairs for them (b002 b008 b035 b041 b077 b080 b094 b095 b098 b106 b108). The generic list
  was extended with function words and topic words (the same device/app list the answer-leak
  check uses) are ignored. `results/bridge/generation.json` records the final rejections.

## Graph + hybrid combinations (2026-09-25)

### D37: Miss analysis, circularity, split and search grid (written before building)
Miss analysis (`fixgraph bench miss-analysis`, `results/combo/miss_analysis.json`): S1 misses
18/171 gold chunks on the 93 main questions and 19/234 on the 156 bridge questions. 13/18 and
12/19 of the misses are already in S1's top-30 reranker candidates (demoted by the
cross-encoder); only 5 and 7 are outside it. Verified-KG routes open 200-480 candidate chunks
per question and carry no usable signal; Apple's article links open about 20 and reach 28%
(main) / 95% (bridge) of misses from S1's top 3. The edge source for every candidate is
therefore Apple's human-authored link graph (`data/corpus/links.parquet`), not the
LLM-extracted KG, and results say so.

Circularity: all 78 bridge pairs were built from an Apple link A -> C (D35), so a link-based
reranker partly wins bridge questions by construction; no unlinked bridge pairs exist to
evaluate instead. Bridge results are reported as non-independent and secondary. The primary,
independent evaluation is the 93 main questions, which were generated from KG paths; 6 of their
49 multi-article questions happen to have directly linked gold articles.

Split (`fixgraph bench combo-split`, seed 13): each set is split 60/40 into dev/test,
stratified by question type; bridge pairs (a/d) stay together. The test qids are hash-frozen in
`results/combo/split.json` before any candidate runs, and no test metric is computed until the
finalists are recorded (D38).

Candidates (all return exactly 8 chunks; the candidate set is S1's top-30 reranker pool plus the
chunks of articles Apple-linked from S1's top-d hits; same bge-reranker-v2-m3 cross-encoder):
- e) link prior: score = CE score + lambda if the chunk is link-reached; lambda in {1, 2, 4}
  (cross-encoder logit units), d in {1, 3}, routing on/off (on: apply only when S1's top CE
  score is below the dev median). 12 configs.
- a) protected expansion: S1's top 6 kept, the 2 remaining slots go to the best-scoring
  link-reached chunks not already in the top 6 (S1's 7-8 if none), d in {1, 3}. 2 configs.
- b) union + rerank: pool re-scored by the cross-encoder alone (lambda = 0), d in {1, 3}. 2 configs.
- d) RRF (S4, D34) and S1 are the references.
Selection on dev by main-set recall@8 (primary), bridge answer-chunk hit@8 as a tie-breaker;
at most 2 finalists. Every configuration's dev scores go to `results/combo/dev_log.json`.
Latency: S1 time plus the measured cross-encoder time for the extra link-reached chunks.
- Split frozen: main 57 dev / 36 test, bridge 47 / 31 pairs (94 / 62 questions); test qid list
  SHA-256 `ff8fb961ced277f1e0322750c4df350a6af867b70b906913aa094ae3ccd57bd3` (`results/combo/split.json`). `combo-cache --split test` refuses to run
  before the finalists are recorded, and `combo-test` refuses a second run.
- Dev amendment (test split untouched): the cross-encoder returns probabilities in [0, 1] (dev
  median S1 top score 0.988), not logits, so lambda in {1, 2, 4} promoted every link-reached
  chunk above every other chunk (identical results for all three; logged in
  `results/combo/dev_log_grid1_logit_scale.json`). The lambda grid is rescaled to
  {0.05, 0.1, 0.2}; nothing else changes. Union + rerank (b) equals S1 on dev: no link-only
  chunk enters the top 8 on cross-encoder score alone.

### D38: Finalists and hypothesis (recorded before the test split was cached or scored)
Dev (`results/combo/dev_log.json`, 57 main + 94 bridge questions): no configuration beats S1's
main recall@8 (0.9298); the routed link priors tie it with zero broken hits. Finalists by the
D37 rule (`results/combo/finalists.json`):
1. `prior-d3-l0.1+route`: S1's top-30 pool plus chunks of articles Apple-linked to S1's top 3;
   score = cross-encoder probability + 0.1 for link-reached chunks; applied only when S1's top
   score < 0.988 (dev median), otherwise S1's ranking unchanged. Dev: main 0.9298 (tie),
   bridge hit 0.851 -> 0.894, +5 / -0.
2. `prior-d1-l0.1+route`: the same, expanding from S1's top 1. Dev: main tie, bridge 0.894,
   +4 / -0.
Hypotheses for the single test run (36 main, 62 bridge questions): (H1, primary, independent)
the finalists do not lower main recall@8 (no significant difference vs S1, broken hits <= 1);
a main-set gain is not expected (dev shows none). (H2, secondary, non-independent) the finalists
raise bridge answer-chunk hit@8 over S1. Reported with n, bootstrap CIs, Holm-corrected paired
permutation tests, recovered/broken per question type and extra latency. No tuning afterwards.
