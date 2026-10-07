"""Cross-encoder reranking behind a protocol (real model or a lexical fake for tests).

Used by: retrieval.hybrid (final stage of S1-S3); built by bench/cli.py and api/app.py.
Uses: retrieval.bm25 (tokenizer for LexicalReranker).
"""

# Imports: gc/logging for model cleanup and logs; the BM25 tokenizer is reused by the fake reranker.
import gc
import logging
from typing import Protocol

from fixgraph.retrieval.bm25 import tokenize

logger = logging.getLogger(__name__)

# Model ids for the default bge reranker, the Qwen alternative and the troubleshooting instruction.
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
QWEN_RERANKER = "Qwen/Qwen3-Reranker-0.6B"
TROUBLESHOOT_PROMPT = (
    "Given a question about a problem with an Apple device, retrieve support article passages "
    "that answer it"
)

# Named reranker settings compared in the reranking study (DECISIONS.md D39):
# name -> (model, max_length in tokens, instruction prompt or None for the model default).
# Qwen3-Reranker runs with batch size 4: a batch of 16 padded to 1024 tokens overflows 6 GB of
# VRAM into the driver's system-memory fallback (13-18 s queries in the first timing run).
RERANKERS: dict[str, tuple[str, int, str | None]] = {
    "bge": (DEFAULT_RERANKER, 512, None),
    "bge1024": (DEFAULT_RERANKER, 1024, None),
    "qwen": (QWEN_RERANKER, 1024, None),
    "qwents": (QWEN_RERANKER, 1024, TROUBLESHOOT_PROMPT),
    # D40: fine-tuned on out-of-corpus synthetic queries (`fixgraph bench ft-train`); local only
    "bgeft": ("data/models/bge-ft", 512, None),
    "qwentsft": ("data/models/qwents-ft", 1024, TROUBLESHOOT_PROMPT),
}


# Interface for rerankers: score(question, passages) gives one relevance score per passage.
class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...

    def release(self) -> None: ...


# Real cross-encoder: reads the question and each passage together and scores how well they match.
class CrossEncoderReranker:
    # Loads the cross-encoder once (float16 on GPU) with its max length and optional instruction.
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str | None = None,
        max_length: int = 512,
        prompt: str | None = None,
        batch_size: int = 16,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self._model = CrossEncoder(
            model_name, device=device, max_length=max_length, model_kwargs=kwargs
        )
        self._batch_size = batch_size
        self._prompt = prompt  # instruction for instruction-tuned rerankers (Qwen3-Reranker)
        logger.info("loaded reranker %s on %s", model_name, device)

    # Builds a reranker from a RERANKERS name; Qwen models get a smaller batch to fit in VRAM.
    @classmethod
    def named(cls, name: str, device: str | None = None) -> "CrossEncoderReranker":
        model, max_length, prompt = RERANKERS[name]
        batch = 4 if model == QWEN_RERANKER or "qwen" in name else 16
        return cls(model, device=device, max_length=max_length, prompt=prompt, batch_size=batch)

    # Scores (question, passage) pairs in batches; HybridRetriever.rerank calls it on the top 30.
    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [(query, d) for d in docs]
        scores = self._model.predict(pairs, batch_size=self._batch_size, prompt=self._prompt)
        return [float(s) for s in scores]

    # Frees the cross-encoder and GPU cache when a stage is finished with it.
    def release(self) -> None:
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Test double that scores passages without loading any model.
class LexicalReranker:
    """Fake reranker: query-term overlap. Deterministic, for tests."""

    # Score = share of the question's terms that also appear in the passage.
    def score(self, query: str, docs: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) / (len(q) or 1) for d in docs]

    # Nothing to free; exists to match the Reranker interface.
    def release(self) -> None:
        pass
