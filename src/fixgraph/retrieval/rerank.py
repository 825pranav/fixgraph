"""Cross-encoder reranking behind a protocol (real model or a lexical fake for tests).

Used by: retrieval.hybrid (final stage of S1-S3); built by bench/cli.py and api/app.py.
Uses: retrieval.bm25 (tokenizer for LexicalReranker).
"""

import gc
import logging
from typing import Protocol

from fixgraph.retrieval.bm25 import tokenize

logger = logging.getLogger(__name__)

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class Reranker(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...

    def release(self) -> None: ...


class CrossEncoderReranker:
    def __init__(
        self, model_name: str = DEFAULT_RERANKER, device: str | None = None, max_length: int = 512
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self._model = CrossEncoder(
            model_name, device=device, max_length=max_length, model_kwargs=kwargs
        )
        logger.info("loaded reranker %s on %s", model_name, device)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        scores = self._model.predict([(query, d) for d in docs], batch_size=16)
        return [float(s) for s in scores]

    def release(self) -> None:
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LexicalReranker:
    """Fake reranker: query-term overlap. Deterministic, for tests."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(d))) / (len(q) or 1) for d in docs]

    def release(self) -> None:
        pass
