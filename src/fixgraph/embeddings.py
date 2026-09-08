"""Text embeddings behind a protocol: Qwen3-Embedding via sentence-transformers, or a fake.

GPU memory rule (spec §5.4): call `release()` when a stage is done with the model.
"""

import gc
import hashlib
import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
Matrix = NDArray[np.float32]


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        """L2-normalized embeddings, one row per text. `query` applies the query instruction."""
        ...

    def release(self) -> None: ...


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
        self._model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
        self.batch_size = batch_size
        self.dim = int(self._model.get_embedding_dimension() or 0)
        self._has_query_prompt = "query" in (self._model.prompts or {})
        logger.info("loaded %s on %s (dim %d)", model_name, device, self.dim)

    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        prompt_name = "query" if query and self._has_query_prompt else None
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            prompt_name=prompt_name,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 2000,
        )
        return np.asarray(emb, dtype=np.float32)

    def release(self) -> None:
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model, no GPU).

    Texts sharing words get similar vectors, so clustering/retrieval logic is testable.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().replace("'", "").split():
                h = int(hashlib.md5(word.strip(".,!?").encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def release(self) -> None:
        pass
