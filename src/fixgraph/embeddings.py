"""Text embeddings behind a protocol: Qwen3-Embedding via sentence-transformers, or a fake.

GPU memory rule (spec §5.4): call `release()` when a stage is done with the model.

Used by: kg.resolve / kg.build (entity clustering), gnn.data (node features), retrieval.index,
retrieval.hybrid and retrieval.linking (dense vectors), and the kg/gnn/bench CLIs and api/app.py.
Tests use FakeEmbedder. No fixgraph imports.
"""

# Imports: numpy for the vector matrices; torch and sentence-transformers load lazily in the class.
import gc
import hashlib
import logging
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

# Module logger so model loading shows up in the run logs.
logger = logging.getLogger(__name__)

# Default dense model for chunks, questions and node names.
# Matrix is the (n_texts, dim) float32 array every encode() returns.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
Matrix = NDArray[np.float32]


# Interface every embedder follows, so callers can swap the real model for the fake one in tests.
class Embedder(Protocol):
    dim: int

    # Takes a list of texts and returns one unit-length vector per text (query=True for questions).
    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        """L2-normalized embeddings, one row per text. `query` applies the query instruction."""
        ...

    # Frees the model so the next pipeline stage can use the GPU memory.
    def release(self) -> None: ...


# Real embedder: wraps a sentence-transformers model (Qwen3-Embedding by default).
class SentenceTransformerEmbedder:
    # Loads the model once on GPU (half precision) if present, else CPU; records the vector size.
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
        # Remember if the model ships a "query" prompt, so questions can get the query instruction.
        self._has_query_prompt = "query" in (self._model.prompts or {})
        logger.info("loaded %s on %s (dim %d)", model_name, device, self.dim)

    # Encodes texts into normalized float32 vectors; dot product then equals cosine similarity.
    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        # Empty input: return an empty matrix with the right width instead of calling the model.
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Only questions get the query prompt; documents (chunks, node names) are encoded as-is.
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

    # Deletes the model and clears the CUDA cache, called when a stage is done with embeddings.
    def release(self) -> None:
        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Test double: hashes words into buckets, so no model download or GPU is needed.
class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder for tests (no model, no GPU).

    Texts sharing words get similar vectors, so clustering/retrieval logic is testable.
    """

    # Only the vector size needs to be set; there is no model to load.
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    # Builds a bag-of-words count vector per text, then L2-normalizes each row.
    def encode(self, texts: list[str], query: bool = False) -> Matrix:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Hash each word to a bucket and count it, so texts sharing words end up close together.
        for i, text in enumerate(texts):
            for word in text.lower().replace("'", "").split():
                h = int(hashlib.md5(word.strip(".,!?").encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        # Normalize rows, guarding against divide-by-zero for empty texts.
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    # Nothing to free for the fake embedder; exists to satisfy the Embedder interface.
    def release(self) -> None:
        pass
