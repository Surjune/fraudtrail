"""Turning text into vectors, locally.

The model runs on this machine through ONNX: no key, no cost, no request leaving the
laptop, and nothing to fail at demo time. Its 384 dimensions are what the vector
attributes in graph/vectors.gsql are declared with, so the two cannot drift apart
silently — a mismatch raises here rather than producing a search that quietly finds
nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

log = logging.getLogger(__name__)

# graph/vectors.gsql declares DIMENSION=384 for every vector attribute.
DIMENSIONS = 384

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# 0 leaves the thread count to the runtime, which is about five times faster here than
# pinning it to one.
THREADS = 0

# How many texts the model holds at once. The default of 256 dies without an exception
# partway through a corpus this size on a laptop, so the work goes through in bounded
# pieces; long runs are also split across processes, one slice each.
EMBED_BATCH = 64

# bge asks for this prefix on the query side only: it is what the model was trained with
# for retrieval, and it measurably improves which case comes back first.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingError(RuntimeError):
    """Text could not be embedded, or the model returned the wrong shape."""


class Embedder(Protocol):
    def documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    """fastembed over ONNX. The first call downloads the model once and caches it."""

    def __init__(self, model_name: str = DEFAULT_MODEL, threads: int = THREADS) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise EmbeddingError("fastembed is not installed") from exc
        log.info("loading embedding model %s", model_name)
        self._model = TextEmbedding(model_name, threads=threads or None)
        self._name = model_name

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = [
                [float(x) for x in vector]
                for vector in self._model.embed(list(texts), batch_size=EMBED_BATCH)
            ]
        except Exception as exc:
            raise EmbeddingError(f"{self._name} could not embed {len(texts)} texts: {exc}") from exc
        for vector in vectors:
            if len(vector) != DIMENSIONS:
                raise EmbeddingError(
                    f"{self._name} returns {len(vector)} dimensions; the graph's vector "
                    f"attributes are declared with {DIMENSIONS}"
                )
        return vectors

    def documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def query(self, text: str) -> list[float]:
        vectors = self._embed([QUERY_PREFIX + text])
        if not vectors:
            raise EmbeddingError("an empty query cannot be embedded")
        return vectors[0]
