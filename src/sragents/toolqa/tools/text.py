"""Text retrieval tools for ToolQA: RetrieveAgenda, RetrieveScirex.

Uses sentence-transformers embeddings + numpy cosine similarity.
Embeddings are persisted to disk (``.embeddings/`` next to each corpus)
so subsequent processes skip the full GPU encode and load vectors
directly. The model itself is lazy-loaded only when ``query()`` is
actually called.

Thread safety: ``TextRetriever`` instances are shared across threads
via ``get_shared_retriever()``. ``_ensure_index()`` uses double-checked
locking to prevent concurrent initialization; ``query()`` is
thread-safe (numpy matmul + ``SentenceTransformer.encode`` produce
new tensors).
"""

import json
import os
import threading
from pathlib import Path

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# Offline fallback: search modelscope cache in project and home dirs
if os.environ.get("HF_HUB_OFFLINE"):
    _candidates = [
        Path(__file__).resolve().parents[6] / ".cache" / "modelscope" / "sentence-transformers" / "all-mpnet-base-v2",
        Path.home() / ".cache" / "modelscope" / "sentence-transformers" / "all-mpnet-base-v2",
    ]
    for _c in _candidates:
        if (_c / "model.safetensors").exists():
            EMBED_MODEL_NAME = str(_c)
            break

# Process-level cache: (corpus_path, text_field) -> TextRetriever (shared)
_retriever_cache: dict[tuple[str, str], "TextRetriever"] = {}
_retriever_cache_lock = threading.Lock()


def get_shared_retriever(
    corpus_path: Path,
    text_field: str,
    model_name: str = EMBED_MODEL_NAME,
    top_k: int = 3,
) -> "TextRetriever":
    """Return a shared TextRetriever, creating it on first call (thread-safe)."""
    key = (str(corpus_path), text_field)
    if key in _retriever_cache:
        return _retriever_cache[key]
    with _retriever_cache_lock:
        if key not in _retriever_cache:
            retriever = TextRetriever(corpus_path, text_field, model_name, top_k)
            retriever._ensure_index()
            _retriever_cache[key] = retriever
    return _retriever_cache[key]


class TextRetriever:
    """Semantic text retriever using sentence-transformers + cosine similarity."""

    def __init__(
        self,
        corpus_path: Path,
        text_field: str,
        model_name: str = EMBED_MODEL_NAME,
        top_k: int = 3,
    ):
        self.corpus_path = Path(corpus_path)
        self.text_field = text_field
        self.model_name = model_name
        self.top_k = top_k

        self._model = None
        self._texts: list[str] | None = None
        self._embeddings: np.ndarray | None = None
        self._init_lock = threading.Lock()

    # Module-level lock: prevents concurrent SentenceTransformer loads across
    # threads (and within the same process).  Without this, the 24-worker
    # ThreadPoolExecutor can trigger N simultaneous GPU allocations during the
    # very first query(), causing CUDA OOM or AttributeError when the model
    # stays None after a failed load.
    _model_lock = threading.Lock()

    def _ensure_model(self):
        """Lazy-load the sentence-transformers model on first query (thread-safe)."""
        if self._model is not None:
            return
        with TextRetriever._model_lock:
            # Double-check: another thread may have loaded while we waited
            if self._model is not None:
                return
            import sentence_transformers
            self._model = sentence_transformers.SentenceTransformer(self.model_name)

    def _ensure_index(self):
        """Lazy-load corpus embeddings on first use.

        Cache hit: load (embeddings, texts) from disk, model stays unloaded.
        Cache miss: read JSONL, encode, persist to disk for next run.
        """
        if self._embeddings is not None:
            return

        with self._init_lock:
            if self._embeddings is not None:
                return

            from sragents.toolqa.tools.embedding_cache import (
                cache_path_for,
                load_if_fresh,
                save_atomic,
            )
            cache_path = cache_path_for(self.corpus_path, self.model_name)

            # 1) Cache hit: load vectors + texts, skip GPU entirely
            hit = load_if_fresh(
                cache_path, self.corpus_path, self.model_name, self.text_field
            )
            if hit is not None:
                self._embeddings, self._texts = hit
                print(
                    f"  Loaded {len(self._texts)} embeddings from cache: {cache_path}"
                )
                return

            # 2) Cache miss: read JSONL, load model, encode, persist
            texts = []
            with open(self.corpus_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    texts.append(item[self.text_field])
            self._texts = texts

            self._ensure_model()
            print(f"  Encoding {len(texts)} documents with {self.model_name}...")
            self._embeddings = self._model.encode(texts, show_progress_bar=True)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = (self._embeddings / norms).astype(np.float32)

            save_atomic(
                cache_path,
                self._embeddings,
                self._texts,
                self.corpus_path,
                self.model_name,
                self.text_field,
            )
            print(f"  Saved embeddings cache: {cache_path}")

    def query(self, query_text: str, top_k: int | None = None) -> str:
        """Return top-k most relevant documents as newline-separated text."""
        self._ensure_index()
        self._ensure_model()
        k = top_k or self.top_k

        # Encode query
        query_emb = self._model.encode([query_text])
        query_norm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        query_norm = np.where(query_norm == 0, 1, query_norm)
        query_emb = query_emb / query_norm

        # Cosine similarity
        scores = (self._embeddings @ query_emb.T).squeeze()
        top_indices = np.argsort(scores)[-k:][::-1]

        results = [self._texts[i] for i in top_indices]
        return "\n".join(results)
