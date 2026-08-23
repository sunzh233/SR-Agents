"""Text retrieval tools for ToolQA: RetrieveAgenda, RetrieveScirex.

Uses sentence-transformers embeddings + numpy cosine similarity.
Embedding model and corpus are lazy-loaded on the first query.

Thread safety: ``TextRetriever`` instances are shared across threads
via ``get_shared_retriever()``. ``_ensure_index()`` uses double-checked
locking to prevent concurrent initialization; ``query()`` is
thread-safe (numpy matmul + ``SentenceTransformer.encode`` produce
new tensors).
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

EMBED_MODEL_NAME = os.environ.get(
    "SRAGENTS_TOOLQA_EMBED_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)
if "SRAGENTS_TOOLQA_EMBED_MODEL" not in os.environ and os.environ.get(
    "HF_HUB_OFFLINE"
):
    # Preserve the release environment's offline ModelScope lookup. Explicit
    # SRAGENTS_TOOLQA_EMBED_MODEL always takes precedence.
    _offline_candidates = [
        Path(__file__).resolve().parents[6]
        / ".cache/modelscope/sentence-transformers/all-mpnet-base-v2",
        Path.home()
        / ".cache/modelscope/sentence-transformers/all-mpnet-base-v2",
    ]
    for _candidate in _offline_candidates:
        if (_candidate / "model.safetensors").exists():
            EMBED_MODEL_NAME = str(_candidate)
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

    # Agenda and SciREX can initialize concurrently. Serialize model loading to
    # avoid duplicating the largest allocation during the first ToolQA query.
    _model_load_lock = threading.Lock()

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

    def _disk_cache_path(self) -> Path | None:
        """Return a content-addressed cache path when shared caching is enabled.

        Skill-LoRA sets ``SRAGENTS_TOOLQA_EMBED_CACHE`` to its shared run cache.
        ``TOOLQA_EMBED_CACHE_DIR`` is retained as the legacy alias. Standalone
        callers can also use ``XDG_CACHE_HOME``. With none of these configured,
        keep process-local behavior rather than writing into a home directory.
        """
        configured = (
            os.environ.get("SRAGENTS_TOOLQA_EMBED_CACHE")
            or os.environ.get("TOOLQA_EMBED_CACHE_DIR")
        )
        if configured:
            root = Path(configured).expanduser()
        else:
            xdg = os.environ.get("XDG_CACHE_HOME")
            if not xdg:
                return None
            root = Path(xdg).expanduser() / "sragents" / "toolqa-embeddings"

        corpus_hash = hashlib.sha256()
        with self.corpus_path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                corpus_hash.update(chunk)
        identity = json.dumps(
            {
                "schema": 1,
                "corpus_sha256": corpus_hash.hexdigest(),
                "text_field": self.text_field,
                "model": self.model_name,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(identity).hexdigest()
        return root / f"{key}.npy"

    @staticmethod
    def _load_cached_embeddings(path: Path, expected_rows: int) -> np.ndarray | None:
        try:
            embeddings = np.load(path, allow_pickle=False, mmap_mode="r")
        except (FileNotFoundError, OSError, ValueError):
            return None
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
            return None
        return embeddings

    @staticmethod
    def _write_cached_embeddings(path: Path, embeddings: np.ndarray) -> None:
        """Publish a complete cache atomically; partial files are never read."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            with temporary.open("wb") as destination:
                np.save(destination, embeddings, allow_pickle=False)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_index(self):
        """Lazy-load corpus and build embedding index on first use.

        Uses double-checked locking so concurrent threads don't duplicate work.
        """
        if self._embeddings is not None:
            return

        with self._init_lock:
            if self._embeddings is not None:
                return

            # Load corpus texts
            texts = []
            with open(self.corpus_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    texts.append(item[self.text_field])
            self._texts = texts

            # The model is still required for query embeddings on a cache hit.
            import sentence_transformers
            configured_device = os.environ.get("SRAGENTS_TOOLQA_EMBED_DEVICE")
            with TextRetriever._model_load_lock:
                self._model = sentence_transformers.SentenceTransformer(
                    self.model_name, device=configured_device,
                )

            cache_path = self._disk_cache_path()
            if cache_path is None:
                self._embeddings = self._build_embeddings(texts)
                return

            # A completed cache is immutable and atomically published, so it is
            # safe to read before taking the construction lock.  This also
            # avoids NFS lock contention on every evaluation shard after a
            # one-time pre-warm.
            cached = self._load_cached_embeddings(cache_path, len(texts))
            if cached is not None:
                print(f"  Loading cached embeddings from {cache_path}...")
                self._embeddings = cached
                return

            # The file lock serializes cache construction across evaluation
            # shards.  Waiting processes load the first shard's atomic result
            # instead of encoding the same 10k-document corpus again.
            import fcntl
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
            with lock_path.open("a+b") as lock:
                # Some shared filesystems turn a blocking flock collision into
                # EAGAIN.  Poll the atomically-published cache while retrying;
                # the builder normally finishes within a few minutes.
                deadline = time.monotonic() + 30 * 60
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX)
                        break
                    except BlockingIOError:
                        cached = self._load_cached_embeddings(cache_path, len(texts))
                        if cached is not None:
                            print(f"  Loading cached embeddings from {cache_path}...")
                            self._embeddings = cached
                            return
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out waiting for ToolQA embedding cache: {cache_path}"
                            )
                        time.sleep(1)
                try:
                    cached = self._load_cached_embeddings(cache_path, len(texts))
                    if cached is None:
                        embeddings = self._build_embeddings(texts)
                        self._write_cached_embeddings(cache_path, embeddings)
                        self._embeddings = embeddings
                    else:
                        print(f"  Loading cached embeddings from {cache_path}...")
                        self._embeddings = cached
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _build_embeddings(self, texts: list[str]) -> np.ndarray:
        print(f"  Encoding {len(texts)} documents with {self.model_name}...")
        embeddings = self._model.encode(texts, show_progress_bar=True)
        embeddings = np.asarray(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def query(self, query_text: str, top_k: int | None = None) -> str:
        """Return top-k most relevant documents as newline-separated text."""
        self._ensure_index()
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
