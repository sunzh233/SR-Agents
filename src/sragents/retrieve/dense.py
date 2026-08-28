"""Dense retrieval using sentence-transformer models (BGE, Contriever, ...).

Two factories are registered:

* ``bge`` → ``BAAI/bge-base-en-v1.5`` with the BGE query prefix.
* ``contriever`` → ``facebook/contriever-msmarco``.

Any other HuggingFace model can be used by calling :class:`DenseRetriever`
directly::

    DenseRetriever(model_name_or_path="intfloat/e5-base-v2",
                   query_prefix="query: ")
"""

import time

from sragents.retrieve.base import register


class DenseRetriever:
    """Dense retriever over any ``sentence-transformers``-compatible model."""

    def __init__(
        self,
        model_name_or_path: str,
        query_prefix: str = "",
        batch_size: int = 256,
        device: str | None = None,
        dtype: str = "float32",
        query_chunk_size: int = 4096,
    ):
        self._model_path = model_name_or_path
        self._query_prefix = query_prefix
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._query_chunk_size = query_chunk_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            print(f"  Loading model: {self._model_path}")
            self._model = SentenceTransformer(self._model_path, device=self._device)
            dtypes = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            if self._dtype not in dtypes:
                raise ValueError(f"unsupported dense retrieval dtype: {self._dtype}")
            if self._dtype != "float32":
                if not str(self._model.device).startswith("cuda"):
                    raise ValueError(f"{self._dtype} dense retrieval requires a CUDA device")
                self._model.to(dtype=dtypes[self._dtype])

    def build_index(self, corpus_ids: list[str], corpus_texts: list[str]) -> None:
        """Encode the corpus once. ``query_prefix`` is **not** applied to
        documents — it is a model-side convention only applied at query
        time (see :meth:`retrieve`)."""
        self._corpus_ids = corpus_ids
        self._load_model()

        print(f"  Encoding corpus ({len(corpus_texts)} docs)...", end=" ", flush=True)
        t0 = time.time()
        self._corpus_emb = self._model.encode(
            corpus_texts,
            batch_size=self._batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        print(f"{time.time() - t0:.1f}s")

    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Encode queries (prepending ``query_prefix`` if set), compute
        cosine similarity against the indexed corpus, return top-K per query
        sorted by descending score."""
        self._load_model()
        query_texts = [self._query_prefix + q for q in queries]
        print(f"  Encoding queries ({len(query_texts)})...", end=" ", flush=True)
        t0 = time.time()
        results = []
        for start in range(0, len(query_texts), self._query_chunk_size):
            query_emb = self._model.encode(
                query_texts[start:start + self._query_chunk_size],
                batch_size=self._batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            scores = query_emb @ self._corpus_emb.T
            values, indices = scores.topk(min(top_k, len(self._corpus_ids)), dim=1)
            for row_indices, row_values in zip(
                indices.cpu().tolist(), values.float().cpu().tolist()
            ):
                results.append([
                    (self._corpus_ids[index], float(score))
                    for index, score in zip(row_indices, row_values)
                ])
        print(f"{time.time() - t0:.1f}s (GPU scoring included)")
        return results


@register("bge")
def _bge_factory(
    model_path: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 256,
    device: str | None = None,
    dtype: str = "float32",
    query_chunk_size: int = 4096,
) -> DenseRetriever:
    return DenseRetriever(
        model_name_or_path=model_path,
        query_prefix="Represent this sentence for searching relevant passages: ",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        query_chunk_size=query_chunk_size,
    )


@register("bge_m3")
def _bge_m3_factory(
    model_path: str = "BAAI/bge-m3",
    batch_size: int = 256,
    device: str | None = None,
    dtype: str = "float32",
    query_chunk_size: int = 4096,
) -> DenseRetriever:
    return DenseRetriever(
        model_name_or_path=model_path,
        query_prefix="",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        query_chunk_size=query_chunk_size,
    )


@register("contriever")
def _contriever_factory(
    model_path: str = "facebook/contriever-msmarco",
    batch_size: int = 256,
    device: str | None = None,
    dtype: str = "float32",
    query_chunk_size: int = 4096,
) -> DenseRetriever:
    return DenseRetriever(
        model_name_or_path=model_path,
        query_prefix="",
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        query_chunk_size=query_chunk_size,
    )
