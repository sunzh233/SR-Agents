"""磁盘缓存 ToolQA TextRetriever 的语料向量，避免每次进程启动都重 encode。

缓存路径：``{corpus_path}.parent / ".embeddings" / {stem}.{model_short}.npz``

文件格式（``np.savez``，不开 pickle）：
- ``embeddings``   (N, D) float32，已行归一化
- ``corpus_texts`` (N,)   object/str，命中时省去重新读 JSONL
- ``manifest``     scalar JSON 字符串，含 staleness 信号

staleness 检查：先比 mtime_ns + size_bytes（O(1)），不一致再算 SHA-256 前 16 字符。
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

__all__ = [
    "cache_path_for",
    "load_if_fresh",
    "save_atomic",
]


def _model_short(model_name: str) -> str:
    """从 HF 模型名或本地路径推导出文件名安全的短名。"""
    last = model_name.rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", last) or "model"


def cache_path_for(corpus_path: Path, model_name: str) -> Path:
    """根据语料路径和模型名推导 .npz 缓存路径。"""
    cache_dir = corpus_path.parent / ".embeddings"
    return cache_dir / f"{corpus_path.stem}.{_model_short(model_name)}.npz"


def _sha256_first16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _build_manifest(
    corpus_path: Path,
    model_name: str,
    text_field: str,
    embeddings: np.ndarray,
) -> dict:
    stat = corpus_path.stat()
    return {
        "model_name": model_name,
        "text_field": text_field,
        "n_docs": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "normalized": True,
        "corpus_mtime_ns": stat.st_mtime_ns,
        "corpus_size_bytes": stat.st_size,
        "corpus_sha256_first16": _sha256_first16(corpus_path),
    }


def _manifest_matches(
    manifest: dict,
    corpus_path: Path,
    model_name: str,
    text_field: str,
) -> bool:
    if manifest.get("model_name") != model_name:
        return False
    if manifest.get("text_field") != text_field:
        return False
    try:
        stat = corpus_path.stat()
    except OSError:
        return False
    # Fast path: mtime + size both unchanged → assume content unchanged, skip sha256.
    # If either differs, fall through to authoritative sha256 check (file may have
    # been touched but not actually modified).
    if (
        manifest.get("corpus_mtime_ns") == stat.st_mtime_ns
        and manifest.get("corpus_size_bytes") == stat.st_size
    ):
        return True
    return _sha256_first16(corpus_path) == manifest.get("corpus_sha256_first16")


def load_if_fresh(
    cache_path: Path,
    corpus_path: Path,
    model_name: str,
    text_field: str,
) -> tuple[np.ndarray, list[str]] | None:
    """读缓存；不存在 / 损坏 / staleness 不匹配则返回 None。

    ``allow_pickle=True`` 是因为 ``corpus_texts`` 存的是变长字符串数组（object dtype）。
    缓存文件由本模块的 ``save_atomic`` 写入、放在可信的数据目录下，无不可信来源风险。
    """
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            manifest_raw = data["manifest"]
            manifest = json.loads(str(manifest_raw))
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            texts = [str(t) for t in data["corpus_texts"].tolist()]
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as e:
        print(f"  [cache] Failed to read {cache_path}: {e}; will rebuild")
        return None

    if not _manifest_matches(manifest, corpus_path, model_name, text_field):
        print(f"  [cache] Stale cache at {cache_path}; will rebuild")
        return None

    return embeddings, texts


def save_atomic(
    cache_path: Path,
    embeddings: np.ndarray,
    corpus_texts: list[str],
    corpus_path: Path,
    model_name: str,
    text_field: str,
) -> None:
    """原子写：先写 .tmp.npz，再 os.replace 成最终名。

    np.savez 会给不以 .npz 结尾的文件名强制追加 .npz，所以 tmp 名必须以 .npz 结尾。
    cache_path.stem 保留 ``{stem}.{model_short}`` 部分（model_short 里的 . 不算后缀分隔）。
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(corpus_path, model_name, text_field, embeddings)
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(
        tmp,
        embeddings=embeddings.astype(np.float32, copy=False),
        corpus_texts=np.asarray(corpus_texts, dtype=object),
        manifest=np.array(json.dumps(manifest)),
    )
    os.replace(tmp, cache_path)
