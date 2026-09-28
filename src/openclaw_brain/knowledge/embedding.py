"""Embedding generation — lazy-loaded sentence-transformers singleton.

Used by the ingestion pipeline (store node embeddings) and at query time
(find semantically similar concepts even when names differ).

The model is loaded on first use and cached in process memory.
Typical load time: ~1-2s on M-series Mac.  Per-call: <1ms.

Supports instruction-asymmetric models (e.g. Qwen3-Embedding): pass
``is_query=True`` with a ``query_instruction`` so queries get the
instruction prefix while documents (node name+definition) are encoded
plain.  ``truncate_dim`` enables MRL truncation (with renormalization)
for models whose native dimension exceeds the index dimension.  With the
defaults (no instruction, truncate_dim=0) behavior is identical to the
historical MiniLM path.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None
_model_name: str = ""
_model_device: "str | None" = None

# Small LRU over encoded vectors — concept mentions repeat heavily across
# chunks of the same document, and consolidation re-encodes node names.
_CACHE_MAX = 4096
_cache: "OrderedDict[tuple[str, bool, int, str], list[float]]" = OrderedDict()


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """Return the cached sentence-transformer model, loading it on first call.

    Honors OPENCLAW_EMBED_DEVICE (e.g. "cpu") to pin the torch device. Pinning to
    cpu keeps embedding off the GPU when another Metal workload (a local LLM server
    such as oMLX) is running concurrently — two saturating Metal workloads can
    starve WindowServer and trigger a watchdog panic on macOS.
    """
    global _model, _model_name, _model_device
    device = os.environ.get("OPENCLAW_EMBED_DEVICE") or None
    if _model is None or _model_name != model_name or _model_device != device:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s (device=%s)", model_name, device or "auto")
        offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}
        _model = SentenceTransformer(model_name, device=device, local_files_only=offline)
        _model_name = model_name
        _model_device = device
        _cache.clear()  # vectors from another model are not comparable
    return _model


def _truncate_renorm(vec: list[float], truncate_dim: int) -> list[float]:
    """MRL truncation: cut to truncate_dim and re-normalize to unit length."""
    if truncate_dim <= 0 or len(vec) <= truncate_dim:
        return vec
    cut = vec[:truncate_dim]
    norm = math.sqrt(sum(x * x for x in cut))
    if norm == 0:
        return cut
    return [x / norm for x in cut]


def _cache_get(key: tuple[str, bool, int, str]) -> list[float] | None:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: tuple[str, bool, int, str], vec: list[float]) -> None:
    _cache[key] = vec
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


def encode(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    is_query: bool = False,
    query_instruction: str = "",
    truncate_dim: int = 0,
) -> list[float]:
    """Encode a single text string into a vector.

    is_query + query_instruction: instruction-asymmetric models prefix
    queries only; documents are encoded plain.  No-ops when the
    instruction is empty (MiniLM behavior).
    """
    if not text or not text.strip():
        return []
    prefixed = f"{query_instruction}{text}" if (is_query and query_instruction) else text
    key = (model_name, is_query and bool(query_instruction), truncate_dim, prefixed)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    model = get_embedder(model_name)
    vec = model.encode(prefixed, convert_to_numpy=True).tolist()
    vec = _truncate_renorm(vec, truncate_dim)
    _cache_put(key, vec)
    return vec


def encode_batch(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    *,
    is_query: bool = False,
    query_instruction: str = "",
    truncate_dim: int = 0,
) -> list[list[float]]:
    """Encode a list of texts into vectors (more efficient than repeated encode())."""
    prefix = query_instruction if (is_query and query_instruction) else ""

    # Serve cache hits; collect misses for one batched model call.
    results: list[list[float] | None] = []
    miss_texts: list[str] = []
    miss_keys: list[tuple[str, bool, int, str]] = []
    for t in texts:
        if not t or not t.strip():
            results.append([])
            continue
        prefixed = f"{prefix}{t}"
        key = (model_name, bool(prefix), truncate_dim, prefixed)
        cached = _cache_get(key)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)
            miss_texts.append(prefixed)
            miss_keys.append(key)

    if miss_texts:
        model = get_embedder(model_name)
        vectors = model.encode(miss_texts, convert_to_numpy=True).tolist()
        vec_iter = iter(vectors)
        key_iter = iter(miss_keys)
        for i, r in enumerate(results):
            if r is None:
                vec = _truncate_renorm(next(vec_iter), truncate_dim)
                _cache_put(next(key_iter), vec)
                results[i] = vec
    return results  # type: ignore[return-value]
