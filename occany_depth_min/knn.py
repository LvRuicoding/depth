"""Online pixel-space KNN-4 completion used as the PromptDA depth prompt."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def knn_complete_sparse_depth(sparse_depth: np.ndarray) -> np.ndarray:
    sparse = np.asarray(sparse_depth, dtype=np.float32)
    if sparse.ndim != 2:
        raise ValueError(f"sparse_depth must be 2-D, got {sparse.shape}.")
    valid = np.isfinite(sparse) & (sparse > 0.0)
    coords = np.column_stack(np.nonzero(valid)).astype(np.float64)
    values = sparse[valid].astype(np.float64)
    if coords.shape[0] < 4:
        raise ValueError(
            f"KNN-4 completion needs at least four projected pixels, got {coords.shape[0]}."
        )
    from scipy.spatial import cKDTree

    height, width = sparse.shape
    grid = np.indices((height, width), dtype=np.float64)
    queries = np.stack((grid[0].reshape(-1), grid[1].reshape(-1)), axis=1)
    tree = cKDTree(coords)
    completed = np.empty(height * width, dtype=np.float64)
    for start in range(0, queries.shape[0], 65536):
        end = min(start + 65536, queries.shape[0])
        try:
            distances, indices = tree.query(queries[start:end], k=4, workers=1)
        except TypeError:
            distances, indices = tree.query(queries[start:end], k=4)
        neighbors = values[indices]
        exact = distances[:, 0] <= 1e-12
        weights = 1.0 / np.maximum(distances, 1e-6)
        values_out = (neighbors * weights).sum(axis=1) / weights.sum(axis=1)
        values_out[exact] = neighbors[exact, 0]
        completed[start:end] = values_out
    result = completed.reshape(height, width).astype(np.float32)
    result[valid] = sparse[valid]
    if not np.isfinite(result).all() or not (result > 0.0).all():
        raise RuntimeError("KNN-4 completion produced invalid depth.")
    return result


def online_knn4_depth_from_sparse(
    sparse_depth: torch.Tensor, sample_ids: Sequence[str] | None = None
) -> torch.Tensor:
    if not isinstance(sparse_depth, torch.Tensor) or sparse_depth.ndim != 4:
        raise RuntimeError("sparse_depth must be a tensor shaped (B,N,H,W).")
    batch_size, num_views = sparse_depth.shape[:2]
    if sample_ids is not None and len(sample_ids) != batch_size:
        raise RuntimeError("sample_ids must match the batch size.")
    source = sparse_depth.detach().cpu().float()
    completed = []
    for batch_index in range(batch_size):
        views = []
        for view_index in range(num_views):
            try:
                dense = knn_complete_sparse_depth(source[batch_index, view_index].numpy())
            except (RuntimeError, ValueError) as exc:
                sample = sample_ids[batch_index] if sample_ids else f"batch[{batch_index}]"
                raise RuntimeError(
                    f"{sample}: KNN-4 completion failed for view {view_index}: {exc}"
                ) from exc
            views.append(torch.from_numpy(dense))
        completed.append(torch.stack(views))
    return torch.stack(completed).float()
