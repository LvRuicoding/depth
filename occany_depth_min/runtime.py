"""Shared model invocation and validation runtime."""
from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.distributed as dist

from .metrics import (
    UnifiedDepthMetricAccumulator,
    canonical_dataset_name,
    restore_sparse_mask_to_native,
    restore_to_native,
)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def forward_batch(
    model: torch.nn.Module, batch: Mapping[str, Any], device: torch.device
) -> Mapping[str, Any]:
    grid = {
        key: move_to_device(batch[key], device)
        for key in ("fusion_vox_origin", "fusion_vox_size", "fusion_vox_grid")
    }
    kwargs: Dict[str, Any] = {"grid_config": grid}
    if "knn_depth" in batch:
        kwargs["knn_depth"] = move_to_device(batch["knn_depth"], device)
    return model(
        move_to_device(batch["views"], device),
        move_to_device(batch["T_target_from_refcam"], device),
        move_to_device(batch["points_per_frame"], device),
        move_to_device(batch["T_cam_from_velo"], device),
        move_to_device(batch["K_per_frame"], device),
        move_to_device(batch["image_hw"], device),
        **kwargs,
    )


def _item(value: Any, index: int, view_index: int = 0) -> torch.Tensor:
    item = value[index]
    if not isinstance(item, torch.Tensor):
        item = torch.as_tensor(item)
    if item.ndim >= 3 and item.shape[0] == 1:
        item = item[view_index]
    return item


def _scalar(value: Any, index: int, view_index: int = 0) -> float | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    item = value[index]
    if item.numel() == 0:
        return None
    if item.ndim > 0 and item.shape[0] > view_index:
        item = item[view_index]
    return float(item.detach().float().reshape(-1)[0].item())


def update_metrics(
    accumulator: UnifiedDepthMetricAccumulator,
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> None:
    pred_batch = outputs["dense_depth"]
    relative_batch = outputs.get("relative_depth")
    for index in range(pred_batch.shape[0]):
        metadata_value = batch.get("resize_metadata")
        metadata = (
            None
            if metadata_value is None
            else metadata_value
            if isinstance(metadata_value, Mapping)
            else metadata_value[index]
        )
        pred = restore_to_native(pred_batch[index, 0], metadata, mode="bilinear")
        relative = (
            None
            if relative_batch is None
            else restore_to_native(relative_batch[index, 0], metadata, mode="bilinear")
        )
        gt = _item(batch.get("native_dense_depth", batch["dense_depth"]), index)
        valid = _item(
            batch.get("native_valid_mask", batch["dense_depth_pixel_mask"]), index
        )
        if "native_sparse_depth_mask" in batch:
            anchor = _item(batch["native_sparse_depth_mask"], index).to(
                device=pred.device, dtype=torch.bool
            )
        else:
            anchor = restore_sparse_mask_to_native(
                _item(batch["sparse_depth_mask"], index), metadata
            )
        accumulator.update_image(
            pred,
            gt.to(pred.device),
            valid_mask=valid.to(pred.device),
            anchor_mask=anchor,
            relative_depth=relative,
            scale_min=_scalar(outputs.get("scale_min"), index),
            scale_max=_scalar(outputs.get("scale_max"), index),
        )


def _autocast(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _merge(accumulator: UnifiedDepthMetricAccumulator) -> UnifiedDepthMetricAccumulator:
    if not dist.is_initialized():
        return accumulator
    states: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(states, accumulator.state_dict())
    merged = UnifiedDepthMetricAccumulator(accumulator.dataset_name)
    for state in states:
        merged.merge(UnifiedDepthMetricAccumulator.from_state_dict(state))
    return merged


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    dataset_name: str,
    device: torch.device,
    *,
    max_batches: int = 0,
    print_freq: int = 20,
) -> Dict[str, Any]:
    was_training = model.training
    model.eval()
    forward_model = model.module if hasattr(model, "module") else model
    accumulator = UnifiedDepthMetricAccumulator(canonical_dataset_name(dataset_name))
    batches = 0
    started = time.time()
    try:
        for step, batch in enumerate(loader):
            if max_batches > 0 and step >= max_batches:
                break
            with _autocast(device):
                outputs = forward_batch(forward_model, batch, device)
            update_metrics(accumulator, outputs, batch)
            batches += 1
            if print_freq > 0 and batches % print_freq == 0:
                values = accumulator.compute()["per_image_macro"]["all_valid"]
                print(
                    f"[{dataset_name} {batches}] images={values['images']:.0f} "
                    f"abs_rel={values['abs_rel']:.6f} delta1={values['delta1']:.6f}",
                    flush=True,
                )
    finally:
        model.train(was_training)
    accumulator = _merge(accumulator)
    total_batches = batches
    if dist.is_initialized():
        count = torch.tensor(batches, device=device, dtype=torch.long)
        dist.all_reduce(count)
        total_batches = int(count.item())
    result = accumulator.compute()
    result.update(
        {
            "prediction_mode": "relative_online_knn_minmax",
            "num_batches": total_batches,
            "local_num_batches": batches,
            "elapsed_seconds": time.time() - started,
        }
    )
    return result
