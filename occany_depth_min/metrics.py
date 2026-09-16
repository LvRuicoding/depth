"""Native-grid six-domain depth metrics used for validation and evaluation."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F

DATASET_NAMES = ("kitti", "ddad", "7scenes", "nyuv2", "sunrgbd", "void")

@dataclasses.dataclass(frozen=True)
class DomainProtocol:
    name: str
    min_depth: float
    max_depth: float
    min_inclusive: bool = True
    max_inclusive: bool = True
    crop: Optional[str] = None
    note: str = ""

DOMAIN_PROTOCOLS: Dict[str, DomainProtocol] = {
    "kitti": DomainProtocol("kitti", 1e-3, 80.0),
    "ddad": DomainProtocol("ddad", 1e-3, 120.0, note="same_sweep_anchor_reconstruction_occany_120m"),
    "7scenes": DomainProtocol("7scenes", 0.1, 4.0, note="raw_depth_coordinate_internal"),
    "nyuv2": DomainProtocol("nyuv2", 1e-3, 10.0, crop="nyuv2_eigen"),
    "sunrgbd": DomainProtocol("sunrgbd", 0.0, 10.0, min_inclusive=False, max_inclusive=False),
    "void": DomainProtocol("void", 0.2, 5.0, min_inclusive=False, max_inclusive=False, note="void_official_0.2_5m"),
}
_ALIASES = {"7-scenes":"7scenes", "seven_scenes":"7scenes", "sevenscenes":"7scenes", "nyu":"nyuv2", "nyu_v2":"nyuv2", "sun_rgbd":"sunrgbd", "sun-rgbd":"sunrgbd"}
METRIC_NAMES = ("abs_rel", "sq_rel", "rmse", "mae", "rmse_log", "silog", "log10", "imae", "irmse", "delta1", "delta2", "delta3")
REGION_NAMES = ("all_valid", "anchor", "non_anchor")

def canonical_dataset_name(name: str) -> str:
    value = str(name).strip().lower()
    value = _ALIASES.get(value, value)
    if value not in DOMAIN_PROTOCOLS:
        raise ValueError(
            f"Unsupported unified depth dataset {name!r}; expected one of {DATASET_NAMES}."
        )
    return value


def resolve_evaluation_split(dataset_name: str, requested_split: str) -> str:
    """Validate the released evaluation split, exposed locally as ``val``."""

    name = canonical_dataset_name(dataset_name)
    if requested_split not in ("val", "test"):
        raise ValueError(f"Evaluation split must be val or test, got {requested_split!r}.")
    if requested_split == "test":
        raise ValueError(
            f"{name} has no local test manifest in this experiment; the released "
            "evaluation split is used as val."
        )
    return requested_split


def nyuv2_eigen_crop_mask(
    height: int,
    width: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the standard NYUv2 crop, scaled from its 480x640 definition.

    The canonical half-open rectangle is rows ``[45, 471)`` and columns
    ``[41, 601)``.  Scaling the rectangle keeps this helper safe if a dataset
    stores a non-canonical native resolution.
    """

    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {(height, width)}.")
    top = int(round(45.0 * height / 480.0))
    bottom = int(round(471.0 * height / 480.0))
    left = int(round(41.0 * width / 640.0))
    right = int(round(601.0 * width / 640.0))
    top, bottom = max(0, top), min(height, max(top + 1, bottom))
    left, right = max(0, left), min(width, max(left + 1, right))
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[top:bottom, left:right] = True
    return mask


def protocol_valid_mask(
    dataset_name: str,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build a benchmark mask using GT only (prediction-independent)."""

    name = canonical_dataset_name(dataset_name)
    protocol = DOMAIN_PROTOCOLS[name]
    gt = _as_hw(gt_depth, "gt_depth").float()
    valid = _protocol_range_valid_mask(protocol, gt, valid_mask)
    if protocol.crop == "nyuv2_eigen":
        valid &= nyuv2_eigen_crop_mask(*gt.shape, device=gt.device)
    return valid


def _protocol_range_valid_mask(
    protocol: DomainProtocol,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    lower = (
        gt >= float(protocol.min_depth)
        if protocol.min_inclusive
        else gt > float(protocol.min_depth)
    )
    upper = (
        gt <= float(protocol.max_depth)
        if protocol.max_inclusive
        else gt < float(protocol.max_depth)
    )
    valid = torch.isfinite(gt) & lower & upper & (gt > 0.0)
    if valid_mask is not None:
        supplied = _as_hw(valid_mask, "valid_mask").to(device=gt.device)
        if supplied.shape != gt.shape:
            raise RuntimeError(
                f"valid_mask shape {tuple(supplied.shape)} != GT shape {tuple(gt.shape)}."
            )
        if supplied.dtype == torch.bool:
            valid &= supplied
        else:
            valid &= torch.isfinite(supplied) & (supplied > 0.5)
    return valid


def _as_hw(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise RuntimeError(f"{name} must reduce to (H,W); got {tuple(value.shape)}.")
    return value


def _int_pair(value: Any, field: str) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"resize_metadata[{field!r}] must contain two integers.")
    return int(value[0]), int(value[1])


def _int_quad(value: Any, field: str) -> tuple[int, int, int, int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"resize_metadata[{field!r}] must contain four integers.")
    return tuple(int(v) for v in value)  # type: ignore[return-value]


def restore_to_native(
    tensor: torch.Tensor,
    resize_metadata: Optional[Mapping[str, Any]],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Undo ``resize + symmetric letterbox`` for one image/view.

    ``resize_metadata`` follows the unified dataset contract:
    ``native_hw``, ``resized_hw`` and ``pad_tblr`` (top, bottom, left, right).
    Leading tensor dimensions are retained.  Boolean tensors always use
    nearest-neighbour restoration.
    """

    if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
        raise TypeError("tensor must be a torch.Tensor with at least two dimensions.")
    if resize_metadata is None:
        return tensor
    native_h, native_w = _int_pair(resize_metadata["native_hw"], "native_hw")
    resized_h, resized_w = _int_pair(resize_metadata["resized_hw"], "resized_hw")
    top, bottom, left, right = _int_quad(
        resize_metadata.get("pad_tblr", (0, 0, 0, 0)), "pad_tblr"
    )
    height, width = tensor.shape[-2:]
    if top < 0 or bottom < 0 or left < 0 or right < 0:
        raise ValueError("resize_metadata padding must be non-negative.")
    if top + resized_h + bottom != height or left + resized_w + right != width:
        raise ValueError(
            "resize metadata does not match tensor grid: "
            f"tensor={(height, width)}, resized={(resized_h, resized_w)}, "
            f"pad={(top, bottom, left, right)}."
        )
    cropped = tensor[..., top : top + resized_h, left : left + resized_w]
    if (resized_h, resized_w) == (native_h, native_w):
        return cropped
    use_nearest = mode == "nearest" or cropped.dtype == torch.bool
    if mode not in ("nearest", "bilinear"):
        raise ValueError(f"Unsupported restoration mode {mode!r}.")
    original_dtype = cropped.dtype
    flat = cropped.reshape(-1, 1, resized_h, resized_w).float()
    if use_nearest:
        restored = F.interpolate(flat, size=(native_h, native_w), mode="nearest")
    else:
        restored = F.interpolate(
            flat,
            size=(native_h, native_w),
            mode="bilinear",
            align_corners=False,
        )
    restored = restored.reshape(*cropped.shape[:-2], native_h, native_w)
    return restored > 0.5 if original_dtype == torch.bool else restored.to(original_dtype)


def restore_sparse_mask_to_native(
    mask: torch.Tensor,
    resize_metadata: Optional[Mapping[str, Any]],
) -> torch.Tensor:
    """Map sparse model-grid anchors back without inflating individual points.

    Nearest-neighbour *image* upsampling turns one sparse pixel into a block.
    Anchor metrics instead need a point mapping, so every occupied resized
    pixel is inverse-mapped to one nearest native pixel (with collision OR).
    """

    sparse = _as_hw(mask, "sparse_mask").to(dtype=torch.bool)
    if resize_metadata is None:
        return sparse
    native_h, native_w = _int_pair(resize_metadata["native_hw"], "native_hw")
    resized_h, resized_w = _int_pair(resize_metadata["resized_hw"], "resized_hw")
    top, bottom, left, right = _int_quad(
        resize_metadata.get("pad_tblr", (0, 0, 0, 0)), "pad_tblr"
    )
    if top + resized_h + bottom != sparse.shape[0] or left + resized_w + right != sparse.shape[1]:
        raise ValueError("resize metadata does not match sparse-mask grid.")
    cropped = sparse[top : top + resized_h, left : left + resized_w]
    occupied = torch.nonzero(cropped, as_tuple=False)
    restored = torch.zeros(
        (native_h, native_w), dtype=torch.bool, device=sparse.device
    )
    if occupied.numel() == 0:
        return restored
    y = torch.round(
        (occupied[:, 0].float() + 0.5) * native_h / resized_h - 0.5
    ).long().clamp_(0, native_h - 1)
    x = torch.round(
        (occupied[:, 1].float() + 0.5) * native_w / resized_w - 0.5
    ).long().clamp_(0, native_w - 1)
    restored[y, x] = True
    return restored


@dataclasses.dataclass
class _MetricSums:
    count: float = 0.0
    abs_sum: float = 0.0
    abs_rel_sum: float = 0.0
    sq_sum: float = 0.0
    sq_rel_sum: float = 0.0
    log_sum: float = 0.0
    log_sq_sum: float = 0.0
    log10_abs_sum: float = 0.0
    inv_abs_sum: float = 0.0
    inv_sq_sum: float = 0.0
    delta1_sum: float = 0.0
    delta2_sum: float = 0.0
    delta3_sum: float = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        pred = pred.detach().reshape(-1).to(dtype=torch.float64)
        gt = gt.detach().reshape(-1).to(device=pred.device, dtype=torch.float64)
        if pred.numel() == 0:
            return
        diff = pred - gt
        abs_error = diff.abs()
        log_diff = torch.log(pred) - torch.log(gt)
        inv_diff = pred.reciprocal() - gt.reciprocal()
        ratio = torch.maximum(pred / gt, gt / pred)
        self.count += float(pred.numel())
        self.abs_sum += float(abs_error.sum().item())
        self.abs_rel_sum += float((abs_error / gt).sum().item())
        self.sq_sum += float(diff.square().sum().item())
        self.sq_rel_sum += float((diff.square() / gt).sum().item())
        self.log_sum += float(log_diff.sum().item())
        self.log_sq_sum += float(log_diff.square().sum().item())
        self.log10_abs_sum += float(
            (torch.log10(pred) - torch.log10(gt)).abs().sum().item()
        )
        self.inv_abs_sum += float(inv_diff.abs().sum().item())
        self.inv_sq_sum += float(inv_diff.square().sum().item())
        self.delta1_sum += float((ratio < 1.25).sum().item())
        self.delta2_sum += float((ratio < 1.25**2).sum().item())
        self.delta3_sum += float((ratio < 1.25**3).sum().item())

    def merge(self, other: "_MetricSums") -> None:
        for field in dataclasses.fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def compute(self) -> Dict[str, float]:
        if self.count <= 0:
            return {"valid_pixels": 0.0, **{name: float("nan") for name in METRIC_NAMES}}
        n = self.count
        log_mean = self.log_sum / n
        log_sq_mean = self.log_sq_sum / n
        return {
            "valid_pixels": n,
            "abs_rel": self.abs_rel_sum / n,
            "sq_rel": self.sq_rel_sum / n,
            "rmse": math.sqrt(self.sq_sum / n),
            "mae": self.abs_sum / n,
            "rmse_log": math.sqrt(self.log_sq_sum / n),
            "silog": math.sqrt(max(log_sq_mean - log_mean * log_mean, 0.0)) * 100.0,
            "log10": self.log10_abs_sum / n,
            "imae": self.inv_abs_sum / n * 1000.0,
            "irmse": math.sqrt(self.inv_sq_sum / n) * 1000.0,
            "delta1": self.delta1_sum / n,
            "delta2": self.delta2_sum / n,
            "delta3": self.delta3_sum / n,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "_MetricSums":
        return cls(**{field.name: float(state.get(field.name, 0.0)) for field in dataclasses.fields(cls)})


class _RegionAccumulator:
    def __init__(self) -> None:
        self.pixel = _MetricSums()
        self.image_metric_sums = {name: 0.0 for name in METRIC_NAMES}
        self.image_metric_counts = {name: 0.0 for name in METRIC_NAMES}
        self.images = 0.0
        self.valid_pixels_sum = 0.0

    def update(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> None:
        if mask.shape != pred.shape or gt.shape != pred.shape:
            raise RuntimeError("Prediction, GT and region mask must share one native grid.")
        pred_values, gt_values = pred[mask], gt[mask]
        if pred_values.numel() == 0:
            return
        per_image = _MetricSums()
        per_image.update(pred_values, gt_values)
        values = per_image.compute()
        self.pixel.merge(per_image)
        self.images += 1.0
        self.valid_pixels_sum += float(pred_values.numel())
        for name in METRIC_NAMES:
            value = float(values[name])
            if math.isfinite(value):
                self.image_metric_sums[name] += value
                self.image_metric_counts[name] += 1.0

    def merge(self, other: "_RegionAccumulator") -> None:
        self.pixel.merge(other.pixel)
        self.images += other.images
        self.valid_pixels_sum += other.valid_pixels_sum
        for name in METRIC_NAMES:
            self.image_metric_sums[name] += other.image_metric_sums[name]
            self.image_metric_counts[name] += other.image_metric_counts[name]

    def compute(self) -> tuple[Dict[str, float], Dict[str, float]]:
        pixel = self.pixel.compute()
        macro = {
            "images": self.images,
            "valid_pixels_mean": (
                self.valid_pixels_sum / self.images if self.images > 0 else float("nan")
            ),
        }
        for name in METRIC_NAMES:
            count = self.image_metric_counts[name]
            macro[name] = self.image_metric_sums[name] / count if count > 0 else float("nan")
        return pixel, macro

    def state_dict(self) -> Dict[str, Any]:
        return {
            "pixel": dataclasses.asdict(self.pixel),
            "image_metric_sums": dict(self.image_metric_sums),
            "image_metric_counts": dict(self.image_metric_counts),
            "images": self.images,
            "valid_pixels_sum": self.valid_pixels_sum,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_RegionAccumulator":
        result = cls()
        result.pixel = _MetricSums.from_dict(state.get("pixel", {}))
        result.images = float(state.get("images", 0.0))
        result.valid_pixels_sum = float(state.get("valid_pixels_sum", 0.0))
        for name in METRIC_NAMES:
            result.image_metric_sums[name] = float(
                state.get("image_metric_sums", {}).get(name, 0.0)
            )
            result.image_metric_counts[name] = float(
                state.get("image_metric_counts", {}).get(name, 0.0)
            )
        return result


class _Diagnostics:
    _ADDITIVE = (
        "images",
        "evaluated_pixels",
        "nonfinite_predictions",
        "scale_images",
        "scale_min_sum",
        "scale_max_sum",
        "scale_range_sum",
        "gt_below_scale_min",
        "gt_above_scale_max",
        "relative_pixels",
        "relative_low_saturation",
        "relative_high_saturation",
        "relative_nonfinite",
    )

    def __init__(self) -> None:
        for name in self._ADDITIVE:
            setattr(self, name, 0.0)
        self.scale_min_seen = float("inf")
        self.scale_max_seen = float("-inf")
        self.scale_range_min_seen = float("inf")
        self.scale_range_max_seen = float("-inf")

    def update(
        self,
        *,
        raw_pred: torch.Tensor,
        gt: torch.Tensor,
        valid: torch.Tensor,
        relative: Optional[torch.Tensor],
        scale_min: Optional[float],
        scale_max: Optional[float],
    ) -> None:
        self.images += 1.0
        self.evaluated_pixels += float(valid.sum().item())
        self.nonfinite_predictions += float((valid & ~torch.isfinite(raw_pred)).sum().item())
        if scale_min is not None and scale_max is not None:
            lo, hi = float(scale_min), float(scale_max)
            if math.isfinite(lo) and math.isfinite(hi):
                span = hi - lo
                self.scale_images += 1.0
                self.scale_min_sum += lo
                self.scale_max_sum += hi
                self.scale_range_sum += span
                self.scale_min_seen = min(self.scale_min_seen, lo)
                self.scale_max_seen = max(self.scale_max_seen, hi)
                self.scale_range_min_seen = min(self.scale_range_min_seen, span)
                self.scale_range_max_seen = max(self.scale_range_max_seen, span)
                self.gt_below_scale_min += float((valid & (gt < lo)).sum().item())
                self.gt_above_scale_max += float((valid & (gt > hi)).sum().item())
        if relative is not None:
            if relative.shape != valid.shape:
                raise RuntimeError("relative_depth must share the restored native grid.")
            rel_valid = valid & torch.isfinite(relative)
            self.relative_pixels += float(valid.sum().item())
            self.relative_nonfinite += float((valid & ~torch.isfinite(relative)).sum().item())
            self.relative_low_saturation += float((rel_valid & (relative <= 0.01)).sum().item())
            self.relative_high_saturation += float((rel_valid & (relative >= 0.99)).sum().item())

    def merge(self, other: "_Diagnostics") -> None:
        for name in self._ADDITIVE:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.scale_min_seen = min(self.scale_min_seen, other.scale_min_seen)
        self.scale_max_seen = max(self.scale_max_seen, other.scale_max_seen)
        self.scale_range_min_seen = min(
            self.scale_range_min_seen, other.scale_range_min_seen
        )
        self.scale_range_max_seen = max(
            self.scale_range_max_seen, other.scale_range_max_seen
        )

    def compute(self) -> Dict[str, Any]:
        scale_n = self.scale_images
        pixel_n = self.evaluated_pixels
        rel_n = self.relative_pixels
        return {
            "images": self.images,
            "evaluated_pixels": pixel_n,
            "nonfinite_predictions": self.nonfinite_predictions,
            "nonfinite_prediction_rate": (
                self.nonfinite_predictions / pixel_n if pixel_n > 0 else float("nan")
            ),
            "scale": {
                "images": scale_n,
                "min_mean": self.scale_min_sum / scale_n if scale_n > 0 else float("nan"),
                "max_mean": self.scale_max_sum / scale_n if scale_n > 0 else float("nan"),
                "range_mean": self.scale_range_sum / scale_n if scale_n > 0 else float("nan"),
                "min_seen": self.scale_min_seen if scale_n > 0 else float("nan"),
                "max_seen": self.scale_max_seen if scale_n > 0 else float("nan"),
                "range_min_seen": self.scale_range_min_seen if scale_n > 0 else float("nan"),
                "range_max_seen": self.scale_range_max_seen if scale_n > 0 else float("nan"),
                "gt_below_min_pixels": self.gt_below_scale_min,
                "gt_above_max_pixels": self.gt_above_scale_max,
                "gt_below_min_rate": self.gt_below_scale_min / pixel_n if pixel_n > 0 else float("nan"),
                "gt_above_max_rate": self.gt_above_scale_max / pixel_n if pixel_n > 0 else float("nan"),
            },
            "relative_saturation": {
                "pixels": rel_n,
                "nonfinite": self.relative_nonfinite,
                "low_le_0.01": self.relative_low_saturation,
                "high_ge_0.99": self.relative_high_saturation,
                "low_rate": self.relative_low_saturation / rel_n if rel_n > 0 else float("nan"),
                "high_rate": self.relative_high_saturation / rel_n if rel_n > 0 else float("nan"),
            },
        }

    def state_dict(self) -> Dict[str, float]:
        names = (*self._ADDITIVE, "scale_min_seen", "scale_max_seen", "scale_range_min_seen", "scale_range_max_seen")
        return {name: float(getattr(self, name)) for name in names}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_Diagnostics":
        result = cls()
        for name in (*cls._ADDITIVE, "scale_min_seen", "scale_max_seen", "scale_range_min_seen", "scale_range_max_seen"):
            if name in state:
                setattr(result, name, float(state[name]))
        return result


class UnifiedDepthMetricAccumulator:
    """Streaming pixel-micro and per-image-macro metrics for one domain."""

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = canonical_dataset_name(dataset_name)
        self.protocol = DOMAIN_PROTOCOLS[self.dataset_name]
        self.regions = {name: _RegionAccumulator() for name in REGION_NAMES}
        self.nyuv2_full_grid = (
            _RegionAccumulator() if self.dataset_name == "nyuv2" else None
        )
        self.diagnostics = _Diagnostics()

    def update_image(
        self,
        pred_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        anchor_mask: Optional[torch.Tensor] = None,
        relative_depth: Optional[torch.Tensor] = None,
        scale_min: Optional[float] = None,
        scale_max: Optional[float] = None,
    ) -> "UnifiedDepthMetricAccumulator":
        raw_pred = _as_hw(pred_depth, "pred_depth").detach().float()
        gt = _as_hw(gt_depth, "gt_depth").to(device=raw_pred.device, dtype=torch.float32)
        if raw_pred.shape != gt.shape:
            raise RuntimeError(
                f"Native prediction shape {tuple(raw_pred.shape)} != GT {tuple(gt.shape)}."
            )
        valid = protocol_valid_mask(self.dataset_name, gt, valid_mask)
        lo = max(float(self.protocol.min_depth), 1e-6)
        hi = float(self.protocol.max_depth)
        # The denominator depends on GT only.  Map invalid predictions to the
        # conservative boundary so NaN/Inf can never improve reported metrics.
        pred = torch.nan_to_num(raw_pred, nan=hi, posinf=hi, neginf=lo).clamp(lo, hi)
        relative = None
        if relative_depth is not None:
            relative = _as_hw(relative_depth, "relative_depth").to(raw_pred.device).float()
        anchor = None
        if anchor_mask is not None:
            anchor = _as_hw(anchor_mask, "anchor_mask").to(raw_pred.device)
            if anchor.shape != gt.shape:
                raise RuntimeError(
                    f"Native anchor shape {tuple(anchor.shape)} != GT {tuple(gt.shape)}."
                )
            anchor = anchor if anchor.dtype == torch.bool else torch.isfinite(anchor) & (anchor > 0.5)

        self.regions["all_valid"].update(pred, gt, valid)
        if self.nyuv2_full_grid is not None:
            full_grid_valid = _protocol_range_valid_mask(
                self.protocol,
                gt,
                None if valid_mask is None else _as_hw(valid_mask, "valid_mask").to(gt.device),
            )
            self.nyuv2_full_grid.update(pred, gt, full_grid_valid)
        if anchor is not None:
            self.regions["anchor"].update(pred, gt, valid & anchor)
            self.regions["non_anchor"].update(pred, gt, valid & ~anchor)
        self.diagnostics.update(
            raw_pred=raw_pred,
            gt=gt,
            valid=valid,
            relative=relative,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        return self

    def merge(self, other: "UnifiedDepthMetricAccumulator") -> "UnifiedDepthMetricAccumulator":
        if self.dataset_name != other.dataset_name:
            raise ValueError(
                f"Cannot merge {other.dataset_name!r} metrics into {self.dataset_name!r}."
            )
        for name in REGION_NAMES:
            self.regions[name].merge(other.regions[name])
        if self.nyuv2_full_grid is not None:
            if other.nyuv2_full_grid is None:
                raise ValueError("NYUv2 full-grid accumulator is missing during merge.")
            self.nyuv2_full_grid.merge(other.nyuv2_full_grid)
        self.diagnostics.merge(other.diagnostics)
        return self

    def compute(self) -> Dict[str, Any]:
        pixel_micro: Dict[str, Any] = {}
        per_image_macro: Dict[str, Any] = {}
        for name in REGION_NAMES:
            pixel_micro[name], per_image_macro[name] = self.regions[name].compute()
        all_count = pixel_micro["all_valid"]["valid_pixels"]
        anchor_count = pixel_micro["anchor"]["valid_pixels"]
        non_anchor_count = pixel_micro["non_anchor"]["valid_pixels"]
        result: Dict[str, Any] = {
            "dataset": self.dataset_name,
            "protocol": dataclasses.asdict(self.protocol),
            "pixel_micro": pixel_micro,
            "per_image_macro": per_image_macro,
            "pixel_region_coverage": {
                "all_valid_pixels": all_count,
                "anchor_pixels": anchor_count,
                "non_anchor_pixels": non_anchor_count,
                "anchor_ratio": anchor_count / all_count if all_count > 0 else float("nan"),
                "non_anchor_ratio": non_anchor_count / all_count if all_count > 0 else float("nan"),
            },
            "diagnostics": self.diagnostics.compute(),
        }
        if self.dataset_name == "void":
            result["void_official"] = _void_official_metrics(pixel_micro, per_image_macro)
        if self.nyuv2_full_grid is not None:
            full_pixel, full_macro = self.nyuv2_full_grid.compute()
            result["nyuv2_full_grid_diagnostic"] = {
                "spatial_protocol": "full_native_valid_grid_no_eigen_crop",
                "pixel_micro": full_pixel,
                "per_image_macro": full_macro,
            }
        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "regions": {name: region.state_dict() for name, region in self.regions.items()},
            "nyuv2_full_grid": (
                None
                if self.nyuv2_full_grid is None
                else self.nyuv2_full_grid.state_dict()
            ),
            "diagnostics": self.diagnostics.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "UnifiedDepthMetricAccumulator":
        result = cls(str(state["dataset_name"]))
        for name in REGION_NAMES:
            result.regions[name] = _RegionAccumulator.from_state_dict(
                state.get("regions", {}).get(name, {})
            )
        if result.nyuv2_full_grid is not None:
            full_state = state.get("nyuv2_full_grid")
            if not isinstance(full_state, Mapping):
                raise ValueError("NYUv2 metric state is missing full-grid diagnostics.")
            result.nyuv2_full_grid = _RegionAccumulator.from_state_dict(full_state)
        result.diagnostics = _Diagnostics.from_state_dict(state.get("diagnostics", {}))
        return result


def _void_official_metrics(
    pixel_micro: Mapping[str, Mapping[str, float]],
    per_image_macro: Mapping[str, Mapping[str, float]],
) -> Dict[str, Any]:
    def convert(branch: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
        converted: Dict[str, Any] = {}
        for region, values in branch.items():
            converted[region] = {
                "mae_mm": float(values.get("mae", float("nan"))) * 1000.0,
                "rmse_mm": float(values.get("rmse", float("nan"))) * 1000.0,
                "imae_1_per_km": float(values.get("imae", float("nan"))),
                "irmse_1_per_km": float(values.get("irmse", float("nan"))),
            }
        return converted

    return {
        "valid_range_m": [0.2, 5.0],
        "range_boundary": "exclusive",
        "pixel_micro": convert(pixel_micro),
        "per_image_macro": convert(per_image_macro),
    }


def macro_average_domains(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Equal-domain summary; intentionally excludes dimensional RMSE/MAE."""

    metric_names = ("abs_rel", "silog", "delta1")
    values: Dict[str, list[float]] = {name: [] for name in metric_names}
    invalid = {name: False for name in metric_names}
    included = []
    for raw_name, result in results.items():
        name = canonical_dataset_name(raw_name)
        branch = result["per_image_macro"]["all_valid"]
        included.append(name)
        if float(branch.get("images", 0.0)) <= 0:
            for metric_name in metric_names:
                invalid[metric_name] = True
            continue
        for metric_name in metric_names:
            value = float(branch.get(metric_name, float("nan")))
            if math.isfinite(value):
                values[metric_name].append(value)
            else:
                invalid[metric_name] = True
    return {
        "weighting": "equal_dataset_of_per_image_macro",
        "datasets": included,
        **{
            name: (
                sum(items) / len(items)
                if items and not invalid[name]
                else float("nan")
            )
            for name, items in values.items()
        },
    }


def best_checkpoint_score(results: Mapping[str, Mapping[str, Any]]) -> float:
    """Return the six-domain equal-weight per-image AbsRel selection score."""

    present = {canonical_dataset_name(name) for name in results}
    missing = [name for name in DATASET_NAMES if name not in present]
    if missing:
        raise ValueError(
            "Best-checkpoint selection requires all six domains; missing "
            + ", ".join(missing)
            + "."
        )
    summary = macro_average_domains(results)
    if len(summary["datasets"]) != len(DATASET_NAMES):
        raise ValueError(
            "Best-checkpoint selection requires at least one valid image in every domain."
        )
    score = float(summary["abs_rel"])
    if not math.isfinite(score):
        raise ValueError(
            "Best-checkpoint selection requires finite per-image AbsRel in every domain."
        )
    return score
