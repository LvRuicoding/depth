"""Retained DA3-small models and DA3-Base KITTI token-fusion variants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .da3 import vit_base, vit_small
from .depth_head import SingleScaleDPTDepthHead
from .initialization import (
    restore_da3_base_head_rng_state,
    restore_reference_head_rng_state,
)
from .projection import projected_lidar_sparse_depth
from .voxel_encoder import PatchDepthBinFeatureEncoder


MODEL_CLASS = (
    "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNN"
    "PromptDAScaledUnified6Model"
)
FUSION_CONTRACT = (
    "lingbot_sparse_log_cat_patchdepth4m_voxel_cat_token_da3_"
    "rgb0_depth1_voxel2_layer11_softplus_metric_v1"
)
SCALE_CONTRACT = "online_knn4_minmax_sigmoid_metric_v1"
ONLINE_KNN_CONTRACT = (
    "unified_sparse_map_pixel_euclidean_inverse_distance_k4_online_no_cache_v1"
)
KITTI_ONLINE_KNN_CONTRACT = (
    "kitti_stage1_lidar_zbuffer_pixel_euclidean_inverse_distance_k4_"
    "online_no_cache_v1"
)
DPT_PROMPT_CONTRACT = "online_knn4_per_frame_minmax_promptda_dpt4_scaled_v1"
INITIALIZATION_CONTRACT = (
    "da3_small_only_random_voxel_prefusion_promptda_dpt_"
    "online_knn_scaled_unified6_v1"
)
DEFAULT_MODEL_VARIANT = "prefusion"
POSTFUSION_MODEL_VARIANT = "postfusion"
DA3_BASE_MODEL_VARIANTS = (
    "image",
    "depth",
    "voxel",
    "voxel_depth",
    "image_scaled",
    "depth_scaled",
    "voxel_scaled",
    "voxel_depth_scaled",
)
MODEL_VARIANTS = (
    DEFAULT_MODEL_VARIANT,
    POSTFUSION_MODEL_VARIANT,
    *DA3_BASE_MODEL_VARIANTS,
)
DIRECT_SCALE_CONTRACT = "direct_softplus_metric_v1"


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    model_class: str
    experiment: str
    fusion_contract: str
    initialization_contract: str
    da3_model_name: str = "da3-small"
    native_dim: int = 384
    token_dim: int = 768
    uses_depth_tokens: bool = True
    uses_voxel_tokens: bool = True
    scaled: bool = True
    prediction_mode: str = "relative_online_knn_minmax"
    depth_scale_contract: str = SCALE_CONTRACT
    online_knn_contract: Optional[str] = ONLINE_KNN_CONTRACT
    dpt_prompt_contract: Optional[str] = DPT_PROMPT_CONTRACT


MODEL_SPECS = {
    DEFAULT_MODEL_VARIANT: ModelSpec(
        variant=DEFAULT_MODEL_VARIANT,
        model_class=MODEL_CLASS,
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "promptda_scaled_unified6"
        ),
        fusion_contract=FUSION_CONTRACT,
        initialization_contract=INITIALIZATION_CONTRACT,
    ),
    POSTFUSION_MODEL_VARIANT: ModelSpec(
        variant=POSTFUSION_MODEL_VARIANT,
        model_class=(
            "Stage1DepthPatchDepth4mVoxelDepthDualWindowPostFusionOnlyOnlineKNN"
            "PromptDAScaledPreAlignedUnified6Model"
        ),
        experiment=(
            "depth_patchdepth4m_voxeldepth_dualwindow_postfusion_only_"
            "promptda_scaled_prefusion_aligned_unified6"
        ),
        fusion_contract=(
            "da3_cat_localglobal_patchdepth4m_voxel_shareddual_shift02_sparse_log_"
            "depth_patch_embed_shareddual_shift02_metric_v4"
        ),
        initialization_contract=(
            "da3_small_only_seeded_logdepth_patchdepth4m_voxel_dualwindow_"
            "promptda_dpt_online_knn_scaled_unified6_v1"
        ),
    ),
}

KITTI_MODEL_SPECS = {
    DEFAULT_MODEL_VARIANT: ModelSpec(
        variant=DEFAULT_MODEL_VARIANT,
        model_class=MODEL_CLASS,
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "promptda_scaled_kitti"
        ),
        fusion_contract=FUSION_CONTRACT,
        initialization_contract=(
            "da3_small_only_random_lingbot_voxel_promptda_last_kitti_v1"
        ),
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
    POSTFUSION_MODEL_VARIANT: ModelSpec(
        variant=POSTFUSION_MODEL_VARIANT,
        model_class=(
            "Stage1DepthPatchDepth4mVoxelDepthDualWindowPostFusionOnlyOnlineKNN"
            "PromptDAScaledPreAlignedKITTIModel"
        ),
        experiment=(
            "depth_patchdepth4m_voxeldepth_dualwindow_postfusion_only_"
            "promptda_scaled_prefusion_aligned"
        ),
        fusion_contract=(
            "da3_cat_localglobal_patchdepth4m_voxel_shareddual_shift02_sparse_log_"
            "depth_patch_embed_shareddual_shift02_metric_v4"
        ),
        initialization_contract=(
            "da3_small_only_seeded_logdepth_patchdepth4m_voxel_dualwindow_"
            "promptda_dpt_online_knn_scaled_kitti_v1"
        ),
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
    "image": ModelSpec(
        variant="image",
        model_class=(
            "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusion"
            "RGBOnlyDirectMetricModel"
        ),
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "rgb_only_direct_metric"
        ),
        fusion_contract=(
            "da3_rgb_only_no_depth_no_voxel_token_layer11_softplus_metric_v1"
        ),
        initialization_contract="da3_base_rgb_only_direct_metric_kitti_v1",
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_depth_tokens=False,
        uses_voxel_tokens=False,
        scaled=False,
        prediction_mode="direct_metric",
        depth_scale_contract=DIRECT_SCALE_CONTRACT,
        online_knn_contract=None,
        dpt_prompt_contract=None,
    ),
    "depth": ModelSpec(
        variant="depth",
        model_class="Stage1DepthLingBotDA3LastDirectMetricModel",
        experiment="depth_lingbot_da3_last_direct_metric",
        fusion_contract=(
            "lingbot_sparse_log_cat_token_da3_rgb0_depth1_"
            "layer11_softplus_metric_v1"
        ),
        initialization_contract="da3_base_depth_direct_metric_kitti_v1",
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_voxel_tokens=False,
        scaled=False,
        prediction_mode="direct_metric",
        depth_scale_contract=DIRECT_SCALE_CONTRACT,
        online_knn_contract=None,
        dpt_prompt_contract=None,
    ),
    "voxel": ModelSpec(
        variant="voxel",
        model_class=(
            "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusion"
            "RGBVoxelDirectMetricModel"
        ),
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "rgbvoxel_direct_metric"
        ),
        fusion_contract=(
            "patchdepth4m_voxel_cat_token_da3_rgb0_voxel2_no_depth_token_"
            "layer11_softplus_metric_v1"
        ),
        initialization_contract="da3_base_rgbvoxel_direct_metric_kitti_v1",
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_depth_tokens=False,
        scaled=False,
        prediction_mode="direct_metric",
        depth_scale_contract=DIRECT_SCALE_CONTRACT,
        online_knn_contract=None,
        dpt_prompt_contract=None,
    ),
    "voxel_depth": ModelSpec(
        variant="voxel_depth",
        model_class=(
            "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionDirectMetricModel"
        ),
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_direct_metric"
        ),
        fusion_contract=(
            "lingbot_sparse_log_cat_patchdepth4m_voxel_cat_token_da3_"
            "rgb0_depth1_voxel2_layer11_softplus_metric_v1"
        ),
        initialization_contract="da3_base_depth_voxel_direct_metric_kitti_v1",
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        scaled=False,
        prediction_mode="direct_metric",
        depth_scale_contract=DIRECT_SCALE_CONTRACT,
        online_knn_contract=None,
        dpt_prompt_contract=None,
    ),
    "image_scaled": ModelSpec(
        variant="image_scaled",
        model_class="DA3BaseRGBOnlyOnlineKNNPromptDAScaledKITTIModel",
        experiment="depth_lingbot_da3_last_rgb_only_promptda_scaled_kitti",
        fusion_contract=(
            "da3_rgb_only_no_depth_no_voxel_token_layer11_promptda_scaled_v1"
        ),
        initialization_contract=(
            "da3_base_only_random_lingbot_rgb_only_promptda_last_kitti_v1"
        ),
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_depth_tokens=False,
        uses_voxel_tokens=False,
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
    "depth_scaled": ModelSpec(
        variant="depth_scaled",
        model_class=(
            "Stage1DepthLingBotDA3LastDepthImageOnlineKNN"
            "PromptDAScaledUnified6Model"
        ),
        experiment="depth_lingbot_da3_last_promptda_scaled_kitti",
        fusion_contract=(
            "lingbot_sparse_log_cat_token_da3_rgb0_depth1_"
            "layer11_softplus_metric_v1"
        ),
        initialization_contract=(
            "da3_base_only_random_lingbot_promptda_last_kitti_v1"
        ),
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_voxel_tokens=False,
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
    "voxel_scaled": ModelSpec(
        variant="voxel_scaled",
        model_class=(
            "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBVoxel"
            "OnlineKNNPromptDAScaledUnified6Model"
        ),
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "rgbvoxel_promptda_scaled_kitti"
        ),
        fusion_contract=(
            "patchdepth4m_voxel_cat_token_da3_rgb0_voxel2_no_depth_token_"
            "layer11_softplus_metric_v1"
        ),
        initialization_contract=(
            "da3_base_only_random_lingbot_rgbvoxel_promptda_last_kitti_v1"
        ),
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        uses_depth_tokens=False,
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
    "voxel_depth_scaled": ModelSpec(
        variant="voxel_depth_scaled",
        model_class=(
            "Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNN"
            "PromptDAScaledUnified6Model"
        ),
        experiment=(
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_"
            "promptda_scaled_kitti"
        ),
        fusion_contract=(
            "lingbot_sparse_log_cat_patchdepth4m_voxel_cat_token_da3_"
            "rgb0_depth1_voxel2_layer11_softplus_metric_v1"
        ),
        initialization_contract=(
            "da3_base_only_random_lingbot_voxel_promptda_last_kitti_v1"
        ),
        da3_model_name="da3-base",
        native_dim=768,
        token_dim=1536,
        online_knn_contract=KITTI_ONLINE_KNN_CONTRACT,
    ),
}


def get_model_spec(variant: str, dataset: str = "unified6") -> ModelSpec:
    if dataset not in ("unified6", "kitti"):
        raise ValueError("dataset must be 'unified6' or 'kitti'.")
    specs = KITTI_MODEL_SPECS if dataset == "kitti" else MODEL_SPECS
    try:
        return specs[str(variant)]
    except KeyError as error:
        raise ValueError(
            f"Model variant {variant!r} is not available for {dataset}; "
            f"expected one of {tuple(specs)}."
        ) from error


class _DinoV2(nn.Module):
    def __init__(self, model_name: str = "da3-small") -> None:
        super().__init__()
        builders = {"da3-small": vit_small, "da3-base": vit_base}
        try:
            self.pretrained = builders[str(model_name)]()
        except KeyError as error:
            raise ValueError(f"Unsupported DA3 model name: {model_name!r}.") from error


class _DA3Net(nn.Module):
    def __init__(self, model_name: str = "da3-small") -> None:
        super().__init__()
        self.backbone = _DinoV2(model_name)


class _DA3Wrapper(nn.Module):
    """Only the state-bearing DA3 hierarchy needed by the experiment."""

    def __init__(self, model_name: str = "da3-small") -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.model = _DA3Net(model_name)

    def _get_pretrained_backbone(self) -> nn.Module:
        return self.model.backbone.pretrained

    def get_backbone_metadata(self) -> Dict[str, object]:
        native_dim = int(self._get_pretrained_backbone().embed_dim)
        return {
            "token_dim": native_dim,
            "feature_dim": native_dim * 2,
            "out_layers": (5, 7, 9, 11),
            "total_layers": 12,
            "cat_token": True,
        }


class DA3Backbone(nn.Module):
    def __init__(
        self,
        *,
        backbone_dtype: torch.dtype = torch.bfloat16,
        freeze: bool = False,
        model_name: str = "da3-small",
    ) -> None:
        super().__init__()
        self.backbone_dtype = backbone_dtype
        self.freeze = bool(freeze)
        self.model_name = str(model_name)
        self.model = _DA3Wrapper(model_name)
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.set_frozen(freeze)

    def load_base_checkpoint(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir) / "model.safetensors"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {self.model_name} model.safetensors: {path}"
            )
        from safetensors.torch import load_file

        all_weights = load_file(str(path), device="cpu")
        weights = {
            key: value
            for key, value in all_weights.items()
            if key.startswith("model.backbone.pretrained.")
        }
        status = self.model.load_state_dict(weights, strict=True)
        if status.missing_keys or status.unexpected_keys:
            raise RuntimeError(f"Invalid {self.model_name} checkpoint: {status}")
        self.set_frozen(self.freeze)

    def set_frozen(self, freeze: bool) -> None:
        self.freeze = bool(freeze)
        for parameter in self.model.parameters():
            parameter.requires_grad = not self.freeze
        if self.freeze:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def stack_images(self, views: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        images = torch.stack([view["img"] for view in views], dim=1).float()
        images = (images + 1.0) * 0.5
        return (images - self._imagenet_mean.to(images)) / self._imagenet_std.to(images)


def _valid_depth_patch_mask(valid_pixels: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = valid_pixels.shape
    if channels != 1 or height % patch_size or width % patch_size:
        raise RuntimeError("Sparse-depth validity map is incompatible with patching.")
    return (
        valid_pixels.reshape(
            batch,
            1,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        .any(dim=3)
        .any(dim=-1)
        .squeeze(1)
    )


def _resolve_image_hw(
    views: List[Dict[str, torch.Tensor]],
    image_hw: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    batch_size = int(views[0]["img"].shape[0])
    if image_hw is None:
        height, width = views[0]["img"].shape[-2:]
        return torch.tensor([height, width], device=device).view(1, 2).expand(
            batch_size, 2
        )
    resolved = image_hw.to(device=device, dtype=torch.long)
    if tuple(resolved.shape) != (batch_size, 2):
        raise RuntimeError(
            f"image_hw must be (B,2), got {tuple(resolved.shape)}."
        )
    return resolved


def _grid_config(
    config: Optional[Dict[str, object]], device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[int, int, int]]]:
    if config is None:
        return None, None, None
    origin = config.get("fusion_vox_origin")
    size = config.get("fusion_vox_size")
    grid = config.get("fusion_vox_grid")
    if isinstance(grid, torch.Tensor):
        if grid.ndim == 2:
            if grid.shape[0] > 1 and not torch.equal(grid, grid[:1].expand_as(grid)):
                raise RuntimeError("fusion_vox_grid must be identical within a batch.")
            grid = grid[0]
        grid = tuple(int(value) for value in grid.detach().cpu().tolist())
    elif grid is not None:
        grid = tuple(int(value) for value in grid)
    return (
        None if origin is None else torch.as_tensor(origin).to(device=device),
        None if size is None else torch.as_tensor(size).to(device=device),
        grid,
    )


class TargetDepthModel(nn.Module):
    """Released DA3-small pre-fusion architecture."""

    fusion_contract = FUSION_CONTRACT
    depth_scale_contract = SCALE_CONTRACT
    online_knn_contract = ONLINE_KNN_CONTRACT
    dpt_prompt_contract = DPT_PROMPT_CONTRACT
    initialization_contract = INITIALIZATION_CONTRACT
    prediction_mode = "relative_online_knn_minmax"
    da3_model_name = "da3-small"

    def __init__(
        self,
        *,
        da3_checkpoint: str | Path | None = None,
        backbone_dtype: torch.dtype = torch.bfloat16,
        freeze_backbone: bool = False,
        load_base: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = 14
        self.num_views = 1
        self.native_dim = 384
        self.feature_dim = 768
        self.uses_depth_tokens = True
        self.uses_voxel_tokens = True
        self.scaled = True
        self.backbone = DA3Backbone(
            backbone_dtype=backbone_dtype, freeze=freeze_backbone
        )
        if load_base:
            if da3_checkpoint is None:
                raise ValueError("da3_checkpoint is required when load_base=True.")
            self.backbone.load_base_checkpoint(da3_checkpoint)

            # Match the discarded native DA3 modules' RNG consumption so a
            # fresh seed-0 training run starts from the released initialization.
            restore_reference_head_rng_state()

        self.dense_depth_head = SingleScaleDPTDepthHead(
            token_dim=768,
            patch_size=14,
            features=128,
            initial_depth=10.0,
            prompt_depth_enabled=True,
            prompt_depth_scale="per_frame_minmax",
            depth_output_mode="normalized_sigmoid",
            refinement_style="promptda",
        )
        vit = self.backbone.model._get_pretrained_backbone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.depth_patch_embed = type(vit.patch_embed)(
                img_size=(168, 518), patch_size=14, in_chans=1, embed_dim=384
            )
        self.voxel_token_encoder = PatchDepthBinFeatureEncoder(
            d_out=384,
            H_t=12,
            W_t=37,
            patch_size=14,
            depth_bin_size=4.0,
            vox_origin=(-60.0, -5.0, 0.0),
            vox_size=(0.4, 0.4, 0.4),
            vox_grid=(300, 25, 300),
            d_token=128,
            hidden=64,
            pe_num_freqs=8,
            dynamic_image_size=True,
        )

    def _sparse_depth_tokens(
        self,
        images: torch.Tensor,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = sparse_mask.bool() & torch.isfinite(sparse_depth) & (sparse_depth > 0.01)
        log_depth = torch.where(
            valid,
            torch.log(sparse_depth.clamp_min(0.01)),
            torch.zeros_like(sparse_depth),
        )
        log_depth = torch.nan_to_num(log_depth)
        batch_size, views, height, width = log_depth.shape
        depth_tokens = self.depth_patch_embed(
            log_depth.reshape(batch_size * views, 1, height, width)
        ).reshape(batch_size, views, -1, self.native_dim)
        valid_patches = _valid_depth_patch_mask(
            valid.reshape(batch_size * views, 1, height, width), 14
        ).flatten(1).reshape(batch_size, views, -1)
        return depth_tokens, valid_patches

    @staticmethod
    def _rope_positions(
        vit: nn.Module,
        grid_height: int,
        grid_width: int,
        depth_indices: torch.Tensor,
        voxel_indices: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_pos = vit.position_getter(1, grid_height, grid_width, device=device) + 1
        special = torch.zeros((1, 1, 2), device=device, dtype=patch_pos.dtype)
        local = torch.cat(
            [special, patch_pos, patch_pos[:, depth_indices], patch_pos[:, voxel_indices]],
            dim=1,
        ).unsqueeze(1)
        global_patch = torch.ones_like(patch_pos)
        global_pos = torch.cat(
            [
                special,
                global_patch,
                global_patch[:, depth_indices],
                global_patch[:, voxel_indices],
            ],
            dim=1,
        ).unsqueeze(1)
        return local, global_pos

    def _prepare_rgb_tokens(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vit = self.backbone.model._get_pretrained_backbone()
        views, _channels, height, width = image.shape
        rgb_tokens = vit.patch_embed(image)
        cls_token = vit.prepare_cls_token(1, views)
        absolute_pos = vit.interpolate_pos_encoding(
            torch.cat([cls_token, rgb_tokens], dim=1), height, width
        )
        return (
            cls_token + absolute_pos[:, :1],
            rgb_tokens + absolute_pos[:, 1:],
            absolute_pos,
        )

    def _encode_one(
        self,
        image: torch.Tensor,
        depth_tokens: Optional[torch.Tensor],
        valid_depth_patches: Optional[torch.Tensor],
        voxel_tokens: List[torch.Tensor],
        voxel_patch_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        vit = self.backbone.model._get_pretrained_backbone()
        views, _channels, height, width = image.shape
        grid_height, grid_width = height // 14, width // 14
        rgb_count = grid_height * grid_width
        cls_token, rgb_tokens, absolute_pos = self._prepare_rgb_tokens(image)
        sequences, local_positions, global_positions = [], [], []
        for view_index in range(views):
            if depth_tokens is None or valid_depth_patches is None:
                depth_indices = torch.zeros(0, dtype=torch.long, device=image.device)
                selected_depth = rgb_tokens.new_zeros((1, 0, self.native_dim))
            else:
                depth_indices = torch.nonzero(
                    valid_depth_patches[view_index], as_tuple=False
                ).flatten()
                selected_depth = (
                    depth_tokens[view_index : view_index + 1, depth_indices]
                    + absolute_pos[:, 1:][:, depth_indices]
                    + 1.0
                )
            voxel_indices = voxel_patch_indices[view_index].to(
                device=image.device, dtype=torch.long
            )
            selected_voxel = voxel_tokens[view_index].to(
                device=image.device, dtype=rgb_tokens.dtype
            ).unsqueeze(0)
            selected_voxel = (
                selected_voxel + absolute_pos[:, 1:][:, voxel_indices] + 2.0
            )
            sequence = torch.cat(
                [
                    cls_token[view_index : view_index + 1],
                    rgb_tokens[view_index : view_index + 1],
                    selected_depth,
                    selected_voxel,
                ],
                dim=1,
            ).unsqueeze(1)
            local_pos, global_pos = self._rope_positions(
                vit,
                grid_height,
                grid_width,
                depth_indices,
                voxel_indices,
                image.device,
            )
            sequences.append(sequence)
            local_positions.append(local_pos)
            global_positions.append(global_pos)

        local_x: Optional[List[torch.Tensor]] = None
        for layer_index, block in enumerate(vit.blocks):
            local_pos = local_positions if layer_index >= 4 else [None] * views
            global_pos = global_positions if layer_index >= 4 else [None] * views
            if layer_index == 4:
                sequences = [
                    torch.cat(
                        [
                            vit.camera_token[
                                :, 0 if view_index == 0 else 1 : (0 if view_index == 0 else 1) + 1
                            ].unsqueeze(2),
                            sequence[:, :, 1:],
                        ],
                        dim=2,
                    )
                    for view_index, sequence in enumerate(sequences)
                ]
            if layer_index >= 4 and layer_index % 2 == 1:
                lengths = [int(sequence.shape[2]) for sequence in sequences]
                packed = torch.cat(sequences, dim=2)
                packed_pos = torch.cat(global_pos, dim=2)
                packed = vit.process_attention(
                    packed, block, "global", pos=packed_pos, attn_mask=None
                )
                sequences = list(torch.split(packed, lengths, dim=2))
            else:
                sequences = [
                    vit.process_attention(sequence, block, "local", pos=local_pos[index])
                    for index, sequence in enumerate(sequences)
                ]
                local_x = sequences

        if local_x is None:
            raise RuntimeError("DA3 local state was not initialized.")
        outputs = []
        rgb_positions = torch.arange(1, rgb_count + 1, device=image.device)
        for local_sequence, sequence in zip(local_x, sequences):
            combined = torch.cat([local_sequence, sequence], dim=-1)
            combined = torch.cat(
                [
                    combined[..., : self.native_dim],
                    vit.norm(combined[..., self.native_dim :]),
                ],
                dim=-1,
            )
            rgb = combined.index_select(2, rgb_positions)
            outputs.append(
                rgb.reshape(1, 1, grid_height, grid_width, self.feature_dim)
            )
        return torch.cat(outputs, dim=1).contiguous()

    def _encode(
        self,
        images: torch.Tensor,
        sparse_depth: Optional[torch.Tensor],
        sparse_mask: Optional[torch.Tensor],
        intrinsics: Optional[torch.Tensor],
        origin: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        grid: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        if self.uses_depth_tokens:
            if sparse_depth is None or sparse_mask is None:
                raise RuntimeError("Sparse depth is required for depth-token fusion.")
            depth_tokens, valid_patches = self._sparse_depth_tokens(
                images, sparse_depth, sparse_mask
            )
        else:
            depth_tokens = valid_patches = None
        if self.uses_voxel_tokens:
            if sparse_depth is None or sparse_mask is None or intrinsics is None:
                raise RuntimeError(
                    "Sparse depth and intrinsics are required for voxel fusion."
                )
            encoded = self.voxel_token_encoder(
                sparse_depth,
                sparse_mask,
                intrinsics,
                vox_origin=origin,
                vox_size=size,
                vox_grid=grid,
            )
            features, _centers, frame_indices, patch_rows, patch_cols = encoded
        else:
            features = frame_indices = patch_rows = patch_cols = None
        batch_size, views = images.shape[:2]
        grid_width = images.shape[-1] // 14
        outputs = []
        for batch_index in range(batch_size):
            tokens_by_view, indices_by_view = [], []
            for view_index in range(views):
                if features is None:
                    tokens_by_view.append(images.new_zeros((0, self.native_dim)))
                    indices_by_view.append(
                        torch.zeros(0, dtype=torch.long, device=images.device)
                    )
                else:
                    assert frame_indices is not None
                    assert patch_rows is not None and patch_cols is not None
                    selected = frame_indices == batch_index * views + view_index
                    tokens_by_view.append(features[selected])
                    indices_by_view.append(
                        patch_rows[selected] * grid_width + patch_cols[selected]
                    )
            outputs.append(
                self._encode_one(
                    images[batch_index],
                    None if depth_tokens is None else depth_tokens[batch_index],
                    None if valid_patches is None else valid_patches[batch_index],
                    tokens_by_view,
                    indices_by_view,
                )
            )
        return torch.cat(outputs).float()

    @staticmethod
    def _knn_bounds(
        knn_depth: Optional[torch.Tensor], reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if knn_depth is None or tuple(knn_depth.shape) != tuple(reference.shape):
            raise RuntimeError("knn_depth must match relative depth (B,N,H,W).")
        depth = knn_depth.to(device=reference.device, dtype=torch.float32)
        if not torch.isfinite(depth).all() or not (depth > 0).all():
            raise RuntimeError("knn_depth must be finite and strictly positive.")
        flat = depth.flatten(2)
        minimum = flat.amin(dim=2).view(*depth.shape[:2], 1, 1)
        maximum = flat.amax(dim=2).view(*depth.shape[:2], 1, 1)
        if not ((maximum - minimum) > 1e-6).all():
            raise RuntimeError("knn_depth range must be non-degenerate per frame.")
        return minimum, maximum

    def forward(
        self,
        views: List[Dict[str, torch.Tensor]],
        T_target_from_refcam: Optional[torch.Tensor] = None,
        points_per_frame: Optional[List[List[torch.Tensor]]] = None,
        T_cam_from_velo: Optional[torch.Tensor] = None,
        K_per_frame: Optional[torch.Tensor] = None,
        image_hw: Optional[torch.Tensor] = None,
        knn_depth: Optional[torch.Tensor] = None,
        gt_depth: Optional[torch.Tensor] = None,
        return_depth: bool = False,
        grid_config: Optional[Dict[str, object]] = None,
    ) -> Dict[str, torch.Tensor]:
        del T_target_from_refcam, gt_depth, return_depth
        device = views[0]["img"].device
        resolved_hw = _resolve_image_hw(views, image_hw, device)
        images = self.backbone.stack_images(views).to(device=device)
        if self.uses_depth_tokens or self.uses_voxel_tokens:
            if (
                points_per_frame is None
                or T_cam_from_velo is None
                or K_per_frame is None
            ):
                raise RuntimeError(
                    "Raw LiDAR, camera transform and intrinsics are required."
                )
            sparse_depth, sparse_mask = projected_lidar_sparse_depth(
                points_per_frame,
                T_cam_from_velo,
                K_per_frame,
                resolved_hw,
                device=device,
                allow_empty_single_view=True,
            )
        else:
            sparse_depth = sparse_mask = None
        origin, size, grid = _grid_config(grid_config, device)
        autocast_enabled = device.type == "cuda" and self.backbone.backbone_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=self.backbone.backbone_dtype,
            enabled=autocast_enabled,
        ):
            tokens = self._encode(
                images,
                None if sparse_depth is None else sparse_depth.to(device),
                None if sparse_mask is None else sparse_mask.to(device),
                None if K_per_frame is None else K_per_frame.to(device),
                origin,
                size,
                grid,
            )
            depth = self.dense_depth_head(
                tokens,
                resolved_hw,
                prompt_depth=knn_depth if self.scaled else None,
            ).float()
        if not self.scaled:
            return {"dense_depth": depth}
        scale_min, scale_max = self._knn_bounds(knn_depth, depth)
        dense_depth = depth * (scale_max - scale_min) + scale_min
        return {
            "dense_depth": dense_depth,
            "relative_depth": depth,
            "scale_min": scale_min,
            "scale_max": scale_max,
        }


class DA3BaseDepthModel(TargetDepthModel):
    """Configurable DA3-Base KITTI token-fusion family."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        da3_checkpoint: str | Path | None = None,
        backbone_dtype: torch.dtype = torch.bfloat16,
        freeze_backbone: bool = False,
        load_base: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        if spec.da3_model_name != "da3-base":
            raise ValueError("DA3BaseDepthModel requires a da3-base model spec.")
        self.variant = spec.variant
        self.expected_experiment = spec.experiment
        self.model_class = spec.model_class
        self.fusion_contract = spec.fusion_contract
        self.initialization_contract = spec.initialization_contract
        self.prediction_mode = spec.prediction_mode
        self.depth_scale_contract = spec.depth_scale_contract
        self.online_knn_contract = spec.online_knn_contract
        self.dpt_prompt_contract = spec.dpt_prompt_contract
        self.da3_model_name = spec.da3_model_name
        self.patch_size = 14
        self.num_views = 1
        self.native_dim = spec.native_dim
        self.feature_dim = spec.token_dim
        self.uses_depth_tokens = spec.uses_depth_tokens
        self.uses_voxel_tokens = spec.uses_voxel_tokens
        self.scaled = spec.scaled

        self.backbone = DA3Backbone(
            backbone_dtype=backbone_dtype,
            freeze=freeze_backbone,
            model_name=spec.da3_model_name,
        )
        if load_base:
            if da3_checkpoint is None:
                raise ValueError("da3_checkpoint is required when load_base=True.")
            self.backbone.load_base_checkpoint(da3_checkpoint)
            restore_da3_base_head_rng_state()

        self.dense_depth_head = SingleScaleDPTDepthHead(
            token_dim=spec.token_dim,
            patch_size=14,
            features=128,
            initial_depth=10.0,
            prompt_depth_enabled=spec.scaled,
            prompt_depth_scale="per_frame_minmax" if spec.scaled else "log",
            depth_output_mode="normalized_sigmoid" if spec.scaled else "metric",
            refinement_style="promptda" if spec.scaled else "baseline",
        )
        vit = self.backbone.model._get_pretrained_backbone()
        if spec.uses_depth_tokens:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.depth_patch_embed = type(vit.patch_embed)(
                    img_size=(168, 518),
                    patch_size=14,
                    in_chans=1,
                    embed_dim=spec.native_dim,
                )
        if spec.uses_voxel_tokens:
            self.voxel_token_encoder = PatchDepthBinFeatureEncoder(
                d_out=spec.native_dim,
                H_t=12,
                W_t=37,
                patch_size=14,
                depth_bin_size=4.0,
                vox_origin=(-25.6, -2.0, 0.0),
                vox_size=(0.4, 0.4, 0.4),
                vox_grid=(128, 16, 128),
                d_token=128,
                hidden=64,
                pe_num_freqs=8,
                dynamic_image_size=False,
            )


def build_model(
    da3_checkpoint: str | Path | None = None,
    *,
    load_base: bool = True,
    variant: str = DEFAULT_MODEL_VARIANT,
) -> nn.Module:
    if variant == DEFAULT_MODEL_VARIANT:
        return TargetDepthModel(da3_checkpoint=da3_checkpoint, load_base=load_base)
    if variant == POSTFUSION_MODEL_VARIANT:
        from .postfusion import PostFusionDepthModel

        return PostFusionDepthModel(
            da3_checkpoint=da3_checkpoint, load_base=load_base
        )
    if variant in DA3_BASE_MODEL_VARIANTS:
        return DA3BaseDepthModel(
            get_model_spec(variant, "kitti"),
            da3_checkpoint=da3_checkpoint,
            load_base=load_base,
        )
    raise ValueError(f"Unknown model variant {variant!r}.")


def load_trained_checkpoint(
    model: nn.Module, checkpoint: str | Path
) -> Dict[str, object]:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload must be a mapping.")
    if getattr(model, "da3_model_name", None) == "da3-base":
        args = payload.get("args", {})
        if isinstance(args, dict):
            get_value = args.get
        else:
            get_value = lambda key, default=None: getattr(args, key, default)
        expected = {
            "exp": getattr(model, "expected_experiment", None),
            "da3_model_name": "da3-base",
            "token_dim": getattr(model, "feature_dim", None),
        }
        mismatches = []
        for key, wanted in expected.items():
            actual = get_value(key, None)
            if actual is not None and wanted is not None and actual != wanted:
                mismatches.append(f"{key}={actual!r}, expected {wanted!r}")
        checkpoint_variant = get_value("model_variant", None)
        if (
            checkpoint_variant is not None
            and checkpoint_variant != getattr(model, "variant", None)
        ):
            mismatches.append(
                f"model_variant={checkpoint_variant!r}, expected {model.variant!r}"
            )
        if mismatches:
            raise RuntimeError("Checkpoint metadata mismatch: " + "; ".join(mismatches))
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload
