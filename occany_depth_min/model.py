"""Exact target model: DA3-small + sparse-depth/voxel pre-fusion + PromptDA."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from .da3 import vit_small
from .depth_head import SingleScaleDPTDepthHead
from .initialization import restore_reference_head_rng_state
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
MODEL_VARIANTS = (DEFAULT_MODEL_VARIANT, POSTFUSION_MODEL_VARIANT)


@dataclass(frozen=True)
class ModelSpec:
    variant: str
    model_class: str
    experiment: str
    fusion_contract: str
    initialization_contract: str


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
    ),
}


def get_model_spec(variant: str, dataset: str = "unified6") -> ModelSpec:
    specs = KITTI_MODEL_SPECS if dataset == "kitti" else MODEL_SPECS
    if dataset not in ("unified6", "kitti"):
        raise ValueError("dataset must be 'unified6' or 'kitti'.")
    try:
        return specs[str(variant)]
    except KeyError as error:
        raise ValueError(
            f"Unknown model variant {variant!r}; expected one of {MODEL_VARIANTS}."
        ) from error


class _DinoV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained = vit_small()


class _DA3Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _DinoV2()


class _DA3Wrapper(nn.Module):
    """Only the state-bearing DA3 hierarchy needed by the experiment."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _DA3Net()

    def _get_pretrained_backbone(self) -> nn.Module:
        return self.model.backbone.pretrained

    @staticmethod
    def get_backbone_metadata() -> Dict[str, object]:
        return {
            "token_dim": 384,
            "feature_dim": 768,
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
    ) -> None:
        super().__init__()
        self.backbone_dtype = backbone_dtype
        self.freeze = bool(freeze)
        self.model = _DA3Wrapper()
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
            raise FileNotFoundError(f"Missing DA3-small model.safetensors: {path}")
        from safetensors.torch import load_file

        all_weights = load_file(str(path), device="cpu")
        weights = {
            key: value
            for key, value in all_weights.items()
            if key.startswith("model.backbone.pretrained.")
        }
        status = self.model.load_state_dict(weights, strict=True)
        if status.missing_keys or status.unexpected_keys:
            raise RuntimeError(f"Invalid DA3-small checkpoint: {status}")
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
    """The sole architecture retained in this repository."""

    fusion_contract = FUSION_CONTRACT
    depth_scale_contract = SCALE_CONTRACT
    online_knn_contract = ONLINE_KNN_CONTRACT
    dpt_prompt_contract = DPT_PROMPT_CONTRACT
    initialization_contract = INITIALIZATION_CONTRACT

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
        ).reshape(batch_size, views, -1, 384)
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
        depth_tokens: torch.Tensor,
        valid_depth_patches: torch.Tensor,
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
                [combined[..., :384], vit.norm(combined[..., 384:])], dim=-1
            )
            rgb = combined.index_select(2, rgb_positions)
            outputs.append(rgb.reshape(1, 1, grid_height, grid_width, 768))
        return torch.cat(outputs, dim=1).contiguous()

    def _encode(
        self,
        images: torch.Tensor,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        origin: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        grid: Optional[Tuple[int, int, int]],
    ) -> torch.Tensor:
        depth_tokens, valid_patches = self._sparse_depth_tokens(
            images, sparse_depth, sparse_mask
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
        batch_size, views = images.shape[:2]
        grid_width = images.shape[-1] // 14
        outputs = []
        for batch_index in range(batch_size):
            tokens_by_view, indices_by_view = [], []
            for view_index in range(views):
                if features is None:
                    tokens_by_view.append(images.new_zeros((0, 384)))
                    indices_by_view.append(
                        torch.zeros(0, dtype=torch.long, device=images.device)
                    )
                else:
                    selected = frame_indices == batch_index * views + view_index
                    tokens_by_view.append(features[selected])
                    indices_by_view.append(
                        patch_rows[selected] * grid_width + patch_cols[selected]
                    )
            outputs.append(
                self._encode_one(
                    images[batch_index],
                    depth_tokens[batch_index],
                    valid_patches[batch_index],
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
        if points_per_frame is None or T_cam_from_velo is None or K_per_frame is None:
            raise RuntimeError("Raw LiDAR, camera transform and intrinsics are required.")
        device = views[0]["img"].device
        resolved_hw = _resolve_image_hw(views, image_hw, device)
        images = self.backbone.stack_images(views).to(device=device)
        sparse_depth, sparse_mask = projected_lidar_sparse_depth(
            points_per_frame,
            T_cam_from_velo,
            K_per_frame,
            resolved_hw,
            device=device,
            allow_empty_single_view=True,
        )
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
                sparse_depth.to(device),
                sparse_mask.to(device),
                K_per_frame.to(device),
                origin,
                size,
                grid,
            )
            relative_depth = self.dense_depth_head(
                tokens, resolved_hw, prompt_depth=knn_depth
            ).float()
        scale_min, scale_max = self._knn_bounds(knn_depth, relative_depth)
        dense_depth = relative_depth * (scale_max - scale_min) + scale_min
        return {
            "dense_depth": dense_depth,
            "relative_depth": relative_depth,
            "scale_min": scale_min,
            "scale_max": scale_max,
        }


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
    get_model_spec(variant)
    raise AssertionError("unreachable")


def load_trained_checkpoint(
    model: nn.Module, checkpoint: str | Path
) -> Dict[str, object]:
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload
