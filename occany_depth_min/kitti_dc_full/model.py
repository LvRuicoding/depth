"""Self-contained DA3-Base models for the full KITTI completion experiment.

Legacy class names, parameter registration order, and seed-0 initialization
are retained so source checkpoints can be loaded and resumed strictly.
"""
from __future__ import annotations

import torch
from torch import nn

from ..depth_head import SingleScaleDPTDepthHead
from ..initialization import restore_da3_base_head_rng_state
from ..model import DA3Backbone, TargetDepthModel, _resolve_image_hw, get_model_spec
from ..voxel_encoder import PatchDepthBinFeatureEncoder
from .fusion import VoxelDepthDualWindowSharedBranchFusionModule
from .raw_voxel import VoxelFeatureEncoder, encode_projected_raw_voxel_tokens


def validate_preprojected_sparse_depth(depth, mask, images):
    """Require one shared, finite z-buffer on the padded image grid."""
    expected = (*images.shape[:2], *images.shape[-2:])
    if not isinstance(depth, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise ValueError("sparse_depth and sparse_depth_mask must be provided together")
    if tuple(depth.shape) != expected or tuple(mask.shape) != expected:
        raise ValueError(f"Preprojected sparse maps must have shape {expected}")
    if mask.dtype != torch.bool:
        raise ValueError("sparse_depth_mask must be boolean")
    depth = depth.to(device=images.device, dtype=torch.float32)
    mask = mask.to(device=images.device)
    if not bool(torch.isfinite(depth).all()) or not bool((depth >= 0).all()):
        raise ValueError("Preprojected depths must be finite and nonnegative")
    if not torch.equal(mask, depth > 0):
        raise ValueError("sparse_depth_mask must exactly identify positive depths")
    return depth, mask


def _backbone(checkpoint, dtype, load_base):
    backbone = DA3Backbone(backbone_dtype=dtype, freeze=False, model_name="da3-base")
    if load_base:
        if checkpoint is None:
            raise ValueError("A DA3-base checkpoint is required when load_base=True")
        backbone.load_base_checkpoint(checkpoint)
    # The full experiment fixes seed=0. The source constructs and discards
    # DA3's native heads before registering experiment-specific parameters.
    restore_da3_base_head_rng_state()
    return backbone


def _head(*, scaled, features=128, initial=10.0):
    return SingleScaleDPTDepthHead(
        token_dim=1536, patch_size=14, features=features, initial_depth=initial,
        depth_output_mode="normalized_sigmoid" if scaled else "metric",
        prompt_depth_enabled=scaled,
        prompt_depth_scale="per_frame_minmax" if scaled else "log",
        refinement_style="promptda" if scaled else "baseline",
    )


def _depth_embed(backbone, seed=0):
    vit = backbone.model._get_pretrained_backbone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return type(vit.patch_embed)(
            img_size=(168, 518), patch_size=14, in_chans=1, embed_dim=768,
        )


def _validate_architecture(da3_model_name, token_dim, patch_size, num_frames,
                           num_views, freeze_backbone, backbone_img_size):
    if (da3_model_name, token_dim, patch_size, num_frames, num_views) != (
        "da3-base", 1536, 14, 1, 1,
    ):
        raise ValueError("Full KITTI requires DA3-base, dim 1536, patch 14, one frame/view")
    if freeze_backbone:
        raise ValueError("Full KITTI requires an unfrozen backbone")
    if tuple(backbone_img_size) != (168, 518):
        raise ValueError("Full KITTI retains the (168,518) parameter reference grid")


class _FullPreFusionModel(TargetDepthModel):
    """RGB, sparse log-depth and optional voxel tokens before all DA3 blocks."""

    uses_depth_tokens = True
    uses_voxel_tokens = True
    scaled = False
    prediction_mode = "direct_metric"
    depth_scale_contract = "direct_softplus_metric_v1"

    def __init__(
        self, *, occany_ckpt=None, da3_model_name="da3-base", token_dim=1536,
        patch_size=14, backbone_img_size=(168, 518), num_frames=1, num_views=1,
        freeze_backbone=False, backbone_dtype=torch.bfloat16,
        dynamic_image_size=True, voxel_encoder="patchdepthbin", depth_embed_seed=0,
        dense_depth_features=128, dense_depth_initial=10.0, load_base=None,
        fusion_vox_origin=(-25.6, -2.0, 0.0), fusion_vox_size=(0.4, 0.4, 0.4),
        fusion_vox_grid=(128, 16, 128), fusion_d_voxel=128,
        fusion_pe_num_freqs=8, fusion_depth_bin_size=4.0,
    ):
        nn.Module.__init__(self)
        _validate_architecture(da3_model_name, token_dim, patch_size, num_frames,
                               num_views, freeze_backbone, backbone_img_size)
        if voxel_encoder not in ("patchdepthbin", "vfe"):
            raise ValueError("voxel_encoder must be patchdepthbin or vfe")
        if voxel_encoder == "vfe" and not self.uses_voxel_tokens:
            raise ValueError("RAW VFE requires an enabled voxel token branch")
        if fusion_depth_bin_size != 4.0:
            raise ValueError("Full KITTI fixes depth bins to 4 metres")
        self.patch_size, self.num_views = 14, 1
        self.native_dim, self.feature_dim = 768, 1536
        self.da3_model_name = "da3-base"
        self.depth_token_branch_enabled = self.uses_depth_tokens
        self.voxel_token_branch_enabled = self.uses_voxel_tokens
        self.voxel_encoder_kind = voxel_encoder
        self.dynamic_image_size = bool(dynamic_image_size)
        variant = next(key for key, cls in MODEL_CLASSES.items() if isinstance(self, cls))
        self.fusion_contract = get_model_spec(variant, "kitti").fusion_contract
        if voxel_encoder == "vfe":
            self.fusion_contract = self.fusion_contract.replace(
                "patchdepth4m_voxel", "raw_cartesian_vfe_voxel",
            )
        if self.scaled:
            self.depth_scale_contract = "online_knn4_minmax_sigmoid_metric_v1"
            self.DPT_PROMPT_CONTRACT = "online_knn4_per_frame_minmax_promptda_dpt4_scaled_v1"
        self.backbone = _backbone(
            occany_ckpt, backbone_dtype,
            occany_ckpt is not None if load_base is None else load_base,
        )
        self.dense_depth_head = _head(
            scaled=self.scaled, features=dense_depth_features,
            initial=dense_depth_initial,
        )
        if self.uses_depth_tokens:
            self.depth_patch_embed = _depth_embed(self.backbone, depth_embed_seed)
        if self.uses_voxel_tokens:
            if voxel_encoder == "vfe":
                self.voxel_token_encoder = VoxelFeatureEncoder(
                    vox_origin=fusion_vox_origin, vox_size=fusion_vox_size,
                    vox_grid=fusion_vox_grid, d_voxel=fusion_d_voxel,
                    d_out=768, hidden=64, pe_num_freqs=fusion_pe_num_freqs,
                    force_fp32_geometry=True,
                )
            else:
                self.voxel_token_encoder = PatchDepthBinFeatureEncoder(
                    d_out=768, H_t=12, W_t=37, patch_size=14,
                    depth_bin_size=4.0, vox_origin=fusion_vox_origin,
                    vox_size=fusion_vox_size, vox_grid=fusion_vox_grid,
                    d_token=fusion_d_voxel, hidden=64,
                    pe_num_freqs=fusion_pe_num_freqs,
                    dynamic_image_size=self.dynamic_image_size,
                )

    def pretrained_parameter_prefixes(self):
        return ("backbone.",)

    def _encode_full(self, images, sparse_depth, sparse_mask, intrinsics,
                     points, transform, image_valid_mask):
        if self.voxel_encoder_kind != "vfe":
            return self._encode(images, sparse_depth, sparse_mask, intrinsics,
                                None, None, None)
        if self.uses_depth_tokens:
            depth_tokens, valid_patches = self._sparse_depth_tokens(
                images, sparse_depth, sparse_mask,
            )
        else:
            depth_tokens = valid_patches = None
        features, _centers, frames, rows, cols = encode_projected_raw_voxel_tokens(
            self.voxel_token_encoder, images, points, transform, intrinsics,
            image_valid_mask, patch_size=14,
        )
        grid_width = images.shape[-1] // 14
        outputs = []
        for index in range(images.shape[0]):
            if features is None:
                tokens = images.new_zeros((0, 768))
                patches = torch.empty(0, dtype=torch.long, device=images.device)
            else:
                selected = frames == index
                tokens = features[selected]
                patches = rows[selected] * grid_width + cols[selected]
            outputs.append(self._encode_one(
                images[index],
                None if depth_tokens is None else depth_tokens[index],
                None if valid_patches is None else valid_patches[index],
                [tokens], [patches],
            ))
        return torch.cat(outputs).float()

    def forward(self, views, *, K_per_frame=None, image_hw=None,
                sparse_depth=None, sparse_depth_mask=None, knn_depth=None,
                points_per_frame=None, T_cam_from_velo=None, image_valid_mask=None):
        images = self.backbone.stack_images(views)
        if images.shape[1] != 1:
            raise ValueError("Full KITTI expects one image view")
        resolved_hw = _resolve_image_hw(views, image_hw, images.device)
        if self.uses_depth_tokens or self.uses_voxel_tokens:
            if K_per_frame is None:
                raise ValueError("Full KITTI conditioning requires camera intrinsics")
            sparse_depth, sparse_mask = validate_preprojected_sparse_depth(
                sparse_depth, sparse_depth_mask, images,
            )
        else:
            sparse_depth = sparse_mask = None
        dtype = self.backbone.backbone_dtype
        enabled = images.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type=images.device.type, dtype=dtype, enabled=enabled):
            tokens = self._encode_full(
                images, sparse_depth, sparse_mask, K_per_frame,
                points_per_frame, T_cam_from_velo, image_valid_mask,
            )
        depth = self.dense_depth_head(
            tokens, resolved_hw, prompt_depth=knn_depth if self.scaled else None,
        ).float()
        if not self.scaled:
            return {"dense_depth": depth}
        minimum, maximum = self._knn_bounds(knn_depth, depth)
        return dict(dense_depth=depth * (maximum - minimum) + minimum,
                    relative_depth=depth, scale_min=minimum, scale_max=maximum)


class Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBOnlyDirectMetricModel(_FullPreFusionModel):
    uses_depth_tokens = False
    uses_voxel_tokens = False


class Stage1DepthLingBotDA3LastDirectMetricModel(_FullPreFusionModel):
    uses_voxel_tokens = False


class Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBVoxelDirectMetricModel(_FullPreFusionModel):
    uses_depth_tokens = False


class Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionDirectMetricModel(_FullPreFusionModel):
    pass


class Stage1DepthLingBotDA3LastDepthImageOnlineKNNPromptDAScaledUnified6Model(_FullPreFusionModel):
    uses_voxel_tokens = False
    scaled = True
    prediction_mode = "relative_online_knn_minmax"


class Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBVoxelOnlineKNNPromptDAScaledUnified6Model(_FullPreFusionModel):
    uses_depth_tokens = False
    scaled = True
    prediction_mode = "relative_online_knn_minmax"


class Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNNPromptDAScaledUnified6Model(_FullPreFusionModel):
    scaled = True
    prediction_mode = "relative_online_knn_minmax"


class Stage1DepthKITTIFullPostFusionDirectMetricModel(nn.Module):
    """RGB-only DA3, then independent voxel and depth SharedDual attention."""

    promptda_scaled = False
    prediction_mode = "direct_metric"

    def __init__(
        self, *, occany_ckpt=None, da3_model_name="da3-base", token_dim=1536,
        patch_size=14, backbone_img_size=(168, 518), num_frames=1, num_views=1,
        freeze_backbone=False, backbone_dtype=torch.bfloat16,
        dynamic_image_size=True, enable_depth=True, enable_voxel=True,
        voxel_encoder="patchdepthbin", depth_embed_seed=0,
        dense_depth_features=128, dense_depth_initial=10.0, load_base=None,
        fusion_num_heads=8, fusion_window=4, fusion_depth_bin_size=4.0,
        fusion_vox_origin=(-25.6, -2.0, 0.0), fusion_vox_size=(0.4, 0.4, 0.4),
        fusion_vox_grid=(128, 16, 128), fusion_d_voxel=128,
        fusion_pe_num_freqs=8,
    ):
        super().__init__()
        _validate_architecture(da3_model_name, token_dim, patch_size, num_frames,
                               num_views, freeze_backbone, backbone_img_size)
        if not dynamic_image_size:
            raise ValueError("Full KITTI post-fusion requires dynamic images")
        if voxel_encoder not in ("patchdepthbin", "vfe"):
            raise ValueError("voxel_encoder must be patchdepthbin or vfe")
        if voxel_encoder == "vfe" and not enable_voxel:
            raise ValueError("RAW VFE requires an enabled voxel token branch")
        if voxel_encoder == "patchdepthbin" and fusion_depth_bin_size != 4.0:
            raise ValueError("Aligned voxel tokens require 4 m depth bins")
        self.depth_token_branch_enabled = bool(enable_depth)
        self.voxel_token_branch_enabled = bool(enable_voxel)
        self.voxel_encoder_kind = voxel_encoder
        self.patch_size, self.branch_dim = 14, 768
        self.backbone = _backbone(
            occany_ckpt, backbone_dtype,
            occany_ckpt is not None if load_base is None else load_base,
        )
        self.fusion = VoxelDepthDualWindowSharedBranchFusionModule(
            d_model=768, H_t=12, W_t=37, patch_size=14,
            num_heads=fusion_num_heads, window=fusion_window,
            vox_origin=fusion_vox_origin, vox_size=fusion_vox_size,
            vox_grid=fusion_vox_grid, depth_bin_size=fusion_depth_bin_size,
            vfe_d_voxel=fusion_d_voxel, pe_num_freqs=fusion_pe_num_freqs,
            dynamic_image_size=True,
            voxel_token_source="cartesian" if voxel_encoder == "vfe" else "patch_depth4m",
            depth_token_source="sparse_log_patch",
            enable_depth=self.depth_token_branch_enabled,
            enable_voxel=self.voxel_token_branch_enabled,
        )
        if voxel_encoder == "vfe":
            self.fusion.voxel_fusion.vfe.force_fp32_geometry = True
        if self.depth_token_branch_enabled:
            self.depth_patch_embed = _depth_embed(self.backbone, depth_embed_seed)
        self.dense_depth_head = _head(
            scaled=self.promptda_scaled, features=dense_depth_features,
            initial=dense_depth_initial,
        )

    def pretrained_parameter_prefixes(self):
        return ("backbone.",)

    def _encode_rgb(self, images):
        # Reuse the same RGB token layout as the prefusion image ablation.
        # This contains no depth/voxel parameters or conditioning tokens.
        empty = torch.empty(0, dtype=torch.long, device=images.device)
        outputs = [TargetDepthModel._encode_one(
            self, image, None, None, [image.new_zeros((0, 768))], [empty],
        ) for image in images]
        return torch.cat(outputs).float()

    native_dim, feature_dim = 768, 1536
    _prepare_rgb_tokens = TargetDepthModel._prepare_rgb_tokens
    _rope_positions = staticmethod(TargetDepthModel._rope_positions)
    _sparse_depth_tokens = TargetDepthModel._sparse_depth_tokens

    def _encode_raw_voxels(self, images, points, transform, intrinsics, valid_mask, reference):
        features, centers, frames, rows, cols = encode_projected_raw_voxel_tokens(
            self.fusion.voxel_fusion.vfe, images, points, transform,
            intrinsics, valid_mask, patch_size=14,
        )
        if features is None:
            features = reference.new_zeros((0, 768))
            centers = reference.new_zeros((0, 3))
            frames = torch.empty(0, dtype=torch.long, device=reference.device)
            rows = cols = frames
        return (features.to(dtype=reference.dtype), centers, frames, rows, cols,
                torch.ones_like(frames, dtype=torch.bool))

    def forward(self, views, *, K_per_frame, image_hw=None,
                sparse_depth=None, sparse_depth_mask=None, knn_depth=None,
                points_per_frame=None, T_cam_from_velo=None, image_valid_mask=None):
        images = self.backbone.stack_images(views)
        if images.shape[1] != 1:
            raise ValueError("Full KITTI post-fusion expects one image view")
        sparse_depth, sparse_mask = validate_preprojected_sparse_depth(
            sparse_depth, sparse_depth_mask, images,
        )
        dtype = self.backbone.backbone_dtype
        enabled = images.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(device_type=images.device.type, dtype=dtype, enabled=enabled):
            tokens = self._encode_rgb(images)
        local, global_ = tokens[..., :768], tokens[..., 768:]
        resolved_hw = _resolve_image_hw(views, image_hw, tokens.device)
        voxel_kwargs = {}
        if self.voxel_encoder_kind == "vfe":
            voxel_kwargs["voxel_encoded"] = self._encode_raw_voxels(
                images, points_per_frame, T_cam_from_velo, K_per_frame,
                image_valid_mask, local,
            )
        depth_kwargs = {}
        if self.depth_token_branch_enabled:
            depth_tokens, valid = self._sparse_depth_tokens(images, sparse_depth, sparse_mask)
            depth_kwargs = dict(depth_tokens=depth_tokens, valid_depth_patches=valid)
        local, global_ = self.fusion(
            local, global_, K_per_frame=K_per_frame, image_hw=resolved_hw,
            sparse_depth=sparse_depth, sparse_mask=sparse_mask,
            **depth_kwargs, **voxel_kwargs,
        )
        fused = torch.cat([local, global_], dim=-1)
        if not self.promptda_scaled:
            return {"dense_depth": self.dense_depth_head(fused, resolved_hw).float()}
        relative = self.dense_depth_head(fused, resolved_hw, prompt_depth=knn_depth).float()
        minimum, maximum = TargetDepthModel._knn_bounds(knn_depth, relative)
        return dict(dense_depth=relative * (maximum - minimum) + minimum,
                    relative_depth=relative, scale_min=minimum, scale_max=maximum)


class Stage1DepthKITTIFullPostFusionPromptDAScaledModel(Stage1DepthKITTIFullPostFusionDirectMetricModel):
    promptda_scaled = True
    prediction_mode = "relative_online_knn_minmax"
    depth_scale_contract = "online_knn4_minmax_sigmoid_metric_v1"
    DPT_PROMPT_CONTRACT = "online_knn4_per_frame_minmax_promptda_dpt4_scaled_v1"


MODEL_CLASSES = {
    "image": Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBOnlyDirectMetricModel,
    "depth": Stage1DepthLingBotDA3LastDirectMetricModel,
    "voxel": Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBVoxelDirectMetricModel,
    "voxel_depth": Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionDirectMetricModel,
    "depth_scaled": Stage1DepthLingBotDA3LastDepthImageOnlineKNNPromptDAScaledUnified6Model,
    "voxel_scaled": Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionRGBVoxelOnlineKNNPromptDAScaledUnified6Model,
    "voxel_depth_scaled": Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNNPromptDAScaledUnified6Model,
}
VFE_MODELS = ("voxel", "voxel_depth", "voxel_scaled", "voxel_depth_scaled")


def model_class(args):
    if args.fusion_mode == "postfusion" and args.model != "image":
        return (Stage1DepthKITTIFullPostFusionPromptDAScaledModel
                if args.model.endswith("scaled")
                else Stage1DepthKITTIFullPostFusionDirectMetricModel)
    return MODEL_CLASSES[args.model]


def legacy_class_name(args):
    return model_class(args).__name__


def build_model(args, *, load_base=True):
    if args.voxel_encoder == "vfe" and args.model not in VFE_MODELS:
        raise ValueError("--voxel-encoder vfe requires a model with voxel tokens")
    kwargs = dict(
        occany_ckpt=str(args.occany_ckpt) if args.occany_ckpt is not None else None,
        backbone_dtype=torch.bfloat16 if args.amp == "bf16" else torch.float32,
        dynamic_image_size=True, load_base=load_base,
    )
    if args.fusion_mode == "postfusion" and args.model != "image":
        kwargs.update(enable_depth=args.model.startswith(("depth", "voxel_depth")),
                      enable_voxel=args.model.startswith("voxel"))
    if args.voxel_encoder == "vfe":
        kwargs["voxel_encoder"] = "vfe"
    return model_class(args)(**kwargs)
