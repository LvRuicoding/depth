"""Minimal dual-window voxel/depth post-fusion PromptDA model."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .depth_head import SingleScaleDPTDepthHead
from .initialization import restore_postfusion_rng_state
from .model import DA3Backbone, _grid_config, _resolve_image_hw, _valid_depth_patch_mask
from .projection import projected_lidar_sparse_depth
from .voxel_encoder import PatchDepthBinFeatureEncoder


MODEL_CLASS = (
    "Stage1DepthPatchDepth4mVoxelDepthDualWindowPostFusionOnlyOnlineKNN"
    "PromptDAScaledPreAlignedUnified6Model"
)
FUSION_CONTRACT = (
    "da3_cat_localglobal_patchdepth4m_voxel_shareddual_shift02_sparse_log_"
    "depth_patch_embed_shareddual_shift02_metric_v4"
)
INITIALIZATION_CONTRACT = (
    "da3_small_only_seeded_logdepth_patchdepth4m_voxel_dualwindow_"
    "promptda_dpt_online_knn_scaled_unified6_v1"
)


class _ProjectedKV(NamedTuple):
    frame_idx: torch.Tensor
    h_t: torch.Tensor
    w_t: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _build_window_layout(
    height: int, width: int, window: int, shift: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    slots = window * window
    window_rows = math.ceil((height + shift) / window)
    window_cols = math.ceil((width + shift) / window)
    row_grid = torch.zeros((window_rows * window_cols, slots), dtype=torch.long)
    col_grid = torch.zeros_like(row_grid)
    valid = torch.zeros_like(row_grid, dtype=torch.bool)
    for window_row in range(window_rows):
        row_start = window_row * window - shift
        for window_col in range(window_cols):
            col_start = window_col * window - shift
            window_index = window_row * window_cols + window_col
            slot = 0
            for row in range(row_start, row_start + window):
                for col in range(col_start, col_start + window):
                    if 0 <= row < height and 0 <= col < width:
                        row_grid[window_index, slot] = row
                        col_grid[window_index, slot] = col
                        valid[window_index, slot] = True
                    slot += 1
    return row_grid, col_grid, valid, window_rows * window_cols


class WindowedCrossAttention(nn.Module):
    """One parameter set shared by local/global and regular/shifted windows."""

    def __init__(
        self,
        *,
        d_model: int = 384,
        num_heads: int = 8,
        window: int = 4,
        height: int = 12,
        width: int = 37,
        ffn_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} is not divisible by heads={num_heads}.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_model // self.num_heads
        self.window = int(window)
        self.height = int(height)
        self.width = int(width)
        self.slots = self.window * self.window
        self.attn_dropout = 0.0

        regular = _build_window_layout(self.height, self.width, self.window, 0)
        shifted = _build_window_layout(
            self.height, self.width, self.window, self.window // 2
        )
        for prefix, layout in (("regular", regular), ("shifted", shifted)):
            rows, cols, valid, count = layout
            self.register_buffer(f"{prefix}_rows", rows, persistent=False)
            self.register_buffer(f"{prefix}_cols", cols, persistent=False)
            self.register_buffer(f"{prefix}_valid", valid, persistent=False)
            setattr(self, f"{prefix}_count", int(count))
        self._runtime_layout_cache: Dict[
            Tuple[int, int, int, torch.device],
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int],
        ] = {}

        self.norm_q = nn.LayerNorm(self.d_model)
        self.norm_kv = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)
        hidden = int(self.d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.d_model),
        )

    def _layout(
        self, height: int, width: int, shift: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if shift not in (0, self.window // 2):
            raise ValueError(f"shift must be 0 or {self.window // 2}, got {shift}.")
        if (height, width) == (self.height, self.width):
            prefix = "regular" if shift == 0 else "shifted"
            return (
                getattr(self, f"{prefix}_rows"),
                getattr(self, f"{prefix}_cols"),
                getattr(self, f"{prefix}_valid"),
                getattr(self, f"{prefix}_count"),
                math.ceil((width + shift) / self.window),
            )
        key = (height, width, shift, device)
        value = self._runtime_layout_cache.get(key)
        if value is None:
            rows, cols, valid, count = _build_window_layout(
                height, width, self.window, shift
            )
            value = (
                rows.to(device),
                cols.to(device),
                valid.to(device),
                count,
                math.ceil((width + shift) / self.window),
            )
            self._runtime_layout_cache[key] = value
        return value

    def prepare_projected_kv(
        self,
        features: torch.Tensor,
        frame_indices: torch.Tensor,
        patch_rows: torch.Tensor,
        patch_cols: torch.Tensor,
        valid: torch.Tensor,
    ) -> Optional[_ProjectedKV]:
        if valid.numel() == 0 or not bool(valid.any().item()):
            return None
        normalized = self.norm_kv(features[valid])
        return _ProjectedKV(
            frame_indices[valid],
            patch_rows[valid],
            patch_cols[valid],
            self.k_proj(normalized),
            self.v_proj(normalized),
        )

    def _can_flash(self, value: torch.Tensor) -> bool:
        available = getattr(torch.backends.cuda, "is_flash_attention_available", None)
        return bool(
            value.is_cuda
            and value.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim % 8 == 0
            and self.head_dim <= 256
            and available is not None
            and available()
            and hasattr(torch.ops.aten, "_flash_attention_forward")
        )

    def forward_branches(
        self,
        image_features: torch.Tensor,
        token_features: torch.Tensor,
        frame_indices: torch.Tensor,
        patch_rows: torch.Tensor,
        patch_cols: torch.Tensor,
        valid: torch.Tensor,
        *,
        shift: int,
        prepared_kv: Optional[_ProjectedKV] = None,
    ) -> torch.Tensor:
        if image_features.ndim != 5:
            raise RuntimeError("image_features must be (branches,frames,H,W,D).")
        branches, frames, height, width, channels = image_features.shape
        if channels != self.d_model:
            raise RuntimeError(f"Expected token dim {self.d_model}, got {channels}.")
        rows, cols, query_mask, window_count, windows_wide = self._layout(
            height, width, shift, image_features.device
        )
        if prepared_kv is None:
            prepared_kv = self.prepare_projected_kv(
                token_features, frame_indices, patch_rows, patch_cols, valid
            )
        if prepared_kv is None:
            return image_features

        local_windows = (
            (prepared_kv.h_t + shift) // self.window * windows_wide
            + (prepared_kv.w_t + shift) // self.window
        )
        global_windows = prepared_kv.frame_idx * window_count + local_windows
        global_windows, order = torch.sort(global_windows)
        compact_k = prepared_kv.k[order]
        compact_v = prepared_kv.v[order]
        use_flash = self._can_flash(compact_k)
        if use_flash:
            active, counts = torch.unique_consecutive(
                global_windows, return_counts=True
            )
        else:
            active, inverse, counts = torch.unique_consecutive(
                global_windows, return_inverse=True, return_counts=True
            )
        active_count = int(active.shape[0])
        max_kv = int(counts.max().item())

        if not use_flash:
            starts = torch.zeros_like(counts)
            starts[1:] = counts.cumsum(0)[:-1]
            slots = torch.arange(global_windows.shape[0], device=global_windows.device)
            slots = slots - starts[inverse]

        active_frame = active // window_count
        active_local = active % window_count
        selected_rows = rows[active_local]
        selected_cols = cols[active_local]
        selected_mask = query_mask[active_local]
        query_indices = (
            (active_frame.unsqueeze(-1) * height + selected_rows) * width
            + selected_cols
        )
        image_flat = image_features.reshape(branches, frames * height * width, channels)
        valid_positions = selected_mask.reshape(-1)
        destinations = query_indices.reshape(-1)[valid_positions]
        query_counts = selected_mask.sum(dim=1, dtype=torch.int32)
        heads, head_dim = self.num_heads, self.head_dim

        if use_flash:
            flash_k = compact_k.view(-1, heads, head_dim)
            flash_v = compact_v.view(-1, heads, head_dim)
            cumulative_q = F.pad(query_counts.cumsum(0, dtype=torch.int32), (1, 0))
            kv_counts = counts.to(dtype=torch.int32)
            cumulative_kv = F.pad(kv_counts.cumsum(0, dtype=torch.int32), (1, 0))
            max_q = int(query_counts.max().item())
        else:
            padded_k = compact_k.new_zeros((active_count, max_kv, channels))
            padded_v = compact_v.new_zeros((active_count, max_kv, channels))
            padded_k[inverse, slots] = compact_k
            padded_v[inverse, slots] = compact_v
            kv_mask = (
                torch.arange(max_kv, device=global_windows.device).unsqueeze(0)
                < counts.unsqueeze(1)
            )
            keys = padded_k.view(active_count, max_kv, heads, head_dim).transpose(1, 2)
            values = padded_v.view(active_count, max_kv, heads, head_dim).transpose(1, 2)
            attention_mask = kv_mask.view(active_count, 1, 1, max_kv).expand(
                active_count, 1, self.slots, max_kv
            )

        outputs = []
        for branch_index in range(branches):
            branch_queries = image_flat[branch_index, destinations]
            projected_queries = self.q_proj(self.norm_q(branch_queries))
            if use_flash:
                attended = torch.ops.aten._flash_attention_forward(
                    projected_queries.view(-1, heads, head_dim),
                    flash_k,
                    flash_v,
                    cumulative_q,
                    cumulative_kv,
                    max_q,
                    max_kv,
                    self.attn_dropout if self.training else 0.0,
                    False,
                    False,
                )[0].reshape(-1, channels)
            else:
                padded_q = projected_queries.new_zeros(
                    (active_count * self.slots, channels)
                )
                padded_q[valid_positions] = projected_queries
                queries = padded_q.view(active_count, self.slots, heads, head_dim)
                queries = queries.transpose(1, 2)
                attended = F.scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    attn_mask=attention_mask,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )
                attended = attended.transpose(1, 2).contiguous().view(-1, channels)
                attended = attended[valid_positions]
            attended = self.out_proj(attended)
            residual = branch_queries + attended
            update = attended + self.ffn(self.norm_ffn(residual))
            output = image_flat[branch_index].clone()
            output[destinations] = output[destinations] + update
            outputs.append(output.view(frames, height, width, channels))
        return torch.stack(outputs, dim=0)


def _empty_encoded(reference: torch.Tensor, channels: int):
    return (
        reference.new_zeros((0, channels)),
        reference.new_zeros((0, 3), dtype=torch.float32),
        torch.zeros(0, dtype=torch.long, device=reference.device),
        torch.zeros(0, dtype=torch.long, device=reference.device),
        torch.zeros(0, dtype=torch.long, device=reference.device),
        torch.zeros(0, dtype=torch.bool, device=reference.device),
    )


class _SharedWindowFusion(nn.Module):
    def __init__(self, *, with_voxel_encoder: bool) -> None:
        super().__init__()
        component = "voxel_attention" if with_voxel_encoder else "depth_attention"
        with torch.random.fork_rng(devices=[]):
            restore_postfusion_rng_state(component)
            self.attention = WindowedCrossAttention()
        self.window_shift = 2
        if with_voxel_encoder:
            with torch.random.fork_rng(devices=[]):
                restore_postfusion_rng_state("voxel_encoder")
                self.depth_encoder = PatchDepthBinFeatureEncoder(
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

    def apply(
        self,
        branches: torch.Tensor,
        encoded: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        features, _centers, frames, rows, cols, valid = encoded
        if features.shape[0] == 0:
            return branches
        prepared = self.attention.prepare_projected_kv(
            features, frames, rows, cols, valid
        )
        for shift in (0, self.window_shift):
            branches = self.attention.forward_branches(
                branches,
                features,
                frames,
                rows,
                cols,
                valid,
                shift=shift,
                prepared_kv=prepared,
            )
        return branches

    def encode_voxels(
        self,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        reference: torch.Tensor,
        origin: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        grid: Optional[Tuple[int, int, int]],
    ):
        encoded = self.depth_encoder(
            sparse_depth.to(reference.device),
            sparse_mask.to(reference.device),
            intrinsics.to(reference.device),
            vox_origin=origin,
            vox_size=size,
            vox_grid=grid,
        )
        features, centers, frames, rows, cols = encoded
        if features is None:
            return _empty_encoded(reference, 384)
        assert centers is not None and frames is not None
        assert rows is not None and cols is not None
        return (
            features.to(dtype=reference.dtype),
            centers,
            frames,
            rows,
            cols,
            torch.ones_like(frames, dtype=torch.bool),
        )

    @staticmethod
    def encode_depth_patches(
        depth_tokens: torch.Tensor,
        valid_patches: torch.Tensor,
        reference: torch.Tensor,
    ):
        batch, views, count, channels = depth_tokens.shape
        _branches, frames, height, width, reference_channels = reference.shape
        if (batch * views, count, channels) != (
            frames,
            height * width,
            reference_channels,
        ):
            raise RuntimeError("Depth patches do not match image token grid.")
        grid = depth_tokens.reshape(frames, height, width, channels)
        mask = valid_patches.bool().reshape(frames, height, width)
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() == 0:
            return _empty_encoded(reference, channels)
        frame_indices, rows, cols = indices.unbind(dim=1)
        features = grid[frame_indices, rows, cols].to(dtype=reference.dtype)
        return (
            features,
            reference.new_zeros((features.shape[0], 3)),
            frame_indices,
            rows,
            cols,
            torch.ones_like(frame_indices, dtype=torch.bool),
        )


class VoxelDepthDualWindowFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.voxel_fusion = _SharedWindowFusion(with_voxel_encoder=True)
        self.depth_fusion = _SharedWindowFusion(with_voxel_encoder=False)

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        *,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        intrinsics: torch.Tensor,
        depth_tokens: torch.Tensor,
        valid_depth_patches: torch.Tensor,
        origin: Optional[torch.Tensor],
        size: Optional[torch.Tensor],
        grid: Optional[Tuple[int, int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if local_tokens.shape != global_tokens.shape:
            raise RuntimeError("Local/global DA3 token shapes must match.")
        batch, views, height, width, channels = local_tokens.shape
        local = local_tokens.reshape(batch * views, height, width, channels).contiguous()
        global_ = global_tokens.reshape(batch * views, height, width, channels).contiguous()
        branches = torch.stack([local, global_], dim=0)
        voxel_encoded = self.voxel_fusion.encode_voxels(
            sparse_depth,
            sparse_mask,
            intrinsics,
            local,
            origin,
            size,
            grid,
        )
        branches = self.voxel_fusion.apply(branches, voxel_encoded)
        depth_encoded = self.depth_fusion.encode_depth_patches(
            depth_tokens, valid_depth_patches, branches
        )
        branches = self.depth_fusion.apply(branches, depth_encoded)
        local, global_ = branches.unbind(dim=0)
        shape = (batch, views, height, width, channels)
        return local.view(shape), global_.view(shape)


class PostFusionDepthModel(nn.Module):
    fusion_contract = FUSION_CONTRACT
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
        self.fusion = VoxelDepthDualWindowFusion()
        with torch.random.fork_rng(devices=[]):
            restore_postfusion_rng_state("head")
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

    @staticmethod
    def _positions(
        vit: nn.Module, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patches = vit.position_getter(1, height, width, device=device) + 1
        special = torch.zeros((1, 1, 2), device=device, dtype=patches.dtype)
        local = torch.cat([special, patches], dim=1).unsqueeze(1)
        global_ = torch.cat([special, torch.ones_like(patches)], dim=1).unsqueeze(1)
        return local, global_

    def _encode_one(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vit = self.backbone.model._get_pretrained_backbone()
        views, _channels, image_height, image_width = images.shape
        height, width = image_height // 14, image_width // 14
        count = height * width
        rgb = vit.patch_embed(images)
        cls = vit.prepare_cls_token(1, views)
        absolute = vit.interpolate_pos_encoding(
            torch.cat([cls, rgb], dim=1), image_height, image_width
        )
        cls = cls + absolute[:, :1]
        rgb = rgb + absolute[:, 1:]
        sequences = [
            torch.cat([cls[index : index + 1], rgb[index : index + 1]], dim=1).unsqueeze(1)
            for index in range(views)
        ]
        positions = [self._positions(vit, height, width, images.device) for _ in range(views)]
        local_state: Optional[List[torch.Tensor]] = None
        for layer_index, block in enumerate(vit.blocks):
            local_positions = [value[0] if layer_index >= 4 else None for value in positions]
            global_positions = [value[1] if layer_index >= 4 else None for value in positions]
            if layer_index == 4:
                sequences = [
                    torch.cat(
                        [
                            vit.camera_token[:, 0 if index == 0 else 1 : (0 if index == 0 else 1) + 1].unsqueeze(2),
                            sequence[:, :, 1:],
                        ],
                        dim=2,
                    )
                    for index, sequence in enumerate(sequences)
                ]
            if layer_index >= 4 and layer_index % 2 == 1:
                lengths = [int(sequence.shape[2]) for sequence in sequences]
                packed = torch.cat(sequences, dim=2)
                packed_positions = torch.cat(global_positions, dim=2)
                packed = vit.process_attention(
                    packed, block, "global", pos=packed_positions, attn_mask=None
                )
                sequences = list(torch.split(packed, lengths, dim=2))
            else:
                sequences = [
                    vit.process_attention(
                        sequence, block, "local", pos=local_positions[index]
                    )
                    for index, sequence in enumerate(sequences)
                ]
                local_state = sequences
        if local_state is None:
            raise RuntimeError("DA3 local state was not initialized.")
        rgb_positions = torch.arange(1, count + 1, device=images.device)
        local_outputs, global_outputs = [], []
        for local, global_ in zip(local_state, sequences):
            local_rgb = local.index_select(2, rgb_positions)
            global_rgb = vit.norm(global_).index_select(2, rgb_positions)
            local_outputs.append(local_rgb.reshape(1, 1, height, width, 384))
            global_outputs.append(global_rgb.reshape(1, 1, height, width, 384))
        return (
            torch.cat(local_outputs, dim=1).contiguous(),
            torch.cat(global_outputs, dim=1).contiguous(),
        )

    def _encode_backbone(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        local, global_ = [], []
        for batch_index in range(images.shape[0]):
            local_one, global_one = self._encode_one(images[batch_index])
            local.append(local_one)
            global_.append(global_one)
        return torch.cat(local).float(), torch.cat(global_).float()

    def _depth_tokens(
        self, sparse_depth: torch.Tensor, sparse_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = sparse_mask.bool() & torch.isfinite(sparse_depth) & (sparse_depth > 0.01)
        values = torch.where(
            valid,
            torch.log(sparse_depth.clamp_min(0.01)),
            torch.zeros_like(sparse_depth),
        )
        values = torch.nan_to_num(values)
        batch, views, height, width = values.shape
        tokens = self.depth_patch_embed(values.reshape(batch * views, 1, height, width))
        tokens = tokens.reshape(batch, views, tokens.shape[1], tokens.shape[2])
        valid_patches = _valid_depth_patch_mask(
            valid.reshape(batch * views, 1, height, width), 14
        ).flatten(1).reshape(batch, views, -1)
        return tokens, valid_patches

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
            local, global_ = self._encode_backbone(images)
            depth_tokens, valid_patches = self._depth_tokens(
                sparse_depth.to(device), sparse_mask.to(device)
            )
            local, global_ = self.fusion(
                local,
                global_,
                sparse_depth=sparse_depth,
                sparse_mask=sparse_mask,
                intrinsics=K_per_frame,
                depth_tokens=depth_tokens,
                valid_depth_patches=valid_patches,
                origin=origin,
                size=size,
                grid=grid,
            )
            relative_depth = self.dense_depth_head(
                torch.cat([local, global_], dim=-1),
                resolved_hw,
                prompt_depth=knn_depth,
            ).float()
        scale_min, scale_max = self._knn_bounds(knn_depth, relative_depth)
        return {
            "dense_depth": relative_depth * (scale_max - scale_min) + scale_min,
            "relative_depth": relative_depth,
            "scale_min": scale_min,
            "scale_max": scale_max,
        }


def build_model(
    da3_checkpoint: str | Path | None = None, *, load_base: bool = True
) -> PostFusionDepthModel:
    return PostFusionDepthModel(da3_checkpoint=da3_checkpoint, load_base=load_base)
