"""DA3-Base local/global SharedDual voxel-then-depth window fusion.

Constructor order retains the discarded RAW VFE and shifted attention so
seeded initialization matches the original full KITTI checkpoints.
"""
from __future__ import annotations
import math
from typing import List, NamedTuple, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from ..voxel_encoder import PatchDepthBinFeatureEncoder
from .raw_voxel import VoxelFeatureEncoder

class _ProjectedVoxelKV(NamedTuple):
    """Compact projected LiDAR K/V shared by branches and window shifts."""

    frame_idx: torch.Tensor
    h_t: torch.Tensor
    w_t: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

def _build_window_layout(
    H_t: int, W_t: int, window: int, shift: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Precompute the static (frame-relative) window <-> patch layout.

    Args:
        H_t, W_t: patch grid size.
        window:   window size (assumed square).
        shift:    spatial shift applied to the grid origin (0 for W-MSA,
                  window//2 for SW-MSA).

    Returns:
        win_h_grid, win_w_grid: (n_win, M_Q) int64 — per-window list of (h_t, w_t)
                                slot coords. Padded with 0 where invalid.
        win_q_mask:             (n_win, M_Q) bool — True where the slot is a
                                real in-bounds patch.
        n_win:                  total window count.
    """
    M_Q = window * window
    n_h = math.ceil((H_t + shift) / window)
    n_w = math.ceil((W_t + shift) / window)
    # Window i along H covers h_t in [i*window - shift, (i+1)*window - shift - 1].
    win_h_grid = torch.zeros((n_h * n_w, M_Q), dtype=torch.long)
    win_w_grid = torch.zeros((n_h * n_w, M_Q), dtype=torch.long)
    win_q_mask = torch.zeros((n_h * n_w, M_Q), dtype=torch.bool)
    for wh in range(n_h):
        h_start = wh * window - shift
        h_end = h_start + window
        for ww in range(n_w):
            w_start = ww * window - shift
            w_end = w_start + window
            win_id = wh * n_w + ww
            slot = 0
            for h in range(h_start, h_end):
                for w in range(w_start, w_end):
                    if 0 <= h < H_t and 0 <= w < W_t:
                        win_h_grid[win_id, slot] = h
                        win_w_grid[win_id, slot] = w
                        win_q_mask[win_id, slot] = True
                    slot += 1
    return win_h_grid, win_w_grid, win_q_mask, n_h * n_w

class WindowedCrossAttnLayer(nn.Module):
    """One layer: cross-attn (Q=image patch tokens, KV=voxel features) + FFN."""

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        window: int = 4,
        shift: int = 0,
        H_t: int = 10,
        W_t: int = 32,
        ffn_ratio: float = 2.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window = window
        self.shift = shift
        self.H_t = H_t
        self.W_t = W_t
        self.attn_dropout = attn_dropout

        wh_grid, ww_grid, q_mask, n_win = _build_window_layout(H_t, W_t, window, shift)
        self.register_buffer("win_h_grid", wh_grid, persistent=False)  # (n_win, M_Q)
        self.register_buffer("win_w_grid", ww_grid, persistent=False)
        self.register_buffer("win_q_mask", q_mask, persistent=False)
        self.n_win = int(n_win)
        self.M_Q = window * window
        # number of windows along width — needed to bucket voxels by window.
        self.n_w = int(math.ceil((W_t + shift) / window))

        alternate_shift = window // 2 if shift == 0 else 0
        alt_wh_grid, alt_ww_grid, alt_q_mask, alt_n_win = _build_window_layout(
            H_t, W_t, window, alternate_shift
        )
        self.register_buffer("alt_win_h_grid", alt_wh_grid, persistent=False)
        self.register_buffer("alt_win_w_grid", alt_ww_grid, persistent=False)
        self.register_buffer("alt_win_q_mask", alt_q_mask, persistent=False)
        self.alt_shift = int(alternate_shift)
        self.alt_n_win = int(alt_n_win)
        self.alt_n_w = int(math.ceil((W_t + alternate_shift) / window))
        self._runtime_layout_cache = {}

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def _voxel_window_id(
        self,
        voxel_h_t: torch.Tensor,
        voxel_w_t: torch.Tensor,
        *,
        shift: int,
        n_w: int,
    ) -> torch.Tensor:
        """Map per-voxel (h_t, w_t) to a frame-relative flat window id."""
        wh = (voxel_h_t + shift) // self.window
        ww = (voxel_w_t + shift) // self.window
        return wh * n_w + ww

    def _select_window_layout(
        self,
        shift: Optional[int],
        *,
        H_t: Optional[int] = None,
        W_t: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        resolved_shift = self.shift if shift is None else int(shift)
        resolved_h = self.H_t if H_t is None else int(H_t)
        resolved_w = self.W_t if W_t is None else int(W_t)
        if resolved_h <= 0 or resolved_w <= 0:
            raise ValueError(
                f"runtime token grid must be positive, got {(resolved_h, resolved_w)}."
            )
        if (resolved_h, resolved_w) != (self.H_t, self.W_t):
            if resolved_shift not in (self.shift, self.alt_shift):
                raise ValueError(
                    f"shift must be {self.shift} or {self.alt_shift}; got "
                    f"{resolved_shift}."
                )
            target_device = self.win_h_grid.device if device is None else device
            cache_key = (
                resolved_h,
                resolved_w,
                resolved_shift,
                target_device,
            )
            layout = self._runtime_layout_cache.get(cache_key)
            if layout is None:
                wh_grid, ww_grid, q_mask, n_win = _build_window_layout(
                    resolved_h,
                    resolved_w,
                    self.window,
                    resolved_shift,
                )
                layout = (
                    wh_grid.to(device=target_device),
                    ww_grid.to(device=target_device),
                    q_mask.to(device=target_device),
                    int(n_win),
                    int(math.ceil((resolved_w + resolved_shift) / self.window)),
                )
                self._runtime_layout_cache[cache_key] = layout
            return (*layout, resolved_shift)
        if resolved_shift == self.shift:
            return (
                self.win_h_grid,
                self.win_w_grid,
                self.win_q_mask,
                self.n_win,
                self.n_w,
                resolved_shift,
            )
        if resolved_shift == self.alt_shift:
            return (
                self.alt_win_h_grid,
                self.alt_win_w_grid,
                self.alt_win_q_mask,
                self.alt_n_win,
                self.alt_n_w,
                resolved_shift,
            )
        raise ValueError(
            f"shift must be {self.shift} or {self.alt_shift}; got {resolved_shift}."
        )

    def prepare_projected_kv(
        self,
        voxel_feat: torch.Tensor,
        voxel_frame_idx: torch.Tensor,
        voxel_h_t: torch.Tensor,
        voxel_w_t: torch.Tensor,
        voxel_valid: torch.Tensor,
    ) -> Optional[_ProjectedVoxelKV]:
        """Project only valid LiDAR tokens before any window padding.

        K/V projection is independent of image branch and window shift. Keeping
        the projected tensors compact avoids running LayerNorm and two Linear
        layers on global-max padding slots and lets shared-attention callers
        reuse the same result for regular and shifted windows.
        """
        if voxel_valid.numel() == 0 or not bool(voxel_valid.any().item()):
            return None
        valid_feat = voxel_feat[voxel_valid]
        valid_norm = self.norm_kv(valid_feat)
        return _ProjectedVoxelKV(
            frame_idx=voxel_frame_idx[voxel_valid],
            h_t=voxel_h_t[voxel_valid],
            w_t=voxel_w_t[voxel_valid],
            k=self.k_proj(valid_norm),
            v=self.v_proj(valid_norm),
        )

    def _can_use_native_varlen_flash(self, projected: torch.Tensor) -> bool:
        """Whether PyTorch's packed FlashAttention kernel supports this layer."""
        flash_available = getattr(
            torch.backends.cuda, "is_flash_attention_available", None
        )
        return bool(
            projected.is_cuda
            and projected.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim % 8 == 0
            and self.head_dim <= 256
            and flash_available is not None
            and flash_available()
            and hasattr(torch.ops.aten, "_flash_attention_forward")
        )

    def forward(
        self,
        image_feat: torch.Tensor,        # (F, H_t, W_t, D)
        voxel_feat: torch.Tensor,        # (V_total, D), already projected to D
        voxel_frame_idx: torch.Tensor,   # (V_total,) int64, 0..F-1
        voxel_h_t: torch.Tensor,         # (V_total,) int64
        voxel_w_t: torch.Tensor,         # (V_total,) int64
        voxel_valid: torch.Tensor,       # (V_total,) bool
        shift: Optional[int] = None,
    ) -> torch.Tensor:
        if image_feat.ndim != 4:
            raise RuntimeError(
                "WindowedCrossAttnLayer expects image_feat (F,H,W,D); got "
                f"{tuple(image_feat.shape)}."
            )
        return self.forward_branches(
            image_feat.unsqueeze(0),
            voxel_feat,
            voxel_frame_idx,
            voxel_h_t,
            voxel_w_t,
            voxel_valid,
            shift=shift,
        )[0]

    def forward_branches(
        self,
        image_feat: torch.Tensor,        # (R, F, H_t, W_t, D)
        voxel_feat: torch.Tensor,        # (V_total, D), shared by all branches
        voxel_frame_idx: torch.Tensor,   # (V_total,) int64, 0..F-1
        voxel_h_t: torch.Tensor,         # (V_total,) int64
        voxel_w_t: torch.Tensor,         # (V_total,) int64
        voxel_valid: torch.Tensor,       # (V_total,) bool
        shift: Optional[int] = None,
        prepared_kv: Optional[_ProjectedVoxelKV] = None,
    ) -> torch.Tensor:
        """Cross-attend multiple image branches to one packed LiDAR KV set.

        Window bucketing, sorting, padding, normalization, and K/V projection
        depend only on LiDAR. Keeping branches as an extra query dimension
        avoids repeating those operations for DA3 local/global tokens.
        """
        if image_feat.ndim != 5:
            raise RuntimeError(
                "forward_branches expects image_feat (R,F,H,W,D); got "
                f"{tuple(image_feat.shape)}."
            )
        num_branches, F_n, H_t, W_t, D = image_feat.shape
        if num_branches <= 0:
            raise RuntimeError("forward_branches requires at least one branch.")
        if D != self.d_model:
            raise RuntimeError(
                "image feature shape does not match the attention layer: "
                f"got HWD={(H_t, W_t, D)}, expected "
                f"token dim {self.d_model}."
            )
        device = image_feat.device
        wh_grid, ww_grid, win_q_mask, n_win, n_w, resolved_shift = (
            self._select_window_layout(
                shift,
                H_t=H_t,
                W_t=W_t,
                device=device,
            )
        )

        if prepared_kv is None:
            prepared_kv = self.prepare_projected_kv(
                voxel_feat,
                voxel_frame_idx,
                voxel_h_t,
                voxel_w_t,
                voxel_valid,
            )
        # If nothing valid → identity.
        if prepared_kv is None:
            return image_feat

        # Window metadata and compact projected K/V have matching row order.
        vf_idx = prepared_kv.frame_idx
        vh_t = prepared_kv.h_t
        vw_t = prepared_kv.w_t
        v_win_local = self._voxel_window_id(
            vh_t, vw_t, shift=resolved_shift, n_w=n_w
        )
        v_global = vf_idx * n_win + v_win_local                    # (V,)

        # Sort voxels by global window id so we can compute per-window slot offsets.
        v_global, order = torch.sort(v_global)
        k_compact = prepared_kv.k[order]
        v_compact = prepared_kv.v[order]
        use_varlen_flash = self._can_use_native_varlen_flash(k_compact)

        # Active windows = unique global window ids that received voxels.
        if use_varlen_flash:
            active, counts = torch.unique_consecutive(
                v_global, return_counts=True
            )
        else:
            active, inv, counts = torch.unique_consecutive(
                v_global, return_inverse=True, return_counts=True
            )
        n_active = int(active.shape[0])
        M_KV = int(counts.max().item())

        if not use_varlen_flash:
            # Slot index within each active window via cumulative group offsets.
            starts = torch.zeros_like(counts)
            starts[1:] = counts.cumsum(0)[:-1]
            slot = torch.arange(v_global.shape[0], device=device) - starts[inv]

        # Pack Q from every image branch. For each active window, gather its
        # M_Q patch tokens once per branch without duplicating the LiDAR KV.
        active_frame = active // n_win
        active_local = active % n_win

        wh_grid_sel = wh_grid[active_local]  # (n_active, M_Q)
        ww_grid_sel = ww_grid[active_local]
        q_mask = win_q_mask[active_local]    # (n_active, M_Q) bool
        # Replace invalid slot coords with 0 (safe gather; we'll mask them).
        wh_safe = wh_grid_sel
        ww_safe = ww_grid_sel
        q_linear = (
            (active_frame.unsqueeze(-1) * H_t + wh_safe) * W_t + ww_safe
        )
        image_flat = image_feat.reshape(num_branches, F_n * H_t * W_t, D)

        Hh = self.num_heads
        Dh = self.head_dim
        valid_pos = q_mask.reshape(-1)
        dst = q_linear.reshape(-1)[valid_pos]
        q_counts = q_mask.sum(dim=1, dtype=torch.int32)

        # On supported CUDA dtypes, use PyTorch's packed varlen FlashAttention
        # operator. It consumes compact Q/K/V plus cumulative window lengths,
        # so neither query-boundary slots nor per-window KV padding participates
        # in attention. CPU, FP32, and unsupported GPUs keep the padded SDPA
        # fallback below.
        if use_varlen_flash:
            k_flash = k_compact.view(-1, Hh, Dh)
            v_flash = v_compact.view(-1, Hh, Dh)
            cu_q = F.pad(q_counts.cumsum(0, dtype=torch.int32), (1, 0))
            kv_counts = counts.to(dtype=torch.int32)
            cu_kv = F.pad(kv_counts.cumsum(0, dtype=torch.int32), (1, 0))
            max_q = int(q_counts.max().item())
        else:
            # Pack already-projected K/V. Padding slots never pass through the
            # expensive LayerNorm/Linear projections and remain masked in SDPA.
            k_pad = k_compact.new_zeros((n_active, M_KV, D))
            v_pad = v_compact.new_zeros((n_active, M_KV, D))
            k_pad[inv, slot] = k_compact
            v_pad[inv, slot] = v_compact
            kv_mask = (
                torch.arange(M_KV, device=device).unsqueeze(0)
                < counts.unsqueeze(1)
            )
            k = k_pad.view(n_active, M_KV, Hh, Dh).transpose(1, 2)
            v = v_pad.view(n_active, M_KV, Hh, Dh).transpose(1, 2)
            # True means the KV position may participate. Every active window
            # has at least one valid KV row, so SDPA never sees an all-False row.
            attn_mask = kv_mask.view(n_active, 1, 1, M_KV).expand(
                n_active, 1, self.M_Q, M_KV
            )

        branch_outputs: List[torch.Tensor] = []
        for branch_idx in range(num_branches):
            # Project only real image positions. The varlen path keeps these
            # rows compact; the fallback scatters them into padded SDPA input.
            branch_q = image_flat[branch_idx, dst]
            q_valid = self.q_proj(self.norm_q(branch_q))
            if use_varlen_flash:
                attn_out = torch.ops.aten._flash_attention_forward(
                    q_valid.view(-1, Hh, Dh),
                    k_flash,
                    v_flash,
                    cu_q,
                    cu_kv,
                    max_q,
                    M_KV,
                    self.attn_dropout if self.training else 0.0,
                    False,
                    False,
                )[0].reshape(-1, D)
            else:
                q_pad = q_valid.new_zeros((n_active * self.M_Q, D))
                q_pad[valid_pos] = q_valid
                q = q_pad.view(n_active, self.M_Q, D)
                q = q.view(n_active, self.M_Q, Hh, Dh).transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout if self.training else 0.0,
                )
                attn_out = attn_out.transpose(1, 2).contiguous().view(-1, D)
                attn_out = attn_out[valid_pos]
            attn_out = self.out_proj(attn_out)
            q_res = branch_q + attn_out
            ffn_out = self.ffn(self.norm_ffn(q_res))
            update = attn_out + ffn_out

            out = image_flat[branch_idx].clone()
            out[dst] = out[dst] + update
            branch_outputs.append(out.view(F_n, H_t, W_t, D))
        return torch.stack(branch_outputs, dim=0)

class SharedDualBranchLidarImageFusionModule(nn.Module):
    def __init__(self, *, d_model=768, num_heads=8, H_t=12, W_t=37,
                 patch_size=14, window=4, vox_origin=(-25.6, -2.0, 0.0),
                 vox_size=(0.4, 0.4, 0.4), vox_grid=(128, 16, 128),
                 vfe_d_voxel=128, vfe_hidden=64, pe_num_freqs=8,
                 ffn_ratio=2.0):
        super().__init__()
        self.d_model, self.H_t, self.W_t = d_model, H_t, W_t
        self.patch_size = patch_size
        self.vfe = VoxelFeatureEncoder(
            vox_origin=vox_origin, vox_size=vox_size, vox_grid=vox_grid,
            d_voxel=vfe_d_voxel, d_out=d_model, hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
        )
        self.attention = WindowedCrossAttnLayer(
            d_model=d_model, num_heads=num_heads, H_t=H_t, W_t=W_t,
            window=window, shift=0, ffn_ratio=ffn_ratio,
        )
        # The source constructs this branch before discarding it; keep its
        # RNG consumption without registering unused trainable parameters.
        WindowedCrossAttnLayer(
            d_model=d_model, num_heads=num_heads, H_t=H_t, W_t=W_t,
            window=window, shift=window // 2, ffn_ratio=ffn_ratio,
        )
        self.window_shift = self.attention.window // 2

    def _apply_shared_attention(
        self,
        image_feat: torch.Tensor,
        voxel_feat: torch.Tensor,
        voxel_frame_idx: torch.Tensor,
        voxel_h_t: torch.Tensor,
        voxel_w_t: torch.Tensor,
        voxel_valid: torch.Tensor,
    ) -> torch.Tensor:
        return self._apply_shared_attention_branches(
            image_feat.unsqueeze(0),
            voxel_feat,
            voxel_frame_idx,
            voxel_h_t,
            voxel_w_t,
            voxel_valid,
        )[0]

    def _apply_shared_attention_branches(
        self,
        image_feat: torch.Tensor,
        voxel_feat: torch.Tensor,
        voxel_frame_idx: torch.Tensor,
        voxel_h_t: torch.Tensor,
        voxel_w_t: torch.Tensor,
        voxel_valid: torch.Tensor,
    ) -> torch.Tensor:
        prepared_kv = self.attention.prepare_projected_kv(
            voxel_feat,
            voxel_frame_idx,
            voxel_h_t,
            voxel_w_t,
            voxel_valid,
        )
        if prepared_kv is None:
            return image_feat
        for shift in (0, self.window_shift):
            image_feat = self.attention.forward_branches(
                image_feat,
                voxel_feat,
                voxel_frame_idx,
                voxel_h_t,
                voxel_w_t,
                voxel_valid,
                shift=shift,
                prepared_kv=prepared_kv,
            )
        return image_feat

class SharedDualBranchDepthBackprojectionFusionModule(SharedDualBranchLidarImageFusionModule):
    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 8,
        H_t: int = 12,
        W_t: int = 37,
        patch_size: int = 14,
        window: int = 4,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        depth_bin_size: float = 4.0,
        vfe_d_voxel: int = 128,
        vfe_hidden: int = 64,
        pe_num_freqs: int = 8,
        ffn_ratio: float = 2.0,
        dynamic_image_size: bool = False,
    ) -> None:
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            H_t=H_t,
            W_t=W_t,
            patch_size=patch_size,
            window=window,
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            vfe_d_voxel=vfe_d_voxel,
            vfe_hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
            ffn_ratio=ffn_ratio,
        )
        # The parent creates the Cartesian VFE before constructing the shared
        # attention. It is deliberately removed from the registered model so
        # this experiment contains only the depth-backprojection encoder.
        del self.vfe
        self.depth_encoder = PatchDepthBinFeatureEncoder(
            d_out=d_model,
            H_t=H_t,
            W_t=W_t,
            patch_size=patch_size,
            depth_bin_size=depth_bin_size,
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            d_token=vfe_d_voxel,
            hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
            dynamic_image_size=dynamic_image_size,
        )
        self.depth_bin_size = float(depth_bin_size)

    def encode_depth_tokens(
        self,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        K_per_frame: torch.Tensor,
        *,
        reference: torch.Tensor,
        fusion_vox_origin: Optional[torch.Tensor] = None,
        fusion_vox_size: Optional[torch.Tensor] = None,
        fusion_vox_grid: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        encoded = self.depth_encoder(
            sparse_depth.to(device=reference.device),
            sparse_mask.to(device=reference.device),
            K_per_frame.to(device=reference.device),
            vox_origin=fusion_vox_origin,
            vox_size=fusion_vox_size,
            vox_grid=fusion_vox_grid,
        )
        feat, center, frame_idx, patch_y, patch_x = encoded
        if feat is None:
            return (
                reference.new_zeros((0, self.d_model)),
                reference.new_zeros((0, 3), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long, device=reference.device),
                torch.zeros((0,), dtype=torch.long, device=reference.device),
                torch.zeros((0,), dtype=torch.long, device=reference.device),
                torch.zeros((0,), dtype=torch.bool, device=reference.device),
            )
        assert center is not None
        assert frame_idx is not None
        assert patch_y is not None
        assert patch_x is not None
        valid = torch.ones_like(frame_idx, dtype=torch.bool)
        return (
            feat.to(dtype=reference.dtype),
            center,
            frame_idx,
            patch_y,
            patch_x,
            valid,
        )

class SharedDualBranchDepthTokenFusionModule(SharedDualBranchLidarImageFusionModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This path consumes already embedded image-aligned depth tokens and
        # therefore must not register the Cartesian point-cloud encoder.
        del self.vfe

    def encode_depth_tokens(
        self,
        depth_tokens: torch.Tensor,
        valid_depth_patches: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select valid patch embeddings on the runtime image grid for attention."""
        if depth_tokens.ndim != 4 or reference.ndim != 4:
            raise RuntimeError("Expected depth tokens (B,N,L,D) and reference (B*N,H_t,W_t,D)")
        B, N, L, D = depth_tokens.shape
        BN, H_t, W_t, reference_dim = reference.shape
        if (B * N, L, D) != (BN, H_t * W_t, reference_dim) or D != self.d_model:
            raise RuntimeError("Depth patch embeddings do not match the runtime image token grid")
        if tuple(valid_depth_patches.shape) != (B, N, L):
            raise RuntimeError("Depth patch mask does not match depth embeddings")
        depth_grid = depth_tokens.reshape(BN, H_t, W_t, D)
        valid_grid = valid_depth_patches.to(device=depth_grid.device, dtype=torch.bool).reshape(BN, H_t, W_t)
        depth_index = torch.nonzero(valid_grid, as_tuple=False)
        frame_idx, patch_y, patch_x = depth_index.unbind(dim=1)
        features = depth_grid[frame_idx, patch_y, patch_x].to(dtype=reference.dtype)
        return (features, reference.new_zeros((features.shape[0], 3)), frame_idx,
                patch_y, patch_x, torch.ones_like(frame_idx, dtype=torch.bool))

class VoxelDepthDualWindowSharedBranchFusionModule(nn.Module):
    """Fuse DA3 branches with voxel then depth SharedDual window attention.

    The voxel and depth attention blocks have independent parameters. Within
    each block, the same parameters are shared across the DA3 local/global
    branches and reused for regular then shifted windows. Voxel tokens update
    both branches first; depth tokens then update the result. The DDAD surround
    variant uses patch-by-depth-bin voxels and sparse log-depth patch embeddings,
    matching both pre-fusion representations with separate trainable encoders.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_heads: int = 8,
        H_t: int = 12,
        W_t: int = 37,
        patch_size: int = 14,
        window: int = 4,
        vox_origin: Tuple[float, float, float] = (-25.6, -2.0, 0.0),
        vox_size: Tuple[float, float, float] = (0.4, 0.4, 0.4),
        vox_grid: Tuple[int, int, int] = (128, 16, 128),
        depth_bin_size: float = 4.0,
        vfe_d_voxel: int = 128,
        vfe_hidden: int = 64,
        pe_num_freqs: int = 8,
        ffn_ratio: float = 2.0,
        dynamic_image_size: bool = False,
        voxel_token_source: str = "cartesian",
        depth_token_source: str = "patch_depth4m",
        enable_voxel: bool = True,
        enable_depth: bool = True,
    ) -> None:
        super().__init__()
        if voxel_token_source not in ("cartesian", "patch_depth4m"):
            raise ValueError(f"Unknown voxel_token_source: {voxel_token_source!r}")
        self.voxel_token_source = voxel_token_source
        if depth_token_source not in ("patch_depth4m", "sparse_log_patch"):
            raise ValueError(f"Unknown depth_token_source: {depth_token_source!r}")
        self.depth_token_source = depth_token_source
        self.enable_voxel = bool(enable_voxel)
        self.enable_depth = bool(enable_depth)
        if not (self.enable_voxel or self.enable_depth):
            raise ValueError("At least one post-fusion modality must be enabled")
        self.d_model = int(d_model)
        self.H_t = int(H_t)
        self.W_t = int(W_t)
        self.dynamic_image_size = bool(dynamic_image_size)
        voxel_cls = SharedDualBranchLidarImageFusionModule
        voxel_kwargs = {}
        if voxel_token_source == "patch_depth4m":
            voxel_cls = SharedDualBranchDepthBackprojectionFusionModule
            voxel_kwargs = dict(depth_bin_size=depth_bin_size,
                                dynamic_image_size=dynamic_image_size)
        self.voxel_fusion = voxel_cls(
            d_model=d_model,
            num_heads=num_heads,
            H_t=H_t,
            W_t=W_t,
            patch_size=patch_size,
            window=window,
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            vfe_d_voxel=vfe_d_voxel,
            vfe_hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
            ffn_ratio=ffn_ratio,
            **voxel_kwargs,
        ) if self.enable_voxel else None
        depth_cls = SharedDualBranchDepthBackprojectionFusionModule
        depth_kwargs = dict(depth_bin_size=depth_bin_size, dynamic_image_size=dynamic_image_size)
        if depth_token_source == "sparse_log_patch":
            depth_cls = SharedDualBranchDepthTokenFusionModule
            depth_kwargs = {}
        self.depth_fusion = depth_cls(
            d_model=d_model,
            num_heads=num_heads,
            H_t=H_t,
            W_t=W_t,
            patch_size=patch_size,
            window=window,
            vox_origin=vox_origin,
            vox_size=vox_size,
            vox_grid=vox_grid,
            vfe_d_voxel=vfe_d_voxel,
            vfe_hidden=vfe_hidden,
            pe_num_freqs=pe_num_freqs,
            ffn_ratio=ffn_ratio,
            **depth_kwargs,
        ) if self.enable_depth else None

    def forward(
        self,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        *,
        points_per_frame: Optional[List[List[torch.Tensor]]] = None,
        T_cam_from_velo: Optional[torch.Tensor] = None,
        K_per_frame: torch.Tensor,
        image_hw: torch.Tensor,
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        fusion_vox_origin: Optional[torch.Tensor] = None,
        fusion_vox_size: Optional[torch.Tensor] = None,
        fusion_vox_grid: Optional[Tuple[int, int, int]] = None,
        depth_token_drop_ratio: float = 0.0,
        voxel_token_drop_ratio: float = 0.0,
        token_dropout_seed: int = 0,
        token_sample_keys: Optional[List[str]] = None,
        depth_tokens: Optional[torch.Tensor] = None,
        valid_depth_patches: Optional[torch.Tensor] = None,
        voxel_encoded: Optional[Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
        ]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if depth_token_drop_ratio or voxel_token_drop_ratio:
            raise ValueError("Full KITTI fixes token dropout to zero")
        if local_tokens.shape != global_tokens.shape:
            raise RuntimeError(
                "local/global token shapes must match; got "
                f"{tuple(local_tokens.shape)} vs {tuple(global_tokens.shape)}."
            )
        B, N, H_t, W_t, D = local_tokens.shape
        expected_grid = (self.H_t, self.W_t)
        expected_shape = (
            ("dynamic", "dynamic", self.d_model)
            if self.dynamic_image_size
            else (self.H_t, self.W_t, self.d_model)
        )
        if D != self.d_model or (
            not self.dynamic_image_size and (H_t, W_t) != expected_grid
        ):
            raise RuntimeError(
                "token grid/dim does not match voxel-depth fusion: got "
                f"{(H_t, W_t, D)}, expected {expected_shape}."
            )
        if self.dynamic_image_size and (H_t <= 0 or W_t <= 0):
            raise RuntimeError(
                f"dynamic voxel-depth fusion requires a positive token grid, got "
                f"{(H_t, W_t)}."
            )
        if tuple(sparse_depth.shape[:2]) != (B, N):
            raise RuntimeError(
                "sparse_depth batch/frame shape must match tokens; got "
                f"{tuple(sparse_depth.shape[:2])} vs {(B, N)}."
            )

        local_feat = local_tokens.reshape(B * N, H_t, W_t, D).contiguous()
        global_feat = global_tokens.reshape(B * N, H_t, W_t, D).contiguous()
        branch_feat = torch.stack([local_feat, global_feat], dim=0)

        if self.enable_voxel:
            if voxel_encoded is not None:
                # Full-image RAW VFE callers already filtered padding and
                # visibility; consume that encoding without another projection.
                if self.voxel_token_source != "cartesian":
                    raise ValueError("Pre-encoded RAW voxels require the Cartesian voxel source")
            elif self.voxel_token_source == "patch_depth4m":
                # Use the same projected sparse-depth geometry and encoder as
                # pre-fusion. This encoder is independent of depth_fusion's encoder.
                voxel_encoded = self.voxel_fusion.encode_depth_tokens(
                    sparse_depth,
                    sparse_mask,
                    K_per_frame,
                    reference=local_feat,
                    fusion_vox_origin=fusion_vox_origin,
                    fusion_vox_size=fusion_vox_size,
                    fusion_vox_grid=fusion_vox_grid,
                )
            else:
                if points_per_frame is None or T_cam_from_velo is None:
                    raise ValueError("Cartesian voxel fusion requires RAW points and calibration")
                raise ValueError("RAW VFE requires pre-encoded visible voxels")
            (
                voxel_feat,
                _voxel_center,
                voxel_frame_idx,
                voxel_h_t,
                voxel_w_t,
                voxel_valid,
            ) = voxel_encoded
            if voxel_feat.shape[0] > 0:
                branch_feat = self.voxel_fusion._apply_shared_attention_branches(
                    branch_feat,
                    voxel_feat,
                    voxel_frame_idx,
                    voxel_h_t,
                    voxel_w_t,
                    voxel_valid,
                )

        if self.enable_depth:
            if self.depth_token_source == "sparse_log_patch":
                if depth_tokens is None or valid_depth_patches is None:
                    raise RuntimeError("Sparse log-depth fusion requires patch embeddings and a valid-patch mask")
                depth_encoded = self.depth_fusion.encode_depth_tokens(
                    depth_tokens, valid_depth_patches, reference=branch_feat[0],
                )
            else:
                depth_encoded = self.depth_fusion.encode_depth_tokens(
                    sparse_depth,
                    sparse_mask,
                    K_per_frame,
                    reference=branch_feat[0],
                    fusion_vox_origin=fusion_vox_origin,
                    fusion_vox_size=fusion_vox_size,
                    fusion_vox_grid=fusion_vox_grid,
                )
            (
                depth_feat,
                _depth_center,
                depth_frame_idx,
                depth_h_t,
                depth_w_t,
                depth_valid,
            ) = depth_encoded
            if depth_feat.shape[0] > 0:
                branch_feat = self.depth_fusion._apply_shared_attention_branches(
                    branch_feat,
                    depth_feat,
                    depth_frame_idx,
                    depth_h_t,
                    depth_w_t,
                    depth_valid,
                )

        local_feat, global_feat = branch_feat.unbind(dim=0)
        shape = (B, N, H_t, W_t, D)
        return local_feat.view(shape), global_feat.view(shape)
