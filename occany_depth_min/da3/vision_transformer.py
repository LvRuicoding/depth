"""Small DA3 DINOv2 transformer used by the released experiment.

This is the runtime subset of Depth Anything 3's Apache-2.0 DINOv2 fork.
"""
from __future__ import annotations

import math
import torch
from einops import rearrange
from torch import nn

from .layers import Block, Mlp, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D


class DinoVisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embed_dim, depth, num_heads = 384, 12, 6
        self.patch_start_idx = 1
        self.num_features = self.embed_dim = embed_dim
        self.alt_start = self.qknorm_start = self.rope_start = 4
        self.cat_token = True
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = 14
        self.num_register_tokens = 0
        self.interpolate_antialias = False
        self.interpolate_offset = 0.1
        self.patch_embed = PatchEmbed(
            img_size=518, patch_size=14, in_chans=3, embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.camera_token = nn.Parameter(torch.randn(1, 2, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.register_tokens = None
        self.rope = RotaryPositionEmbedding2D(frequency=100)
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=nn.LayerNorm,
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=1.0,
                    qk_norm=index >= 4,
                    rope=self.rope if index >= 4 else None,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def interpolate_pos_encoding(
        self, x: torch.Tensor, width: int, height: int
    ) -> torch.Tensor:
        previous_dtype = x.dtype
        patch_count = x.shape[1] - 1
        source_count = self.pos_embed.shape[1] - 1
        if patch_count == source_count and width == height:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        width_patches = width // self.patch_size
        height_patches = height // self.patch_size
        source_side = int(math.sqrt(source_count))
        if source_count != source_side * source_side:
            raise RuntimeError("DA3 absolute position grid is not square.")
        scale = (
            float(width_patches + self.interpolate_offset) / source_side,
            float(height_patches + self.interpolate_offset) / source_side,
        )
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, source_side, source_side, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            scale_factor=scale,
        )
        if patch_pos_embed.shape[-2:] != (width_patches, height_patches):
            raise RuntimeError(
                f"Interpolated position grid {patch_pos_embed.shape[-2:]} != "
                f"{(width_patches, height_patches)}."
            )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_cls_token(self, batch_size: int, views: int) -> torch.Tensor:
        return self.cls_token.expand(batch_size, views, -1).reshape(
            batch_size * views, 1, self.embed_dim
        )

    @staticmethod
    def process_attention(
        x: torch.Tensor,
        block: nn.Module,
        attention_type: str = "global",
        pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, views, _tokens = x.shape[:3]
        if attention_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attention_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attention_type}")
        x = block(x, pos=pos, attn_mask=attn_mask)
        if attention_type == "local":
            return rearrange(x, "(b s) n c -> b s n c", b=batch_size, s=views)
        return rearrange(x, "b (s n) c -> b s n c", b=batch_size, s=views)


def vit_small() -> DinoVisionTransformer:
    return DinoVisionTransformer()
