"""Behavior checks for full KITTI shared local/global window fusion."""
import pytest
import torch

from occany_depth_min.kitti_dc_full.fusion import (
    VoxelDepthDualWindowSharedBranchFusionModule,
)
from occany_depth_min.kitti_dc_full.model import validate_preprojected_sparse_depth


def fusion(**kwargs):
    return VoxelDepthDualWindowSharedBranchFusionModule(
        d_model=8, num_heads=2, H_t=2, W_t=3, patch_size=14,
        voxel_token_source="patch_depth4m", depth_token_source="sparse_log_patch",
        dynamic_image_size=True, **kwargs,
    )


def inputs(height=2, width=3, empty=False):
    local = torch.randn(1, 1, height, width, 8, requires_grad=True)
    global_ = torch.randn_like(local, requires_grad=True)
    depth = torch.zeros(1, 1, height * 14, width * 14)
    valid = torch.zeros(1, 1, height * width, dtype=torch.bool)
    if not empty:
        depth[0, 0, 3, 4] = 5
        depth[0, 0, -2, -3] = 10
        valid[0, 0, 0] = valid[0, 0, -1] = True
    return local, global_, dict(
        K_per_frame=torch.tensor([[[[100., 0., width * 7], [0., 100., height * 7], [0., 0., 1.]]]]),
        image_hw=torch.tensor([[height * 14, width * 14]]),
        sparse_depth=depth, sparse_mask=depth > 0,
        depth_tokens=torch.randn(1, 1, height * width, 8, requires_grad=True),
        valid_depth_patches=valid,
    )


def test_fusion_reuses_attention_voxel_then_depth_on_dynamic_grids(monkeypatch):
    model = fusion()
    calls = []
    for branch in ("voxel", "depth"):
        attention = getattr(model, branch + "_fusion").attention
        original = attention.forward_branches

        def record(*args, branch=branch, original=original, **kwargs):
            calls.append((branch, kwargs["shift"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(attention, "forward_branches", record)
    for height, width in ((2, 3), (3, 2)):
        local, global_, kwargs = inputs(height, width)
        result = model(local, global_, **kwargs)
        assert calls == [("voxel", 0), ("voxel", 2), ("depth", 0), ("depth", 2)]
        calls.clear()
        assert result[0].shape == local.shape and result[1].shape == global_.shape
        (result[0].square().mean() + result[1].square().mean()).backward()
        assert local.grad.abs().sum() > 0 and global_.grad.abs().sum() > 0
        for branch in ("voxel", "depth"):
            params = getattr(model, branch + "_fusion").parameters()
            grads = [p.grad for p in params if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
        model.zero_grad(set_to_none=True)


def test_empty_conditioning_preserves_rgb_branches_exactly():
    local, global_, kwargs = inputs(empty=True)
    result = fusion()(local, global_, **kwargs)
    torch.testing.assert_close(result[0], local, atol=0, rtol=0)
    torch.testing.assert_close(result[1], global_, atol=0, rtol=0)


@pytest.mark.parametrize("branch", ["voxel", "depth"])
def test_disabled_branch_registers_no_parameters(branch):
    model = fusion(**{"enable_" + branch: False})
    assert not any(name.startswith(branch + "_fusion.") for name, _ in model.named_parameters())
    local, global_, kwargs = inputs()
    assert all(torch.isfinite(value).all() for value in model(local, global_, **kwargs))


@pytest.mark.parametrize("bad", ["missing", "mask", "negative", "nan"])
def test_shared_projection_rejects_inconsistent_maps(bad):
    images = torch.zeros(1, 1, 3, 28, 42)
    depth = torch.zeros(1, 1, 28, 42)
    depth[0, 0, 3, 4] = 5
    mask = depth > 0
    if bad == "missing":
        mask = None
    elif bad == "mask":
        mask.zero_()
    elif bad == "negative":
        depth[0, 0, 1, 1] = -1
    else:
        depth[0, 0, 1, 1] = torch.nan
    with pytest.raises(ValueError):
        validate_preprojected_sparse_depth(depth, mask, images)
