"""Representation checks for the RAW VFE branch used by full KITTI."""
import pytest
import torch

from occany_depth_min.kitti_dc_full.raw_voxel import (
    VoxelFeatureEncoder, encode_projected_raw_voxel_tokens,
)


def encoder():
    torch.manual_seed(0)
    return VoxelFeatureEncoder(
        d_out=16, d_voxel=8, hidden=8, force_fp32_geometry=True,
    )


def project(vfe, points, mask=None):
    images = torch.zeros(1, 1, 3, 28, 42)
    intrinsics = torch.tensor([[[[20., 0., 21.], [0., 20., 14.], [0., 0., 1.]]]])
    return encode_projected_raw_voxel_tokens(
        vfe, images, [[points]], torch.eye(4)[None, None], intrinsics,
        torch.ones(1, 1, 28, 42, dtype=torch.bool) if mask is None else mask,
        patch_size=14,
    )


def test_raw_projection_keeps_both_depths_without_z_buffer():
    # Two distinct camera voxels project to one pixel/patch. Both contribute
    # to window fusion, independently of the sparse-depth branch's z-buffer.
    features, centers, frames, rows, cols = project(
        encoder(), torch.tensor([[0., 0., 5., .2], [0., 0., 10., .7]]),
    )
    assert features.shape == (2, 16)
    assert (frames == 0).all() and (rows == 1).all() and (cols == 1).all()
    assert centers[0, 2] != centers[1, 2]


def test_raw_intensity_changes_features_and_backpropagates():
    vfe = encoder()
    points = torch.tensor([[0., 0., 5., .2], [1., 1., 10., .7]])
    first = project(vfe, points)[0]
    changed = points.clone()
    changed[:, 3] += 1
    second = project(vfe, changed)[0]
    assert not torch.equal(first, second)
    first.square().mean().backward()
    grad = vfe.point_mlp[0].weight.grad[:, 3]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_raw_projection_excludes_padding_and_handles_empty_frames():
    vfe = encoder()
    features = project(vfe, torch.empty(0, 4))
    assert features == (None, None, None, None, None)
    mask = torch.ones(1, 1, 28, 42, dtype=torch.bool)
    mask[:, :, 14:] = False
    features, _, frames, rows, cols = project(
        vfe, torch.tensor([[0., 0., 5., .2]]), mask,
    )
    assert features.shape == (0, 16)
    assert frames.numel() == rows.numel() == cols.numel() == 0


def test_vfe_geometry_stays_float32_under_bfloat16_autocast():
    vfe = encoder()
    points = torch.tensor([[.21, .19, 5.01, .2], [1.1, .9, 10.01, .7]])
    baseline = project(vfe, points)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = project(vfe, points)
    for index in range(1, 5):
        torch.testing.assert_close(baseline[index], mixed[index], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["points", "transform", "mask"])
def test_raw_projection_requires_all_geometry_inputs(missing):
    arguments = dict(
        points_per_frame=[[torch.empty(0, 4)]],
        T_cam_from_velo=torch.eye(4)[None, None],
        image_valid_mask=torch.ones(1, 1, 28, 42, dtype=torch.bool),
    )
    names = dict(points="points_per_frame", transform="T_cam_from_velo", mask="image_valid_mask")
    arguments[names[missing]] = None
    with pytest.raises(ValueError, match="RAW VFE requires"):
        encode_projected_raw_voxel_tokens(
            encoder(), torch.zeros(1, 1, 3, 28, 42),
            K_per_frame=torch.eye(3)[None, None], patch_size=14, **arguments,
        )
