import hashlib
import os
from pathlib import Path

import pytest
import torch

from occany_depth_min.model import (
    DA3_BASE_MODEL_VARIANTS,
    build_model,
    get_model_spec,
    load_trained_checkpoint,
)
from occany_depth_min.train import KITTI_EPOCHS, _fixed_args, parser as train_parser


REFERENCE = Path(os.environ.get("OCCANY_REFERENCE_CHECKPOINT", "__missing__"))
POSTFUSION_REFERENCE = Path(
    os.environ.get("OCCANY_POSTFUSION_REFERENCE_CHECKPOINT", "__missing__")
)


def test_state_dict_contract() -> None:
    model = build_model(load_base=False)
    assert len(model.state_dict()) == 315
    assert sum(parameter.numel() for parameter in model.parameters()) == 29_570_945


def test_reference_checkpoint_strict_load() -> None:
    if not REFERENCE.is_file():
        pytest.skip("set OCCANY_REFERENCE_CHECKPOINT to exercise strict compatibility")
    model = build_model(load_base=False)
    payload = load_trained_checkpoint(model, REFERENCE)
    assert payload["epoch"] == 9


def test_postfusion_state_dict_contract() -> None:
    torch.manual_seed(12345)
    model = build_model(load_base=False, variant="postfusion")
    assert len(model.state_dict()) == 351
    assert sum(parameter.numel() for parameter in model.parameters()) == 31_940_225
    expected = {
        "fusion.": "367d4dca203af487d03198a99124a6226ae0db5993188cda7488d153ec0d9e2d",
        "dense_depth_head.": "c166855a88394ced346f0f62f753935b5a408552c5cfa18bfa45d5e6f0e8d4c7",
        "depth_patch_embed.": "773c393a19161e4f1b52efd27e1526032856e16a26bb07df92cb8a9a3d88ab08",
    }
    state = model.state_dict()
    for prefix, expected_hash in expected.items():
        digest = hashlib.sha256()
        for key, value in state.items():
            if key.startswith(prefix):
                digest.update(key.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        assert digest.hexdigest() == expected_hash


def test_postfusion_reference_checkpoint_strict_load() -> None:
    if not POSTFUSION_REFERENCE.is_file():
        pytest.skip(
            "set OCCANY_POSTFUSION_REFERENCE_CHECKPOINT to exercise strict compatibility"
        )
    model = build_model(load_base=False, variant="postfusion")
    payload = load_trained_checkpoint(model, POSTFUSION_REFERENCE)
    assert payload["epoch"] == 9


def test_kitti_only_training_contracts() -> None:
    expected = {
        "prefusion": (
            "depth_lingbot_da3_last_patchdepth4m_voxel_prefusion_promptda_scaled_kitti",
            "da3_small_only_random_lingbot_voxel_promptda_last_kitti_v1",
        ),
        "postfusion": (
            "depth_patchdepth4m_voxeldepth_dualwindow_postfusion_only_"
            "promptda_scaled_prefusion_aligned",
            "da3_small_only_seeded_logdepth_patchdepth4m_voxel_dualwindow_"
            "promptda_dpt_online_knn_scaled_kitti_v1",
        ),
    }
    for variant, (experiment, initialization) in expected.items():
        spec = get_model_spec(variant, "kitti")
        assert spec.experiment == experiment
        assert spec.initialization_contract == initialization
        args = train_parser().parse_args(
            [
                "--dataset",
                "kitti",
                "--model",
                variant,
                "--da3-checkpoint",
                "base",
                "--kitti-root",
                "data",
                "--output-dir",
                "output",
            ]
        )
        fixed = _fixed_args(args, 4)
        assert fixed["epochs"] == KITTI_EPOCHS == 20
        assert fixed["dataset"] == "kitti"
        assert fixed["dynamic_image_size"] is None
        assert fixed["online_knn_contract"].startswith("kitti_stage1_lidar_zbuffer")


@pytest.mark.parametrize(
    ("variant", "keys", "parameters"),
    (
        ("image", 269, 93_502_785),
        ("depth", 271, 93_654_081),
        ("voxel", 281, 93_648_641),
        ("voxel_depth", 283, 93_799_937),
        ("image_scaled", 301, 94_754_625),
        ("depth_scaled", 303, 94_905_921),
        ("voxel_scaled", 313, 94_900_481),
        ("voxel_depth_scaled", 315, 95_051_777),
    ),
)
def test_da3_base_state_dict_contracts(
    variant: str, keys: int, parameters: int
) -> None:
    model = build_model(load_base=False, variant=variant)
    spec = get_model_spec(variant, "kitti")
    assert len(model.state_dict()) == keys
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters
    assert model.da3_model_name == "da3-base"
    assert model.native_dim == 768
    assert model.feature_dim == spec.token_dim == 1536
    assert hasattr(model, "depth_patch_embed") is spec.uses_depth_tokens
    assert hasattr(model, "voxel_token_encoder") is spec.uses_voxel_tokens


def test_da3_base_variants_are_kitti_only() -> None:
    assert len(DA3_BASE_MODEL_VARIANTS) == 8
    for variant in DA3_BASE_MODEL_VARIANTS:
        with pytest.raises(ValueError, match="not available for unified6"):
            get_model_spec(variant, "unified6")


def test_da3_base_kitti_training_metadata() -> None:
    for variant in DA3_BASE_MODEL_VARIANTS:
        args = train_parser().parse_args(
            [
                "--dataset",
                "kitti",
                "--model",
                variant,
                "--da3-checkpoint",
                "base",
                "--kitti-root",
                "data",
                "--output-dir",
                "output",
            ]
        )
        fixed = _fixed_args(args, 4)
        spec = get_model_spec(variant, "kitti")
        assert fixed["da3_model_name"] == "da3-base"
        assert fixed["token_dim"] == 1536
        assert fixed["epochs"] == 20
        assert fixed["prediction_mode"] == spec.prediction_mode
        assert fixed["depth_scale_contract"] == spec.depth_scale_contract


def test_image_scaled_uses_knn_only_for_prompt_and_scale() -> None:
    model = build_model(load_base=False, variant="image_scaled")
    assert not model.uses_depth_tokens
    assert not model.uses_voxel_tokens
    assert model.dense_depth_head.prompt_depth_enabled
    reference = torch.ones(1, 1, 2, 2)
    with pytest.raises(RuntimeError, match="must match"):
        model._knn_bounds(None, reference)
    with pytest.raises(RuntimeError, match="strictly positive"):
        model._knn_bounds(torch.zeros_like(reference), reference)
    with pytest.raises(RuntimeError, match="non-degenerate"):
        model._knn_bounds(torch.ones_like(reference), reference)
    low, high = model._knn_bounds(
        torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]), reference
    )
    assert low.item() == 1.0
    assert high.item() == 4.0


def test_da3_base_checkpoint_metadata_mismatch_is_rejected(tmp_path: Path) -> None:
    class Stub(torch.nn.Module):
        da3_model_name = "da3-base"
        expected_experiment = "expected"
        feature_dim = 1536
        variant = "image"

    checkpoint = tmp_path / "mismatch.pth"
    torch.save(
        {
            "args": {
                "exp": "different",
                "da3_model_name": "da3-small",
                "token_dim": 768,
            },
            "model": {},
        },
        checkpoint,
    )
    with pytest.raises(RuntimeError, match="Checkpoint metadata mismatch"):
        load_trained_checkpoint(Stub(), checkpoint)
