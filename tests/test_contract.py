import hashlib
import os
from pathlib import Path

import pytest
import torch

from occany_depth_min.model import build_model, load_trained_checkpoint


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
