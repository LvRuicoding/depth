"""Small source-derived contracts; no source repository or weights required."""
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def signature(state, *, values=False):
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        if values:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


REFERENCE = json.loads((Path(__file__).parent / "fixtures/kitti_dc_full_reference.json").read_text())


@pytest.mark.parametrize("configuration", sorted(REFERENCE))
def test_source_model_structure_and_initialization(configuration):
    from occany_depth_min.kitti_dc_full.model import build_model, legacy_class_name

    fusion, encoder, variant = configuration.split("/")
    args = SimpleNamespace(
        model=variant, fusion_mode=fusion, voxel_encoder=encoder,
        occany_ckpt=None, da3_checkpoint=None, amp="none",
    )
    torch.manual_seed(0)
    model = build_model(args, load_base=False)
    assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() == REFERENCE[configuration]["cpu_rng_sha256"]
    state = model.state_dict()
    expected = REFERENCE[configuration]
    assert legacy_class_name(args) == expected["model_class"]
    assert len(state) == expected["keys"]
    assert sum(parameter.numel() for parameter in model.parameters()) == expected["parameters"]
    assert signature(state) == expected["structure_sha256"]
    assert signature({key: value for key, value in state.items() if not key.startswith("backbone.")}, values=True) == expected["non_backbone_sha256"]
    del model, state
    gc.collect()
