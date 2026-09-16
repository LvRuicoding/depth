import os
from pathlib import Path

import pytest
import torch

from occany_depth_min.model import build_model, load_trained_checkpoint


REFERENCE = Path(os.environ.get("OCCANY_REFERENCE_CHECKPOINT", "__missing__"))


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
