"""Opt-in CPU fidelity audit against an external OccAny checkout.

Run from this repository with --reference-root and --da3-checkpoint. Source
imports occur only in this audit, never in the migrated runtime or default tests.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def reset_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def inputs():
    height, width = 56, 70
    intrinsics = torch.tensor([[40., 0., 35.], [0., 40., 28.], [0., 0., 1.]])
    rows = torch.tensor([23., 26., 28., 30., 32., 34., 24., 29.])
    cols = torch.tensor([10., 20., 30., 40., 50., 59., 34., 45.])
    depth = torch.tensor([8., 10., 12., 14., 16., 18., 20., 24.])
    points = torch.stack(((cols-35.)*depth/40., (rows-28.)*depth/40., depth,
                          torch.linspace(.1, .9, len(depth))), dim=-1)
    sparse = torch.zeros(1, 1, height, width)
    sparse[0, 0, rows.long(), cols.long()] = depth
    from occany_depth_min.knn import knn_complete_sparse_depth
    prompt = torch.from_numpy(knn_complete_sparse_depth(sparse[0, 0].numpy()))[None, None]
    image = torch.linspace(-1., 1., 3*height*width).reshape(1, 3, height, width)
    return {
        "views": [{"img": image, "true_shape": np.array([[height, width]], dtype=np.int32)}],
        "K_per_frame": intrinsics[None, None], "image_hw": torch.tensor([[height, width]]),
        "sparse_depth": sparse, "sparse_depth_mask": sparse > 0, "knn_depth": prompt,
        "points_per_frame": [[points]], "T_cam_from_velo": torch.eye(4)[None],
        "image_valid_mask": torch.ones_like(sparse, dtype=torch.bool),
    }


def compare_outputs(left, right):
    assert left.keys() == right.keys(), (left.keys(), right.keys())
    maximum = 0.
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1e-5, atol=1e-6, msg=key)
        maximum = max(maximum, float((left[key]-right[key]).abs().max()))
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--da3-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--configuration", action="append", help="fusion/encoder/model; default: all 22")
    options = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(options.reference_root.resolve()))
    from ft.kitti_stage1_5f.tools import train_kitti_dc_full as source
    from occany_depth_min.kitti_dc_full import train as migrated
    from occany_depth_min.kitti_dc_full.model import build_model
    from occany_depth_min.depth_head import dense_metric_depth_loss

    configurations = options.configuration or [
        f"{fusion}/{encoder}/{model}"
        for fusion in ("prefusion", "postfusion")
        for encoder in ("patchdepthbin", "vfe")
        for model in (source.VFE_MODELS if encoder == "vfe" else source.MODEL_CLASSES)
    ]
    batch, report = inputs(), {}
    target = torch.linspace(7., 25., 56*70).reshape(1, 1, 56, 70)
    device = torch.device("cpu")
    for configuration in configurations:
        fusion, encoder, variant = configuration.split("/")
        args = source.get_parser().parse_args([
            "--model", variant, "--fusion_mode", fusion, "--voxel_encoder", encoder,
            "--occany_ckpt", str(options.da3_checkpoint), "--amp", "none",
        ])
        reset_seed()
        original = source.build_model(args).eval()
        reference_rng = torch.get_rng_state().clone()
        reset_seed()
        copied = build_model(args).eval()
        assert torch.equal(reference_rng, torch.get_rng_state()), (configuration, "constructor RNG")
        before, after = original.state_dict(), copied.state_dict()
        assert list(before) == list(after), configuration
        for key in before:
            assert torch.equal(before[key], after[key]), (configuration, key, "initialization")
        assert [key for key, _ in original.named_parameters()] == [key for key, _ in copied.named_parameters()], configuration
        del before, after
        model_batch = dict(batch)
        if not variant.endswith("scaled"):
            model_batch.pop("knn_depth")
        result_source = source.forward_batch(original, model_batch, device)
        result_target = migrated.forward_batch(copied, model_batch, device)
        maximum = compare_outputs(result_source, result_target)
        loss_source = source.dense_metric_depth_loss(result_source["dense_depth"], target,
                                                    torch.ones(1, 1, dtype=torch.bool), loss_weight=.1)[0]
        loss_target = dense_metric_depth_loss(result_target["dense_depth"], target,
                                             torch.ones(1, 1, dtype=torch.bool), loss_weight=.1)[0]
        torch.testing.assert_close(loss_source, loss_target, rtol=1e-5, atol=1e-6)
        loss_source.backward()
        loss_target.backward()
        gradients = 0
        for (name, left), (_, right) in zip(original.named_parameters(), copied.named_parameters()):
            assert (left.grad is None) == (right.grad is None), (configuration, name, "gradient presence")
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6, msg=f"{configuration}: {name}")
                gradients += 1
        record = {"initialization": "exact", "output_max_abs_error": maximum,
                  "loss": float(loss_target.detach()), "gradient_tensors": gradients,
                  "strict_checkpoint": None}
        del result_source, result_target, loss_source, loss_target, original
        copied.zero_grad(set_to_none=True)
        if options.checkpoint_root:
            root_name = "kitti_dc_full_postfusion" if fusion == "postfusion" else "kitti_dc_full"
            experiment = variant + ("_vfe" if encoder == "vfe" else "")
            run = options.checkpoint_root / root_name / "single_frame" / f"da3_base_{experiment}" / "left_long1232_10ep_seed0"
            checkpoint = run / "checkpoint-last.pth"
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
                copied.load_state_dict(payload["model"], strict=True)
                record["strict_checkpoint"] = str(checkpoint)
                del payload
        del copied
        gc.collect()
        report[configuration] = record
        options.report_json.write_text(json.dumps(report, indent=2)+"\n")
        print(configuration, json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
