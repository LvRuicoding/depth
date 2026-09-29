"""Strict four-GPU Unified6 or KITTI evaluation for retained checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset

from .data import (
    DOMAIN_NAMES,
    build_kitti_depth_dataset,
    build_unified_depth_dataset,
    collate_unified_depth_online_knn4,
)
from .metrics import best_checkpoint_score, macro_average_domains
from .model import (
    DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
    build_model,
    get_model_spec,
    load_trained_checkpoint,
)
from .runtime import evaluate_loader


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_results(payload: Mapping[str, Any], output_json: str | Path) -> None:
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    output.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for domain, result in safe["datasets"].items():
        path = output.with_name(f"{output.stem}_{domain}{output.suffix or '.json'}")
        domain_payload = {
            "checkpoint": safe["checkpoint"],
            "checkpoint_epoch": safe["checkpoint_epoch"],
            "weights": safe["weights"],
            "zero_shot": False,
            "experiment": safe["experiment"],
            "prediction_mode": safe["prediction_mode"],
            "depth_scale_contract": safe["depth_scale_contract"],
            "split": "val",
            "dataset": result,
        }
        path.write_text(
            json.dumps(domain_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _distributed_device() -> tuple[torch.device, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("Evaluation must be launched with torchrun on exactly four GPUs.")
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"Exact reproduction requires world_size=4, got {world}.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), rank, world


def _root(args: argparse.Namespace, domain: str) -> str:
    return str(getattr(args, domain.replace("7scenes", "seven_scenes") + "_root"))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--dataset", choices=("unified6", "kitti"), default="unified6")
    value.add_argument("--manifest-dir")
    value.add_argument("--kitti-root")
    value.add_argument("--ddad-root")
    value.add_argument("--seven-scenes-root")
    value.add_argument("--nyuv2-root")
    value.add_argument("--sunrgbd-root")
    value.add_argument("--void-root")
    value.add_argument("--output-json", required=True)
    value.add_argument(
        "--model",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help="Checkpoint architecture to evaluate.",
    )
    value.add_argument(
        "--domains",
        default="all",
        help="Evaluation stage control: all or one comma-separated subset.",
    )
    value.add_argument("--max-batches", type=int, default=0, help=argparse.SUPPRESS)
    return value


def _validate_paths(args: argparse.Namespace) -> None:
    get_model_spec(args.model, args.dataset)
    required = ["kitti_root"]
    if args.dataset == "unified6":
        required.extend(
            (
                "manifest_dir",
                "ddad_root",
                "seven_scenes_root",
                "nyuv2_root",
                "sunrgbd_root",
                "void_root",
            )
        )
    missing = [f"--{name.replace('_', '-')}" for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"{args.dataset} evaluation requires {', '.join(missing)}.")


def main() -> None:
    args = parser().parse_args()
    _validate_paths(args)
    device, rank, world = _distributed_device()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if checkpoint.is_dir():
        checkpoint = checkpoint / (
            "checkpoint-last.pth" if args.dataset == "kitti" else "checkpoint-best.pth"
        )
    spec = get_model_spec(args.model, args.dataset)
    model = build_model(load_base=False, variant=args.model)
    payload = load_trained_checkpoint(model, checkpoint)
    model.to(device)
    model.eval()

    if args.domains == "all":
        domains = ("kitti",) if args.dataset == "kitti" else DOMAIN_NAMES
    else:
        domains = tuple(
            item.strip() for item in args.domains.split(",") if item.strip()
        )
    invalid = [name for name in domains if name not in DOMAIN_NAMES]
    if invalid:
        raise ValueError(f"Unknown evaluation domains: {invalid}")
    if args.dataset == "kitti" and domains != ("kitti",):
        raise ValueError("KITTI-only evaluation accepts only --domains kitti.")

    results: Dict[str, Dict[str, Any]] = {}
    for domain in domains:
        if args.dataset == "kitti":
            dataset = build_kitti_depth_dataset(
                str(args.kitti_root), "val", strict_count=True
            )
        else:
            dataset = build_unified_depth_dataset(
                domain,
                _root(args, domain),
                "val",
                manifest_dir=str(args.manifest_dir),
                input_long_side=518,
                patch_size=14,
                synthetic_sparse_points=500,
                sampling_seed=0,
                strict_count=True,
            )
        shard = Subset(dataset, list(range(rank, len(dataset), world)))
        loader = DataLoader(
            shard,
            sampler=SequentialSampler(shard),
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_unified_depth_online_knn4,
        )
        result = evaluate_loader(
            model,
            loader,
            domain,
            device,
            max_batches=args.max_batches,
            print_freq=20 if rank == 0 else 0,
        )
        result["samples"] = len(dataset) if args.max_batches == 0 else result["num_batches"]
        result["split"] = "val"
        results[domain] = result

    if rank == 0:
        macro = macro_average_domains(results)
        output: Dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "weights": str(checkpoint),
            "zero_shot": False,
            "experiment": spec.experiment,
            "model_variant": spec.variant,
            "model_class": spec.model_class,
            "prediction_mode": spec.prediction_mode,
            "depth_scale_contract": spec.depth_scale_contract,
            "online_knn_contract": spec.online_knn_contract,
            "dpt_prompt_contract": spec.dpt_prompt_contract,
            "initialization_contract": spec.initialization_contract,
            "requested_split": "val",
            "dataset_splits": {name: "val" for name in domains},
            "datasets": results,
            "cross_domain_macro": macro,
        }
        if args.dataset == "unified6" and tuple(domains) == DOMAIN_NAMES and args.max_batches == 0:
            output["best_checkpoint_score_abs_rel"] = best_checkpoint_score(results)
        write_results(output, args.output_json)
        print(json.dumps(_json_safe(macro), indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
