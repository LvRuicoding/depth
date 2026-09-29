"""Exact four-GPU trainer for the retained Unified6 and KITTI depth runs."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler, Subset

from .data import (
    DOMAIN_NAMES,
    EXPECTED_KITTI_COUNTS,
    DomainBalancedDistributedSampler,
    build_kitti_depth_dataset,
    build_unified_depth_dataset,
    build_unified_six_dataset,
    collate_unified_depth_online_knn4,
)
from .depth_head import dense_metric_depth_loss
from .eval import write_results
from .metrics import best_checkpoint_score, macro_average_domains
from .model import (
    DEFAULT_MODEL_VARIANT,
    DA3_BASE_MODEL_VARIANTS,
    MODEL_VARIANTS,
    POSTFUSION_MODEL_VARIANT,
    build_model,
    get_model_spec,
)
from .runtime import evaluate_loader, forward_batch, move_to_device


EPOCHS = 10
KITTI_EPOCHS = 20
LEARNING_RATE = 1.0e-4
MIN_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 1.0e-4
WARMUP_EPOCHS = 1
BALANCED_SAMPLING_CONTRACT = (
    "equal_domain_persistent_no_replacement_full_coverage_over_20_epochs_v1"
)
NATURAL_SAMPLING_CONTRACT = "natural_concat_distributed_shuffle_full_pass_per_epoch_v1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--da3-checkpoint", required=True)
    value.add_argument("--dataset", choices=("unified6", "kitti"), default="unified6")
    value.add_argument("--manifest-dir")
    value.add_argument("--kitti-root")
    value.add_argument("--ddad-root")
    value.add_argument("--seven-scenes-root")
    value.add_argument("--nyuv2-root")
    value.add_argument("--sunrgbd-root")
    value.add_argument("--void-root")
    value.add_argument("--output-dir", required=True)
    value.add_argument("--resume", default=None)
    value.add_argument(
        "--model",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help="Architecture to train.",
    )
    value.add_argument(
        "--sampling",
        choices=("balanced", "natural"),
        default="balanced",
        help="balanced reproduces the original run; natural keeps raw domain proportions.",
    )
    value.add_argument("--smoke-steps", type=int, default=0, help=argparse.SUPPRESS)
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
        raise ValueError(f"{args.dataset} training requires {', '.join(missing)}.")
    if args.dataset == "kitti" and args.sampling != "balanced":
        raise ValueError("KITTI-only training has one dataset and does not accept --sampling natural.")


def _roots(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "kitti": str(args.kitti_root),
        "ddad": str(args.ddad_root),
        "7scenes": str(args.seven_scenes_root),
        "nyuv2": str(args.nyuv2_root),
        "sunrgbd": str(args.sunrgbd_root),
        "void": str(args.void_root),
    }


def _init_distributed() -> tuple[torch.device, int, int, int]:
    if "RANK" not in os.environ:
        raise RuntimeError("Training must be launched with torchrun on exactly four GPUs.")
    dist.init_process_group("nccl", init_method="env://")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"Exact reproduction requires world_size=4, got {world}.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), rank, world, local_rank


def _fixed_args(args: argparse.Namespace, world: int) -> Dict[str, Any]:
    spec = get_model_spec(args.model, args.dataset)
    is_kitti = args.dataset == "kitti"
    epochs = KITTI_EPOCHS if is_kitti else EPOCHS
    sampling_contract = (
        BALANCED_SAMPLING_CONTRACT
        if args.sampling == "balanced"
        else NATURAL_SAMPLING_CONTRACT
    )
    fixed = {
        **vars(args),
        "exp": (
            spec.experiment
            if is_kitti or args.sampling == "balanced"
            else f"{spec.experiment}_natural"
        ),
        "model_variant": spec.variant,
        "backbone": "da3",
        "da3_model_name": spec.da3_model_name,
        "num_frames": 1,
        "num_views": 1,
        "width": 518,
        "height": 168,
        "patch_size": 14,
        "token_dim": spec.token_dim,
        "batch_size": 1,
        "num_workers": 4,
        "epochs": epochs,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "min_lr": MIN_LEARNING_RATE,
        "amp": "bf16",
        "dense_depth_features": 128,
        "dense_depth_loss_weight": 0.1,
        "dynamic_image_size": None if is_kitti else True,
        "fusion_vox_origin": None if is_kitti else [-60.0, -5.0, 0.0],
        "fusion_vox_size": None if is_kitti else [0.4, 0.4, 0.4],
        "fusion_vox_grid": None if is_kitti else [300, 25, 300],
        "fusion_depth_bin_size": 4.0,
        "seed": 0,
        "world_size": world,
        "model_class": spec.model_class,
        "fusion_contract": spec.fusion_contract,
        "prediction_mode": spec.prediction_mode,
        "depth_scale_contract": spec.depth_scale_contract,
        "online_knn_contract": spec.online_knn_contract,
        "dpt_prompt_contract": spec.dpt_prompt_contract,
        "initialization_contract": spec.initialization_contract,
    }
    if is_kitti:
        fixed.update(
            {
                "sampling_mode": "single_dataset_distributed_shuffle",
                "sampling_contract": "kitti_full_pass_distributed_shuffle_v1",
                "depth_cache_policy": "forbidden_no_read_no_write_v1",
                "eval_freq": 1,
                "save_freq": 2,
            }
        )
        for name in (
            "unified_input_long_side",
            "unified_sparse_points",
            "unified_samples_per_epoch",
            "unified_sampling_mode",
            "unified_sampling_contract",
        ):
            fixed.pop(name, None)
    else:
        fixed.update(
            {
                "unified_input_long_side": 518,
                "unified_sparse_points": 500,
                "unified_samples_per_epoch": 0,
                "unified_sampling_mode": args.sampling,
                "unified_sampling_contract": sampling_contract,
                "eval_freq": 5,
                "save_freq": 2,
            }
        )
    if args.model == DEFAULT_MODEL_VARIANT:
        fixed["lingbot_da3_voxel_prefusion_contract"] = spec.fusion_contract
    elif args.model == POSTFUSION_MODEL_VARIANT:
        fixed["patch_depth4m_voxeldepth_dualwindow_contract"] = spec.fusion_contract
        fixed["depth_token_source"] = "sparse_log_patch"
        fixed["voxel_token_source"] = "patch_depth4m"
    elif args.model in DA3_BASE_MODEL_VARIANTS:
        fixed["da3_base_token_fusion_contract"] = spec.fusion_contract
        fixed["depth_token_source"] = (
            "sparse_log_patch" if spec.uses_depth_tokens else None
        )
        fixed["voxel_token_source"] = (
            "patch_depth4m" if spec.uses_voxel_tokens else None
        )
    return fixed


def build_training_sampler(
    dataset: torch.utils.data.Dataset,
    *,
    sampling: str,
    rank: int,
    world: int,
) -> torch.utils.data.Sampler[int]:
    """Build either the released equal-domain sampler or a natural full pass."""
    if sampling == "balanced":
        return DomainBalancedDistributedSampler(
            dataset,  # type: ignore[arg-type]
            samples_per_epoch=0,
            coverage_epochs=EPOCHS,
            seed=0,
            num_replicas=world,
            rank=rank,
            shuffle=True,
        )
    if sampling == "natural":
        # UnifiedSixDataset is a natural concatenation. DistributedSampler
        # shuffles that single global index space, so domain probability is
        # exactly proportional to each manifest's size. The current 96,648
        # samples divide evenly over four ranks and require no padding.
        return DistributedSampler(
            dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False,
        )
    raise ValueError(f"Unknown sampling mode: {sampling!r}")


def _checkpoint_args(payload: Dict[str, Any]) -> Dict[str, Any]:
    saved_args = payload.get("args", {})
    if isinstance(saved_args, dict):
        return saved_args
    if hasattr(saved_args, "__dict__"):
        return vars(saved_args)
    return {}


def _checkpoint_sampling_contract(payload: Dict[str, Any]) -> str | None:
    value = _checkpoint_args(payload).get("unified_sampling_contract")
    return None if value is None else str(value)


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch_fraction: float, total_epochs: int = EPOCHS
) -> float:
    if epoch_fraction < WARMUP_EPOCHS:
        learning_rate = LEARNING_RATE * epoch_fraction / WARMUP_EPOCHS
    else:
        progress = (epoch_fraction - WARMUP_EPOCHS) / (total_epochs - WARMUP_EPOCHS)
        learning_rate = MIN_LEARNING_RATE + 0.5 * (
            LEARNING_RATE - MIN_LEARNING_RATE
        ) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _validation_loaders(
    args: argparse.Namespace, rank: int, world: int
) -> Dict[str, DataLoader]:
    if args.dataset == "kitti":
        dataset = build_kitti_depth_dataset(str(args.kitti_root), "val", strict_count=True)
        shard = Subset(dataset, list(range(rank, len(dataset), world)))
        return {
            "kitti": DataLoader(
                shard,
                sampler=SequentialSampler(shard),
                batch_size=1,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_unified_depth_online_knn4,
            )
        }
    loaders = {}
    for domain, root in _roots(args).items():
        dataset = build_unified_depth_dataset(
            domain,
            root,
            "val",
            manifest_dir=str(args.manifest_dir),
            input_long_side=518,
            patch_size=14,
            synthetic_sparse_points=500,
            sampling_seed=0,
            strict_count=True,
        )
        shard = Subset(dataset, list(range(rank, len(dataset), world)))
        loaders[domain] = DataLoader(
            shard,
            sampler=SequentialSampler(shard),
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_unified_depth_online_knn4,
        )
    return loaders


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    fixed_args: Dict[str, Any],
    best_score: float,
    validation: Dict[str, Any] | None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": {},
        "epoch": epoch,
        "args": fixed_args,
        "backbone_hash": None,
        "best_macro_abs_rel": best_score,
    }
    if validation is not None:
        if fixed_args["dataset"] == "unified6":
            payload["unified6_val_macro"] = macro_average_domains(validation)
        else:
            payload["kitti_val"] = validation["kitti"]
    return payload


def main() -> None:
    args = parser().parse_args()
    _validate_paths(args)
    device, rank, world, local_rank = _init_distributed()
    seed = rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    output_dir = Path(args.output_dir).expanduser().resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(device_ids=[local_rank])

    fixed_args = _fixed_args(args, world)
    model = build_model(
        args.da3_checkpoint, load_base=True, variant=args.model
    ).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
        static_graph=False,
    )
    optimizer = torch.optim.AdamW(
        ddp.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    start_epoch, best_score = 0, float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_args = _checkpoint_args(resume)
        mismatches = []
        for name in (
            "exp",
            "model_variant",
            "da3_model_name",
            "token_dim",
            "initialization_contract",
        ):
            saved = saved_args.get(name)
            expected = fixed_args[name]
            if saved is not None and saved != expected:
                mismatches.append(f"{name}={saved!r}, expected {expected!r}")
        if mismatches:
            raise RuntimeError(
                "Cannot resume with a different model architecture: "
                + "; ".join(mismatches)
            )
        if args.dataset == "unified6":
            saved_contract = _checkpoint_sampling_contract(resume)
            expected_contract = str(fixed_args["unified_sampling_contract"])
            if saved_contract is not None and saved_contract != expected_contract:
                raise RuntimeError(
                    "Cannot resume with a different sampling policy: "
                    f"checkpoint={saved_contract!r}, requested={expected_contract!r}."
                )
            if args.sampling == "natural" and saved_contract is None:
                raise RuntimeError(
                    "A natural-sampling run cannot resume a checkpoint that does not "
                    "record its sampling contract."
                )
        ddp.module.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume.get("best_macro_abs_rel", float("inf")))

    if args.dataset == "kitti":
        train_dataset = build_kitti_depth_dataset(
            str(args.kitti_root), "train", strict_count=True
        )
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=0,
            drop_last=False,
        )
    else:
        train_dataset = build_unified_six_dataset(
            _roots(args),
            "train",
            manifest_dir=str(args.manifest_dir),
            input_long_side=518,
            patch_size=14,
            synthetic_sparse_points=500,
            sampling_seed=0,
            strict_count=True,
        )
        sampler = build_training_sampler(
            train_dataset,
            sampling=args.sampling,
            rank=rank,
            world=world,
        )
    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_unified_depth_online_knn4,
    )
    validation_loaders = None if args.smoke_steps else _validation_loaders(args, rank, world)

    if rank == 0:
        if args.dataset == "kitti":
            data_config = {
                "train_samples": len(train_dataset),
                "validation_samples": EXPECTED_KITTI_COUNTS["val"],
                "sampling_mode": "single_dataset_distributed_shuffle",
                "sampling_contract": fixed_args["sampling_contract"],
                "logical_samples_per_epoch": len(train_dataset),
                "samples_per_rank": len(sampler),
            }
        else:
            data_config = {
                "train_samples": len(train_dataset),
                "sampling_mode": args.sampling,
                "sampling_contract": fixed_args["unified_sampling_contract"],
                "domain_samples_per_epoch": (
                    {
                        name: sampler.samples_per_domain
                        for name in train_dataset.domain_names  # type: ignore[attr-defined]
                    }
                    if isinstance(sampler, DomainBalancedDistributedSampler)
                    else {
                        name: len(domain_dataset)
                        for name, domain_dataset in zip(
                            train_dataset.domain_names,  # type: ignore[attr-defined]
                            train_dataset.datasets,  # type: ignore[attr-defined]
                        )
                    }
                ),
                "logical_samples_per_epoch": (
                    sampler.logical_samples_per_epoch
                    if isinstance(sampler, DomainBalancedDistributedSampler)
                    else len(train_dataset)
                ),
                "samples_per_rank": len(sampler),
            }
        config = {
            "args": fixed_args,
            "model": {
                "variant": args.model,
                "class": fixed_args["model_class"],
                "total_params": sum(parameter.numel() for parameter in ddp.module.parameters()),
                "trainable_params": sum(
                    parameter.numel()
                    for parameter in ddp.module.parameters()
                    if parameter.requires_grad
                ),
            },
            "data": data_config,
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    total_steps = 0
    total_epochs = int(fixed_args["epochs"])
    eval_frequency = int(fixed_args["eval_freq"])
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch)
        ddp.train()
        loss_sum = torch.zeros((), device=device)
        step_count = torch.zeros((), device=device)
        started = time.time()
        for step, batch in enumerate(train_loader):
            learning_rate = _adjust_learning_rate(
                optimizer,
                epoch + step / max(len(train_loader), 1),
                total_epochs,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = forward_batch(ddp, batch, device)
                loss, raw_loss, valid_pixels, valid_frames = dense_metric_depth_loss(
                    outputs["dense_depth"],
                    move_to_device(batch["dense_depth"], device),
                    frame_mask=move_to_device(batch["dense_depth_frame_mask"], device),
                    loss_weight=0.1,
                )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite depth loss at epoch={epoch}, step={step}.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            step_count += 1
            total_steps += 1
            if rank == 0 and (step + 1) % 20 == 0:
                print(
                    f"epoch={epoch} step={step + 1}/{len(train_loader)} "
                    f"loss={float(loss):.6f} raw={float(raw_loss):.6f} "
                    f"valid={int(valid_pixels)} frames={int(valid_frames)} "
                    f"lr={learning_rate:.8g}",
                    flush=True,
                )
            if args.smoke_steps and total_steps >= args.smoke_steps:
                break

        dist.all_reduce(loss_sum)
        dist.all_reduce(step_count)
        if rank == 0:
            print(
                f"epoch={epoch} mean_loss={float(loss_sum / step_count):.6f} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if args.smoke_steps:
            if rank == 0:
                smoke = _checkpoint_payload(
                    ddp.module, optimizer, epoch, fixed_args, best_score, None
                )
                torch.save(smoke, output_dir / "checkpoint-smoke.pth")
            break

        validation = None
        is_best = False
        if (epoch + 1) % eval_frequency == 0:
            assert validation_loaders is not None
            validation = {
                domain: evaluate_loader(
                    ddp.module,
                    loader,
                    domain,
                    device,
                    print_freq=20 if rank == 0 else 0,
                )
                for domain, loader in validation_loaders.items()
            }
            current = (
                best_checkpoint_score(validation)
                if args.dataset == "unified6"
                else float(validation["kitti"]["pixel_micro"]["all_valid"]["abs_rel"])
            )
            is_best = current < best_score
            best_score = min(best_score, current)
            if rank == 0:
                macro = macro_average_domains(validation)
                write_results(
                    {
                        "checkpoint": str(output_dir / "checkpoint-best.pth"),
                        "checkpoint_epoch": epoch,
                        "weights": str(output_dir / "checkpoint-best.pth"),
                        "zero_shot": False,
                        "experiment": fixed_args["exp"],
                        "prediction_mode": fixed_args["prediction_mode"],
                        "depth_scale_contract": fixed_args["depth_scale_contract"],
                        "requested_split": "val",
                        "dataset_splits": {
                            name: "val" for name in validation
                        },
                        "datasets": validation,
                        "cross_domain_macro": macro,
                        "best_checkpoint_score_abs_rel": current,
                    },
                    output_dir
                    / (
                        f"eval_unified_epoch{epoch}.json"
                        if args.dataset == "unified6"
                        else f"eval_depth_epoch{epoch}.json"
                    ),
                )

        if rank == 0:
            payload = _checkpoint_payload(
                ddp.module, optimizer, epoch, fixed_args, best_score, validation
            )
            if (epoch + 1) % 2 == 0 or validation is not None or epoch + 1 == total_epochs:
                torch.save(payload, output_dir / "checkpoint-last.pth")
            if (epoch + 1) % 5 == 0:
                torch.save(
                    {"model": ddp.module.state_dict(), "epoch": epoch, "args": fixed_args},
                    output_dir / f"checkpoint-{epoch}.pth",
                )
            if is_best:
                torch.save(payload, output_dir / "checkpoint-best.pth")
        dist.barrier(device_ids=[local_rank])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
