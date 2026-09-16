"""Exact four-GPU trainer for the one retained Unified6 depth model."""
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
from torch.utils.data import DataLoader, SequentialSampler, Subset

from .data import (
    DOMAIN_NAMES,
    DomainBalancedDistributedSampler,
    build_unified_depth_dataset,
    build_unified_six_dataset,
    collate_unified_depth_online_knn4,
)
from .depth_head import dense_metric_depth_loss
from .eval import EXPERIMENT, write_results
from .metrics import best_checkpoint_score, macro_average_domains
from .model import (
    DPT_PROMPT_CONTRACT,
    FUSION_CONTRACT,
    INITIALIZATION_CONTRACT,
    MODEL_CLASS,
    ONLINE_KNN_CONTRACT,
    SCALE_CONTRACT,
    build_model,
)
from .runtime import evaluate_loader, forward_batch, move_to_device


EPOCHS = 10
LEARNING_RATE = 1.0e-4
MIN_LEARNING_RATE = 1.0e-6
WEIGHT_DECAY = 1.0e-4
WARMUP_EPOCHS = 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--da3-checkpoint", required=True)
    value.add_argument("--manifest-dir", required=True)
    value.add_argument("--kitti-root", required=True)
    value.add_argument("--ddad-root", required=True)
    value.add_argument("--seven-scenes-root", required=True)
    value.add_argument("--nyuv2-root", required=True)
    value.add_argument("--sunrgbd-root", required=True)
    value.add_argument("--void-root", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--resume", default=None)
    value.add_argument("--smoke-steps", type=int, default=0, help=argparse.SUPPRESS)
    return value


def _roots(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "kitti": args.kitti_root,
        "ddad": args.ddad_root,
        "7scenes": args.seven_scenes_root,
        "nyuv2": args.nyuv2_root,
        "sunrgbd": args.sunrgbd_root,
        "void": args.void_root,
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
    return {
        **vars(args),
        "exp": EXPERIMENT,
        "dataset": "unified6",
        "backbone": "da3",
        "da3_model_name": "da3-small",
        "num_frames": 1,
        "num_views": 1,
        "width": 518,
        "height": 168,
        "patch_size": 14,
        "token_dim": 768,
        "batch_size": 1,
        "num_workers": 4,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "min_lr": MIN_LEARNING_RATE,
        "amp": "bf16",
        "dense_depth_features": 128,
        "dense_depth_loss_weight": 0.1,
        "dynamic_image_size": True,
        "unified_input_long_side": 518,
        "unified_sparse_points": 500,
        "unified_samples_per_epoch": 0,
        "fusion_vox_origin": [-60.0, -5.0, 0.0],
        "fusion_vox_size": [0.4, 0.4, 0.4],
        "fusion_vox_grid": [300, 25, 300],
        "fusion_depth_bin_size": 4.0,
        "seed": 0,
        "world_size": world,
        "model_class": MODEL_CLASS,
        "lingbot_da3_voxel_prefusion_contract": FUSION_CONTRACT,
        "depth_scale_contract": SCALE_CONTRACT,
        "online_knn_contract": ONLINE_KNN_CONTRACT,
        "dpt_prompt_contract": DPT_PROMPT_CONTRACT,
        "initialization_contract": INITIALIZATION_CONTRACT,
    }


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch_fraction: float
) -> float:
    if epoch_fraction < WARMUP_EPOCHS:
        learning_rate = LEARNING_RATE * epoch_fraction / WARMUP_EPOCHS
    else:
        progress = (epoch_fraction - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        learning_rate = MIN_LEARNING_RATE + 0.5 * (
            LEARNING_RATE - MIN_LEARNING_RATE
        ) * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _validation_loaders(
    args: argparse.Namespace, rank: int, world: int
) -> Dict[str, DataLoader]:
    loaders = {}
    for domain, root in _roots(args).items():
        dataset = build_unified_depth_dataset(
            domain,
            root,
            "val",
            manifest_dir=args.manifest_dir,
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
        payload["unified6_val_macro"] = macro_average_domains(validation)
    return payload


def main() -> None:
    args = parser().parse_args()
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
    model = build_model(args.da3_checkpoint, load_base=True).to(device)
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
        ddp.module.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume.get("best_macro_abs_rel", float("inf")))

    train_dataset = build_unified_six_dataset(
        _roots(args),
        "train",
        manifest_dir=args.manifest_dir,
        input_long_side=518,
        patch_size=14,
        synthetic_sparse_points=500,
        sampling_seed=0,
        strict_count=True,
    )
    sampler = DomainBalancedDistributedSampler(
        train_dataset,
        samples_per_epoch=0,
        coverage_epochs=EPOCHS,
        seed=0,
        num_replicas=world,
        rank=rank,
        shuffle=True,
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
        config = {
            "args": fixed_args,
            "model": {
                "class": MODEL_CLASS,
                "total_params": sum(parameter.numel() for parameter in ddp.module.parameters()),
                "trainable_params": sum(
                    parameter.numel()
                    for parameter in ddp.module.parameters()
                    if parameter.requires_grad
                ),
            },
            "data": {
                "train_samples": len(train_dataset),
                "logical_samples_per_epoch": sampler.logical_samples_per_epoch,
                "samples_per_rank": len(sampler),
            },
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    total_steps = 0
    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        ddp.train()
        loss_sum = torch.zeros((), device=device)
        step_count = torch.zeros((), device=device)
        started = time.time()
        for step, batch in enumerate(train_loader):
            learning_rate = _adjust_learning_rate(
                optimizer, epoch + step / max(len(train_loader), 1)
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
        if (epoch + 1) % 5 == 0:
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
            current = best_checkpoint_score(validation)
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
                        "experiment": EXPERIMENT,
                        "prediction_mode": "relative_online_knn_minmax",
                        "depth_scale_contract": SCALE_CONTRACT,
                        "requested_split": "val",
                        "dataset_splits": {name: "val" for name in DOMAIN_NAMES},
                        "datasets": validation,
                        "cross_domain_macro": macro,
                        "best_checkpoint_score_abs_rel": current,
                    },
                    output_dir / f"eval_unified_epoch{epoch}.json",
                )

        if rank == 0:
            payload = _checkpoint_payload(
                ddp.module, optimizer, epoch, fixed_args, best_score, validation
            )
            if (epoch + 1) % 2 == 0 or validation is not None or epoch + 1 == EPOCHS:
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
