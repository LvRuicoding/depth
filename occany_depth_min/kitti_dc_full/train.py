"""Train/evaluate DA3-Base ablations, including RAW VFE, on full KITTI DC.

This entrypoint keeps the historical processed-KITTI contracts independent.
Use the smoke subcommand for bounded validation; it never saves training checkpoints.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from .data import (
    PROTOCOL, KITTIDepthCompletionDataset, FullCoverageDistributedSampler,
    collate_kitti_dc_full,
)
from .model import MODEL_CLASSES, VFE_MODELS, build_model, model_class, legacy_class_name
from ..depth_head import dense_metric_depth_loss
from ..metrics import UnifiedDepthMetricAccumulator, restore_to_native
from ..runtime import update_metrics as update_accumulator_from_batch, _merge as _merge_distributed_accumulator


REPO_ROOT = Path(__file__).resolve().parents[2]


def experiment_name(args):
    return args.model + ("_vfe" if args.voxel_encoder == "vfe" else "")


def output_root(args):
    name = "kitti_dc_full_postfusion" if args.fusion_mode == "postfusion" else "kitti_dc_full"
    return REPO_ROOT / "output/depth" / name


def validate_output_path(path, args):
    path = Path(path).resolve()
    if path.is_relative_to((REPO_ROOT / "output/depth/kitti").resolve()):
        raise ValueError("Full KITTI DC requires its own output directory, outside output/depth/kitti")
    if args.fusion_mode == "postfusion" and path.is_relative_to((REPO_ROOT / "output/depth/kitti_dc_full").resolve()):
        raise ValueError("Post-fusion outputs must be outside the existing prefusion kitti_dc_full directory")


def get_parser(*, mode="train"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_CLASSES), required=True)
    parser.add_argument("--fusion-mode", choices=("prefusion", "postfusion"), default="prefusion")
    parser.add_argument("--voxel-encoder", choices=("patchdepthbin", "vfe"), default="patchdepthbin")
    parser.add_argument("--kitti-dc-root", default=os.environ.get("KITTI_DC_ROOT", str(REPO_ROOT / "raw_data/kitti_full")))
    parser.add_argument("--da3-checkpoint", dest="occany_ckpt", default=os.environ.get("DA3_CHECKPOINT", str(REPO_ROOT / "checkpoints/DA3-BASE")))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Training checkpoint for eval-val")
    parser.add_argument("--prediction-dir", type=Path, help="Optional native-grid float32 .npy predictions for eval-val")
    parser.add_argument("--smoke-steps", type=int, default=2 if mode == "smoke" else 0)
    # The released full-KITTI protocol is fixed; these are not tuning flags.
    parser.set_defaults(dataset="kitti_dc_full", eval_split="val", input_long_side=1232,
                        epochs=10, batch_size=1, amp="bf16", seed=0, lr=1e-4,
                        weight_decay=1e-4, warmup_epochs=1, min_lr=1e-6,
                        dense_depth_loss_weight=0.1, eval_only=mode == "eval-val", mode=mode)
    return parser


def initialize_runtime(args):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in ((1, 4) if args.smoke_steps else (4,)):
        raise ValueError("Formal full-KITTI train/eval requires four GPUs; smoke supports one or four GPUs")
    if not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid distributed rank")
    if not torch.cuda.is_available():
        raise RuntimeError("Full-KITTI train/eval/smoke requires CUDA; no CUDA device is available")
    if local_rank >= torch.cuda.device_count():
        raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, world_size, local_rank, device


def forward_batch(model, batch, device):
    unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    views = [{key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
              for key, value in view.items()} for view in batch["views"]]
    kwargs = dict(K_per_frame=batch["K_per_frame"].to(device, non_blocking=True),
                  image_hw=batch["image_hw"].to(device, non_blocking=True))
    if getattr(unwrapped, "depth_token_branch_enabled", False) or getattr(unwrapped, "voxel_token_branch_enabled", False):
        kwargs.update(sparse_depth=batch["sparse_depth"].to(device, non_blocking=True),
                      sparse_depth_mask=batch["sparse_depth_mask"].to(device, non_blocking=True))
    if "knn_depth" in batch:
        kwargs["knn_depth"] = batch["knn_depth"].to(device, non_blocking=True)
    if getattr(unwrapped, "voxel_encoder_kind", "patchdepthbin") == "vfe":
        kwargs.update(points_per_frame=batch["points_per_frame"],
                      T_cam_from_velo=batch["T_cam_from_velo"].to(device, non_blocking=True),
                      image_valid_mask=batch["image_valid_mask"].to(device, non_blocking=True))
    return model(views, **kwargs)


def amp_context(args):
    return torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" else contextlib.nullcontext()


def loader(dataset, args, *, training, rank, world_size):
    if training:
        # A bounded smoke run reaches a real partial final DDP step, exercising
        # zero-weight padding rather than just checking the first full batches.
        smoke_samples = min(len(dataset), (args.smoke_steps - 1) * world_size + 1) if args.smoke_steps else None
        sampler = FullCoverageDistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                                 seed=args.seed, max_samples=smoke_samples)
    else:
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
        sampler = torch.utils.data.SequentialSampler(dataset)
    return DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=args.num_workers,
                      pin_memory=True, drop_last=False, collate_fn=collate_kitti_dc_full,
                      persistent_workers=args.num_workers > 0)


def lr_at(args, progress):
    if progress < args.warmup_epochs:
        return args.lr * progress / max(args.warmup_epochs, 1e-12)
    fraction = (progress - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-12)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * fraction))


def protocol_config(args, train_dataset, val_dataset, world_size, saved_protocol=None):
    saved_protocol = saved_protocol or {}
    weights = Path(args.occany_ckpt) / "model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    protocol = dict(protocol=PROTOCOL, dataset=args.dataset, camera="image_02",
                lidar_source="synchronized_RAW_bin", projection="shared_fp32_round_zbuffer_image_mask",
                rgb_resize="PIL_LANCZOS", gt_resize="INTER_NEAREST_EXACT", intrinsic_resize="half_pixel",
                input_long_side=args.input_long_side, no_upscale=True, patch_size=14,
                model=args.model, model_class=legacy_class_name(args),
                da3_model_name="da3-base", pretrained_sha256=digest.hexdigest(), token_dim=1536,
                num_frames=1, freeze_backbone=False, epochs=args.epochs, world_size=world_size,
                batch_size=1, seed=args.seed, amp=args.amp, lr=args.lr, weight_decay=args.weight_decay,
                betas=(0.9, 0.95), warmup_epochs=args.warmup_epochs, min_lr=args.min_lr,
                loss="log_l1_plus_relative_l1", loss_weight=args.dense_depth_loss_weight,
                train_samples=len(train_dataset) if train_dataset is not None else saved_protocol.get("train_samples"),
                val_samples=len(val_dataset),
                train_manifest_sha256=train_dataset.manifest_sha256 if train_dataset is not None else saved_protocol.get("train_manifest_sha256"),
                val_manifest_sha256=val_dataset.manifest_sha256,
                evaluation="original_grid_pixel_micro_and_per_image_macro_1e-3_to_80m",
                scaled_prompt="valid_image_online_knn4_edge_pad" if args.model.endswith("scaled") else None,
                checkpoint_interval_epochs=5, evaluation_interval_epochs=5,
                eval_split="val", primary_metric="pixel_micro.all_valid.rmse", smoke=bool(args.smoke_steps))
    # Leave historical seven-model protocols byte-for-byte equivalent. VFE
    # records its different input/representation explicitly for checkpoint IO.
    if args.voxel_encoder == "vfe":
        protocol.update(voxel_encoder="vfe", voxel_encoder_class="VoxelFeatureEncoder",
                        experiment=experiment_name(args), voxel_input="synchronized_RAW_xyz_intensity",
                        voxel_intensity=True, voxel_geometry="fp32_transform_grid_center_projection",
                        voxel_visibility="center_in_unpadded_image_without_zbuffer",
                        voxel_origin=(-25.6, -2.0, 0.0), voxel_size=(0.4, 0.4, 0.4),
                        voxel_grid=(128, 16, 128), voxel_feature_dim=128, voxel_pe_num_freqs=8)
    if args.fusion_mode == "postfusion":
        has_depth = args.model.startswith(("depth", "voxel_depth"))
        has_voxel = args.model.startswith("voxel")
        protocol.update(fusion_mode="postfusion", fusion_stage="none" if args.model == "image" else "after_da3",
                        depth_token_source="sparse_log_patch" if has_depth else None,
                        voxel_token_source=("cartesian_vfe" if args.voxel_encoder == "vfe" else "patch_depth4m") if has_voxel else None,
                        fusion_order=[name for name, enabled in (("voxel", has_voxel), ("depth", has_depth)) if enabled],
                        fusion_window=4 if args.model != "image" else None,
                        fusion_shifts=(0, 2) if args.model != "image" else None,
                        depth_embed_seed=0 if has_depth else None,
                        initialization="da3_base_only_random_encoders_fusion_dpt")
    return protocol


def validate_checkpoint(payload, protocol, *, resume):
    saved = payload.get("protocol", {})
    if saved.get("fusion_mode", "prefusion") != protocol.get("fusion_mode", "prefusion"):
        raise ValueError("Incompatible KITTI DC checkpoint fields: fusion_mode")
    if saved.get("voxel_encoder", "patchdepthbin") != protocol.get("voxel_encoder", "patchdepthbin"):
        raise ValueError("Incompatible KITTI DC checkpoint fields: voxel_encoder")
    # JSON normalizes tuples, so compare canonical serialized contracts.
    ignored = {"world_size"} if not resume else set()
    differences = [key for key in protocol if key not in ignored and
                   json.dumps(saved.get(key), sort_keys=True) != json.dumps(protocol[key], sort_keys=True)]
    if differences:
        raise ValueError("Incompatible KITTI DC checkpoint fields: " + ", ".join(differences))
    required = ("model", "optimizer", "epoch", "rng_states") if resume else ("model", "epoch")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint is missing {missing}")
    if resume and protocol.get("world_size") is not None:
        states = payload["rng_states"]
        if not isinstance(states, (list, tuple)) or len(states) != protocol["world_size"]:
            raise ValueError("Checkpoint rng_states must contain one state per rank")


def validate_existing_config(output_dir, protocol):
    """Reject accidental reuse of an output directory across experiment protocols."""
    for name in ("training_config.json", "eval_config.json", "smoke_config.json"):
        path = Path(output_dir) / name
        if path.is_file():
            saved = json.loads(path.read_text())["protocol"]
            validate_checkpoint(dict(model={}, epoch=0, protocol=saved), protocol, resume=False)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


@torch.no_grad()
def evaluate(model, validation_loader, args, device, rank, expected_samples):
    model.eval()
    accumulator = UnifiedDepthMetricAccumulator("kitti")
    if args.prediction_dir:
        args.prediction_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for step, batch in enumerate(validation_loader):
        with amp_context(args):
            outputs = forward_batch(model, batch, device)
        if not bool(torch.isfinite(outputs["dense_depth"]).all()):
            raise RuntimeError(f"Nonfinite prediction: {batch['sample_id']}")
        update_accumulator_from_batch(accumulator, outputs, batch)
        if args.prediction_dir:
            native = restore_to_native(outputs["dense_depth"][0, 0].float(), batch["resize_metadata"][0])
            drive, frame, camera = batch["sample_id"][0].split("/")
            path = args.prediction_dir / drive / camera / f"{frame}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, native.cpu().numpy())
        if rank == 0 and args.print_freq > 0 and (step + 1) % args.print_freq == 0:
            print(f"[val rank0] {step + 1}/{len(validation_loader)}", flush=True)
    result = _merge_distributed_accumulator(accumulator).compute()
    images = result["per_image_macro"]["all_valid"]["images"]
    if images != expected_samples:
        raise RuntimeError(f"Evaluation visited {images} images, expected {expected_samples}")
    result.update(source_dataset=args.dataset, protocol_id=PROTOCOL, val_samples=expected_samples,
                  elapsed_seconds=time.monotonic() - started)
    result["dataset"] = args.dataset
    result["split"] = "val"
    return result


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device))


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)


def _run(args):
    if args.voxel_encoder == "vfe" and args.model not in VFE_MODELS:
        raise ValueError("--voxel-encoder vfe requires a model with voxel tokens")
    if args.epochs <= 0 or args.smoke_steps < 0 or args.num_workers < 0 or args.print_freq < 0:
        raise ValueError("Invalid epochs, smoke steps, worker count or print frequency")
    if args.eval_only != bool(args.checkpoint):
        raise ValueError("eval-val requires --checkpoint; --checkpoint is evaluation-only")
    if args.prediction_dir and not args.eval_only:
        raise ValueError("--prediction-dir is only for eval-val")
    if args.mode == "smoke" and args.smoke_steps <= 0:
        raise ValueError("Smoke checks require a positive --smoke-steps")
    if args.mode != "smoke" and args.smoke_steps:
        raise ValueError("--smoke-steps is only for smoke")
    if args.resume and args.eval_only:
        raise ValueError("eval-val cannot resume training")
    if args.smoke_steps and (args.resume or args.eval_only):
        raise ValueError("Smoke checks cannot resume or evaluate training checkpoints")
    rank, world_size, local_rank, device = initialize_runtime(args)
    args.occany_ckpt = Path(args.occany_ckpt).resolve()
    pretrained_config = json.loads((args.occany_ckpt / "config.json").read_text())
    if pretrained_config.get("model_name") != "da3-base":
        raise ValueError("The seven full-KITTI experiments require DA3-BASE")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = (args.output_dir or output_root(args) / "single_frame" /
                       f"da3_base_{experiment_name(args)}" / f"left_long{args.input_long_side}_{args.epochs}ep_seed{args.seed}").resolve()
    validate_output_path(args.output_dir, args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not (args.resume or args.eval_only or args.smoke_steps):
        raise ValueError(f"Output directory is nonempty; use --resume explicitly: {args.output_dir}")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    checkpoint = args.resume or args.checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint else None
    use_lidar = args.model != "image"
    dataset_kwargs = dict(input_long_side=args.input_long_side, use_lidar=use_lidar,
                          scaled=args.model.endswith("scaled"), return_raw_points=args.voxel_encoder == "vfe")
    # Standalone val evaluation requires only labeled val, its RGB/RAW scans,
    # calibration and the checkpoint. It never enumerates training or test GT.
    train_dataset = None if args.eval_only else KITTIDepthCompletionDataset(args.kitti_dc_root, "train", **dataset_kwargs)
    val_dataset = KITTIDepthCompletionDataset(args.kitti_dc_root, "val", **dataset_kwargs)
    protocol = protocol_config(args, train_dataset, val_dataset, world_size,
                               payload.get("protocol", {}) if payload else None)
    validate_existing_config(args.output_dir, protocol)
    if payload is not None:
        validate_checkpoint(payload, protocol, resume=bool(args.resume))
    if args.smoke_steps:
        training_data = train_dataset
        validation_data = Subset(val_dataset, range(min(len(val_dataset), 2 * world_size + 1)))
    else:
        training_data, validation_data = train_dataset, val_dataset
    train_loader = None if args.eval_only else loader(training_data, args, training=True, rank=rank, world_size=world_size)
    val_loader = loader(validation_data, args, training=False, rank=rank, world_size=world_size)
    model = build_model(args).to(device)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(payload["model"], strict=True)
        if args.resume:
            optimizer.load_state_dict(payload["optimizer"])
            start_epoch = payload["epoch"] + 1
    if rank == 0:
        config_name = "smoke_config.json" if args.smoke_steps else "eval_config.json" if args.eval_only else "training_config.json"
        write_json(args.output_dir / config_name,
                   dict(protocol=protocol, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
        print(json.dumps(protocol, ensure_ascii=False), flush=True)
    if args.eval_only:
        metrics = evaluate(model, val_loader, args, device, rank, len(validation_data))
        if rank == 0:
            write_json(args.output_dir / f"eval_val_epoch{payload['epoch']}.json",
                       dict(epoch=payload["epoch"], checkpoint=str(checkpoint), datasets={args.dataset: metrics}))
        return
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    if args.resume:
        restore_rng(payload["rng_states"][rank], device)
        del payload
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        loss_sum, real_samples, steps = 0.0, 0, 0
        smoke_gradients = {}
        for step, batch in enumerate(train_loader):
            lr = lr_at(args, epoch + step / len(train_loader))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args):
                outputs = forward_batch(model, batch, device)
                gt = batch["dense_depth"].to(device, non_blocking=True)
                valid = batch["dense_depth_pixel_mask"].to(device, non_blocking=True)
                gt = torch.where(valid, gt, 0)
                loss, raw_loss, _, _ = dense_metric_depth_loss(outputs["dense_depth"], gt,
                    batch["dense_depth_frame_mask"].to(device), loss_weight=args.dense_depth_loss_weight)
                loss = loss * batch["ddp_loss_scale"].to(device).reshape(())
            if not bool(torch.isfinite(outputs["dense_depth"]).all()) or not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite training prediction/loss: {batch['sample_id']}")
            loss.backward()
            if args.smoke_steps:
                base = model.module if world_size > 1 else model
                prefixes = ["backbone", "dense_depth_head"]
                if getattr(base, "depth_token_branch_enabled", False):
                    prefixes.append("depth_patch_embed")
                if getattr(base, "voxel_token_branch_enabled", False):
                    if args.fusion_mode == "postfusion":
                        encoder = "vfe" if args.voxel_encoder == "vfe" else "depth_encoder"
                        prefixes.append("fusion.voxel_fusion." + encoder)
                    else:
                        prefixes.append("voxel_token_encoder")
                if args.fusion_mode == "postfusion" and args.model != "image":
                    prefixes.extend("fusion." + name + "_fusion.attention" for name in ("voxel", "depth")
                                    if getattr(base, name + "_token_branch_enabled"))
                for prefix in prefixes:
                    gradients = [p.grad.detach().float().square().sum() for name, p in base.named_parameters()
                                 if name.startswith(prefix + ".") and p.grad is not None]
                    if gradients:
                        norm = float(torch.stack(gradients).sum().sqrt())
                        if not math.isfinite(norm) or norm == 0:
                            raise RuntimeError(f"Invalid {prefix} gradient norm: {norm}")
                        smoke_gradients[prefix] = norm
                    elif args.fusion_mode == "postfusion":
                        raise RuntimeError(f"Missing enabled {prefix} gradients")
            optimizer.step()
            loss_sum += float(loss.detach())
            real_samples += int((~batch["sampler_padding"]).sum())
            steps += 1
            if rank == 0 and (step == 0 or args.print_freq > 0 and (step + 1) % args.print_freq == 0):
                print(f"[train] model={args.model} epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(train_loader)} "
                      f"loss={float(raw_loss):.6f} lr={lr:.8g} elapsed={time.monotonic() - started:.1f}s "
                      f"peak_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}", flush=True)
            if args.smoke_steps and steps >= args.smoke_steps:
                break
        totals = torch.tensor([loss_sum, real_samples, steps], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(totals)
        if not args.smoke_steps and int(totals[1]) != len(train_dataset):
            raise RuntimeError(f"Incomplete epoch: {int(totals[1])} != {len(train_dataset)}")
        training_elapsed = time.monotonic() - started
        base_model = model.module if world_size > 1 else model
        scheduled = (epoch + 1) % 5 == 0
        # Save the current epoch before evaluating it. All ranks wait for the
        # atomic checkpoint writes, then evaluate this same model state on val.
        if scheduled and not args.smoke_steps:
            states = [rng_state(device)]
            if world_size > 1:
                states = [None] * world_size
                dist.all_gather_object(states, rng_state(device))
            if rank == 0:
                payload = dict(model=base_model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                               rng_states=states, protocol=protocol)
                for name in (f"checkpoint-epoch{epoch + 1}.pth", "checkpoint-last.pth"):
                    temporary = args.output_dir / (name + ".tmp")
                    torch.save(payload, temporary)
                    temporary.replace(args.output_dir / name)
            if world_size > 1:
                dist.barrier()
        log = dict(epoch=epoch, train_loss=float(totals[0] / totals[2]), train_real_samples=int(totals[1]),
                   train_elapsed_seconds=training_elapsed,
                   peak_allocated_GiB=torch.cuda.max_memory_allocated(device) / 2**30,
                   smoke=bool(args.smoke_steps))
        if scheduled or args.smoke_steps:
            metrics = evaluate(base_model, val_loader, args, device, rank, len(validation_data))
            if not args.smoke_steps:
                # Validation loader RNG must not change the continuation from
                # the state recorded just before validation in the checkpoint.
                restore_rng(states[rank], device)
            log["datasets"] = {args.dataset: metrics}
            if not args.smoke_steps:
                log["checkpoint"] = str(args.output_dir / f"checkpoint-epoch{epoch + 1}.pth")
            if rank == 0:
                if not args.smoke_steps:
                    write_json(args.output_dir / f"eval_val_epoch{epoch}.json", log)
                print(f"[val] epoch={epoch + 1} images={len(validation_data)} "
                      f"AbsRel={metrics['pixel_micro']['all_valid']['abs_rel']:.6f} "
                      f"RMSE={metrics['pixel_micro']['all_valid']['rmse']:.6f}", flush=True)
        if args.smoke_steps:
            log["gradient_norms"] = smoke_gradients
            log["train_sampler_padding_slots"] = len(train_loader.sampler) * world_size - train_loader.sampler.length
            if rank == 0:
                write_json(args.output_dir / "smoke_metrics.json", log)
        if rank == 0:
            with (args.output_dir / "log.jsonl").open("a") as handle:
                handle.write(json.dumps(log) + "\n")
        if args.smoke_steps:
            break
        if world_size > 1:
            dist.barrier()
    if rank == 0:
        print("Smoke validation complete; no training checkpoints saved." if args.smoke_steps else "Training complete.", flush=True)


def main(argv=None, *, mode="train"):
    if mode not in ("train", "eval-val", "smoke"):
        raise ValueError(f"Unsupported mode: {mode}")
    args = get_parser(mode=mode).parse_args(argv)
    try:
        return _run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
