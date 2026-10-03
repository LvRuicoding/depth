#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

usage() {
  cat <<'USAGE'
Usage: scripts/run_kitti_dc_full.sh [train|eval-val|smoke|summarize] [arguments]
  --kitti-dc-root PATH      External KITTI full root (required except summarize)
  --da3-checkpoint PATH     External DA3-BASE directory (required except summarize)
  --output-root PATH        Suite output root (required)
  --fusion-mode MODE        prefusion (default) or postfusion
  --voxel-encoder ENCODER   patchdepthbin (default) or vfe
  --models "MODEL ..."      Ordered subset; defaults to seven models, or four for VFE
  --input-long-side N       Resize long side before patch-14 padding (default: 1232)
  --epochs N               Training schedule length (default: 10)
  --checkpoint-epochs "5 10"  Eval checkpoints (default: every 5 epochs up to --epochs)
  --num-workers N           Data-loader workers (default: 4)
  --print-freq N            Logging interval (default: 20; smoke: 1)
  --smoke-steps N           Bounded smoke iterations (default: 2)
  --nproc-per-node N        4; smoke also accepts 1
  --python PATH            Python executable (default: python from PATH)
  --torchrun PATH          Torchrun executable (default: torchrun from PATH)
  --dry-run                Preview commands without creating outputs or querying GPUs
Environment equivalents: KITTI_DC_ROOT, DA3_CKPT, OUTPUT_ROOT, FUSION_MODE,
VOXEL_ENCODER, MODELS, INPUT_LONG_SIDE, EPOCHS, CKPT_EPOCHS, NUM_WORKERS, PRINT_FREQ, SMOKE_STEPS,
NPROC_PER_NODE, PYTHON, TORCHRUN, DRY_RUN, CUDA_VISIBLE_DEVICES.
Postfusion VFE waits for each selected GPU to use <5120 MiB; configure NVIDIA_SMI,
GPU_MEMORY_LIMIT_MIB and GPU_POLL_SECONDS when necessary.
USAGE
}
die() { echo "$*" >&2; exit 2; }
need_value() { (( $# >= 2 )) && [[ -n "$2" && "$2" != --* ]] || die "Missing value for $1"; }

stage=train
if [[ $# -gt 0 && "$1" != -* ]]; then stage="$1"; shift; fi
python_command=${PYTHON:-python}
torchrun_command=${TORCHRUN:-torchrun}
nproc_per_node=${NPROC_PER_NODE:-4}
kitti_dc_root=${KITTI_DC_ROOT:-}
da3_checkpoint=${DA3_CKPT:-}
output_root=${OUTPUT_ROOT:-}
fusion_mode=${FUSION_MODE:-prefusion}
voxel_encoder=${VOXEL_ENCODER:-patchdepthbin}
models=${MODELS:-}
input_long_side=${INPUT_LONG_SIDE:-1232}
epochs=${EPOCHS:-10}
checkpoint_epochs=${CKPT_EPOCHS:-}
num_workers=${NUM_WORKERS:-4}
print_freq=${PRINT_FREQ:-}
smoke_steps=${SMOKE_STEPS:-2}
dry_run=${DRY_RUN:-0}

while (( $# )); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --python) need_value "$@"; python_command="$2"; shift 2 ;;
    --torchrun) need_value "$@"; torchrun_command="$2"; shift 2 ;;
    --nproc-per-node) need_value "$@"; nproc_per_node="$2"; shift 2 ;;
    --kitti-dc-root) need_value "$@"; kitti_dc_root="$2"; shift 2 ;;
    --da3-checkpoint) need_value "$@"; da3_checkpoint="$2"; shift 2 ;;
    --output-root) need_value "$@"; output_root="$2"; shift 2 ;;
    --fusion-mode) need_value "$@"; fusion_mode="$2"; shift 2 ;;
    --voxel-encoder) need_value "$@"; voxel_encoder="$2"; shift 2 ;;
    --models) need_value "$@"; models="$2"; shift 2 ;;
    --input-long-side) need_value "$@"; input_long_side="$2"; shift 2 ;;
    --epochs) need_value "$@"; epochs="$2"; shift 2 ;;
    --checkpoint-epochs) need_value "$@"; checkpoint_epochs="$2"; shift 2 ;;
    --num-workers) need_value "$@"; num_workers="$2"; shift 2 ;;
    --print-freq) need_value "$@"; print_freq="$2"; shift 2 ;;
    --smoke-steps) need_value "$@"; smoke_steps="$2"; shift 2 ;;
    *) die "Unknown argument: $1 (use --help)" ;;
  esac
done
case "$stage" in train|eval-val|smoke|summarize) ;; *) die "Unknown stage: $stage" ;; esac
[[ "$fusion_mode" == prefusion || "$fusion_mode" == postfusion ]] || die "Invalid fusion mode: $fusion_mode"
[[ "$voxel_encoder" == patchdepthbin || "$voxel_encoder" == vfe ]] || die "Invalid voxel encoder: $voxel_encoder"
[[ -z "${KITTI_DC_FULL_SUITE_FUSION:-}" || "$fusion_mode" == "$KITTI_DC_FULL_SUITE_FUSION" ]] || die "This suite requires fusion mode $KITTI_DC_FULL_SUITE_FUSION"
[[ -z "${KITTI_DC_FULL_SUITE_ENCODER:-}" || "$voxel_encoder" == "$KITTI_DC_FULL_SUITE_ENCODER" ]] || die "This suite requires voxel encoder $KITTI_DC_FULL_SUITE_ENCODER"
[[ "$input_long_side" =~ ^[1-9][0-9]*$ ]] || die "--input-long-side must be a positive integer"
[[ "$epochs" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
[[ -z "${KITTI_DC_FULL_SUITE_INPUT_LONG_SIDE:-}" || "$input_long_side" == "$KITTI_DC_FULL_SUITE_INPUT_LONG_SIDE" ]] || die "This suite requires input long side $KITTI_DC_FULL_SUITE_INPUT_LONG_SIDE"
[[ -z "${KITTI_DC_FULL_SUITE_EPOCHS:-}" || "$epochs" == "$KITTI_DC_FULL_SUITE_EPOCHS" ]] || die "This suite requires epochs $KITTI_DC_FULL_SUITE_EPOCHS"
[[ "$dry_run" == 0 || "$dry_run" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ -n "$output_root" ]] || die "--output-root (or OUTPUT_ROOT) is required"
# Resolve relative paths against the repository, matching the original suites.
[[ "$output_root" == /* ]] || output_root="$repo_dir/$output_root"
# Check even the smoke/validation subtree before tee or mkdir can write there.
"$python_command" -c '
import sys
from pathlib import Path
project = Path(sys.argv[1]).resolve()
fusion = sys.argv[2]
for value in sys.argv[3:]:
    output = Path(value).resolve()
    if output.is_relative_to((project / "output/depth/kitti").resolve()):
        raise SystemExit("Full KITTI DC outputs must be outside output/depth/kitti.")
    if fusion == "postfusion" and output.is_relative_to((project / "output/depth/kitti_dc_full").resolve()):
        raise SystemExit("Post-fusion outputs must be outside the existing prefusion kitti_dc_full directory.")
' "$repo_dir" "$fusion_mode" "$output_root" "$output_root/validation"

run_stage() {
  local name="$1"
  shift
  if [[ "$dry_run" == 1 ]]; then
    printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
  else
    mkdir -p "$output_root/suite_logs"
    "$@" 2>&1 | tee -a "$output_root/suite_logs/$name.log"
  fi
}
if [[ "$stage" == summarize ]]; then
  run_stage summarize "$python_command" -m occany_depth_min.kitti_dc_full summarize --output-root "$output_root"
  exit
fi

[[ -n "$kitti_dc_root" ]] || die "--kitti-dc-root (or KITTI_DC_ROOT) is required"
[[ -n "$da3_checkpoint" ]] || die "--da3-checkpoint (or DA3_CKPT) is required"
[[ "$nproc_per_node" == 4 || ( "$stage" == smoke && "$nproc_per_node" == 1 ) ]] || die "Train/eval require 4 GPUs; smoke accepts 1 or 4"
[[ "$num_workers" =~ ^[0-9]+$ ]] || die "--num-workers must be a nonnegative integer"
[[ "$smoke_steps" =~ ^[1-9][0-9]*$ ]] || die "--smoke-steps must be a positive integer"
if [[ -z "$print_freq" ]]; then
  print_freq=20
  [[ "$stage" != smoke ]] || print_freq=1
fi
[[ "$print_freq" =~ ^[0-9]+$ ]] || die "--print-freq must be a nonnegative integer"
if [[ -z "$models" ]]; then
  if [[ "$voxel_encoder" == vfe ]]; then
    models="voxel voxel_depth voxel_scaled voxel_depth_scaled"
  else
    models="image depth voxel voxel_depth depth_scaled voxel_scaled voxel_depth_scaled"
  fi
fi
read -r -a model_names <<< "$models"
(( ${#model_names[@]} )) || die "--models must contain at least one model"
declare -A seen_models=()
for model in "${model_names[@]}"; do
  case "$model" in image|depth|voxel|voxel_depth|depth_scaled|voxel_scaled|voxel_depth_scaled) ;; *) die "Unknown model: $model" ;; esac
  [[ "$voxel_encoder" != vfe || "$model" == voxel* ]] || die "VFE requires a voxel model: $model"
  [[ -z "${seen_models[$model]+set}" ]] || die "Duplicate model: $model"
  seen_models[$model]=1
done
if [[ "$stage" == eval-val ]]; then
  eval_epochs=()
  if [[ -n "$checkpoint_epochs" ]]; then
    read -r -a eval_epochs <<< "$checkpoint_epochs"
  else
    for (( epoch=5; epoch<=epochs; epoch+=5 )); do eval_epochs+=("$epoch"); done
  fi
  (( ${#eval_epochs[@]} )) || die "No scheduled checkpoint epochs; evaluation requires at least 5 training epochs"
  for epoch in "${eval_epochs[@]}"; do
    [[ "$epoch" =~ ^[1-9][0-9]*$ ]] && (( epoch % 5 == 0 && epoch <= epochs )) || die "Checkpoint epochs must be positive multiples of five and not exceed EPOCHS=$epochs"
  done
fi
if [[ "$nproc_per_node" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
(( ${#gpu_ids[@]} == nproc_per_node )) || die "CUDA_VISIBLE_DEVICES count must match --nproc-per-node"
declare -A seen_gpu_ids=()
for gpu_id in "${gpu_ids[@]}"; do
  [[ "$gpu_id" =~ ^([0-9]+|GPU-[[:alnum:]-]+)$ ]] || die "Invalid GPU ID: $gpu_id"
  [[ -z "${seen_gpu_ids[$gpu_id]+set}" ]] || die "Duplicate GPU ID: $gpu_id"
  seen_gpu_ids[$gpu_id]=1
done
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

wait_for_gpu_memory() {
  local threshold=${GPU_MEMORY_LIMIT_MIB:-5120}
  local poll_seconds=${GPU_POLL_SECONDS:-60}
  local nvidia_smi=${NVIDIA_SMI:-nvidia-smi}
  [[ "$threshold" =~ ^[1-9][0-9]*$ ]] || die "GPU_MEMORY_LIMIT_MIB must be a positive integer"
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || die "GPU_POLL_SECONDS must be a positive integer"
  if [[ "$dry_run" == 1 ]]; then
    echo "[dry-run] wait until GPUs $CUDA_VISIBLE_DEVICES each use less than $threshold MiB"
    return
  fi
  command -v "$nvidia_smi" >/dev/null 2>&1 || die "nvidia-smi is unavailable: $nvidia_smi"
  local gpu_id memory_used all_below_limit
  local -a usage_summary
  while true; do
    all_below_limit=1
    usage_summary=()
    for gpu_id in "${gpu_ids[@]}"; do
      memory_used=$("$nvidia_smi" --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits) || die "Failed to query GPU $gpu_id"
      memory_used=${memory_used//[[:space:]]/}
      [[ "$memory_used" =~ ^[0-9]+$ ]] || die "Invalid memory usage for GPU $gpu_id: $memory_used"
      usage_summary+=("GPU$gpu_id=${memory_used}MiB")
      if (( 10#$memory_used >= threshold )); then all_below_limit=0; fi
    done
    echo "[gpu-wait] ${usage_summary[*]} threshold=<$threshold MiB"
    if (( all_below_limit )); then return; fi
    echo "[gpu-wait] Checking again in ${poll_seconds}s"
    sleep "$poll_seconds"
  done
}
if [[ "$fusion_mode" == postfusion && "$voxel_encoder" == vfe ]]; then wait_for_gpu_memory; fi

for model in "${model_names[@]}"; do
  experiment="$model"
  [[ "$voxel_encoder" != vfe ]] || experiment="${model}_vfe"
  run_dir="$output_root/single_frame/da3_base_$experiment/left_long${input_long_side}_${epochs}ep_seed0"
  if [[ "$stage" == smoke ]]; then run_dir="$output_root/validation/$experiment"; fi
  command=("$torchrun_command" --standalone --nproc_per_node="$nproc_per_node"
    -m occany_depth_min.kitti_dc_full "$stage"
    --model "$model" --fusion-mode "$fusion_mode" --voxel-encoder "$voxel_encoder"
    --kitti-dc-root "$kitti_dc_root" --da3-checkpoint "$da3_checkpoint"
    --input-long-side "$input_long_side" --epochs "$epochs"
    --output-dir "$run_dir" --num-workers "$num_workers" --print-freq "$print_freq")
  case "$stage" in
    train)
      if [[ -f "$run_dir/checkpoint-last.pth" ]]; then command+=(--resume "$run_dir/checkpoint-last.pth"); fi
      run_stage "train_$experiment" "${command[@]}"
      ;;
    eval-val)
      for epoch in "${eval_epochs[@]}"; do
        run_stage "eval-val_epoch${epoch}_$experiment" "${command[@]}" --checkpoint "$run_dir/checkpoint-epoch$epoch.pth"
      done
      ;;
    smoke) run_stage "smoke_$experiment" "${command[@]}" --smoke-steps "$smoke_steps" ;;
  esac
done
if [[ "$stage" != smoke ]]; then
  run_stage summarize "$python_command" -m occany_depth_min.kitti_dc_full summarize --output-root "$output_root"
fi
