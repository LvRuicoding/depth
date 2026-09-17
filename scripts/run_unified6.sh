#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {train|train-natural|eval|train-postfusion|train-postfusion-natural|eval-postfusion|train-kitti|eval-kitti|train-postfusion-kitti|eval-postfusion-kitti} [path arguments...]" >&2
  exit 2
fi

stage="$1"
shift
case "$stage" in
  train)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@"
    ;;
  train-natural)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@" --sampling natural
    ;;
  train-postfusion)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@" --model postfusion
    ;;
  train-postfusion-natural)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@" --model postfusion --sampling natural
    ;;
  eval)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.eval "$@"
    ;;
  eval-postfusion)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.eval "$@" --model postfusion
    ;;
  train-kitti)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@" --dataset kitti
    ;;
  eval-kitti)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.eval "$@" --dataset kitti
    ;;
  train-postfusion-kitti)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.train "$@" --dataset kitti --model postfusion
    ;;
  eval-postfusion-kitti)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.eval "$@" --dataset kitti --model postfusion
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
