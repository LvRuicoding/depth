#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {train|train-natural|eval} [path arguments...]" >&2
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
  eval)
    exec torchrun --standalone --nproc_per_node=4 -m occany_depth_min.eval "$@"
    ;;
  *)
    echo "unknown stage: $stage (expected train, train-natural or eval)" >&2
    exit 2
    ;;
esac
