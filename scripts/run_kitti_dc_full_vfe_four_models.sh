#!/usr/bin/env bash
set -euo pipefail

# Original suite entry point; its fusion mode and encoder are fixed.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FUSION_MODE=prefusion
export VOXEL_ENCODER=vfe
export KITTI_DC_FULL_SUITE_FUSION=prefusion
export KITTI_DC_FULL_SUITE_ENCODER=vfe
exec bash "$script_dir/run_kitti_dc_full.sh" "$@"
