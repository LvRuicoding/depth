#!/usr/bin/env bash
set -euo pipefail

# Seven single-frame left-camera prefusion experiments at 518x168 for five epochs.
# Keep the full FOV: resize the long side to 518, then pad to patch size 14.
# Initialize only the backbone from DA3-BASE; all parameters remain trainable.
# External KITTI_DC_ROOT and DA3_CKPT are required except for summarize.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
export OUTPUT_ROOT=${OUTPUT_ROOT:-${repo_dir}/output/depth/kitti_dc_full_518x168_5ep}
export MODELS="image depth voxel voxel_depth depth_scaled voxel_scaled voxel_depth_scaled"
export FUSION_MODE=prefusion
export VOXEL_ENCODER=patchdepthbin
export INPUT_LONG_SIDE=518
export EPOCHS=5
export CKPT_EPOCHS=5
export KITTI_DC_FULL_SUITE_FUSION=prefusion
export KITTI_DC_FULL_SUITE_ENCODER=patchdepthbin
export KITTI_DC_FULL_SUITE_INPUT_LONG_SIDE=518
export KITTI_DC_FULL_SUITE_EPOCHS=5
exec bash "$script_dir/run_kitti_dc_full.sh" "$@"
