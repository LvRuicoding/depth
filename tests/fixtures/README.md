# Full KITTI reference contracts

`kitti_dc_full_reference.json` records the ordered state-dict structure,
parameter count, experiment-module initialization and final CPU RNG state
for all 22 configurations. It contains hashes, not weights or experiment
outputs.

The reference is OccAny commit `ec177ca1011d62a52e530e4df9c50bdf21a16306`,
constructed with seed 0, float32 backbone execution and DA3-BASE weights
whose `model.safetensors` SHA256 is
`e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`.
State signatures hash the key, shape and dtype in registration order;
initialization signatures also hash contiguous tensor bytes, excluding the
pretrained `backbone.` keys. RNG signatures hash `torch.get_rng_state()` bytes.

Default tests need neither the source checkout nor pretrained weights. For
an additional source/target output, loss, gradient, constructor RNG and
checkpoint audit, run `tests/verify_kitti_dc_full_reference.py` with explicit
external reference and weight paths as shown in the repository README.
