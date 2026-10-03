"""Summarize native-grid val results separately for baseline and VFE runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(output_root):
    output_root = Path(output_root)
    groups = {"patchdepthbin": [], "vfe": []}
    for run_dir in sorted((output_root / "single_frame").glob("da3_base_*/*")):
        files = sorted(run_dir.glob("eval_val_epoch*.json"), key=lambda p: int(p.stem.removeprefix("eval_val_epoch")))
        results = [json.loads(path.read_text()) for path in files]
        if not results:
            continue
        config = json.loads((run_dir / "training_config.json").read_text())["protocol"]
        encoder = config.get("voxel_encoder", "patchdepthbin")
        for result in results:
            metrics = result["datasets"]["kitti_dc_full"]
            row = dict(model=config["model"], experiment=config.get("experiment", config["model"]),
                       fusion_mode=config.get("fusion_mode", "prefusion"),
                       fusion_stage=config.get("fusion_stage", "none" if config["model"] == "image" else "before_da3"),
                       depth_token_source=config.get("depth_token_source"),
                       voxel_token_source=config.get("voxel_token_source"),
                       voxel_encoder=encoder, checkpoint_kind="epoch", epoch=result["epoch"] + 1,
                       split="val", images=metrics["val_samples"], input_long_side=config["input_long_side"],
                       checkpoint=str(run_dir / f"checkpoint-epoch{result['epoch'] + 1}.pth"))
            for average in ("pixel_micro", "per_image_macro"):
                for key in ("abs_rel", "rmse", "mae", "irmse", "imae", "delta1", "valid_pixels"):
                    if key in metrics[average]["all_valid"]:
                        row[f"{average}_{key}"] = metrics[average]["all_valid"][key]
            groups[encoder].append(row)
    if not any(groups.values()):
        print("No completed full-KITTI val results yet.")
        return
    for encoder, rows in groups.items():
        if not rows:
            continue
        name = "vfe_four_models_val_summary.csv" if encoder == "vfe" else "seven_models_val_summary.csv"
        path = output_root / name
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", dest="output_root", required=True)
    summarize(parser.parse_args(argv).output_root)


if __name__ == "__main__":
    main()
