from __future__ import annotations

import csv
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from occany_depth_min.kitti_dc_full.data import (
    KITTIDepthCompletionDataset, FullCoverageDistributedSampler, collate_kitti_dc_full,
    letterbox_full_image, raw_left_geometry,
)
from occany_depth_min.kitti_dc_full.train import validate_checkpoint, get_parser, lr_at, protocol_config
from occany_depth_min.kitti_dc_full.summary import summarize
from occany_depth_min.metrics import restore_to_native, UnifiedDepthMetricAccumulator


@pytest.mark.parametrize("w,h", [(1242, 375), (1224, 370), (1238, 374), (1226, 370), (1241, 376)])
@pytest.mark.parametrize("long_side,padded_hw", [(1232, (378, 1232)), (518, (168, 518))])
def test_full_fov_geometry_and_half_pixel_projection(w, h, long_side, padded_hw):
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    rgb[:, 0], rgb[:, -1] = [255, 0, 0], [0, 255, 0]
    gt = np.full((h, w), 7.5, dtype=np.float32)
    K = np.array([[720., 0, w / 2 - 7], [0, 710., h / 2 - 11], [0, 0, 1]])
    out, depth, mask, resized_K, metadata = letterbox_full_image(rgb, gt, K, long_side=long_side)
    assert out.shape == (*padded_hw, 3)
    assert depth.shape == mask.shape == padded_hw
    nh, nw = metadata["resized_hw"]
    top, bottom, left, right = metadata["pad_tblr"]
    assert nh <= h and nw <= w
    assert mask.sum() == nh * nw
    assert np.all(depth[mask] == 7.5) and np.all(depth[~mask] == 0)
    assert out[top + nh // 2, left, 0] > out[top + nh // 2, left, 1]
    assert out[top + nh // 2, left + nw - 1, 1] > out[top + nh // 2, left + nw - 1, 0]
    xyz = np.array([2., 1., 10.])
    old_uv = (K @ xyz)[:2] / xyz[2]
    new_uv = (resized_K @ xyz)[:2] / xyz[2]
    expected = (old_uv + 0.5) * np.array(metadata["scale_xy"]) - 0.5 + [left, top]
    np.testing.assert_allclose(new_uv, expected, atol=5e-5)
    native = restore_to_native(torch.from_numpy(depth), metadata)
    assert native.shape == (h, w)
    torch.testing.assert_close(native, torch.full((h, w), 7.5))


def test_raw_calibration_includes_rectification_and_left_translation(tmp_path):
    P = np.array([[8., 0, 8., 0.8], [0, 8., 4., 0.4], [0, 0, 1., 0.02]])
    R = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    raw_R = np.array([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
    raw_t = np.array([0.1, 0.2, 0.3])
    (tmp_path / "calib_cam_to_cam.txt").write_text(
        "P_rect_02: " + " ".join(map(str, P.ravel())) + "\nR_rect_00: " + " ".join(map(str, R.ravel())) + "\n")
    (tmp_path / "calib_velo_to_cam.txt").write_text(
        "R: " + " ".join(map(str, raw_R.ravel())) + "\nT: " + " ".join(map(str, raw_t)) + "\n")
    K, T = raw_left_geometry(tmp_path)
    raw = np.eye(4); raw[:3, :3], raw[:3, 3] = raw_R, raw_t
    rect = np.eye(4); rect[:3, :3] = R
    np.testing.assert_allclose(K @ T[:3], P @ rect @ raw, atol=1e-10)


@pytest.fixture
def small_raw_root(tmp_path):
    date, drive = "2011_09_26", "2011_09_26_drive_0001_sync"
    date_root = tmp_path / date
    date_root.mkdir()
    (date_root / "calib_cam_to_cam.txt").write_text(
        "P_rect_02: 8 0 8 0 0 8 4 0 0 0 1 0\nR_rect_00: 1 0 0 0 1 0 0 0 1\n")
    (date_root / "calib_velo_to_cam.txt").write_text("R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n")
    gt_dir = tmp_path / "val" / drive / "proj_depth/groundtruth/image_02"
    rgb_dir = date_root / drive / "image_02/data"
    bin_dir = date_root / drive / "velodyne_points/data"
    for path in (gt_dir, rgb_dir, bin_dir, tmp_path / "train"):
        path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 16), 10 * 256, dtype=np.uint16)).save(gt_dir / "0000000005.png")
    Image.fromarray(np.full((8, 16, 3), 128, dtype=np.uint8)).save(rgb_dir / "0000000005.png")
    # Last point lands on padded pixels; it must not become an observation.
    points = np.array([[-3.75, -1.25, 5, 0], [-3.75, 0, 10, 0], [0, 3.75, 15, 0],
                       [7.5, -5, 20, 0], [-10, 0, 5, 0]], dtype=np.float32)
    points.tofile(bin_dir / "0000000005.bin")
    return tmp_path, bin_dir


def test_raw_projection_shared_with_knn_and_excludes_padding(small_raw_root):
    root, _ = small_raw_root
    dataset = KITTIDepthCompletionDataset(root, "val", scaled=True, require_full=False)
    sample = dataset[0]
    assert sample["native_dense_depth"].shape == (8, 16)
    assert not sample["sparse_depth_mask"][~sample["image_valid_mask"]].any()
    assert int(sample["sparse_depth_mask"].sum()) == 4
    assert set(sample["sparse_depth"][sample["sparse_depth_mask"]].tolist()) == {5., 10., 15., 20.}
    prompt = sample["knn_depth"]
    torch.testing.assert_close(prompt[sample["sparse_depth_mask"]], sample["sparse_depth"][sample["sparse_depth_mask"]])
    assert prompt.min() == 5 and prompt.max() == 20
    batch = collate_kitti_dc_full([sample])
    assert batch["sparse_depth"].shape == (1, 1, 14, 28)
    accumulator = UnifiedDepthMetricAccumulator("kitti")
    accumulator.update_image(torch.full((8, 16), 10.), sample["native_dense_depth"])
    assert accumulator.compute()["pixel_micro"]["all_valid"]["valid_pixels"] == 128


def test_image_model_does_not_read_lidar(small_raw_root):
    root, bin_dir = small_raw_root
    for path in bin_dir.iterdir():
        path.unlink()
    sample = KITTIDepthCompletionDataset(root, "val", use_lidar=False, require_full=False)[0]
    assert not sample["sparse_depth_mask"].any()
    with pytest.raises(FileNotFoundError):
        KITTIDepthCompletionDataset(root, "val", use_lidar=True, require_full=False)


def test_no_test_split_or_silent_partial_dataset(small_raw_root):
    root, _ = small_raw_root
    with pytest.raises(ValueError, match="Unsupported split"):
        KITTIDepthCompletionDataset(root, "test", require_full=False)
    with pytest.raises(ValueError, match="expected 3426"):
        KITTIDepthCompletionDataset(root, "val")
    with pytest.raises(SystemExit):
        get_parser().parse_args(["--model", "image", "--eval-split", "test"])


def test_val_only_needs_neither_train_nor_test_gt(small_raw_root):
    root, _ = small_raw_root
    (root / "train").rmdir()
    dataset = KITTIDepthCompletionDataset(root, "val", require_full=False)
    assert len(dataset) == 1
    args = get_parser().parse_args(["--model", "depth", "--da3-checkpoint", str(root)])
    (root / "model.safetensors").write_bytes(b"test-reference-weights")
    protocol = protocol_config(args, None, dataset, 1,
                               {"train_samples": 42949, "train_manifest_sha256": "saved-training-index"})
    assert protocol["train_samples"] == 42949
    assert protocol["train_manifest_sha256"] == "saved-training-index"
    assert protocol["val_samples"] == 1 and protocol["eval_split"] == "val"


def test_summary_lists_every_saved_epoch(tmp_path):
    run = tmp_path / "single_frame/da3_base_depth/left_long1232_10ep_seed0"
    run.mkdir(parents=True)
    (run / "training_config.json").write_text(json.dumps({"protocol": {"model": "depth", "input_long_side": 1232}}))
    for epoch, rmse, abs_rel in [(4, 2., .9), (9, 3., .1)]:
        metrics = dict(val_samples=3426, pixel_micro={"all_valid": {"rmse": rmse, "abs_rel": abs_rel}},
                       per_image_macro={"all_valid": {"rmse": rmse + .1, "abs_rel": abs_rel + .01}})
        (run / f"eval_val_epoch{epoch}.json").write_text(json.dumps(
            {"epoch": epoch, "datasets": {"kitti_dc_full": metrics}}))
    summarize(tmp_path)
    with (tmp_path / "seven_models_val_summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert [(row["checkpoint_kind"], row["epoch"], row["split"]) for row in rows] == [("epoch", "5", "val"), ("epoch", "10", "val")]

    assert [row["checkpoint"].split("/")[-1] for row in rows] == ["checkpoint-epoch5.pth", "checkpoint-epoch10.pth"]


@pytest.mark.parametrize("length", [1, 3, 4, 5, 42949])
def test_ddp_full_coverage_and_tail_weights(length):
    samplers = [FullCoverageDistributedSampler(range(length), num_replicas=4, rank=rank) for rank in range(4)]
    entries = [list(sampler) for sampler in samplers]
    visited = [index for rank in entries for index, weight, padding in rank if not padding]
    assert Counter(visited) == Counter(range(length))
    for step in zip(*entries):
        assert sum(weight for _, weight, _ in step) == 4
        assert all(weight == 0 for _, weight, padding in step if padding)
    samplers[0].set_epoch(1)
    if length > 4:
        assert list(samplers[0]) != entries[0]


def test_resume_rejects_data_resolution_or_schedule_changes():
    protocol = dict(dataset="kitti_dc_full", input_long_side=1232, epochs=10, world_size=4,
                    model="depth", train_manifest_sha256="manifest-a", betas=(0.9, 0.95))
    payload = dict(protocol=json.loads(json.dumps(protocol)), model={}, optimizer={}, epoch=0,
                   rng_states=[{}] * 4, best_rmse=5.)
    validate_checkpoint(payload, protocol, resume=True)
    for key, value in [("input_long_side", 1024), ("epochs", 20), ("model", "voxel"),
                       ("dataset", "kitti"), ("train_manifest_sha256", "manifest-b"), ("world_size", 1)]:
        with pytest.raises(ValueError, match=key):
            validate_checkpoint(payload, {**protocol, key: value}, resume=True)
    validate_checkpoint(payload, {**protocol, "world_size": 1}, resume=False)


def test_smoke_sampler_reaches_partial_tail_without_changing_full_data():
    samplers = [FullCoverageDistributedSampler(range(42949), num_replicas=4, rank=rank, max_samples=5)
                for rank in range(4)]
    entries = [list(sampler) for sampler in samplers]
    assert all(len(rank) == 2 for rank in entries)
    assert sum(not padding for rank in entries for _, _, padding in rank) == 5
    assert sum(padding for rank in entries for _, _, padding in rank) == 3
    assert sum(weight for rank in entries for _, weight, _ in rank) == 8




@pytest.mark.parametrize("long_side,epochs", [(1232, 10), (518, 5)])
def test_training_saves_before_val_only_every_five_epochs(tmp_path, monkeypatch, long_side, epochs):
    from occany_depth_min.kitti_dc_full import train as trainer

    pretrained = tmp_path / 'pretrained'
    pretrained.mkdir()
    (pretrained / 'config.json').write_text('{"model_name": "da3-base"}')
    output = tmp_path / 'run'
    model = torch.nn.Linear(1, 1, bias=False)
    batch = dict(dense_depth=torch.ones(1), dense_depth_pixel_mask=torch.ones(1, dtype=torch.bool),
                 dense_depth_frame_mask=torch.ones(1, dtype=torch.bool), ddp_loss_scale=torch.ones(1),
                 sampler_padding=torch.zeros(1, dtype=torch.bool), sample_id=['fixture'])
    current_epoch = [0]

    class TrainingLoader:
        sampler = SimpleNamespace(set_epoch=lambda epoch: current_epoch.__setitem__(0, epoch))

        def __len__(self):
            return 1

        def __iter__(self):
            yield batch

    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('LOCAL_RANK', '0')
    cpu = torch.device('cpu')
    monkeypatch.setattr(trainer, 'initialize_runtime', lambda args: (0, 1, 0, cpu))
    monkeypatch.setattr(trainer, 'amp_context', lambda args: __import__('contextlib').nullcontext())
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *args: 0)
    monkeypatch.setattr(trainer, 'rng_state', lambda device: {})
    monkeypatch.setattr(trainer, 'restore_rng', lambda *args: None)
    dataset_sizes = []

    def dataset(*args, **kwargs):
        dataset_sizes.append(kwargs['input_long_side'])
        return [0]

    monkeypatch.setattr(trainer, 'KITTIDepthCompletionDataset', dataset)
    monkeypatch.setattr(trainer, 'protocol_config', lambda *args: {})
    monkeypatch.setattr(trainer, 'loader', lambda *args, training, **kwargs: TrainingLoader() if training else [0])
    monkeypatch.setattr(trainer, 'build_model', lambda args: model)
    monkeypatch.setattr(trainer, 'forward_batch', lambda *args: {'dense_depth': model(torch.ones(1, 1)).reshape(1)})
    monkeypatch.setattr(trainer, 'dense_metric_depth_loss',
                        lambda pred, *args, **kwargs: (pred.square().mean(), pred.square().mean(), None, None))
    evaluated = []

    def evaluate(saved_model, *args):
        epoch = current_epoch[0] + 1
        evaluated.append(epoch)
        checkpoint = torch.load(output / f'checkpoint-epoch{epoch}.pth', weights_only=False)
        assert checkpoint['epoch'] == epoch - 1
        assert 'optimizer' in checkpoint and 'rng_states' in checkpoint
        for name, value in saved_model.state_dict().items():
            torch.testing.assert_close(checkpoint['model'][name], value)
        return {'pixel_micro': {'all_valid': {'rmse': 1., 'abs_rel': .1}}}

    monkeypatch.setattr(trainer, 'evaluate', evaluate)
    trainer.main(['--model', 'image', '--num-workers', '0',
                  '--input-long-side', str(long_side), '--epochs', str(epochs),
                  '--da3-checkpoint', str(pretrained), '--output-dir', str(output)])
    expected_epochs = list(range(5, epochs + 1, 5))
    assert dataset_sizes == [long_side, long_side]
    assert evaluated == expected_epochs
    assert {path.name for path in output.glob('*.pth')} == {
        *(f'checkpoint-epoch{epoch}.pth' for epoch in expected_epochs), 'checkpoint-last.pth'}
    assert {path.name for path in output.glob('eval_val_*.json')} == {
        f'eval_val_epoch{epoch - 1}.json' for epoch in expected_epochs}
    logs = [json.loads(line) for line in (output / 'log.jsonl').read_text().splitlines()]
    assert len(logs) == epochs
    assert [log['epoch'] + 1 for log in logs if 'datasets' in log] == expected_epochs


def test_raw_points_returned_once_with_intensity_and_ragged_collate(small_raw_root, monkeypatch):
    root, bin_dir = small_raw_root
    point_path = bin_dir / '0000000005.bin'
    original = np.fromfile(point_path, np.float32).reshape(-1, 4)
    original[:, 3] = [.1, .2, .3, .4, .5]
    original.tofile(point_path)
    read = np.fromfile
    reads = []

    def counted_read(*args, **kwargs):
        reads.append(args[0])
        return read(*args, **kwargs)

    monkeypatch.setattr(np, 'fromfile', counted_read)
    baseline = KITTIDepthCompletionDataset(root, 'val', require_full=False, scaled=True)[0]
    dataset = KITTIDepthCompletionDataset(root, 'val', require_full=False, scaled=True, return_raw_points=True)
    reads.clear()
    raw = dataset[0]
    assert reads == [point_path]
    torch.testing.assert_close(raw['points_per_frame'][0], torch.from_numpy(original))
    for key in ('sparse_depth', 'sparse_depth_mask', 'knn_depth'):
        torch.testing.assert_close(raw[key], baseline[key])
    second = {**raw, 'points_per_frame': [raw['points_per_frame'][0][:2]]}
    batch = collate_kitti_dc_full([raw, second])
    assert [sample[0].shape for sample in batch['points_per_frame']] == [(5, 4), (2, 4)]
    with pytest.raises(ValueError, match='use_lidar'):
        KITTIDepthCompletionDataset(root, 'val', require_full=False, use_lidar=False, return_raw_points=True)




def test_vfe_protocol_rejects_cross_encoder_both_directions(small_raw_root):
    from occany_depth_min.kitti_dc_full.train import experiment_name

    root, _ = small_raw_root
    dataset = KITTIDepthCompletionDataset(root, 'val', require_full=False)
    (root / 'model.safetensors').write_bytes(b'test-reference-weights')
    args = get_parser().parse_args(['--model', 'voxel_depth_scaled', '--da3-checkpoint', str(root)])
    baseline = protocol_config(args, dataset, dataset, 4)
    assert 'voxel_encoder' not in baseline
    args.voxel_encoder = 'vfe'
    vfe = protocol_config(args, dataset, dataset, 4)
    assert experiment_name(args) == 'voxel_depth_scaled_vfe'
    assert vfe['voxel_intensity'] is True and vfe['voxel_grid'] == (128, 16, 128)
    for saved, current in [(baseline, vfe), (vfe, baseline)]:
        with pytest.raises(ValueError, match='voxel_encoder'):
            validate_checkpoint(dict(model={}, epoch=4, protocol=saved), current, resume=False)
    for protocol in (baseline, vfe):
        validate_checkpoint(dict(model={}, epoch=4, protocol=protocol), protocol, resume=False)
    for field in ('voxel_intensity', 'voxel_input', 'voxel_grid'):
        with pytest.raises(ValueError, match=field):
            validate_checkpoint(dict(model={}, epoch=4, protocol=vfe), {**vfe, field: None}, resume=False)


@pytest.mark.parametrize('model', ['image', 'depth', 'depth_scaled'])
def test_vfe_rejects_models_without_voxel_tokens_before_initialization(model):
    from occany_depth_min.kitti_dc_full.train import main

    with pytest.raises(ValueError, match='voxel tokens'):
        main(['--model', model, '--voxel-encoder', 'vfe'])


def test_summary_separates_vfe_and_baselines(tmp_path):
    for encoder in ('patchdepthbin', 'vfe'):
        experiment = 'voxel' + ('_vfe' if encoder == 'vfe' else '')
        run = tmp_path / 'single_frame' / f'da3_base_{experiment}' / 'left_long1232_10ep_seed0'
        run.mkdir(parents=True)
        protocol = dict(model='voxel', input_long_side=1232)
        if encoder == 'vfe':
            protocol.update(voxel_encoder=encoder, experiment=experiment)
        (run / 'training_config.json').write_text(json.dumps({'protocol': protocol}))
        metrics = dict(val_samples=3426, pixel_micro={'all_valid': {'rmse': 2.}},
                       per_image_macro={'all_valid': {'rmse': 2.1}})
        (run / 'eval_val_epoch4.json').write_text(json.dumps({'epoch': 4, 'datasets': {'kitti_dc_full': metrics}}))
    summarize(tmp_path)
    for filename, encoder in [('seven_models_val_summary.csv', 'patchdepthbin'), ('vfe_four_models_val_summary.csv', 'vfe')]:
        with (tmp_path / filename).open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1 and rows[0]['voxel_encoder'] == encoder
        assert rows[0]['experiment'] == ('voxel_vfe' if encoder == 'vfe' else 'voxel')




def test_rgb_normalization_matches_original_torchvision_arithmetic():
    from occany_depth_min.kitti_dc_full.data import normalize_rgb
    values = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.stack((values, values[::-1], values.T), axis=-1)
    original = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255
    original = (original - .5) / .5
    torch.testing.assert_close(normalize_rgb(rgb), original, rtol=0, atol=0)


def test_training_defaults_and_mode_validation(monkeypatch):
    from occany_depth_min.kitti_dc_full import train
    args = train.get_parser().parse_args(['--model', 'depth'])
    assert (args.epochs, args.input_long_side, args.batch_size, args.amp, args.seed) == (10, 1232, 1, 'bf16', 0)
    assert (args.lr, args.min_lr, args.weight_decay, args.warmup_epochs) == (1e-4, 1e-6, 1e-4, 1)
    assert [lr_at(args, progress) for progress in (0, 1, 10)] == [0, 1e-4, 1e-6]
    small = train.get_parser().parse_args(['--model', 'depth', '--input-long-side', '518', '--epochs', '5'])
    assert (small.epochs, small.input_long_side, small.batch_size, small.amp, small.seed) == (5, 518, 1, 'bf16', 0)
    assert [lr_at(small, progress) for progress in (0, 1, 5)] == [0, 1e-4, 1e-6]
    with pytest.raises(ValueError, match='requires --checkpoint'):
        train.main(['--model', 'image'], mode='eval-val')
    with pytest.raises(ValueError, match='only for smoke'):
        train.main(['--model', 'image', '--smoke-steps', '1'])
    with pytest.raises(ValueError, match='positive'):
        train.main(['--model', 'image', '--smoke-steps', '0'], mode='smoke')
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='four GPUs'):
        train.main(['--model', 'image'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no CUDA'):
        train.main(['--model', 'image'], mode='smoke')


@pytest.mark.parametrize("flag", ['--input-long-side', '--epochs'])
@pytest.mark.parametrize("value", ['0', '-1', '5.5'])
def test_resolution_and_schedule_reject_invalid_values(flag, value):
    with pytest.raises(SystemExit):
        get_parser().parse_args(['--model', 'image', flag, value])


def test_protocol_fusion_compatibility_and_legacy_defaults(tmp_path):
    from occany_depth_min.kitti_dc_full import train
    (tmp_path / 'model.safetensors').write_bytes(b'reference')

    class Data:
        manifest_sha256 = 'manifest'

        def __len__(self):
            return 3

    args = train.get_parser().parse_args(['--model', 'voxel_depth_scaled', '--da3-checkpoint', str(tmp_path)])
    pre = train.protocol_config(args, Data(), Data(), 4)
    assert 'fusion_mode' not in pre and 'voxel_encoder' not in pre
    assert pre['model_class'] == 'Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNNPromptDAScaledUnified6Model'
    args.fusion_mode = 'postfusion'
    after = train.protocol_config(args, Data(), Data(), 4)
    assert after['fusion_order'] == ['voxel', 'depth']
    assert after['model_class'] == 'Stage1DepthKITTIFullPostFusionPromptDAScaledModel'
    for saved, current in ((pre, after), (after, pre)):
        with pytest.raises(ValueError, match='fusion_mode'):
            train.validate_checkpoint(dict(model={}, epoch=4, protocol=saved), current, resume=False)
    for protocol in (pre, after):
        train.validate_checkpoint(dict(model={}, epoch=4, optimizer={}, rng_states=[{}] * 4,
                                       protocol=protocol), protocol, resume=True)
    args.model = 'image'
    rgb = train.protocol_config(args, Data(), Data(), 4)
    assert rgb['fusion_stage'] == 'none' and rgb['fusion_order'] == []
    assert train.model_class(args) is train.MODEL_CLASSES['image']


def _stub_cpu_training(monkeypatch):
    """Run real AdamW/save/resume logic while substituting costly I/O and CUDA."""
    import contextlib
    import random
    from occany_depth_min.kitti_dc_full import train
    cpu = torch.device('cpu')
    monkeypatch.setattr(train, 'initialize_runtime', lambda args: (0, 1, 0, cpu))
    monkeypatch.setattr(train, 'amp_context', lambda args: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *args: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *args: 0)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda *args: torch.tensor([1, 2], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda *args: None)
    monkeypatch.setattr(train, 'KITTIDepthCompletionDataset', lambda *args, **kwargs: [0])
    monkeypatch.setattr(train, 'protocol_config', lambda *args: {'world_size': 1})
    batch = dict(dense_depth=torch.ones(1, 1, 1, 1),
                 dense_depth_pixel_mask=torch.ones(1, 1, 1, 1, dtype=torch.bool),
                 dense_depth_frame_mask=torch.ones(1, 1, dtype=torch.bool), ddp_loss_scale=torch.ones(1),
                 sampler_padding=torch.zeros(1, dtype=torch.bool), sample_id=['fixture'])

    class TrainingLoader:
        sampler = FullCoverageDistributedSampler([0])

        def __len__(self):
            return 1

        def __iter__(self):
            yield batch

    monkeypatch.setattr(train, 'loader', lambda *args, training, **kwargs: TrainingLoader() if training else [0])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1, bias=False)
            self.dense_depth_head = torch.nn.Linear(1, 1, bias=False)

    monkeypatch.setattr(train, 'build_model', lambda args: Model())

    def forward_batch(model, *args):
        value = 1 + torch.rand(1, 1) + np.random.uniform() + random.random()
        return {'dense_depth': model.dense_depth_head(model.backbone(value)).sigmoid().reshape(1, 1, 1, 1)}

    monkeypatch.setattr(train, 'forward_batch', forward_batch)

    def evaluate(*args):
        # Evaluation consumes every RNG; training continuation restores them.
        torch.rand(3)
        np.random.rand(3)
        [random.random() for _ in range(3)]
        return {'pixel_micro': {'all_valid': {'rmse': 1., 'abs_rel': .1}}}

    monkeypatch.setattr(train, 'evaluate', evaluate)
    return train


def test_resume_restores_optimizer_and_all_rngs(tmp_path, monkeypatch):
    train = _stub_cpu_training(monkeypatch)
    pretrained = tmp_path / 'pretrained'
    pretrained.mkdir()
    (pretrained / 'config.json').write_text('{"model_name": "da3-base"}')
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    base_args = ['--model', 'image', '--da3-checkpoint', str(pretrained), '--num-workers', '0', '--print-freq', '0']
    train.main([*base_args, '--output-dir', str(full)])
    train.main([*base_args, '--output-dir', str(resumed), '--resume', str(full / 'checkpoint-epoch5.pth')])
    original = torch.load(full / 'checkpoint-epoch10.pth', weights_only=False)
    restored = torch.load(resumed / 'checkpoint-epoch10.pth', weights_only=False)
    assert original['epoch'] == restored['epoch'] == 9
    for key in original['model']:
        torch.testing.assert_close(original['model'][key], restored['model'][key], rtol=0, atol=0)
    assert original['optimizer']['param_groups'] == restored['optimizer']['param_groups']
    for param, values in original['optimizer']['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, restored['optimizer']['state'][param][key], rtol=0, atol=0)
    original_rng, restored_rng = original['rng_states'][0], restored['rng_states'][0]
    assert original_rng['python'] == restored_rng['python']
    np.testing.assert_array_equal(original_rng['numpy'][1], restored_rng['numpy'][1])
    torch.testing.assert_close(original_rng['torch'], restored_rng['torch'], rtol=0, atol=0)
    assert [json.loads(line)['epoch'] for line in (resumed / 'log.jsonl').read_text().splitlines()] == list(range(5, 10))


def test_smoke_performs_bounded_update_without_checkpoints(tmp_path, monkeypatch):
    train = _stub_cpu_training(monkeypatch)
    pretrained, output = tmp_path / 'pretrained', tmp_path / 'smoke'
    pretrained.mkdir()
    (pretrained / 'config.json').write_text('{"model_name": "da3-base"}')
    train.main(['--model', 'image', '--da3-checkpoint', str(pretrained), '--output-dir', str(output),
                '--num-workers', '0', '--smoke-steps', '1'], mode='smoke')
    assert not list(output.glob('*.pth'))
    assert not (output / 'training_config.json').exists()
    assert (output / 'smoke_config.json').exists()
    metrics = json.loads((output / 'smoke_metrics.json').read_text())
    assert metrics['smoke'] is True and metrics['epoch'] == 0
    assert metrics['gradient_norms']['backbone'] > 0
    assert metrics['gradient_norms']['dense_depth_head'] > 0


def test_eval_restores_full_native_grid_and_writes_float32_predictions(small_raw_root, tmp_path, monkeypatch):
    import contextlib
    from occany_depth_min.kitti_dc_full import train
    root, _ = small_raw_root
    sample = KITTIDepthCompletionDataset(root, 'val', require_full=False)[0]
    batch = collate_kitti_dc_full([sample])
    outputs = {'dense_depth': torch.full_like(batch['dense_depth'], 10.)}
    monkeypatch.setattr(train, 'forward_batch', lambda *args: outputs)
    monkeypatch.setattr(train, 'amp_context', lambda args: contextlib.nullcontext())
    args = SimpleNamespace(prediction_dir=tmp_path / 'predictions', print_freq=0, dataset='kitti_dc_full')
    metrics = train.evaluate(torch.nn.Identity(), [batch], args, torch.device('cpu'), 0, 1)
    assert metrics['split'] == 'val' and metrics['val_samples'] == 1
    assert metrics['pixel_micro']['all_valid']['valid_pixels'] == 128
    assert metrics['pixel_micro']['all_valid']['rmse'] == 0
    assert metrics['per_image_macro']['all_valid']['images'] == 1
    prediction = np.load(next(args.prediction_dir.rglob('*.npy')))
    assert prediction.shape == (8, 16) and prediction.dtype == np.float32
    outputs['dense_depth'][0, 0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='Nonfinite prediction'):
        train.evaluate(torch.nn.Identity(), [batch], args, torch.device('cpu'), 0, 1)


def test_existing_output_config_rejects_cross_protocol_before_overwrite(tmp_path):
    from occany_depth_min.kitti_dc_full.train import validate_existing_config
    config_path = tmp_path / 'training_config.json'
    original = json.dumps({'protocol': {'model': 'depth', 'fusion_mode': 'postfusion', 'voxel_encoder': 'vfe'}})
    config_path.write_text(original)
    for key, value in [('fusion_mode', 'prefusion'), ('voxel_encoder', 'patchdepthbin'), ('model', 'voxel')]:
        protocol = dict(model='depth', fusion_mode='postfusion', voxel_encoder='vfe')
        protocol[key] = value
        with pytest.raises(ValueError, match=key):
            validate_existing_config(tmp_path, protocol)
        assert config_path.read_text() == original


def test_resume_requires_per_rank_random_states():
    protocol = {'world_size': 4}
    with pytest.raises(ValueError, match='one state per rank'):
        validate_checkpoint(dict(model={}, epoch=4, optimizer={}, protocol=protocol, rng_states=[{}]), protocol, resume=True)


def test_summary_contains_postfusion_representation(tmp_path):
    run = tmp_path / 'single_frame/da3_base_depth/left_long1232_10ep_seed0'
    run.mkdir(parents=True)
    config = dict(model='depth', input_long_side=1232, fusion_mode='postfusion',
                  fusion_stage='after_da3', depth_token_source='sparse_log_patch')
    (run / 'training_config.json').write_text(json.dumps(dict(protocol=config)))
    metrics = dict(val_samples=3426, pixel_micro={'all_valid': {'rmse': 2.}}, per_image_macro={'all_valid': {'rmse': 2.1}})
    (run / 'eval_val_epoch4.json').write_text(json.dumps(dict(epoch=4, datasets={'kitti_dc_full': metrics})))
    summarize(tmp_path)
    with (tmp_path / 'seven_models_val_summary.csv').open() as source:
        row = next(csv.DictReader(source))
    assert row['fusion_mode'] == 'postfusion' and row['depth_token_source'] == 'sparse_log_patch'
    assert row['epoch'] == '5' and row['images'] == '3426'


def test_eval_val_main_enumerates_only_validation(tmp_path, monkeypatch):
    train = _stub_cpu_training(monkeypatch)
    pretrained, output = tmp_path / 'pretrained', tmp_path / 'eval'
    pretrained.mkdir()
    (pretrained / 'config.json').write_text('{"model_name": "da3-base"}')
    args = train.get_parser(mode='eval-val').parse_args(['--model', 'image'])
    model = train.build_model(args)
    checkpoint = tmp_path / 'source-epoch5.pth'
    torch.save(dict(model=model.state_dict(), epoch=4, protocol={'world_size': 1}), checkpoint)
    splits = []

    def dataset(root, split, **kwargs):
        splits.append(split)
        assert split == 'val'
        return [0]

    monkeypatch.setattr(train, 'KITTIDepthCompletionDataset', dataset)
    train.main(['--model', 'image', '--da3-checkpoint', str(pretrained), '--output-dir', str(output),
                '--checkpoint', str(checkpoint)], mode='eval-val')
    assert splits == ['val']
    assert (output / 'eval_config.json').exists()
    metrics = json.loads((output / 'eval_val_epoch4.json').read_text())
    assert metrics['epoch'] == 4 and metrics['checkpoint'] == str(checkpoint)
    assert not (output / 'training_config.json').exists() and not list(output.glob('*.pth'))
