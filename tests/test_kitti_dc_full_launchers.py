"""Exercise suite orchestration with executable mocks, without GPUs or data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SEVEN = ["image", "depth", "voxel", "voxel_depth", "depth_scaled", "voxel_scaled", "voxel_depth_scaled"]
VFE = ["voxel", "voxel_depth", "voxel_scaled", "voxel_depth_scaled"]
SUITES = [
    ("run_kitti_dc_full_seven_models.sh", "prefusion", "patchdepthbin", SEVEN),
    ("run_kitti_dc_full_vfe_four_models.sh", "prefusion", "vfe", VFE),
    ("run_kitti_dc_full_postfusion_seven_models.sh", "postfusion", "patchdepthbin", SEVEN),
    ("run_kitti_dc_full_postfusion_vfe_four_models.sh", "postfusion", "vfe", VFE),
]
SUITE_518 = "run_kitti_dc_full_seven_models_518x168_5ep.sh"


def executable(path, body):
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def runner(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_file = tmp_path / "calls.jsonl"
    runtime = executable(bin_dir / "runtime", '''import json, os, sys
if len(sys.argv) > 1 and sys.argv[1] == "-c":
    # Run the launcher's read-only output guard using the selected interpreter.
    code = sys.argv[2]
    sys.argv = ["-c", *sys.argv[3:]]
    exec(code)
    raise SystemExit(0)
args = sys.argv[1:]
with open(os.environ["CALLS_FILE"], "a") as handle:
    handle.write(json.dumps(["runtime", *args]) + "\\n")
print("mock runtime: " + " ".join(args))
if "--model" in args and args[args.index("--model") + 1] == os.environ.get("FAIL_MODEL"):
    raise SystemExit(7)
''')
    nvidia = executable(bin_dir / "nvidia-smi", '''import json, os, sys
counter_file = os.environ["GPU_COUNTER_FILE"]
try:
    with open(counter_file) as handle:
        count = int(handle.read())
except FileNotFoundError:
    count = 0
with open(counter_file, "w") as handle:
    handle.write(str(count + 1))
with open(os.environ["CALLS_FILE"], "a") as handle:
    handle.write(json.dumps(["gpu", *sys.argv[1:]]) + "\\n")
if os.environ.get("FAIL_GPU") == "1":
    raise SystemExit(9)
print(5120 if os.environ.get("GPU_BUSY_ONCE") == "1" and count == 0 else 5119)
''')
    executable(bin_dir / "sleep", '''import json, os, sys
with open(os.environ["CALLS_FILE"], "a") as handle:
    handle.write(json.dumps(["sleep", *sys.argv[1:]]) + "\\n")
''')
    env = os.environ.copy()
    for name in (
        "PYTHON", "TORCHRUN", "NPROC_PER_NODE", "KITTI_DC_ROOT", "DA3_CKPT", "OUTPUT_ROOT",
        "FUSION_MODE", "VOXEL_ENCODER", "MODELS", "CKPT_EPOCHS", "NUM_WORKERS", "PRINT_FREQ",
        "SMOKE_STEPS", "DRY_RUN", "CUDA_VISIBLE_DEVICES", "KITTI_DC_FULL_SUITE_FUSION",
        "KITTI_DC_FULL_SUITE_ENCODER", "GPU_MEMORY_LIMIT_MIB", "GPU_POLL_SECONDS", "NVIDIA_SMI",
        "INPUT_LONG_SIDE", "EPOCHS", "KITTI_DC_FULL_SUITE_INPUT_LONG_SIDE", "KITTI_DC_FULL_SUITE_EPOCHS",
    ):
        env.pop(name, None)
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        PYTHON=runtime,
        TORCHRUN=runtime,
        CALLS_FILE=str(calls_file),
        GPU_COUNTER_FILE=str(tmp_path / "gpu_count"),
        NVIDIA_SMI=nvidia,
        GPU_POLL_SECONDS="1",
    )

    class Runner:
        output = tmp_path / "suite with space"

        def __call__(self, script="run_kitti_dc_full.sh", *args, paths=True, output=True, env_updates=None):
            command = ["bash", str(SCRIPTS / script), *map(str, args)]
            if paths:
                command += ["--kitti-dc-root", "/external/full data", "--da3-checkpoint", "/external/base weights"]
            if output:
                command += ["--output-root", str(self.output)]
            actual_env = env.copy()
            actual_env.update(env_updates or {})
            return subprocess.run(command, env=actual_env, text=True, capture_output=True, timeout=15)

        def calls(self, kind="runtime"):
            if not calls_file.exists():
                return []
            return [line[1:] for line in map(json.loads, calls_file.read_text().splitlines()) if line[0] == kind]

        def dry_commands(self, result):
            return [shlex.split(line)[1:] for line in result.stdout.splitlines() if line.startswith("[dry-run]") and " wait until " not in line]

    return Runner()


def value(args, flag):
    return args[args.index(flag) + 1]


@pytest.mark.parametrize("script,fusion,encoder,models", SUITES)
def test_original_suites_route_all_22_variants(runner, script, fusion, encoder, models):
    result = runner(script, "train")
    assert result.returncode == 0, result.stderr
    calls = runner.calls()
    assert len(calls) == len(models) + 1
    for args, model in zip(calls, models):
        assert args[:5] == ["--standalone", "--nproc_per_node=4", "-m", "occany_depth_min.kitti_dc_full", "train"]
        assert value(args, "--model") == model
        assert value(args, "--fusion-mode") == fusion
        assert value(args, "--voxel-encoder") == encoder
        assert value(args, "--kitti-dc-root") == "/external/full data"
        assert value(args, "--da3-checkpoint") == "/external/base weights"
        assert value(args, "--input-long-side") == "1232"
        assert value(args, "--epochs") == "10"
        suffix = "_vfe" if encoder == "vfe" else ""
        expected = runner.output / "single_frame" / f"da3_base_{model}{suffix}" / "left_long1232_10ep_seed0"
        assert value(args, "--output-dir") == str(expected)
    assert calls[-1] == ["-m", "occany_depth_min.kitti_dc_full", "summarize", "--output-root", str(runner.output)]
    assert (runner.output / "suite_logs" / "summarize.log").is_file()


def test_eval_default_and_subset_epochs(runner):
    result = runner("run_kitti_dc_full.sh", "eval-val", "--models", "depth voxel")
    assert result.returncode == 0, result.stderr
    calls = runner.calls()
    assert [value(args, "--model") for args in calls[:-1]] == ["depth", "depth", "voxel", "voxel"]
    assert [Path(value(args, "--checkpoint")).name for args in calls[:-1]] == [
        "checkpoint-epoch5.pth", "checkpoint-epoch10.pth", "checkpoint-epoch5.pth", "checkpoint-epoch10.pth"
    ]
    result = runner("run_kitti_dc_full.sh", "eval-val", "--models", "depth", "--checkpoint-epochs", "5", "--dry-run")
    commands = runner.dry_commands(result)
    assert result.returncode == 0, result.stderr
    assert len(commands) == 2
    assert Path(value(commands[0], "--checkpoint")).name == "checkpoint-epoch5.pth"


@pytest.mark.parametrize("env_preview", [False, True])
def test_dry_run_writes_nothing_and_bypasses_gpu_query(runner, env_preview):
    flags = [] if env_preview else ["--dry-run"]
    result = runner("run_kitti_dc_full_postfusion_vfe_four_models.sh", "train", *flags,
                    env_updates={"DRY_RUN": "1"} if env_preview else None)
    assert result.returncode == 0, result.stderr
    assert "wait until GPUs 0,1,2,3 each use less than 5120 MiB" in result.stdout
    assert len(runner.dry_commands(result)) == 5
    assert not runner.output.exists()
    assert not runner.calls()
    assert not runner.calls("gpu")


@pytest.mark.parametrize("script,run_name", [
    ("run_kitti_dc_full.sh", "left_long1232_10ep_seed0"),
    (SUITE_518, "left_long518_5ep_seed0"),
])
def test_auto_resume_is_per_model(runner, script, run_name):
    checkpoint = runner.output / "single_frame/da3_base_voxel" / run_name / "checkpoint-last.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    result = runner(script, "train", "--models", "voxel depth")
    assert result.returncode == 0, result.stderr
    calls = runner.calls()
    assert value(calls[0], "--resume") == str(checkpoint)
    assert "--resume" not in calls[1]


def test_runtime_failure_stops_before_next_model_or_summary(runner):
    result = runner("run_kitti_dc_full.sh", "train", "--models", "depth voxel", env_updates={"FAIL_MODEL": "depth"})
    assert result.returncode == 7
    calls = runner.calls()
    assert len(calls) == 1
    assert value(calls[0], "--model") == "depth"
    assert "mock runtime:" in (runner.output / "suite_logs/train_depth.log").read_text()


def test_single_card_smoke_is_bounded_and_skips_summary(runner):
    result = runner("run_kitti_dc_full.sh", "smoke", "--models", "voxel_scaled", "--nproc-per-node", "1", "--smoke-steps", "3")
    assert result.returncode == 0, result.stderr
    args, = runner.calls()
    assert "--nproc_per_node=1" in args
    assert value(args, "--smoke-steps") == "3"
    assert value(args, "--print-freq") == "1"
    assert value(args, "--output-dir") == str(runner.output / "validation/voxel_scaled")


def test_postfusion_vfe_waits_at_exact_threshold_then_starts(runner):
    result = runner("run_kitti_dc_full_postfusion_vfe_four_models.sh", "train", "--models", "voxel",
                    env_updates={"GPU_BUSY_ONCE": "1"})
    assert result.returncode == 0, result.stderr
    assert len(runner.calls("gpu")) == 8
    assert runner.calls("sleep") == [["1"]]
    assert len(runner.calls()) == 2
    assert "threshold=<5120 MiB" in result.stdout


def test_failed_gpu_query_prevents_launch_and_output_writes(runner):
    result = runner("run_kitti_dc_full_postfusion_vfe_four_models.sh", "train", env_updates={"FAIL_GPU": "1"})
    assert result.returncode != 0
    assert "Failed to query GPU 0" in result.stderr
    assert not runner.calls()
    assert not runner.output.exists()


def test_summary_needs_only_output_root(runner):
    result = runner("run_kitti_dc_full.sh", "summarize", paths=False)
    assert result.returncode == 0, result.stderr
    assert runner.calls() == [["-m", "occany_depth_min.kitti_dc_full", "summarize", "--output-root", str(runner.output)]]
    assert not runner.calls("gpu")


@pytest.mark.parametrize("flags,error", [
    (["train", "--nproc-per-node", "1"], "Train/eval require 4 GPUs"),
    (["eval-val", "--checkpoint-epochs", "15"], "Checkpoint epochs"),
    (["eval-val", "--epochs", "5", "--checkpoint-epochs", "10"], "Checkpoint epochs"),
    (["eval-val", "--checkpoint-epochs", "0"], "Checkpoint epochs"),
    (["eval-val", "--checkpoint-epochs", "-5"], "Checkpoint epochs"),
    (["eval-val", "--checkpoint-epochs", "4"], "Checkpoint epochs"),
    (["eval-val", "--epochs", "3"], "No scheduled checkpoint epochs"),
    (["train", "--input-long-side", "0"], "positive integer"),
    (["train", "--input-long-side", "-518"], "positive integer"),
    (["train", "--input-long-side", "518.5"], "positive integer"),
    (["train", "--epochs", "0"], "positive integer"),
    (["train", "--epochs", "-5"], "positive integer"),
    (["train", "--epochs", "5.5"], "positive integer"),
    (["train", "--voxel-encoder", "vfe", "--models", "image"], "VFE requires a voxel model"),
    (["train", "--models", "voxel voxel"], "Duplicate model"),
    (["smoke", "--smoke-steps", "0"], "positive integer"),
])
def test_invalid_suite_requests_stop_without_outputs(runner, flags, error):
    result = runner("run_kitti_dc_full.sh", *flags)
    assert result.returncode == 2
    assert error in result.stderr
    assert not runner.calls()
    assert not runner.output.exists()


@pytest.mark.parametrize("script,fusion,encoder,models", SUITES + [(SUITE_518, "prefusion", "patchdepthbin", SEVEN)])
def test_wrapper_rejects_conflicting_suite_flags(runner, script, fusion, encoder, models):
    other_fusion = "postfusion" if fusion == "prefusion" else "prefusion"
    result = runner(script, "train", "--fusion-mode", other_fusion)
    assert result.returncode == 2
    assert "This suite requires fusion mode" in result.stderr
    other_encoder = "vfe" if encoder == "patchdepthbin" else "patchdepthbin"
    result = runner(script, "train", "--voxel-encoder", other_encoder)
    assert result.returncode == 2
    assert "This suite requires voxel encoder" in result.stderr
    assert not runner.output.exists()


@pytest.mark.parametrize("stage,fusion,forbidden", [
    ("train", "prefusion", "kitti"),
    ("smoke", "prefusion", "kitti"),
    ("train", "postfusion", "kitti_dc_full"),
    ("smoke", "postfusion", "kitti_dc_full"),
])
def test_output_guard_resolves_symlinks_before_log_writes(runner, tmp_path, stage, fusion, forbidden):
    # A symlink can point into a forbidden directory without creating it.
    link = tmp_path / "unsafe_output"
    target = ROOT / "output/depth" / forbidden / "launcher_test"
    link.symlink_to(target, target_is_directory=True)
    runner.output = link
    result = runner("run_kitti_dc_full.sh", stage, "--fusion-mode", fusion, "--dry-run")
    assert result.returncode != 0
    assert "outputs must be outside" in result.stderr
    assert not target.exists()
    assert not runner.calls()


def test_default_commands_use_path_executables_and_explicit_roots(runner):
    result = runner("run_kitti_dc_full.sh", "train", "--models", "image", "--dry-run",
                    env_updates={"PYTHON": "python", "TORCHRUN": "torchrun"})
    assert result.returncode == 0, result.stderr
    commands = runner.dry_commands(result)
    assert commands[0][0] == "torchrun"
    assert commands[1][0] == "python"
    assert "/home/dataset-local" not in result.stdout


def test_help_and_shell_syntax(runner):
    result = runner("run_kitti_dc_full.sh", "--help", paths=False)
    assert result.returncode == 0
    assert "--output-root" in result.stdout
    assert "--input-long-side" in result.stdout and "--epochs" in result.stdout
    for script in ["run_kitti_dc_full.sh", SUITE_518, *(suite[0] for suite in SUITES)]:
        assert subprocess.run(["bash", "-n", str(SCRIPTS / script)], capture_output=True).returncode == 0


@pytest.mark.parametrize("stage", ["train", "eval-val", "smoke"])
def test_518_suite_routes_seven_models_and_epoch_five(runner, stage):
    flags = ["--nproc-per-node", "1"] if stage == "smoke" else []
    result = runner(SUITE_518, stage, *flags)
    assert result.returncode == 0, result.stderr
    calls = runner.calls()
    assert len(calls) == len(SEVEN) + (stage != "smoke")
    for args, model in zip(calls, SEVEN):
        assert args[2:5] == ["-m", "occany_depth_min.kitti_dc_full", stage]
        assert value(args, "--model") == model
        assert value(args, "--fusion-mode") == "prefusion"
        assert value(args, "--voxel-encoder") == "patchdepthbin"
        assert value(args, "--input-long-side") == "518"
        assert value(args, "--epochs") == "5"
        expected = runner.output / "single_frame" / f"da3_base_{model}" / "left_long518_5ep_seed0"
        if stage == "smoke":
            expected = runner.output / "validation" / model
            assert value(args, "--smoke-steps") == "2"
        elif stage == "eval-val":
            assert value(args, "--checkpoint") == str(expected / "checkpoint-epoch5.pth")
        assert value(args, "--output-dir") == str(expected)
    if stage != "smoke":
        assert calls[-1] == ["-m", "occany_depth_min.kitti_dc_full", "summarize", "--output-root", str(runner.output)]


def test_518_suite_defaults_override_conflicting_environment_without_writes(runner):
    output = ROOT / "output/depth/kitti_dc_full_518x168_5ep"
    existed = output.exists()
    result = runner(SUITE_518, "eval-val", "--dry-run", output=False, env_updates={
        "INPUT_LONG_SIDE": "1232", "EPOCHS": "10", "CKPT_EPOCHS": "5 10",
        "FUSION_MODE": "postfusion", "VOXEL_ENCODER": "vfe", "MODELS": "voxel",
    })
    assert result.returncode == 0, result.stderr
    commands = runner.dry_commands(result)
    assert [value(args, "--model") for args in commands[:-1]] == SEVEN
    for args, model in zip(commands, SEVEN):
        expected = output / "single_frame" / f"da3_base_{model}" / "left_long518_5ep_seed0"
        assert value(args, "--output-dir") == str(expected)
        assert value(args, "--checkpoint") == str(expected / "checkpoint-epoch5.pth")
    assert output.exists() == existed
    assert not runner.calls() and not runner.calls("gpu")


@pytest.mark.parametrize("flag,value_,error", [
    ("--input-long-side", "1232", "requires input long side 518"),
    ("--epochs", "10", "requires epochs 5"),
    ("--checkpoint-epochs", "10", "Checkpoint epochs"),
])
def test_518_suite_rejects_conflicting_schedule(runner, flag, value_, error):
    result = runner(SUITE_518, "eval-val", flag, value_)
    assert result.returncode == 2
    assert error in result.stderr
    assert not runner.calls() and not runner.output.exists()


@pytest.mark.parametrize("use_environment", [False, True])
def test_custom_schedule_derives_all_checkpoint_epochs(runner, use_environment):
    flags = [] if use_environment else ["--input-long-side", "518", "--epochs", "15"]
    environment = {"INPUT_LONG_SIDE": "518", "EPOCHS": "15"} if use_environment else {}
    result = runner("run_kitti_dc_full.sh", "eval-val", "--models", "depth", *flags,
                    "--dry-run", env_updates=environment)
    assert result.returncode == 0, result.stderr
    commands = runner.dry_commands(result)
    assert [Path(value(args, "--checkpoint")).name for args in commands[:-1]] == [
        "checkpoint-epoch5.pth", "checkpoint-epoch10.pth", "checkpoint-epoch15.pth",
    ]
    for args in commands[:-1]:
        assert value(args, "--input-long-side") == "518"
        assert value(args, "--epochs") == "15"
        assert Path(value(args, "--output-dir")).name == "left_long518_15ep_seed0"
    assert not runner.output.exists() and not runner.calls()


def test_518_summary_needs_only_output_root(runner):
    result = runner(SUITE_518, "summarize", paths=False)
    assert result.returncode == 0, result.stderr
    assert runner.calls() == [["-m", "occany_depth_min.kitti_dc_full", "summarize", "--output-root", str(runner.output)]]
