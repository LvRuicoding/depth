# OccAny-Depth-Min

这个仓库保留两个 DA3-small 模型的 Unified6 与 KITTI-only 训练/评测代码，并提供八个 DA3-Base 单帧 KITTI-only 模型、22 个 KITTI full 模型配置，以及 518×168、5 epoch 的七模型前融合套件：

`Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNNPromptDAScaledUnified6Model`

对应原实验目录：

`output/depth/unified6/single_frame/da3_small_lingbot_tokenfusion_last_patchdepth4m_voxel_prefusion_promptda_scaled`

`Stage1DepthPatchDepth4mVoxelDepthDualWindowPostFusionOnlyOnlineKNNPromptDAScaledPreAlignedUnified6Model`

对应原实验目录：

`output/depth/unified6/single_frame/da3_small_patchdepth4m_voxeldepth_dualwindow_postfusion_only_promptda_scaled_prefusion_aligned_depthlogpatch_voxelpatch4m`

| `--model` | DA3 输入 token | 深度输出 |
| --- | --- | --- |
| `image` | RGB | 直接 metric |
| `depth` | RGB + sparse-depth | 直接 metric |
| `voxel` | RGB + voxel | 直接 metric |
| `voxel_depth` | RGB + sparse-depth + voxel | 直接 metric |
| `image_scaled` | RGB | PromptDA + 在线 KNN-4 min/max scale |
| `depth_scaled` | RGB + sparse-depth | PromptDA + 在线 KNN-4 min/max scale |
| `voxel_scaled` | RGB + voxel | PromptDA + 在线 KNN-4 min/max scale |
| `voxel_depth_scaled` | RGB + sparse-depth + voxel | PromptDA + 在线 KNN-4 min/max scale |

这八个 DA3-Base 变体仅用于 KITTI。`image_scaled` 的 DA3 token 序列保持纯 RGB；LiDAR 生成的在线 KNN 深度只传给 PromptDA 头，并用于恢复每帧输出的 min/max metric scale。

仓库不包含数据、Unified6 v3 manifests、DA3 基础权重、训练 checkpoint 或输出。这些路径通过参数或环境变量指定。正式训练和评测固定使用 4 张 GPU；KITTI full 的有界 smoke 也支持单卡。模型结构、优化器、学习率和评测协议固定；KITTI full 可通过 `--input-long-side`、`--epochs` 选择输入长边和训练周期，其他入口的输入尺寸与训练轮数仍固定。Unified6 训练提供原实验的域均衡模式，以及不做域均衡的自然比例模式。

Unified6 固定训练 10 个 epoch；KITTI-only 复现原实验的 3834/815 train/val 划分、518×168 principal-point crop、KITTI 体素网格、在线 LiDAR KNN-4 和 20 个 epoch。KITTI-only 不读取 Unified manifest。

## 模型一：DA3 前融合训练

```bash
scripts/run_unified6.sh train \
  --da3-checkpoint /home/dataset-local/lr/code/OccAny/checkpoints/da3_small \
  --manifest-dir /home/dataset-local/lr/code/OccAny/data/unified_depth_splits_v3 \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --ddad-root /home/dataset-local/lr/code/OccAny/data/ddad_processed \
  --seven-scenes-root /home/dataset-local/lr/code/OccAny/raw_data/7-Scenes/OpenDataLab___7-Scenes/raw \
  --nyuv2-root /home/dataset-local/lr/code/OccAny/raw_data/NYUv2 \
  --sunrgbd-root /home/dataset-local/lr/code/OccAny/raw_data/SUN_RGB-D/OpenDataLab___SUN_RGB-D/raw/SUNRGBD \
  --void-root /home/dataset-local/lr/code/OccAny/raw_data/VOID \
  --output-dir /path/to/output
```

续训只需追加 `--resume /path/to/checkpoint-last.pth`。

固定训练协议：4 GPU、每卡 batch size 1、bf16、AdamW、`lr=1e-4`、`weight_decay=1e-4`、1 epoch warmup、cosine 到 `1e-6`、10 epochs、每域每 epoch 4826 个样本、dense-depth loss 权重 0.1。

## 不使用域均衡的多数据集训练

把上述训练命令的入口改为 `train-natural`，其余参数不变：

```bash
scripts/run_unified6.sh train-natural \
  --da3-checkpoint /home/dataset-local/lr/code/OccAny/checkpoints/da3_small \
  --manifest-dir /home/dataset-local/lr/code/OccAny/data/unified_depth_splits_v3 \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --ddad-root /home/dataset-local/lr/code/OccAny/data/ddad_processed \
  --seven-scenes-root /home/dataset-local/lr/code/OccAny/raw_data/7-Scenes/OpenDataLab___7-Scenes/raw \
  --nyuv2-root /home/dataset-local/lr/code/OccAny/raw_data/NYUv2 \
  --sunrgbd-root /home/dataset-local/lr/code/OccAny/raw_data/SUN_RGB-D/OpenDataLab___SUN_RGB-D/raw/SUNRGBD \
  --void-root /home/dataset-local/lr/code/OccAny/raw_data/VOID \
  --output-dir /path/to/natural-output
```

该模式直接打乱六个训练集拼接后的全局索引，不为数据域设置配额，也不对小域做过采样。每个 epoch 全量遍历 96,648 个样本：KITTI 3,659、DDAD 12,650、7-Scenes 26,000、NYUv2 795、SUN RGB-D 5,285、VOID 48,259；4 卡时每卡 24,162 个样本。其余训练超参数与上面的域均衡模式一致。

自然比例训练的 checkpoint 会记录独立的采样协议；续训仍追加 `--resume`，但不能在域均衡与自然比例 checkpoint 之间交叉续训。

## 模型二：双窗口后融合训练

第二个模型复用相同的路径参数。域均衡训练使用：

```bash
scripts/run_unified6.sh train-postfusion \
  --da3-checkpoint /home/dataset-local/lr/code/OccAny/checkpoints/da3_small \
  --manifest-dir /home/dataset-local/lr/code/OccAny/data/unified_depth_splits_v3 \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --ddad-root /home/dataset-local/lr/code/OccAny/data/ddad_processed \
  --seven-scenes-root /home/dataset-local/lr/code/OccAny/raw_data/7-Scenes/OpenDataLab___7-Scenes/raw \
  --nyuv2-root /home/dataset-local/lr/code/OccAny/raw_data/NYUv2 \
  --sunrgbd-root /home/dataset-local/lr/code/OccAny/raw_data/SUN_RGB-D/OpenDataLab___SUN_RGB-D/raw/SUNRGBD \
  --void-root /home/dataset-local/lr/code/OccAny/raw_data/VOID \
  --output-dir /path/to/postfusion-output
```

不使用域均衡时，将入口改为 `train-postfusion-natural`。它同样每个 epoch 自然比例遍历全部 96,648 个训练样本。两种模式都支持追加 `--resume`，且会校验 checkpoint 的模型与采样协议。

该模型先用标准 DA3 生成 384 维 local/global 分支，再分别以共享参数的 regular/shifted window attention 依次融合 patch×4m voxel token 和稀疏 log-depth patch token，最后拼接为 768 维输入 PromptDA DPT 头。

## 评测

```bash
scripts/run_unified6.sh eval \
  --checkpoint /path/to/checkpoint-best.pth \
  --manifest-dir /home/dataset-local/lr/code/OccAny/data/unified_depth_splits_v3 \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --ddad-root /home/dataset-local/lr/code/OccAny/data/ddad_processed \
  --seven-scenes-root /home/dataset-local/lr/code/OccAny/raw_data/7-Scenes/OpenDataLab___7-Scenes/raw \
  --nyuv2-root /home/dataset-local/lr/code/OccAny/raw_data/NYUv2 \
  --sunrgbd-root /home/dataset-local/lr/code/OccAny/raw_data/SUN_RGB-D/OpenDataLab___SUN_RGB-D/raw/SUNRGBD \
  --void-root /home/dataset-local/lr/code/OccAny/raw_data/VOID \
  --output-json /path/to/eval_unified_best_val.json
```

默认评测 KITTI、DDAD、7-Scenes、NYUv2、SUN RGB-D、VOID 全部官方 val manifest，并输出综合 JSON 与六个分域 JSON。调试时可用 `--domains kitti` 只跑一个域。

评测第二个模型时使用同一组参数，将入口改为：

```bash
scripts/run_unified6.sh eval-postfusion \
  --checkpoint /path/to/postfusion-checkpoint-best.pth \
  --manifest-dir /home/dataset-local/lr/code/OccAny/data/unified_depth_splits_v3 \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --ddad-root /home/dataset-local/lr/code/OccAny/data/ddad_processed \
  --seven-scenes-root /home/dataset-local/lr/code/OccAny/raw_data/7-Scenes/OpenDataLab___7-Scenes/raw \
  --nyuv2-root /home/dataset-local/lr/code/OccAny/raw_data/NYUv2 \
  --sunrgbd-root /home/dataset-local/lr/code/OccAny/raw_data/SUN_RGB-D/OpenDataLab___SUN_RGB-D/raw/SUNRGBD \
  --void-root /home/dataset-local/lr/code/OccAny/raw_data/VOID \
  --output-json /path/to/postfusion-eval.json
```

## KITTI-only 训练与评测

前融合模型：

```bash
scripts/run_unified6.sh train-kitti \
  --da3-checkpoint /home/dataset-local/lr/code/OccAny/checkpoints/da3_small \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --output-dir /path/to/kitti-output

scripts/run_unified6.sh eval-kitti \
  --checkpoint /path/to/kitti-output/checkpoint-last.pth \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --output-json /path/to/kitti-eval.json
```

双窗口后融合模型将入口分别改为 `train-postfusion-kitti` 和 `eval-postfusion-kitti`，参数不变。两种训练都支持 `--resume /path/to/checkpoint-last.pth`；也可以直接评测原仓库相应的 KITTI-only checkpoint。

八个 DA3-Base 模型使用同一个入口，通过 `--model` 选择。基础权重目录必须包含 `model.safetensors`：

```bash
scripts/run_unified6.sh train-kitti \
  --model voxel_depth_scaled \
  --da3-checkpoint /home/dataset-local/lr/code/OccAny/checkpoints/DA3-BASE \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --output-dir /path/to/voxel-depth-scaled

scripts/run_unified6.sh eval-kitti \
  --model voxel_depth_scaled \
  --checkpoint /path/to/voxel-depth-scaled/checkpoint-last.pth \
  --kitti-root /home/dataset-local/lr/code/OccAny/data/kitti_processed \
  --output-json /path/to/voxel-depth-scaled/eval.json
```

把示例中的模型名替换为表中任意一个即可。续训追加 `--resume`；代码会同时校验实验名、模型变体、DA3 尺寸、token 维度和初始化合同。评测加载使用严格 state-dict 校验，因此把某个 checkpoint 配给错误变体会立即报错。

KITTI-only 固定使用 4 GPU、每卡 batch size 1、bf16、AdamW、`lr=1e-4`、1 epoch warmup、cosine 到 `1e-6`、20 epochs，并在每个 epoch 后只评测 KITTI val。缺少 `dense_depthmap` 的训练帧仍保留在采样序列中，但由 frame mask 跳过 dense-depth loss；因此训练集是原协议的 3834 帧，而不是 Unified6 manifest 中的 3659 帧。

## KITTI full：官方 Depth Completion 全量实验

`occany_depth_min.kitti_dc_full` 独立于上面的 KITTI-only 与 Unified6 入口。它迁移原仓库 `output/depth/kitti_dc_full`、`output/depth/kitti_dc_full_postfusion` 和 `output/depth/kitti_dc_full_518x168_5ep` 对应的代码、训练协议和 checkpoint 合同；数据和 DA3 权重通过外部路径读取，实验输出根目录可以指定，无需等待原实验跑完。

前融合与后融合各提供七个基础模型：`image depth voxel voxel_depth depth_scaled voxel_scaled voxel_depth_scaled`；各提供四个 RAW xyz/intensity VFE 模型：`voxel voxel_depth voxel_scaled voxel_depth_scaled`，共 22 个配置。这里没有 `image_scaled`。`--fusion-mode` 选择 `prefusion`/`postfusion`，`--voxel-encoder` 选择 `patchdepthbin`/`vfe`；VFE 只支持四个含 voxel 的变体。后融合在 DA3 local/global 分支之后依次融合 voxel、sparse-depth，`image` 始终保持纯 RGB。

固定协议为官方左相机 train/val 的 42,949/3,426 帧、完整视场、不放大原图、半像素内参更新、对称补边到 patch 14、共享 FP32 RAW LiDAR 投影；默认长边 1232。scaled 模型在有效图像区域计算在线 KNN-4。正式训练使用 4 GPU、每卡 batch 1、bf16、seed 0、AdamW（`lr=1e-4`、`weight_decay=1e-4`、`betas=(0.9,0.95)`）、1 epoch warmup 后 cosine 到 `1e-6`，dense loss 权重 0.1；默认训练 10 epochs。每五轮先保存 checkpoint，再在原始图像网格评测 val，记录 pixel-micro 与 per-image-macro 两套指标，范围为 `1e-3 <= depth <= 80m`。不提供测试集评测入口。

数据根目录保留原目录结构：`train/`、`val/` 下的官方 `*/proj_depth/groundtruth/image_02/*.png`，以及各日期目录下的标定文件、`*_sync/image_02/data/*.png` 和 `*_sync/velodyne_points/data/*.bin`。DA3-BASE 目录包含 `model.safetensors`。批量入口要求显式传路径，或设置以下环境变量：

```bash
export KITTI_DC_ROOT=/path/to/OccAny/raw_data/kitti_full
export DA3_CKPT=/path/to/OccAny/checkpoints/DA3-BASE
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

四套批量训练入口保留原脚本名，使用两个独立输出根目录：

```bash
scripts/run_kitti_dc_full_seven_models.sh train \
  --output-root /path/to/output/depth/kitti_dc_full
scripts/run_kitti_dc_full_vfe_four_models.sh train \
  --output-root /path/to/output/depth/kitti_dc_full
scripts/run_kitti_dc_full_postfusion_seven_models.sh train \
  --output-root /path/to/output/depth/kitti_dc_full_postfusion
scripts/run_kitti_dc_full_postfusion_vfe_four_models.sh train \
  --output-root /path/to/output/depth/kitti_dc_full_postfusion
```

也可以逐条传入 `--kitti-dc-root` 和 `--da3-checkpoint`。公共入口 `scripts/run_kitti_dc_full.sh` 接受同一组参数，加上 `--fusion-mode`、`--voxel-encoder`、`--input-long-side` 和 `--epochs`；后两个参数也可通过 `INPUT_LONG_SIDE`、`EPOCHS` 设置，命令行参数优先。包装入口固定各自融合方式和编码器，冲突的显式参数会报错。`--models "voxel voxel_depth_scaled"` 选择有序子集；模型顺序执行、日志追加到 `suite_logs/`，任一命令失败立即停止。若当前运行目录已有 `checkpoint-last.pth`，训练自动续训。运行目录为 `single_frame/da3_base_<model>[_vfe]/left_long<输入长边>_<训练轮数>ep_seed0`，默认保持原来的 `left_long1232_10ep_seed0`；前后融合应使用不同输出根，trainer 拒绝覆盖或加载不匹配的协议。

将阶段改为 `eval-val` 默认评测训练周期内每五轮的 checkpoint，10 epochs 对应第五、十轮，5 epochs 只评第五轮。`--checkpoint-epochs "5"` 也可显式选择第五轮；指定轮数必须为正整数、五的倍数且不超过 `--epochs`。评测时的 `--input-long-side` 和 `--epochs` 必须匹配 checkpoint 的原训练协议。沿用 `checkpoint-epoch5.pth`/`checkpoint-epoch10.pth` 与从零开始的 `eval_val_epoch4.json`/`eval_val_epoch9.json`。train/eval 批量完成后自动汇总，或独立执行：

```bash
scripts/run_kitti_dc_full.sh summarize \
  --output-root /path/to/output/depth/kitti_dc_full
```

基础模型和 VFE 分别写入 `seven_models_val_summary.csv` 与 `vfe_four_models_val_summary.csv`；尚未生成 val JSON 的实验不会阻止汇总。`--dry-run`（或 `DRY_RUN=1`）预览所有命令，不创建输出、不查询 GPU。`--python`、`--torchrun` 默认从 PATH 查找，也可指定环境中的可执行文件；`--num-workers` 与 `--print-freq` 可调整。后融合 VFE 套件执行前等待每张选中 GPU 的显存使用量严格小于 5120 MiB；可通过 `NVIDIA_SMI`、`GPU_MEMORY_LIMIT_MIB`、`GPU_POLL_SECONDS` 配置查询命令、阈值和轮询间隔。

单卡 smoke 使用有界训练步，不保存训练 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_kitti_dc_full.sh smoke \
  --fusion-mode postfusion --voxel-encoder vfe --models "voxel_depth_scaled" \
  --kitti-dc-root "$KITTI_DC_ROOT" --da3-checkpoint "$DA3_CKPT" \
  --output-root /path/to/smoke --nproc-per-node 1 --smoke-steps 2
```

单个原仓库 checkpoint 的评测与续训直接使用 Python 入口；这些命令仍要求 4 GPU，并严格校验模型、融合、编码器、数据和训练协议，续训同时恢复 optimizer、epoch 与各 rank 随机状态：

```bash
torchrun --standalone --nproc_per_node=4 -m occany_depth_min.kitti_dc_full eval-val \
  --model voxel_depth_scaled --fusion-mode postfusion --voxel-encoder vfe \
  --kitti-dc-root "$KITTI_DC_ROOT" --da3-checkpoint "$DA3_CKPT" \
  --checkpoint /path/to/OccAny/output/depth/kitti_dc_full_postfusion/single_frame/da3_base_voxel_depth_scaled_vfe/left_long1232_10ep_seed0/checkpoint-epoch5.pth \
  --output-dir /path/to/evaluation --prediction-dir /path/to/predictions

torchrun --standalone --nproc_per_node=4 -m occany_depth_min.kitti_dc_full train \
  --model voxel_depth_scaled --fusion-mode postfusion --voxel-encoder vfe \
  --kitti-dc-root "$KITTI_DC_ROOT" --da3-checkpoint "$DA3_CKPT" \
  --resume /path/to/OccAny/output/depth/kitti_dc_full_postfusion/single_frame/da3_base_voxel_depth_scaled_vfe/left_long1232_10ep_seed0/checkpoint-last.pth \
  --output-dir /path/to/new-training-output
```

`--prediction-dir` 仅用于 `eval-val`，按原图大小输出 float32 `.npy`。迁移后的生产代码不导入原 OccAny 仓库；已有 checkpoint 可直接严格加载，未完成实验的最后一个 checkpoint 可继续训练。

默认测试包含 22 个配置的源实现结构、初始化与构造结束随机状态摘要，运行 `python -m pytest -q` 即可，不需要数据、权重或原仓库。可选的外部源实现审计会在 CPU 上使用小型固定输入，比较全部初始化张量、参数顺序、输出、损失和梯度，并严格加载已有各变体 checkpoint：

```bash
python tests/verify_kitti_dc_full_reference.py \
  --reference-root /path/to/OccAny \
  --da3-checkpoint /path/to/OccAny/checkpoints/DA3-BASE \
  --checkpoint-root /path/to/OccAny/output/depth \
  --report-json /tmp/kitti-full-fidelity.json
```

迁移验证中，全部 22 个配置的 FP32 输出与源实现差异为零，初始化、损失、梯度与构造随机状态对照通过，13 个已有变体 checkpoint 严格加载通过；三个真实 val 样本的数据及 batch 逐字段一致。正式四卡 CUDA/bf16 的有界 smoke 需要在 GPU 可用的环境中运行，CPU 对照不替代这项硬件验证。

### 518×168、5 epoch 七模型套件

`scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh` 对应原实验目录 `output/depth/kitti_dc_full_518x168_5ep`。它固定单帧左相机、prefusion、patchdepthbin、长边 518 和 5 epochs，默认按 `image depth voxel voxel_depth depth_scaled voxel_scaled voxel_depth_scaled` 顺序执行。保留完整视场，缩放后补边到 518×168；仅 DA3-BASE backbone 从预训练权重初始化，其他模块保持原随机初始化，全部参数可训练。

数据与权重仍使用外部路径。以下命令使用当前机器的已有环境；也可激活自己的环境并从 PATH 使用 `python`、`torchrun`：

```bash
export PYTHON=/home/dataset-local/envs/occany/bin/python
export TORCHRUN=/home/dataset-local/envs/occany/bin/torchrun
export KITTI_DC_ROOT=/home/dataset-local/lr/data/kitti_full
export DA3_CKPT=/home/dataset-local/lr/code/OccAny/checkpoints/DA3-BASE
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 预览命令；不创建实验输出
bash scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh train --dry-run
# 训练或自动恢复当前模型的 checkpoint-last.pth
bash scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh train
# 只评测第五轮 checkpoint，并汇总七个模型
bash scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh eval-val
# 无需数据、权重或 GPU，汇总已有 val JSON
bash scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh summarize
# 单卡、两步 smoke：可选择部分模型，不保存训练 checkpoint
CUDA_VISIBLE_DEVICES=0 bash scripts/run_kitti_dc_full_seven_models_518x168_5ep.sh smoke \
  --models "voxel_depth_scaled" --nproc-per-node 1 --smoke-steps 2
```

默认输出根为本仓库的 `output/depth/kitti_dc_full_518x168_5ep`，可通过 `OUTPUT_ROOT` 或 `--output-root` 覆盖。正式运行目录为 `single_frame/da3_base_<model>/left_long518_5ep_seed0`，smoke 写入输出根的 `validation/<model>`。第五轮保存 `checkpoint-epoch5.pth` 和 `checkpoint-last.pth`，先保存再评测原图网格，生成 `eval_val_epoch4.json`；汇总文件为 `seven_models_val_summary.csv`。该入口拒绝显式修改固定的融合方式、编码器、长边或训练周期，以及评测超过第五轮的 checkpoint。

单个源实验 checkpoint 可直接使用 Python 入口；评测或续训均需加上 `--input-long-side 518 --epochs 5`，并分别提供 `--checkpoint` 或 `--resume`，其余路径参数与上面的 KITTI full 示例相同。

该套件的迁移验证已通过：七个模型的训练协议与源实现一致；三个真实 val 样本及 batch 在 518×168 下逐字段一致；135 项相关回归测试通过，覆盖启动命令、第五轮保存与评测、自动续训、几何变换和原有模型初始化。未启动完整五轮训练。

## 保真约束

- 两个原 checkpoint 均可严格加载：前融合模型为 315 个 state-dict 键、29,570,945 个参数；双窗口后融合模型为 351 个键、31,940,225 个参数。
- 七个已有 DA3-Base checkpoint 已逐个通过严格加载；新定义的 `image_scaled` 使用同一套 PromptDA-scale 合同。八个模型的 state-dict 键数依次为 269、271、281、283、301、303、313、315。
- v3 manifest 只读并校验 schema、样本数与 SHA256，不生成 split 或深度缓存。
- 训练初始化复现原 DA3-small、DA3-Base wrapper 和后融合构造链中丢弃模块所消耗的 seed-0 RNG 状态；DA3-Base 的 DPT、depth patch 和 voxel encoder 初值已与原构造链逐位校验。
- Unified6 指标在撤销 letterbox 后的原生图像网格计算，保留 NYUv2 Eigen crop、VOID 官方范围以及 anchor/non-anchor 诊断；KITTI-only 指标按原实验在固定 518×168 网格计算。
