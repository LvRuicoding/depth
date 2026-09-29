# OccAny-Depth-Min

这个仓库保留两个 DA3-small 模型的 Unified6 与 KITTI-only 训练/评测代码，并提供八个 DA3-Base 单帧 KITTI 模型：

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

仓库不包含数据、Unified6 v3 manifests、DA3 基础权重、训练 checkpoint 或输出。所有这些均通过路径参数传入。训练和评测固定使用 4 张 GPU；模型结构、优化器、学习率、输入尺寸、训练轮数和评测协议均不提供可变参数。Unified6 训练提供原实验的域均衡模式，以及不做域均衡的自然比例模式。

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

## 保真约束

- 两个原 checkpoint 均可严格加载：前融合模型为 315 个 state-dict 键、29,570,945 个参数；双窗口后融合模型为 351 个键、31,940,225 个参数。
- 七个已有 DA3-Base checkpoint 已逐个通过严格加载；新定义的 `image_scaled` 使用同一套 PromptDA-scale 合同。八个模型的 state-dict 键数依次为 269、271、281、283、301、303、313、315。
- v3 manifest 只读并校验 schema、样本数与 SHA256，不生成 split 或深度缓存。
- 训练初始化复现原 DA3-small、DA3-Base wrapper 和后融合构造链中丢弃模块所消耗的 seed-0 RNG 状态；DA3-Base 的 DPT、depth patch 和 voxel encoder 初值已与原构造链逐位校验。
- Unified6 指标在撤销 letterbox 后的原生图像网格计算，保留 NYUv2 Eigen crop、VOID 官方范围以及 anchor/non-anchor 诊断；KITTI-only 指标按原实验在固定 518×168 网格计算。
