# OccAny-Depth-Min

这个仓库只保留以下模型的训练与六域原生分辨率评测代码：

`Stage1DepthLingBotDA3LastPatchDepth4mVoxelPreFusionOnlineKNNPromptDAScaledUnified6Model`

它对应原实验目录：

`output/depth/unified6/single_frame/da3_small_lingbot_tokenfusion_last_patchdepth4m_voxel_prefusion_promptda_scaled`

仓库不包含数据、Unified6 v3 manifests、DA3-small 基础权重、训练 checkpoint 或输出。所有这些均通过路径参数传入。训练和评测固定使用 4 张 GPU；模型结构、优化器、学习率、10 个 epoch、均衡采样、输入尺寸、稀疏点数和评测协议均不提供可变参数。

## 训练

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

## 保真约束

- 原 checkpoint 可按 315 个 state-dict 键严格加载；总参数量为 29,570,945。
- v3 manifest 只读并校验 schema、样本数与 SHA256，不生成 split 或深度缓存。
- 训练初始化复现原 DA3 wrapper 丢弃模块所消耗的 seed-0 RNG 状态，因此保留下来的 DPT、depth patch 与 voxel 分支初值逐位一致。
- 指标在撤销 letterbox 后的原生图像网格计算，保留 NYUv2 Eigen crop、VOID 官方范围以及 anchor/non-anchor 诊断。
