# 高光谱图像去雾 — 物理引导潜在扩散模型（LDM）

基于 **RSyntHyperPDID** 数据集（182 波段高光谱遥感图像）的物理引导扩散模型去雾项目。
核心思路：AutoEncoder 压缩 182 波段 → 16×64×64 latent，条件 UNet + SpatialFiLM 在 latent 空间做去雾扩散，测试时叠加 DCP（暗通道先验）梯度物理引导。

## 仓库结构

## 数据准备

数据集来自 HuggingFace [nikos74/RSyntHyperPDID](https://huggingface.co/datasets/nikos74/RSyntHyperPDID)（或 hf-mirror 镜像），解压后按如下目录放置：

```
data/
├── clear/            # 清晰GT: XXXX.npy, (256,256,182) float32
├── test/             # 测试雾图: XXXX_YYYY.npy（官方 22 个场景）
└── train/            # 训练雾图: XXXX_YYYY.npy（官方 196 个场景）
```

- **配对规则**：雾图 `XXXX_YYYY.npy` 的 GT 为 `clear/XXXX.npy`（XXXX 为场景 ID）
- 自检：`python scripts/rshpdid_dataset.py`

## 环境依赖

- Python 3.10+
- `torch`（CUDA 版，项目用 GPU 训练）、`numpy`
- 其余均为标准库（`serve.py` 只用 `http.server`）

## 复现流程

### 阶段2：PSGNet 全监督基线

```bash
python scripts/train_psgnet_phase2.py --train_hazy_dir data/train --epochs 100
python scripts/test_psgnet_phase2.py
```

### 阶段3：自编码器（AE）

```bash
python models/diffusion/ldm_hsi_phase3.py
# 产物: checkpoints/ldm_hsi_autoencoder_best_phase3.pth
```

### 阶段3：条件LDM + FiLM（当前最优扩散路线）

```bash
python scripts/train_conditional_ldm_phase3_full.py \
  --module film --seed 42 \
  --run_suffix _mfilm --log_subdir conditional_ldm_phase3_run2
# 产物: checkpoints/conditional_ldm_phase3_full_mfilm_best.pth
#       checkpoints/conditional_ldm_phase3_full_mfilm_z_stats.pth
#       checkpoints/latent_cache_phase3/  (z_train_hazy/z_test_hazy 等 latent 缓存)
```

### 评测：DDIM 少步采样 + 测试时物理引导

```bash
python scripts/eval_ddim_guidance.py --module film \
  --ckpt checkpoints/conditional_ldm_phase3_full_mfilm_best.pth \
  --z_stats checkpoints/conditional_ldm_phase3_full_mfilm_z_stats.pth \
  --ae_ckpt checkpoints/ldm_hsi_autoencoder_best_phase3.pth \
  --lat_cache checkpoints/latent_cache_phase3 \
  --ddim_steps 10 25 50 100 \
  --guidance_s 0 0.8 --guide_start_t 50 \
  --seeds 42 --batch_size 32 \
  --name mphase3_film_ddim
```

### 一键编排（AE → 条件LDM(FiLM) → DDIM 评测）

```bash
python run_retrain.py        # 全链路
python run_retrain_step2.py  # 跳过已训 AE，仅 LDM 训练 + 评测
```

## 训练进度实时监控

```bash
python serve.py              # 启动 http://localhost:8010
```

浏览器打开 `webui/dashboard.html`（总览）或各阶段详情页，页面每 4 秒轮询 `/api/status` 展示损失曲线、验证 PSNR、采样可视化。

## 技术要点与已知问题

- 扩散模型天花板受 AE 重建上界限制（latent 压缩极限）；扩散采样环节仍有压缩空间（方向 B 待攻关）
- latent 缓存落盘为**未归一化** latent，评测脚本加载后须用 `z_stats` 做 `(z-mean)/std` 归一化（eval\_ddim\_guidance.py 已内置）
- DDIM 的 `x0_pred` **不可** clamp 到 \[-1,1]（latent 空间值域远超图像域，clamp 会致多步采样崩塌）
- 测试时物理引导（DCP 梯度）训练零改动，仅在 `t ≤ guide_start_t` 且 `s>0` 时生效

