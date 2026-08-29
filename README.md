# 高光谱图像去雾 — 物理引导潜在扩散模型（LDM）

基于 **RSyntHyperPDID** 数据集（182 波段高光谱遥感图像）的物理引导扩散模型去雾项目。
核心思路：AutoEncoder 压缩 182 波段 → 16×64×64 latent，条件 UNet + SpatialFiLM 在 latent 空间做去雾扩散，测试时叠加 DCP（暗通道先验）梯度物理引导。

## 仓库结构

```
├── models/
│   ├── diffusion/          # 阶段3核心：HSIAutoEncoder、条件UNet(conditional_ldm_phase3.py)、DDPM
│   ├── physics/            # 阶段4候选模块：ASM / 逐波段条件注入（被证伪/待验路线）
│   └── psgnet/             # 阶段2基线：PSGNet 全监督模型
├── scripts/
│   ├── rshpdid_dataset.py          # 数据集加载（配对规则、懒加载）
│   ├── train_psgnet_phase2.py      # 阶段2：PSGNet 基线训练
│   ├── test_psgnet_phase2.py       # 阶段2：PSGNet 评测
│   ├── train_conditional_ldm_phase3_full.py  # 阶段3：条件LDM(FiLM) 训练
│   ├── eval_ddim_guidance.py       # DDIM 少步采样 + 测试时 DCP 物理引导评测
│   ├── eval_physics_guidance.py    # 100 步 DDPM + 物理引导评测
│   └── ...                         # 其余为历史实验脚本（可忽略）
├── run_retrain.py           # 一键重训编排：AE → 条件LDM(FiLM) → DDIM评测
├── run_retrain_step2.py     # 编排后半段（跳过已训 AE）
├── serve.py                 # 训练进度实时监控服务（标准库 http.server，端口 8010）
└── webui/                   # 监控网页（dashboard + 各阶段详情页，每 4s 轮询）
```

**不入库**（体积大/生成物）：`data/`、`checkpoints/`、`logs/`、`results/`、`__pycache__/`。

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

### 阶段2：PSGNet 全监督基线（结果：Best 28.84dB / SSIM 0.8911 / SAM 6.13°）

```bash
python scripts/train_psgnet_phase2.py --train_hazy_dir data/train --epochs 100
python scripts/test_psgnet_phase2.py
```

### 阶段3：自编码器（AE 重建上界 ≈ 24.95dB）

```bash
python models/diffusion/ldm_hsi_phase3.py
# 产物: checkpoints/ldm_hsi_autoencoder_best_phase3.pth
```

### 阶段3：条件LDM + FiLM（当前最优扩散路线，Final 20.63dB / SSIM 0.6891 / SAM 10.39°）

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

## 关键结论坐标系（统一评测口径：best ckpt + 全测试集 264 对）

| 方法 | PSNR | SSIM | SAM |
|------|------|------|-----|
| PSGNet 基线（阶段2） | **28.84** | 0.8911 | 6.13° |
| AE 重建上界（latent 压缩极限） | **24.95** | — | — |
| 条件LDM+FiLM 100步（阶段3） | **20.63** | 0.6891 | 10.39° |
| DDIM 少步（10/25/50步） | ≈19.0 | — | — |

已知问题：LDM 天花板受 AE 重建上界 24.95dB 限制；扩散采样仍损失 ~4dB（方向 B 待攻关）；DDIM 步数 25→50 已无增益，瓶颈在条件注入通路而非步数。

## 注意事项

- latent 缓存落盘为**未归一化** latent，评测脚本加载后须用 `z_stats` 做 `(z-mean)/std` 归一化（eval_ddim_guidance.py 已内置）
- DDIM 的 `x0_pred` **不可** clamp 到 [-1,1]（latent 空间值域远超图像域，clamp 会致 50 步采样崩塌）
- 测试时物理引导（DCP 梯度）训练零改动，仅在 `t ≤ guide_start_t` 且 `s>0` 时生效
