"""
L3 物理桥训练：'清晰→雾' 桥潜在扩散模型（全量 data/train）
文件名：train_bridge_ldm.py
对应：实验任务清单_v3.0.md【L3】；fork 自 train_conditional_ldm_phase3_full.py

与 phase3_full 的差异（共 3 处实质改动，其余结构逐行保留）：
  1. 前向过程：ddpm.add_noise(z_clear, t)  →  bridge.q_sample(z_clear, z_hazy, t)
     即 z_t = s_t·z_clear + (1-s_t)·z_hazy + σ_t·ε（扩散方向=物理退化方向，
     消除"物理梯度与扩散目标拔河"的历史失败机制，见 v3.0 清单 §1）；
  2. 预测目标：ε 预测  →  x0 预测（直接预测清晰 latent，采样全程无除法，
     规避 §11 DDIM 的 s→0 除法放大病理；详见 models/diffusion/bridge_ldm.py 头注释）；
  3. 采样起点：纯噪声 randn  →  bridge.start = z_hazy + σ_max·ε（雾+噪声），
     路径从"噪声→清晰"缩短为"雾→清晰"，少步采样天然成立（方向 B 复活）。

复用资产：AE 权重、latent 磁盘缓存（checkpoints/latent_cache_phase3/，缓存为
原始未归一化 latent，与 phase3_full 完全同源）、ConditionalLatentUNet+FiLM、
warmup+cosine 调度、断点续训、统一评测口径（blend=1.0 + per-image 指标）。

用法（本地 F 盘数据）：
  python scripts/train_bridge_ldm.py --train_hazy_dir F:/data/train \
      --test_hazy_dir F:/data/test --clear_dir F:/data/clear \
      --module film --seed 42 --run_suffix _mfilm --log_subdir bridge_ldm_run1
服务器上（数据在相对路径 data/ 下）：
  python scripts/train_bridge_ldm.py --module film --seed 42 \
      --run_suffix _mfilm --log_subdir bridge_ldm_run1
"""
import sys
import argparse
import time
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import RSHPDIDDataset
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
)
from models.diffusion.bridge_ldm import HazeBridge, sample_bridge


def bridge_sample_denorm(model, z_hazy, bridge, device, denorm=None, eta=0.0, steps=None):
    """桥采样得到清晰 latent 并反归一化（decode 前必须 denorm，漏掉只剩 ~16.5dB）。"""
    z = sample_bridge(model, z_hazy, bridge, steps or bridge.timesteps,
                      eta=eta, device=device)
    if denorm is not None:
        z = denorm(z)
    return z


def encode_to_latent(dataset, autoencoder, device, batch_size=16, desc=""):
    """流式编码数据集到 latent 空间，只保留 latent 张量（内存友好）。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    z_hazy, z_clear = [], []
    n = len(dataset)
    with torch.no_grad():
        for i, (hazy, clear) in enumerate(loader):
            z_hazy.append(autoencoder.encode(hazy.to(device)).cpu())
            z_clear.append(autoencoder.encode(clear.to(device)).cpu())
            if (i + 1) % 25 == 0 or (i + 1) * batch_size >= n:
                print(f"  [{desc}] 编码中 {min((i+1)*batch_size, n)}/{n}")
    return torch.cat(z_hazy), torch.cat(z_clear)


def to_rgb(img, brighten=1.0):
    """(C,H,W) numpy [0,1] -> RGB，取 R/G/B 波段。"""
    B = img.shape[0]
    r_idx = min(int(B * 0.72), B - 1)
    g_idx = int(B * 0.46)
    b_idx = int(B * 0.20)
    rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
    return np.clip(rgb * brighten, 0, 1)


def build_sample_figure(hazy_rgb, pred_rgb, clear_rgb, save_path, n=4):
    """生成 雾霾输入 | 去雾结果 | GT 对比图。"""
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        for c, (im, title) in enumerate([
            (hazy_rgb[r], "雾霾输入"),
            (pred_rgb[r], "桥LDM去雾结果"),
            (clear_rgb[r], "清晰参考(GT)"),
        ]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    plt.suptitle("L3 物理桥LDM去雾效果（训练中实时采样）", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()


def train(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    sfx = args.run_suffix
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Haze-Bridge LDM (full data/train) training on {device}")

    # ---------------- 数据 ----------------
    print("Building datasets...")
    train_ds = RSHPDIDDataset(args.train_hazy_dir, args.clear_dir, max_samples=args.max_train_samples)
    test_ds = RSHPDIDDataset(args.test_hazy_dir, args.clear_dir)
    print(f"Train pairs: {len(train_ds)}, Test pairs: {len(test_ds)}")

    in_ch = train_ds.num_bands()
    H, W = train_ds.spatial_size()
    print(f"Bands: {in_ch}, size: {H}x{W}")

    # ---------------- 冻结自编码器 ----------------
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.ae_ckpt) if args.ae_ckpt else Path(args.checkpoint_dir) / "ldm_hsi_autoencoder_best_phase3.pth"
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"自编码器 checkpoint 不存在: {ae_ckpt}，请先运行 ldm_hsi_phase3.py")
    print(f"Loading AE from {ae_ckpt}")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # ---------------- 流式编码到 latent（带磁盘缓存，与 phase3_full 同源可复用） ----------------
    lat_tag = "" if args.latent_ch == 16 else f"_lc{args.latent_ch}"
    latent_cache_dir = Path(args.checkpoint_dir) / f"latent_cache_phase3{lat_tag}"
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "z_train_hazy": latent_cache_dir / "z_train_hazy.pt",
        "z_train_clear": latent_cache_dir / "z_train_clear.pt",
        "z_test_hazy": latent_cache_dir / "z_test_hazy.pt",
        "z_test_clear": latent_cache_dir / "z_test_clear.pt",
    }
    if all(p.exists() for p in cache_paths.values()):
        print("Loading cached latents...")
        z_train_hazy = torch.load(cache_paths["z_train_hazy"])
        z_train_clear = torch.load(cache_paths["z_train_clear"])
        z_test_hazy = torch.load(cache_paths["z_test_hazy"])
        z_test_clear = torch.load(cache_paths["z_test_clear"])
    else:
        print("Encoding train set to latent space (streaming)...")
        t0 = time.time()
        z_train_hazy, z_train_clear = encode_to_latent(train_ds, autoencoder, device, desc="train")
        print(f"Train latent encoded in {time.time()-t0:.1f}s, shapes: {z_train_hazy.shape}, {z_train_clear.shape}")

        print("Encoding test set to latent space (streaming)...")
        t0 = time.time()
        z_test_hazy, z_test_clear = encode_to_latent(test_ds, autoencoder, device, desc="test")
        print(f"Test latent encoded in {time.time()-t0:.1f}s, shapes: {z_test_hazy.shape}, {z_test_clear.shape}")
        for key, p in cache_paths.items():
            torch.save(locals()[key], p)
        print(f"Latents cached to {latent_cache_dir}")

    # ---------------- Latent 标准化（与 phase3_full 完全一致） ----------------
    z_mean = z_train_clear.mean()
    z_std = z_train_clear.std()
    z_train_clear = (z_train_clear - z_mean) / z_std
    z_train_hazy = (z_train_hazy - z_mean) / z_std
    z_test_clear = (z_test_clear - z_mean) / z_std
    z_test_hazy = (z_test_hazy - z_mean) / z_std
    z_stats = {"mean": float(z_mean), "std": float(z_std)}
    print(f"Latent 标准化: mean={z_mean:.4f} std={z_std:.4f} -> 归一化后 std={z_train_clear.std():.4f}")
    # 桥的插值/噪声均作用于归一化空间；hazy latent 与 clear 同尺度（同一 AE + 同一 stats）
    print(f"归一化后 hazy latent std={z_train_hazy.std():.4f}（应≈1，桥插值两端同尺度）")

    def denorm(z):
        return z * z_std + z_mean

    # 保留少量原始 HSI 用于可视化（内存小）
    vis_hazy = torch.stack([test_ds[i][0] for i in range(min(args.viz_samples, len(test_ds)))])
    vis_clear = torch.stack([test_ds[i][1] for i in range(min(args.viz_samples, len(test_ds)))])
    val_clear_raw = torch.stack([test_ds[i][1] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    val_hazy_raw = torch.stack([test_ds[i][0] for i in range(min(args.val_samples, len(test_ds)))]).to(device)

    # ---------------- 训练数据 ----------------
    train_ds_latent = TensorDataset(z_train_hazy, z_train_clear)
    train_loader = DataLoader(train_ds_latent, batch_size=args.batch_size, shuffle=True, num_workers=0)

    val_hazy = z_test_hazy[:args.val_samples].to(device)

    # ---------------- 模型 + 桥调度 ----------------
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Conditional U-Net parameters: {n_params/1e6:.2f}M (depth={args.depth}, module={args.module})")

    bridge = HazeBridge(timesteps=args.timesteps, sigma_max=args.sigma_max,
                        sched=args.sched, device=device)
    print(f"HazeBridge: sched={args.sched} sigma_max={args.sigma_max} timesteps={args.timesteps}")
    print(f"  s: {bridge.s[0]:.4f} -> {bridge.s[-1]:.4f}, "
          f"sigma: {bridge.sigma[0]:.4f} -> {bridge.sigma[-1]:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr 调度：线性 warmup + cosine 衰减（沿用 phase3_full）
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(0.03 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------------- 日志目录 ----------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / (args.log_subdir + sfx)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_log.txt"
    sample_file = log_dir / "sample_latest.png"
    resume_ckpt = checkpoint_dir / f"bridge_ldm_phase3{sfx}_resume.pth"

    def log_print(msg, echo=True):
        if echo:
            print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # ---------------- 训练循环 ----------------
    history = {"train_loss": [], "val_psnr": [], "val_ssim": [], "val_sam": [], "epoch_time": []}
    best_psnr = -1.0
    start_epoch = 0
    total_t0 = time.time()

    if args.resume and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"]
        history.update(state["history"])
        best_psnr = state.get("best_psnr", -1.0)
        log_print(f"[resume] 从第 {start_epoch} 轮续训，best PSNR={best_psnr:.2f}dB，历史 {len(history['train_loss'])} 条")

    def save_resume(epoch_done):
        torch.save({
            "epoch": epoch_done + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_psnr": best_psnr,
            "z_stats": z_stats,
        }, resume_ckpt)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_t0 = time.time()
        losses = []
        for z_hazy, z_clear in train_loader:
            z_hazy, z_clear = z_hazy.to(device), z_clear.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (z_clear.size(0),), device=device).long()
            # 【改动1】桥前向：z_t = s_t·z_clear + (1-s_t)·z_hazy + σ_t·ε
            z_t = bridge.q_sample(z_clear, z_hazy, t)
            # 【改动2】x0 预测：直接预测清晰 latent（无 ε 反解，采样无除法）
            x0_hat = model(z_t, z_hazy, t)
            loss = F.mse_loss(x0_hat, z_clear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses))
        ep_time = time.time() - ep_t0
        history["train_loss"].append(avg_loss)
        history["epoch_time"].append(ep_time)

        # 每 eval_every 轮在验证子集上采样评估
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # 固定验证起点噪声：best-ckpt 选择跨 epoch 可比
                torch.manual_seed(1234)
                z_pred = bridge_sample_denorm(model, val_hazy, bridge, device,
                                               denorm=denorm, eta=args.sample_eta)
                pred = autoencoder.decode(z_pred).clamp(0, 1)
                residual_blend = args.residual_blend
                pred = residual_blend * pred + (1 - residual_blend) * val_hazy_raw
                psnr = compute_psnr(pred, val_clear_raw).item()
                ssim = compute_ssim(pred, val_clear_raw).item()
                sam = compute_sam(pred, val_clear_raw).item()
            history["val_psnr"].append(psnr)
            history["val_ssim"].append(ssim)
            history["val_sam"].append(sam)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), checkpoint_dir / f"bridge_ldm_phase3{sfx}_best.pth")
                torch.save(z_stats, checkpoint_dir / f"bridge_ldm_phase3{sfx}_z_stats.pth")
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} "
                      f"test_psnr={psnr:.2f} test_ssim={ssim:.4f} test_sam={sam:.2f} "
                      f"time={ep_time:.1f}s elapsed={time.time()-total_t0:.0f}s")
        else:
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} "
                      f"time={ep_time:.1f}s elapsed={time.time()-total_t0:.0f}s")

        # 周期生成采样可视化（供网页实时显示）
        if (epoch + 1) % args.viz_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_v = z_test_hazy[:args.viz_samples].to(device)
                z_vp = bridge_sample_denorm(model, z_v, bridge, device,
                                            denorm=denorm, eta=args.sample_eta)
                pred_v = autoencoder.decode(z_vp).clamp(0, 1)
                pred_v = args.residual_blend * pred_v + (1 - args.residual_blend) * vis_hazy[:args.viz_samples].to(device)
                pred_v = pred_v.cpu().numpy()
            build_sample_figure(
                [to_rgb(v.numpy(), 1.3) for v in vis_hazy],
                [to_rgb(p) for p in pred_v],
                [to_rgb(v.numpy()) for v in vis_clear],
                sample_file,
                n=args.viz_samples,
            )
            log_print(f"[viz] 采样可视化已更新: {sample_file}", echo=False)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_resume(epoch)

    # ---------------- 保存 ----------------
    torch.save(model.state_dict(), checkpoint_dir / f"bridge_ldm_phase3{sfx}.pth")
    save_resume(args.epochs - 1)
    with open(checkpoint_dir / f"history_bridge_ldm_phase3{sfx}.json", "w") as f:
        json.dump(history, f, indent=2)
    log_print(f"Training complete in {time.time()-total_t0:.0f}s. Best val PSNR: {best_psnr:.2f}dB")

    # 损失曲线图
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_loss"], color="tab:blue", label="train loss (x0 MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bridge Loss (MSE)")
    ax.set_title("L3 物理桥LDM 训练损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    if history["val_psnr"]:
        ax2 = ax.twinx()
        ev_epochs = [e for e in range(1, args.epochs + 1) if (e % args.eval_every == 0 or e == args.epochs)]
        ax2.plot(ev_epochs[:len(history["val_psnr"])], history["val_psnr"], "o-",
                 color="tab:red", label="val PSNR")
        ax2.set_ylabel("Val PSNR (dB)", color="tab:red")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_bridge_ldm_loss_curve{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- 全量测试集评估（统一口径：best ckpt + blend=1.0 + per-image） ----------------
    print("Evaluating on full test set (重载最优 checkpoint, 统一 best 口径)...")
    best_ckpt = checkpoint_dir / f"bridge_ldm_phase3{sfx}_best.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best checkpoint: {best_ckpt} (best val PSNR={best_psnr:.2f}dB)")
    model.eval()
    test_metrics = {"psnr": [], "ssim": [], "sam": []}
    n_test = len(z_test_hazy)
    bs_test = 8
    with torch.no_grad():
        for i in range(0, n_test, bs_test):
            z_h = z_test_hazy[i:i+bs_test].to(device)
            hazy_raw = torch.stack([test_ds[k][0] for k in range(i, min(i+bs_test, n_test))]).to(device)
            gt_raw = torch.stack([test_ds[k][1] for k in range(i, min(i+bs_test, n_test))]).to(device)
            z_p = bridge_sample_denorm(model, z_h, bridge, device,
                                       denorm=denorm, eta=args.sample_eta)
            pred = autoencoder.decode(z_p).clamp(0, 1)
            pred = args.residual_blend * pred + (1 - args.residual_blend) * hazy_raw
            for j in range(pred.size(0)):
                p, g = pred[j:j+1], gt_raw[j:j+1]
                test_metrics["psnr"].append(compute_psnr(p, g).item())
                test_metrics["ssim"].append(compute_ssim(p, g).item())
                test_metrics["sam"].append(compute_sam(p, g).item())
            print(f"  test progress {min(i+bs_test, n_test)}/{n_test}")

    avg = {k: float(np.mean(v)) for k, v in test_metrics.items()}
    print(f"Test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / f"metrics_bridge_ldm_phase3{sfx}.json", "w") as f:
        json.dump(avg, f, indent=2)
    log_print(f"Final test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hazy_dir", type=str, default="data/train")
    parser.add_argument("--test_hazy_dir", type=str, default="data/test")
    parser.add_argument("--clear_dir", type=str, default="data/clear")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--latent_ch", type=int, default=16)
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="自编码器 checkpoint 路径（默认 checkpoints/ldm_hsi_autoencoder_best_phase3.pth）")
    parser.add_argument("--ae_base_ch", type=int, default=64)
    parser.add_argument("--base_ch", type=int, default=192, help="条件 U-Net 基础通道数")
    parser.add_argument("--depth", type=int, default=1, help="条件 U-Net 下采样级数(1=旧结构)")
    parser.add_argument("--module", type=str, default="none",
                        choices=["none", "se", "cbam", "film"],
                        help="容量增强模块: none/se/cbam/film")
    parser.add_argument("--time_emb_dim", type=int, default=256, help="时间嵌入维度")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--sigma_max", type=float, default=0.5,
                        help="桥雾端噪声强度 σ_max（多样性与质量的权衡轴；注意续训时必须与首次训练一致）")
    parser.add_argument("--sched", type=str, default="linear", choices=["linear", "ddpm"],
                        help="桥调度形状: linear=雾浓度严格线性(物理叙事)，ddpm=沿用 ᾱ 形状重标定")
    parser.add_argument("--sample_eta", type=float, default=0.0,
                        help="验证/测试采样步进噪声(0=确定性 DDIM 式, 1=与训练边缘分布一致)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=8)
    parser.add_argument("--viz_samples", type=int, default=4)
    parser.add_argument("--residual_blend", type=float, default=1.0,
                        help="输出域残差融合权重（统一口径用 1.0，即直接输出）")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_subdir", type=str, default="bridge_ldm_run1")
    parser.add_argument("--save_every", type=int, default=5, help="每N轮保存一次可恢复checkpoint")
    parser.add_argument("--resume", action="store_true", help="从最近的可恢复checkpoint断点续训")
    parser.add_argument("--seed", type=int, default=None, help="随机种子(多种子统计用)")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="产物名后缀(消融/多种子运行，如 _seed0)，避免覆盖默认结果")
    args = parser.parse_args()
    train(args)
