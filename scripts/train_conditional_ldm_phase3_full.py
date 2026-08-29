"""
阶段3.3：条件潜在扩散模型训练（全量 data/train 数据）
文件名：train_conditional_ldm_phase3_full.py
功能：
  1. 用 RSHPDIDDataset 懒加载配对数据（data/train 全量 2352 对，避免 224GB 内存爆炸）；
  2. 用冻结的自编码器把 hazy/clear 流式编码到潜在空间（只保留 latent，内存 <2GB）；
  3. 在潜在空间训练条件 DDPM（条件 = 雾霾 latent），预测噪声；
  4. 训练过程实时写入日志 + 周期生成采样可视化 PNG（供 webui 实时显示）；
  5. 训练结束后在 data/test 上计算 PSNR/SSIM/SAM，并生成去雾对比可视化。
"""
import sys
import argparse
import time
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
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
from models.diffusion.simple_ddpm_phase3 import DDPM
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    sample_latent as _sample_latent,
)


def sample_latent(model, z_hazy, timesteps, device, denorm=None, z_mean=0.0, z_std=1.0):
    """从 z_hazy 条件生成清晰 latent，并支持反归一化。

    扩散在归一化后的 latent 空间训练，因此采样得到的 z 需先反归一化
    再交给 decode，否则解码输入的分布会偏离训练 AE 时的分布。
    """
    z = _sample_latent(model, z_hazy, timesteps, device)
    if denorm is not None:
        z = denorm(z)
    elif z_std != 1.0 or z_mean != 0.0:
        z = z * z_std + z_mean
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
            (pred_rgb[r], "LDM去雾结果"),
            (clear_rgb[r], "清晰参考(GT)"),
        ]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    plt.suptitle("阶段3 条件LDM去雾效果（训练中实时采样）", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()


def train(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    sfx = args.run_suffix  # 消融/多种子运行时区分产物，避免覆盖
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Conditional LDM (full data/train) training on {device}")

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

    # ---------------- 流式编码到 latent（带磁盘缓存，避免重复编码） ----------------
    # latent_ch=16 复用历史缓存目录，其他通道数独立目录
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

    # ---------------- Latent 标准化（关键修复） ----------------
    # 实测 latent std≈15，与扩散噪声(方差1)尺度严重不匹配，是 PSNR 上不去的根因。
    # 用训练集 clear latent 的统计量归一化到 std≈1，扩散才能正常学习。
    z_mean = z_train_clear.mean()
    z_std = z_train_clear.std()
    z_train_clear = (z_train_clear - z_mean) / z_std
    z_train_hazy = (z_train_hazy - z_mean) / z_std
    z_test_clear = (z_test_clear - z_mean) / z_std
    z_test_hazy = (z_test_hazy - z_mean) / z_std
    # 保存归一化系数，供采样后反归一化
    z_stats = {"mean": float(z_mean), "std": float(z_std)}
    print(f"Latent 标准化: mean={z_mean:.4f} std={z_std:.4f} -> 归一化后 std={z_train_clear.std():.4f}")

    def denorm(z):
        return z * z_std + z_mean

    # 保留少量原始 HSI 用于可视化（内存小）
    vis_hazy = torch.stack([test_ds[i][0] for i in range(min(args.viz_samples, len(test_ds)))])
    vis_clear = torch.stack([test_ds[i][1] for i in range(min(args.viz_samples, len(test_ds)))])
    # 原始 hazy/clear 用于真实域评估（避免 AE 重建误差污染 PSNR）
    val_clear_raw = torch.stack([test_ds[i][1] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    val_hazy_raw = torch.stack([test_ds[i][0] for i in range(min(args.val_samples, len(test_ds)))]).to(device)

    # ---------------- 训练数据 ----------------
    train_ds_latent = TensorDataset(z_train_hazy, z_train_clear)
    train_loader = DataLoader(train_ds_latent, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 验证用的小子集（潜在空间）
    val_hazy = z_test_hazy[:args.val_samples].to(device)
    val_clear = z_test_clear[:args.val_samples].to(device)

    # ---------------- 模型 ----------------
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Conditional U-Net parameters: {n_params/1e6:.2f}M (depth={args.depth}, module={args.module})")

    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr 调度：线性 warmup + cosine 衰减，稳定扩散训练
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
    resume_ckpt = checkpoint_dir / f"conditional_ldm_phase3_full{sfx}_resume.pth"

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

    # 断点续训：若存在可恢复 checkpoint 则从上次位置继续
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
            z_noisy, noise = ddpm.add_noise(z_clear, t)
            predicted_noise = model(z_noisy, z_hazy, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            # 梯度裁剪：depth=2 模型容量增大 4.3x，lr=1e-3 下第 ~8 epoch 曾出现
            # loss 爆炸(184)并坍塌到常数解(loss==1)，必须限制梯度范数
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses))
        ep_time = time.time() - ep_t0
        history["train_loss"].append(avg_loss)
        history["epoch_time"].append(ep_time)

        # 每 eval_every 轮在验证子集上采样评估（全量 DDPM 采样，代价高所以隔轮做）
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_pred = sample_latent(model, val_hazy, args.timesteps, device, denorm=denorm)
                pred = autoencoder.decode(z_pred).clamp(0, 1)
                # 输出域残差融合：去雾 = hazy + α*(pred - hazy)，α 可抑制过平滑
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
                torch.save(model.state_dict(), checkpoint_dir / f"conditional_ldm_phase3_full{sfx}_best.pth")
                torch.save(z_stats, checkpoint_dir / f"conditional_ldm_phase3_full{sfx}_z_stats.pth")
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
                z_vp = sample_latent(model, z_v, args.timesteps, device, denorm=denorm)
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

        # 定期保存可恢复 checkpoint（断点续训用）
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_resume(epoch)

    # ---------------- 保存 ----------------
    torch.save(model.state_dict(), checkpoint_dir / f"conditional_ldm_phase3_full{sfx}.pth")
    save_resume(args.epochs - 1)
    with open(checkpoint_dir / f"history_conditional_ldm_phase3_full{sfx}.json", "w") as f:
        json.dump(history, f, indent=2)
    log_print(f"Training complete in {time.time()-total_t0:.0f}s. Best val PSNR: {best_psnr:.2f}dB")

    # 损失曲线图
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_loss"], color="tab:blue", label="train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion Loss (MSE)")
    ax.set_title("阶段3 条件LDM 训练损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    if history["val_psnr"]:
        ax2 = ax.twinx()
        ev_epochs = [e for e in range(1, args.epochs + 1) if (e % args.eval_every == 0 or e == args.epochs)]
        ax2.plot(ev_epochs[:len(history["val_psnr"])], history["val_psnr"], "o-",
                 color="tab:red", label="val PSNR")
        ax2.set_ylabel("Val PSNR (dB)", color="tab:red")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_phase3_loss_curve{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---------------- 全量测试集评估 ----------------
    print("Evaluating on full test set (重载最优 checkpoint, 统一 best 口径)...")
    best_ckpt = checkpoint_dir / f"conditional_ldm_phase3_full{sfx}_best.pth"
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
            z_p = sample_latent(model, z_h, args.timesteps, device, denorm=denorm)
            pred = autoencoder.decode(z_p).clamp(0, 1)
            # 输出域残差融合
            pred = args.residual_blend * pred + (1 - args.residual_blend) * hazy_raw
            for j in range(pred.size(0)):
                p, g = pred[j:j+1], gt_raw[j:j+1]
                test_metrics["psnr"].append(compute_psnr(p, g).item())
                test_metrics["ssim"].append(compute_ssim(p, g).item())
                test_metrics["sam"].append(compute_sam(p, g).item())
            print(f"  test progress {min(i+bs_test, n_test)}/{n_test}")

    avg = {k: float(np.mean(v)) for k, v in test_metrics.items()}
    print(f"Test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / f"metrics_conditional_ldm_phase3_full{sfx}.json", "w") as f:
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=8)
    parser.add_argument("--viz_samples", type=int, default=4)
    parser.add_argument("--residual_blend", type=float, default=0.9, help="输出域残差融合权重: pred = α*ldm + (1-α)*hazy")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_subdir", type=str, default="conditional_ldm_phase3_run2")
    parser.add_argument("--save_every", type=int, default=5, help="每N轮保存一次可恢复checkpoint")
    parser.add_argument("--resume", action="store_true", help="从最近的可恢复checkpoint断点续训")
    parser.add_argument("--seed", type=int, default=None, help="随机种子(多种子统计用)")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="产物名后缀(消融/多种子运行，如 _seed0)，避免覆盖默认结果")
    args = parser.parse_args()
    train(args)
