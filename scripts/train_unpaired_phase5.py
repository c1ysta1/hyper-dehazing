"""
阶段五：无配对（Unpaired）物理自监督扩散去雾训练
文件名：train_unpaired_phase5.py
核心思想（对照实验任务清单【任务5.1】）：
  - 训练监督信号只来自雾霾图，清晰图不参与训练（测试集 GT 仅用于评估）；
  - 物理自监督：
      1) 伪清晰目标初始化：暗通道先验(DCP)反演 J_init = (I - A(1-t)) / t，
         完全由雾霾图 + 物理先验得到，不接触任何 clear 数据；
      2) 扩散训练：以雾霾 latent 为条件，向伪清晰 latent 学去噪（L_diff）；
      3) 物理循环一致性（L_asm）：本步 x0 估计 -> 解码 J_pred -> 冻结的预训练 ASM
         重建 I_rec = J_pred*t + A(1-t)，要求 I_rec ≈ 雾霾输入（J->I 循环）；
      4) 自举(Bootstrap)：每隔 N epoch 用当前模型快速采样，与伪目标 EMA 混合，
         使伪清晰目标随模型能力提升而细化（self-training）。
  - 总损失 L = L_diff + λ_asm * L_asm（L_asm 即物理循环一致性损失）；
  - ASM 使用阶段4预训练权重并冻结（固定物理先验，防止无配对场景下 t->1 平凡解）；
  - 日志格式与阶段4一致，供 webui 实时展示；结束后全 test 集评估 PSNR/SSIM/SAM。
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
from models.diffusion.simple_ddpm_phase3 import DDPM
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    sample_latent as _sample_latent,
)
from models.physics.asm_module_phase4 import AtmosphericScatteringModel


def sample_latent(model, z_hazy, timesteps, device, denorm=None):
    z = _sample_latent(model, z_hazy, timesteps, device)
    if denorm is not None:
        z = denorm(z)
    return z


def estimate_atmosphere_light(hazy, frac=0.001):
    """每波段取最亮 frac 比例像素的均值作为大气光 A（经典 DCP 做法）。
    hazy: (B,C,H,W) -> A: (B,C,1,1)"""
    flat = hazy.flatten(2)                                   # B,C,HW
    k = max(1, int(flat.shape[-1] * frac))
    topk = flat.topk(k, dim=-1).values                       # B,C,k
    A = topk.mean(dim=-1)[:, :, None, None]
    return A


def dcp_dehaze(hazy, window=15, omega=0.95):
    """暗通道先验反演：只用雾霾图得到初始伪清晰图。
    hazy: (B,C,H,W) -> J_init: (B,C,H,W) in [0,1]"""
    pad = window // 2
    dark = -F.max_pool2d(-hazy, kernel_size=window, stride=1, padding=pad)  # 每波段 erode
    t = (1.0 - omega * dark).clamp(min=0.1)
    A = estimate_atmosphere_light(hazy)
    J = (hazy - A * (1.0 - t)) / t
    return J.clamp(0, 1), t, A


def to_rgb(img, brighten=1.0):
    """(C,H,W) numpy [0,1] -> RGB。"""
    B = img.shape[0]
    r_idx = min(int(B * 0.72), B - 1)
    g_idx = int(B * 0.46)
    b_idx = int(B * 0.20)
    rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
    return np.clip(rgb * brighten, 0, 1)


def build_sample_figure(hazy_rgb, pred_rgb, clear_rgb, save_path, n=4):
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        for c, (im, title) in enumerate([
            (hazy_rgb[r], "雾霾输入"), (pred_rgb[r], "无配对去雾"), (clear_rgb[r], "清晰参考(GT)"),
        ]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    plt.suptitle("阶段5 无配对LDM去雾效果（训练中实时采样）", fontsize=13)
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
    log_line = (f"Unpaired physics self-supervised LDM on {device}, "
                f"lambda_asm={args.lambda_asm}, epochs={args.epochs}, "
                f"bootstrap_every={args.bootstrap_every} ema={args.ema_beta}")
    print(log_line)

    print("Building datasets (训练只用雾霾图)...")
    train_ds = RSHPDIDDataset(args.train_hazy_dir, args.clear_dir, max_samples=args.max_train_samples)
    test_ds = RSHPDIDDataset(args.test_hazy_dir, args.clear_dir)
    print(f"Train hazy: {len(train_ds)}, Test pairs(eval only): {len(test_ds)}")

    in_ch = train_ds.num_bands()
    H, W = train_ds.spatial_size()
    print(f"Bands: {in_ch}, size: {H}x{W}")

    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.ae_ckpt) if args.ae_ckpt else Path(args.checkpoint_dir) / "ldm_hsi_autoencoder_best_phase3.pth"
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"自编码器 checkpoint 不存在: {ae_ckpt}")
    print(f"Loading AE from {ae_ckpt}")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # ---------------- 复用阶段4 latent 缓存（z_hazy/z_stats/hazy_low） ----------------
    # latent_ch=16 复用历史缓存目录，其他通道数独立目录（与阶段4一致）
    lat_tag = "" if args.latent_ch == 16 else f"_lc{args.latent_ch}"
    lat_cache = Path(args.lat_cache) if args.lat_cache else Path(args.checkpoint_dir) / f"latents_phase4{lat_tag}"
    need = ["z_train_hazy", "z_test_hazy", "z_test_clear", "hazy_low", "z_stats"]
    if not all((lat_cache / f"{n}.pt").exists() for n in need):
        raise FileNotFoundError(f"缺少阶段4 latent 缓存({lat_cache})，请先运行 train_physics_ldm_phase4.py")
    print(f"Loading hazy latents from {lat_cache}...")
    z_train_hazy = torch.load(lat_cache / "z_train_hazy.pt")
    z_test_hazy = torch.load(lat_cache / "z_test_hazy.pt")
    z_test_clear = torch.load(lat_cache / "z_test_clear.pt")
    hazy_low_train = torch.load(lat_cache / "hazy_low.pt")
    s = torch.load(lat_cache / "z_stats.pt")
    z_mean, z_std = float(s["mean"]), float(s["std"])
    print(f"hazy latents: train={tuple(z_train_hazy.shape)}, test={tuple(z_test_hazy.shape)}")

    def denorm(z):
        return z * z_std + z_mean

    # ---------------- 伪清晰目标：DCP 反演初始化（只用雾霾图） ----------------
    p5_cache = Path(args.checkpoint_dir) / f"latents_phase5{lat_tag}"
    p5_cache.mkdir(parents=True, exist_ok=True)
    z_pseudo_path = p5_cache / "z_pseudo_init.pt"
    if z_pseudo_path.exists() and not args.no_cache:
        z_pseudo = torch.load(z_pseudo_path)
        print(f"Loaded pseudo targets from cache: {tuple(z_pseudo.shape)}")
    else:
        print("Building DCP pseudo-clear targets (hazy only, once)...")
        t0 = time.time()
        z_ps = []
        bs = 16
        with torch.no_grad():
            for i in range(0, len(train_ds), bs):
                hazy = torch.stack([train_ds[k][0] for k in range(i, min(i + bs, len(train_ds)))]).to(device)
                J_init, _, _ = dcp_dehaze(hazy)
                z = autoencoder.encode(J_init).float()
                z = (z - z_mean) / z_std
                z_ps.append(z.cpu())
                if (i // bs) % 25 == 0:
                    print(f"  [DCP] {min(i + bs, len(train_ds))}/{len(train_ds)}")
        z_pseudo = torch.cat(z_ps)
        torch.save(z_pseudo, z_pseudo_path)
        print(f"DCP pseudo targets built in {time.time()-t0:.1f}s -> {z_pseudo_path}")

    # 小样本调试时截断缓存张量与伪目标对齐（全量训练时无影响）
    n_use = len(z_pseudo)
    if len(z_train_hazy) > n_use:
        print(f"[align] 截断缓存张量 {len(z_train_hazy)} -> {n_use}（小样本模式）")
        z_train_hazy = z_train_hazy[:n_use]
        hazy_low_train = hazy_low_train[:n_use]

    # DCP 初始伪目标质量参考（在 val 子集上，仅评估参考）
    val_hazy_raw = torch.stack([test_ds[i][0] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    val_clear_raw = torch.stack([test_ds[i][1] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    vis_hazy = torch.stack([test_ds[i][0] for i in range(min(args.viz_samples, len(test_ds)))])
    vis_clear = torch.stack([test_ds[i][1] for i in range(min(args.viz_samples, len(test_ds)))])
    val_hazy = z_test_hazy[:args.val_samples].to(device)
    with torch.no_grad():
        J_dcp, _, _ = dcp_dehaze(val_hazy_raw)
        dcp_psnr = compute_psnr(J_dcp, val_clear_raw).item()
    print(f"[ref] DCP 反演初始伪目标质量: val PSNR={dcp_psnr:.2f}dB (无配对起点)")

    # ---------------- 冻结的预训练 ASM（物理先验，防平凡解） ----------------
    asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=args.ae_base_ch).to(device)
    asm_ckpt = Path(args.checkpoint_dir) / "asm_phase4.pth"
    if asm_ckpt.exists():
        asm.load_state_dict(torch.load(asm_ckpt, map_location=device))
        print(f"Loaded pretrained ASM from {asm_ckpt} (frozen)")
    else:
        print("[warn] 未找到 asm_phase4.pth，ASM 随机初始化(frozen)")
    asm.eval()
    for p in asm.parameters():
        p.requires_grad = False

    train_loader = DataLoader(
        TensorDataset(z_train_hazy, z_pseudo, hazy_low_train),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device)
    if args.init_from_phase4:
        p4 = Path(args.checkpoint_dir) / "physics_ldm_phase4_best.pth"
        if p4.exists():
            model.load_state_dict(torch.load(p4, map_location=device))
            print(f"[init] 从阶段4最优权重热启动: {p4}")
        else:
            print(f"[warn] 未找到 {p4}，从零训练")
    print(f"Conditional U-Net parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (depth={args.depth})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = max(1, int(0.03 * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ddpm = DDPM(timesteps=args.timesteps, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir) / (args.log_subdir + sfx)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_log.txt"
    sample_file = log_dir / "sample_latest.png"

    def log_print(msg, echo=True):
        if echo:
            print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_print(log_line)
    log_print(f"[ref] DCP 反演初始伪目标质量: val PSNR={dcp_psnr:.2f}dB (无配对起点)")

    history = {"diff_loss": [], "asm_loss": [], "total_loss": [],
               "val_psnr": [], "val_ssim": [], "val_sam": [], "epoch_time": []}
    best_psnr = -1.0
    total_t0 = time.time()

    # 自举子集索引（随机抽取，与 latent 索引对齐；长度以伪目标为准，兼容小样本调试）
    n_pool = len(z_pseudo)
    boot_idx = torch.randperm(n_pool)[:min(args.bootstrap_subset, n_pool)]

    def bootstrap_pseudo_targets():
        """用当前模型快速采样，EMA 混合更新伪清晰目标（self-training）。"""
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(boot_idx), args.batch_size):
                sel = boot_idx[i:i + args.batch_size]
                z_h = z_train_hazy[sel].to(device)
                z_out = _sample_latent(model, z_h, args.bootstrap_steps, device)  # 归一化空间
                z_pseudo[sel] = (args.ema_beta * z_pseudo[sel].cpu()
                                 + (1 - args.ema_beta) * z_out.cpu()).float()
        model.train()
        return time.time() - t0

    for epoch in range(args.epochs):
        # 自举更新伪目标
        if args.bootstrap_every > 0 and epoch > 0 and epoch % args.bootstrap_every == 0:
            bt = bootstrap_pseudo_targets()
            log_print(f"[bootstrap] pseudo-target EMA updated: subset={len(boot_idx)} "
                      f"steps={args.bootstrap_steps} ema={args.ema_beta} time={bt:.0f}s")

        model.train()
        ep_t0 = time.time()
        d_losses, a_losses, t_losses = [], [], []
        for step, (z_hazy, z_pseudo_b, hz_low) in enumerate(train_loader):
            z_hazy, z_pseudo_b = z_hazy.to(device), z_pseudo_b.to(device)
            hz_low = hz_low.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (z_pseudo_b.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_pseudo_b, t)
            pred_noise = model(z_noisy, z_hazy, t)
            diff_loss = F.mse_loss(pred_noise, noise)

            # 物理循环一致性（解码昂贵，每隔 asm_every 步算一次）
            # 注: ASM 参数已 requires_grad=False(冻结)，此处正常计算以保持
            # J_pred -> z0_est -> pred_noise 的梯度回传（no_grad 会切断该路径）
            if step % args.asm_every == 0:
                ac = ddpm.alphas_cumprod[t][:, None, None, None]
                z0_est = (z_noisy - (1.0 - ac).clamp(min=1e-8).sqrt() * pred_noise) / ac.clamp(min=1e-8).sqrt()
                J_pred = autoencoder.decode(denorm(z0_est)).clamp(0, 1)
                if args.asm_scale > 0 and J_pred.shape[-1] != args.asm_scale:
                    jp_low = F.interpolate(J_pred, size=(args.asm_scale, args.asm_scale), mode="area")
                else:
                    jp_low = J_pred
                asm_loss, _, _ = asm.physics_loss(hz_low, jp_low)
                total_loss = diff_loss + args.lambda_asm * asm_loss
            else:
                asm_loss = None
                total_loss = diff_loss
            total_loss.backward()
            # 梯度裁剪：防止 depth=2 大模型训练不稳定（参考阶段3 depth=2 loss 爆炸坍塌教训）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            d_losses.append(diff_loss.item())
            a_losses.append(asm_loss.item() if asm_loss is not None else 0.0)
            t_losses.append(total_loss.item())

        avg_d = float(np.mean(d_losses))
        avg_a = float(np.mean(a_losses))
        avg_t = float(np.mean(t_losses))
        ep_time = time.time() - ep_t0
        history["diff_loss"].append(avg_d)
        history["asm_loss"].append(avg_a)
        history["total_loss"].append(avg_t)
        history["epoch_time"].append(ep_time)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_pred = sample_latent(model, val_hazy, args.timesteps, device, denorm=denorm)
                pred = autoencoder.decode(z_pred).clamp(0, 1)
                pred = args.residual_blend * pred + (1 - args.residual_blend) * val_hazy_raw
                psnr = compute_psnr(pred, val_clear_raw).item()
                ssim = compute_ssim(pred, val_clear_raw).item()
                sam = compute_sam(pred, val_clear_raw).item()
            history["val_psnr"].append(psnr)
            history["val_ssim"].append(ssim)
            history["val_sam"].append(sam)
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), checkpoint_dir / f"unpaired_phase5{sfx}_best.pth")
                torch.save({"mean": z_mean, "std": z_std}, checkpoint_dir / f"unpaired_phase5{sfx}_z_stats.pth")
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_t:.4f} "
                      f"diff={avg_d:.4f} asm={avg_a:.4f} "
                      f"test_psnr={psnr:.2f} test_ssim={ssim:.4f} test_sam={sam:.2f} "
                      f"time={ep_time:.1f}s elapsed={time.time()-total_t0:.0f}s")
        else:
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_t:.4f} "
                      f"diff={avg_d:.4f} asm={avg_a:.4f} "
                      f"time={ep_time:.1f}s elapsed={time.time()-total_t0:.0f}s")

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

    torch.save(model.state_dict(), checkpoint_dir / f"unpaired_phase5{sfx}.pth")
    with open(checkpoint_dir / f"history_unpaired_phase5{sfx}.json", "w") as f:
        json.dump(history, f, indent=2)
    log_print(f"Training complete in {time.time()-total_t0:.0f}s. Best val PSNR: {best_psnr:.2f}dB")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["diff_loss"], color="tab:blue", label="diffusion loss")
    ax.plot(history["asm_loss"], color="tab:orange", label="asm(cycle) loss")
    ax.plot(history["total_loss"], color="tab:green", ls="--", label="total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("阶段5 无配对LDM 训练损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    if history["val_psnr"]:
        ax2 = ax.twinx()
        ev_epochs = [e for e in range(1, args.epochs + 1) if (e % args.eval_every == 0 or e == args.epochs)]
        ax2.plot(ev_epochs[:len(history["val_psnr"])], history["val_psnr"], "o-",
                 color="tab:red", label="val PSNR")
        ax2.set_ylabel("Val PSNR (dB)", color="tab:red")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_phase5_unpaired_loss_curve{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Evaluating on full test set (重载最优 checkpoint, 统一 best 口径)...")
    best_ckpt = checkpoint_dir / f"unpaired_phase5{sfx}_best.pth"
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
            pred = args.residual_blend * pred + (1 - args.residual_blend) * hazy_raw
            for j in range(pred.size(0)):
                p, g = pred[j:j+1], gt_raw[j:j+1]
                test_metrics["psnr"].append(compute_psnr(p, g).item())
                test_metrics["ssim"].append(compute_ssim(p, g).item())
                test_metrics["sam"].append(compute_sam(p, g).item())
            print(f"  test progress {min(i+bs_test, n_test)}/{n_test}")

    avg = {k: float(np.mean(v)) for k, v in test_metrics.items()}
    print(f"Test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / f"metrics_unpaired_phase5{sfx}.json", "w") as f:
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
    parser.add_argument("--lat_cache", type=str, default=None,
                        help="阶段4 latent 缓存目录（默认 checkpoints/latents_phase4[_lc{N}]）")
    parser.add_argument("--ae_base_ch", type=int, default=64)
    parser.add_argument("--base_ch", type=int, default=192)
    parser.add_argument("--depth", type=int, default=1, help="条件 U-Net 下采样级数(1=旧结构)")
    parser.add_argument("--module", type=str, default="none",
                        choices=["none", "se", "cbam", "film"],
                        help="容量增强模块: none/se/cbam/film")
    parser.add_argument("--time_emb_dim", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=10)
    parser.add_argument("--val_samples", type=int, default=8)
    parser.add_argument("--viz_samples", type=int, default=4)
    parser.add_argument("--residual_blend", type=float, default=0.9)
    parser.add_argument("--lambda_asm", type=float, default=0.1)
    parser.add_argument("--asm_scale", type=int, default=64,
                        help="物理循环一致性损失计算分辨率(0=全分辨率)")
    parser.add_argument("--asm_every", type=int, default=4,
                        help="每隔多少步计算一次物理损失(含解码)")
    parser.add_argument("--bootstrap_every", type=int, default=20,
                        help="每隔多少 epoch 自举更新伪目标(0=关闭)")
    parser.add_argument("--bootstrap_subset", type=int, default=512)
    parser.add_argument("--bootstrap_steps", type=int, default=100)
    parser.add_argument("--ema_beta", type=float, default=0.7)
    parser.add_argument("--init_from_phase4", type=int, default=0,
                        help="1=从阶段4最优权重热启动(默认0,从零训练保持无配对纯粹性)")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_subdir", type=str, default="unpaired_phase5_run1")
    parser.add_argument("--seed", type=int, default=None, help="随机种子(多种子统计用)")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="产物名后缀(消融/多种子运行，如 _noasm/_seed0)，避免覆盖默认结果")
    args = parser.parse_args()
    train(args)
