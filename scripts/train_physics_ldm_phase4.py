"""
阶段四：物理引导扩散训练（物理一致性损失 + 实时进度日志/可视化）
文件名：train_physics_ldm_phase4.py
功能：
  1. 用 RSHPDIDDataset 懒加载 + 流式编码，避免一次性加载 182 波段全量数据造成内存爆炸；
  2. 在归一化 latent 空间训练条件 DDPM（条件 = 雾霾 latent）；
  3. 物理一致性损失作用于"模型输出的清晰图"：
       本步 t 下的 one-step 去噪估计 x0 -> 反归一化 -> 解码 J_pred -> ASM 重建
       L_phys = || hazy - (J_pred*t + A*(1-t)) ||_1
       使生成结果满足大气散射退化，从而引导扩散输出物理可解释；
  4. 总损失 = 扩散损失 + λ * 物理一致性损失；
  5. 训练过程实时写日志 + 周期生成采样可视化（供 webui 实时显示）；
  6. 训练结束后在 data/test 上计算 PSNR/SSIM/SAM。
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


def run_latent_stats(dataset, autoencoder, device, batch_size=16):
    """单遍流式计算 train clear latent 的 mean/std，用于归一化。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cnt, s, s2 = 0, 0.0, 0.0
    with torch.no_grad():
        for hazy, clear in loader:
            z = autoencoder.encode(clear.to(device)).float()
            s += z.sum().item()
            s2 += (z ** 2).sum().item()
            cnt += z.numel()
    mean = s / cnt
    var = max(s2 / cnt - mean ** 2, 1e-8)
    return mean, float(np.sqrt(var))


def stream_encode(dataset, autoencoder, device, batch_size=16, desc="", mean=None, std=None):
    """流式编码；若给 mean/std 则归一化到 latent 空间。只保留 latent 张量（内存友好）。"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    z_hazy, z_clear = [], []
    n = len(dataset)
    with torch.no_grad():
        for i, (hazy, clear) in enumerate(loader):
            zh = autoencoder.encode(hazy.to(device)).float()
            zc = autoencoder.encode(clear.to(device)).float()
            if mean is not None and std is not None:
                zh = (zh - mean) / std
                zc = (zc - mean) / std
            z_hazy.append(zh.cpu())
            z_clear.append(zc.cpu())
            if (i + 1) % 25 == 0 or (i + 1) * batch_size >= n:
                print(f"  [{desc}] 编码中 {min((i+1)*batch_size, n)}/{n}")
    return torch.cat(z_hazy), torch.cat(z_clear)


class PhysLatentDataset(Dataset):
    """每步从 npy 懒读 raw hazy（供 ASM 物理损失），latent 从内存缓存取（索引对齐）。"""
    def __init__(self, z_hazy, z_clear, raw_ds):
        self.z_hazy = z_hazy
        self.z_clear = z_clear
        self.raw_ds = raw_ds
        assert len(raw_ds) == len(z_clear)

    def __len__(self):
        return len(self.z_clear)

    def __getitem__(self, idx):
        hazy_raw = self.raw_ds[idx][0]
        return self.z_hazy[idx], self.z_clear[idx], hazy_raw


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
            (hazy_rgb[r], "雾霾输入"), (pred_rgb[r], "物理引导去雾"), (clear_rgb[r], "清晰参考(GT)"),
        ]):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    plt.suptitle("阶段4 物理引导LDM去雾效果（训练中实时采样）", fontsize=13)
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
    log_line = f"Physics-guided LDM on {device}, lambda_phys={args.lambda_phys}, epochs={args.epochs}"
    print(log_line)

    print("Building datasets...")
    train_ds = RSHPDIDDataset(args.train_hazy_dir, args.clear_dir, max_samples=args.max_train_samples)
    test_ds = RSHPDIDDataset(args.test_hazy_dir, args.clear_dir)
    print(f"Train pairs: {len(train_ds)}, Test pairs: {len(test_ds)}")

    in_ch = train_ds.num_bands()
    H, W = train_ds.spatial_size()
    print(f"Bands: {in_ch}, size: {H}x{W}")

    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.ae_ckpt) if args.ae_ckpt else Path(args.checkpoint_dir) / "ldm_hsi_autoencoder_best_phase3.pth"
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"自编码器 checkpoint 不存在: {ae_ckpt}，请先运行 ldm_hsi_phase3.py")
    print(f"Loading AE from {ae_ckpt}")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # ---------------- Latent 标准化 + 磁盘缓存（避免重复编码 220GB npy） ----------------
    # latent_ch=16 复用历史缓存目录，其他通道数独立目录
    lat_tag = "" if args.latent_ch == 16 else f"_lc{args.latent_ch}"
    lat_cache = Path(args.checkpoint_dir) / f"latents_phase4{lat_tag}"
    lat_cache.mkdir(parents=True, exist_ok=True)
    cf = lambda n: lat_cache / f"{n}.pt"

    if (not args.no_cache and cf("z_train_hazy").exists() and cf("z_train_clear").exists()
            and cf("z_test_hazy").exists() and cf("z_test_clear").exists() and cf("hazy_low").exists()):
        print("Loading latents from cache...")
        z_train_hazy = torch.load(cf("z_train_hazy"))
        z_train_clear = torch.load(cf("z_train_clear"))
        z_test_hazy = torch.load(cf("z_test_hazy"))
        z_test_clear = torch.load(cf("z_test_clear"))
        hazy_low_train = torch.load(cf("hazy_low"))
        s = torch.load(cf("z_stats"))
        z_mean, z_std = s["mean"], s["std"]
        print(f"Cache loaded: train={tuple(z_train_clear.shape)}, text_={tuple(z_test_clear.shape)}, hazy_low={tuple(hazy_low_train.shape)}")
    else:
        print("Building latent + low-res hazy cache (once)...")
        t0 = time.time()
        train_enc_loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
        z_zh, z_zc, lows = [], [], []
        with torch.no_grad():
            for hazy, clear in train_enc_loader:
                zh = autoencoder.encode(hazy.to(device)).float()
                zc = autoencoder.encode(clear.to(device)).float()
                low = F.interpolate(hazy.to(device), size=(64, 64), mode="area").float().cpu()
                z_zh.append(zh.cpu()); z_zc.append(zc.cpu()); lows.append(low)
        z_train_hazy = torch.cat(z_zh); z_train_clear = torch.cat(z_zc)
        hazy_low_train = torch.cat(lows)
        z_mean = float(z_train_clear.reshape(-1).mean())
        z_std = float(z_train_clear.reshape(-1).std().clamp(min=1e-8))
        z_train_hazy = (z_train_hazy - z_mean) / z_std
        z_train_clear = (z_train_clear - z_mean) / z_std
        test_enc_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
        z_th, z_tc = [], []
        with torch.no_grad():
            for hazy, clear in test_enc_loader:
                z_th.append(autoencoder.encode(hazy.to(device)).float().cpu())
                z_tc.append(autoencoder.encode(clear.to(device)).float().cpu())
        z_test_hazy = torch.cat(z_th); z_test_clear = torch.cat(z_tc)
        z_test_hazy = (z_test_hazy - z_mean) / z_std
        z_test_clear = (z_test_clear - z_mean) / z_std
        for name, obj in [("z_train_hazy", z_train_hazy), ("z_train_clear", z_train_clear),
                          ("z_test_hazy", z_test_hazy), ("z_test_clear", z_test_clear),
                          ("hazy_low", hazy_low_train), ("z_stats", {"mean": float(z_mean), "std": float(z_std)})]:
            torch.save(obj, cf(name))
        print(f"Cache built+saved in {time.time()-t0:.1f}s -> {lat_cache}")
    z_stats = {"mean": float(z_mean), "std": float(z_std)}

    def denorm(z):
        return z * z_std + z_mean

    vis_hazy = torch.stack([test_ds[i][0] for i in range(min(args.viz_samples, len(test_ds)))])
    vis_clear = torch.stack([test_ds[i][1] for i in range(min(args.viz_samples, len(test_ds)))])
    val_clear_raw = torch.stack([test_ds[i][1] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    val_hazy_raw = torch.stack([test_ds[i][0] for i in range(min(args.val_samples, len(test_ds)))]).to(device)
    val_hazy = z_test_hazy[:args.val_samples].to(device)

    train_loader = DataLoader(
        TensorDataset(z_train_hazy, z_train_clear, hazy_low_train),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device)
    asm = None
    if args.lambda_phys > 0:
        asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=args.ae_base_ch).to(device)
    print(f"Conditional U-Net parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (depth={args.depth})")
    if asm:
        print(f"ASM parameters: {sum(p.numel() for p in asm.parameters())/1e6:.2f}M")

    params = list(model.parameters())
    if asm:
        params += list(asm.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) / (args.log_subdir + sfx)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_log.txt"
    sample_file = log_dir / "sample_latest.png"

    def log_print(msg, echo=True):
        if echo:
            print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    log_print(log_line)

    history = {"diff_loss": [], "phys_loss": [], "total_loss": [],
               "val_psnr": [], "val_ssim": [], "val_sam": [], "epoch_time": []}
    best_psnr = -1.0
    total_t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        if asm:
            asm.train()
        ep_t0 = time.time()
        d_losses, p_losses, t_losses = [], [], []
        for step, (z_hazy, z_clear, hz_low) in enumerate(train_loader):
            z_hazy, z_clear = z_hazy.to(device), z_clear.to(device)
            hz_low = hz_low.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (z_clear.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_clear, t)
            pred_noise = model(z_noisy, z_hazy, t)
            diff_loss = F.mse_loss(pred_noise, noise)

            # 解码+物理一致性较昂贵，每隔 phys_every 步才计算一次（hazy_low 已预降采样到内存）
            if asm and (step % args.phys_every == 0):
                ac = ddpm.alphas_cumprod[t][:, None, None, None]
                z0_est = (z_noisy - (1.0 - ac).clamp(min=1e-8).sqrt() * pred_noise) / ac.clamp(min=1e-8).sqrt()
                J_pred = autoencoder.decode(denorm(z0_est)).clamp(0, 1)
                if args.stop_grad_pred:
                    J_pred = J_pred.detach()
                if args.phys_scale > 0 and J_pred.shape[-1] != args.phys_scale:
                    jp_low = F.interpolate(J_pred, size=(args.phys_scale, args.phys_scale), mode="area")
                else:
                    jp_low = J_pred
                phys_loss, t_map, A_map = asm.physics_loss(hz_low, jp_low)
                total_loss = diff_loss + args.lambda_phys * phys_loss
            else:
                phys_loss = None
                total_loss = diff_loss
            total_loss.backward()
            # 梯度裁剪：防止 depth=2 大模型训练不稳定（参考阶段3 depth=2 loss 爆炸坍塌教训）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            d_losses.append(diff_loss.item())
            p_losses.append(phys_loss.item() if (asm and phys_loss is not None) else 0.0)
            t_losses.append(total_loss.item())

        avg_d = float(np.mean(d_losses))
        avg_p = float(np.mean(p_losses))
        avg_t = float(np.mean(t_losses))
        ep_time = time.time() - ep_t0
        history["diff_loss"].append(avg_d)
        history["phys_loss"].append(avg_p)
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
                torch.save(model.state_dict(), checkpoint_dir / f"physics_ldm_phase4{sfx}_best.pth")
                torch.save(z_stats, checkpoint_dir / f"physics_ldm_phase4{sfx}_z_stats.pth")
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_t:.4f} "
                      f"diff={avg_d:.4f} phys={avg_p:.4f} "
                      f"test_psnr={psnr:.2f} test_ssim={ssim:.4f} test_sam={sam:.2f} "
                      f"time={ep_time:.1f}s elapsed={time.time()-total_t0:.0f}s")
        else:
            log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_t:.4f} "
                      f"diff={avg_d:.4f} phys={avg_p:.4f} "
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

    torch.save(model.state_dict(), checkpoint_dir / f"physics_ldm_phase4{sfx}.pth")
    if asm and not sfx:
        torch.save(asm.state_dict(), checkpoint_dir / "asm_phase4.pth")
    with open(checkpoint_dir / f"history_physics_ldm_phase4{sfx}.json", "w") as f:
        json.dump(history, f, indent=2)
    log_print(f"Training complete in {time.time()-total_t0:.0f}s. Best val PSNR: {best_psnr:.2f}dB")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["diff_loss"], color="tab:blue", label="diffusion loss")
    if asm:
        ax.plot(history["phys_loss"], color="tab:orange", label="physics loss")
    ax.plot(history["total_loss"], color="tab:green", ls="--", label="total loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("阶段4 物理引导LDM 训练损失曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    if history["val_psnr"]:
        ax2 = ax.twinx()
        ev_epochs = [e for e in range(1, args.epochs + 1) if (e % args.eval_every == 0 or e == args.epochs)]
        ax2.plot(ev_epochs[:len(history["val_psnr"])], history["val_psnr"], "o-",
                 color="tab:red", label="val PSNR")
        ax2.set_ylabel("Val PSNR (dB)", color="tab:red")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig_phase4_physics_loss_curve{sfx}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Evaluating on full test set (重载最优 checkpoint, 统一 best 口径)...")
    best_ckpt = checkpoint_dir / f"physics_ldm_phase4{sfx}_best.pth"
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
    with open(out_dir / f"metrics_physics_ldm_phase4{sfx}.json", "w") as f:
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
    parser.add_argument("--lambda_phys", type=float, default=0.1)
    parser.add_argument("--phys_scale", type=int, default=64,
                        help="物理一致性损失计算分辨率(0=全分辨率),降低它可大幅提速")
    parser.add_argument("--phys_every", type=int, default=4,
                        help="每隔多少步计算一次物理损失(含解码),降低可避免解码成为瓶颈")
    parser.add_argument("--no_cache", action="store_true",
                        help="忽略 latent 磁盘缓存,强制重新编码")
    parser.add_argument("--stop_grad_pred", type=int, default=0,
                        help="对 J_pred 是否切断梯度(1=切断,只训ASM)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_subdir", type=str, default="physics_ldm_phase4_run1")
    parser.add_argument("--seed", type=int, default=None, help="随机种子(多种子统计用)")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="产物名后缀(消融/多种子运行，如 _seed0)，避免覆盖默认结果")
    args = parser.parse_args()
    train(args)
