"""
统一评测脚本（阶段6修订，修复评测口径）
文件名：eval_unified_phase6b.py
修复的口径问题：
  1) 统一评测各阶段 best checkpoint（而非 last epoch 权重）；
  2) blend=1.0（纯模型输出）为主口径，与 PSGNet 直接预测公平可比；
     blend=0.9（旧口径，混入 10% 雾霾图）仅作参考对照；
  3) 固定采样种子，DDPM 采样结果可复现；
  4) 多种子/消融 run 通过 --ckpt/--z_stats 指定，结果统一写入 results/metrics_unified/。
用法示例：
  python scripts/eval_unified_phase6b.py \
      --ckpt checkpoints/physics_ldm_phase4_best.pth \
      --z_stats checkpoints/physics_ldm_phase4_z_stats.pth --name phase4_orig
"""
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import RSHPDIDDataset
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    sample_latent,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="UNet best checkpoint 路径")
    parser.add_argument("--z_stats", type=str, required=True, help="对应 z_stats 路径")
    parser.add_argument("--name", type=str, required=True, help="结果名（输出文件名）")
    parser.add_argument("--test_hazy_dir", type=str, default="data/test")
    parser.add_argument("--clear_dir", type=str, default="data/clear")
    parser.add_argument("--latent_ch", type=int, default=16)
    parser.add_argument("--ae_ckpt", type=str, default=None,
                        help="自编码器 checkpoint 路径（默认 checkpoints/ldm_hsi_autoencoder_best_phase3.pth）")
    parser.add_argument("--lat_cache", type=str, default=None,
                        help="latent 缓存目录（默认 checkpoints/latents_phase4[_lc{N}]）")
    parser.add_argument("--ae_base_ch", type=int, default=64)
    parser.add_argument("--base_ch", type=int, default=192)
    parser.add_argument("--depth", type=int, default=1, help="条件 U-Net 下采样级数(须与训练一致)")
    parser.add_argument("--module", type=str, default="none",
                        choices=["none", "se", "cbam", "film"],
                        help="容量增强模块(须与训练一致)")
    parser.add_argument("--time_emb_dim", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--blends", type=str, default="1.0,0.9", help="逗号分隔的 blend 列表")
    parser.add_argument("--seed", type=int, default=42, help="采样种子（固定以复现）")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--out_dir", type=str, default="results/metrics_unified")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[unified-eval] {args.name} on {device}, ckpt={args.ckpt}")

    test_ds = RSHPDIDDataset(args.test_hazy_dir, args.clear_dir)
    n_test = len(test_ds)
    in_ch = test_ds.num_bands()

    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.ae_ckpt) if args.ae_ckpt else Path(args.checkpoint_dir) / "ldm_hsi_autoencoder_best_phase3.pth"
    print(f"Loading AE from {ae_ckpt}")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()

    s = torch.load(args.z_stats, map_location="cpu")
    z_mean, z_std = float(s["mean"]), float(s["std"])

    def denorm(z):
        return z * z_std + z_mean

    # 复用阶段4 latent 缓存作为测试输入（与训练一致的预处理）
    lat_tag = "" if args.latent_ch == 16 else f"_lc{args.latent_ch}"
    lat_cache = Path(args.lat_cache) if args.lat_cache else Path(args.checkpoint_dir) / f"latents_phase4{lat_tag}"
    z_test_hazy = torch.load(lat_cache / "z_test_hazy.pt")
    assert len(z_test_hazy) == n_test, f"latent 缓存数量 {len(z_test_hazy)} != 测试集 {n_test}"

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    blends = [float(b) for b in args.blends.split(",")]
    metrics = {b: {"psnr": [], "ssim": [], "sam": []} for b in blends}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n_test, args.batch_size):
            z_h = z_test_hazy[i:i + args.batch_size].to(device)
            hazy_raw = torch.stack([test_ds[k][0] for k in range(i, min(i + args.batch_size, n_test))]).to(device)
            gt_raw = torch.stack([test_ds[k][1] for k in range(i, min(i + args.batch_size, n_test))]).to(device)
            z_p = denorm(sample_latent(model, z_h, args.timesteps, device))
            pred = autoencoder.decode(z_p).clamp(0, 1)
            for b in blends:
                pb = b * pred + (1 - b) * hazy_raw
                for j in range(pb.size(0)):
                    p, g = pb[j:j + 1], gt_raw[j:j + 1]
                    metrics[b]["psnr"].append(compute_psnr(p, g).item())
                    metrics[b]["ssim"].append(compute_ssim(p, g).item())
                    metrics[b]["sam"].append(compute_sam(p, g).item())
            if (i // args.batch_size) % 10 == 0:
                print(f"  progress {min(i + args.batch_size, n_test)}/{n_test} "
                      f"({time.time() - t0:.0f}s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "ckpt": args.ckpt, "seed": args.seed,
               "timesteps": args.timesteps, "device": str(device),
               "eval_time_s": round(time.time() - t0, 1)}
    for b in blends:
        avg = {k: float(np.mean(v)) for k, v in metrics[b].items()}
        summary[f"blend_{b}"] = avg
        print(f"[{args.name}] blend={b}: PSNR={avg['psnr']:.2f}dB "
              f"SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / f"{args.name}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_dir / (args.name + '.json')}")


if __name__ == "__main__":
    main()
