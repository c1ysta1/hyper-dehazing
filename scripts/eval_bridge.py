"""
L3 物理桥评测：'清晰→雾' 桥扩散的少步采样评测（方向 B 复活验证）
文件名：eval_bridge.py
对应：实验任务清单_v3.0.md【L3】；fork 自 eval_ddim_guidance.py

评测口径（与全项目统一，保证可比）：
  - best ckpt + blend=1.0（直接输出）+ 固定采样种子 + 全测试集 264 对 + per-image 指标
  - 输入 = checkpoints/latent_cache_phase3 的【归一化】latent（缓存为原始值，须先 (z-mean)/std）
  - 采样输出解码前必须 denorm

对比坐标系（对照结论判定）：
  | 基线 | PSNR(dB) |
  | AE 上限（per-image 口径复测 2026-08-31） | 25.38 |
  | 阶段3+FiLM DDPM100 | 20.95 |
  | 阶段3+FiLM DDPM100 + 测试时引导 s=0.8 | 21.14 |
  | DDIM 少步（§11 已证伪） | ~19 |
桥的判定标准：
  1) bridge100(eta) > 21.14 → 桥本身优于"噪声起点"的扩散；
  2) bridge10/25 ≥ bridge100 − 0.3dB → 少步成立（方向 B 经桥复活）；
  3) bridge1 = "带噪输入回归"极限 → 衡量轨迹插值相对直接回归的价值。

用法（本地 F 盘数据）：
  python scripts/eval_bridge.py --module film \
      --ckpt checkpoints/bridge_ldm_phase3_mfilm_best.pth \
      --z_stats checkpoints/bridge_ldm_phase3_mfilm_z_stats.pth \
      --lat_cache checkpoints/latent_cache_phase3 \
      --test_hazy_dir F:/data/test --clear_dir F:/data/clear \
      --steps 1 5 10 25 50 100 --etas 0 1 --seeds 42 \
      --sigma_max 0.5 --name bridge_mfilm
输出：results/metrics_guided/{name}_bridge.json
"""
import sys
import time
import argparse
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
)
from models.diffusion.bridge_ldm import HazeBridge, sample_bridge


@torch.no_grad()
def eval_one(model, ae, z_stats, z_test_hazy, gt_all, device, bridge,
             n_steps, eta, batch_size, seed=42):
    """单个 (steps, eta, seed) 配置在全测试集上的评测。

    gt_all: 预加载的全量 GT 清晰图张量 (N,C,H,W)（避免多配置重复读盘）。
    """
    z_mean, z_std = z_stats["mean"], z_stats["std"]
    denorm = lambda z: z * z_std + z_mean
    n_test = len(z_test_hazy)
    psnrs, ssims, sams = [], [], []
    for i in range(0, n_test, batch_size):
        z_h = z_test_hazy[i:i + batch_size].to(device)
        gt_raw = gt_all[i:i + batch_size].to(device)
        # 固定起点噪声：同一 (steps, eta) 下不同 seed 可比；
        # 同一 seed 下不同 steps/eta 用同一起点，差异仅来自轨迹
        torch.manual_seed(seed + i)
        start_noise = torch.randn_like(z_h)
        z_pred = sample_bridge(model, z_h, bridge, n_steps,
                                eta=eta, start_noise=start_noise, device=device)
        pred = ae.decode(denorm(z_pred)).clamp(0, 1)
        for j in range(pred.size(0)):
            psnrs.append(compute_psnr(pred[j:j + 1], gt_raw[j:j + 1]).item())
            ssims.append(compute_ssim(pred[j:j + 1], gt_raw[j:j + 1]).item())
            sams.append(compute_sam(pred[j:j + 1], gt_raw[j:j + 1]).item())
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(sams))


def main():
    p = argparse.ArgumentParser(description="L3 物理桥少步采样评测")
    p.add_argument("--ckpt", type=str, required=True, help="桥模型 checkpoint")
    p.add_argument("--z_stats", type=str, required=True, help="latent 归一化统计量")
    p.add_argument("--ae_ckpt", type=str,
                   default="checkpoints/ldm_hsi_autoencoder_best_phase3.pth")
    p.add_argument("--lat_cache", type=str, default="checkpoints/latent_cache_phase3",
                   help="latent 缓存目录（原始未归一化，与训练同源）")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--test_hazy_dir", type=str, default="data/test")
    p.add_argument("--clear_dir", type=str, default="data/clear")
    p.add_argument("--latent_ch", type=int, default=16)
    p.add_argument("--ae_base_ch", type=int, default=64)
    p.add_argument("--base_ch", type=int, default=192)
    p.add_argument("--time_emb_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--module", type=str, default="none",
                   choices=["none", "se", "cbam", "film"])
    p.add_argument("--timesteps", type=int, default=100,
                   help="训练所用的总时间步（桥调度重建依据）")
    p.add_argument("--sigma_max", type=float, default=0.5,
                   help="桥雾端噪声强度（必须与训练时一致）")
    p.add_argument("--sched", type=str, default="linear", choices=["linear", "ddpm"],
                   help="桥调度形状（必须与训练时一致）")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 10, 25, 50, 100],
                   help="桥采样步数列表（1=回归极限，100=全轨迹）")
    p.add_argument("--etas", type=float, nargs="+", default=[0.0, 1.0],
                   help="步进噪声系数列表（0=确定性，1=与训练边缘分布一致）")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  module={args.module}  ckpt={args.ckpt}", flush=True)
    print(f"bridge: sched={args.sched} sigma_max={args.sigma_max} "
          f"timesteps={args.timesteps}", flush=True)

    ds_probe = RSHPDIDDataset(hazy_dir=args.test_hazy_dir, clear_dir=args.clear_dir)
    in_ch = ds_probe.num_bands()
    print(f"num_bands={in_ch}", flush=True)

    ae = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch,
                        base_ch=args.ae_base_ch).to(device).eval()
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device))
    for prm in ae.parameters():
        prm.requires_grad_(False)

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch,
                                  depth=args.depth, module=args.module).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    z_stats = torch.load(args.z_stats, map_location=device)

    bridge = HazeBridge(timesteps=args.timesteps, sigma_max=args.sigma_max,
                        sched=args.sched, device=device)

    lat_cache = Path(args.lat_cache)
    z_test_hazy_raw = torch.load(lat_cache / "z_test_hazy.pt")
    n_test = len(ds_probe)
    assert len(z_test_hazy_raw) == n_test, \
        f"latent 缓存 {len(z_test_hazy_raw)} != 测试集 {n_test}"
    # 缓存保存的是【原始未归一化】latent，模型在归一化空间(std≈1)训练，必须先归一化
    z_test_hazy = (z_test_hazy_raw - z_stats["mean"]) / z_stats["std"]
    # 预加载全量 GT（读盘一次，供所有 steps×eta×seed 配置复用）
    print(f"Preloading {n_test} GT clear images (一次性读盘)...", flush=True)
    t0 = time.time()
    gt_all = torch.stack([ds_probe[k][1] for k in range(n_test)])
    print(f"GT preloaded in {time.time()-t0:.1f}s, shape={tuple(gt_all.shape)}", flush=True)
    print(f"test set: {n_test} pairs, steps={args.steps}, etas={args.etas}, "
          f"seeds={args.seeds}", flush=True)
    print(f"latent 归一化: mean={z_stats['mean']:.4f} std={z_stats['std']:.4f} "
          f"-> 缓存归一化后 std={z_test_hazy.std():.4f}", flush=True)

    results = {}
    for steps in args.steps:
        for eta in args.etas:
            key = f"bridge{steps}_eta{eta}"
            vals = []
            for seed in args.seeds:
                psnr, ssim, sam = eval_one(
                    model, ae, z_stats, z_test_hazy, ds_probe, device, bridge,
                    steps, eta, args.batch_size, seed=seed)
                vals.append((psnr, ssim, sam))
                print(f"  {key} seed={seed}: PSNR={psnr:.2f}  SSIM={ssim:.4f}  "
                      f"SAM={sam:.2f}", flush=True)
            arr = np.array(vals)
            results[key] = {
                "steps": steps, "eta": eta,
                "psnr_mean": float(arr[:, 0].mean()), "psnr_std": float(arr[:, 0].std()),
                "ssim_mean": float(arr[:, 1].mean()), "ssim_std": float(arr[:, 1].std()),
                "sam_mean": float(arr[:, 2].mean()), "sam_std": float(arr[:, 2].std()),
                "per_seed": [{"psnr": float(p), "ssim": float(q), "sam": float(r)}
                             for p, q, r in vals],
            }
            print(f"  {key}: PSNR={arr[:,0].mean():.2f}±{arr[:,0].std():.2f}  "
                  f"SSIM={arr[:,1].mean():.4f}±{arr[:,1].std():.4f}  "
                  f"SAM={arr[:,2].mean():.2f}±{arr[:,2].std():.2f}", flush=True)

    out_dir = Path(__file__).resolve().parent.parent / "results" / "metrics_guided"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_bridge.json"
    with open(out_path, "w") as f:
        json.dump({"meta": vars(args), "results": results, "note": (
            "L3 物理桥少步采样评测。输入为归一化 latent 缓存，解码前已 denorm；"
            "blend=1.0；固定采样种子；steps=1 为'带噪输入回归'极限对照；"
            "eta=0 确定性步进(DDIM 类比)，eta=1 与训练边缘分布一致(DDPM 类比)。"
            "对比基线: AE上限 25.38 / DDPM100 20.95 / 引导后 21.14 / DDIM少步 ~19。"
        )}, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
