"""
测试时物理引导评测（暗通道先验 DPS 式引导）
文件名：eval_physics_guidance.py
背景：物理训练 loss 已被证伪（三次实验模型越强越伤），改为测试时引导，训练完全不变。
用法：
  python scripts/eval_physics_guidance.py --module film \
    --ckpt checkpoints/conditional_ldm_phase3_full_mfilm_best.pth \
    --z_stats checkpoints/conditional_ldm_phase3_full_mfilm_z_stats.pth \
    --guidance_s 0 0.25 0.5 1.0 2.0 --guide_start_t 50 --name mphase3_film
输出：results/metrics_guided/{name}_guided.json（含每个 s 的 PSNR/SSIM/SAM）
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import RSHPDIDDataset
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    sample_latent_guided,
)


def denorm_factory(z_mean, z_std):
    """z_stats 的 mean/std 为 float 标量，直接闭包使用。"""
    return lambda z: z * z_std + z_mean


@torch.no_grad()
def encode_batch(ae, x, device):
    return ae.encode(x.to(device))


def eval_one_s(model, ae, z_stats, z_test_hazy, test_ds, device, s,
               guide_start_t, timesteps, batch_size, seed=42):
    """与 eval_unified_phase6b 完全对齐：latent 缓存输入 + denorm + clamp。

    关键：扩散在“归一化 latent”空间训练，输入 z_hazy 必须直接用缓存
    z_test_hazy（已归一化），采样输出再 denorm 后解码。现场 ae.encode 得到
    的是未归一化 latent，会偏离训练分布导致输出全坏（PSNR=-inf）。
    seed 为采样种子，用于多种子稳定性验证。
    """
    z_mean, z_std = z_stats["mean"], z_stats["std"]
    denorm = denorm_factory(z_mean, z_std)
    n_test = len(test_ds)
    psnrs, ssims, sams = [], [], []
    for i in range(0, n_test, batch_size):
        z_h = z_test_hazy[i:i + batch_size].to(device)
        gt_raw = torch.stack(
            [test_ds[k][1] for k in range(i, min(i + batch_size, n_test))]).to(device)
        # 固定初始噪声：不同 s 用同一 z_init，采样差异仅来自引导本身
        torch.manual_seed(seed + i)
        z_init = torch.randn_like(z_h)
        z_pred = sample_latent_guided(
            model, z_h, timesteps, device, ae,
            denorm=denorm, guidance_s=s, guide_start_t=guide_start_t,
            z_init=z_init)
        # 采样输出在归一化 latent 空间(std~1)，须 denorm 回 AE 训练分布(std~10.6)再解码
        pred = ae.decode(denorm(z_pred)).clamp(0, 1)
        for j in range(pred.size(0)):
            psnrs.append(compute_psnr(pred[j:j+1], gt_raw[j:j+1]).item())
            ssims.append(compute_ssim(pred[j:j+1], gt_raw[j:j+1]).item())
            sams.append(compute_sam(pred[j:j+1], gt_raw[j:j+1]).item())
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(sams))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--z_stats", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str,
                   default="checkpoints/ldm_hsi_autoencoder_best_phase3.pth")
    p.add_argument("--lat_cache", type=str,
                   default="checkpoints/latents_phase4",
                   help="归一化 latent 缓存目录（含 z_test_hazy.pt，须与训练一致）")
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
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--guidance_s", type=float, nargs="+", default=[0, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--guide_start_t", type=int, default=50,
                   help="仅当 t <= 该值时引导（默认后半段）")
    p.add_argument("--seeds", type=int, nargs="+", default=[42],
                   help="采样种子列表，多种子时输出 mean±std")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  module={args.module}  ckpt={args.ckpt}")

    # in_ch 须从数据集动态获取（RSyntHyperPDID 为 182 波段，与 AE ckpt 一致）
    ds_probe = RSHPDIDDataset(hazy_dir=args.test_hazy_dir, clear_dir=args.clear_dir)
    in_ch = ds_probe.num_bands()
    print(f"num_bands={in_ch}")

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
    # mean/std 为 float 标量，denorm 闭包直接引用即可

    # 与 eval_unified 一致：用预计算的归一化 latent 缓存作为扩散输入
    lat_cache = Path(args.lat_cache)
    z_test_hazy = torch.load(lat_cache / "z_test_hazy.pt")
    n_test = len(ds_probe)
    assert len(z_test_hazy) == n_test, \
        f"latent 缓存 {len(z_test_hazy)} != 测试集 {n_test}"
    print(f"test set: {n_test} pairs, batch={args.batch_size}, timesteps={args.timesteps}")

    results = {}
    for s in args.guidance_s:
        vals = []
        for seed in args.seeds:
            psnr, ssim, sam = eval_one_s(
                model, ae, z_stats, z_test_hazy, ds_probe, device,
                s, args.guide_start_t, args.timesteps, args.batch_size, seed=seed)
            vals.append((psnr, ssim, sam))
            print(f"  seed={seed} s={s}: PSNR={psnr:.2f}  SSIM={ssim:.4f}  SAM={sam:.2f}", flush=True)
        arr = np.array(vals)
        results[f"s={s}"] = {
            "psnr_mean": float(arr[:, 0].mean()), "psnr_std": float(arr[:, 0].std()),
            "ssim_mean": float(arr[:, 1].mean()), "ssim_std": float(arr[:, 1].std()),
            "sam_mean": float(arr[:, 2].mean()), "sam_std": float(arr[:, 2].std()),
            "per_seed": [{"psnr": p, "ssim": q, "sam": r} for p, q, r in vals],
        }
        print(f"  s={s}: PSNR={arr[:,0].mean():.2f}±{arr[:,0].std():.2f}  "
              f"SSIM={arr[:,1].mean():.4f}±{arr[:,1].std():.4f}  "
              f"SAM={arr[:,2].mean():.2f}±{arr[:,2].std():.2f}", flush=True)

    out_dir = Path(__file__).resolve().parent.parent / "results" / "metrics_guided"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}_guided.json"
    with open(out_path, "w") as f:
        json.dump({"meta": vars(args), "results": results}, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
