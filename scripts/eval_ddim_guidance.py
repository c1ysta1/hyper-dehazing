"""
方向 B 前端落地：DDIM 少步采样 + 测试时物理引导（DCP）叠加评测
文件名：eval_ddim_guidance.py

背景/定位：
  交接文档方向 B = 压缩扩散采样损失（≈3.8dB）。核心手段之一是 DDIM/DPM-Solver
  少步采样；且采样的关键瓶颈已知除步数外还有“条件注入通路”（FiLM 已补+2.61），
  故本脚本在【当前最优 FiLM 链路】上评测“少步 + 物理引导”的组合，而非旧 ddim_sampler_improved.py
  所连的历史模型（physics_ldm_phase4.pth / 无 module / 现场 encode / 无 denorm）。

评测口径（与 eval_physics_guidance.py 完全对齐，保证可比）：
  - best ckpt（--module film）+ blend=1.0（直接输出）+ 固定采样种子 + 全测试集 264 对
  - 输入 = 预计算的【归一化】latent 缓存 checkpoints/latents_phase4/z_test_hazy.pt
  - 采样输出在归一化 latent 空间(std~1)，解码前必须 ae.decode(denorm(z))（漏 denorm 只剩 ~16.5dB）
  - s=0 退化为纯 DDIM 基线（可视为对 100 步 DDPM 的步数压缩对照）

用法示例：
  python scripts/eval_ddim_guidance.py --module film \
      --ckpt checkpoints/conditional_ldm_phase3_full_mfilm_best.pth \
      --z_stats checkpoints/conditional_ldm_phase3_full_mfilm_z_stats.pth \
      --ddim_steps 10 25 50 --guidance_s 0 0.8 --guide_start_t 50 --seeds 42 100 200 \
      --name mphase3_film_ddim
输出：results/metrics_guided/{name}_ddim.json（含每 (steps, s) 的 PSNR/SSIM/SAM 与多种子 mean±std）
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import RSHPDIDDataset
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.simple_ddpm_phase3 import DDPM
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    _dark_channel,
)


def denorm_factory(z_mean, z_std):
    return lambda z: z * z_std + z_mean


def ddim_timesteps(ddpm_timesteps, n_steps):
    """从训练时间步中挑出 n_steps 个边界，返回降序 t 列表（用于 ddim 步进）。"""
    # 含首/末边界的均匀网格：保证包含 0（最后去噪彻底）与 T-1（初始噪声）
    idx = np.linspace(0, ddpm_timesteps - 1, n_steps + 1).round().astype(int).tolist()
    # 去重（步数多于可用时间步时 linspace 会重叠）
    uniq = []
    for v in idx:
        if v not in uniq:
            uniq.append(v)
    # (cur, prev) 按去噪顺序递减：prev 为当前 t 的上一大步（更小）
    ts = uniq[::-1]  # 从大到小
    return ts


def ddim_step(model, z, z_hazy, ddpm, t, t_prev, device, eta=0.0):
    """单步 DDIM（eta=0 为确定性），返回 x0_pred 与 z_next。"""
    t_batch = torch.full((z.size(0),), t, device=device, dtype=torch.long)
    with torch.no_grad():
        predicted_noise = model(z, z_hazy, t_batch)
    alpha_cumprod = ddpm.alphas_cumprod[t].float()
    alpha_cumprod_prev = ddpm.alphas_cumprod[t_prev].float()

    y = 1 - alpha_cumprod
    yp = 1 - alpha_cumprod_prev

    sqrt_ac = torch.sqrt(alpha_cumprod)
    # 注意：绝不能对 x0_pred clamp 到 [-1,1]——那是【图像空间】DDIM 的做法。
    # 本模型在归一化 latent 空间(std≈1.19, mean≈0.29)采样，真实 latent 大量超出
    # [-1,1]；clamp 每步都截断信号能量，误差随步数累积：实测 10 步 18.81dB、
    # 25 步 18.17dB、50 步崩塌到 9.09dB。标准 DDIM 公式无需 clamp。
    x0_pred = (z - torch.sqrt(y) * predicted_noise) / sqrt_ac

    sigma = eta * torch.sqrt(
        (yp / y) * (1 - alpha_cumprod / alpha_cumprod_prev))
    dir_xt = torch.sqrt(torch.clamp(yp - sigma ** 2, min=0.0)) * predicted_noise
    noise = sigma * torch.randn_like(z) if eta > 0 else torch.zeros_like(z)
    z_next = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt + noise
    return x0_pred, z_next


def apply_physics_guidance(model_ae, denorm, x0_pred, z_next, guidance_s,
                           dark_kernel=15):
    """对 z_next 施加 DCP 引导：梯度过暗通道，归一化后按 s 修正（与 sample_latent_guided 一致）。

    用本步的 x0_pred（干净 latent 估计）解出暗通道损失，对 z_next 直接修正。
    """
    if guidance_s <= 0:
        return z_next
    # 注意：eval_one 外层有 @torch.no_grad()，必须在局部开启梯度域，
    # 否则 x0_g.requires_grad_(True) 的叶子不会构建计算图，autograd 报
    # "element 0 of tensors does not require grad"。
    with torch.enable_grad():
        x0_g = x0_pred.detach().requires_grad_(True)
        img_hat = model_ae.decode(denorm(x0_g))
        dark = _dark_channel(img_hat, kernel=dark_kernel)
        L_phys = dark.mean()
        grad = torch.autograd.grad(L_phys, x0_g)[0]
    gnorm = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
    return z_next - guidance_s * (grad / gnorm)


def sample_ddim_guided(model, ae, z_hazy, ddpm, denorm, n_steps,
                       guidance_s=0.0, guide_start_t=None, eta=0.0,
                       dark_kernel=15, z_init=None, device='cpu'):
    """DDIM 少步采样，可选叠加测试时 DCP 物理引导。"""
    ts = ddim_timesteps(ddpm.timesteps, n_steps)
    # ts 为降序，下一大步 prev 即后一个（更小的）t；ts[-1]==0 时 prev=0
    z = torch.randn_like(z_hazy) if z_init is None else z_init.to(device)
    z = z.to(device)
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else ts[-1]
        x0_pred, z_next = ddim_step(model, z, z_hazy, ddpm, t, t_prev,
                                    device, eta=eta)
        # 物理引导：只在后半段（t <= guide_start_t）且 s>0 时
        if guidance_s > 0 and t <= guide_start_t:
            z_next = apply_physics_guidance(ae, denorm, x0_pred, z_next,
                                            guidance_s, dark_kernel)
        z = z_next.detach()
    return z


@torch.no_grad()
def eval_one(model, ae, z_stats, z_test_hazy, test_ds, device,
             n_steps, guidance_s, guide_start_t, timesteps,
             batch_size, seed=42, eta=0.0):
    z_mean, z_std = z_stats["mean"], z_stats["std"]
    denorm = denorm_factory(z_mean, z_std)
    ddpm = DDPM(timesteps=timesteps, device=device)
    n_test = len(test_ds)
    psnrs, ssims, sams = [], [], []
    for i in range(0, n_test, batch_size):
        z_h = z_test_hazy[i:i + batch_size].to(device)
        gt_raw = torch.stack(
            [test_ds[k][1] for k in range(i, min(i + batch_size, n_test))]).to(device)
        # 固定初始噪声：同一 n_steps 下不同 s 用同一 z_init，差异仅来自引导
        torch.manual_seed(seed + i)
        z_init = torch.randn_like(z_h)
        z_pred = sample_ddim_guided(
            model, ae, z_h, ddpm, denorm, n_steps,
            guidance_s=guidance_s, guide_start_t=guide_start_t,
            eta=eta, z_init=z_init, device=device)
        pred = ae.decode(denorm(z_pred)).clamp(0, 1)
        for j in range(pred.size(0)):
            psnrs.append(compute_psnr(pred[j:j + 1], gt_raw[j:j + 1]).item())
            ssims.append(compute_ssim(pred[j:j + 1], gt_raw[j:j + 1]).item())
            sams.append(compute_sam(pred[j:j + 1], gt_raw[j:j + 1]).item())
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(sams))


def main():
    p = argparse.ArgumentParser(description="DDIM 少步 + 测试时物理引导评测（FiLM 链路，方向 B）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--z_stats", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str,
                   default="checkpoints/ldm_hsi_autoencoder_best_phase3.pth")
    p.add_argument("--lat_cache", type=str, default="checkpoints/latents_phase4")
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
                   help="训练所用的总时间步（DDPM schedule 重建依据）")
    p.add_argument("--ddim_steps", type=int, nargs="+", default=[50],
                   help="DDIM 采样步数列表（如 10 25 50）")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM 随机性，0=确定性")
    p.add_argument("--guidance_s", type=float, nargs="+", default=[0, 0.8])
    p.add_argument("--guide_start_t", type=int, default=50,
                   help="仅当 t <= 该值时物理引导（默认后半段）")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  module={args.module}  ckpt={args.ckpt}", flush=True)

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

    lat_cache = Path(args.lat_cache)
    z_test_hazy_raw = torch.load(lat_cache / "z_test_hazy.pt")
    n_test = len(ds_probe)
    assert len(z_test_hazy_raw) == n_test, \
        f"latent 缓存 {len(z_test_hazy_raw)} != 测试集 {n_test}"
    # 缓存保存的是【原始未归一化】latent（train_conditional_ldm_phase3_full.py 在
    # 归一化前落盘），而模型在归一化空间(std≈1)训练，必须先做 (z-mean)/std 归一化，
    # 采样输出再 denorm 解码。若跳过归一化，输入分布偏离训练分布，PSNR 会大幅劣化。
    z_test_hazy = (z_test_hazy_raw - z_stats["mean"]) / z_stats["std"]
    print(f"test set: {n_test} pairs, ddpm_timesteps={args.timesteps}, "
          f"ddim_steps={args.ddim_steps}, seeds={args.seeds}", flush=True)
    print(f"latent 归一化: mean={z_stats['mean']:.4f} std={z_stats['std']:.4f} "
          f"-> 缓存归一化后 std={z_test_hazy.std():.4f}", flush=True)

    results = {}
    for steps in args.ddim_steps:
        for s in args.guidance_s:
            key = f"ddim{steps}_s{s}"
            vals = []
            for seed in args.seeds:
                psnr, ssim, sam = eval_one(
                    model, ae, z_stats, z_test_hazy, ds_probe, device,
                    steps, s, args.guide_start_t, args.timesteps,
                    args.batch_size, seed=seed, eta=args.eta)
                vals.append((psnr, ssim, sam))
                print(f"  {key} seed={seed}: PSNR={psnr:.2f}  SSIM={ssim:.4f}  "
                      f"SAM={sam:.2f}", flush=True)
            arr = np.array(vals)
            results[key] = {
                "ddim_steps": steps, "guidance_s": s,
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
    out_path = out_dir / f"{args.name}_ddim.json"
    with open(out_path, "w") as f:
        json.dump({"meta": vars(args), "results": results, "note": (
            "输入为归一化 latent 缓存，解码前已 denorm；blend=1.0；固定采样种子；"
            "s=0 为纯 DDIM 少步基线（可比 checkpoints/metrics_guided 100 步 DDPM 结果）。",
        )}, f, indent=2, ensure_ascii=False)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()