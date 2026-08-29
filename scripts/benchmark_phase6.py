"""
阶段六：统一基准测试（参数量 / 推理时间 / 透射率形式消融）
文件名：benchmark_phase6.py
功能：
  1. 统计各方法参数量: PSGNet / 条件LDM / 物理引导LDM / 无配对LDM；
  2. 测量单样本(256x256x182)去雾推理时间:
       PSGNet = 一次前向; LDM系 = AE编码 + T步扩散采样 + AE解码;
  3. 消融: 波段自适应透射率 vs 统一透射率(对波段取均值) 的 ASM 重建误差;
  4. 输出 results/phase6_benchmark.json 供 experiment_summary.md 引用。
"""
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import RSHPDIDDataset
from models.psgnet.psgnet_core_phase2 import PSGNet
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder
from models.diffusion.conditional_ldm_phase3 import ConditionalLatentUNet, sample_latent
from models.physics.asm_module_phase4 import AtmosphericScatteringModel

CKPT = Path("checkpoints")


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def time_ldm_inference(model, autoencoder, hazy, timesteps, z_stats, n_rep=3):
    """LDM 系单样本推理: encode + 采样 + decode, 计时取平均(ms)。首次作为 warmup 不计时。"""
    device = hazy.device
    mean, std = z_stats["mean"], z_stats["std"]
    times = []
    with torch.no_grad():
        for rep in range(n_rep + 1):
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            z_h = (autoencoder.encode(hazy) - mean) / std
            z_pred = sample_latent(model, z_h, timesteps, device)
            out = autoencoder.decode(z_pred).clamp(0, 1)
            torch.cuda.synchronize() if device.type == "cuda" else None
            if rep > 0:  # 跳过 warmup
                times.append((time.time() - t0) * 1000)
    return float(np.mean(times))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark on {device}")
    out = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}

    test_ds = RSHPDIDDataset("data/test", "data/clear", max_samples=16)
    in_ch = test_ds.num_bands()
    hazy0 = test_ds[0][0].unsqueeze(0).to(device)
    print(f"Test samples: {len(test_ds)}, bands: {in_ch}")

    # ---------------- PSGNet ----------------
    print("\n[1/4] PSGNet...")
    psgnet = PSGNet(in_ch=in_ch, base_ch=64).to(device)
    ck = torch.load(CKPT / "psgnet_phase2_best.pth", map_location=device)
    psgnet.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    psgnet.eval()
    out["psgnet_params_M"] = round(count_params(psgnet), 2)
    times = []
    with torch.no_grad():
        for rep in range(7):
            t0 = time.time()
            _ = psgnet(hazy0)
            torch.cuda.synchronize() if device.type == "cuda" else None
            if rep >= 2:  # 前2次 warmup
                times.append((time.time() - t0) * 1000)
    out["psgnet_time_ms"] = round(float(np.mean(times)), 1)
    print(f"  params: {out['psgnet_params_M']}M, time: {out['psgnet_time_ms']}ms/sample")

    # ---------------- 自编码器 + 三个 LDM ----------------
    print("\n[2/4] AutoEncoder + LDM variants...")
    ae = HSIAutoEncoder(in_ch=in_ch, latent_ch=16, base_ch=64).to(device)
    ae.load_state_dict(torch.load(CKPT / "ldm_hsi_autoencoder_best_phase3.pth", map_location=device))
    ae.eval()
    out["ae_params_M"] = round(count_params(ae), 2)

    def make_ldm(ckpt_name):
        m = ConditionalLatentUNet(in_ch=16, cond_ch=16, time_emb_dim=256, base_ch=192).to(device)
        sd = torch.load(CKPT / ckpt_name, map_location=device)
        m.load_state_dict(sd)
        m.eval()
        return m

    ldm_specs = {
        "phase3": ("conditional_ldm_phase3_full_best.pth", "conditional_ldm_phase3_full_z_stats.pth", 100),
        "phase4": ("physics_ldm_phase4_best.pth", "physics_ldm_phase4_z_stats.pth", 100),
        "phase5": ("unpaired_phase5_best.pth", "unpaired_phase5_z_stats.pth", 100),
    }
    for key, (ckpt_m, ckpt_s, tsteps) in ldm_specs.items():
        m = make_ldm(ckpt_m)
        zs = torch.load(CKPT / ckpt_s, map_location=device)
        t_ms = time_ldm_inference(m, ae, hazy0, tsteps, zs)
        out[f"{key}_unet_params_M"] = round(count_params(m), 2)
        out[f"{key}_time_ms"] = round(t_ms, 1)
        print(f"  {key}: UNet {out[f'{key}_unet_params_M']}M + AE {out['ae_params_M']}M, "
              f"time: {out[f'{key}_time_ms']}ms/sample ({tsteps} steps)")
        del m
        torch.cuda.empty_cache()

    # ASM 参数量(阶段4/5共用)
    asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=64).to(device)
    asm.load_state_dict(torch.load(CKPT / "asm_phase4.pth", map_location=device))
    asm.eval()
    out["asm_params_M"] = round(count_params(asm), 2)

    # ---------------- 消融: 波段自适应 vs 统一透射率 ----------------
    print("\n[3/4] 消融: 透射率形式(波段自适应 vs 统一)...")
    hazy_all = torch.stack([test_ds[i][0] for i in range(len(test_ds))]).to(device)
    clear_all = torch.stack([test_ds[i][1] for i in range(len(test_ds))]).to(device)
    l_bandwise, l_unified = [], []
    bs = 8
    with torch.no_grad():
        for i in range(0, len(hazy_all), bs):
            hz, cl = hazy_all[i:i+bs], clear_all[i:i+bs]
            t, A = asm(hz, cl)
            recon_bw = asm.reconstruct(cl, t, A)
            l_bandwise.append(F.l1_loss(recon_bw, hz).item())
            # 统一透射率: t 对波段取均值(同一空间位置所有波段共享 t)
            t_uni = t.mean(dim=1, keepdim=True).expand_as(t)
            recon_uni = asm.reconstruct(cl, t_uni, A)
            l_unified.append(F.l1_loss(recon_uni, hz).item())
    out["ablation_transmission"] = {
        "bandwise_adaptive_l1": round(float(np.mean(l_bandwise)), 5),
        "unified_l1": round(float(np.mean(l_unified)), 5),
    }
    print(f"  波段自适应 t 重建 L1: {out['ablation_transmission']['bandwise_adaptive_l1']}")
    print(f"  统一 t       重建 L1: {out['ablation_transmission']['unified_l1']}")

    # t 值统计(供汇报: 不同波段透射率差异)
    with torch.no_grad():
        t, A = asm(hazy_all[:8], clear_all[:8])
    t_band_mean = t.mean(dim=(0, 2, 3)).cpu().numpy()
    out["t_band_stats"] = {
        "short_wave_first5": [round(float(v), 3) for v in t_band_mean[:5]],
        "mid_wave": [round(float(v), 3) for v in t_band_mean[88:93]],
        "long_wave_last5": [round(float(v), 3) for v in t_band_mean[-5:]],
        "t_std_across_bands": round(float(t_band_mean.std()), 4),
    }
    print(f"  波段间 t 标准差: {out['t_band_stats']['t_std_across_bands']}")

    # ---------------- 数据集统计 ----------------
    print("\n[4/4] 数据集统计...")
    train_ds = RSHPDIDDataset("data/train", "data/clear")
    out["dataset"] = {
        "train_pairs": len(train_ds),
        "test_pairs": len(RSHPDIDDataset("data/test", "data/clear")),
        "bands": in_ch,
        "spatial": "256x256",
    }

    with open("results/phase6_benchmark.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> results/phase6_benchmark.json")


if __name__ == "__main__":
    main()
