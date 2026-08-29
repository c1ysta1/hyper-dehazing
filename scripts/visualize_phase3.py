"""
阶段3 结果可视化：条件LDM去雾效果汇总
文件名：visualize_phase3.py
生成：
  1. 去雾对比图（雾霾输入 | LDM去雾 | GT | 误差图）
  2. 光谱曲线对比（同一像素）
  3. 指标报告（PSNR/SSIM/SAM）
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
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
    sample_latent,
)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

WL_START, WL_STEP = 391.3, 2.4


def to_rgb(img, brighten=1.0):
    """(C,H,W) numpy [0,1] -> RGB，取 R/G/B 波段。"""
    B = img.shape[0]
    r_idx = min(int(B * 0.72), B - 1)
    g_idx = int(B * 0.46)
    b_idx = int(B * 0.20)
    rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
    return np.clip(rgb * brighten, 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = RSHPDIDDataset("data/test", "data/clear")
    in_ch = test_ds.num_bands()

    # 模型
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=16, base_ch=64).to(device)
    autoencoder.load_state_dict(torch.load("checkpoints/ldm_hsi_autoencoder_best_phase3.pth",
                                           map_location=device))
    autoencoder.eval()

    model = ConditionalLatentUNet(in_ch=16, cond_ch=16, time_emb_dim=128, base_ch=128).to(device)
    model.load_state_dict(torch.load("checkpoints/conditional_ldm_phase3_full_best.pth",
                                     map_location=device))
    model.eval()

    n_vis = 6
    print(f"Loading {n_vis} test samples...")
    hazy_list, clear_list = [], []
    for i in range(n_vis):
        h, c = test_ds[i]
        hazy_list.append(h)
        clear_list.append(c)
    hazy = torch.stack(hazy_list).to(device)
    clear = torch.stack(clear_list).to(device)

    # 采样
    print("Sampling (DDPM reverse, 100 steps)...")
    with torch.no_grad():
        z_h = autoencoder.encode(hazy)
        z_pred = sample_latent(model, z_h, 100, device)
        pred = autoencoder.decode(z_pred).clamp(0, 1).cpu().numpy()
    hazy_np = hazy.cpu().numpy()
    clear_np = clear.cpu().numpy()

    # 指标（逐样本）
    metrics = {"psnr": [], "ssim": [], "sam": []}
    for i in range(n_vis):
        p = torch.from_numpy(pred[i:i+1]).to(device)
        g = torch.from_numpy(clear_np[i:i+1]).to(device)
        metrics["psnr"].append(compute_psnr(p, g).item())
        metrics["ssim"].append(compute_ssim(p, g).item())
        metrics["sam"].append(compute_sam(p, g).item())
    print(f"Visualization samples avg: PSNR={np.mean(metrics['psnr']):.2f}dB "
          f"SSIM={np.mean(metrics['ssim']):.4f} SAM={np.mean(metrics['sam']):.2f}°")

    # ---- 图1：去雾对比 ----
    fig, axes = plt.subplots(n_vis, 4, figsize=(15, 3.4 * n_vis))
    for r in range(n_vis):
        err = np.abs(pred[r] - clear_np[r]).mean(axis=0)
        row = [
            (to_rgb(hazy_np[r], 1.3), f"雾霾输入 (PSNR={metrics['psnr'][r]:.2f}dB)"),
            (to_rgb(pred[r]), f"LDM去雾 (PSNR={metrics['psnr'][r]:.2f}dB)"),
            (to_rgb(clear_np[r]), "清晰参考(GT)"),
            (plt.cm.viridis(err / max(err.max(), 1e-6)), "误差图 |pred-GT|"),
        ]
        for c_i, (im, title) in enumerate(row):
            ax = axes[r, c_i]
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.suptitle("阶段3 条件LDM去雾效果（data/test 测试集）", fontsize=14, y=1.005)
    plt.tight_layout()
    p1 = RESULTS / "fig_phase3_dehazing.png"
    plt.savefig(p1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"图1: {p1}")

    # ---- 图2：光谱曲线对比 ----
    wl = WL_START + np.arange(in_ch) * WL_STEP
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i in range(n_vis):
        img = clear_np[i]
        n_b = img.shape[0]
        veg = img[int(n_b * 0.62)] - img[int(n_b * 0.35)]  # 近红外-红
        y, x = np.unravel_index(np.argmax(veg), veg.shape)
        ax = axes[i // 3, i % 3]
        ax.plot(wl, hazy_np[i, :, y, x], "b--", label="雾霾输入", alpha=0.7, linewidth=1.2)
        ax.plot(wl, pred[i, :, y, x], "r-", label="LDM去雾", linewidth=1.5)
        ax.plot(wl, clear_np[i, :, y, x], "g-", label="清晰GT", linewidth=1.5)
        ax.set_title(f"样本{i+1} 像素(y={y},x={x})")
        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("反射率")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("光谱曲线对比（植被像素：LDM去雾 vs GT）", fontsize=13, y=1.01)
    plt.tight_layout()
    p2 = RESULTS / "fig_phase3_spectral_curves.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"图2: {p2}")

    # ---- 指标报告 ----
    report = {
        "samples": [{"file": test_ds.names[i],
                     "psnr": float(metrics["psnr"][i]),
                     "ssim": float(metrics["ssim"][i]),
                     "sam": float(metrics["sam"][i])} for i in range(n_vis)],
        "avg": {k: float(np.mean(v)) for k, v in metrics.items()},
    }
    with open(RESULTS / "visualization_phase3_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("指标报告已保存: results/visualization_phase3_report.json")


if __name__ == "__main__":
    main()
