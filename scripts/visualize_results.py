"""
实验效果可视化汇总（生成汇报用图表）
文件名：visualize_results.py
生成：
  1. 各模型训练损失曲线
  2. 方法对比柱状图（含参照基线）
  3. PSGNet 去雾效果可视化（雾图/去雾结果/GT + 误差图）
  4. 光谱曲线对比（同一像素）
  5. 采样器/模型容量对比图
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
from rshpdid_dataset import load_rshpdid_tensors

from models.psgnet.psgnet_core_phase2 import PSGNet
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.conditional_ldm_phase3 import compute_sam, compute_ssim

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# 波段波长（用于光谱曲线）：优先读取 RSyntHyperPDID 自带波长文件
_WL_FILE = Path("data/wavelengths_synthyper.npy")
WAVELENGTHS = np.load(_WL_FILE) if _WL_FILE.exists() else None
WL_START, WL_STEP = 391.3, 2.4
BANDS = 182


def load_json(fp):
    with open(fp) as f:
        return json.load(f)


# ============================================================
# 图1：训练损失曲线
# ============================================================
def plot_training_curves():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PSGNet
    h = load_json("checkpoints/history_psgnet_phase2.json")
    ax = axes[0]
    ax.plot(h["train_loss"], label="train loss", color="tab:blue")
    if "test_loss" in h and h["test_loss"]:
        ax.plot(h["test_loss"], label="test loss", color="tab:orange")
    ax.set_title("PSGNet 训练曲线 (100轮)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # 自编码器
    h = load_json("checkpoints/history_ldm_hsi_phase3.json")
    ax = axes[1]
    key = "train_psnr" if "train_psnr" in h else ("test_psnr" if "test_psnr" in h else None)
    if "train_loss" in h:
        ax.plot(h["train_loss"], label="train loss", color="tab:green")
    if key:
        ax2 = ax.twinx()
        ax2.plot(h[key], label=key, color="tab:red")
        ax2.set_ylabel("PSNR (dB)", color="tab:red")
    ax.set_title("自编码器训练 (50轮)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # 扩散模型对比
    ax = axes[2]
    colors = {"conditional_ldm_phase3": ("tab:blue", "条件LDM"),
              "physics_ldm_phase4": ("tab:red", "物理引导LDM"),
              "unpaired_phase5": ("tab:purple", "无配对LDM")}
    for name, (c, label) in colors.items():
        fp = f"checkpoints/history_{name}.json"
        if Path(fp).exists():
            h = load_json(fp)
            if "diff_loss" in h:
                ax.plot(h["diff_loss"], label=label, color=c)
    ax.set_title("扩散模型训练对比 (200轮)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Diffusion Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "fig1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图1: fig1_training_curves.png")


# ============================================================
# 图2：方法对比柱状图（含基线）
# ============================================================
def plot_method_comparison():
    base = load_json(RESULTS / "eval_baselines.json")
    methods = [
        ("雾霾输入\n(不处理)", base["baseline_hazy_input"], "gray"),
        ("均值图\n(平凡解)", base["baseline_mean_image"], "gray"),
        ("暗通道先验\n(DCP)", base["baseline_dcp"], "gray"),
        ("无配对LDM", base["unpaired_ldm_200ep"], "tab:purple"),
        ("条件LDM", base["cond_ldm_200ep"], "tab:blue"),
        ("物理引导LDM", base["physics_ldm_200ep"], "tab:red"),
        ("PSGNet", base["psgnet_100ep"], "tab:green"),
    ]
    names = [m[0] for m in methods]
    psnrs = [m[1]["psnr"] for m in methods]
    ssims = [m[1]["ssim"] for m in methods]
    sams = [m[1]["sam"] for m in methods]
    colors = [m[2] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title, better in [
        (axes[0], psnrs, "PSNR (dB) ↑", "high"),
        (axes[1], ssims, "SSIM ↑", "high"),
        (axes[2], sams, "SAM (°) ↓", "low"),
    ]:
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
        best_idx = int(np.argmax(vals)) if better == "high" else int(np.argmin(vals))
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("方法对比（含参照基线）— 69个测试patch", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig2_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图2: fig2_method_comparison.png")


# ============================================================
# 图3：PSGNet 去雾可视化
# ============================================================
def to_rgb(img, brighten=1.0):
    """(C,H,W) -> RGB, 选R/G/B波段"""
    B = img.shape[0]
    r_idx, g_idx, b_idx = min(int(B * 0.72), B - 1), int(B * 0.46), int(B * 0.20)
    rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
    return np.clip(rgb * brighten, 0, 1)


def plot_psgnet_dehazing():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading test set from npy...")
    test = load_rshpdid_tensors("data/test", "data/clear", max_samples=16)
    hazy_all, clear_all = test["hazy"], test["clear"]

    model = PSGNet(in_ch=hazy_all.shape[1], base_ch=64).to(device)
    ckpt = torch.load("checkpoints/psgnet_phase2_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    n_vis = 4
    idxs = np.linspace(0, len(hazy_all) - 1, n_vis, dtype=int)
    fig, axes = plt.subplots(n_vis, 4, figsize=(14, 3.2 * n_vis))

    with torch.no_grad():
        preds = model(hazy_all[idxs].to(device)).clamp(0, 1).cpu()

    for r in range(n_vis):
        h, p, c = hazy_all[idxs[r]], preds[r], clear_all[idxs[r]]
        err = (p - c).abs().mean(dim=0)
        row = [
            (to_rgb(h.numpy(), 1.3), "雾霾输入"),
            (to_rgb(p.numpy()), f"PSGNet输出 (PSNR={compute_psnr(preds[r:r+1], clear_all[idxs[r]:idxs[r]+1]):.2f}dB)"),
            (to_rgb(c.numpy()), "清晰参考(GT)"),
            (plt.cm.viridis(err / max(err.max(), 1e-6)), "误差图 |pred-GT|"),
        ]
        for c_i, (im, title) in enumerate(row):
            ax = axes[r, c_i] if n_vis > 1 else axes[c_i]
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.suptitle("PSGNet 去雾效果（合成雾霾测试集）", fontsize=13, y=1.005)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig3_psgnet_dehazing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图3: fig3_psgnet_dehazing.png")


# ============================================================
# 图4：光谱曲线对比（同一像素位置）
# ============================================================
def plot_spectral_curves():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading test set from npy...")
    test = load_rshpdid_tensors("data/test", "data/clear", max_samples=16)
    hazy_all, clear_all = test["hazy"], test["clear"]

    model = PSGNet(in_ch=hazy_all.shape[1], base_ch=64).to(device)
    ckpt = torch.load("checkpoints/psgnet_phase2_best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    with torch.no_grad():
        preds = model(hazy_all[:3].to(device)).clamp(0, 1).cpu()

    wl = WAVELENGTHS if WAVELENGTHS is not None else WL_START + np.arange(hazy_all.shape[1]) * WL_STEP
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        # 选植被特征明显的像素（红边位置）
        img = clear_all[i]
        n_b = img.shape[0]
        veg = (img[int(n_b * 0.62)] - img[int(n_b * 0.35)])  # 近红外-红, (H,W)
        y, x = np.unravel_index(np.argmax(veg), veg.shape)
        axes[i].plot(wl, hazy_all[i, :, y, x], "b--", label="雾霾输入", alpha=0.7)
        axes[i].plot(wl, preds[i, :, y, x], "r-", label="PSGNet去雾")
        axes[i].plot(wl, clear_all[i, :, y, x], "g-", label="清晰GT")
        axes[i].set_title(f"样本{i+1} 植被像素光谱 (y={y},x={x})")
        axes[i].set_xlabel("波长 (nm)")
        axes[i].set_ylabel("反射率")
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)
    plt.suptitle("光谱曲线对比（植被像素：红边~720nm与近红外平台）", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(RESULTS / "fig4_spectral_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图4: fig4_spectral_curves.png")


# ============================================================
# 图5：消融实验（训练轮数 / 采样器 / 模型容量）
# ============================================================
def plot_ablations():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 5.1 训练轮数影响
    epochs_data = {
        "5轮": {"psnr": 26.40, "ssim": 0.6879, "sam": 6.61},
        "50轮": {"psnr": 26.27, "ssim": 0.7395, "sam": 7.90},
        "200轮": {"psnr": 26.56, "ssim": 0.7820, "sam": 7.75},
    }
    ax = axes[0]
    xs = list(epochs_data.keys())
    ax.plot(xs, [epochs_data[k]["psnr"] for k in xs], "o-", label="PSNR", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(xs, [epochs_data[k]["ssim"] for k in xs], "s-", label="SSIM", color="tab:red")
    ax.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("SSIM")
    ax.set_title("条件LDM: 训练轮数影响")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.grid(alpha=0.3)

    # 5.2 采样器对比
    d = load_json(RESULTS / "ddim_comparison_improved.json")
    ax = axes[1]
    names = list(d.keys())
    psnr_vals = [d[k]["psnr"] for k in names]
    ssim_vals = [d[k]["ssim"] for k in names]
    x = np.arange(len(names))
    ax.bar(x - 0.15, psnr_vals, 0.3, label="PSNR", color="tab:blue")
    ax3 = ax.twinx()
    ax3.bar(x + 0.15, ssim_vals, 0.3, label="SSIM", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax3.set_ylabel("SSIM")
    ax.set_title("采样器对比 (200轮checkpoint)")
    ax.grid(alpha=0.3, axis="y")

    # 5.3 模型容量
    cap = [
        ("小模型\n1.42M,100步", 26.56, 0.7820),
        ("小模型\n1.42M,1000步", 26.09, 0.7108),
        ("大模型\n5.44M,1000步", 26.12, 0.7123),
        ("大物理模型\n5.44M,1000步", 26.07, 0.7099),
    ]
    ax = axes[2]
    xs = [c[0] for c in cap]
    psnr_vals = [c[1] for c in cap]
    ssim_vals = [c[2] for c in cap]
    x = np.arange(len(xs))
    ax.bar(x - 0.15, psnr_vals, 0.3, label="PSNR", color="tab:green")
    ax4 = ax.twinx()
    ax4.bar(x + 0.15, ssim_vals, 0.3, label="SSIM", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(xs, fontsize=8)
    ax.set_ylabel("PSNR (dB)")
    ax4.set_ylabel("SSIM")
    ax.set_ylim(25.8, 26.7)
    ax.set_title("模型容量与时间步（数据不足→过拟合）")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS / "fig5_ablations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图5: fig5_ablations.png")


# ============================================================
# 图6：真实数据雾霾证据（波段对比度 + 预览图对比）
# ============================================================
def plot_haze_evidence():
    fig = plt.figure(figsize=(15, 5))

    # 6.1 波段对比度曲线（已保存数据重绘）
    data_dir = Path(r"data/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc")
    img_fp = data_dir / "XiongAn" / "XiongAn.img"
    H, W = 1580, 3750
    contrasts, wls = [], []
    with open(img_fp, "rb") as f:
        for band_idx in range(0, 256, 6):
            f.seek(band_idx * H * W * 2)
            band = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W).astype(np.float32)
            valid = band[band > 0]
            if len(valid) == 0:
                continue
            p1, p99 = np.percentile(valid, [1, 99])
            contrasts.append((p99 - p1) / (p99 + 1e-8))
            wls.append(WL_START + band_idx * WL_STEP)

    ax1 = fig.add_subplot(131)
    ax1.plot(wls, contrasts, "o-", color="tab:red", markersize=3)
    ax1.annotate("391nm 对比度=0.13\n(被雾淹没)", xy=(391, 0.127), xytext=(430, 0.25),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax1.annotate("775nm 对比度=0.74\n(雾影响小)", xy=(775, 0.740), xytext=(800, 0.55),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax1.set_xlabel("波长 (nm)")
    ax1.set_ylabel("对比度 (p99-p1)/p99")
    ax1.set_title("XiongAn数据雾霭证据:\n对比度随波长上升=瑞利散射指纹")
    ax1.grid(alpha=0.3)

    # 6.2/6.3 预览图对比
    from PIL import Image
    orig = np.array(Image.open(data_dir / "Original_data.png").convert("RGB").resize((600, 300)))
    result = np.array(Image.open(data_dir / "Result_data.png").convert("RGB").resize((600, 300)))
    ax2 = fig.add_subplot(132)
    ax2.imshow(orig)
    ax2.set_title(f"Original_data.png\n暗通道={0.497:.2f}(重雾)")
    ax2.axis("off")
    ax3 = fig.add_subplot(133)
    ax3.imshow(result)
    ax3.set_title(f"Result_data.png\n暗通道={0.254:.2f}(似去雾结果)")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS / "fig6_haze_evidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("图6: fig6_haze_evidence.png")


if __name__ == "__main__":
    plot_training_curves()
    plot_method_comparison()
    plot_psgnet_dehazing()
    plot_spectral_curves()
    plot_ablations()
    plot_haze_evidence()
    print("\n全部图表已生成到 results/")
