"""
测试时物理引导结果可视化（汇报用）
文件名：make_guidance_fig.py
输出：results/fig_physics_guidance.png
对比：基线(s=0) vs 不同引导强度的 PSNR/SSIM，标注最优 s=0.8（后半段引导）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

G = Path("results/metrics_guided")

def load(name):
    d = json.load(open(G / f"{name}_guided.json"))["results"]
    out = {}
    for k, v in d.items():
        s = float(k.split("=")[1])
        out[s] = v
    return out

# 后半段引导（guide_start_t=50）：合并细扫 + 粗扫
half = {}
half.update(load("mphase3_film"))
half.update(load("mphase3_film_coarse"))
# 全程引导（guide_start_t=100）
full = load("mphase3_film_full")

hs = sorted(half)
h_psnr = [half[s]["psnr"] for s in hs]
h_ssim = [half[s]["ssim"] for s in hs]
fs = sorted(full)
f_psnr = [full[s]["psnr"] for s in fs]
f_ssim = [full[s]["ssim"] for s in fs]

base_psnr = half[0.0]["psnr"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# ---- PSNR 曲线 ----
ax1.axhline(base_psnr, color="#94a3b8", ls="--", lw=1.5, label=f"基线 s=0 ({base_psnr:.2f} dB)")
ax1.plot(hs, h_psnr, "o-", color="#2563eb", lw=2, ms=6, label="后半段引导 (t≤50)")
ax1.plot(fs, f_psnr, "s--", color="#dc2626", lw=2, ms=6, label="全程引导 (t≤100)")
best_s, best_p = 0.8, half[0.8]["psnr"]
ax1.scatter([best_s], [best_p], s=200, facecolor="none", edgecolor="#16a34a", lw=2.5, zorder=5)
ax1.annotate(f"最优 s=0.8\n{best_p:.2f} dB (+{best_p-base_psnr:.2f})",
             xy=(best_s, best_p), xytext=(1.0, base_psnr + 0.02),
             fontsize=11, fontweight="bold", color="#16a34a",
             arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5))
ax1.set_xlabel("引导强度 s", fontsize=12)
ax1.set_ylabel("PSNR (dB)", fontsize=12)
ax1.set_title("测试时物理引导提升去雾精度", fontsize=13, fontweight="bold")
ax1.legend(fontsize=10, loc="lower right")
ax1.grid(alpha=0.3, ls="--")

# ---- SSIM 曲线 ----
ax2.axhline(half[0.0]["ssim"], color="#94a3b8", ls="--", lw=1.5, label=f"基线 s=0 ({half[0.0]['ssim']:.4f})")
ax2.plot(hs, h_ssim, "o-", color="#2563eb", lw=2, ms=6, label="后半段引导 (t≤50)")
ax2.plot(fs, f_ssim, "s--", color="#dc2626", lw=2, ms=6, label="全程引导 (t≤100)")
ax2.axvline(0.8, color="#16a34a", ls=":", lw=1.5)
ax2.text(0.82, ax2.get_ylim()[0], "s=0.8", color="#16a34a", fontsize=10, fontweight="bold", va="bottom")
ax2.set_xlabel("引导强度 s", fontsize=12)
ax2.set_ylabel("SSIM", fontsize=12)
ax2.set_title("过大 s 损伤结构（过冲）", fontsize=13, fontweight="bold")
ax2.legend(fontsize=10, loc="lower left")
ax2.grid(alpha=0.3, ls="--")

fig.suptitle("物理引导替代物理训练 loss：保住『物理优化扩散』且稳定提升",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = Path("results/fig_physics_guidance.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print(f"基线 s=0: {base_psnr:.2f}  最优 s=0.8(后半段): {best_p:.2f}  (+{best_p-base_psnr:.2f} dB)")
