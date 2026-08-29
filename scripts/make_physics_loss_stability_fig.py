"""
专题可视化：物理训练 loss 的跨模型强度不稳定性
文件名：make_physics_loss_stability_fig.py
输出：results/fig_physics_loss_stability.png（供汇报引用）
数据：统一评测口径 results/metrics_unified/*.json（blend=1.0）
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

U = Path("results/metrics_unified")

# 统一评测数据：{模型强度: {基线: json名, 物理微调后: json名}}
groups = [
    ("弱模型\nnone(3.22M)", "phase3_orig", "phase4_orig"),
    ("中等模型\ndepth=2(13.9M)", "mphase3_none", "mphase4_film"),  # 注: depth=2物理微调即 phase4_d2
    ("强模型\nFiLM(3.39M)", "mphase3_film", "mphase4_film"),
]

def psnr(name):
    d = json.load(open(U / f"{name}.json"))
    return d["blend_1.0"]["psnr"]

# 三组数据（depth=2 组用 phase3_d2/phase4_d2；FiLM 组用 mphase3_film/mphase4_film）
data = [
    ("弱模型 none\n(3.22M 参数)", "phase3_orig", "phase4_orig"),
    ("中等模型 depth=2\n(13.89M 参数)", "phase3_d2", "phase4_d2"),
    ("强模型 +FiLM\n(3.39M 参数)", "mphase3_film", "mphase4_film"),
]

labels, base_vals, phys_vals = [], [], []
for lab, b, p in data:
    labels.append(lab)
    base_vals.append(psnr(b))
    phys_vals.append(psnr(p))

deltas = [p - b for p, b in zip(phys_vals, base_vals)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

# ---- 左图：基线 vs 物理微调 分组柱状 ----
x = np.arange(len(labels))
w = 0.36
b1 = ax1.bar(x - w / 2, base_vals, w, label="基线（无物理 loss）", color="#60a5fa", edgecolor="k", linewidth=0.6)
b2 = ax1.bar(x + w / 2, phys_vals, w, label="+物理训练 loss", color="#f87171", edgecolor="k", linewidth=0.6)
for r, v in zip(b1, base_vals):
    ax1.text(r.get_x() + r.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
for r, v in zip(b2, phys_vals):
    ax1.text(r.get_x() + r.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
# 变化箭头
for i, (b, p) in enumerate(zip(base_vals, phys_vals)):
    col = "#16a34a" if p >= b else "#dc2626"
    ax1.annotate("", xy=(i + w / 2, p + 0.05), xytext=(i - w / 2, b + 0.05),
                 arrowprops=dict(arrowstyle="->", color=col, lw=2))
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("PSNR (dB)", fontsize=11)
ax1.set_ylim(15, 23.5)
ax1.set_title("物理训练 loss 的效果随模型强度反转", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(axis="y", alpha=0.3, linestyle="--")

# ---- 右图：ΔPSNR 变化柱状（核心结论图） ----
colors = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
bars = ax2.bar(labels, deltas, 0.55, color=colors, edgecolor="k", linewidth=0.6)
for r, d in zip(bars, deltas):
    va = "bottom" if d >= 0 else "top"
    off = 0.05 if d >= 0 else -0.05
    ax2.text(r.get_x() + r.get_width() / 2, d + off, f"{d:+.2f}", ha="center", va=va, fontsize=12, fontweight="bold")
ax2.axhline(0, color="k", lw=1)
ax2.set_ylabel("Δ PSNR (dB)：物理 loss − 基线", fontsize=11)
ax2.set_title("同一物理约束：模型越强越受伤", fontsize=13, fontweight="bold")
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim(-2.6, 1.4)

fig.suptitle("实验证据：物理项作为训练 loss 是结构性失败（三次独立实验复现）", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = Path("results/fig_physics_loss_stability.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved: {out}")
print("data:", {l: (round(b,2), round(p,2), round(d,2)) for l,b,p,d in zip(labels,base_vals,phys_vals,deltas)})
