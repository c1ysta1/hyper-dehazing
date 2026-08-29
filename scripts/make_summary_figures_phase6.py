"""
阶段六：生成汇总可视化图表（方法对比 / 消融 / 无配对差距 / 效率）
文件名：make_summary_figures_phase6.py
输出: results/fig6_*.png，供 experiment_summary.md 与汇报 PPT 引用。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

R = Path("results")
bench = json.load(open(R / "phase6_benchmark.json"))

# ---------------- 数据（2026-08-23 统一口径修订: best ckpt + blend=1.0 + 固定采样种子, 多种子 mean±std） ----------------
import glob

uni = {}
for f in glob.glob(str(R / "metrics_unified" / "*.json")):
    d = json.load(open(f))
    uni[Path(f).stem] = d["blend_1.0"]

def ms(names, key):
    vals = [uni[n][key] for n in names]
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s

P3 = ["phase3_orig", "phase3_seed0", "phase3_seed1"]
P4 = ["phase4_orig", "phase4_seed0", "phase4_seed1"]
P5 = ["phase5_orig", "phase5_seed0", "phase5_seed1"]
P5N = ["phase5_noasm_seed0", "phase5_noasm_seed1"]

methods = ["PSGNet\n(全监督基线)", "条件LDM\n(阶段3)", "物理引导LDM\n(阶段4)", "无配对LDM\n(阶段5)"]
colors = ["#94a3b8", "#60a5fa", "#4ade80", "#a78bfa"]
# PSGNet 为单一确定性前向(无采样随机性), 其余为 3 种子 mean
psnr_m = [28.84, ms(P3, "psnr")[0], ms(P4, "psnr")[0], ms(P5, "psnr")[0]]
psnr_s = [0.0, ms(P3, "psnr")[1], ms(P4, "psnr")[1], ms(P5, "psnr")[1]]
ssim_m = [0.8911, ms(P3, "ssim")[0], ms(P4, "ssim")[0], ms(P5, "ssim")[0]]
ssim_s = [0.0, ms(P3, "ssim")[1], ms(P4, "ssim")[1], ms(P5, "ssim")[1]]
sam_m = [6.13, ms(P3, "sam")[0], ms(P4, "sam")[0], ms(P5, "sam")[0]]
sam_s = [0.0, ms(P3, "sam")[1], ms(P4, "sam")[1], ms(P5, "sam")[1]]

# 图1: 三指标方法对比 (统一口径, 误差棒=种子间标准差)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, mvals, svals, title, better in [
    (axes[0], psnr_m, psnr_s, "PSNR (dB)", "高"),
    (axes[1], ssim_m, ssim_s, "SSIM", "高"),
    (axes[2], sam_m, sam_s, "SAM (°)", "低"),
]:
    bars = ax.bar(methods, mvals, yerr=svals, capsize=4, color=colors,
                  edgecolor="#334155", linewidth=0.8)
    for b, mv, sv in zip(bars, mvals, svals):
        label = f"{mv:.2f}" if sv == 0 else f"{mv:.2f}±{sv:.2f}"
        ax.text(b.get_x() + b.get_width() / 2, mv + sv, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"{title} (越{better}越好)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8.5)
plt.suptitle("方法对比: RSyntHyperPDID 测试集 (统一口径: best ckpt + blend=1.0, LDM系为3种子 mean±std)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(R / "fig6_method_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 图2: 消融1 - 有/无物理约束 (3种子配对散点: 增益不随种子稳定)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
seed_labels = ["orig", "seed0", "seed1"]
for ax, key, unit in [(axes[0], "psnr", "dB"), (axes[1], "ssim", ""), (axes[2], "sam", "°")]:
    v3 = [uni[f"phase3_{s}"][key] for s in seed_labels]
    v4 = [uni[f"phase4_{s}"][key] for s in seed_labels]
    ax.scatter(["无物理约束\n(阶段3)"] * 3, v3, c="#60a5fa", s=90, zorder=3, label="阶段3 各种子")
    ax.scatter(["+ASM物理约束\n(阶段4)"] * 3, v4, c="#4ade80", s=90, zorder=3, label="阶段4 各种子")
    ax.plot([0, 1], [v3, v4], color="#94a3b8", lw=1, alpha=0.6, zorder=2)
    m3, m4 = np.mean(v3), np.mean(v4)
    ax.scatter([0], [m3], c="#1d4ed8", marker="_", s=400, lw=3, zorder=4)
    ax.scatter([1], [m4], c="#15803d", marker="_", s=400, lw=3, zorder=4)
    d = m4 - m3
    ax.set_title(f"{key.upper()} {'(dB)' if unit else ''}: 均值差 {d:+.2f}{unit}\n(3种子配对, 增益不稳定)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks([0, 1])
    ax.set_xlim(-0.6, 1.6)
plt.suptitle("消融实验1: ASM 物理约束 (统一口径 3 种子) -- 配对散点显示增益未超过种子间波动", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(R / "fig6_ablation_physics.png", dpi=150, bbox_inches="tight")
plt.close()

# 图3: 消融2 - 透射率形式
ab = bench["ablation_transmission"]
tb = bench["t_band_stats"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["波段自适应 t\n(逐波段独立)", "统一 t\n(波段共享)"],
            [ab["bandwise_adaptive_l1"], ab["unified_l1"]],
            color=["#4ade80", "#94a3b8"], width=0.5, edgecolor="#334155")
axes[0].set_ylabel("ASM 重建 L1 误差 (越低越好)")
axes[0].set_title(f"消融实验2: 透射率形式\n(自适应比统一低 {100*(1-ab['bandwise_adaptive_l1']/ab['unified_l1']):.1f}%)", fontsize=11)
for i, v in enumerate([ab["bandwise_adaptive_l1"], ab["unified_l1"]]):
    axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)

groups = ["短波前5波段", "中波段", "长波后5波段"]
gvals = [np.mean(tb["short_wave_first5"]), np.mean(tb["mid_wave"]), np.mean(tb["long_wave_last5"])]
bars = axes[1].bar(groups, gvals, color=["#60a5fa", "#4ade80", "#fb923c"], width=0.5, edgecolor="#334155")
for b, v in zip(bars, gvals):
    axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontweight="bold")
axes[1].set_ylabel("平均透射率 t")
axes[1].set_title(f"波段间透射率差异 (跨波段标准差={tb['t_std_across_bands']})\n→ 支持波段自适应设计", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(R / "fig6_ablation_transmission.png", dpi=150, bbox_inches="tight")
plt.close()

# 图4: 无配对训练的价值 (DCP起点 -> 无配对 -> 有配对, 统一口径)
fig, ax = plt.subplots(figsize=(9, 4.5))
stages = ["DCP 物理先验\n(零学习)", "无配对LDM-ASM\n(阶段5消融, 仅雾霾图)", "无配对LDM\n(阶段5, 仅雾霾图)", "物理引导LDM\n(阶段4, 有配对)"]
vals = [12.89, ms(P5N, "psnr")[0], ms(P5, "psnr")[0], ms(P4, "psnr")[0]]
errs = [0.0, ms(P5N, "psnr")[1], ms(P5, "psnr")[1], ms(P4, "psnr")[1]]
cols = ["#f87171", "#c084fc", "#a78bfa", "#4ade80"]
bars = ax.bar(stages, vals, yerr=errs, capsize=4, color=cols, width=0.55, edgecolor="#334155")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f} dB", ha="center", fontweight="bold", fontsize=11)
ax.annotate("", xy=(2, ms(P5, "psnr")[0]), xytext=(0, 12.89),
            arrowprops=dict(arrowstyle="->", color="#a78bfa", lw=2))
ax.annotate(f"+{ms(P5, 'psnr')[0]-12.89:.2f} dB\n(物理自监督+扩散先验)", xy=(1.1, 15.2), ha="center", color="#a78bfa", fontsize=10, fontweight="bold")
ax.annotate("", xy=(3, ms(P4, "psnr")[0]), xytext=(2, ms(P5, "psnr")[0]),
            arrowprops=dict(arrowstyle="->", color="#4ade80", lw=2))
ax.annotate(f"差距 {ms(P4,'psnr')[0]-ms(P5,'psnr')[0]:.2f} dB\n(配对信息价值)", xy=(2.5, 19.3), ha="center", color="#4ade80", fontsize=10, fontweight="bold")
ax.set_ylabel("测试集 PSNR (dB)")
ax.set_ylim(10, 26)
ax.set_title(f"无配对训练效果: 达到有配对方法 {ms(P5,'psnr')[0]/ms(P4,'psnr')[0]*100:.1f}% 的 PSNR 水平 (统一口径)", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", labelsize=8.5)
plt.tight_layout()
plt.savefig(R / "fig6_unpaired_value.png", dpi=150, bbox_inches="tight")
plt.close()

# 图5: 参数量与推理效率
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
labels = ["PSGNet", "条件LDM\n(UNet+AE)", "物理LDM\n(UNet+AE)", "无配对LDM\n(UNet+AE)"]
params = [bench["psgnet_params_M"], bench["phase3_unet_params_M"] + bench["ae_params_M"],
          bench["phase4_unet_params_M"] + bench["ae_params_M"],
          bench["phase5_unet_params_M"] + bench["ae_params_M"]]
times = [bench["psgnet_time_ms"], bench["phase3_time_ms"], bench["phase4_time_ms"], bench["phase5_time_ms"]]
axes[0].bar(labels, params, color=colors, edgecolor="#334155", width=0.55)
for i, v in enumerate(params):
    axes[0].text(i, v, f"{v:.2f}M", ha="center", va="bottom", fontweight="bold")
axes[0].set_ylabel("参数量 (M)")
axes[0].set_title("参数量对比: LDM 系仅为 PSGNet 的 23%", fontsize=11)
axes[0].grid(axis="y", alpha=0.3)
axes[0].tick_params(axis="x", labelsize=9)

axes[1].bar(labels, times, color=colors, edgecolor="#334155", width=0.55)
for i, v in enumerate(times):
    axes[1].text(i, v, f"{v:.1f}ms", ha="center", va="bottom", fontweight="bold")
axes[1].set_ylabel("单样本推理时间 (ms)")
axes[1].set_title(f"推理时间对比 (RTX 5090 实测, 含 warmup; LDM 含100步采样)", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)
axes[1].tick_params(axis="x", labelsize=9)
plt.tight_layout()
plt.savefig(R / "fig6_efficiency.png", dpi=150, bbox_inches="tight")
plt.close()

# 图6: 各阶段训练曲线汇总 (val/test PSNR)
fig, ax = plt.subplots(figsize=(10, 4.5))
h3 = json.load(open("checkpoints/history_conditional_ldm_phase3_full.json"))
h4 = json.load(open("checkpoints/history_physics_ldm_phase4.json"))
h5 = json.load(open("checkpoints/history_unpaired_phase5.json"))
ev3 = [i * 10 for i in range(1, len(h3["val_psnr"]) + 1)]
ev4 = [i * 10 for i in range(1, len(h4["val_psnr"]) + 1)]
ev5 = [i * 10 for i in range(1, len(h5["val_psnr"]) + 1)]
ax.plot(ev3, h3["val_psnr"], "o-", color="#60a5fa", label="阶段3 条件LDM (300ep)", ms=4)
ax.plot(ev4, h4["val_psnr"], "s-", color="#4ade80", label="阶段4 物理引导LDM (120ep)", ms=4)
ax.plot(ev5, h5["val_psnr"], "^-", color="#a78bfa", label="阶段5 无配对LDM (120ep)", ms=4)
ax.axhline(12.89, color="#f87171", ls="--", lw=1.2, label="DCP 物理先验起点 (12.89dB)")
ax.set_xlabel("Epoch")
ax.set_ylabel("验证集 PSNR (dB)")
ax.set_title("各阶段扩散模型验证 PSNR 曲线（同一潜空间/同一评测协议）", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(R / "fig6_all_val_curves.png", dpi=150, bbox_inches="tight")
plt.close()

print("已生成图表:")
for f in sorted(R.glob("fig6_*.png")):
    print(f"  {f}")
