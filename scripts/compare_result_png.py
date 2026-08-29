"""
对比 Result_data.png 与 XiongAn.img / Original_data.png 的空间结构关系
文件名：compare_result_png.py
"""
import numpy as np
from PIL import Image
from pathlib import Path

data_dir = Path(r"d:\Onedrive\OneDrive - neau.edu.cn\高光谱\实验代码\data\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc")

orig = Image.open(data_dir / "Original_data.png").convert("RGB")
result = Image.open(data_dir / "Result_data.png").convert("RGB")
print(f"Original: {orig.size}, Result: {result.size}")

# 统一缩放到小尺寸比较结构
size = (512, 256)
orig_s = np.array(orig.resize(size)).astype(np.float32) / 255.0
result_s = np.array(result.resize(size)).astype(np.float32) / 255.0

# 灰度结构相关性
og = orig_s.mean(axis=2)
rg = result_s.mean(axis=2)
corr = np.corrcoef(og.flatten(), rg.flatten())[0, 1]
print(f"灰度结构相关性(缩放到512x256): {corr:.3f}")

# 分块相关性（检查是否部分对齐）
H, W = og.shape
bh, bw = H // 4, W // 4
print("\n分块相关性矩阵 (Original vs Result):")
for i in range(4):
    row = []
    for j in range(4):
        a = og[i*bh:(i+1)*bh, j*bw:(j+1)*bw].flatten()
        b = rg[i*bh:(i+1)*bh, j*bw:(j+1)*bw].flatten()
        row.append(np.corrcoef(a, b)[0, 1])
    print("  " + "  ".join(f"{v:+.2f}" for v in row))

# 差异统计
diff = np.abs(orig_s - result_s)
print(f"\n像素差异: mean={diff.mean():.3f}")
print(f"Original 暗通道: {orig_s.min(axis=2).mean():.3f}")
print(f"Result   暗通道: {result_s.min(axis=2).mean():.3f}")
print(f"Original 饱和度: {(orig_s.max(axis=2)-orig_s.min(axis=2)).mean():.3f}")
print(f"Result   饱和度: {(result_s.max(axis=2)-result_s.min(axis=2)).mean():.3f}")

# 保存并排对比图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0, 0].imshow(orig_s)
axes[0, 0].set_title(f"Original_data.png {orig.size}")
axes[0, 0].axis("off")
axes[0, 1].imshow(result_s)
axes[0, 1].set_title(f"Result_data.png {result.size}")
axes[0, 1].axis("off")
# 直方图对比
axes[1, 0].hist(og.flatten(), bins=50, alpha=0.5, label="Original", density=True)
axes[1, 0].hist(rg.flatten(), bins=50, alpha=0.5, label="Result", density=True)
axes[1, 0].set_xlabel("Gray value")
axes[1, 0].legend()
axes[1, 0].set_title("Gray histogram")
# RGB散点
axes[1, 1].scatter(og.flatten()[::37], rg.flatten()[::37], s=1, alpha=0.3)
axes[1, 1].set_xlabel("Original gray")
axes[1, 1].set_ylabel("Result gray")
axes[1, 1].set_title(f"Scatter (corr={corr:.3f})")
plt.tight_layout()
plt.savefig("results/orig_vs_result_comparison.png", dpi=120)
print("\n对比图保存: results/orig_vs_result_comparison.png")
