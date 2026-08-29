"""
验证XiongAn.img与Original_data.png的关系，并检查短波段的雾霾特征
文件名：verify_hazy_relation.py
"""
import numpy as np
from PIL import Image
from pathlib import Path

data_dir = Path(r"d:\Onedrive\OneDrive - neau.edu.cn\高光谱\实验代码\data\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc")

H, W, B = 1580, 3750, 256
fp_img = data_dir / "XiongAn" / "XiongAn.img"

# 读取RGB三个波段 (B=36:~468nm, G=72:~562nm, R=120:~691nm，参考default bands {120,72,36})
bands = {}
with open(fp_img, "rb") as f:
    for band_idx in [36, 72, 120]:
        f.seek(band_idx * H * W * 2)
        bands[band_idx] = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W).astype(np.float32)

R, G, Bc = bands[120], bands[72], bands[36]
rgb = np.stack([R, G, Bc], axis=-1)
mask = rgb.max(axis=-1) > 0
print(f"有效像素比例: {mask.mean():.3f}")

# 归一化渲染
vmax = np.percentile(rgb[mask], 99.5)
rgb_n = np.clip(rgb / vmax, 0, 1)
rgb_n[~mask] = 0
img_render = (rgb_n * 255).astype(np.uint8)

# 保存渲染图
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
Image.fromarray(img_render).save(out_dir / "xiongan_rgb_render.png")

# 与Original_data.png对比（缩放到相同尺寸后计算相关性）
orig = np.array(Image.open(data_dir / "Original_data.png").convert("RGB").resize((W, H)))
render_small = img_render

# 有效区域相关性
for c, cn in enumerate("RGB"):
    a = orig[:, :, c].astype(np.float32)[mask]
    b = render_small[:, :, c].astype(np.float32)[mask]
    if a.std() > 0 and b.std() > 0:
        corr = np.corrcoef(a, b)[0, 1]
        print(f"  {cn}通道相关性: {corr:.3f}")

# 波长依赖的对比度分析（雾霾的物理指纹）
print("\n=== 全波段对比度剖面（雾霾指纹：短波低对比度） ===")
with open(fp_img, "rb") as f:
    for band_idx in [0, 20, 40, 60, 80, 100, 130, 160, 190, 220, 250]:
        f.seek(band_idx * H * W * 2)
        band = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W).astype(np.float32)
        valid = band[mask]
        if len(valid) == 0 or valid.max() == 0:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        contrast = (p99 - p1) / (p99 + 1e-8)
        wl = 391.3 + band_idx * 2.4  # 近似波长
        print(f"  band {band_idx:3d} (~{wl:.0f}nm): mean={valid.mean():6.0f} std={valid.std():6.0f} "
              f"p99/p1={p99/p1:5.2f} contrast={contrast:.3f}")

# 保存波段对比度曲线
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

contrasts, wls, means = [], [], []
with open(fp_img, "rb") as f:
    for band_idx in range(0, 256, 8):
        f.seek(band_idx * H * W * 2)
        band = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W).astype(np.float32)
        valid = band[mask]
        if len(valid) == 0:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        contrasts.append((p99 - p1) / (p99 + 1e-8))
        wls.append(391.3 + band_idx * 2.4)
        means.append(valid.mean())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(wls, contrasts, "o-")
axes[0].set_xlabel("Wavelength (nm)")
axes[0].set_ylabel("Contrast (p99-p1)/p99")
axes[0].set_title("Band contrast profile (haze fingerprint)")
axes[0].grid(alpha=0.3)
axes[1].plot(wls, means, "o-")
axes[1].set_xlabel("Wavelength (nm)")
axes[1].set_ylabel("Mean DN")
axes[1].set_title("Mean brightness per band")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "xiongan_haze_analysis.png", dpi=120)
print(f"\n分析图已保存: results/xiongan_haze_analysis.png")
print(f"RGB渲染图已保存: results/xiongan_rgb_render.png")
