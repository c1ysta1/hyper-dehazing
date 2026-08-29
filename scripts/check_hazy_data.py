"""
检查数据是否为雾天数据：分析 Original_data.png / Result_data.png / XiongAn.img 的雾霾特征
文件名：check_hazy_data.py
"""
import numpy as np
from PIL import Image
from pathlib import Path

data_dir = Path(r"d:\Onedrive\OneDrive - neau.edu.cn\高光谱\实验代码\data\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc\xiong_an_ma_ti_wan_cun_hang_kong_gao_etc")

# 1. 分析两个PNG预览图
for name in ["Original_data.png", "Result_data.png"]:
    fp = data_dir / name
    img = np.array(Image.open(fp))
    print(f"\n=== {name} ===")
    print(f"  shape: {img.shape}, dtype: {img.dtype}")
    if img.ndim == 3:
        # RGB统计
        for c, cn in enumerate(["R", "G", "B"][:img.shape[2]]):
            ch = img[:, :, c].astype(np.float32) / 255.0
            print(f"  {cn}: mean={ch.mean():.3f} std={ch.std():.3f} min={ch.min():.3f} max={ch.max():.3f}")
        # 饱和度（低饱和度=雾霾特征）
        if img.shape[2] >= 3:
            rgb = img[:, :, :3].astype(np.float32) / 255.0
            mx = rgb.max(axis=2)
            mn = rgb.min(axis=2)
            sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0)
            # 暗通道（暗通道值高=雾霾特征，He Kaiming DCP）
            dark = mn
            print(f"  饱和度 mean={sat.mean():.3f}  (雾霾图像饱和度低)")
            print(f"  暗通道 mean={dark.mean():.3f}, top1%={np.percentile(dark, 99):.3f}  (雾霾图像暗通道值高)")
            print(f"  对比度(std of gray): {rgb.mean(axis=2).std():.3f}  (雾霾图像对比度低)")

# 2. 分析XiongAn.img的几个波段
print("\n=== XiongAn.img (ENVI, uint16, 1580x3750x256) ===")
fp_img = data_dir / "XiongAn" / "XiongAn.img"
# BSQ格式: band-major。读取3个波段(绿色~60, 红~90, 近红外~120)
H, W, B = 1580, 3750, 256
with open(fp_img, "rb") as f:
    results = {}
    for band_idx in [10, 60, 90, 120, 200]:
        f.seek(band_idx * H * W * 2)  # uint16
        band = np.frombuffer(f.read(H * W * 2), dtype=np.uint16).reshape(H, W).astype(np.float32)
        # 排除mask=0区域
        valid = band[band > 0]
        if len(valid) == 0:
            continue
        p1, p99 = np.percentile(valid, [1, 99])
        results[band_idx] = band
        print(f"  band {band_idx}: mean={valid.mean():.0f} std={valid.std():.0f} "
              f"p1={p1:.0f} p99={p99:.0f} contrast={(p99-p1)/(p99+1e-8):.3f}")

# 3. 暗通道分析（He Kaiming暗通道先验：无雾图像暗通道趋近0，有雾图像暗通道值高）
print("\n=== 暗通道先验分析（XiongAn.img, 采样1/4区域） ===")
# 用绿色/红/近红外三个波段模拟RGB暗通道
if all(k in results for k in [60, 90, 120]):
    step = 4
    rgb = np.stack([results[120][::step, ::step], results[90][::step, ::step], results[60][::step, ::step]], axis=-1)
    mask = rgb.max(axis=-1) > 0  # 有效区域
    rgb_n = rgb / (rgb.max() + 1e-8)
    dark = rgb_n.min(axis=-1)[mask]
    print(f"  采样点数: {mask.sum()}")
    print(f"  暗通道 mean={dark.mean():.3f}, median={np.median(dark):.3f}, p90={np.percentile(dark, 90):.3f}")
    print(f"  (He判据: 无雾图像暗通道均值通常<0.1; 有雾图像>0.3)")

    # 归一化后整体统计
    valid_rgb = rgb[mask]
    print(f"\n  归一化RGB统计:")
    print(f"  亮度 mean={valid_rgb.mean()/valid_rgb.max():.3f}  (雾天图像整体偏亮)")
    gray = valid_rgb.mean(axis=-1)
    print(f"  对比度(gray std/max): {gray.std()/gray.max():.3f}  (雾天图像对比度低)")
