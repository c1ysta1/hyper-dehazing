"""
评价体系补充：计算各参照基线分数
文件名：eval_baselines.py
功能：
  1. 雾霾输入直接作为输出（"什么都不做"基线）
  2. 全图均值作为输出（"均值预测"基线）
  3. 已训练模型的分数
  统一在同一测试集上计算，便于横向解释。
"""
import sys
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.diffusion.ldm_hsi_phase3 import compute_psnr
from models.diffusion.conditional_ldm_phase3 import compute_sam, compute_ssim


def main():
    test_data = torch.load('data/synthetic_haze/test/test_tensor.pth')
    hazy = test_data['hazy'].float()
    clear = test_data['clear'].float()
    print(f"Test set: {hazy.shape[0]} samples, {hazy.shape[1]} bands, {hazy.shape[2]}x{hazy.shape[3]}")

    results = {}

    # 基线1：雾霾图直接当输出（什么都不做）
    p, s, a = [], [], []
    for i in range(len(hazy)):
        p.append(compute_psnr(hazy[i:i+1], clear[i:i+1]).item())
        s.append(compute_ssim(hazy[i:i+1], clear[i:i+1]).item())
        a.append(compute_sam(hazy[i:i+1], clear[i:i+1]).item())
    results['baseline_hazy_input'] = {'psnr': float(np.mean(p)), 'ssim': float(np.mean(s)), 'sam': float(np.mean(a))}
    print(f"雾霾输入直接当输出: PSNR={results['baseline_hazy_input']['psnr']:.2f} "
          f"SSIM={results['baseline_hazy_input']['ssim']:.4f} SAM={results['baseline_hazy_input']['sam']:.2f}")

    # 基线2：全数据集均值图当输出
    mean_img = clear.mean(dim=0, keepdim=True)
    p, s, a = [], [], []
    for i in range(len(clear)):
        p.append(compute_psnr(mean_img, clear[i:i+1]).item())
        s.append(compute_ssim(mean_img, clear[i:i+1]).item())
        a.append(compute_sam(mean_img, clear[i:i+1]).item())
    results['baseline_mean_image'] = {'psnr': float(np.mean(p)), 'ssim': float(np.mean(s)), 'sam': float(np.mean(a))}
    print(f"均值图当输出:       PSNR={results['baseline_mean_image']['psnr']:.2f} "
          f"SSIM={results['baseline_mean_image']['ssim']:.4f} SAM={results['baseline_mean_image']['sam']:.2f}")

    # 基线3：暗通道先验去雾（经典物理方法，非学习）
    # 简化DCP: t = 1 - 0.95*min_channel / A, A取图像最亮0.1%像素均值
    p, s, a = [], [], []
    for i in range(len(hazy)):
        I = hazy[i:i+1]
        # 各波段暗通道（局部最小，这里用全局最小近似）
        dark = I.min(dim=1, keepdim=True)[0]
        # 大气光估计
        flat = I.mean(dim=1).flatten()
        topk = torch.topk(flat, max(1, int(flat.numel() * 0.001)))[0]
        A = topk.mean().clamp(0.1, 1.0)
        # 透射率
        t = 1 - 0.95 * dark / (A + 1e-8)
        t = t.clamp(0.1, 1.0)
        # 恢复
        J = (I - A) / (t + 1e-8) + A
        J = J.clamp(0, 1)
        p.append(compute_psnr(J, clear[i:i+1]).item())
        s.append(compute_ssim(J, clear[i:i+1]).item())
        a.append(compute_sam(J, clear[i:i+1]).item())
    results['baseline_dcp'] = {'psnr': float(np.mean(p)), 'ssim': float(np.mean(s)), 'sam': float(np.mean(a))}
    print(f"暗通道先验(DCP):    PSNR={results['baseline_dcp']['psnr']:.2f} "
          f"SSIM={results['baseline_dcp']['ssim']:.4f} SAM={results['baseline_dcp']['sam']:.2f}")

    # 汇总已有模型分数
    model_files = {
        'psgnet_100ep': 'results/metrics_psgnet_phase2.json',
        'cond_ldm_200ep': 'results/metrics_conditional_ldm_phase3.json',
        'physics_ldm_200ep': 'results/metrics_physics_ldm_phase4.json',
        'unpaired_ldm_200ep': 'results/metrics_unpaired_phase5.json',
    }
    for name, fp in model_files.items():
        if Path(fp).exists():
            with open(fp) as f:
                m = json.load(f)
            results[name] = m
            print(f"{name}: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}")

    with open('results/eval_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/eval_baselines.json")


if __name__ == '__main__':
    main()
