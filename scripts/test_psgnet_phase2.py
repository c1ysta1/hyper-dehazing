"""
阶段二：PSGNet 测试与指标计算
文件名：test_psgnet_phase2.py
功能：加载训练好的 PSGNet checkpoint，在测试集上计算 PSNR、SSIM、SAM、推理时间，
      并可视化5个测试样本对比图。
"""
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.psgnet.psgnet_core_phase2 import PSGNet
from models.diffusion.conditional_ldm_phase3 import compute_sam, compute_ssim


def compute_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def test_psgnet(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing PSGNet on {device}")

    test_data = torch.load(args.test_data)
    test_hazy = test_data['hazy'].float()
    test_clear = test_data['clear'].float()
    in_ch = test_hazy.shape[1]

    model = PSGNet(in_ch=in_ch, base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    metrics = {'psnr': [], 'ssim': [], 'sam': [], 'time_ms': []}
    preds = []
    with torch.no_grad():
        for i in range(len(test_hazy)):
            hazy = test_hazy[i:i+1].to(device)
            clear = test_clear[i:i+1].to(device)
            t0 = time.time()
            out = model(hazy)
            t1 = time.time()
            out = out.clamp(0, 1)
            preds.append(out.cpu())
            metrics['psnr'].append(compute_psnr(out, clear).item())
            metrics['ssim'].append(compute_ssim(out, clear).item())
            metrics['sam'].append(compute_sam(out, clear).item())
            metrics['time_ms'].append((t1 - t0) * 1000)

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"PSNR: {avg['psnr']:.2f}dB")
    print(f"SSIM: {avg['ssim']:.4f}")
    print(f"SAM:  {avg['sam']:.2f}°")
    print(f"Inference time: {avg['time_ms']:.2f}ms / sample")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'metrics_psgnet_phase2.json', 'w') as f:
        json.dump(avg, f, indent=2)

    # 可视化5个样本
    B, C, H, W = test_hazy.shape
    r_idx = min(int(C * 0.7), C - 1)
    g_idx = int(C * 0.45)
    b_idx = int(C * 0.2)

    def to_rgb(img):
        # img: (C, H, W) numpy array
        rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
        return np.clip(rgb, 0, 1)

    n_vis = min(5, len(test_hazy))
    fig, axes = plt.subplots(n_vis, 3, figsize=(9, 3 * n_vis))
    if n_vis == 1:
        axes = axes[None, :]
    for i in range(n_vis):
        axes[i, 0].imshow(to_rgb(test_hazy[i, 0].numpy() if test_hazy[i].dim() == 4 else test_hazy[i].numpy()))
        axes[i, 0].set_title('Hazy input')
        axes[i, 0].axis('off')
        pred_i = preds[i]
        pred_arr = pred_i[0].numpy() if pred_i.dim() == 4 else pred_i.numpy()
        axes[i, 1].imshow(to_rgb(pred_arr))
        axes[i, 1].set_title('PSGNet output')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(to_rgb(test_clear[i, 0].numpy() if test_clear[i].dim() == 4 else test_clear[i].numpy()))
        axes[i, 2].set_title('GT clear')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / 'psgnet_vis_phase2.png', dpi=150)
    plt.close()
    print(f"Visualization saved to {out_dir / 'psgnet_vis_phase2.png'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='data/synthetic_haze/test/test_tensor.pth')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/psgnet_phase2_best.pth')
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    test_psgnet(args)
