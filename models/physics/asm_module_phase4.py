"""
阶段四：大气散射模型（ASM）模块
文件名：asm_module_phase4.py
功能：
  1. 基于网络或暗通道先验估计波段自适应透射率 t(x)；
  2. 估计大气光 A（全局或按波段）；
  3. 计算物理一致性损失 L_phys = ||I - (J * t + A * (1-t))||_1；
  4. 单独测试并可视化透射率灰度图，打印不同波段 t 值。
"""
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TransmissionEstimator(nn.Module):
    """
    波段自适应透射率估计网络。
    输入：清晰图 J 与雾霾图 I（均归一化到 [0,1]），形状 (B, C, H, W)。
    输出：每个波段的透射率 t_k，形状 (B, C, H, W)。
    """
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch * 2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, hazy, clear):
        x = torch.cat([hazy, clear], dim=1)
        t = self.net(x)
        # 保证透射率在合理范围
        t = 0.1 + 0.9 * t
        return t


class AtmosphereLightEstimator(nn.Module):
    """
    波段自适应大气光估计网络。
    输出：每个波段的大气光 A_k，形状 (B, C, 1, 1)。
    """
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch * 2, base_ch),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch, in_ch),
            nn.Sigmoid(),
        )

    def forward(self, hazy, clear):
        feat_hazy = self.pool(hazy).squeeze(-1).squeeze(-1)
        feat_clear = self.pool(clear).squeeze(-1).squeeze(-1)
        feat = torch.cat([feat_hazy, feat_clear], dim=1)
        A = self.fc(feat)
        A = A[:, :, None, None]
        A = 0.05 + 0.45 * A  # 限制在 [0.05, 0.5]
        return A


class AtmosphericScatteringModel(nn.Module):
    """完整 ASM 模块：估计 t、A，并计算物理一致性损失。"""
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        self.t_estimator = TransmissionEstimator(in_ch, base_ch)
        self.A_estimator = AtmosphereLightEstimator(in_ch, base_ch)

    def forward(self, hazy, clear):
        t = self.t_estimator(hazy, clear)
        A = self.A_estimator(hazy, clear)
        return t, A

    def reconstruct(self, clear, t, A):
        return clear * t + A * (1.0 - t)

    def physics_loss(self, hazy, clear):
        t, A = self.forward(hazy, clear)
        recon = self.reconstruct(clear, t, A)
        loss = F.l1_loss(recon, hazy)
        return loss, t, A


def dark_channel_prior(hazy, window_size=15):
    """
    简化暗通道先验估计透射率（按波段最小值滤波）。
    输入 hazy: (B, C, H, W)
    输出 t: (B, C, H, W)，粗略估计。
    """
    pad = window_size // 2
    dark = -F.max_pool2d(-hazy, kernel_size=window_size, stride=1, padding=pad)
    t = 1.0 - 0.95 * dark
    return t


def test_asm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"ASM test on {device}")

    data = torch.load(args.data_path)
    hazy = data['hazy'][:4].float().to(device)
    clear = data['clear'][:4].float().to(device)
    B, C, H, W = hazy.shape

    asm = AtmosphericScatteringModel(in_ch=C).to(device)
    t, A = asm(hazy, clear)
    loss, t2, A2 = asm.physics_loss(hazy, clear)

    print(f"Input shape: {hazy.shape}")
    print(f"Transmission t shape: {t.shape}, range [{t.min():.4f}, {t.max():.4f}], mean {t.mean():.4f}")
    print(f"Atmosphere light A shape: {A.shape}, range [{A.min():.4f}, {A.max():.4f}], mean {A.mean():.4f}")
    print(f"Physics consistency loss: {loss.item():.4f}")

    # 打印不同波段的 t 均值（短波长 vs 长波长）
    band_means = t.mean(dim=(0, 2, 3)).detach().cpu().numpy()
    print(f"First 5 band t means: {band_means[:5]}")
    print(f"Last 5 band t means: {band_means[-5:]}")

    # 可视化第一个样本的第一个波段透射率
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_vis = t[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.imshow(t_vis, cmap='gray')
    plt.title('Estimated Transmission t(x) (band 0)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / 'asm_transmission_phase4.png', dpi=150)
    plt.close()
    print(f"Transmission visualization saved to {out_dir / 'asm_transmission_phase4.png'}")

    # 比较暗通道先验
    t_dcp = dark_channel_prior(hazy)
    print(f"Dark channel prior t range: [{t_dcp.min():.4f}, {t_dcp.max():.4f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/synthetic_haze/test/test_tensor.pth')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    test_asm(args)
