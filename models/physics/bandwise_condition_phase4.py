"""
阶段四：波段自适应物理条件
文件名：bandwise_condition_phase4.py
功能：
  1. 设计波段分组策略（5组）；
  2. 每组共享一个透射率估计网络；
  3. 打印每组的透射率均值，验证物理直觉；
  4. 提供消融接口：统一透射率 vs 波段自适应透射率。
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
from rshpdid_dataset import load_rshpdid_tensors


class BandwiseTransmissionEstimator(nn.Module):
    """
    波段自适应透射率估计：将波段分为 num_groups 组，每组一个轻量网络。
    组内所有波段共享同一透射率图，再通过 1x1 卷积生成每个波段的 t。
    """
    def __init__(self, in_ch, num_groups=5, base_ch=32):
        super().__init__()
        self.in_ch = in_ch
        self.num_groups = num_groups
        # 不均等分组：前 (num_groups - rem) 组大小为 base，后 rem 组大小为 base+1
        base = in_ch // num_groups
        rem = in_ch % num_groups
        self.group_bounds = []
        s = 0
        for g in range(num_groups):
            size = base + (1 if g >= num_groups - rem else 0)
            self.group_bounds.append((s, s + size))
            s += size
        # 每组一个网络：输入 hazy+clear 的组内均值，输出该组透射率图
        self.group_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2, base_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_ch, 1, 3, padding=1),
                nn.Sigmoid(),
            ) for _ in range(num_groups)
        ])
        # 1x1 卷积将组级透射率映射到波段级
        self.to_bandwise = nn.Conv2d(num_groups, in_ch, 1)

    def forward(self, hazy, clear):
        B, C, H, W = hazy.shape
        group_t = []
        for g in range(self.num_groups):
            s, e = self.group_bounds[g]
            hazy_g = hazy[:, s:e].mean(dim=1, keepdim=True)
            clear_g = clear[:, s:e].mean(dim=1, keepdim=True)
            inp = torch.cat([hazy_g, clear_g], dim=1)
            t_g = self.group_nets[g](inp)  # (B, 1, H, W)
            group_t.append(t_g)
        group_t = torch.cat(group_t, dim=1)  # (B, num_groups, H, W)
        t = self.to_bandwise(group_t)  # (B, C, H, W)
        t = 0.1 + 0.9 * t
        return t, group_t


class UniformTransmissionEstimator(nn.Module):
    """统一透射率：所有波段共享一张透射率图（用于消融对比）。"""
    def __init__(self, in_ch, base_ch=32):
        super().__init__()
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Conv2d(2, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, hazy, clear):
        inp = torch.cat([hazy.mean(dim=1, keepdim=True),
                         clear.mean(dim=1, keepdim=True)], dim=1)
        t_one = self.net(inp)  # (B,1,H,W)
        t = t_one.expand(-1, self.in_ch, -1, -1).contiguous()
        t = 0.1 + 0.9 * t
        return t, t_one


def test_bandwise(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_rshpdid_tensors(args.hazy_dir, args.clear_dir, max_samples=4)
    hazy = data['hazy'].to(device)
    clear = data['clear'].to(device)
    B, C, H, W = hazy.shape

    print(f"Input: {hazy.shape}, num_groups={args.num_groups}, group_size={C // args.num_groups}")

    # 波段自适应
    bw = BandwiseTransmissionEstimator(in_ch=C, num_groups=args.num_groups).to(device)
    t_bw, group_t = bw(hazy, clear)
    group_means = group_t.mean(dim=(0, 2, 3)).detach().cpu().numpy()
    print("Bandwise group t means:", [f"{v:.4f}" for v in group_means])

    # 统一透射率
    uni = UniformTransmissionEstimator(in_ch=C).to(device)
    t_uni, _ = uni(hazy, clear)
    print(f"Uniform t mean: {t_uni.mean().item():.4f}, std: {t_uni.std().item():.4f}")
    print(f"Bandwise t mean: {t_bw.mean().item():.4f}, std: {t_bw.std().item():.4f}")

    # 物理一致性损失对比
    A = 0.3
    recon_bw = clear * t_bw + A * (1 - t_bw)
    recon_uni = clear * t_uni + A * (1 - t_uni)
    loss_bw = F.l1_loss(recon_bw, hazy).item()
    loss_uni = F.l1_loss(recon_uni, hazy).item()
    print(f"Physics loss - bandwise: {loss_bw:.4f}, uniform: {loss_uni:.4f}")

    out = {
        'num_groups': args.num_groups,
        'group_t_means': group_means.tolist(),
        'uniform_t_mean': float(t_uni.mean().item()),
        'bandwise_t_mean': float(t_bw.mean().item()),
        'loss_bandwise': loss_bw,
        'loss_uniform': loss_uni,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'bandwise_condition_phase4.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Bandwise condition test complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazy_dir', type=str, default='data/test')
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    parser.add_argument('--num_groups', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    test_bandwise(args)
