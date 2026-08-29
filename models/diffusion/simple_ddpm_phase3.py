"""
阶段三：简化版 DDPM 实现与验证
文件名：simple_ddpm_phase3.py
功能：
  1. 实现基础 DDPM 前向加噪与反向去噪过程；
  2. 实现 2D U-Net 去噪网络；
  3. 在 MNIST 上训练并验证从纯噪声生成清晰数字；
  4. 将网络适配到高光谱数据（输入通道从 1 改为 236）。
"""
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步嵌入。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """轻量 U-Net 去噪网络，可接受任意输入通道数。"""
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.enc1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc2 = nn.Conv2d(base_ch, base_ch * 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, 2)
        self.dec = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        self.time_proj1 = nn.Linear(time_emb_dim, base_ch)
        self.time_proj2 = nn.Linear(time_emb_dim, base_ch * 2)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = F.relu(self.enc1(x) + self.time_proj1(t_emb)[:, :, None, None])
        e2 = self.pool(e1)
        e2 = F.relu(self.enc2(e2) + self.time_proj2(t_emb)[:, :, None, None])
        b = self.bottleneck(e2)
        d = F.relu(self.up(b))
        if d.shape[-2:] != e1.shape[-2:]:
            d = F.interpolate(d, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e1], dim=1)
        d = F.relu(self.dec(d))
        return self.out(d)


class DDPM:
    """简化 DDPM 定义。"""
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def sample(self, model, shape):
        """从纯噪声生成样本。"""
        model.eval()
        x = torch.randn(shape).to(self.device)
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t_batch)
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
                if t > 0:
                    x = x + torch.sqrt(beta) * noise
        return x


def train_mnist(args):
    """在 MNIST 上训练 DDPM 并验证。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"MNIST DDPM using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SimpleUNet(in_ch=1, base_ch=64).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (x.size(0),), device=device).long()
            x_noisy, noise = ddpm.add_noise(x, t)
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"MNIST Epoch [{epoch+1}/{args.epochs}] loss={np.mean(losses):.4f}")

    # 生成可视化
    samples = ddpm.sample(model, (16, 1, 28, 28))
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1).cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i in range(16):
        axes[i // 4, i % 4].imshow(samples[i, 0], cmap='gray')
        axes[i // 4, i % 4].axis('off')
    plt.tight_layout()
    save_path = out_dir / 'ddpm_mnist_samples_phase3.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"MNIST samples saved to {save_path}")

    torch.save(model.state_dict(), out_dir / 'ddpm_mnist_phase3.pth')


def test_hsi_shape(in_ch=236, H=64, W=64, base_ch=64):
    """验证 DDPM U-Net 在高光谱尺寸上的输入输出。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(in_ch=in_ch, base_ch=base_ch).to(device)
    ddpm = DDPM(timesteps=100, device=device)
    x = torch.randn(1, in_ch, H, W).to(device)
    t = torch.randint(0, 100, (1,), device=device).long()
    out = model(x, t)
    print(f"HSI U-Net test: input {x.shape} -> output {out.shape}")
    assert out.shape == x.shape

    # 测试采样过程，但不生成完整样本
    x_noisy, noise = ddpm.add_noise(x, t)
    predicted_noise = model(x_noisy, t)
    print(f"HSI noise prediction shape: {predicted_noise.shape}")
    assert predicted_noise.shape == x.shape


def main(args):
    if args.mode == 'mnist':
        train_mnist(args)
    elif args.mode == 'hsi_shape':
        test_hsi_shape(in_ch=args.in_ch, H=args.h, W=args.w, base_ch=args.base_ch)
    elif args.mode == 'both':
        train_mnist(args)
        test_hsi_shape(in_ch=args.in_ch, H=args.h, W=args.w, base_ch=args.base_ch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='both', choices=['mnist', 'hsi_shape', 'both'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--in_ch', type=int, default=236)
    parser.add_argument('--h', type=int, default=64)
    parser.add_argument('--w', type=int, default=64)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    main(args)
