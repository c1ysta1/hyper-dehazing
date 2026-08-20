"""
阶段三：条件潜在扩散模型（高光谱去雾）
文件名：conditional_ldm_phase3.py
功能：
  1. 加载预训练自编码器，将清晰图/雾霾图编码到潜在空间；
  2. 在潜在空间中训练条件 DDPM，条件为雾霾图 latent；
  3. 测试时：输入雾霾图 -> 扩散采样得到清晰 latent -> 解码得到去雾结果；
  4. 计算 PSNR/SSIM/SAM 作为基线。
"""
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.simple_ddpm_phase3 import DDPM, SinusoidalPositionEmbeddings


class ConditionalLatentUNet(nn.Module):
    """在潜在空间 (B, C, H, W) 中运行的条件 U-Net。"""
    def __init__(self, in_ch=16, cond_ch=16, time_emb_dim=128, base_ch=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        self.enc1 = nn.Conv2d(in_ch + cond_ch, base_ch, 3, padding=1)
        self.enc2 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.bottleneck = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.up = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        self.t_proj1 = nn.Linear(time_emb_dim, base_ch)
        self.t_proj2 = nn.Linear(time_emb_dim, base_ch * 2)

    def forward(self, x, cond, t):
        # x: noisy latent, cond: hazy latent
        inp = torch.cat([x, cond], dim=1)
        t_emb = self.time_mlp(t)

        e1 = F.relu(self.enc1(inp) + self.t_proj1(t_emb)[:, :, None, None])
        e2 = F.relu(self.enc2(e1) + self.t_proj2(t_emb)[:, :, None, None])
        b = self.bottleneck(e2)
        d = F.relu(self.up(b))
        if d.shape[-2:] != e1.shape[-2:]:
            d = F.interpolate(d, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e1], dim=1)
        d = F.relu(self.dec(d))
        return self.out(d)


def compute_sam(img1, img2):
    """Spectral Angle Mapper (SAM)，输入 (B, C, H, W)。"""
    img1 = img1.reshape(img1.size(0), img1.size(1), -1)
    img2 = img2.reshape(img2.size(0), img2.size(1), -1)
    num = torch.sum(img1 * img2, dim=1)
    denom = torch.sqrt(torch.sum(img1 ** 2, dim=1) * torch.sum(img2 ** 2, dim=1) + 1e-8)
    cos_angle = torch.clamp(num / (denom + 1e-8), -1.0, 1.0)
    angle = torch.acos(cos_angle)
    return torch.mean(angle) * 180.0 / np.pi


def compute_ssim(img1, img2, window_size=11, max_val=1.0):
    """简化 SSIM（按通道平均）。"""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim.mean()


def train_conditional_ldm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Conditional LDM training on {device}")

    # 加载数据
    train_data = torch.load(args.train_data)
    test_data = torch.load(args.test_data)
    train_hazy = train_data['hazy'].float()
    train_clear = train_data['clear'].float()
    test_hazy = test_data['hazy'].float()
    test_clear = test_data['clear'].float()

    in_ch = train_clear.shape[1]
    H, W = train_clear.shape[-2:]

    # 加载并固定自编码器
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_ckpt}. Please run ldm_hsi_phase3.py first.")
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # 预计算 latent
    print("Encoding training data to latent space...")
    with torch.no_grad():
        z_train_hazy = []
        z_train_clear = []
        bs = 8
        for i in range(0, len(train_hazy), bs):
            z_train_hazy.append(autoencoder.encode(train_hazy[i:i+bs].to(device)).cpu())
            z_train_clear.append(autoencoder.encode(train_clear[i:i+bs].to(device)).cpu())
        z_train_hazy = torch.cat(z_train_hazy, dim=0)
        z_train_clear = torch.cat(z_train_clear, dim=0)

    train_ds = TensorDataset(z_train_hazy, z_train_clear)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # 条件扩散模型
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    print(f"Conditional U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': []}
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for z_hazy, z_clear in train_loader:
            z_hazy, z_clear = z_hazy.to(device), z_clear.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (z_clear.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_clear, t)
            predicted_noise = model(z_noisy, z_hazy, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        history['train_loss'].append(avg_loss)
        print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), checkpoint_dir / 'conditional_ldm_phase3.pth')
    with open(checkpoint_dir / 'history_conditional_ldm_phase3.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 测试：在 test 集上采样并解码
    print("Evaluating on test set...")
    model.eval()
    test_metrics = {'psnr': [], 'ssim': [], 'sam': []}
    with torch.no_grad():
        z_test_hazy = autoencoder.encode(test_hazy.to(device))
        # 使用较少的步长进行快速采样
        z_pred = sample_latent(model, z_test_hazy, args.timesteps, device)
        pred = autoencoder.decode(z_pred)
        test_clear_d = test_clear.to(device)
        for i in range(len(pred)):
            p = pred[i:i+1]
            g = test_clear_d[i:i+1]
            test_metrics['psnr'].append(compute_psnr(p, g).item())
            test_metrics['ssim'].append(compute_ssim(p, g).item())
            test_metrics['sam'].append(compute_sam(p, g).item())

    avg_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
    print(f"Test metrics: PSNR={avg_metrics['psnr']:.2f}dB SSIM={avg_metrics['ssim']:.4f} SAM={avg_metrics['sam']:.2f}°")
    with open(out_dir / 'metrics_conditional_ldm_phase3.json', 'w') as f:
        json.dump(avg_metrics, f, indent=2)


def sample_latent(model, z_hazy, timesteps, device):
    """从 z_hazy 条件生成清晰 latent。"""
    ddpm = DDPM(timesteps=timesteps, device=device)
    z = torch.randn_like(z_hazy)
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_batch = torch.full((z.size(0),), t, device=device, dtype=torch.long)
            predicted_noise = model(z, z_hazy, t_batch)
            alpha = ddpm.alphas[t]
            alpha_cumprod = ddpm.alphas_cumprod[t]
            beta = ddpm.betas[t]
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = (z - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            if t > 0:
                z = z + torch.sqrt(beta) * noise
    return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/synthetic_haze/train/train_tensor.pth')
    parser.add_argument('--test_data', type=str, default='data/synthetic_haze/test/test_tensor.pth')
    parser.add_argument('--latent_ch', type=int, default=16)
    parser.add_argument('--ae_base_ch', type=int, default=64)
    parser.add_argument('--base_ch', type=int, default=128)
    parser.add_argument('--time_emb_dim', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    train_conditional_ldm(args)
