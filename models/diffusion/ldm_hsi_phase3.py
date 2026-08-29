"""
阶段三：高光谱 Latent Diffusion 自编码器
文件名：ldm_hsi_phase3.py
功能：
  1. 编码器：高光谱 (236ch) -> 潜在空间 (16ch 或配置通道)；
  2. 解码器：潜在空间 -> 高光谱 (236ch)；
  3. 训练自编码器重建清晰图；
  4. 验证重建 PSNR > 30dB，并打印潜在空间尺寸与压缩比。
"""
import sys
import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
from rshpdid_dataset import ClearHSIDataset, scene_ids_in


class HSIAutoEncoder(nn.Module):
    """轻量自编码器，将高光谱图压缩到潜在空间。"""
    def __init__(self, in_ch=236, latent_ch=16, base_ch=64):
        super().__init__()
        # 编码器
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),   # /2
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),  # /4
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, latent_ch, 3, padding=1),
        )
        # 解码器
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def encode(self, x):
        z = self.enc(x)
        return z

    def decode(self, z):
        x = self.dec(z)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


def compute_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def train_autoencoder(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"HSI AutoEncoder training on {device}")
    if args.seed is not None:
        import random
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # latent_ch=16 保持历史产物名（向后兼容），其他通道数加 _lc{N} 后缀
    tag = "" if args.latent_ch == 16 else f"_lc{args.latent_ch}"
    ae_best_name = f"ldm_hsi_autoencoder_best_phase3{tag}.pth"
    ae_last_name = f"ldm_hsi_autoencoder_last_phase3{tag}.pth"

    # RSyntHyperPDID: 用非测试场景的 clear 图训练自编码器，测试场景的 clear 图做验证
    test_scenes = scene_ids_in(args.test_hazy_dir)
    train_ds = ClearHSIDataset(args.clear_dir, exclude_scenes=test_scenes,
                               max_samples=args.max_train_samples)
    test_ds = ClearHSIDataset(args.clear_dir, include_scenes=test_scenes)
    in_ch = train_ds.num_bands()
    H, W = train_ds.spatial_size()
    print(f"AE train scenes: {len(train_ds)}, val scenes: {len(test_ds)} (excluded test scenes)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.base_ch).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'test_loss': [], 'test_psnr': []}
    best_psnr = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        test_psnr = 0.0
        with torch.no_grad():
            for x in test_loader:
                x = x.to(device)
                recon = model(x)
                loss = criterion(recon, x)
                test_loss += loss.item()
                test_psnr += compute_psnr(recon, x).item()
        test_loss /= len(test_loader)
        test_psnr /= len(test_loader)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)
        print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_psnr={test_psnr:.2f}dB")

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save(model.state_dict(), checkpoint_dir / ae_best_name)

    # 保存最终模型与历史
    torch.save(model.state_dict(), checkpoint_dir / ae_last_name)
    with open(checkpoint_dir / f'history_ldm_hsi_phase3{tag}.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 记录潜在空间尺寸与压缩比
    with torch.no_grad():
        x0 = test_ds[0].unsqueeze(0).to(device)
        z = model.encode(x0)
        print(f"Latent space shape (batch=1): {z.shape}, original shape: {tuple(x0.shape)}")
        compression_ratio = (in_ch * H * W) / (args.latent_ch * z.shape[-2] * z.shape[-1])
        print(f"Compression ratio: {compression_ratio:.2f}x")
    info = {
        'in_ch': in_ch,
        'latent_ch': args.latent_ch,
        'original_shape': [in_ch, H, W],
        'latent_shape': list(z[0].shape),
        'compression_ratio': compression_ratio,
        'best_psnr': best_psnr,
    }
    with open(out_dir / f'ldm_hsi_info_phase3{tag}.json', 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Autoencoder training complete. Best PSNR: {best_psnr:.2f}dB -> {ae_best_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    parser.add_argument('--test_hazy_dir', type=str, default='data/test',
                        help='用于确定测试场景（这些场景的clear图只做验证）')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--latent_ch', type=int, default=16)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    train_autoencoder(args)
