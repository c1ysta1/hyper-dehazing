"""
改进实验：DDIM采样器 + 大容量U-Net + 1000时间步
文件名：ddim_sampler_improved.py
功能：
  1. 实现DDIM采样器（非马尔可夫，支持更少步数更高质量）
  2. 用现有200轮checkpoint测试DDIM vs DDPM
  3. 支持配置更大容量的U-Net（base_ch=256）
  4. 支持训练时使用1000时间步
"""
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rshpdid_dataset import load_rshpdid_tensors

from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.simple_ddpm_phase3 import DDPM, SinusoidalPositionEmbeddings
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
)
from models.physics.asm_module_phase4 import AtmosphericScatteringModel


class DDIMSampler:
    """DDIM采样器：支持更少步数、更高质量的采样。"""
    def __init__(self, ddpm, ddim_steps=50, eta=0.0):
        """
        Args:
            ddpm: DDPM对象（含alphas, betas等）
            ddim_steps: DDIM采样步数（远少于训练时间步）
            eta: 控制随机性，0=确定性DDIM，1=DDPM等价
        """
        self.ddpm = ddpm
        self.ddim_steps = ddim_steps
        self.eta = eta
        # 从训练时间步中均匀选取ddim_steps个子步骤
        self.ddim_timesteps = np.round(
            np.linspace(0, ddpm.timesteps - 1, ddim_steps)
        ).astype(int).tolist()
        self.ddim_timesteps.reverse()  # 从大到小

    def sample(self, model, shape, cond=None, device='cpu'):
        """DDIM采样。
        Args:
            model: 去噪U-Net
            shape: 采样形状 (B, C, H, W)
            cond: 条件特征（可选），形状同shape
        """
        x = torch.randn(shape).to(device)
        with torch.no_grad():
            for i, t in enumerate(self.ddim_timesteps):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                # 预测噪声
                if cond is not None:
                    predicted_noise = model(x, cond, t_batch)
                else:
                    predicted_noise = model(x, t_batch)

                # DDIM更新公式
                alpha = self.ddpm.alphas[t]
                alpha_cumprod = self.ddpm.alphas_cumprod[t]

                # 从预测噪声计算 x_0
                x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
                x0_pred = torch.clamp(x0_pred, -1, 1)

                if i < len(self.ddim_timesteps) - 1:
                    t_prev = self.ddim_timesteps[i + 1]
                    alpha_cumprod_prev = self.ddpm.alphas_cumprod[t_prev]
                else:
                    alpha_cumprod_prev = torch.tensor(1.0, device=device)

                # DDIM sigma
                sigma = self.eta * torch.sqrt(
                    (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * (1 - alpha_cumprod / alpha_cumprod_prev)
                )

                # DDIM方向
                dir_xt = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * predicted_noise
                # DDIM前向
                noise = sigma * torch.randn_like(x) if self.eta > 0 else 0
                x = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt + noise

        return x


def evaluate_with_sampling(model, autoencoder, test_hazy, test_clear, ddpm, sampler_type='ddim',
                           ddim_steps=50, eta=0.0, device='cpu'):
    """用指定采样器评测模型。"""
    model.eval()
    autoencoder.eval()
    metrics = {'psnr': [], 'ssim': [], 'sam': []}

    with torch.no_grad():
        z_test_hazy = autoencoder.encode(test_hazy.to(device))
        if sampler_type == 'ddim':
            sampler = DDIMSampler(ddpm, ddim_steps=ddim_steps, eta=eta)
            z_pred = sampler.sample(model, z_test_hazy.shape, cond=z_test_hazy, device=device)
        else:
            # 原始DDPM采样
            z_pred = ddpm_sample_latent(model, z_test_hazy, ddpm.timesteps, device)

        pred = autoencoder.decode(z_pred).clamp(0, 1)
        test_clear_d = test_clear.to(device)
        for i in range(len(pred)):
            p = pred[i:i+1]
            g = test_clear_d[i:i+1]
            metrics['psnr'].append(compute_psnr(p, g).item())
            metrics['ssim'].append(compute_ssim(p, g).item())
            metrics['sam'].append(compute_sam(p, g).item())

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def ddpm_sample_latent(model, z_hazy, timesteps, device):
    """原始DDPM采样（从conditional_ldm_phase3.py复用）。"""
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


def test_ddim(args):
    """用现有checkpoint测试DDIM vs DDPM采样。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DDIM vs DDPM test on {device}")

    # 加载数据
    print("Loading test set from npy...")
    test_data = load_rshpdid_tensors(args.test_hazy_dir, args.clear_dir)
    test_hazy = test_data['hazy']
    test_clear = test_data['clear']
    in_ch = test_hazy.shape[1]

    # 加载自编码器
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()

    # 加载条件LDM（200轮checkpoint）
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    ckpt_cond = Path(args.checkpoint_dir) / 'conditional_ldm_phase3.pth'
    model.load_state_dict(torch.load(ckpt_cond, map_location=device))
    model.eval()

    # DDPM配置（与训练时一致：100步）
    ddpm = DDPM(timesteps=100, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. DDPM 100步（基线）
    print("Testing DDPM 100 steps...")
    t0 = time.time()
    m = evaluate_with_sampling(model, autoencoder, test_hazy, test_clear, ddpm,
                               sampler_type='ddpm', device=device)
    t1 = time.time()
    m['time_s'] = t1 - t0
    results['ddpm_100'] = m
    print(f"  DDPM 100: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}° time={m['time_s']:.1f}s")

    # 2. DDIM 50步 eta=0（确定性）
    for steps in [20, 50, 100]:
        for eta in [0.0, 0.5]:
            print(f"Testing DDIM {steps} steps eta={eta}...")
            t0 = time.time()
            m = evaluate_with_sampling(model, autoencoder, test_hazy, test_clear, ddpm,
                                       sampler_type='ddim', ddim_steps=steps, eta=eta, device=device)
            t1 = time.time()
            m['time_s'] = t1 - t0
            key = f'ddim_{steps}_eta{eta}'
            results[key] = m
            print(f"  DDIM {steps} eta={eta}: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}° time={m['time_s']:.1f}s")

    # 加载物理引导LDM（200轮checkpoint）
    ckpt_phys = Path(args.checkpoint_dir) / 'physics_ldm_phase4.pth'
    if ckpt_phys.exists():
        model_phys = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                           time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
        model_phys.load_state_dict(torch.load(ckpt_phys, map_location=device))
        model_phys.eval()
        print("\n--- Physics-guided LDM (200ep, λ=0.5) ---")
        print("Testing DDIM 50 steps eta=0...")
        m = evaluate_with_sampling(model_phys, autoencoder, test_hazy, test_clear, ddpm,
                                   sampler_type='ddim', ddim_steps=50, eta=0.0, device=device)
        results['physics_ddim_50_eta0'] = m
        print(f"  DDIM 50 eta=0: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}°")

    with open(out_dir / 'ddim_comparison_improved.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'ddim_comparison_improved.json'}")


def train_large_model(args):
    """训练大容量条件LDM（base_ch=256, timesteps=1000）。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Large model training on {device}, base_ch={args.base_ch}, timesteps={args.timesteps}")

    print("Loading data from npy...")
    train_data = load_rshpdid_tensors(args.train_hazy_dir, args.clear_dir,
                                      max_samples=args.max_train_samples)
    test_data = load_rshpdid_tensors(args.test_hazy_dir, args.clear_dir)
    train_hazy = train_data['hazy']
    train_clear = train_data['clear']
    test_hazy = test_data['hazy']
    test_clear = test_data['clear']
    in_ch = train_clear.shape[1]

    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # 预编码
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

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    print(f"Large U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

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
        avg = float(np.mean(losses))
        history['train_loss'].append(avg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={avg:.4f}")

    torch.save(model.state_dict(), checkpoint_dir / f'conditional_ldm_large_phase3.pth')
    with open(checkpoint_dir / 'history_conditional_ldm_large.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 用DDIM评测
    print("Evaluating with DDIM 50 steps eta=0...")
    m = evaluate_with_sampling(model, autoencoder, test_hazy, test_clear, ddpm,
                               sampler_type='ddim', ddim_steps=50, eta=0.0, device=device)
    print(f"Test: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}°")
    with open(out_dir / 'metrics_conditional_ldm_large_improved.json', 'w') as f:
        json.dump(m, f, indent=2)


def train_large_physics(args):
    """训练大容量物理引导LDM（base_ch=256, timesteps=1000, λ=0.5）。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Large physics LDM training on {device}, base_ch={args.base_ch}, timesteps={args.timesteps}, λ={args.lambda_phys}")

    print("Loading data from npy...")
    train_data = load_rshpdid_tensors(args.train_hazy_dir, args.clear_dir,
                                      max_samples=args.max_train_samples)
    test_data = load_rshpdid_tensors(args.test_hazy_dir, args.clear_dir)
    train_hazy = train_data['hazy']
    train_clear = train_data['clear']
    test_hazy = test_data['hazy']
    test_clear = test_data['clear']
    in_ch = train_clear.shape[1]

    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    with torch.no_grad():
        z_train_hazy = []
        z_train_clear = []
        bs = 8
        for i in range(0, len(train_hazy), bs):
            z_train_hazy.append(autoencoder.encode(train_hazy[i:i+bs].to(device)).cpu())
            z_train_clear.append(autoencoder.encode(train_clear[i:i+bs].to(device)).cpu())
        z_train_hazy = torch.cat(z_train_hazy, dim=0)
        z_train_clear = torch.cat(z_train_clear, dim=0)

    train_ds = TensorDataset(z_train_hazy, z_train_clear, train_hazy, train_clear)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=args.ae_base_ch).to(device)
    print(f"Large U-Net parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = optim.Adam(list(model.parameters()) + list(asm.parameters()), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {'diff_loss': [], 'phys_loss': [], 'total_loss': []}
    for epoch in range(args.epochs):
        model.train()
        asm.train()
        dl, pl, tl = [], [], []
        for z_hazy, z_clear, hazy, clear in train_loader:
            z_hazy, z_clear = z_hazy.to(device), z_clear.to(device)
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, args.timesteps, (z_clear.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_clear, t)
            predicted_noise = model(z_noisy, z_hazy, t)
            diff_loss = F.mse_loss(predicted_noise, noise)
            phys_loss, _, _ = asm.physics_loss(hazy, clear)
            total_loss = diff_loss + args.lambda_phys * phys_loss
            total_loss.backward()
            optimizer.step()
            dl.append(diff_loss.item())
            pl.append(phys_loss.item())
            tl.append(total_loss.item())
        history['diff_loss'].append(float(np.mean(dl)))
        history['phys_loss'].append(float(np.mean(pl)))
        history['total_loss'].append(float(np.mean(tl)))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] diff={history['diff_loss'][-1]:.4f} phys={history['phys_loss'][-1]:.4f}")

    torch.save(model.state_dict(), checkpoint_dir / 'physics_ldm_large_phase4.pth')
    torch.save(asm.state_dict(), checkpoint_dir / 'asm_large_phase4.pth')
    with open(checkpoint_dir / 'history_physics_ldm_large.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("Evaluating with DDIM 50 steps eta=0...")
    m = evaluate_with_sampling(model, autoencoder, test_hazy, test_clear, ddpm,
                               sampler_type='ddim', ddim_steps=50, eta=0.0, device=device)
    print(f"Test: PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f} SAM={m['sam']:.2f}°")
    with open(out_dir / 'metrics_physics_ldm_large_improved.json', 'w') as f:
        json.dump(m, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test_ddim', choices=['test_ddim', 'train_large', 'train_large_physics'])
    # 数据
    parser.add_argument('--train_hazy_dir', type=str, default='data/test',
                        help='训练雾图目录（train.zip 下载解压后改为 data/train）')
    parser.add_argument('--test_hazy_dir', type=str, default='data/test')
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    parser.add_argument('--max_train_samples', type=int, default=None)
    # 模型
    parser.add_argument('--latent_ch', type=int, default=16)
    parser.add_argument('--ae_base_ch', type=int, default=64)
    parser.add_argument('--base_ch', type=int, default=256, help='大模型用256，原模型128')
    parser.add_argument('--time_emb_dim', type=int, default=128)
    parser.add_argument('--timesteps', type=int, default=1000, help='训练时间步，改进用1000')
    # 训练
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lambda_phys', type=float, default=0.5)
    # 路径
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    if args.mode == 'test_ddim':
        # 测试DDIM时用原始base_ch=128
        args.base_ch = 128
        args.timesteps = 100
        test_ddim(args)
    elif args.mode == 'train_large':
        train_large_model(args)
    elif args.mode == 'train_large_physics':
        train_large_physics(args)
