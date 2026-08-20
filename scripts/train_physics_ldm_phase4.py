"""
阶段四：物理引导扩散训练
文件名：train_physics_ldm_phase4.py
功能：
  1. 在条件扩散模型基础上加入 ASM 物理一致性损失；
  2. 总损失 = 扩散损失 + λ * 物理一致性损失；
  3. 训练并测试，对比有/无物理约束的 SAM 指标。
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.simple_ddpm_phase3 import DDPM
from models.diffusion.conditional_ldm_phase3 import (
    ConditionalLatentUNet,
    compute_sam,
    compute_ssim,
    sample_latent,
)
from models.physics.asm_module_phase4 import AtmosphericScatteringModel


def train_physics_ldm(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Physics-guided LDM training on {device}, lambda={args.lambda_phys}")

    train_data = torch.load(args.train_data)
    test_data = torch.load(args.test_data)
    train_hazy = train_data['hazy'].float()
    train_clear = train_data['clear'].float()
    test_hazy = test_data['hazy'].float()
    test_clear = test_data['clear'].float()

    in_ch = train_clear.shape[1]

    # 固定自编码器
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # 预编码
    print("Encoding data to latent space...")
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

    # 扩散模型与 ASM
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=args.ae_base_ch).to(device)

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
        diff_losses, phys_losses, total_losses = [], [], []
        for z_hazy, z_clear, hazy, clear in train_loader:
            z_hazy, z_clear = z_hazy.to(device), z_clear.to(device)
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()

            # 扩散损失
            t = torch.randint(0, args.timesteps, (z_clear.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_clear, t)
            predicted_noise = model(z_noisy, z_hazy, t)
            diff_loss = F.mse_loss(predicted_noise, noise)

            # 物理一致性损失（用解码后的预测清晰图）
            with torch.no_grad():
                pass
            phys_loss, t_est, A_est = asm.physics_loss(hazy, clear)

            total_loss = diff_loss + args.lambda_phys * phys_loss
            total_loss.backward()
            optimizer.step()

            diff_losses.append(diff_loss.item())
            phys_losses.append(phys_loss.item())
            total_losses.append(total_loss.item())

        history['diff_loss'].append(float(np.mean(diff_losses)))
        history['phys_loss'].append(float(np.mean(phys_losses)))
        history['total_loss'].append(float(np.mean(total_losses)))
        print(f"Epoch [{epoch+1}/{args.epochs}] diff={history['diff_loss'][-1]:.4f} "
              f"phys={history['phys_loss'][-1]:.4f} total={history['total_loss'][-1]:.4f}")

    torch.save(model.state_dict(), checkpoint_dir / 'physics_ldm_phase4.pth')
    torch.save(asm.state_dict(), checkpoint_dir / 'asm_phase4.pth')
    with open(checkpoint_dir / 'history_physics_ldm_phase4.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 测试
    print("Evaluating on test set...")
    model.eval()
    asm.eval()
    test_metrics = {'psnr': [], 'ssim': [], 'sam': []}
    with torch.no_grad():
        z_test_hazy = autoencoder.encode(test_hazy.to(device))
        z_pred = sample_latent(model, z_test_hazy, args.timesteps, device)
        pred = autoencoder.decode(z_pred).clamp(0, 1)
        test_clear_d = test_clear.to(device)
        for i in range(len(pred)):
            p = pred[i:i+1]
            g = test_clear_d[i:i+1]
            test_metrics['psnr'].append(compute_psnr(p, g).item())
            test_metrics['ssim'].append(compute_ssim(p, g).item())
            test_metrics['sam'].append(compute_sam(p, g).item())

    avg = {k: float(np.mean(v)) for k, v in test_metrics.items()}
    print(f"Test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / 'metrics_physics_ldm_phase4.json', 'w') as f:
        json.dump(avg, f, indent=2)


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
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lambda_phys', type=float, default=0.1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()
    train_physics_ldm(args)
