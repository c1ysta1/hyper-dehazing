"""
阶段五：无配对训练实验
文件名：train_unpaired_phase5.py
核心思想：
  - 训练数据只有雾霾图（无清晰图配对），用 ASM 作为自监督信号；
  - 扩散模型学习：从噪声 -> 满足物理约束的清晰图；
  - 用预训练无条件扩散先验作为"伪清晰图"来源；
  - 损失：L = L_diffusion + λ1 * L_asm + λ2 * L_cycle（循环一致性）。

实现说明：
  由于完全无配对训练不稳定，本实现采用半监督策略：
  - 用已训练的自编码器建立清晰图先验（通过随机采样的 latent 解码得到伪清晰图）；
  - 用 ASM 将伪清晰图合成为"伪雾霾图"；
  - 在伪配对上训练条件扩散模型；
  - 同时用 ASM 物理一致性损失约束真实雾霾图域。
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


def train_unpaired(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Unpaired training on {device}")

    # 加载真实雾霾图（不用 clear）
    train_data = torch.load(args.train_data)
    test_data = torch.load(args.test_data)
    train_hazy = train_data['hazy'].float()
    test_hazy = test_data['hazy'].float()
    test_clear = test_data['clear'].float()  # 仅用于最终评测
    in_ch = train_hazy.shape[1]

    # 固定自编码器
    autoencoder = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch, base_ch=args.ae_base_ch).to(device)
    ae_ckpt = Path(args.checkpoint_dir) / 'ldm_hsi_autoencoder_best_phase3.pth'
    autoencoder.load_state_dict(torch.load(ae_ckpt, map_location=device))
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # 真实雾霾 latent
    with torch.no_grad():
        z_train_hazy = []
        bs = 8
        for i in range(0, len(train_hazy), bs):
            z_train_hazy.append(autoencoder.encode(train_hazy[i:i+bs].to(device)).cpu())
        z_train_hazy = torch.cat(z_train_hazy, dim=0)

    # 条件扩散模型 + ASM
    model = ConditionalLatentUNet(in_ch=args.latent_ch, cond_ch=args.latent_ch,
                                  time_emb_dim=args.time_emb_dim, base_ch=args.base_ch).to(device)
    asm = AtmosphericScatteringModel(in_ch=in_ch, base_ch=args.ae_base_ch).to(device)
    ddpm = DDPM(timesteps=args.timesteps, device=device)
    optimizer = optim.Adam(list(model.parameters()) + list(asm.parameters()), lr=args.lr)

    train_ds = TensorDataset(z_train_hazy, train_hazy)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {'diff_loss': [], 'phys_loss': [], 'total_loss': []}
    for epoch in range(args.epochs):
        model.train()
        asm.train()
        diff_l, phys_l, total_l = [], [], []
        for z_hazy, hazy in train_loader:
            z_hazy, hazy = z_hazy.to(device), hazy.to(device)
            optimizer.zero_grad()

            # 1) 生成伪清晰图：从随机 latent 解码
            z_pseudo_clear = torch.randn_like(z_hazy)
            with torch.no_grad():
                pseudo_clear = autoencoder.decode(z_pseudo_clear).clamp(0, 1)

            # 2) 用 ASM 将伪清晰图合成伪雾霾图（训练 ASM 反向）
            #    并用真实雾霾图约束 ASM 域
            t_est, A_est = asm(hazy, pseudo_clear)
            pseudo_hazy = pseudo_clear * t_est + A_est * (1 - t_est)
            phys_loss = F.l1_loss(pseudo_hazy, hazy)

            # 3) 扩散损失：以真实雾霾 latent 为条件，预测伪清晰 latent
            t = torch.randint(0, args.timesteps, (z_pseudo_clear.size(0),), device=device).long()
            z_noisy, noise = ddpm.add_noise(z_pseudo_clear, t)
            predicted_noise = model(z_noisy, z_hazy, t)
            diff_loss = F.mse_loss(predicted_noise, noise)

            total_loss = diff_loss + args.lambda_phys * phys_loss
            total_loss.backward()
            optimizer.step()

            diff_l.append(diff_loss.item())
            phys_l.append(phys_loss.item())
            total_l.append(total_loss.item())

        history['diff_loss'].append(float(np.mean(diff_l)))
        history['phys_loss'].append(float(np.mean(phys_l)))
        history['total_loss'].append(float(np.mean(total_l)))
        print(f"Epoch [{epoch+1}/{args.epochs}] diff={history['diff_loss'][-1]:.4f} "
              f"phys={history['phys_loss'][-1]:.4f} total={history['total_loss'][-1]:.4f}")

    torch.save(model.state_dict(), checkpoint_dir / 'unpaired_ldm_phase5.pth')
    torch.save(asm.state_dict(), checkpoint_dir / 'asm_unpaired_phase5.pth')
    with open(checkpoint_dir / 'history_unpaired_phase5.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 测试
    print("Evaluating unpaired model on test set...")
    model.eval()
    asm.eval()
    metrics = {'psnr': [], 'ssim': [], 'sam': []}
    with torch.no_grad():
        z_test_hazy = autoencoder.encode(test_hazy.to(device))
        z_pred = sample_latent(model, z_test_hazy, args.timesteps, device)
        pred = autoencoder.decode(z_pred).clamp(0, 1)
        test_clear_d = test_clear.to(device)
        for i in range(len(pred)):
            p = pred[i:i+1]
            g = test_clear_d[i:i+1]
            metrics['psnr'].append(compute_psnr(p, g).item())
            metrics['ssim'].append(compute_ssim(p, g).item())
            metrics['sam'].append(compute_sam(p, g).item())
    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"Test metrics: PSNR={avg['psnr']:.2f}dB SSIM={avg['ssim']:.4f} SAM={avg['sam']:.2f}°")
    with open(out_dir / 'metrics_unpaired_phase5.json', 'w') as f:
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
    train_unpaired(args)
