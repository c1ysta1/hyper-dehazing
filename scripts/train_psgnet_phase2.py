"""
阶段二：PSGNet 基线训练
文件名：train_psgnet_phase2.py
功能：加载本地合成配对数据，训练 PSGNet 去雾模型，保存 checkpoint 与日志。
"""
import os
import sys
import argparse
import time
import json
from pathlib import Path

# 将项目根目录加入 Python 路径，以便导入 models
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rshpdid_dataset import RSHPDIDDataset

from models.psgnet.psgnet_core_phase2 import PSGNet


def compute_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cpu':
        raise RuntimeError("CUDA 不可用！拒绝在 CPU 上训练（避免极慢的无效训练），请检查 GPU 环境。")

    # 加载数据（RSyntHyperPDID: 雾图目录 + clear GT 目录，懒加载）
    if Path(args.train_hazy_dir).resolve() == Path(args.test_hazy_dir).resolve():
        print("[提示] 训练集临时使用 data/test（train.zip 下载解压后请改为 --train_hazy_dir data/train）")
    train_ds = RSHPDIDDataset(args.train_hazy_dir, args.clear_dir, max_samples=args.max_train_samples)
    test_ds = RSHPDIDDataset(args.test_hazy_dir, args.clear_dir)

    in_ch = train_ds.num_bands()
    H, W = train_ds.spatial_size()
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    print(f"Input channels: {in_ch}, spatial size: {H}x{W}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = PSGNet(in_ch=in_ch, base_ch=args.base_ch).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # 日志
    log_dir = Path(args.log_dir) / f'psgnet_phase2_{time.strftime("%m%d_%H%M")}'
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train_log.txt'

    def log_print(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    best_psnr = 0.0
    history = {'train_loss': [], 'test_loss': [], 'test_psnr': []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (hazy, clear) in enumerate(train_loader):
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            out = model(hazy)
            loss = criterion(out, clear)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 测试
        model.eval()
        test_loss = 0.0
        test_psnr = 0.0
        with torch.no_grad():
            for hazy, clear in test_loader:
                hazy, clear = hazy.to(device), clear.to(device)
                out = model(hazy)
                loss = criterion(out, clear)
                test_loss += loss.item()
                test_psnr += compute_psnr(out, clear).item()
        test_loss /= len(test_loader)
        test_psnr /= len(test_loader)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_psnr'].append(test_psnr)

        log_print(f"Epoch [{epoch+1}/{args.epochs}] train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_psnr={test_psnr:.2f}dB")

        # 保存 checkpoint
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = checkpoint_dir / f'psgnet_phase2_epoch{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_psnr': test_psnr,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        if test_psnr > best_psnr:
            best_psnr = test_psnr
            best_path = checkpoint_dir / 'psgnet_phase2_best.pth'
            torch.save(model.state_dict(), best_path)

        scheduler.step()

    with open(checkpoint_dir / 'history_psgnet_phase2.json', 'w') as f:
        json.dump(history, f, indent=2)
    log_print(f"Training complete. Best test PSNR: {best_psnr:.2f}dB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hazy_dir', type=str, default='data/test',
                        help='训练雾图目录（train.zip 下载解压后改为 data/train）')
    parser.add_argument('--test_hazy_dir', type=str, default='data/test')
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='限制训练样本数（快速验证用）')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = parser.parse_args()
    train(args)
