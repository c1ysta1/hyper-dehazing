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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
from rshpdid_dataset import load_rshpdid_tensors

from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.simple_ddpm_phase3 import DDPM, SinusoidalPositionEmbeddings


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力（极小参数量）。"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(8, ch // reduction)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module（通道+空间双重注意力）。"""
    def __init__(self, ch, reduction=8, k=7):
        super().__init__()
        mid = max(8, ch // reduction)
        self.ch_fc = nn.Sequential(
            nn.Linear(ch, mid), nn.ReLU(), nn.Linear(mid, ch))
        self.sp_conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]
        # 通道注意力：avg+max 双路径共享 MLP
        w_avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        w_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.ch_fc(w_avg) + self.ch_fc(w_max)).view(b, c, 1, 1)
        x = x * w
        # 空间注意力：avg/max 通道统计 -> 单通道显著图
        s = torch.sigmoid(self.sp_conv(
            torch.cat([x.mean(dim=1, keepdim=True),
                       x.max(dim=1, keepdim=True)[0]], dim=1)))
        return x * s


class SpatialFiLM(nn.Module):
    """空间条件调制：由雾霾 latent 图生成 scale/shift，显式强化条件通路。"""
    def __init__(self, cond_ch, feat_ch):
        super().__init__()
        self.to_scale = nn.Conv2d(cond_ch, feat_ch, 3, padding=1)
        self.to_shift = nn.Conv2d(cond_ch, feat_ch, 3, padding=1)

    def forward(self, f, cond):
        # cond 自适应下采样到 f 的分辨率
        if cond.shape[-2:] != f.shape[-2:]:
            cond = F.adaptive_avg_pool2d(cond, f.shape[-2:])
        return f * (1.0 + self.to_scale(cond)) + self.to_shift(cond)


class ConditionalLatentUNet(nn.Module):
    """在潜在空间 (B, C, H, W) 中运行的条件 U-Net。

    depth=下采样次数。depth=1 与旧结构逐位等价：
      enc0(32->192) -> enc1(/2, 192->384) -> bottleneck(384->384)
      -> up(384->192) -> dec(concat 192+192 -> 192) -> out(192->16)
    depth>1 增加一级下采样/上采样，加强扩散去噪容量。
    module=可选容量增强模块（默认 none，插入点：bottleneck 后 + 每级 dec 后）：
      none = 旧结构（逐位等价基线）
      se   = SE 通道注意力
      cbam = CBAM 通道+空间注意力
      film = SpatialFiLM 条件空间调制（利用雾霾 latent 显式调制特征）
    """
    def __init__(self, in_ch=16, cond_ch=16, time_emb_dim=128, base_ch=128,
                 depth=1, module="none"):
        super().__init__()
        self.depth = depth
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )

        # 编码器: depth+1 个卷积（第 0 层保持 64x64，之后每层 stride=2 降采样一次），
        # 通道逐级 base_ch * 2^i；depth=1 时 = [enc0(32->192,64x64), enc1(/2, 192->384,32x32)]
        self.encs = nn.ModuleList()
        ch = in_ch + cond_ch
        for i in range(depth + 1):
            out_ch = base_ch * (2 ** i)
            stride = 2 if i > 0 else 1
            self.encs.append(nn.Conv2d(ch, out_ch, 3, stride=stride, padding=1))
            ch = out_ch
        bot_ch = base_ch * (2 ** depth)
        self.bottleneck = nn.Conv2d(ch, bot_ch, 3, padding=1)

        # 解码器: depth 级上采样，每级与对应 skip 拼接（含最外层的 64x64 级）
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        c = bot_ch
        for i in range(depth, 0, -1):
            out_ch = base_ch * (2 ** (i - 1))
            self.ups.append(nn.ConvTranspose2d(c, out_ch, 2, stride=2))
            self.decs.append(nn.Conv2d(out_ch + out_ch, out_ch, 3, padding=1))
            c = out_ch
        self.out = nn.Conv2d(base_ch, in_ch, 1)

        self.t_projs = nn.ModuleList()
        for i in range(depth + 1):
            self.t_projs.append(nn.Linear(time_emb_dim, base_ch * (2 ** i)))

        # 可选容量增强模块：插入点 = [bottleneck 后] + 每级 dec 输出后
        assert module in ("none", "se", "cbam", "film"), f"未知 module: {module}"
        self.module = module
        if module in ("se", "cbam"):
            dims = [bot_ch] + [base_ch * (2 ** (i - 1)) for i in range(depth, 0, -1)]
            self.attns = nn.ModuleList(
                SEBlock(d) if module == "se" else CBAM(d) for d in dims)
        elif module == "film":
            dims = [bot_ch] + [base_ch * (2 ** (i - 1)) for i in range(depth, 0, -1)]
            self.films = nn.ModuleList(SpatialFiLM(cond_ch, d) for d in dims)

    def _enhance(self, h, cond, idx):
        """在插入点应用所选模块。"""
        if self.module in ("se", "cbam"):
            return self.attns[idx](h)
        if self.module == "film":
            return self.films[idx](h, cond)
        return h

    def forward(self, x, cond, t):
        # x: noisy latent, cond: hazy latent
        inp = torch.cat([x, cond], dim=1)
        t_emb = self.time_mlp(t)

        skips = []
        h = inp
        for i in range(self.depth + 1):
            h = F.relu(self.encs[i](h) + self.t_projs[i](t_emb)[:, :, None, None])
            if i < self.depth:
                skips.append(h)  # 每个降采样前的特征保留
        b = self.bottleneck(h)
        h = b
        ai = 0
        if self.module != "none":
            h = self._enhance(h, cond, ai); ai += 1
        for i in range(self.depth):
            h = self.ups[i](h)
            skip = skips[self.depth - 1 - i]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = F.relu(self.decs[i](h))
            if self.module != "none":
                h = self._enhance(h, cond, ai); ai += 1
        return self.out(h)


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

    # 加载数据（RSyntHyperPDID npy 配对数据）
    print("Loading training pairs from npy...")
    train_data = load_rshpdid_tensors(args.train_hazy_dir, args.clear_dir,
                                      max_samples=args.max_train_samples)
    test_data = load_rshpdid_tensors(args.test_hazy_dir, args.clear_dir)
    train_hazy = train_data['hazy']
    train_clear = train_data['clear']
    test_hazy = test_data['hazy']
    test_clear = test_data['clear']

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


def sample_latent(model, z_hazy, timesteps, device, z_init=None):
    """从 z_hazy 条件生成清晰 latent（标准 DDPM 反向采样，无引导）。

    z_init: 可选初始噪声；为 None 时随机采样。传入固定噪声可让多次调用可比。
    """
    ddpm = DDPM(timesteps=timesteps, device=device)
    z = torch.randn_like(z_hazy) if z_init is None else z_init.to(device)
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


def _dark_channel(img, kernel=15):
    """暗通道先验：对图像 (B, C, H, W) 先取通道最小再空间最小池化。

    清晰无雾图像的暗通道接近 0（除天空等亮区外）；雾霾图暗通道被抬高。
    对 img_hat 最小化该值，即施加“去雾”方向的物理先验。
    """
    # 取 RGB 三通道做 DCP（182 波段取代表性三波段与 to_rgb 一致）
    B = img.size(1)
    r_idx = min(int(B * 0.72), B - 1)
    g_idx = int(B * 0.46)
    b_idx = int(B * 0.20)
    rgb = img[:, [r_idx, g_idx, b_idx]]  # (B,3,H,W)
    dark = rgb.min(dim=1, keepdim=True).values  # (B,1,H,W)
    dark = -F.max_pool2d(-dark, kernel, stride=1, padding=kernel // 2)
    return dark


def sample_latent_guided(model, z_hazy, timesteps, device, autoencoder,
                         denorm=None, guidance_s=1.0, guide_start_t=None,
                         dark_kernel=15, z_init=None):
    """测试时物理引导采样（DPS 式暗通道先验，不参与训练）。

    每个反向扩散步骤：
      1) 正常预测噪声得到 z0_hat（本步模型估计的干净 latent）；
      2) 仅在后半段 (t <= guide_start_t) 且 s>0 时：
         解码 z0_hat -> 图像，计算暗通道物理损失 L = mean(dark_channel)，
         对当前 z 求梯度并把下一步 z_{t-1} 向“降低暗通道”方向推一步；
      3) s=0 时退化为 sample_latent（天然对齐基线 20.95dB）。

    参数:
      guidance_s  : 引导强度，0=不引导；
      guide_start_t: 仅当 t <= 该值时引导（None 表示全程）；
    """
    if guidance_s == 0:
        return sample_latent(model, z_hazy, timesteps, device, z_init=z_init)

    if guide_start_t is None:
        guide_start_t = timesteps  # 默认全程引导

    ddpm = DDPM(timesteps=timesteps, device=device)
    z = torch.randn_like(z_hazy) if z_init is None else z_init.to(device)
    for t in reversed(range(timesteps)):
        t_batch = torch.full((z.size(0),), t, device=device, dtype=torch.long)
        with torch.no_grad():
            predicted_noise = model(z, z_hazy, t_batch)
        alpha = ddpm.alphas[t]
        alpha_cumprod = ddpm.alphas_cumprod[t]
        beta = ddpm.betas[t]

        # 本步干净 latent 估计（对 t=0..T 通用）
        sqrt_one_minus_ac = torch.sqrt(1 - alpha_cumprod)
        sqrt_ac = torch.sqrt(alpha_cumprod)
        z0_hat = (z - sqrt_one_minus_ac * predicted_noise) / sqrt_ac

        # 标准 DDPM 反向一步
        with torch.no_grad():
            if t > 0:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z_next = (z - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            if t > 0:
                z_next = z_next + torch.sqrt(beta) * noise

        # ---- 测试时物理引导：只在后半段、且 s>0 ----
        if t <= guide_start_t:
            z0_hat_g = z0_hat.detach().requires_grad_(True)
            z_in = denorm(z0_hat_g) if denorm is not None else z0_hat_g
            img_hat = autoencoder.decode(z_in)
            # 物理先验：清晰图暗通道应接近 0，最小化之即去雾方向
            dark = _dark_channel(img_hat, kernel=dark_kernel)
            L_phys = dark.mean()
            grad = torch.autograd.grad(L_phys, z0_hat_g)[0]
            # 梯度量级极小(~1e-6)，直接乘 s 无效果；归一化到单位方向，
            # 让 s 直接表示每步在 latent 空间的修正幅度（与 latent 量级~1可比）
            gnorm = grad.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
            z_next = z_next - guidance_s * (grad / gnorm)

        z = z_next.detach()
    return z


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hazy_dir', type=str, default='data/test',
                        help='训练雾图目录（train.zip 下载解压后改为 data/train）')
    parser.add_argument('--test_hazy_dir', type=str, default='data/test')
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    parser.add_argument('--max_train_samples', type=int, default=None)
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
