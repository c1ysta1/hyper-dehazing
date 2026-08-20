"""
阶段二：PSGNet 核心模块实现与单元测试
文件名：psgnet_core_phase2.py
功能：实现 SpectralGrouping、PSEB、FCB（简化版）以及 U-Net 编码器-解码器，
      并验证输入输出尺寸与参数量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGrouping(nn.Module):
    """
    将 C 个通道按组数 num_groups 分成若干组，每组独立进行 3x3 卷积，
    最后通过 1x1 卷积聚合各组信息。
    """
    def __init__(self, in_ch, out_ch, num_groups=4):
        super().__init__()
        assert in_ch % num_groups == 0 and out_ch % num_groups == 0
        self.num_groups = num_groups
        self.group_in = in_ch // num_groups
        self.group_out = out_ch // num_groups
        self.group_convs = nn.ModuleList([
            nn.Conv2d(self.group_in, self.group_out, kernel_size=3, padding=1)
            for _ in range(num_groups)
        ])
        self.fuse = nn.Conv2d(out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        xs = torch.split(x, self.group_in, dim=1)
        outs = [conv(xi) for conv, xi in zip(self.group_convs, xs)]
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return out


class ChannelAttention(nn.Module):
    """通道注意力：SE-style。"""
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """空间注意力：基于通道极值。"""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, max_val], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class PSEB(nn.Module):
    """
    Physics-aware Spectral Enhancement Block。
    包含 SpectralGrouping + 通道/空间注意力 + 残差。
    """
    def __init__(self, ch, num_groups=4):
        super().__init__()
        self.sg = SpectralGrouping(ch, ch, num_groups=num_groups)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = self.sg(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ca(out)
        out = self.sa(out)
        out = self.bn2(out)
        return out + identity


class FCB(nn.Module):
    """
    Frequency-domain Construction Block 简化版。
    使用多尺度空洞卷积捕获不同频率/尺度的信息，替代 FFT。
    """
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(ch, ch // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(ch, ch // 4, kernel_size=3, dilation=2, padding=2)
        self.conv7 = nn.Conv2d(ch, ch // 4, kernel_size=3, dilation=3, padding=3)
        self.fuse = nn.Conv2d(ch, ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = torch.cat([
            self.conv1(x),
            self.conv3(x),
            self.conv5(x),
            self.conv7(x)
        ], dim=1)
        out = self.fuse(out)
        out = self.bn(out)
        out = self.relu(out)
        return out + identity


class DownBlock(nn.Module):
    """下采样块：Conv + BN + ReLU + MaxPool。"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.pool(feat)
        return feat, down


class UpBlock(nn.Module):
    """上采样块：Upsample + Concat + Conv。"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PSGNet(nn.Module):
    """
    简化版 PSGNet：U-Net + SpectralGrouping + PSEB + FCB。
    """
    def __init__(self, in_ch=172, base_ch=64, num_groups=4):
        super().__init__()
        # 输入分组增强
        self.input_sg = SpectralGrouping(in_ch, base_ch, num_groups=num_groups)
        self.input_pseb = PSEB(base_ch, num_groups=num_groups)

        # 编码器
        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.pseb1 = PSEB(base_ch * 2, num_groups=4)
        self.fcb1 = FCB(base_ch * 2)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.pseb2 = PSEB(base_ch * 4, num_groups=4)
        self.fcb2 = FCB(base_ch * 4)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8)
        self.pseb3 = PSEB(base_ch * 8, num_groups=4)
        self.fcb3 = FCB(base_ch * 8)

        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            PSEB(base_ch * 8, num_groups=4),
            FCB(base_ch * 8)
        )

        # 解码器
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch)

        # 输出
        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x):
        x = self.input_sg(x)
        x = self.input_pseb(x)

        skip1, x = self.down1(x)
        x = self.pseb1(x)
        x = self.fcb1(x)

        skip2, x = self.down2(x)
        x = self.pseb2(x)
        x = self.fcb2(x)

        skip3, x = self.down3(x)
        x = self.pseb3(x)
        x = self.fcb3(x)

        x = self.bottleneck(x)

        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.out_conv(x)
        return x


def test_modules():
    """单元测试：验证每个模块与完整网络尺寸和参数量。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 1, 172, 256, 256
    x = torch.randn(B, C, H, W).to(device)

    print("=" * 60)
    print("PSGNet Phase 2 Unit Tests")
    print("=" * 60)

    # SpectralGrouping
    sg = SpectralGrouping(C, C, num_groups=4).to(device)
    y = sg(x)
    print(f"SpectralGrouping: {x.shape} -> {y.shape}")
    assert y.shape == x.shape

    # PSEB
    pseb = PSEB(C, num_groups=4).to(device)
    y = pseb(x)
    print(f"PSEB: {x.shape} -> {y.shape}")
    assert y.shape == x.shape

    # FCB
    fcb = FCB(C).to(device)
    y = fcb(x)
    print(f"FCB: {x.shape} -> {y.shape}")
    assert y.shape == x.shape

    # Full PSGNet
    model = PSGNet(in_ch=C, base_ch=64).to(device)
    y = model(x)
    print(f"PSGNet full: {x.shape} -> {y.shape}")
    assert y.shape == x.shape

    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,} ({params / 1e6:.2f}M)")
    print("All tests passed.")


if __name__ == '__main__':
    test_modules()
