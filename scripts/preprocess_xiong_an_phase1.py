"""
阶段一：数据预处理与雾霾合成
文件名：preprocess_xiong_an_phase1.py
功能：读取本地ENVI高光谱数据，切patch，基于大气散射模型合成配对雾霾图，
      划分train/test，保存为PyTorch张量，并可视化样本。
"""
import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from spectral import envi
from tqdm import tqdm
from pathlib import Path


def read_envi_data(hdr_path, img_path=None):
    """读取ENVI格式高光谱数据，返回 (H, W, B) numpy array."""
    if img_path is None:
        img = envi.open(hdr_path)
    else:
        img = envi.open(hdr_path, image=img_path)
    arr = img.load()  # memmap-like, load into memory
    return np.array(arr, dtype=np.float32)


def remove_bands(data, remove_first=10, remove_last=10):
    """剔除前后边缘波段，保留中间有效波段。"""
    B = data.shape[-1]
    keep = list(range(remove_first, B - remove_last))
    return data[..., keep], keep


def normalize(data, method='max'):
    """将数据归一化到 [0, 1]。"""
    if method == 'max':
        vmax = data.max()
        vmin = data.min()
    elif method == 'percentile':
        vmin = np.percentile(data, 0.5)
        vmax = np.percentile(data, 99.5)
    else:
        raise ValueError(method)
    data = np.clip((data - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    return data, (vmin, vmax)


def generate_haze(clear, t_min=0.3, t_max=0.95, a_min=0.1, a_max=0.5):
    """
    基于大气散射模型合成雾霾图：
        I = J * t + A * (1 - t)
    输入 clear: (B, H, W)，已归一化到 [0,1]
    输出 hazy, t, A
    """
    B, H, W = clear.shape
    # 全局透射率：每个patch一个标量，或空间变化的随机图
    t_global = random.uniform(t_min, t_max)
    # 空间变化：增加非均匀雾霾，让透射率与“深度”相关
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)
    depth = 0.5 + 0.5 * np.sin(2 * np.pi * (xv + yv + random.random()))
    t = t_global * (0.7 + 0.3 * depth)
    t = t.astype(np.float32)[None, ...]  # (1, H, W)
    # 大气光：每个波段一个值
    A = np.random.uniform(a_min, a_max, size=(B, 1, 1)).astype(np.float32)
    hazy = clear * t + A * (1.0 - t)
    hazy = np.clip(hazy, 0.0, 1.0)
    return hazy, t, A


def crop_patches(data, patch_size=128, stride=None, max_patches=None):
    """从 (H, W, B) 数据中切patch，返回 list of (B, H, W) patches."""
    H, W, B = data.shape
    if stride is None:
        stride = patch_size // 2
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = data[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch.transpose(2, 0, 1).copy())  # (B, H, W)
            if max_patches and len(patches) >= max_patches:
                return patches
    return patches


def visualize_samples(clear_patches, hazy_patches, save_path, num=3):
    """可视化清晰图、雾霾图、差异图。选择3个波段近似RGB。"""
    B = clear_patches[0].shape[0]
    # 选择波段索引：近红、绿、蓝（简单取首中尾）
    if B >= 3:
        r_idx = min(int(B * 0.7), B - 1)
        g_idx = int(B * 0.45)
        b_idx = int(B * 0.2)
    else:
        r_idx = g_idx = b_idx = 0

    def to_rgb(img):
        rgb = np.stack([img[r_idx], img[g_idx], img[b_idx]], axis=-1)
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    fig, axes = plt.subplots(num, 3, figsize=(9, 3 * num))
    if num == 1:
        axes = axes[None, :]
    for i in range(num):
        clear = to_rgb(clear_patches[i])
        hazy = to_rgb(hazy_patches[i])
        diff = np.abs(clear - hazy)
        axes[i, 0].imshow(clear)
        axes[i, 0].set_title('Clear')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(hazy)
        axes[i, 1].set_title('Hazy (synthetic)')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(diff)
        axes[i, 2].set_title('Difference')
        axes[i, 2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'train').mkdir(exist_ok=True)
    (out_dir / 'test').mkdir(exist_ok=True)

    print(f"Reading ENVI data from {args.hdr_path}")
    data = read_envi_data(args.hdr_path, args.img_path)
    print(f"Original shape: {data.shape}")

    # 剔除波段
    data, kept_bands = remove_bands(data, args.remove_first, args.remove_last)
    print(f"After band removal: {data.shape}, kept bands {len(kept_bands)} ({kept_bands[0]}-{kept_bands[-1]})")

    # 归一化
    data, norm_params = normalize(data, args.norm_method)
    print(f"Normalized to [0,1] using {args.norm_method}: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")
    np.save(out_dir / 'norm_params.npy', np.array(norm_params))

    # 切patch
    print(f"Cropping {args.patch_size}x{args.patch_size} patches with stride {args.stride}")
    patches = crop_patches(data, args.patch_size, args.stride, max_patches=args.max_patches_total)
    print(f"Total patches: {len(patches)}")

    # 随机划分 train / test
    random.shuffle(patches)
    n_test = min(args.num_test, len(patches) // 5)
    n_train = len(patches) - n_test
    train_patches = patches[:n_train]
    test_patches = patches[n_train:]
    print(f"Train: {len(train_patches)}, Test: {len(test_patches)}")

    def save_split(patches, split_name):
        split_dir = out_dir / split_name
        clear_list = []
        hazy_list = []
        for idx, clear in enumerate(tqdm(patches, desc=f"Generating {split_name}")):
            hazy, t, A = generate_haze(clear)
            clear_list.append(torch.from_numpy(clear))
            hazy_list.append(torch.from_numpy(hazy))
            # 保存单个样本以便后续检查
            np.savez(
                split_dir / f"sample_{idx:04d}.npz",
                clear=clear.astype(np.float16),
                hazy=hazy.astype(np.float16),
                t=t.astype(np.float16),
                A=A.astype(np.float16),
            )
        # 保存整个 split 张量
        torch.save(
            {'clear': torch.stack(clear_list), 'hazy': torch.stack(hazy_list)},
            split_dir / f'{split_name}_tensor.pth'
        )
        print(f"Saved {split_name} tensor: clear shape {clear_list[0].shape}, samples {len(clear_list)}")

    save_split(train_patches, 'train')
    save_split(test_patches, 'test')

    # 可视化
    vis_dir = Path(args.result_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualize_samples(train_patches[:3], [generate_haze(p)[0] for p in train_patches[:3]],
                      vis_dir / 'vis_xiong_an_phase1.png')

    # 保存配置
    with open(out_dir / 'info.txt', 'w') as f:
        f.write(f"hdr_path: {args.hdr_path}\n")
        f.write(f"img_path: {args.img_path}\n")
        f.write(f"original_shape: {data.shape}\n")
        f.write(f"kept_bands: {len(kept_bands)}\n")
        f.write(f"patch_size: {args.patch_size}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"train_samples: {len(train_patches)}\n")
        f.write(f"test_samples: {len(test_patches)}\n")
        f.write(f"norm_params: {norm_params}\n")

    print("Preprocessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_path', type=str,
                        default='data/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc/XiongAn.hdr')
    parser.add_argument('--img_path', type=str,
                        default='data/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc/xiong_an_ma_ti_wan_cun_hang_kong_gao_etc/XiongAn/XiongAn.img')
    parser.add_argument('--output_dir', type=str, default='data/synthetic_haze')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--max_patches_total', type=int, default=600,
                        help='最多生成多少个patch，防止数据过大')
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--remove_first', type=int, default=10)
    parser.add_argument('--remove_last', type=int, default=10)
    parser.add_argument('--norm_method', type=str, default='max', choices=['max', 'percentile'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
