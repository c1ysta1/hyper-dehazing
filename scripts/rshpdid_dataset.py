"""
RSyntHyperPDID 数据集加载模块（适配 HyperHazeOff 基准数据集）
文件名：rshpdid_dataset.py

数据集结构（解压自 HuggingFace nikos74/RSyntHyperPDID 的 clear.zip / train.zip / test.zip）:
  data/clear/XXXX.npy        清晰GT，(256,256,182) float32，HWC
  data/test/XXXX_YYYY.npy    测试雾图，YYYY 为雾合成参数
  data/train/XXXX_YYYY.npy   训练雾图（train.zip 下载解压后自动可用）

配对规则: 雾图 XXXX_YYYY.npy 的 GT 为 clear/XXXX.npy（XXXX 为场景ID）

说明:
  - train.zip 下载解压前，训练脚本默认临时使用 data/test 作为训练数据；
    解压出 data/train 后，将 --train_hazy_dir 改为 data/train 即可。
  - 官方划分: 22 个测试场景（data/test），196 个训练场景（data/train），无场景重叠。
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def scene_id_of(hazy_filename):
    """从雾图文件名提取场景ID: '0010_0030.npy' -> '0010'"""
    return Path(hazy_filename).stem.split('_')[0]


def list_hazy_files(hazy_dir):
    """列出雾图目录下所有 npy 文件（排序保证可复现）。"""
    hazy_dir = Path(hazy_dir)
    if not hazy_dir.exists():
        raise FileNotFoundError(f"雾图目录不存在: {hazy_dir}")
    files = sorted(hazy_dir.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f"雾图目录为空: {hazy_dir}")
    return files


def scene_ids_in(hazy_dir):
    """提取雾图目录中出现的所有场景ID集合。"""
    return {scene_id_of(f.name) for f in list_hazy_files(hazy_dir)}


def build_pairs(hazy_dir, clear_dir):
    """
    扫描雾图目录并与 clear 目录配对。
    返回 [(hazy_path, clear_path), ...]；缺失 GT 的雾图会被跳过并打印警告。
    """
    clear_dir = Path(clear_dir)
    pairs, missing = [], []
    for f in list_hazy_files(hazy_dir):
        gt = clear_dir / f"{scene_id_of(f.name)}.npy"
        if gt.exists():
            pairs.append((f, gt))
        else:
            missing.append(f.name)
    if missing:
        print(f"[警告] {len(missing)} 张雾图缺少GT，已跳过: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if not pairs:
        raise FileNotFoundError(f"未找到任何有效配对: hazy={hazy_dir}, clear={clear_dir}")
    return pairs


def _load_chw(path):
    """读取单个 npy (H,W,C) -> torch (C,H,W) float32。"""
    arr = np.load(path)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()


class RSHPDIDDataset(Dataset):
    """
    RSyntHyperPDID 配对数据集（懒加载，按需读盘，避免一次性占用内存/显存）。
    __getitem__ 返回 (hazy, clear)，均为 (C,H,W) float32 张量。
    """

    def __init__(self, hazy_dir, clear_dir, max_samples=None):
        self.pairs = build_pairs(hazy_dir, clear_dir)
        if max_samples is not None:
            self.pairs = self.pairs[:max_samples]
        self.names = [p[0].name for p in self.pairs]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, clear_path = self.pairs[idx]
        return _load_chw(hazy_path), _load_chw(clear_path)

    def num_bands(self):
        arr = np.load(self.pairs[0][0], mmap_mode='r')
        return arr.shape[2]

    def spatial_size(self):
        arr = np.load(self.pairs[0][0], mmap_mode='r')
        return arr.shape[0], arr.shape[1]


class ClearHSIDataset(Dataset):
    """
    只加载 clear 图的数据集（用于自编码器训练等）。
    可通过 include_scenes / exclude_scenes 控制使用哪些场景。
    """

    def __init__(self, clear_dir, include_scenes=None, exclude_scenes=None, max_samples=None):
        clear_dir = Path(clear_dir)
        if not clear_dir.exists():
            raise FileNotFoundError(f"clear 目录不存在: {clear_dir}")
        files = sorted(clear_dir.glob('*.npy'))
        if include_scenes is not None:
            files = [f for f in files if f.stem in include_scenes]
        if exclude_scenes is not None:
            files = [f for f in files if f.stem not in exclude_scenes]
        if max_samples is not None:
            files = files[:max_samples]
        if not files:
            raise FileNotFoundError(f"clear 目录无可用文件: {clear_dir}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return _load_chw(self.files[idx])

    def num_bands(self):
        arr = np.load(self.files[0], mmap_mode='r')
        return arr.shape[2]

    def spatial_size(self):
        arr = np.load(self.files[0], mmap_mode='r')
        return arr.shape[0], arr.shape[1]


def load_rshpdid_tensors(hazy_dir, clear_dir, max_samples=None):
    """
    将配对数据一次性加载为张量字典 {'hazy': (N,C,H,W), 'clear': (N,C,H,W)}，
    与旧版 *.pth 数据格式兼容（供阶段三/四/五等需要整体张量的脚本使用）。
    注意: 全量训练集约 224GB 内存，仅在内存充足时使用；训练优先用 RSHPDIDDataset 懒加载。
    """
    pairs = build_pairs(hazy_dir, clear_dir)
    if max_samples is not None:
        pairs = pairs[:max_samples]
    hazy_list, clear_list = [], []
    for i, (hp, cp) in enumerate(pairs):
        hazy_list.append(_load_chw(hp))
        clear_list.append(_load_chw(cp))
        if (i + 1) % 50 == 0:
            print(f"  已加载 {i + 1}/{len(pairs)} 对样本")
    return {'hazy': torch.stack(hazy_list), 'clear': torch.stack(clear_list)}


def load_hazy_tensors(hazy_dir, max_samples=None):
    """只加载雾图张量 (N,C,H,W)（供阶段五无配对训练使用）。"""
    files = list_hazy_files(hazy_dir)
    if max_samples is not None:
        files = files[:max_samples]
    return torch.stack([_load_chw(f) for f in files])


if __name__ == '__main__':
    # 自检: python scripts/rshpdid_dataset.py [--hazy_dir data/test]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hazy_dir', type=str, default='data/test')
    parser.add_argument('--clear_dir', type=str, default='data/clear')
    args = parser.parse_args()

    ds = RSHPDIDDataset(args.hazy_dir, args.clear_dir)
    print(f"样本数: {len(ds)}, 波段数: {ds.num_bands()}, 空间尺寸: {ds.spatial_size()}")
    print(f"涉及场景数: {len(set(scene_id_of(n) for n in ds.names))}")
    hazy, clear = ds[0]
    print(f"hazy: {tuple(hazy.shape)} range=[{hazy.min():.3f},{hazy.max():.3f}]")
    print(f"clear: {tuple(clear.shape)} range=[{clear.min():.3f},{clear.max():.3f}]")
    print("数据集模块自检通过。")
