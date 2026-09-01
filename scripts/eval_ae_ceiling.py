"""
AE 重建上限复测（诊断脚本，2026-08-31）
文件名：eval_ae_ceiling.py

目的：回答"AE 上限是不是有问题、为什么跑不过 CNN（PSGNet 28.84dB）"。
方法：用同一份 best 自编码器权重，分别在【训练场景 clear】与【测试场景 clear】上
     评测重建 PSNR/SSIM/SAM，用两者差距区分失败模式：
       - train ≈ test（都 ~25dB）  => 压缩本身有损：182x 信息瓶颈 + 4 层 plain conv
                                      架构的天花板，与见没见过数据无关
       - train >> test（差 >1dB） => 泛化受限：AE 记住了训练场景，上限可靠
                                      更多训练/lr 衰减/更强架构抬升
背景证据（来自 history_ldm_hsi_phase3.json）：
       - 50 epoch 无 lr 衰减，test_psnr 曲线末端仍在 23.5~24.9 波动爬升，未充分收敛；
       - 通道消融 16→32/64ch 反而变差（24.63/24.71 < 24.90），说明瓶颈在架构不在通道数。

数据：外部磁盘 F:/data/{clear, test}（注意本项目 data/ 已迁移到 F 盘）。
用法：
  python scripts/eval_ae_ceiling.py \
      --clear_dir F:/data/clear --test_hazy_dir F:/data/test \
      --ae_ckpt checkpoints/ldm_hsi_autoencoder_best_phase3.pth
"""
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rshpdid_dataset import ClearHSIDataset, scene_ids_in
from models.diffusion.ldm_hsi_phase3 import HSIAutoEncoder, compute_psnr
from models.diffusion.conditional_ldm_phase3 import compute_sam, compute_ssim


def eval_split(ae, ds, device, batch_size=4):
    """对整个数据集做 encode->decode 重建评测，逐图计算三指标。"""
    psnrs, ssims, sams = [], [], []
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            x = torch.stack([ds[k] for k in range(i, min(i + batch_size, len(ds)))]).to(device)
            recon = ae(x)
            for j in range(x.size(0)):
                psnrs.append(compute_psnr(recon[j:j + 1], x[j:j + 1]).item())
                ssims.append(compute_ssim(recon[j:j + 1], x[j:j + 1]).item())
                sams.append(compute_sam(recon[j:j + 1], x[j:j + 1]).item())
    return psnrs, ssims, sams


def summarize(psnrs, ssims, sams):
    return {
        "n": len(psnrs),
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "psnr_min": float(np.min(psnrs)), "psnr_max": float(np.max(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "sam_mean": float(np.mean(sams)), "sam_std": float(np.std(sams)),
    }


def main():
    p = argparse.ArgumentParser(description="AE 重建上限复测（train vs test clear）")
    p.add_argument("--clear_dir", type=str, default="F:/data/clear")
    p.add_argument("--test_hazy_dir", type=str, default="F:/data/test",
                   help="用于确定测试场景 ID（与 AE 训练时的划分完全一致）")
    p.add_argument("--ae_ckpt", type=str, default="checkpoints/ldm_hsi_autoencoder_best_phase3.pth")
    p.add_argument("--latent_ch", type=int, default=16)
    p.add_argument("--ae_base_ch", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--out", type=str, default="results/metrics_guided/ae_ceiling_recheck.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ae_ckpt={args.ae_ckpt}", flush=True)

    # 与 AE 训练时完全一致的划分：test 场景 clear 做验证，其余做训练
    test_scenes = scene_ids_in(args.test_hazy_dir)
    train_ds = ClearHSIDataset(args.clear_dir, exclude_scenes=test_scenes)
    test_ds = ClearHSIDataset(args.clear_dir, include_scenes=test_scenes)
    in_ch = train_ds.num_bands()
    print(f"train clear: {len(train_ds)} 张 | test clear: {len(test_ds)} 张 | 波段: {in_ch}",
          flush=True)

    ae = HSIAutoEncoder(in_ch=in_ch, latent_ch=args.latent_ch,
                        base_ch=args.ae_base_ch).to(device).eval()
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device))

    results = {}
    for name, ds in [("train_clear", train_ds), ("test_clear", test_ds)]:
        psnrs, ssims, sams = eval_split(ae, ds, device, args.batch_size)
        results[name] = summarize(psnrs, ssims, sams)
        r = results[name]
        print(f"[{name}] n={r['n']}  PSNR={r['psnr_mean']:.2f}±{r['psnr_std']:.2f} "
              f"(min={r['psnr_min']:.2f}, max={r['psnr_max']:.2f})  "
              f"SSIM={r['ssim_mean']:.4f}±{r['ssim_std']:.4f}  "
              f"SAM={r['sam_mean']:.2f}±{r['sam_std']:.2f}", flush=True)

    gap = results["train_clear"]["psnr_mean"] - results["test_clear"]["psnr_mean"]
    results["generalization_gap_db"] = float(gap)
    print(f"\ntrain - test 泛化差距: {gap:+.2f}dB")
    if gap > 1.0:
        print("=> 结论倾向【泛化受限】：AE 记住了训练场景，上限可通过更多训练抬升")
    else:
        print("=> 结论倾向【压缩有损】：train/test 同水位，上限是 182x 压缩 + 架构问题")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
