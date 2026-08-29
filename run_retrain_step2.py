# 高光谱去雾 — 续跑编排 v2（跳过已完成的 AE，仅 LDM 训练 + DDIM 评测）
# 前置: checkpoints/ldm_hsi_autoencoder_best_phase3.pth 已生成
#       且 data/test/0191_0030.npy 已修复（47,710,208 字节）
# 用法: python run_retrain_step2.py
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_step(label, cmd):
    print(f"\n===== {label} =====", flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT)
    p.wait()
    print(f"===== {label} exit={p.returncode} =====", flush=True)
    return p.returncode


t0 = time.time()
# [2] 条件LDM + FiLM（复用 AE best + 重新编码 latent；产出 _mfilm_best.pth + _z_stats.pth + latent 缓存）
rc = run_step("条件LDM(FiLM) 训练", [
    PY, "scripts/train_conditional_ldm_phase3_full.py",
    "--module", "film", "--seed", "42",
    "--run_suffix", "_mfilm", "--log_subdir", "conditional_ldm_phase3_run2",
])

# [3] 复制 history 到网页监控的固定路径
hist = ROOT / "checkpoints" / "history_conditional_ldm_phase3_full_mfilm.json"
if hist.exists():
    import shutil
    shutil.copy(hist, ROOT / "checkpoints" / "history_conditional_ldm_phase3_full.json")
    print("history 已复制到 dashboard 监控路径", flush=True)

# [4] DDIM 少步 + 物理引导评测
rc = run_step("DDIM+引导 评测", [
    PY, "scripts/eval_ddim_guidance.py",
    "--module", "film",
    "--ckpt", "checkpoints/conditional_ldm_phase3_full_mfilm_best.pth",
    "--z_stats", "checkpoints/conditional_ldm_phase3_full_mfilm_z_stats.pth",
    "--ae_ckpt", "checkpoints/ldm_hsi_autoencoder_best_phase3.pth",
    "--lat_cache", "checkpoints/latent_cache_phase3",
    "--ddim_steps", "10", "25", "50",
    "--guidance_s", "0", "0.8", "--guide_start_t", "50",
    "--seeds", "42", "100", "200",
    "--name", "mphase3_film_ddim",
])

print(f"\n===== 全部完成, 总耗时 {time.time()-t0:.0f}s =====", flush=True)
