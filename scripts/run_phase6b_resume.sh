#!/bin/bash
# 阶段6修订实验补跑脚本（GPU 掉线中断后恢复用）
# 已完成的 run（phase5 x4、phase4 x2、orig 统一评测）自动跳过；
# 只补：phase3 seed0/seed1 训练 + [3] 统一评测 + [4] GPU 基准。
# 用法: nohup bash scripts/run_phase6b_resume.sh > logs/phase6b_resume.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

log(){ echo "[$(date '+%H:%M:%S')] $1"; }

echo "=== 等待 GPU 可用 ==="
for i in $(seq 1 120); do
  if python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" > /dev/null 2>&1; then
    log "GPU 已恢复: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
    sleep 15   # 稳定后再开始，避免刚恢复又掉线
    break
  fi
  log "GPU 不可用 (第 ${i} 次检测)，30s 后重试..."
  sleep 30
  if [ "$i" = "120" ]; then
    echo "=== 等待超时(60分钟)，退出 ==="
    exit 1
  fi
done

echo "=== [补1] 训练 phase3 seed0/seed1 (latent 缓存已就位，约28min/个) ==="
python scripts/train_conditional_ldm_phase3_full.py --seed 0 --run_suffix _seed0 --resume
python scripts/train_conditional_ldm_phase3_full.py --seed 1 --run_suffix _seed1

echo "=== [补2] 新 checkpoint 统一评测 ==="
for sfx in _seed0 _seed1; do
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/conditional_ldm_phase3_full${sfx}_best.pth" --z_stats "checkpoints/conditional_ldm_phase3_full${sfx}_z_stats.pth" --name "phase3${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/physics_ldm_phase4${sfx}_best.pth" --z_stats "checkpoints/physics_ldm_phase4${sfx}_z_stats.pth" --name "phase4${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/unpaired_phase5${sfx}_best.pth" --z_stats "checkpoints/unpaired_phase5${sfx}_z_stats.pth" --name "phase5${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/unpaired_phase5_noasm${sfx}_best.pth" --z_stats "checkpoints/unpaired_phase5_noasm${sfx}_z_stats.pth" --name "phase5_noasm${sfx}"
done

echo "=== [补3] GPU 基准测试 (覆盖 results/phase6_benchmark.json) ==="
python scripts/benchmark_phase6.py

echo "=== 补跑全部完成: 结果在 results/metrics_unified/ 与 results/phase6_benchmark.json ==="
