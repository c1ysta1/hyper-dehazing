#!/bin/bash
# 阶段6修订实验一键脚本（需要 GPU）
# 内容: 统一评测口径重测 + 阶段5无ASM消融 + 三阶段多种子 + GPU基准
# 用法: bash scripts/run_phase6b_experiments.sh 2>&1 | tee logs/phase6b_runs.log
set -e
cd "$(dirname "$0")/.."

echo "=== [0/4] GPU 检查 ==="
python -c "import torch; assert torch.cuda.is_available(), 'GPU 不可用'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== [1/4] 原有 checkpoint 统一评测 (best ckpt + blend=1.0主口径/0.9旧口径) ==="
python scripts/eval_unified_phase6b.py --ckpt checkpoints/conditional_ldm_phase3_full_best.pth --z_stats checkpoints/conditional_ldm_phase3_full_z_stats.pth --name phase3_orig
python scripts/eval_unified_phase6b.py --ckpt checkpoints/physics_ldm_phase4_best.pth --z_stats checkpoints/physics_ldm_phase4_z_stats.pth --name phase4_orig
python scripts/eval_unified_phase6b.py --ckpt checkpoints/unpaired_phase5_best.pth --z_stats checkpoints/unpaired_phase5_z_stats.pth --name phase5_orig

echo "=== [2/4] 训练: 阶段5无ASM消融 ×2种子 + 阶段3/4/5 各种子0/1 ==="
python scripts/train_unpaired_phase5.py --lambda_asm 0 --seed 0 --run_suffix _noasm_seed0
python scripts/train_unpaired_phase5.py --lambda_asm 0 --seed 1 --run_suffix _noasm_seed1
python scripts/train_unpaired_phase5.py --seed 0 --run_suffix _seed0
python scripts/train_unpaired_phase5.py --seed 1 --run_suffix _seed1
python scripts/train_physics_ldm_phase4.py --seed 0 --run_suffix _seed0
python scripts/train_physics_ldm_phase4.py --seed 1 --run_suffix _seed1
python scripts/train_conditional_ldm_phase3_full.py --seed 0 --run_suffix _seed0
python scripts/train_conditional_ldm_phase3_full.py --seed 1 --run_suffix _seed1

echo "=== [3/4] 新 checkpoint 统一评测 ==="
for sfx in _seed0 _seed1; do
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/conditional_ldm_phase3_full${sfx}_best.pth" --z_stats "checkpoints/conditional_ldm_phase3_full${sfx}_z_stats.pth" --name "phase3${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/physics_ldm_phase4${sfx}_best.pth" --z_stats "checkpoints/physics_ldm_phase4${sfx}_z_stats.pth" --name "phase4${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/unpaired_phase5${sfx}_best.pth" --z_stats "checkpoints/unpaired_phase5${sfx}_z_stats.pth" --name "phase5${sfx}"
  python scripts/eval_unified_phase6b.py --ckpt "checkpoints/unpaired_phase5_noasm${sfx}_best.pth" --z_stats "checkpoints/unpaired_phase5_noasm${sfx}_z_stats.pth" --name "phase5_noasm${sfx}"
done

echo "=== [4/4] GPU 基准测试 (覆盖 results/phase6_benchmark.json) ==="
python scripts/benchmark_phase6.py

echo "=== 全部完成: 结果在 results/metrics_unified/ 与 results/phase6_benchmark.json ==="
