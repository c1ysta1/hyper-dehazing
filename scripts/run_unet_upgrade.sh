#!/bin/bash
# 加强 UNet 实验一键脚本（2026-08-23 决策执行）
# 背景: AE 通道扩展(32/64ch)证无效(24.63/24.71 < 16ch 24.95dB), 瓶颈在扩散 UNet 容量
# 方案: depth=2 UNet(13.89M, 旧结构4.3x), 复用 16ch AE 与 latent 缓存, 重训全链路
# 流程:
#   [1] 阶段3 条件LDM(depth=2, 300ep) -> 阶段4 物理引导(depth=2, 120ep) -> 阶段5 无配对(depth=2, 120ep)
#   [2] 统一口径评测 x3 (best ckpt + blend=1.0 + seed42)
# 产物: *_d2 后缀, 与既有 16ch depth=1 结果完全隔离
# 用法: env -u LD_PRELOAD nohup bash scripts/run_unet_upgrade.sh > logs/unet_upgrade.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

log(){ echo "[$(date '+%H:%M:%S')] $1"; }

echo "=== [0] GPU 检查 ==="
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== [1/4] 阶段3 条件LDM 重训 (depth=2, 300ep, lr=2e-4) ==="
# 教训: lr=1e-3 + 无梯度裁剪时 depth=2 在第8个epoch loss爆炸(184)坍塌(loss恒1.0)
python scripts/train_conditional_ldm_phase3_full.py --depth 2 --lr 2e-4 --seed 42 --run_suffix _d2

echo "=== [2/4] 阶段4 物理引导LDM 重训 (depth=2, 120ep, lr=2e-4) ==="
python scripts/train_physics_ldm_phase4.py --depth 2 --lr 2e-4 --seed 42 --run_suffix _d2

echo "=== [3/4] 阶段5 无配对LDM 重训 (depth=2, 120ep, lr=2e-4) ==="
python scripts/train_unpaired_phase5.py --depth 2 --lr 2e-4 --seed 42 --run_suffix _d2

echo "=== [4/4] 统一口径评测 x3 ==="
python scripts/eval_unified_phase6b.py --depth 2 \
  --ckpt "checkpoints/conditional_ldm_phase3_full_d2_best.pth" \
  --z_stats "checkpoints/conditional_ldm_phase3_full_d2_z_stats.pth" \
  --name "phase3_d2"
python scripts/eval_unified_phase6b.py --depth 2 \
  --ckpt "checkpoints/physics_ldm_phase4_d2_best.pth" \
  --z_stats "checkpoints/physics_ldm_phase4_d2_z_stats.pth" \
  --name "phase4_d2"
python scripts/eval_unified_phase6b.py --depth 2 \
  --ckpt "checkpoints/unpaired_phase5_d2_best.pth" \
  --z_stats "checkpoints/unpaired_phase5_d2_z_stats.pth" \
  --name "phase5_d2"

echo "=== 全部完成: depth=2 链路结果在 results/metrics_unified/phase*_d2.json ==="
