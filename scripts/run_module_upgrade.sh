#!/bin/bash
# 模块消融实验一键脚本（2026-08-27 决策执行）
# 背景: AE通道扩展与UNet加深(depth=2)均未突破 20.34dB; 转向 U-Net 内部模块增强
# 方案: depth=1 + 三种轻量模块 vs none 对照(3.22M同构), 阶段3筛选 -> 最优模块全链路
#   se: SE通道注意力(+0.05M) / cbam: CBAM双注意力(+0.05M) / film: 空间条件调制(+0.17M)
# 注意: module=none 与旧结构逐位等价, phase_orig(19.48dB) 可作历史参考但本流水线自带 none 新对照
# 流程:
#   [1] 阶段3 x4 (none/se/cbam/film, 300ep) -> [2] 统一评测x4 -> [3] 选最优
#   [4] 阶段4/5 (最优模块) -> [5] 统一评测
# 用法: env -u LD_PRELOAD nohup bash scripts/run_module_upgrade.sh > logs/module_upgrade.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

log(){ echo "[$(date '+%H:%M:%S')] $1"; }

echo "=== [0] GPU 检查 ==="
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"

echo "=== [1/5] 阶段3 条件LDM 模块消融 (depth=1, 300ep x4) ==="
for m in none se cbam film; do
  log "phase3 训练开始: module=$m"
  python scripts/train_conditional_ldm_phase3_full.py --module $m --seed 42 \
    --run_suffix "_m$m" --log_subdir conditional_ldm_phase3_run2
done

echo "=== [2/5] 统一口径评测 x4 (blend=1.0 + 全测试集) ==="
for m in none se cbam film; do
  python scripts/eval_unified_phase6b.py --module "$m" \
    --ckpt "checkpoints/conditional_ldm_phase3_full_m${m}_best.pth" \
    --z_stats "checkpoints/conditional_ldm_phase3_full_m${m}_z_stats.pth" \
    --name "mphase3_${m}"
done

echo "=== [3/5] 选择阶段3最优模块 ==="
BEST_M=$(python - <<'PYEOF'
import json
res = {}
for m in ["none", "se", "cbam", "film"]:
    try:
        d = json.load(open(f"results/metrics_unified/mphase3_{m}.json"))
        res[m] = d["blend_1.0"]["psnr"]
    except Exception as e:
        print(f"[warn] mphase3_{m}.json 读取失败: {e}", file=__import__('sys').stderr)
assert res, "无任何有效评测结果"
best = max(res, key=res.get)
print(best)
print(" --- PSNR 汇总:", {k: round(v, 2) for k, v in sorted(res.items(), key=lambda kv: -kv[1])}, file=__import__('sys').stderr)
PYEOF
)
log "阶段3最优模块 = $BEST_M"

echo "=== [4/5] 阶段4 物理引导 + 阶段5 无配对 (module=$BEST_M) ==="
python scripts/train_physics_ldm_phase4.py --module "$BEST_M" --seed 42 \
  --run_suffix "_m$BEST_M"
python scripts/train_unpaired_phase5.py --module "$BEST_M" --seed 42 \
  --run_suffix "_m$BEST_M"

echo "=== [5/5] 统一口径评测 阶段4/5 ==="
python scripts/eval_unified_phase6b.py --module "$BEST_M" \
  --ckpt "checkpoints/physics_ldm_phase4_m${BEST_M}_best.pth" \
  --z_stats "checkpoints/physics_ldm_phase4_m${BEST_M}_z_stats.pth" \
  --name "mphase4_${BEST_M}"
python scripts/eval_unified_phase6b.py --module "$BEST_M" \
  --ckpt "checkpoints/unpaired_phase5_m${BEST_M}_best.pth" \
  --z_stats "checkpoints/unpaired_phase5_m${BEST_M}_z_stats.pth" \
  --name "mphase5_${BEST_M}"

echo "=== 全部完成: 结果在 results/metrics_unified/mphase*.json ==="
