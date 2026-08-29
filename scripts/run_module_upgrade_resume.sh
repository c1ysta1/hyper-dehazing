#!/bin/bash
# 模块消融续跑脚本: 阶段3四组已训完(none ckpt已评), 从 se/cbam/film 评测开始
# 背景: 首次流水线 eval 循环漏传 --module 导致 set -e 中止; 训练产物完好无需重训
set -e
cd "$(dirname "$0")/.."

log(){ echo "[$(date '+%H:%M:%S')] $1"; }

echo "=== [2R] 统一口径评测 se/cbam/film ==="
for m in se cbam film; do
  python scripts/eval_unified_phase6b.py --module "$m" \
    --ckpt "checkpoints/conditional_ldm_phase3_full_m${m}_best.pth" \
    --z_stats "checkpoints/conditional_ldm_phase3_full_m${m}_z_stats.pth" \
    --name "mphase3_${m}"
done

echo "=== [3R] 选择阶段3最优模块 ==="
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

echo "=== [4R] 阶段4 物理引导 + 阶段5 无配对 (module=$BEST_M) ==="
python scripts/train_physics_ldm_phase4.py --module "$BEST_M" --seed 42 \
  --run_suffix "_m$BEST_M"
python scripts/train_unpaired_phase5.py --module "$BEST_M" --seed 42 \
  --run_suffix "_m$BEST_M"

echo "=== [5R] 统一口径评测 阶段4/5 ==="
python scripts/eval_unified_phase6b.py --module "$BEST_M" \
  --ckpt "checkpoints/physics_ldm_phase4_m${BEST_M}_best.pth" \
  --z_stats "checkpoints/physics_ldm_phase4_m${BEST_M}_z_stats.pth" \
  --name "mphase4_${BEST_M}"
python scripts/eval_unified_phase6b.py --module "$BEST_M" \
  --ckpt "checkpoints/unpaired_phase5_m${BEST_M}_best.pth" \
  --z_stats "checkpoints/unpaired_phase5_m${BEST_M}_z_stats.pth" \
  --name "mphase5_${BEST_M}"

echo "=== 全部完成: 结果在 results/metrics_unified/mphase*.json ==="
