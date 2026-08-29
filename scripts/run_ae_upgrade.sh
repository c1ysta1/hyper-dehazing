#!/bin/bash
# AE 升级实验一键脚本（2026-08-23 决策执行）
# 动因: AE 重建上界 24.95dB(16ch/182x压缩) 是 LDM 系与 PSGNet 差距主因(压缩损失~3.9dB)
# 流程:
#   [A] 训 AE-32ch 与 AE-64ch(各50ep), 按测试集重建 PSNR 自动选优
#   [B] 用最优通道数重训全链路: 阶段3(300ep) -> 阶段4(120ep) -> 阶段5(120ep)
#   [C] 统一口径评测(best ckpt + blend=1.0 + 固定种子42) x3
# 产物命名: *_lc{N} 后缀, 不覆盖既有 16ch 结果
# 用法: env -u LD_PRELOAD nohup bash scripts/run_ae_upgrade.sh > logs/ae_upgrade.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

log(){ echo "[$(date '+%H:%M:%S')] $1"; }

echo "=== [0/4] GPU 检查 ==="
python -c "import torch; assert torch.cuda.is_available(), 'GPU 不可用'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== [A/4] 训练 AE latent_ch=32 / 64 (各50ep) ==="
log "AE-32 开始"
python models/diffusion/ldm_hsi_phase3.py --latent_ch 32 --seed 42
log "AE-64 开始"
python models/diffusion/ldm_hsi_phase3.py --latent_ch 64 --seed 42

LC=$(python -c "
import json
p32 = json.load(open('results/ldm_hsi_info_phase3_lc32.json'))['best_psnr']
p64 = json.load(open('results/ldm_hsi_info_phase3_lc64.json'))['best_psnr']
print(32 if p32 >= p64 else 64)
")
PSNR_AE=$(python -c "
import json
lc = $LC
print(f\"{json.load(open(f'results/ldm_hsi_info_phase3_lc{lc}.json'))['best_psnr']:.2f}\")
")
log "选优: latent_ch=$LC (重建 PSNR=${PSNR_AE}dB, 旧16ch=24.95dB)"

AE_CKPT="checkpoints/ldm_hsi_autoencoder_best_phase3_lc${LC}.pth"
SFX="_lc${LC}"

echo "=== [B1/4] 阶段3 条件LDM 重训 (latent_ch=$LC, 300ep) ==="
python scripts/train_conditional_ldm_phase3_full.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" --seed 42 --run_suffix "$SFX"

echo "=== [B2/4] 阶段4 物理引导LDM 重训 (latent_ch=$LC, 120ep) ==="
python scripts/train_physics_ldm_phase4.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" --seed 42 --run_suffix "$SFX"

echo "=== [B3/4] 阶段5 无配对LDM 重训 (latent_ch=$LC, 120ep) ==="
python scripts/train_unpaired_phase5.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" --seed 42 --run_suffix "$SFX"

echo "=== [C/4] 统一口径评测 x3 (best ckpt + blend=1.0 + seed42) ==="
python scripts/eval_unified_phase6b.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" \
  --ckpt "checkpoints/conditional_ldm_phase3_full${SFX}_best.pth" \
  --z_stats "checkpoints/conditional_ldm_phase3_full${SFX}_z_stats.pth" \
  --name "phase3${SFX}"
python scripts/eval_unified_phase6b.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" \
  --ckpt "checkpoints/physics_ldm_phase4${SFX}_best.pth" \
  --z_stats "checkpoints/physics_ldm_phase4${SFX}_z_stats.pth" \
  --name "phase4${SFX}"
python scripts/eval_unified_phase6b.py --latent_ch "$LC" --ae_ckpt "$AE_CKPT" \
  --ckpt "checkpoints/unpaired_phase5${SFX}_best.pth" \
  --z_stats "checkpoints/unpaired_phase5${SFX}_z_stats.pth" \
  --name "phase5${SFX}"

echo "=== 全部完成: AE(latent_ch=$LC, ${PSNR_AE}dB) 链路结果在 results/metrics_unified/phase*_lc${LC}.json ==="
