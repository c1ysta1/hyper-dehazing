# 从 phase2 训练日志生成 dashboard 所需 history json
# 输出: checkpoints/history_psgnet_phase2.json
#   test_psnr: 训练日志 100 个 epoch 的真实值 + 末尾追加统一口径最终评测值(metrics_psgnet_phase2.json)
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

log_text = (ROOT / "logs" / "psgnet_phase2_0821_1523" / "train_log.txt").read_text(encoding="utf-8", errors="ignore")

train_loss, test_loss, test_psnr = [], [], []
for line in log_text.splitlines():
    m = re.match(r"Epoch \[\d+/\d+\] train_loss=([\d.]+) test_loss=([\d.]+) test_psnr=([\d.]+)dB", line)
    if m:
        train_loss.append(float(m.group(1)))
        test_loss.append(float(m.group(2)))
        test_psnr.append(float(m.group(3)))

# 统一口径最终评测值（best ckpt + 全测试集 264 对）
final = json.loads((ROOT / "results" / "metrics_psgnet_phase2.json").read_text(encoding="utf-8"))
final_psnr = final["psnr"]
print(f"日志解析: {len(test_psnr)} epochs, 训练曲线 best={max(test_psnr):.2f}dB")
print(f"统一口径最终评测: PSNR={final_psnr:.2f}dB (追加为曲线终点)")

history = {
    "train_loss": train_loss,
    "test_loss": test_loss,
    "test_psnr": test_psnr + [final_psnr],
    "final_metrics": final,
}
out = ROOT / "checkpoints" / "history_psgnet_phase2.json"
out.write_text(json.dumps(history, indent=2), encoding="utf-8")
print(f"saved: {out} (曲线点数 {len(history['test_psnr'])})")
