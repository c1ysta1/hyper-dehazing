"""多阶段实时进度站点：汇总入口 + 各阶段详情页 + 训练状态 API。

用法: python serve.py [port]
  /            -> webui/dashboard.html     (阶段2/3/4/5 汇总, 实时刷新)
  /api/status  -> JSON: 各阶段日志路径/进度/运行状态(基于日志mtime+进程扫描)
  其余路径按静态文件提供(项目根为站点根)。
"""
import http.server
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8010

# 各阶段日志文件匹配模式(取 mtime 最新者)
PHASE_PATTERNS = {
    "phase2": ["logs/psgnet_phase2_*/train_log.txt"],
    "phase3": ["logs/*ldm_phase3*/train_log.txt", "logs/conditional_ldm_phase3_*/train_log.txt"],
    "phase4": ["logs/*phase4*/train_log.txt"],
    "phase5": ["logs/*phase5*/train_log.txt", "logs/*unpaired*/train_log.txt",
               "logs/*phase5*.log", "logs/*unpaired*.log"],
}

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
RUNNING_STALE_SEC = 60  # 日志 60 秒内有更新视为训练中


def scan_running_scripts():
    """扫描 /proc 找正在运行的训练脚本, 返回 {phase_key: 脚本名}。"""
    running = {}
    try:
        pids = os.listdir("/proc")
    except OSError:
        return running
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        if "python" not in cmd:
            continue
        m = re.search(r"train_\w*phase(\d)", cmd)
        if m:
            running[f"phase{m.group(1)}"] = cmd.split("\x00")[1].split("/")[-1]
            continue
        if re.search(r"train_\w*unpaired\w*\.py", cmd):
            running["phase5"] = cmd.split("\x00")[1].split("/")[-1]
    return running


def phase_status():
    """汇总各阶段: 日志路径 / epoch 进度 / 是否运行 / 是否完成。"""
    running = scan_running_scripts()
    now = time.time()
    out = {}
    for key, patterns in PHASE_PATTERNS.items():
        info = {"found": False, "running": False, "done": False,
                "epoch": 0, "total": 0, "log": None, "script": None, "mtime_age": None}
        if key in running:
            info["running"] = True
            info["script"] = running[key]
        candidates = []
        for pat in patterns:
            candidates.extend(ROOT.glob(pat))
        if candidates:
            p = max(candidates, key=lambda x: x.stat().st_mtime)
            info["found"] = True
            info["log"] = "/" + str(p.relative_to(ROOT))
            age = now - p.stat().st_mtime
            info["mtime_age"] = round(age, 1)
            if age < RUNNING_STALE_SEC:
                info["running"] = True
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                text = ""
            eps = EPOCH_RE.findall(text)
            if eps:
                ep, tot = int(eps[-1][0]), int(eps[-1][1])
                info["epoch"], info["total"] = ep, tot
                if ep >= tot:
                    info["done"] = True
                    info["running"] = False
        out[key] = info
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard", "/dashboard.html"):
            self.path = "/webui/dashboard.html"
            return super().do_GET()
        if path == "/api/status":
            return self.api_status()
        return super().do_GET()

    def api_status(self):
        try:
            body = json.dumps({"now": time.time(), "phases": phase_status()}).encode()
            code, ctype = 200, "application/json; charset=utf-8"
        except Exception as e:  # 状态接口异常不影响静态服务
            body = json.dumps({"error": str(e)}).encode()
            code, ctype = 500, "application/json; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):  # 静默访问日志
        pass


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Serving {ROOT} at http://0.0.0.0:{PORT} (dashboard + /api/status)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
