"""
L3 物理桥方案：'清晰→雾' 潜在扩散桥（Haze Bridge LDM）
文件名：bridge_ldm.py
对应：实验任务清单_v3.0.md【L3】

设计核心：把扩散前向过程从"清晰→高斯噪声"改为物理退化桥"清晰→雾"：
    q(z_t | z_clear, z_hazy) = s_t·z_clear + (1-s_t)·z_hazy + σ_t·ε
  - s_t: 1→0（t=0 纯清晰，t=T-1 纯雾），t 的大小即雾浓度，物理可解释；
  - σ_t: 0→σ_max（雾端注入随机性，是生成多样性的唯一来源）；
  - 模型任务：x0 预测——直接从 (z_t, z_hazy, t) 预测清晰 latent z_clear。

【重要设计决策】为什么用 x0 预测而不是 ε 预测（有本项目历史教训）：
  ε 预测的采样需要反解 x̂0 = (z_t − (1-s_t)·z_hazy − σ_t·ε̂) / s_t，
  在 t→T-1 处 s_t→0：除以 0 未定义，且 1/s_t 倍放大预测误差——与 §11 中
  DDIM 的 √ᾱ→0 除法放大误差（clamp bug 崩到 9dB 的根源）是同一病理。
  x0 预测全程无除法，采样步进只做凸组合：
    z_{t'} = s_{t'}·x̂0 + (1-s_{t'})·z_hazy + η·σ_{t'}·ε_new
  数值上按构造稳定，且 1 步采样也合法（退化为"带噪输入的回归"极限）。

调度形状 sched：
  - linear（默认）：s_t = 1 − t/(T−1)，雾浓度严格线性 = t/(T−1)，物理叙事最干净；
  - ddpm：复用 DDPM 的 ᾱ_t 形状重标定端点（注意 T=100, beta_end=0.02 时
    ᾱ_{T-1}≈0.37≠0，直接用 √ᾱ_t 当 s_t 到不了纯雾，必须重标定）。
"""
import numpy as np
import torch


class HazeBridge:
    """'清晰→雾' 潜在桥调度器：前向加噪 q_sample + 采样步进 step。"""

    def __init__(self, timesteps=100, sigma_max=0.5, sched="linear",
                 beta_start=1e-4, beta_end=0.02, device="cpu"):
        assert sched in ("linear", "ddpm"), f"未知调度形状: {sched}"
        self.timesteps = timesteps
        self.sigma_max = sigma_max
        self.sched = sched
        t = torch.arange(timesteps, dtype=torch.float64)
        if sched == "linear":
            # s: 1→0 严格线性；σ: 0→σ_max 按 sqrt（方差线性增长，布朗运动式）
            frac = t / (timesteps - 1)
            s = 1.0 - frac
            sigma = sigma_max * torch.sqrt(frac)
        else:
            # DDPM ᾱ 形状重标定：保证 s_0≈1, s_{T-1}=0 严格成立
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
            ac = torch.cumprod(1.0 - betas, dim=0)
            ac_end = ac[-1]
            s = torch.sqrt((ac - ac_end) / (1.0 - ac_end))
            sigma = sigma_max * torch.sqrt((1.0 - ac) / (1.0 - ac_end))
        self.s = s.float().to(device)
        self.sigma = sigma.float().to(device)

    def to(self, device):
        self.s = self.s.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def q_sample(self, z_clear, z_hazy, t, noise=None):
        """桥前向加噪：q(z_t|z_clear, z_hazy) = s_t·z_clear + (1-s_t)·z_hazy + σ_t·ε。

        t 为 (B,) 的 long 张量（与 DDPM.add_noise 用法对齐）。
        """
        if noise is None:
            noise = torch.randn_like(z_clear)
        s = self.s[t][:, None, None, None]
        sg = self.sigma[t][:, None, None, None]
        return s * z_clear + (1 - s) * z_hazy + sg * noise

    def start(self, z_hazy, noise=None):
        """采样起点：t=T-1 处 s=0，即 z_hazy + σ_max·ε（不再是纯噪声！）。"""
        if noise is None:
            noise = torch.randn_like(z_hazy)
        return z_hazy + self.sigma[-1] * noise

    def step(self, x0_hat, z_hazy, t_prev, noise=None, eta=0.0):
        """Euler 步进到 t_prev（凸组合，无除法，数值稳定）。

        z_{t'} = s_{t'}·x̂0 + (1-s_{t'})·z_hazy + η·σ_{t'}·ε_new
        eta=0 确定性（DDIM 类比）；eta=1 时状态分布与训练边缘分布 q(z_{t'}|x̂0,z_hazy) 一致。
        t_prev 为 python 标量（采样循环中逐网格点步进）。
        """
        if noise is None:
            noise = torch.zeros_like(x0_hat)
        s = self.s[t_prev]
        sg = self.sigma[t_prev]
        return s * x0_hat + (1 - s) * z_hazy + eta * sg * noise


def timestep_grid(timesteps, n_steps):
    """从 [0, T-1] 挑 n_steps 个边界，返回降序 t 列表（含 T-1 起点与 0 终点）。

    与 eval_ddim_guidance.py 的 ddim_timesteps 同一约定，保证少步评测口径可比。
    """
    idx = np.linspace(0, timesteps - 1, n_steps + 1).round().astype(int).tolist()
    uniq = []
    for v in idx:
        if v not in uniq:
            uniq.append(v)
    return uniq[::-1]  # 从大到小


def sample_bridge(model, z_hazy, bridge, n_steps, eta=0.0,
                  start_noise=None, device="cpu"):
    """桥采样：从'雾+σ_max·ε'出发沿去雾轨迹走到清晰 latent。

    每步：x̂0 = model(z_t, z_hazy, t)（x0 预测），再 Euler 投影回 t' 的桥状态。
    注意：x̂0 绝不 clamp 到 [-1,1]——那是图像空间做法，latent 空间（std≈1.19,
    mean≈0.29 归一化前）大量真实值超出该范围，clamp 会截断信号（§11 教训）。
    """
    ts = timestep_grid(bridge.timesteps, n_steps)
    z = bridge.start(z_hazy, noise=start_noise)
    with torch.no_grad():
        for i, t in enumerate(ts):
            t_batch = torch.full((z.size(0),), t, device=device, dtype=torch.long)
            x0_hat = model(z, z_hazy, t_batch)
            t_prev = ts[i + 1] if i + 1 < len(ts) else ts[-1]
            noise = torch.randn_like(z) if (eta > 0 and t_prev > 0) else None
            z = bridge.step(x0_hat, z_hazy, t_prev, noise=noise, eta=eta)
    return z


if __name__ == "__main__":
    # 自检：调度端点/单调性 + 采样形状
    b = HazeBridge(timesteps=100, sigma_max=0.5, sched="linear")
    assert abs(b.s[0] - 1.0) < 1e-6 and abs(b.s[-1]) < 1e-6
    assert abs(b.sigma[0]) < 1e-6 and abs(b.sigma[-1] - 0.5) < 1e-6
    assert (b.s[1:] <= b.s[:-1]).all() and (b.sigma[1:] >= b.sigma[:-1]).all()
    print(f"linear: s {b.s[0]:.4f}->{b.s[-1]:.4f}, sigma {b.sigma[0]:.4f}->{b.sigma[-1]:.4f}")
    b2 = HazeBridge(timesteps=100, sigma_max=0.5, sched="ddpm")
    assert abs(b2.s[-1]) < 1e-6 and (b2.s[1:] <= b2.s[:-1]).all()
    print(f"ddpm:   s {b2.s[0]:.4f}->{b2.s[-1]:.4f}, sigma {b2.sigma[0]:.4f}->{b2.sigma[-1]:.4f}")
    for n in (1, 10, 25, 50, 100):
        g = timestep_grid(100, n)
        assert g[0] == 99 and g[-1] == 0
        print(f"grid({n}): {len(g)} 个 t, 首={g[0]} 末={g[-1]}")
    print("HazeBridge 自检通过。")
