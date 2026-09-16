"""极简条件扩散策略（Mini-Diffusion Policy）核心模块 —— W4 Task 2。

把 Task 1 的**确定性**回归头（MLP + MSE → 唯一动作块 ``a = f(s)``）升级为基于
去噪扩散概率模型（DDPM）的**概率式**动作块生成器 ``p(a | s)``。

四块内容，对应扩散策略的四步数学
--------------------------------
1. :class:`DiffusionSchedule` —— 前向加噪的调度表 ``β_t`` / ``ᾱ_t`` 与全部闭式系数：

   * ``q_sample``：``a^t = sqrt(ᾱ_t)·a^0 + sqrt(1 - ᾱ_t)·ε``（训练用的加噪）
   * ``predict_x0_from_noise``：``a^0 = (a^t - sqrt(1-ᾱ_t)·ε_θ) / sqrt(ᾱ_t)``
   * ``q_posterior_mean_variance``：``q(a^{t-1} | a^t, a^0)`` 的闭式均值 / 方差（DDPM 采样用）
   * ``ddpm_step`` / ``ddim_step``：单步反向去噪（随机 ancestral / 确定性 DDIM）
2. :class:`NoisePredictor` —— 噪声预测网络 ``ε_θ(s, a^t, t)``：对状态 ``s`` 与扩散步
   ``t`` 双条件化（FiLM / AdaLN 调制 MLP），输出与动作块同形状的噪声。
3. :class:`DiffusionPolicy` —— 把「调度表 + 噪声网络 + 采样器」封装成与 Task 1 策略
   **完全同接口**的模块（``forward`` / ``forward_actions`` / ``training``），从而
   ``src/datasets.py`` 的闭环 rollout、``src/engine.py`` 的训练循环都能直接复用。
4. :func:`build_diffusion_policy` —— 由配置字典装配策略（``configs/diffusion_config.yaml``）。

与确定性动作块 BC 的关系
------------------------
* **动作空间约定**：扩散在归一化空间 ``[-1, 1]`` 上工作（``a / action_scale``），
  采样结果乘回 ``action_scale`` 并截断到 ``[-action_scale, action_scale]``；
* **标签约定**：动作块 ``(batch, K, action_dim)`` 与 Task 1 完全一致（同一份
  ``build_action_chunks`` 产物），因此两套实验的**数据、切分、评估协议逐样本对齐**；
* **唯一变量**：输出是"分布的样本"而非"条件均值"。这正是它能在**多峰**动作
  分布上避免"撞墙的平均动作"的根本原因。

术语速记
--------
* ``K``   —— 动作块长度（一次预测未来多少步动作，本模块输出形状 ``(B, K, A)``）；
* ``T``   —— 训练用的加噪步数（``num_train_timesteps``，默认 100）；
* ``N``   —— 推理时的去噪步数（``sample_steps``，默认 16，DDIM 可跨步采样）；
* ``ε_θ`` —— 噪声预测网络；反向采样时每步调用它一次，故"一次规划 = N 次前向"。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "DiffusionSchedule",
    "SinusoidalTimeEmbedding",
    "FiLMBlock",
    "NoisePredictor",
    "DiffusionPolicy",
    "build_diffusion_policy",
    "count_parameters",
]


# =============================================================================
# 1. 噪声调度表（前向加噪 + 反向去噪的全部闭式系数）
# =============================================================================
class DiffusionSchedule:
    """DDPM 噪声调度表：预计算 ``β_t``、``ᾱ_t`` 及其全部推导系数。

    Args:
        num_train_timesteps: 训练加噪步数 ``T``（默认 100）。
        schedule: ``cosine``（推荐，扩散策略论文采用）| ``linear`` | ``scaled_linear``。
        beta_start / beta_end: ``linear`` / ``scaled_linear`` 的端点。
        cosine_s: 余弦调度的偏移常数（``s=0.008``，避免 ``t=0`` 处 β 过小）。
        dtype: 系数张量精度（默认 float32；float64 便于数值自检）。

    Note:
        时间步编号一律 ``t ∈ {0, ..., T-1}``：``t=0`` 最干净、``t=T-1`` 最吵
        （``ᾱ_{T-1} ≈ 0``，接近纯高斯噪声）；采样时从 ``T-1`` 递减到 ``0``。
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        if self.num_train_timesteps <= 0:
            raise ValueError(f"num_train_timesteps 必须为正整数，当前 {num_train_timesteps}")
        self.schedule = str(schedule).lower()
        self.dtype = dtype

        t_total = self.num_train_timesteps
        if self.schedule == "cosine":
            steps = torch.arange(t_total + 1, dtype=torch.float64)
            f_t = torch.cos((steps / t_total + cosine_s) / (1.0 + cosine_s) * math.pi / 2.0) ** 2
            alphas_cumprod = f_t / f_t[0]
            betas = (1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(1e-8, 0.999)
        elif self.schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, t_total, dtype=torch.float64)
        elif self.schedule == "scaled_linear":
            betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, t_total, dtype=torch.float64
            ) ** 2
        else:
            raise ValueError(f"未知调度表 '{schedule}'，可选: cosine | linear | scaled_linear")

        self.betas = betas.to(dtype)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas.to(torch.float64), dim=0).to(dtype)
        # ᾱ_{t-1}：t=0 时约定为 1（即"加噪前"完全干净）
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=dtype), self.alphas_cumprod[:-1]], dim=0
        )

        # ---- 前向加噪 q(a^t | a^0) ----
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        # ---- 反向去噪 q(a^{t-1} | a^t, a^0) ----
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 采样时方差下限截断（t=0 处方差为 0，取对数需要 eps 保护）
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp_min(1e-20))
        self.posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * self.alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def to(self, device: torch.device) -> "DiffusionSchedule":
        """把全部系数张量搬到 ``device``（原地，返回 self）。"""
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def _extract(self, table: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """按时间步索引取值并广播到 ``ndim`` 维（形状 ``(B, 1, ..., 1)``）。"""
        out = table.gather(0, t.reshape(-1).long())
        return out.reshape(t.shape[0], *([1] * (ndim - 1)))

    def sample_timesteps(self, n_steps: Optional[int] = None) -> List[int]:
        """返回推理用的**降序**时间步序列（**必含 0**，即最后一步一定落在干净端点）。

        ``n_steps >= T`` → 全部 ``T`` 步；否则在 ``[0, T-1]`` 上等距取 ``n_steps`` 步
        （DDIM 跨步采样，正是"训练 100 步、推理 10~16 步"得以实现的原因）。
        ``n_steps = 1`` 时退化为"一步到位"：从最吵的 ``T-1`` 直接跳到 ``0``（共 2 步）。
        """
        total = self.num_train_timesteps
        if n_steps is None or int(n_steps) >= total:
            return list(range(total - 1, -1, -1))
        n = max(1, int(n_steps))
        idx = torch.linspace(total - 1, 0, n).round().long()
        # 保证末端一定是 0（干净端点），同时去重并按降序返回
        idx = torch.unique(torch.cat([idx, torch.zeros(1, dtype=torch.long)]))
        return [int(t) for t in reversed(idx.tolist())]

    # ------------------------------------------------------------------ #
    # 前向加噪 / 闭式反解
    # ------------------------------------------------------------------ #
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """前向加噪：``a^t = sqrt(ᾱ_t)·a^0 + sqrt(1 - ᾱ_t)·ε``。

        Args:
            x0: 干净动作块 ``(B, K, A)``（归一化空间 ``[-1, 1]``）。
            t: 时间步 ``(B,)``，取值 ``[0, T-1]``。
            noise: 可选的外部噪声 ``ε``（便于复现 / 自检）；``None`` 时现场采样。
            generator: 采样噪声用的随机源（配合 ``torch.Generator`` 保证可复现）。
        """
        if noise is None:
            noise = torch.randn(
                x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
            )
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x0.dim())
        sqrt_omacp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.dim())
        return sqrt_acp * x0 + sqrt_omacp * noise

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        clip: Optional[float] = None,
    ) -> torch.Tensor:
        """由噪声预测反解干净动作块：``a^0 = (a^t - sqrt(1-ᾱ_t)·ε) / sqrt(ᾱ_t)``。

        Args:
            clip: 非 ``None`` 时把结果截断到 ``[-clip, clip]``（归一化空间，
                即实践中的 "x0 clamp" 技巧，可抑制个别时刻的极端外推）。
        """
        sqrt_acp = self._extract(self.sqrt_alphas_cumprod, t, x_t.dim())
        sqrt_omacp = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.dim())
        x0 = (x_t - sqrt_omacp * eps) / sqrt_acp.clamp_min(1e-8)
        if clip is not None:
            x0 = x0.clamp(-float(clip), float(clip))
        return x0

    def q_posterior_mean_variance(
        self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``q(a^{t-1} | a^t, a^0) = N(mean, variance)`` 的闭式解（DDPM 原文式 (6)(7)）。

        Returns:
            ``(mean, variance, log_variance)``，形状均为 ``(B, K, A)``。
        """
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.dim()) * x0
            + self._extract(self.posterior_mean_coef2, t, x_t.dim()) * x_t
        )
        variance = self._extract(self.posterior_variance, t, x_t.dim())
        log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.dim())
        return mean, variance, log_variance

    # ------------------------------------------------------------------ #
    # 单步反向去噪（采样器的原子操作）
    # ------------------------------------------------------------------ #
    def ddpm_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        clip: Optional[float] = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDPM 随机 ancestral 一步：``a^{t-1} = μ_θ(a^t, a^0) + σ_t·z``。

        每个时刻都注入新噪声 ``z ~ N(0, I)``，因此**同一次规划内逐时刻随机**；
        完整的 DDPM 采样需要 ``T`` 步（默认 100 次网络前向）。
        """
        x0 = self.predict_x0_from_noise(x_t, t, eps, clip=clip)
        mean, variance, _ = self.q_posterior_mean_variance(x0, x_t, t)
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        # t = 0 时后验方差为 0（已到达干净端点，不再加噪）
        nonzero = (t.reshape(-1, 1, 1) > 0).to(x_t.dtype)
        return mean + variance.sqrt() * noise * nonzero

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eps: torch.Tensor,
        eta: float = 0.0,
        clip: Optional[float] = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM 一步去噪（Song et al. 2020），支持跨步采样。

        公式:: ``a^{t-1} = sqrt(ᾱ_{t-1})·â^0 + sqrt(1 - ᾱ_{t-1} - σ²)·ε + σ·z``，
        ``σ = η·sqrt((1-ᾱ_{t-1})/(1-ᾱ_t))·sqrt(1 - ᾱ_t/ᾱ_{t-1})``；``η=0`` 时
        ``σ=0``，更新**完全确定**（给定 ``ε_θ`` 唯一）。``t_prev < 0`` 表示干净端点
        （``ᾱ_{-1} := 1``）。

        Note:
            ``η=0`` 并不等于"策略确定"：起始噪声 ``a^N ~ N(0, I)`` 每次规划都会重新
            采样，不同随机数仍给出**同一分布的不同样本**——多峰正是由此体现。
        """
        t_prev_idx = t_prev.reshape(-1).clamp_min(0).long()
        acp_prev = self.alphas_cumprod.gather(0, t_prev_idx)
        acp_prev = torch.where(
            t_prev.reshape(-1) < 0, torch.ones_like(acp_prev), acp_prev
        ).reshape(-1, 1, 1)
        acp_t = self._extract(self.alphas_cumprod, t, x_t.dim())
        x0 = self.predict_x0_from_noise(x_t, t, eps, clip=clip)

        eta = float(max(0.0, eta))
        if eta <= 0.0:
            return acp_prev.sqrt() * x0 + (1.0 - acp_prev).clamp_min(0.0).sqrt() * eps
        sigma = (
            eta
            * ((1.0 - acp_prev) / (1.0 - acp_t).clamp_min(1e-8)).sqrt()
            * (1.0 - acp_t / acp_prev.clamp_min(1e-8)).clamp_min(0.0).sqrt()
        )
        noise = torch.randn(x_t.shape, device=x_t.device, dtype=x_t.dtype, generator=generator)
        return (
            acp_prev.sqrt() * x0
            + (1.0 - acp_prev - sigma ** 2).clamp_min(0.0).sqrt() * eps
            + sigma * noise
        )


# =============================================================================
# 2. 噪声预测网络 ε_θ(s, a^t, t)
# =============================================================================
def count_parameters(module: nn.Module) -> int:
    """统计可训练参数量。"""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


class SinusoidalTimeEmbedding(nn.Module):
    """扩散步 ``t`` 的正弦位置编码（Transformer 式），把标量标量扩成 ``dim`` 维向量。

    扩散步是**离散的全局标量**：直接用 ``t/T`` 会让网络难以分辨相邻步；正弦编码把
    它展开成多频三角函数，网络可用线性组合逼近任意"关于 t 的平滑函数"。
    """

    def __init__(self, dim: int = 64, max_period: float = 1e4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``(B,)`` → ``(B, dim)``。"""
        t = t.reshape(-1).float()
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(1, half)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:                       # 奇数维时补一列零，保证输出维度正确
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FiLMBlock(nn.Module):
    """FiLM（Feature-wise Linear Modulation / AdaLN）残差块。

    ``h = h + MLP( (1 + γ(c)) * LN(h) + β(c) )``：条件向量 ``c`` 只调制尺度 ``γ``
    与偏置 ``β``，"谁在什么状态、什么扩散步"这一信息因此能作用于每一层中间特征。
    """

    def __init__(self, hidden_dim: int, cond_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale_shift = nn.Linear(cond_dim, hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            _resolve_activation(activation)(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        # 零初始化调制层：训练初期等价于恒等映射，深层扩散网络更易优化
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return h + self.mlp(self.norm(h) * (1.0 + scale) + shift)


def _resolve_activation(name: str):
    """按名称解析激活函数（与 ``src/networks.py`` 保持同一命名空间）。"""
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"未知激活函数 '{name}'，可选: {list(mapping)}")
    return mapping[key]


class NoisePredictor(nn.Module):
    """条件噪声预测网络 ``ε_θ(s, a^t, t)``。

    Args:
        state_dim: 状态维度（Pendulum 为 3）。
        action_dim: 动作维度（Pendulum 为 1）。
        chunk_size: 动作块长度 ``K``。
        hidden_dim: 隐藏层宽度。
        num_layers: FiLM 残差块层数。
        time_embed_dim: 扩散步正弦编码维度。
        activation: 激活函数名（``gelu`` / ``silu`` / ``relu`` ...）。

    输入输出::

        输入: state (B, state_dim) | x_t (B, K, action_dim) | t (B,)
        输出: 预测噪声 eps (B, K, action_dim)，与输入 x_t 同形状

    结构：``x_t`` 展平后线性嵌入 → 与 ``state`` / ``t`` 的条件向量相加 →
    ``num_layers`` 个 FiLM 残差块 → 线性输出头 → 还原成 ``(B, K, action_dim)``。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        chunk_size: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embed_dim: int = 64,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = max(1, int(chunk_size))
        self.hidden_dim = int(hidden_dim)
        self.action_flat_dim = self.chunk_size * self.action_dim

        act = _resolve_activation(activation)
        self.x_embed = nn.Sequential(
            nn.Linear(self.action_flat_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_embed = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            act(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            FiLMBlock(hidden_dim, hidden_dim, activation) for _ in range(max(1, int(num_layers)))
        )
        self.out = nn.Linear(hidden_dim, self.action_flat_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """预测噪声 ``ε_θ``；``t`` 可为 ``(B,)`` 或标量张量。"""
        if x_t.dim() == 2:                                  # (B, A) → (B, 1, A)
            x_t = x_t.unsqueeze(1)
        batch = x_t.shape[0]
        if t.dim() == 0:
            t = t.expand(batch)
        cond = self.state_embed(state) + self.time_mlp(self.time_embed(t))
        h = self.x_embed(x_t.reshape(batch, -1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out(h).reshape(batch, self.chunk_size, self.action_dim)



# =============================================================================
# 3. 扩散策略：调度表 + 噪声网络 + 采样器（与 Task 1 策略同接口）
# =============================================================================
class DiffusionPolicy(nn.Module):
    """条件扩散动作块生成器，接口与 Task 1 的确定性策略**完全一致**。

    对上层（``src/datasets.py`` 的闭环 rollout、``src/engine.py`` 的训练循环）
    暴露：

    * ``forward(state)`` / ``forward_actions(state)`` → ``(B, K, action_dim)``
      真实动作块（内部完成 ``N`` 步反向去噪 + 反归一化 + 截断）；
    * ``training`` 属性、``train()`` / ``eval()``、``state_dict()`` 等标准 ``nn.Module`` 语义；
    * ``p_losses(state, actions, valid)`` → 训练用的**噪声预测 MSE**（DDPM 简化目标）。

    归一化约定（扩散论文的标准做法）：网络在 ``[-1, 1]`` 空间上预测，真实动作为
    ``a = action_scale · a_norm``；因此 ``x0 = clamp(a / action_scale, -1, 1)``。

    Args:
        noise_net: :class:`NoisePredictor`。
        schedule: :class:`DiffusionSchedule`。
        action_scale: 动作缩放（Pendulum 动作空间 ``[-2, 2]`` → 2.0）。
        sample_steps: 推理去噪步数 ``N``（10 / 16 / 32 / 100...）。
        sampler: ``ddim``（确定性、支持跨步，默认）| ``ddpm``（随机 ancestral，需 ``N=T``）。
        eta: DDIM 的随机性系数（``0`` 完全确定；``>0`` 逐步注入噪声）。
        clip_x0: ``x0`` 预测截断（归一化空间，默认 1.0）。
        seed: 固定采样噪声的随机种子（``None`` = 每次规划都重新随机，即真正的采样）。
    """

    def __init__(
        self,
        noise_net: NoisePredictor,
        schedule: DiffusionSchedule,
        action_scale: float = 2.0,
        sample_steps: int = 16,
        sampler: str = "ddim",
        eta: float = 0.0,
        clip_x0: Optional[float] = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.noise_net = noise_net
        self.schedule = schedule
        self.action_scale = float(action_scale)
        self.chunk_size = int(noise_net.chunk_size)
        self.action_dim = int(noise_net.action_dim)
        self.sample_steps = int(sample_steps)
        self.sampler = str(sampler).lower()
        if self.sampler not in ("ddim", "ddpm"):
            raise ValueError(f"未知采样器 '{sampler}'，可选: ddim | ddpm")
        self.eta = float(eta)
        self.clip_x0 = clip_x0
        self.seed = seed
        self._generator: Optional[torch.Generator] = None
        # 统计量：每次规划实际调用了多少次噪声网络（= 推理步数），供"推理开销"度量
        self.n_network_calls = 0

    # ------------------------------------------------------------------ #
    # 归一化 / 反归一化
    # ------------------------------------------------------------------ #
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """真实动作 → 扩散空间 ``[-1, 1]``（并截断到合法范围）。"""
        return (actions / max(self.action_scale, 1e-8)).clamp(-1.0, 1.0)

    def denormalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """扩散空间 ``[-1, 1]`` → 真实动作（并截断到动作空间上限）。"""
        return (x * self.action_scale).clamp(-self.action_scale, self.action_scale)

    # ------------------------------------------------------------------ #
    # 训练目标：噪声预测 MSE（DDPM 简化损失）
    # ------------------------------------------------------------------ #
    def p_losses(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算 ``E_{t, ε} [ ||ε - ε_θ(s, a^t, t)||² ]``（掩码掉零填充位置）。

        每个 batch 现场随机采样扩散步 ``t ~ U{0, ..., T-1}`` 与噪声 ``ε ~ N(0, I)``，
        然后按闭式加噪得到 ``a^t``，回归目标就是**自己注入的那个噪声**。

        Args:
            state: ``(B, state_dim)``。
            actions: ``(B, K, action_dim)`` 真实动作块（未归一化）。
            valid: ``(B, K)`` bool 掩码（``False`` = 轨迹末端零填充，不参与损失）。
            generator: 噪声随机源（可复现性）。
            t: 可选的外部时间步（自检用）。

        Returns:
            ``(loss, info)``；``info`` 含 ``eps_mse`` / ``x0_mse``（干净的"动作块重建
            误差"，便于与 Task 1 的标签 MSE 直觉对齐）。
        """
        actions = self.normalize_actions(actions)
        if actions.dim() == 2:                       # (B, A) → (B, 1, A)
            actions = actions.unsqueeze(1)
        batch = actions.shape[0]
        if t is None:
            t = torch.randint(
                0, self.schedule.num_train_timesteps, (batch,),
                device=actions.device, generator=generator,
            )
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype,
                            generator=generator)
        x_t = self.schedule.q_sample(actions, t, noise=noise)
        eps_pred = self.noise_net(state, x_t, t)

        if valid is None:
            weight = torch.ones_like(actions)
        else:
            weight = valid.to(actions.dtype).reshape(batch, self.chunk_size, 1).expand_as(actions)
        denom = weight.sum().clamp_min(1.0)
        loss = ((eps_pred - noise) ** 2 * weight).sum() / denom

        with torch.no_grad():
            x0_pred = self.schedule.predict_x0_from_noise(x_t, t, eps_pred, clip=self.clip_x0)
            x0_mse = ((x0_pred - actions) ** 2 * weight).sum() / denom
        return loss, {"eps_mse": loss.detach(), "x0_mse": x0_mse.detach()}



    # ------------------------------------------------------------------ #
    # 反向采样（推理）
    # ------------------------------------------------------------------ #
    def _generator_for(self, device: torch.device) -> Optional[torch.Generator]:
        """返回采样噪声随机源：``seed=None`` → 用全局随机源（真随机采样）。"""
        if self.seed is None:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self.seed))
        return self._generator

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        n_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """从纯高斯噪声 ``a^N ~ N(0, I)`` 出发，反向去噪出干净动作块。

        每次调用消耗 ``N`` 次 ``ε_θ`` 前向（``N = n_steps``；``ddpm`` 采样器下
        ``N = T``）——这正是扩散策略"推理更贵"的量化来源。

        Args:
            state: ``(B, state_dim)``。
            n_steps: 覆盖实例默认的推理步数（用于"步数 vs 效果"消融）。
            generator: 采样噪声随机源（``None`` 时用 ``self.seed`` 或全局随机源）。
            return_trajectory: ``True`` 时额外返回整条去噪轨迹 ``(N+1, B, K, A)``
                （教学可视化用：噪声 → 动作块的逐渐成形）。

        Returns:
            ``(B, K, action_dim)`` 归一化动作块（``return_trajectory`` 时返回元组）。
        """
        batch = state.shape[0]
        device, dtype = state.device, state.dtype
        if generator is None:
            generator = self._generator_for(device)

        x = torch.randn(batch, self.chunk_size, self.action_dim, device=device, dtype=dtype,
                        generator=generator)
        trajectory = [x] if return_trajectory else None

        if self.sampler == "ddpm":
            times = list(range(self.schedule.num_train_timesteps - 1, -1, -1))
        else:
            times = self.schedule.sample_timesteps(n_steps or self.sample_steps)

        for i, t_scalar in enumerate(times):
            t = torch.full((batch,), int(t_scalar), device=device, dtype=torch.long)
            eps = self.noise_net(state, x, t)
            self.n_network_calls += 1
            if self.sampler == "ddpm":
                x = self.schedule.ddpm_step(x, t, eps, clip=self.clip_x0, generator=generator)
            else:
                t_prev = times[i + 1] if i + 1 < len(times) else -1
                x = self.schedule.ddim_step(
                    x, t, torch.full((batch,), int(t_prev), device=device, dtype=torch.long),
                    eps, eta=self.eta, clip=self.clip_x0, generator=generator,
                )
            if return_trajectory:
                trajectory.append(x)
        return (x, torch.stack(trajectory)) if return_trajectory else x

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """与确定性策略同名同义：一次前向（采样）即得到真实动作块 ``(B, K, A)``。"""
        return self.forward_actions(state)

    def forward_actions(self, state: torch.Tensor) -> torch.Tensor:
        """闭环 rollout 依赖的统一接口：采样 → 反归一化 → 截断到动作空间。"""
        was_training = self.training
        self.eval()                                   # 采样阶段禁用 dropout/BN 随机性
        x0 = self.sample(state)
        if was_training:
            self.train()
        return self.denormalize_actions(x0)

    def set_sample_steps(self, n_steps: int) -> "DiffusionPolicy":
        """调整推理步数 ``N``（消融实验用），返回 self 便于链式调用。"""
        self.sample_steps = max(1, int(n_steps))
        return self



    @property
    def sample_cost(self) -> int:
        """单次规划的网络前向次数（与 Task 1 的 1 次形成直观对比）。"""
        return (self.schedule.num_train_timesteps if self.sampler == "ddpm"
                else len(self.schedule.sample_timesteps(self.sample_steps)))


# =============================================================================
# 4. 装配：由配置字典构建扩散策略
# =============================================================================
def build_diffusion_policy(config: Dict[str, Any]) -> DiffusionPolicy:
    """按配置构建 :class:`DiffusionPolicy`。

    配置段（见 ``configs/diffusion_config.yaml``）::

        diffusion:
          num_train_timesteps: 100      # T：训练加噪步数
          schedule: cosine              # cosine | linear | scaled_linear
          sample_steps: 16              # N：推理去噪步数（DDIM）
          sampler: ddim                 # ddim | ddpm
          eta: 0.0                      # DDIM 随机性；0 = 确定性
          clip_x0: 1.0                  # x0 预测截断（归一化空间）
          seed: null                    # null = 每次规划都重新采样（真随机）
        noise_net:
          hidden_dim: 128
          num_layers: 3
          time_embed_dim: 64
          activation: gelu

    ``model`` 段复用 Task 1 的字段（``state_dim`` / ``action_dim`` /
    ``chunk_size`` / ``action_scale``），因此两套实验共享同一份数据形状约定。
    """
    diff_cfg = config.get("diffusion", {}) or {}
    net_cfg = config.get("noise_net", {}) or {}
    model_cfg = config.get("model", {}) or {}

    schedule = DiffusionSchedule(
        num_train_timesteps=int(diff_cfg.get("num_train_timesteps", 100)),
        schedule=str(diff_cfg.get("schedule", "cosine")),
        beta_start=float(diff_cfg.get("beta_start", 1e-4)),
        beta_end=float(diff_cfg.get("beta_end", 0.02)),
        cosine_s=float(diff_cfg.get("cosine_s", 0.008)),
    )
    noise_net = NoisePredictor(
        state_dim=int(model_cfg.get("state_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 1)),
        chunk_size=int(model_cfg.get("chunk_size", 1)),
        hidden_dim=int(net_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 128))),
        num_layers=int(net_cfg.get("num_layers", 3)),
        time_embed_dim=int(net_cfg.get("time_embed_dim", 64)),
        activation=str(net_cfg.get("activation", "gelu")),
    )
    seed = diff_cfg.get("seed", None)
    return DiffusionPolicy(
        noise_net=noise_net,
        schedule=schedule,
        action_scale=float(model_cfg.get("action_scale", 2.0)),
        sample_steps=int(diff_cfg.get("sample_steps", 16)),
        sampler=str(diff_cfg.get("sampler", "ddim")),
        eta=float(diff_cfg.get("eta", 0.0)),
        clip_x0=diff_cfg.get("clip_x0", 1.0),
        seed=None if seed is None else int(seed),
    )
