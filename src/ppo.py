"""手写 PPO（PPO-Clip + GAE-λ）核心模块 —— W8 Task 1。

在 **Pendulum-v1**（连续动作空间 ``[-2.0, 2.0]``，单回合 200 步）上从零实现
Proximal Policy Optimization：Actor-Critic 采样 rollout → GAE-λ 估计优势 →
PPO-Clip 损失 → 多轮小批量（mini-batch）更新。

一句话数学::

    r_t(θ)      = exp( logπ_θ(a_t|s_t) − logπ_{θ_old}(a_t|s_t) )          # 重要性采样比值
    L^{CLIP}(θ) = − E_t [ min( r_t·Â_t , clip(r_t, 1−ε, 1+ε)·Â_t ) ]       # 策略项（负号 = 把"最大化"译成"最小化"）
    L^{VF}(φ)   = ½ E_t [ ( V_φ(s_t) − Target_t )² ]                       # 价值项
    L^{total}   = L^{CLIP} + c₁·L^{VF} − c₂·H(π_θ)                         # 熵正则（>0 鼓励探索）

    δ_t      = r_t + γ·V(s_{t+1}) − V(s_t)                                 # TD 残差
    Â_t      = δ_t + (γλ)·(1 − done_t)·Â_{t+1}                             # GAE-λ（t 从 T−1 倒序递推）
    Target_t = Â_t + V(s_t)                                                # 价值回归目标

模块内八个可独立测试的部件
--------------------------
1. :class:`Actor` —— 连续动作策略 ``π_θ(a|s) = N(μ_θ(s), σ²)``，``σ = exp(log_std)``，
   ``log_std`` 为**可学习参数**并夹紧到 ``[LOG_STD_MIN, LOG_STD_MAX]`` 防数值退化。
2. :class:`Critic` —— 状态价值 ``V_φ(s)``（标量，输出形 ``(B, 1)``）。
3. :class:`ActorCritic` —— Actor-Critic 容器：默认**独立双网络**（作业要求），
   也可配置 ``shared_backbone=True`` 共享主干、双头输出；统一提供
   ``mu / distribution / log_prob / entropy / act / value`` 接口。
4. :class:`RolloutBuffer` —— 环形缓存 ``(s, a_raw, logπ_old, r, terminated, truncated, V)``。
5. :func:`compute_gae` —— 纯函数 GAE-λ（倒序递推 + ``done`` 掩码 + rollout 边界自举），
   与朴素 ``O(T²)`` 实现逐位一致（见 ``tests/test_ppo.py``）。
6. :func:`ppo_losses` —— 手写 PPO-Clip / 价值 / 熵三项损失，并给出
   ``ratio / approx_kl / clip_fraction`` 等诊断量。
7. :func:`ppo_update` —— ``update_epochs`` × mini-batch 的完整更新（梯度裁剪、
   优势标准化、可选 KL 早停），返回全部训练诊断量。
8. :class:`RolloutCollector` / :func:`evaluate_policy` —— 采样与独立评估
   （含手写 :class:`RunningMeanStd` 状态归一化）。

连续动作空间的坑：``clip(action, -2, 2)`` 时 ``log_prob`` 要注意什么（W8 思考题）
--------------------------------------------------------------------------------
采样得到 ``a_raw ~ N(μ, σ²)`` 后，环境只接受 ``[-2, 2]``，因此必须
``a_env = clip(a_raw, −2, 2)`` 再交给环境**执行**；但 ``log_prob`` 必须用
**截断前的** ``a_raw`` 计算。三个理由（单元测试逐条量化）：

1. **密度函数的"变量"不能被替换**。``logπ(a_env)`` 是把 ``a_env`` 当成高斯变量
   算出来的密度，而 ``a_env`` 并不服从该高斯 —— 它在 ``±2`` 处还持有概率质量
   ``P(a_raw < −2)`` / ``P(a_raw > 2)``（连续部分 + 两个点质量的混合，根本没有密度）。
   策略梯度定理要求 ``∇_θ logπ_θ(a)·Â`` 中 ``a`` 与 ``π_θ`` 描述**同一个随机变量**：
   ``E_{a~π}[∇logπ(a)·Â] = ∇J`` 这一步的关键正是"积分变量与密度一致"。
2. **用 ``clip(a)`` 会引入系统性（非零均值）偏置梯度**。高斯得分函数满足
   ``E_{a~N(μ,σ²)}[∇_μ logπ(a)] = E[(a−μ)/σ²] = 0``（无偏地"零信号"），换成
   ``clip(a)`` 后期望变成 ``(E[clip(a)] − μ)/σ²``：只要 ``μ`` 越过边界就变成**非零
   伪梯度**——``μ`` 在界外时 ``E[clip(a)]`` 恒等于边界值，于是伪梯度 ≈ ``(边界−μ)/σ²``
   把 ``μ`` 硬拽向边界（``test_clipped_action_log_prob_is_biased`` 实测 ``μ=±5, σ=0.1``
   时 ≈ ∓300，而正确做法恒为零均值、量级小两个数量级）。它**与奖励无关**，
   却会在 ``×Â_t`` 后主导更新方向。
3. **PPO 的重要性采样比值会直接坏掉**。更新时要重算 ``logπ_θ(a_t)``：若 buffer 里存的
   是 ``a_env``，则"参数还没变的第 0 个 epoch"里 ``r_t ≠ 1``（新旧 logπ 算在**两个不同
   动作点**上），``min(r·Â, clip(r)·Â)`` 的裁剪区间失去基准意义；存 ``a_raw`` 则恒有
   ``r_t = 1``（``test_ratio_is_one_only_when_raw_action_is_stored``）。

**本模块的做法**：把 ``a_raw`` 视为策略采样出的动作变量，``clip`` 只是环境侧一个确定的
（不可导）变换 —— 等价于"环境奖励函数换成 ``r(clip(a_raw))``"，策略梯度定理依旧成立；
因此 ``a_raw`` 进 buffer、``a_env`` 只进 ``env.step``（buffer 字段就命名为 ``actions_raw``；
``RolloutCollector.collect`` 同时记录 ``last_actions_raw`` / ``last_actions_env``，
单测 ``test_buffer_keeps_raw_action_while_env_gets_clipped_action`` 断言
``a_env == clip(a_raw, low, high)`` 且 buffer 里存的是 ``a_raw``）。若希望策略
**自身**输出受限动作，数学上正确的替代是 **tanh 压缩高斯**（``a_env = tanh(a_raw)`` 且
``logπ`` 减去 ``Σ log|∂a_env/∂a_raw|`` 的 Jacobian 项），本模块保留为扩展方向，配置里默认
``model.action_transform: clip``。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

try:  # 兼容"作为包导入"与"直接运行"两种场景
    from .reinforce import (
        DEFAULT_MAX_STEPS,
        count_parameters,
        env_max_steps,
        make_env,
    )
except ImportError:  # pragma: no cover
    from reinforce import (  # type: ignore
        DEFAULT_MAX_STEPS,
        count_parameters,
        env_max_steps,
        make_env,
    )

__all__ = [
    "LOG_STD_MIN",
    "LOG_STD_MAX",
    "RunningMeanStd",
    "gaussian_log_prob",
    "entropy_of_normal",
    "Actor",
    "Critic",
    "ActorCritic",
    "build_actor_critic",
    "build_ppo_optimizer",
    "RolloutBuffer",
    "compute_gae",
    "policy_loss_clip",
    "value_loss",
    "ppo_losses",
    "ppo_update",
    "RolloutCollector",
    "evaluate_policy",
    "explained_variance",
    "make_env",
    "env_max_steps",
    "count_parameters",
]


# =============================================================================
# 0. 常量与内部工具
# =============================================================================
LOG_STD_MIN = -20.0     # log σ 下界：σ ≈ 2e-9，远离 Normal 的数值病态区
LOG_STD_MAX = 2.0       # log σ 上界：σ ≈ 7.39，远大于 Pendulum 的动作幅度 2.0

DEFAULT_ACTION_LOW = -2.0    # Pendulum-v1 动作下界
DEFAULT_ACTION_HIGH = 2.0    # Pendulum-v1 动作上界


def _resolve_activation(name: str) -> type:
    """按名称解析激活函数层（与 ``src/networks.py`` / ``src/reinforce.py`` 同一套可选值）。"""
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"未知激活函数 '{name}'，可选: {list(mapping)}")
    return mapping[key]


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    activation: str = "relu",
) -> nn.Sequential:
    """构建含 ``num_layers`` 个隐藏层的 MLP（输出层**不带激活**）。

    输出层不做压缩：Actor 的均值 ``μ(s)`` 需要能取到边界之外的值（越界由环境侧 clip
    处理），Critic 的价值更不该被 tanh 限制在 ``(−1, 1)``。
    """
    layers: List[nn.Module] = []
    last = int(in_dim)
    for _ in range(max(1, int(num_layers))):
        layers += [nn.Linear(last, int(hidden_dim)), _resolve_activation(activation)()]
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


def _as_bound(value: Union[float, Sequence[float], torch.Tensor], device, dtype) -> torch.Tensor:
    """把动作边界规范化为可广播张量（标量 → 形状 ``(1,)``）。"""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype).clone()
    if isinstance(value, (list, tuple, np.ndarray)):
        return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device, dtype=dtype)
    return torch.as_tensor([float(value)], device=device, dtype=dtype)


def gaussian_log_prob(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """**手写**对角高斯对数密度（对动作维求和，保持 ``(B, 1)``）::

        log N(a; μ, σ²) = −½ Σ_j [ (a_j − μ_j)² / σ_j² + log(2π σ_j²) ]

    与 ``torch.distributions.Normal(mu, std).log_prob(a).sum(-1, keepdim=True)`` 逐位一致
    （``test_handwritten_gaussian_log_prob_matches_torch``）；单独抽出来是为了让
    "``log_prob`` 究竟算在哪个动作上"这件事在代码里一眼可见。
    """
    var = std.pow(2)
    log_prob = -0.5 * ((action - mean).pow(2) / var + torch.log(2.0 * math.pi * var))
    return log_prob.sum(dim=-1, keepdim=True)


def entropy_of_normal(log_std: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """**手写**对角高斯微分熵（对动作维求和）::

        H[N(μ, σ²)] = ½ Σ_j [ 1 + log(2π σ_j²) ] = Σ_j ( ½(1 + log 2π) + log σ_j )

    ``log_std`` 已夹紧到 ``[LOG_STD_MIN, LOG_STD_MAX]``，故熵有界：
    ``[LOG_STD_MIN + ½(1+log2π)]·A ≤ H ≤ [LOG_STD_MAX + ½(1+log2π)]·A``。
    """
    ent = 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std
    return ent.sum(dim=-1, keepdim=keepdim)

# =============================================================================
# 0.5 状态归一化：在线均值 / 方差
# =============================================================================
class RunningMeanStd:
    """在线均值 / 方差估计（Welford 并行更新），用于**状态归一化**。

    为什么 Pendulum 需要它：状态是 ``[cos θ, sin θ, θ̇]``，前两维在 ``[-1, 1]`` 而
    ``θ̇ ∈ [-8, 8]`` —— 量纲差近 8 倍。不归一化时网络把容量大量花在拟合角速度的尺度上，
    ``V(s)`` 回归的条件数很差；归一化后三维持平（消融开关：
    ``--set model.normalize_obs=false``）。

    实现要点：用"并行方差合成"（Chan's parallel algorithm）把一批样本的
    ``(batch_mean, batch_var, batch_count)`` 合并进已有统计量，**无需保存历史样本**；
    ``count`` 初始化为 ``ε=1e-4``（而非 0）以避免前几步除零、并抑制统计量早期抖动。
    """

    def __init__(self, shape: Union[int, Sequence[int]] = (), epsilon: float = 1e-4, clip: float = 10.0) -> None:
        if shape is None:
            shape_t: Tuple[int, ...] = ()
        elif isinstance(shape, (int, np.integer)):
            shape_t = (int(shape),)
        else:
            shape_t = tuple(int(s) for s in shape)
        self.shape: Tuple[int, ...] = shape_t
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)
        self.clip = float(clip)      # 归一化后截断到 ±clip（防离群状态毁掉单步前向）

    @property
    def std(self) -> np.ndarray:
        """标准差（``+1e-8`` 防除零）。"""
        return np.sqrt(self.var) + 1e-8

    def update(self, x: Union[np.ndarray, torch.Tensor, Sequence[float], float]) -> None:
        """用一批样本（形 ``(N, *shape)``；单样本会自动补 batch 维）更新统计量。"""
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == len(self.shape):
            arr = arr.reshape((1,) + arr.shape)
        batch_count = int(arr.shape[0]) if arr.ndim else 1
        if batch_count <= 0:
            return
        arr = arr.reshape(batch_count, *self.shape) if self.shape else arr.reshape(batch_count, 1)
        self._update_from_moments(arr.mean(axis=0), arr.var(axis=0), batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """把一批样本的 ``(均值, 方差, 样本数)`` 合并进当前统计量。"""
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def normalize_np(self, x: Union[np.ndarray, Sequence[float]], clip: Optional[float] = None) -> np.ndarray:
        """``(x − μ) / (σ + 1e-8)``，可选截断到 ``±clip``（默认取构造参数；``clip<=0`` 关闭）。"""
        out = (np.asarray(x, dtype=np.float64) - self.mean) / self.std
        limit = self.clip if clip is None else float(clip)
        if limit and limit > 0:
            out = np.clip(out, -limit, limit)
        return out.astype(np.float32)

    def normalize_torch(self, x: torch.Tensor, clip: Optional[float] = None) -> torch.Tensor:
        """Tensor 版归一化：与 :meth:`normalize_np` 数值一致，避免每步 numpy↔torch 往返。"""
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        out = (x - mean) / std
        limit = self.clip if clip is None else float(clip)
        if limit and limit > 0:
            out = torch.clamp(out, -limit, limit)
        return out

    def state_dict(self) -> Dict[str, Any]:
        """导出统计量（随检查点保存：续训与评估必须用同一套归一化参数）。"""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "shape": list(self.shape),
            "clip": float(self.clip),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        """恢复统计量（``None`` 时保持当前值不变）。"""
        if not state:
            return
        self.mean = np.asarray(state["mean"], dtype=np.float64).reshape(self.shape)
        self.var = np.asarray(state["var"], dtype=np.float64).reshape(self.shape)
        self.count = float(state.get("count", 1.0))
        if "clip" in state:
            self.clip = float(state["clip"])

# =============================================================================
# 1. Actor / Critic 网络
# =============================================================================
class Actor(nn.Module):
    """连续动作策略网络 ``π_θ(a|s) = N(μ_θ(s), σ²)``，``σ = exp(log_std)``。

    Args:
        state_dim: 状态维度（Pendulum 为 3）。
        action_dim: 动作维度（Pendulum 为 1）。
        hidden_dim / num_layers / activation: MLP 结构。
        log_std_init: ``log_std`` 初值。Pendulum 常用 ``-0.5``（σ ≈ 0.61，
            相对动作幅度 2.0 是"适度探索"）。
        log_std_min / log_std_max: ``log_std`` 夹紧区间，防止 σ → 0（熵崩塌、
            梯度爆炸）或 σ → ∞（动作完全随机）。
        action_low / action_high: 环境动作边界，只用于**执行侧**截断，
            **不参与** ``log_prob`` 计算（见模块 docstring 的思考题）。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        log_std_init: float = -0.5,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        action_low: float = DEFAULT_ACTION_LOW,
        action_high: float = DEFAULT_ACTION_HIGH,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        self.mu_net = _mlp(self.state_dim, self.action_dim, self.hidden_dim, self.num_layers, self.activation)
        # 可学习参数：log_std 与状态无关（对角高斯，各动作维一个标量），初值 log(σ)=log_std_init
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))

    # ---------------------------------------------------------------- #
    # 前向 / 分布
    # ---------------------------------------------------------------- #
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作均值 ``μ_θ(s)``，形 ``(B, action_dim)``（不压缩：允许越界，由环境侧 clip）。"""
        return self.mu_net(state)

    def clipped_log_std(self) -> torch.Tensor:
        """夹紧后的 ``log_std``（形 ``(action_dim,)``）。夹紧点之外梯度为 0，属预期的"软边界"。"""
        return torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

    def std(self) -> torch.Tensor:
        """标准差 ``σ = exp(clip(log_std))``（形 ``(action_dim,)``，可自动广播到 batch）。"""
        return torch.exp(self.clipped_log_std())

    def distribution(self, state: torch.Tensor) -> Normal:
        """动作分布 ``N(μ_θ(s), σ²)``（``torch.distributions.Normal``）。"""
        return Normal(self.forward(state), self.std())

    def log_prob_of(self, state: torch.Tensor, action_raw: torch.Tensor) -> torch.Tensor:
        """**用截断前的原始动作** ``a_raw`` 计算 ``logπ_θ(a_raw|s)``，返回 ``(B, 1)``。

        这是本模块唯一允许的 log-prob 接口 —— 传入 ``a_env = clip(a_raw)`` 会让
        策略梯度出现系统性偏置、并破坏 PPO 的 ``r_t = 1`` 初值（思考题，见模块 docstring）。
        """
        mu = self.forward(state)
        return gaussian_log_prob(action_raw, mu, self.std())

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """微分熵 ``H[N(μ, σ²)]``，返回 ``(B, 1)``（与状态无关，但按 batch 广播便于求和）。"""
        log_std = self.clipped_log_std().unsqueeze(0).expand(state.shape[0], -1)
        return entropy_of_normal(log_std)

    # ---------------------------------------------------------------- #
    # 采样与动作执行
    # ---------------------------------------------------------------- #
    def env_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        """把采样动作截断到环境边界：``a_env = clip(a_raw, low, high)``（**仅用于执行**）。"""
        return torch.clamp(action_raw, self.action_low, self.action_high)

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """采样一个动作，返回采样过程中的全部量（全部 ``detach``，不建计算图）。

        Returns:
            ``action_raw``：``dist.sample()`` 的原始值（**进 buffer、算 logπ 用这个**）；
            ``action_env``：截断到环境边界后的动作（**只喂给 env.step**）；
            ``log_prob``：``(B, 1)`` 的 ``logπ_θ(action_raw|s)``；
            ``entropy``：``(B, 1)``；``mean`` / ``std``：分布参数（诊断用）。
        """
        mu = self.forward(state)
        std = self.std()
        dist = Normal(mu, std)
        action_raw = mu if deterministic else dist.sample()
        return {
            "action_raw": action_raw,
            "action_env": self.env_action(action_raw),
            "log_prob": gaussian_log_prob(action_raw, mu, std),   # ★ 永远算在 a_raw 上
            "entropy": self.entropy(state),
            "mean": mu,
            "std": std.expand(mu.shape[0], -1),
        }


class Critic(nn.Module):
    """状态价值网络 ``V_φ(s)``：``state_dim`` 维状态 → 标量价值（输出形 ``(B, 1)``）。"""

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.net = _mlp(self.state_dim, 1, self.hidden_dim, self.num_layers, self.activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回 ``V_φ(s)``，形 ``(B, 1)``。"""
        return self.net(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """:meth:`forward` 的语义化别名。"""
        return self.forward(state)

# =============================================================================
# 2. Actor-Critic 容器（独立双网络 / 共享主干双头）
# =============================================================================
class ActorCritic(nn.Module):
    """把 Actor 与 Critic 组合成一个模块，统一对外提供策略 / 价值接口。

    ``shared_backbone=False``（默认，作业要求）：**独立双网络** —— ``actor`` 与
    ``critic`` 各自一套 MLP、互不干扰，策略目标不会因价值回归的梯度而漂移。
    ``shared_backbone=True``：共享前 ``num_layers−1`` 层的表征，再分出两个线性头
    （SB3 风格初始化：``mu_head`` 权重尺度 0.01 ⇒ 初始 ``μ ≈ 0``，``value_head``
    尺度 1.0），参数更省；Pendulum 只有 3 维状态，两种结构差异不大，默认不共享。

    ``actor_parameters()`` / ``critic_parameters()`` 给出**互不相交**的参数集合：
    共享主干归 actor 组（一个参数只能有一个"主人"，否则两个 param group 会把它更新两次）。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
        shared_backbone: bool = False,
        log_std_init: float = -0.5,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        action_low: float = DEFAULT_ACTION_LOW,
        action_high: float = DEFAULT_ACTION_HIGH,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.shared_backbone = bool(shared_backbone)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        if self.shared_backbone:
            # 主干取 (num_layers − 1) 层，再各接一个线性头 ⇒ 总深度与独立双网络一致
            self.trunk = _mlp(state_dim, hidden_dim, hidden_dim, max(1, num_layers - 1), activation)
            self.mu_head = nn.Linear(hidden_dim, action_dim)
            self.value_head = nn.Linear(hidden_dim, 1)
            nn.init.orthogonal_(self.mu_head.weight, 0.01)     # 初始 μ ≈ 0（动作居中）
            nn.init.zeros_(self.mu_head.bias)
            nn.init.orthogonal_(self.value_head.weight, 1.0)
            nn.init.zeros_(self.value_head.bias)
            self._log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        else:
            self.actor = Actor(
                state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                num_layers=num_layers, activation=activation, log_std_init=log_std_init,
                log_std_min=log_std_min, log_std_max=log_std_max,
                action_low=action_low, action_high=action_high,
            )
            self.critic = Critic(
                state_dim=state_dim, hidden_dim=hidden_dim,
                num_layers=num_layers, activation=activation,
            )

    # ---------------------------------------------------------------- #
    # 策略 / 价值前向
    # ---------------------------------------------------------------- #
    def mu(self, state: torch.Tensor) -> torch.Tensor:
        """动作均值 ``μ_θ(s)``，形 ``(B, action_dim)``。"""
        if self.shared_backbone:
            return self.mu_head(self.trunk(state))
        return self.actor.forward(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """状态价值 ``V_φ(s)``，形 ``(B, 1)``。"""
        if self.shared_backbone:
            return self.value_head(self.trunk(state))
        return self.critic.value(state)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """一次前向同时得到 ``(μ(s), V(s))``（共享主干时 trunk 只算一次）。"""
        if self.shared_backbone:
            h = self.trunk(state)
            return self.mu_head(h), self.value_head(h)
        return self.actor.forward(state), self.critic.value(state)

    def log_std_parameter(self) -> nn.Parameter:
        """``log_std`` 参数张量（两种模式下都能拿到，便于日志 / 测试）。"""
        return self._log_std if self.shared_backbone else self.actor.log_std

    def clipped_log_std(self) -> torch.Tensor:
        """夹紧后的 ``log_std``（形 ``(action_dim,)``）。"""
        return torch.clamp(self.log_std_parameter(), self.log_std_min, self.log_std_max)

    def std(self) -> torch.Tensor:
        """``σ = exp(clip(log_std))``（形 ``(action_dim,)``，可广播到 batch）。"""
        return torch.exp(self.clipped_log_std())

    def distribution(self, state: torch.Tensor) -> Normal:
        """动作分布 ``N(μ_θ(s), σ²)``。"""
        return Normal(self.mu(state), self.std())

    def log_prob(self, state: torch.Tensor, action_raw: torch.Tensor) -> torch.Tensor:
        """``logπ_θ(action_raw|s)``，形 ``(B, 1)``（**必须传 a_raw**）。"""
        return gaussian_log_prob(action_raw, self.mu(state), self.std())

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """微分熵，形 ``(B, 1)``。"""
        log_std = self.clipped_log_std().unsqueeze(0).expand(state.shape[0], -1)
        return entropy_of_normal(log_std)

    def env_action(self, action_raw: torch.Tensor) -> torch.Tensor:
        """``a_env = clip(a_raw, low, high)``（仅用于执行）。"""
        return torch.clamp(action_raw, self.action_low, self.action_high)
    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """采样一个动作 + 估计 ``V(s)``（**一次前向**同时得到 ``μ(s)`` 与 ``V(s)``）。

        返回字典比 :meth:`Actor.act` 多一个 ``"value"`` —— 因为 rollout 采样时既需要
        ``V(s_t)``（GAE 的起点）又需要动作，用同一次前向计算可省掉一次网络前向。

        Returns:
            ``action_raw``（进 buffer / 算 logπ）/ ``action_env``（只喂 env）/
            ``log_prob`` ``(B,1)`` / ``entropy`` ``(B,1)`` / ``mean`` / ``std`` / ``value`` ``(B,1)``。
        """
        mu, value = self.forward(state)
        std = self.std()
        dist = Normal(mu, std)
        action_raw = mu if deterministic else dist.sample()
        return {
            "action_raw": action_raw,
            "action_env": self.env_action(action_raw),
            "log_prob": gaussian_log_prob(action_raw, mu, std),    # ★ 永远算在 a_raw 上
            "entropy": entropy_of_normal(self.clipped_log_std().unsqueeze(0).expand(mu.shape[0], -1)),
            "mean": mu,
            "std": std.expand(mu.shape[0], -1),
            "value": value,
        }

    # ---------------------------------------------------------------- #
    # 参数分组（供给两个不同学习率的 param group）
    # ---------------------------------------------------------------- #
    def actor_parameters(self) -> List[nn.Parameter]:
        """策略侧参数（共享模式下含主干与 ``log_std``）。"""
        if self.shared_backbone:
            return list(self.trunk.parameters()) + list(self.mu_head.parameters()) + [self._log_std]
        return list(self.actor.parameters())

    def critic_parameters(self) -> List[nn.Parameter]:
        """价值侧参数（共享模式下只有 ``value_head``；主干归 actor 组，避免被更新两次）。"""
        if self.shared_backbone:
            return list(self.value_head.parameters())
        return list(self.critic.parameters())


def build_actor_critic(config: Dict[str, Any]) -> ActorCritic:
    """按配置构建 Actor-Critic（读取 ``model`` 段；缺省值即 Pendulum-v1 的设置）。

    支持的 ``model`` 字段：``state_dim / action_dim / hidden_dim / num_layers /
    activation / shared_backbone / log_std_init / log_std_min / log_std_max /
    action_low / action_high / action_transform``。
    """
    model_cfg = config.get("model", config) or {}
    transform = str(model_cfg.get("action_transform", "clip")).lower()
    if transform != "clip":
        raise ValueError(
            f"未知 action_transform '{transform}'（本模块只实现 clip：tanh 压缩高斯需要"
            "额外的 Jacobian 修正项 log|∂a_env/∂a_raw|，见模块 docstring）"
        )
    return ActorCritic(
        state_dim=int(model_cfg.get("state_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 1)),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        activation=str(model_cfg.get("activation", "relu")),
        shared_backbone=bool(model_cfg.get("shared_backbone", False)),
        log_std_init=float(model_cfg.get("log_std_init", -0.5)),
        log_std_min=float(model_cfg.get("log_std_min", LOG_STD_MIN)),
        log_std_max=float(model_cfg.get("log_std_max", LOG_STD_MAX)),
        action_low=float(model_cfg.get("action_low", DEFAULT_ACTION_LOW)),
        action_high=float(model_cfg.get("action_high", DEFAULT_ACTION_HIGH)),
    )


def build_ppo_optimizer(
    model: ActorCritic,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    weight_decay: float = 0.0,
) -> optim.Adam:
    """构建 Adam：**两个 param group** 分别持有 ``lr_actor`` / ``lr_critic``。

    为什么用"一个优化器 + 两个 param group"而不是两个优化器：Adam 是**逐参数**更新的
    （每个参数各自维护 ``exp_avg / exp_avg_sq / step``，更新量只依赖该参数自己的梯度与
    学习率），因此"两个 param group 的单优化器"与"两个独立 Adam"在数学上**完全等价**
    （``test_two_param_groups_equiv_two_separate_adam`` 逐步逐参数比对），
    而检查点里只需要保存一份优化器状态 —— 断点续训逻辑与 W3/W4/W6 完全复用。
    """
    return optim.Adam(
        [
            {"params": [p for p in model.actor_parameters() if p.requires_grad], "lr": float(lr_actor)},
            {"params": [p for p in model.critic_parameters() if p.requires_grad], "lr": float(lr_critic)},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(weight_decay),
    )

# =============================================================================
# 3. RolloutBuffer 与 GAE-λ
# =============================================================================
def _to_col(x: Union[torch.Tensor, Sequence[float]], device=None, dtype=torch.float32) -> torch.Tensor:
    """把任意输入规范化为列向量 ``(T, 1)``。"""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=dtype)
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return x.to(dtype=dtype) if device is None else x.to(device=device, dtype=dtype)


def compute_gae(
    rewards: Union[torch.Tensor, np.ndarray, Sequence[float]],
    values: Union[torch.Tensor, np.ndarray, Sequence[float]],
    terminated: Union[torch.Tensor, np.ndarray, Sequence[float]],
    truncated: Optional[Union[torch.Tensor, np.ndarray, Sequence[float]]] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: Union[float, torch.Tensor] = 0.0,
    bootstrap_on_truncation: bool = True,
) -> Dict[str, torch.Tensor]:
    """**手写** GAE-λ（广义优势估计），沿轨迹从 ``T−1`` 倒序递推::

        δ_t      = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)
        Â_t      = δ_t + (γλ)·(1 − done_t)·Â_{t+1}
        Target_t = Â_t + V(s_t)

    边界与终止处理（Pendulum 的关键细节）：

    * ``done_t`` 用**真终止** ``terminated_t``（``bootstrap_on_truncation=True``，默认）：
      Pendulum-v1 永远不会 ``terminated``，回合结束只有"到点"（``truncated``，200 步
      TimeLimit）。到点只是"被时间截断"，不是"掉进吸收态"，所以 **继续自举**
      ``γ·V(s_200)`` 才是无偏的做法（CleanRL 的 ``use_proper_time_limits``；
      等价于把任务看作"永不终止的连续控制"）。若把 ``truncated`` 也当作终止，
      ``V(s)`` 就永远学不到"200 步之后的价值"，对 ``γ=0.99`` 的 200 步任务是有偏的
      （消融：``--set training.bootstrap_on_truncation=false``）。
    * rollout 的最后一步用 ``last_value = V(s_T)`` 自举；若恰好在上一步回合结束，
      调用方应传 ``last_value = 0``（此时 ``done = 1``，自举项被掩掉）。

    实现细节：倒序递推在 **CPU float64** 上做。公式是逐标量递推（``T ≈ 2048`` 次），
    numpy 标量运算每次约 0.1 µs，而 GPU 上每个标量算子有 5–10 µs 的启动开销 ——
    在 CPU 上算比在 GPU 上快两个数量级；结果再搬回原设备即可（优势是常数，不需梯度）。

    Returns:
        ``{"advantages": (T,1), "returns": (T,1), "deltas": (T,1), "done": (T,1)}``，
        其中 ``returns = advantages + values``（即 ``Target_t``）。
    """
    ref = rewards if isinstance(rewards, torch.Tensor) else (
        values if isinstance(values, torch.Tensor) else torch.zeros(1)
    )
    device, dtype = ref.device, (torch.float32 if ref.dtype not in (torch.float32, torch.float64) else ref.dtype)

    r = _to_col(rewards).detach().cpu().numpy().reshape(-1).astype(np.float64)
    v = _to_col(values).detach().cpu().numpy().reshape(-1).astype(np.float64)
    term = _to_col(terminated).detach().cpu().numpy().reshape(-1).astype(np.float64)
    trunc = (np.zeros_like(term) if truncated is None
             else _to_col(truncated).detach().cpu().numpy().reshape(-1).astype(np.float64))
    if not (r.shape == v.shape == term.shape == trunc.shape):
        raise ValueError(f"GAE 输入形状不一致: r{r.shape} v{v.shape} term{term.shape} trunc{trunc.shape}")

    done = term if bootstrap_on_truncation else np.clip(term + trunc, 0.0, 1.0)
    T = r.shape[0]
    adv = np.zeros(T, dtype=np.float64)
    deltas = np.zeros(T, dtype=np.float64)
    gae = 0.0
    next_value = float(last_value.detach().cpu().item()) if isinstance(last_value, torch.Tensor) else float(last_value)
    for t in range(T - 1, -1, -1):
        nonterminal = 1.0 - done[t]
        delta = r[t] + float(gamma) * next_value * nonterminal - v[t]
        gae = delta + float(gamma) * float(gae_lambda) * nonterminal * gae
        deltas[t] = delta
        adv[t] = gae
        next_value = v[t]

    advantages = torch.as_tensor(adv, dtype=dtype, device=device).reshape(T, 1)
    return {
        "advantages": advantages,
        "returns": advantages + _to_col(values, device=device, dtype=dtype),
        "deltas": torch.as_tensor(deltas, dtype=dtype, device=device).reshape(T, 1),
        "done": torch.as_tensor(done, dtype=dtype, device=device).reshape(T, 1),
    }
class RolloutBuffer:
    """定长 rollout 缓存：一次"采样阶段"（``rollout_steps`` 步）的全部转移。

    字段（均为 ``(capacity, ·)`` 张量，存在同一设备上）::

        states       (T, state_dim)   经过归一化的状态（网络实际吃到的输入）
        actions_raw  (T, action_dim)  ★ 截断前的原始动作 a_raw —— logπ 与更新都只用它
        log_probs    (T, 1)           采样时的 logπ_θold(a_raw|s)
        rewards      (T, 1)           即时奖励（可选乘 reward_scale）
        values       (T, 1)           采样时的 V_θold(s)
        terminated   (T, 1)           真终止（Pendulum 恒为 0）
        truncated    (T, 1)           TimeLimit 截断（Pendulum 每 200 步为 1）
        advantages   (T, 1)           GAE 得到的 Â_t（常数，不带梯度）
        returns      (T, 1)           Target_t = Â_t + V_θold(s_t)

    为什么"定长 + 复位"而不是动态 list：容量与 ``rollout_steps`` 对齐后张量只分配一次，
    98 次 rollout 的分配次数从 ~20 万次降到 0；``reset()`` 只把写指针归零
    （不清空数据，便于崩溃 / 异常时查看现场）。
    """

    FIELD_NAMES = (
        "states", "actions_raw", "log_probs", "rewards", "values",
        "terminated", "truncated", "advantages", "returns",
    )

    def __init__(
        self,
        capacity: int = 2048,
        state_dim: int = 3,
        action_dim: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.states = torch.zeros((self.capacity, self.state_dim), dtype=torch.float32, device=self.device)
        self.actions_raw = torch.zeros((self.capacity, self.action_dim), dtype=torch.float32, device=self.device)
        for name in ("log_probs", "rewards", "values", "terminated", "truncated", "advantages", "returns"):
            setattr(self, name, torch.zeros((self.capacity, 1), dtype=torch.float32, device=self.device))
        self._pos = 0

    # ---------------------------------------------------------------- #
    # 写入 / 复位
    # ---------------------------------------------------------------- #
    def reset(self) -> None:
        """复位写指针（**不清空**已有数据，便于异常时查看现场）。"""
        self._pos = 0

    def __len__(self) -> int:
        """已缓存的步数。"""
        return self._pos

    @property
    def is_full(self) -> bool:
        """是否已写满（容量 = rollout_steps）。"""
        return self._pos >= self.capacity

    def add(
        self,
        state: Union[torch.Tensor, np.ndarray, Sequence[float]],
        action_raw: Union[torch.Tensor, np.ndarray, Sequence[float]],
        log_prob: Union[torch.Tensor, float],
        reward: Union[torch.Tensor, float],
        value: Union[torch.Tensor, float],
        terminated: Union[torch.Tensor, bool, float] = False,
        truncated: Union[torch.Tensor, bool, float] = False,
    ) -> int:
        """写入一步转移，返回写入位置的下标（缓冲区满时抛 ``IndexError``）。"""
        if self._pos >= self.capacity:
            raise IndexError(f"RolloutBuffer 已满（capacity={self.capacity}），请先 reset()")
        i = self._pos
        as_t = lambda v: torch.as_tensor(v, dtype=torch.float32, device=self.device).reshape(-1)  # noqa: E731
        self.states[i] = as_t(state)[:self.state_dim]
        self.actions_raw[i] = as_t(action_raw)[:self.action_dim]
        self.log_probs[i] = as_t(log_prob)[:1]
        self.rewards[i] = as_t(reward)[:1]
        self.values[i] = as_t(value)[:1]
        self.terminated[i] = float(bool(terminated))
        self.truncated[i] = float(bool(truncated))
        self._pos += 1
        return i

    # ---------------------------------------------------------------- #
    # GAE / mini-batch
    # ---------------------------------------------------------------- #
    def compute_gae(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        last_value: Union[float, torch.Tensor] = 0.0,
        bootstrap_on_truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """对已缓存的 ``T`` 步调用 :func:`compute_gae`，结果写回 ``advantages`` / ``returns``。

        Args:
            last_value: ``V(s_T)`` —— rollout 边界之后那个状态的**价值估计**（采样阶段在
                rollout 结束时多做一次 ``forward`` 得到）；若回合恰好在边界结束则传 0
                （此时 ``done=1``，自举项被掩掉，传什么都无所谓）。

        Returns:
            ``{"advantages", "returns", "deltas", "done"}``，均为 ``(T, 1)`` 切片（``T = len(self)``）。
        """
        n = self._pos
        out = compute_gae(
            rewards=self.rewards[:n], values=self.values[:n],
            terminated=self.terminated[:n], truncated=self.truncated[:n],
            gamma=gamma, gae_lambda=gae_lambda, last_value=last_value,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )
        self.advantages[:n] = out["advantages"]
        self.returns[:n] = out["returns"]
        out["advantages"] = self.advantages[:n]
        out["returns"] = self.returns[:n]
        return out

    def get_minibatches(
        self,
        batch_size: int = 64,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> List[torch.Tensor]:
        """**随机打乱**后切分 mini-batch，返回下标张量列表（不含数据本身）。

        每个 epoch 重新调用 ⇒ 每个 epoch 的划分都不同（PPO 标准做法：同一批 rollout 数据
        被"多轮、以不同顺序"复用）。返回"下标"而不是"切好的数据"：一小批有 6 个字段，
        切 32 批会产生 192 个视图对象，而按需 gather 只要 6 次索引 —— GPU 上更快更省显存。

        打乱在 **CPU** 上做（``torch.randperm``）：CUDA 张量要求 CUDA generator，而
        CPU 置换索引同样能对 CUDA 张量做高级索引，省掉额外维护一个 CUDA generator 的麻烦；
        续训时全局 ``torch`` RNG 由检查点恢复，因此**划分顺序也可复现**。

        Args:
            batch_size: 每个 mini-batch 的样本数（最后一批可能不足，``drop_last=True`` 则丢弃）。
            generator: 可选 ``torch.Generator``（单测里固定划分用）。
            drop_last: 是否丢弃不足一批的尾批（训练时 ``False``，避免丢数据）。
        """
        n = self._pos
        if n == 0:
            return []
        perm = torch.randperm(n, generator=generator)          # CPU 置换
        batch_size = max(1, int(batch_size))
        batches: List[torch.Tensor] = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if drop_last and idx.numel() < batch_size:
                continue
            batches.append(idx.to(device or self.device))
        return batches

    # ---------------------------------------------------------------- #
    # 诊断
    # ---------------------------------------------------------------- #
    def stats(self, clip_low: float = DEFAULT_ACTION_LOW, clip_high: float = DEFAULT_ACTION_HIGH) -> Dict[str, float]:
        """缓存内容概览（写 TensorBoard 与单测断言用；空缓存返回全 0）。"""
        n = self._pos
        if n == 0:
            return dict(n_steps=0.0, reward_mean=0.0, reward_sum=0.0, value_mean=0.0,
                        advantage_mean=0.0, advantage_std=0.0, advantage_abs_mean=0.0,
                        terminated_count=0.0, truncated_count=0.0, action_abs_mean=0.0,
                        action_clip_fraction=0.0)
        action = self.actions_raw[:n]
        return {
            "n_steps": float(n),
            "reward_mean": float(self.rewards[:n].mean()),
            "reward_sum": float(self.rewards[:n].sum()),
            "value_mean": float(self.values[:n].mean()),
            "advantage_mean": float(self.advantages[:n].mean()),
            "advantage_std": float(self.advantages[:n].std(unbiased=False)),
            "advantage_abs_mean": float(self.advantages[:n].abs().mean()),
            "terminated_count": float(self.terminated[:n].sum()),
            "truncated_count": float(self.truncated[:n].sum()),
            "action_abs_mean": float(action.abs().mean()),
            # a_raw 越界的比例：衡量"clip 到底影响了多少步"（思考题的定量背景）
            "action_clip_fraction": float(((action < clip_low) | (action > clip_high)).float().mean()),
        }

# =============================================================================
# 4. PPO 损失（PPO-Clip / 价值 / 熵）与多轮小批量更新
# =============================================================================
def policy_loss_clip(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    normalize_advantage: bool = False,
    normalize_eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """**手写** PPO-Clip 策略损失::

        r_t(θ) = exp( logπ_θ(a_t|s_t) − logπ_θold(a_t|s_t) )
        L^{CLIP}(θ) = − mean_t [ min( r_t·Â_t , clip(r_t, 1−ε, 1+ε)·Â_t ) ]

    三个性质（均有单测）：

    * **负号**：``−min(...)`` 让"最大化代理目标"变成优化器能执行的"最小化"
      （与 W6 REINFORCE 的负号同理；取负后梯度方向就是提升 ``r·Â`` 的方向）；
    * **双侧裁剪**：``Â_t > 0``（好动作）时长概率不能一次涨超过 ``1+ε``；
      ``Â_t < 0``（坏动作）时概率不能被压低于 ``1−ε`` ⇒ "信任域"效果；
    * ``Â_t`` 做 ``detach`` ⇒ 梯度**只**流向策略参数（不会去改优势本身，PPO 它是常数）。

    Args:
        new_log_probs / old_log_probs: 形 ``(B, 1)``；二者在同一动作 ``a_raw`` 上取值，
            且 ``new`` 那侧带梯度。
        advantages: 形 ``(B, 1)``，GAE 得到的 ``Â_t``。
        clip_eps: 裁剪阈值 ``ε``（0.2 ⇒ 比值允许在 ``[0.8, 1.2]``）。
        normalize_advantage: 是否在**本 mini-batch 内**把优势标准化为零均值、单位标准差
            （尺度不变，避免"奖励尺度变化导致有效步长漂移"；统计量只做 ``detach`` 运算）。

    Returns:
        ``(loss, stats)``，``stats`` 含 ``ratio_mean / ratio_std / clip_fraction / approx_kl``。
    """
    adv = advantages.detach()
    if normalize_advantage and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + normalize_eps)

    ratio = torch.exp(new_log_probs - old_log_probs)                 # (B, 1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        stats = {
            "ratio_mean": float(ratio.mean()),
            "ratio_std": float(ratio.std(unbiased=False)),
            "clip_fraction": float(((ratio - 1.0).abs() > float(clip_eps)).float().mean()),
            # approx_kl 用 k3 估计量 (r − 1) − log r ≥ 0：比 mean(logπ_old − logπ_new) 方差更小
            "approx_kl": float(((ratio - 1.0) - torch.log(ratio)).mean()),
        }
    return loss, stats


def value_loss(
    new_values: torch.Tensor,
    targets: torch.Tensor,
    old_values: Optional[torch.Tensor] = None,
    clip_eps: Optional[float] = None,
) -> torch.Tensor:
    """**手写** 价值回归损失::

        L^{VF}(φ) = ½ · mean_t [ ( V_φ(s_t) − Target_t )² ]        # Target_t = Â_t + V_θold(s_t)

    ``clip_eps`` 非空时启用**Clipped Value Loss**（PPO2 论文的可选项）::

        L^{VF} = ½ · mean_t [ max( (V_φ − Target)², (V_clip − Target)² ) ]
        V_clip = V_θold + clip( V_φ − V_θold, −ε, +ε )

    即"价值也不能在一次更新里跑太远"；默认 ``None``（关闭），与本作业给出的
    ``L^VF = ½ E[(V − Target)²]`` 严格一致。
    """
    vf_unclipped = (new_values - targets).pow(2)
    if clip_eps is None or old_values is None:
        return 0.5 * vf_unclipped.mean()
    vf_clipped = old_values.detach() + torch.clamp(
        new_values - old_values.detach(), -float(clip_eps), float(clip_eps)
    )
    return 0.5 * torch.maximum(vf_unclipped, (vf_clipped - targets).pow(2)).mean()


def explained_variance(values: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> float:
    """``1 − Var(Target − V) / Var(Target)``：价值网络的解释度（1=完美，0=与常数预测持平）。"""
    y = targets.detach().reshape(-1)
    diff = (targets.detach() - values.detach()).reshape(-1)
    var_y = float(y.var(unbiased=False))
    if var_y < eps:
        return 0.0
    return float(1.0 - diff.var(unbiased=False) / var_y)


def ppo_losses(
    model: ActorCritic,
    states: torch.Tensor,
    actions_raw: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    targets: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    value_clip_eps: Optional[float] = None,
    normalize_advantage: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``L^{total} = L^{CLIP} + c₁·L^{VF} − c₂·H(π_θ)``：一次 mini-batch 的完整损失。

    关键点：``new_log_probs`` 由 ``gaussian_log_prob(actions_raw, μ_θ(s), σ)`` 得到 ——
    **动作是 buffer 里存好的 ``a_raw``**，绝不用截断后的动作（思考题）。
    熵项的符号是"**减** ``c₂·H``"：优化器最小化总损失 ⇒ 等价于最大化熵（鼓励探索）。
    """
    mu, values = model.forward(states)
    std = model.std()
    new_log_probs = gaussian_log_prob(actions_raw, mu, std)
    entropy = entropy_of_normal(model.clipped_log_std().unsqueeze(0).expand(states.shape[0], -1)).mean()

    pg_loss, pg_stats = policy_loss_clip(
        new_log_probs, old_log_probs, advantages, clip_eps,
        normalize_advantage=normalize_advantage,
    )
    vf_loss = value_loss(values, targets, old_values, clip_eps=value_clip_eps)
    total = pg_loss + float(value_coef) * vf_loss - float(entropy_coef) * entropy

    with torch.no_grad():
        stats: Dict[str, float] = {
            "policy_loss": float(pg_loss),
            "value_loss": float(vf_loss),
            "entropy": float(entropy),
            "total_loss": float(total),
            "log_std_mean": float(model.clipped_log_std().mean()),
            "value_mean": float(values.mean()),
            "value_abs_error_mean": float((values - targets).abs().mean()),
            "explained_variance": explained_variance(values, targets),
        }
    stats.update(pg_stats)
    return total, stats
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    clip_eps: float = 0.2,
    update_epochs: int = 10,
    batch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = True,
    value_clip_eps: Optional[float] = None,
    target_kl: Optional[float] = None,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """用一整批 rollout 做 ``update_epochs`` 轮 mini-batch 更新（PPO 的核心循环）。

    数据流::

        buffer（T 步、固定不动）
          └─ 每个 epoch：重新 randperm → 切成 ⌈T/batch⌉ 个 mini-batch
               └─ 每个 mini-batch：forward(μ,V) → L^CLIP + c₁L^VF − c₂H → backward
                    → clip_grad_norm_(max_grad_norm) → Adam.step()

    为什么"同一批数据反复用 K 轮"：rollout 只有 2048 步，单次梯度估计噪声很大；
    用 10 个 epoch × 32 个 mini-batch = 320 次更新把每步样本"榨干"，样本效率显著提升。
    PPO 的 Clip 就是为这种"重度复用同一批数据"设计的：比值一旦超出 ``[1−ε, 1+ε]``
    就**不再产生梯度**，从而不用像 TRPO 那样解二阶约束。

    Args:
        buffer: 已填好 ``states/actions_raw/log_probs/rewards/values/terminated/truncated``
            并调用过 :meth:`RolloutBuffer.compute_gae` 的缓存。
        update_epochs: 轮数 ``K``（配置项 ``training.update_epochs``）。
        batch_size: mini-batch 大小（``training.batch_size``）。
        max_grad_norm: 全局梯度裁剪阈值（``training.max_grad_norm``；``<=0`` 关闭）。
        target_kl: 非空且某 epoch 的平均 ``approx_kl`` 超过它时提前停止（默认关闭）。
        generator: 可选 ``torch.Generator``（单测里固定划分顺序）。

    Returns:
        诊断字典：损失三项、``ratio_mean`` / ``clip_fraction`` / ``approx_kl`` / ``grad_norm`` /
        ``explained_variance`` 等（全部为 rollout 内所有更新的均值），可直接写 TensorBoard。
    """
    n = len(buffer)
    if n == 0:
        raise ValueError("RolloutBuffer 为空，无法更新（请先采样）")
    device = torch.device(device) if device is not None else buffer.device
    model.train()

    # 常数张量只切一次（注意：这些是 buffer 里的**原张量视图**，不会被优化器改动）
    states = buffer.states[:n].to(device)
    actions_raw = buffer.actions_raw[:n].to(device)
    old_log_probs = buffer.log_probs[:n].to(device)
    old_values = buffer.values[:n].to(device)
    advantages = buffer.advantages[:n].to(device)
    targets = buffer.returns[:n].to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    keys: List[str] = []
    acc: Dict[str, List[float]] = {}
    n_updates = 0
    epochs_run = 0
    stopped_early = False

    for _epoch in range(max(1, int(update_epochs))):
        batches = buffer.get_minibatches(batch_size, generator=generator, drop_last=drop_last, device=device)
        epoch_kl: List[float] = []
        for idx in batches:
            total, stats = ppo_losses(
                model,
                states=states[idx], actions_raw=actions_raw[idx],
                old_log_probs=old_log_probs[idx], old_values=old_values[idx],
                advantages=advantages[idx], targets=targets[idx],
                clip_eps=clip_eps, value_coef=value_coef, entropy_coef=entropy_coef,
                value_clip_eps=value_clip_eps, normalize_advantage=normalize_advantage,
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if max_grad_norm and float(max_grad_norm) > 0:
                # clip_grad_norm_ 返回的是**裁剪前**的全局范数：正好是我们要监控的量
                grad_norm = float(nn.utils.clip_grad_norm_(params, float(max_grad_norm)))
            else:
                grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)))
            optimizer.step()

            stats["grad_norm"] = grad_norm
            for key, value in stats.items():
                if key not in acc:
                    keys.append(key)
                    acc[key] = []
                acc[key].append(float(value))
            epoch_kl.append(stats["approx_kl"])
            n_updates += 1
        epochs_run += 1
        if target_kl is not None and epoch_kl and float(np.mean(epoch_kl)) > float(target_kl):
            stopped_early = True
            break

    out: Dict[str, float] = {key: float(np.mean(acc[key])) for key in keys}
    with torch.no_grad():
        out.update({
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std(unbiased=False)),
            "advantage_abs_mean": float(advantages.abs().mean()),
            "target_mean": float(targets.mean()),
            "n_steps": float(n),
            "n_minibatches": float(n_updates),
            "epochs_run": float(epochs_run),
            "kl_early_stop": float(bool(stopped_early)),
        })
    return out

# =============================================================================
# 5. 采样器（rollout）与独立评估
# =============================================================================
class RolloutCollector:
    """在环境里采样 ``rollout_steps`` 步，写进 :class:`RolloutBuffer` 并算好 GAE。

    设计要点（每条都有对应的单测 / 文档验证）：

    * **惰性 reset**：只有"上一步真的结束了回合"才在下一个动作前 ``reset``。于是 rollout
      边界处的自举值 ``V(s_T)`` 是"从真正的 s_T 继续"的价值；若像很多实现那样在回合结束处
      立刻 auto-reset，边界自举会拿到**新回合初始状态**的价值（错误信号）。
    * **确定性 reset 种子**：第 ``i`` 次重置用 ``seed + i`` 播种，初始状态序列与全局 torch
      RNG 解耦 —— 每次运行、每次续训都得到同一串初始状态，实验之间可公平对比。
    * **状态归一化**：先 ``update`` 统计量再 ``normalize``（SB3 ``VecNormalize`` 的做法）；
      buffer 里存的是**网络实际吃到的输入**，更新阶段重算 ``logπ`` 时用的就是同一批数值。
    * **reward_scale 只作用于学习目标**：buffer 中的奖励可乘缩放系数（改善价值回归的
      数值条件），但日志里的 episode return 永远是环境原始回报，避免"曲线看着变好"的假象。
    * **a_raw / a_env 分离**：``buffer.actions_raw = a_raw``（算 logπ、更新用），
      ``env.step(a_env = clip(a_raw))``；``last_actions_raw`` / ``last_actions_env`` 保留最近
      一次 rollout 的两种动作，便于断言"喂给环境的永远是截断后的动作"。
    """

    def __init__(
        self,
        env: Any,
        model: ActorCritic,
        buffer: RolloutBuffer,
        device: Union[str, torch.device] = "cpu",
        normalizer: Optional[RunningMeanStd] = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        reward_scale: float = 1.0,
        bootstrap_on_truncation: bool = True,
        seed: int = 42,
        normalize_obs: bool = True,
        n_resets: int = 0,
    ) -> None:
        self.env = env
        self.model = model
        self.buffer = buffer
        self.device = torch.device(device)
        self.normalize_obs = bool(normalize_obs)
        self.normalizer = normalizer if self.normalize_obs else None
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reward_scale = float(reward_scale)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.seed = int(seed)

        self._obs: Optional[np.ndarray] = None      # 原始（未归一化）状态；None ⇒ 下一步前 reset
        self._ep_return = 0.0
        self._ep_length = 0
        self.n_resets = int(n_resets)               # 续训时接续原计数 ⇒ 初始状态序列可复现
        self.episode_returns: List[float] = []      # 全部已完成回合的原始回报
        self.episode_lengths: List[int] = []
        self.last_actions_raw: Optional[torch.Tensor] = None
        self.last_actions_env: Optional[torch.Tensor] = None
        self.last_last_value: float = 0.0

    def reset_episode(self) -> np.ndarray:
        """开新回合：用 ``seed + n_resets`` 播种（确定性、可复现），返回原始状态。"""
        obs, _ = self.env.reset(seed=self.seed + self.n_resets)
        self.n_resets += 1
        self._obs = np.asarray(obs, dtype=np.float32)
        self._ep_return, self._ep_length = 0.0, 0
        return self._obs

    def _policy_input(self) -> torch.Tensor:
        """当前状态的"网络输入"：按需 reset → 更新归一化统计量 → 归一化，返回 ``(1, state_dim)``。"""
        obs = self._obs if self._obs is not None else self.reset_episode()
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.normalizer is not None:
            self.normalizer.update(obs)                 # ★ 先并入当前样本，再归一化
            tensor = self.normalizer.normalize_torch(tensor)
        return tensor

    @torch.no_grad()
    def collect(self, steps: int) -> Dict[str, Any]:
        """采样 ``steps`` 步：写 buffer → 边界自举 → 算 GAE → 返回统计。

        Returns:
            ``n_steps`` / ``last_value`` / ``episode_returns`` / ``episode_lengths`` /
            ``mean_episode_return`` / ``n_episodes`` / ``action_clip_fraction`` /
            ``std_mean`` / ``gae``。
        """
        self.buffer.reset()
        raw_actions: List[torch.Tensor] = []
        env_actions: List[torch.Tensor] = []
        n_before = len(self.episode_returns)

        for _ in range(int(steps)):
            obs_norm = self._policy_input()                     # (1, state_dim)
            out = self.model.act(obs_norm, deterministic=False)
            action_raw = out["action_raw"]                      # (1, action_dim) 未截断
            action_env = out["action_env"]                      # (1, action_dim) 已截断到 [-2, 2]

            next_obs, reward, terminated, truncated, _info = self.env.step(
                action_env[0].detach().cpu().numpy()
            )
            self.buffer.add(
                state=obs_norm[0], action_raw=action_raw[0], log_prob=out["log_prob"][0],
                reward=float(reward) * self.reward_scale, value=out["value"][0],
                terminated=terminated, truncated=truncated,
            )
            raw_actions.append(action_raw[0].detach())
            env_actions.append(action_env[0].detach())

            # 统计**原始回报**（不受 reward_scale 影响，曲线始终可比）
            self._ep_return += float(reward)
            self._ep_length += 1
            if terminated or truncated:
                self.episode_returns.append(self._ep_return)
                self.episode_lengths.append(self._ep_length)
                self._obs = None                                # 下一步前才 reset（惰性）
            else:
                self._obs = np.asarray(next_obs, dtype=np.float32)

        # ---- rollout 边界的自举值 V(s_T)：回合恰好在边界结束 ⇒ 传 0（done=1 会掩掉自举项）----
        if self._obs is None:
            last_value: Union[float, torch.Tensor] = 0.0
        else:
            last_value = self.model.value(self._policy_input()).detach()
        self.last_last_value = float(np.mean(np.atleast_1d(
            last_value.detach().cpu().numpy() if isinstance(last_value, torch.Tensor) else float(last_value)
        )))

        gae = self.buffer.compute_gae(
            gamma=self.gamma, gae_lambda=self.gae_lambda, last_value=last_value,
            bootstrap_on_truncation=self.bootstrap_on_truncation,
        )
        self.last_actions_raw = torch.stack(raw_actions)
        self.last_actions_env = torch.stack(env_actions)

        new_returns = self.episode_returns[n_before:]
        return {
            "n_steps": int(len(self.buffer)),
            "last_value": float(self.last_last_value),
            "episode_returns": new_returns,
            "episode_lengths": self.episode_lengths[n_before:],
            "mean_episode_return": float(np.mean(new_returns)) if new_returns else float("nan"),
            "n_episodes": len(new_returns),
            "action_clip_fraction": float(
                ((self.last_actions_raw < self.model.action_low)
                 | (self.last_actions_raw > self.model.action_high)).float().mean()
            ),
            "std_mean": float(self.model.std().mean()),
            "gae": gae,
        }


@torch.no_grad()
def evaluate_policy(
    model: ActorCritic,
    env_name: str = "Pendulum-v1",
    device: Union[str, torch.device] = "cpu",
    episodes: int = 20,
    seed: int = 12345,
    gamma: float = 0.99,
    deterministic: bool = True,
    normalizer: Optional[RunningMeanStd] = None,
    max_steps: Optional[int] = None,
    env: Optional[Any] = None,
) -> Dict[str, float]:
    """用**独立评估回合**衡量策略水平（不动训练环境、不算梯度、不更新归一化统计量）。

    与训练采样的区别：

    * ``deterministic=True``：动作取 ``μ(s)``（再截断到 ``[-2, 2]``），评估无采样噪声；
    * 每回合用 ``seed + i`` 播种 ⇒ 同一策略的评估可复现、不同策略可对比；
    * 归一化统计量**只读**（评估不 ``update``），保证"评估分布"与训练末端一致。

    Returns:
        ``{"mean_return", "std_return", "min_return", "max_return", "mean_length",
           "mean_reward_per_step", "episodes", "n_episodes", "deterministic"}``。
        Pendulum 的回报 = 各步奖励之和（负值，越接近 0 越好；收敛策略约 −150 ~ −200）。
    """
    own_env = env is None
    env = env if env is not None else make_env(env_name)
    limit = max_steps if max_steps is not None else env_max_steps(env, default=DEFAULT_MAX_STEPS)
    returns: List[float] = []
    lengths: List[int] = []
    try:
        for i in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + i)
            ep_return, steps = 0.0, 0
            while True:
                tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
                if normalizer is not None:
                    tensor = normalizer.normalize_torch(tensor)
                out = model.act(tensor, deterministic=deterministic)
                obs, reward, terminated, truncated, _info = env.step(out["action_env"][0].cpu().numpy())
                ep_return += float(reward)
                steps += 1
                if terminated or truncated or steps >= int(limit):
                    break
            returns.append(ep_return)
            lengths.append(steps)
    finally:
        if own_env:
            env.close()

    arr = np.asarray(returns, dtype=np.float32) if returns else np.zeros(0, dtype=np.float32)
    n = int(arr.size)
    return {
        "mean_return": float(arr.mean()) if n else 0.0,
        "std_return": float(arr.std(ddof=0)) if n else 0.0,
        "min_return": float(arr.min()) if n else 0.0,
        "max_return": float(arr.max()) if n else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_reward_per_step": float(arr.sum() / max(1, int(np.sum(lengths)))) if n else 0.0,
        "episodes": float(n),
        "n_episodes": float(n),             # 兼容 report_closed_loop 的字段名
        "deterministic": float(bool(deterministic)),
    }

