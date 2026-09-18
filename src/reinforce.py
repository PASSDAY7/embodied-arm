"""手写 REINFORCE（带 Baseline 的策略梯度）核心模块 —— W6 Task 1。

在 CartPole-v1 上从零实现**策略梯度（Policy Gradient）**流水线：不再需要专家数据，
策略通过与环境的交互自己收集轨迹，用蒙特卡洛回报直接估计策略梯度。

一句话数学（策略梯度定理）::

    ∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t  ∇_θ log π_θ(a_t | s_t) · (G_t − b(s_t)) ]

本模块把定理拆成五个可单独测试的部件
------------------------------------
1. :class:`PolicyNetwork` —— 策略网络 ``π_θ(a | s)``：输入 ``state_dim`` 维状态，
   输出 ``action_dim`` 维 **logits**，经 ``Categorical`` 得到离散动作分布；提供
   ``distribution`` / ``log_prob`` / ``entropy`` / ``act`` 四个接口。
2. :func:`compute_returns` —— 累积折扣回报 ``G_t = r_t + γ·G_{t+1}``（从回合末尾反向
   递推），并可选做 **Return 归一化** ``(G − mean) / (std + ε)``。
3. :func:`compute_advantages` —— 优势 ``A_t = G_t − b(s_t)``：基线 ``b`` 支持
   ``none``（原始回报）/ ``batch_mean``（批内回报均值）/ ``value``（小 Critic ``V_φ(s)``）。
4. :func:`reinforce_loss` —— 目标损失 ``L(θ) = −Σ_t log π_θ(a_t|s_t)·A_t``。
   **负号**的作用是把"最大化期望回报"翻译成 PyTorch 优化器默认执行的"最小化"。
5. :func:`run_episode` / :func:`update_policy` —— 回合采样（缓存 ``(s, a, r, logπ)``）
   与一次策略改进（重算 logπ → 求优势 → 反向传播），构成"回合更新"闭环。

两个实现细节（都会在单元测试里被验证）
--------------------------------------
* **logπ 既缓存又重算**：采样时缓存的 ``log π`` 是 ``detach`` 过的标量（用于诊断与
  一致性校验）；真正反传的 logπ 由 :func:`update_policy` 用 **(states, actions) 重新
  前向**得到。因为"采样 → 更新"之间参数没有变，重算值与缓存值逐位一致
  （见 ``tests/test_reinforce.py::test_recomputed_log_probs_match_cached``），
  但重算避免让整条回合的计算图常驻显存，也让"多回合批量更新"成为可能。
* **单优化器 = 双优化器**：当基线为 ``value`` 时，策略损失与 Critic 回归损失写成一个
  联合损失、交给同一个 Adam（``advantage = (G_t − V(s_t)).detach()`` 已切断流向
  Critic 参数的梯度）。因为两项损失各自只依赖**互不相交**的参数集合，而 Adam 又是
  逐参数更新的，所以"一个优化器"与"两个独立优化器"在数学上**完全等价**
  （见 ``test_joint_optimizer_matches_two_separate_optimizers``）—— 好处是检查点里
  只需保存一份优化器状态，断点续训逻辑与 W3/W4 完全复用。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:  # gymnasium 为可选依赖，仅环境交互时需要
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

__all__ = [
    "PolicyNetwork",
    "ValueNetwork",
    "build_policy",
    "build_value_network",
    "make_env",
    "env_max_steps",
    "compute_returns",
    "compute_advantages",
    "reinforce_loss",
    "value_loss",
    "run_episode",
    "stack_episodes",
    "update_policy",
    "evaluate_policy",
    "count_parameters",
    "BASELINES",
    "DEFAULT_MAX_STEPS",
]

# CartPole-v1 的回合上限（TimeLimit 截断步数）：单回合最多 500 步
DEFAULT_MAX_STEPS = 500

# 支持的基线类型：无基线 / 批内回报均值 / 价值网络 V_φ(s)
BASELINES: Tuple[str, ...] = ("none", "batch_mean", "value")

# =============================================================================
# 0. 内部工具
# =============================================================================
def _resolve_activation(name: str) -> type:
    """按名称解析激活函数层（与 ``src/networks.py`` 保持同一套可选值）。"""
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


def count_parameters(module: nn.Module) -> int:
    """统计可训练参数量。"""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _mlp(
    in_dim: int,
    out_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 2,
    activation: str = "relu",
) -> nn.Sequential:
    """构建含 ``num_layers`` 个隐藏层的 MLP（输出层不带激活，保留 logits / 价值原值）。"""
    layers: List[nn.Module] = []
    last = int(in_dim)
    for _ in range(max(1, int(num_layers))):
        layers += [nn.Linear(last, int(hidden_dim)), _resolve_activation(activation)()]
        last = int(hidden_dim)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


# =============================================================================
# 1. 策略网络 π_θ(a | s) 与价值网络 V_φ(s)
# =============================================================================
class PolicyNetwork(nn.Module):
    """离散动作策略网络：``state_dim`` 维状态 → ``action_dim`` 维动作 logits。

    Args:
        state_dim: 状态维度（CartPole 为 4）。
        action_dim: 离散动作个数（CartPole 为 2，即"向左/向右推小车"）。
        hidden_dim: 隐藏层宽度。
        num_layers: 隐藏层数量。
        activation: 激活函数名（``relu`` / ``tanh`` / ``gelu`` / ``leaky_relu`` / ``elu``）。

    Note:
        输出是 **logits 而不是概率**：``Categorical(logits=...)`` 内部做 log-softmax，
        数值上比"先 softmax 再取 log"更稳定（避免 ``log(0) = -inf``）。
    """

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.net = _mlp(self.state_dim, self.action_dim, hidden_dim, num_layers, activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作 logits，形状 ``(batch, action_dim)``。"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """返回动作分布 ``π_θ(· | s)``。"""
        return torch.distributions.Categorical(logits=self.forward(state))

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``log π_θ(a | s)``，形状 ``(batch,)`` —— 策略梯度里的得分函数。"""
        dist = self.distribution(state)
        return dist.log_prob(action.reshape(-1).long())

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """动作分布熵 ``H[π_θ(·|s)]``，形状 ``(batch,)``（熵奖励用于鼓励探索）。"""
        return self.distribution(state).entropy()

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """与环境交互用的动作选择。

        Args:
            state: 单个状态 ``(state_dim,)`` 或一批状态 ``(batch, state_dim)``。
            deterministic: ``True`` 取 ``argmax``（评估用贪心策略）；
                ``False`` 按分布采样（训练用，探索的来源）。

        Returns:
            ``{"action": (batch,) long, "log_prob": (batch,), "entropy": (batch,)}``，
            均为 ``detach`` 后的结果（采样阶段不需建图）。
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return {
            "action": action.detach(),
            "log_prob": dist.log_prob(action).detach(),
            "entropy": dist.entropy().detach(),
        }


class ValueNetwork(nn.Module):
    """状态价值网络 ``V_φ(s)``：作为策略梯度的**进阶基线** ``b(s_t)``。

    只用状态作为输入、回归"原始折扣回报 ``G_t``"（单位与真实回报一致），
    因此优势 ``A_t = G_t − V_φ(s_t)`` 的方差远小于直接用 ``G_t`` 加权。
    """

    def __init__(
        self,
        state_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.net = _mlp(self.state_dim, 1, hidden_dim, num_layers, activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回 ``(batch, 1)`` 的状态价值。"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """返回 ``(batch,)`` 的状态价值（去掉最后一维，便于与回报相减）。"""
        return self.forward(state).squeeze(-1)


def build_policy(config: Dict[str, Any]) -> PolicyNetwork:
    """按配置字典构建策略网络（读取 ``config["model"]`` 段，字段名与 W3/W4 对齐）。"""
    model_cfg = config.get("model", config)
    return PolicyNetwork(
        state_dim=int(model_cfg.get("state_dim", 4)),
        action_dim=int(model_cfg.get("action_dim", 2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        activation=str(model_cfg.get("activation", "relu")),
    )


def build_value_network(config: Dict[str, Any]) -> ValueNetwork:
    """按配置字典构建 Critic（``algorithm.critic`` 段优先，缺省回落到 ``model`` 段）。"""
    model_cfg = config.get("model", config)
    critic_cfg = (config.get("algorithm", {}) or {}).get("critic", {}) or {}
    return ValueNetwork(
        state_dim=int(critic_cfg.get("state_dim", model_cfg.get("state_dim", 4))),
        hidden_dim=int(critic_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 128))),
        num_layers=int(critic_cfg.get("num_layers", model_cfg.get("num_layers", 2))),
        activation=str(critic_cfg.get("activation", model_cfg.get("activation", "relu"))),
    )


# =============================================================================
# 2. 环境工具（CartPole-v1）
# =============================================================================
def make_env(env_name: str = "CartPole-v1", render: bool = False) -> "gym.Env":
    """创建 Gymnasium 环境；``render=True`` 时开人类可视化窗口。"""
    if gym is None:
        raise ImportError("未安装 gymnasium，无法创建环境（pip install gymnasium）。")
    return gym.make(env_name, render_mode="human" if render else None)


def env_max_steps(env: Any, default: int = DEFAULT_MAX_STEPS) -> int:
    """读取环境的回合步数上限（``TimeLimit`` 截断步数），读取失败时返回 ``default``。

    CartPole-v1 在 500 步处 ``truncated``（而不是 ``terminated``）—— 这个区分很重要：
    ``truncated`` 不代表策略失败，所以回报递推在回合结束时直接置 0 即可（不做自举）。
    """
    spec = getattr(env, "spec", None)
    limit = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if limit is None:
        limit = getattr(env, "_max_episode_steps", None)  # TimeLimit 包装器
    if limit is None:
        spec = getattr(getattr(env, "unwrapped", None), "spec", None)
        limit = getattr(spec, "max_episode_steps", None) if spec is not None else None
    try:
        return int(limit) if limit else int(default)
    except (TypeError, ValueError):  # pragma: no cover
        return int(default)


# =============================================================================
# 3. 回报 / 优势 / 损失（策略梯度定理的三个代数部件）
# =============================================================================
def compute_returns(
    rewards: Union[Sequence[float], torch.Tensor],
    gamma: float = 0.99,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """把一条轨迹的即时奖励 ``[r_0, ..., r_{T-1}]`` 变成**累积折扣回报** ``[G_0, ..., G_{T-1}]``。

    从回合末尾反向递推（``O(T)``，与朴素的双重循环 ``O(T²)`` 结果逐位一致）::

        G_{T-1} = r_{T-1}
        G_t     = r_t + γ · G_{t+1}

    为什么必须"从后往前"：``G_t`` 依赖 ``t`` 之后的所有奖励，反向递推让每步只做一次乘加。

    Args:
        rewards: **单条轨迹**的即时奖励序列（列表 / 张量）。
        gamma: 折扣因子 ``γ ∈ [0, 1]``（CartPole 用 0.99）。
        normalize: 是否做 **Return 归一化** ``(G − mean(G)) / (std(G) + ε)``。
            同一批（默认就是同一回合）内的回报被拉到零均值、单位标准差，
            它的作用相当于"批内均值基线 + 自适应梯度尺度"。
        eps: 防除零小常数（回合内所有回报相同时 ``std = 0``）。

    Returns:
        ``{"returns": 待使用的回报, "raw_returns": 原始回报, "raw_mean": float,
           "raw_std": float, "norm_std": float, "normalized": bool}``。
    """
    if torch.is_tensor(rewards):
        arr = rewards.detach().to("cpu").reshape(-1).to(torch.float32)
    else:
        arr = torch.as_tensor(np.asarray(list(rewards), dtype=np.float32)).reshape(-1)

    values = arr.tolist()
    discounted = 0.0
    out = [0.0] * len(values)
    for i in range(len(values) - 1, -1, -1):            # 反向递推
        discounted = float(values[i]) + float(gamma) * discounted
        out[i] = discounted
    raw = torch.tensor(out, dtype=torch.float32)

    if raw.numel() == 0:
        mean = std = 0.0
    else:
        mean = float(raw.mean())
        std = float(raw.std(unbiased=False))            # 总体标准差（与 numpy 默认 ddof=0 一致）
    returns = (raw - mean) / (std + eps) if normalize else raw.clone()
    return {
        "returns": returns,
        "raw_returns": raw,
        "raw_mean": mean,
        "raw_std": std,
        "norm_std": float(returns.std(unbiased=False)) if returns.numel() else 0.0,
        "normalized": bool(normalize),
    }


def compute_advantages(
    returns: Union[Sequence[float], torch.Tensor],
    values: Optional[Union[Sequence[float], torch.Tensor]] = None,
    normalize: bool = False,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """计算策略梯度的权重项 —— 优势 ``A_t = G_t − b(s_t)``。

    基线 ``b`` 的三种形态（与 :func:`src.reinforce.update_policy` 的 ``baseline`` 参数一致）::

        b = 0                     → A_t = G_t                        （原始回报，方差最大）
        b = mean(G)（批内均值）    → A_t = G_t − mean(G)               （最简 Baseline）
        b = V_φ(s_t)（价值网络）   → A_t = G_t − V_φ(s_t)              （进阶 Baseline）

    基线只改变方差、不改变期望：因为 ``E[∇log π(a|s)·b(s)] = 0``（对任意只依赖 ``s`` 的
    ``b`` 成立），所以减去基线是**无偏**的方差缩减技巧。

    Args:
        returns: 折扣回报 ``G_t``。
        values: 基线 ``b``（``None`` 表示不减去任何基线）。
        normalize: 是否再把优势标准化到零均值、单位标准差（PPO/A2C 的常见做法）。

    Returns:
        ``{"advantages", "raw_advantages", "advantage_mean", "advantage_std",
           "advantage_abs_mean", "baseline_mean", "normalize"}``。
    """
    ret = torch.as_tensor(np.asarray(_to_numpy(returns), dtype=np.float32)).reshape(-1)
    if values is None:
        adv = ret.clone()
        baseline_mean = 0.0
    else:
        base = torch.as_tensor(np.asarray(_to_numpy(values), dtype=np.float32)).reshape(-1)
        if base.numel() != ret.numel():
            raise ValueError(f"基线长度 {base.numel()} 与回报长度 {ret.numel()} 不一致")
        if base.requires_grad:
            base = base.detach()                        # 关键：基线不参与策略梯度
        adv = ret - base
        baseline_mean = float(base.mean()) if base.numel() else 0.0

    raw_adv = adv.clone()
    if normalize and adv.numel() > 0:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
    return {
        "advantages": adv,
        "raw_advantages": raw_adv,
        "advantage_mean": float(adv.mean()) if adv.numel() else 0.0,
        "advantage_std": float(adv.std(unbiased=False)) if adv.numel() else 0.0,
        "advantage_abs_mean": float(adv.abs().mean()) if adv.numel() else 0.0,
        "baseline_mean": baseline_mean,
        "normalize": bool(normalize),
    }


def _to_numpy(data: Union[Sequence[float], torch.Tensor]) -> np.ndarray:
    """把张量 / 序列统一成 numpy 数组（只做搬运，不做拷贝语义假设）。"""
    if torch.is_tensor(data):
        return data.detach().to("cpu").numpy()
    return np.asarray(data)



def reinforce_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    entropies: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """REINFORCE 目标损失 —— **为什么要加负号**的答案就在这里。

    策略梯度定理给出的是"如何**最大化**期望回报"：::

        ∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · A_t ]
        J(θ)     = E[ Σ_t log π_θ(a_t|s_t) · A_t ]              （蒙特卡洛估计）

    而 ``torch.optim`` 的所有优化器（SGD / Adam / ...）默认都在**最小化**目标函数
    （``optimizer.step()`` 执行的是 ``θ ← θ − lr · ∇L``）。因此必须定义::

        L(θ) = − Σ_t log π_θ(a_t|s_t) · A_t

    这样 ``∇L = −∇J``，优化器做的"梯度下降"就等价于在 ``J`` 上"梯度上升"：
    当 ``A_t > 0``（这个动作比基线好）时，``−log π`` 下降 ⇒ ``log π`` 上升
    ⇒ **该动作的概率被提高**；``A_t < 0`` 时该动作概率被压低。
    （不写负号会发生什么：等价于最大化"坏动作的概率"，往返报的反方向跑，
    见 ``tests/test_reinforce.py::test_minus_sign_is_required_for_ascent``。）

    Args:
        log_probs: ``log π_θ(a_t | s_t)``，形状 ``(N,)``（N = 更新批内总步数）。
        advantages: 优势 ``A_t``，形状 ``(N,)``。
        entropies: 可选的动作分布熵（配合 ``entropy_coef > 0`` 鼓励探索）。
        entropy_coef: 熵奖励系数 ``β``：``L = −Σ logπ·A − β·Σ H``。
        reduction: ``"mean"``（对步数取均值，梯度尺度与回合长度无关）/ ``"sum"``（教材写法）。

    Returns:
        ``(loss, stats)``：``loss`` 为可反传标量；``stats`` 记录策略梯度项与熵项数值。
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"未知 reduction '{reduction}'，可选: mean | sum")
    lp = log_probs.reshape(-1)
    adv = advantages.reshape(-1).detach()               # 优势是常数权重，不参与求导
    if lp.numel() != adv.numel():
        raise ValueError(f"log_probs 长度 {lp.numel()} 与 advantages 长度 {adv.numel()} 不一致")

    agg = torch.sum if reduction == "sum" else torch.mean
    pg_term = -agg(lp * adv)                            # ★ 负号：最大化 → 最小化
    loss = pg_term
    entropy_value = 0.0
    if entropies is not None:
        ent = entropies.reshape(-1)
        if ent.numel() != lp.numel():
            raise ValueError(f"entropies 长度 {ent.numel()} 与 log_probs 长度 {lp.numel()} 不一致")
        # 熵统计**始终**记录（便于观察探索程度），但只有 β>0 时才进入损失函数：
        # 丢弃熵项等价于 β=0，此时它只是一个诊断量，不影响梯度。
        entropy_value = float(agg(ent).detach())
        if float(entropy_coef):
            loss = loss - float(entropy_coef) * agg(ent)  # 最大化熵 ⇒ 在损失里减去熵
    stats = {
        "policy_gradient": float(pg_term.detach()),
        "entropy": entropy_value,
        "entropy_coef": float(entropy_coef),
        "reduction": reduction,
        "n_steps": int(lp.numel()),
        "loss": float(loss.detach()),
    }
    return loss, stats


def value_loss(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Critic 的回归损失 ``MSE(V_φ(s_t), G_t)``（``G_t`` 为**原始**折扣回报，单位与回报一致）。"""
    v = values.reshape(-1)
    t = targets.reshape(-1).detach()
    if v.numel() != t.numel():
        raise ValueError(f"values 长度 {v.numel()} 与 targets 长度 {t.numel()} 不一致")
    return nn.functional.mse_loss(v, t)



# =============================================================================
# 4. 回合采样（数据从哪来：策略与环境的交互）
# =============================================================================
@torch.no_grad()
def run_episode(
    env: Any,
    policy: PolicyNetwork,
    device: Union[str, torch.device] = "cpu",
    gamma: float = 0.99,
    normalize_returns: bool = True,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """跑一整条回合，缓存 ``(s_t, a_t, logπ, r_t)`` 并就地算出折扣回报。

    采样阶段整体包在 ``@torch.no_grad()`` 里：策略梯度是 **off-policy 更新前的
    蒙特卡洛估计**，只需要 ``(s_t, a_t, G_t)`` 这些**数值**，不需要保留采样时的计算图
    （更新时用同样的 ``(s_t, a_t)`` 重新前向即可，见 :func:`update_policy`）。

    Args:
        env: Gymnasium 环境（CartPole-v1）。
        policy: 策略网络。
        device: 前向所用设备。
        gamma: 折扣因子。
        normalize_returns: 本回合回报是否归一化（批量更新时会按整个更新批重算，此处仅作文档/参考）。
        max_steps: 额外步数上限（``None`` 时用 :func:`env_max_steps`）。
        seed: 仅首个回合传 ``seed`` 可复现环境随机性（后续回合靠环境自身连续采样）。
        deterministic: 训练时用 ``False``（按分布采样，保证探索）。

    Returns:
        轨迹字典，键包括 ``states`` ``(T, state_dim)``、``actions`` ``(T,)``、
        ``log_probs`` ``(T,)``、``entropies`` ``(T,)``、``rewards`` ``(T,)``、
        ``returns`` ``(T,)``、``raw_returns`` ``(T,)``、``episode_return``（未折扣总回报）、
        ``discounted_return``（``G_0``）、``length``、``terminated``、``truncated``。

    Note:
        CartPole-v1 在 500 步处是 ``truncated=True``（时间上限），不是失败。这里按惯例
        把它当回合结束处理（不做自举），因为 500 步就是任务定义的满分。
    """
    limit = int(max_steps) if max_steps is not None else env_max_steps(env)
    state, _ = env.reset(seed=seed)

    states: List[np.ndarray] = []
    actions: List[int] = []
    log_probs: List[float] = []
    entropies: List[float] = []
    rewards: List[float] = []
    terminated = truncated = False

    while True:
        s_t = torch.as_tensor(np.asarray(state, dtype=np.float32).reshape(-1),
                              dtype=torch.float32, device=device)
        out = policy.act(s_t, deterministic=deterministic)
        action = int(out["action"].reshape(-1)[0].item())

        states.append(np.asarray(state, dtype=np.float32).reshape(-1))
        actions.append(action)
        log_probs.append(float(out["log_prob"].reshape(-1)[0].item()))
        entropies.append(float(out["entropy"].reshape(-1)[0].item()))

        state, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))

        if terminated or truncated:
            break
        if len(rewards) >= limit:                        # 兜底：环境没有 TimeLimit 时手动截断
            truncated = True
            break

    ret = compute_returns(rewards, gamma=gamma, normalize=normalize_returns)
    return {
        "states": torch.tensor(np.stack(states), dtype=torch.float32),
        "actions": torch.tensor(actions, dtype=torch.long),
        "log_probs": torch.tensor(log_probs, dtype=torch.float32),
        "entropies": torch.tensor(entropies, dtype=torch.float32),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "returns": ret["returns"],                       # 归一化后（或原始，取决于开关）
        "raw_returns": ret["raw_returns"],               # 未归一化
        "raw_mean": ret["raw_mean"],
        "raw_std": ret["raw_std"],
        "episode_return": float(np.sum(rewards)),        # 未折扣总回报 = 存活步数
        "discounted_return": float(ret["raw_returns"][0].item()) if rewards else 0.0,
        "length": len(rewards),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "gamma": float(gamma),
    }


def stack_episodes(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把多条轨迹拼成一个更新批（``(N,)``，N 为批内总步数）。

    与"逐回合更新"的唯一区别是损失里的 ``Σ_t`` 覆盖多回合的步；
    同时保留 ``episode_index`` / ``episode_lengths``，方便做回合级诊断与分段统计。
    """
    if not episodes:
        raise ValueError("episodes 为空，无法拼接更新批")
    states = torch.cat([e["states"].to(torch.float32) for e in episodes], dim=0)
    actions = torch.cat([e["actions"].to(torch.long) for e in episodes], dim=0)
    log_probs = torch.cat([e["log_probs"].to(torch.float32) for e in episodes], dim=0)
    entropies = torch.cat([e["entropies"].to(torch.float32) for e in episodes], dim=0)
    rewards = torch.cat([e["rewards"].to(torch.float32) for e in episodes], dim=0)
    raw_returns = torch.cat([e["raw_returns"].to(torch.float32) for e in episodes], dim=0)
    episode_index = torch.cat(
        [torch.full((e["states"].shape[0],), i, dtype=torch.long) for i, e in enumerate(episodes)],
        dim=0,
    )
    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs,
        "entropies": entropies,
        "rewards": rewards,
        "raw_returns": raw_returns,
        "episode_index": episode_index,
        "episode_lengths": [int(e["states"].shape[0]) for e in episodes],
        "episode_returns": [float(e["episode_return"]) for e in episodes],
        "n_transitions": int(states.shape[0]),
        "n_episodes": len(episodes),
    }


def _grad_norm(parameters: Sequence[torch.Tensor]) -> float:
    """所有参数梯度的全局 L2 范数（即 ``clip_grad_norm_`` 返回的裁剪前指标）。"""
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return float(np.sqrt(total))


# =============================================================================
# 5. 一次策略改进：重算 logπ → 算优势 → 反向传播
# =============================================================================
def update_policy(
    policy: PolicyNetwork,
    episodes: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    critic: Optional[ValueNetwork] = None,
    device: Optional[Union[str, torch.device]] = None,
    gamma: float = 0.99,
    normalize_returns: bool = True,
    baseline: str = "batch_mean",
    normalize_advantages: bool = False,
    critic_coef: float = 0.5,
    entropy_coef: float = 0.0,
    max_grad_norm: float = 0.0,
    reduction: str = "mean",
) -> Dict[str, float]:
    """用一批（一条或多条）轨迹做一步 REINFORCE 更新，返回全部诊断量。

    数据流::

        (s, a) ──重新前向──> logπ_θ(a|s), H ─┐
        G_t ──减基线──> A_t ──归一化──> Â_t ──┤──> L = −Σ logπ·Â (+ c·MSE) ──> backward ──> step

    三个关键性质：

    * **优势是常数权重**：:func:`reinforce_loss` 内部对 ``advantages`` 做 ``detach``，
      所以梯度**只会**流向策略参数（``baseline="value"`` 时 Critic 仅由 MSE 项训练）。
    * **``baseline`` 与 ``normalize_returns`` 正交**：前者决定"减掉什么中心"
      （``none`` / ``batch_mean`` / ``value``），后者决定"要不要再除以标准差"。
      默认组合 ``batch_mean + True`` 就是作业中 ``A = (G − μ) / (σ + ε)`` 的 Return 归一化。
    * **单优化器等价于双优化器**（见模块 docstring）。

    Args:
        policy: 策略网络。
        episodes: 单条轨迹字典、:func:`stack_episodes` 的批字典，或多条轨迹组成的列表。
        optimizer: 优化器；``None`` 时只计算 ``loss`` 与诊断量、**不更新参数**（单元测试 / 评估用）。
        critic: 价值网络（``baseline="value"`` 时必填）。
        device: 前向设备（默认取策略参数所在设备）。
        gamma: 折扣因子（用于记录；回报已在 :func:`run_episode` 中算好）。
        normalize_returns: 是否把"回报 − 基线"标准化到零均值、单位标准差。
        baseline: ``"none"`` / ``"batch_mean"`` / ``"value"``。
        normalize_advantages: 归一化之后是否**再**标准化一次优势（幂等开关，便于消融）。
        critic_coef: Critic 损失权重 ``c``（``L = L_policy + c·L_critic``）。
        entropy_coef: 熵奖励系数（``L = … − β·ΣH``）。
        max_grad_norm: ``>0`` 时做全局梯度裁剪（CartPole 通常不必开启）。
        reduction: ``"mean"`` / ``"sum"``。

    Returns:
        诊断字典（见函数末尾 ``stats``），可用于 TensorBoard 与单测断言。
    """
    baseline = str(baseline).lower()
    if baseline not in BASELINES:
        raise ValueError(f"未知基线 '{baseline}'，可选: {list(BASELINES)}")
    if baseline == "value" and critic is None:
        raise ValueError("baseline='value' 需要传入 critic（价值网络）")

    # 兼容三种入参：单条轨迹字典 / 批字典 / 轨迹列表（统一补齐批统计字段）
    if isinstance(episodes, dict):
        batch = episodes if "n_transitions" in episodes else stack_episodes([episodes])
    else:
        batch = stack_episodes(list(episodes))
    if device is None:
        device = next(policy.parameters()).device
    device = torch.device(device)

    states = batch["states"].to(device=device, dtype=torch.float32)
    actions = batch["actions"].to(device=device, dtype=torch.long)
    raw_returns = batch["raw_returns"].to(device=device, dtype=torch.float32)

    # --- 1) 重算 logπ / 熵（带梯度）；缓存的 logπ 只用于一致性校验 ---------------
    dist = policy.distribution(states)                   # 一次前向同时拿到 logπ 与熵
    log_probs = dist.log_prob(actions.reshape(-1).long())
    entropies = dist.entropy()
    cached = batch["log_probs"].to(device=device, dtype=torch.float32)
    log_prob_drift = float((log_probs.detach() - cached).abs().max().item()) if cached.numel() else 0.0

    # --- 2) 基线 → 优势 --------------------------------------------------------
    critic_loss = None
    critic_values = None
    if baseline == "batch_mean":
        baseline_offset = float(raw_returns.mean().detach())
        adv = raw_returns - raw_returns.mean()
    elif baseline == "none":
        baseline_offset = 0.0
        adv = raw_returns.clone()
    else:  # baseline == "value"
        critic_values = critic.value(states)              # (N,) 带梯度，供 MSE 训练
        adv = raw_returns - critic_values.detach()         # ★ detach：优势不训练 Critic
        baseline_offset = float(critic_values.detach().mean())
        critic_loss = value_loss(critic_values, raw_returns)

    # --- 3) 归一化（默认开启）：(A − mean) / (std + ε) --------------------------
    adv_used = adv
    if normalize_returns and adv_used.numel() > 0:
        adv_used = (adv_used - adv_used.mean()) / (adv_used.std(unbiased=False) + 1e-8)
    if normalize_advantages and adv_used.numel() > 0:
        adv_used = (adv_used - adv_used.mean()) / (adv_used.std(unbiased=False) + 1e-8)

    # --- 4) 损失：策略项（含负号）+ Critic 项 -----------------------------------
    policy_loss, loss_stats = reinforce_loss(
        log_probs,
        adv_used,
        entropies=entropies,
        entropy_coef=float(entropy_coef),
        reduction=str(reduction),
    )
    total_loss = policy_loss
    if critic_loss is not None:
        total_loss = total_loss + float(critic_coef) * critic_loss

    # --- 5) 反向传播 + 参数更新 ------------------------------------------------
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if critic is not None:
        parameters += [p for p in critic.parameters() if p.requires_grad]

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    grad_norm = _grad_norm(parameters)                       # 裁剪前梯度范数
    if max_grad_norm and float(max_grad_norm) > 0:
        torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
    if optimizer is not None:
        optimizer.step()

    with torch.no_grad():
        stats: Dict[str, float] = {
            # 损失
            "policy_loss": float(policy_loss.detach()),
            "loss": float(total_loss.detach()),
            "policy_gradient": float(loss_stats["policy_gradient"]),
            "critic_loss": float(critic_loss.detach()) if critic_loss is not None else 0.0,
            "critic_coef": float(critic_coef) if critic_loss is not None else 0.0,
            "critic_value_mean": float(critic_values.mean()) if critic_values is not None else 0.0,
            # 探索
            "entropy": float(loss_stats["entropy"]),
            "entropy_coef": float(entropy_coef),
            # 基线 / 优势
            "baseline": baseline,
            "baseline_offset": float(baseline_offset),
            "advantage_mean": float(adv_used.mean()) if adv_used.numel() else 0.0,
            "advantage_std": float(adv_used.std(unbiased=False)) if adv_used.numel() else 0.0,
            "advantage_abs_mean": float(adv_used.abs().mean()) if adv_used.numel() else 0.0,
            "advantage_raw_mean": float(adv.mean()) if adv.numel() else 0.0,
            "advantage_raw_std": float(adv.std(unbiased=False)) if adv.numel() else 0.0,
            "return_raw_mean": float(raw_returns.mean()) if raw_returns.numel() else 0.0,
            "return_raw_std": float(raw_returns.std(unbiased=False)) if raw_returns.numel() else 0.0,
            # 开关与批信息
            "normalize_returns": float(bool(normalize_returns)),
            "normalize_advantages": float(bool(normalize_advantages)),
            "reduction": loss_stats["reduction"],
            "log_prob_drift": log_prob_drift,
            "grad_norm": grad_norm,
            "n_steps": float(batch["n_transitions"]),
            "n_episodes": float(batch["n_episodes"]),
            "gamma": float(gamma),
        }
    return stats


# =============================================================================
# 6. 独立评估：衡量"当前策略到底行不行"
# =============================================================================
@torch.no_grad()
def evaluate_policy(
    policy: PolicyNetwork,
    env_name: str = "CartPole-v1",
    device: Union[str, torch.device] = "cpu",
    episodes: int = 20,
    seed: int = 12345,
    gamma: float = 0.99,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    env: Optional[Any] = None,
) -> Dict[str, float]:
    """用**独立评估回合**衡量当前策略的真实水平（不回写训练环境的随机流）。

    与训练回合的区别：
    * ``deterministic=True``：取 ``argmax`` 动作（贪心策略），评估不再有采样噪声；
    * 每个回合用 ``seed + i`` 显式播种，保证同一策略的评估结果可复现、可对比；
    * 不参与任何梯度计算。

    Returns:
        ``{"mean_return", "std_return", "min_return", "max_return", "mean_length",
           "episodes", "deterministic"}``（CartPole 的 ``return`` 就是存活步数）。
    """
    own_env = env is None
    env = env if env is not None else make_env(env_name)
    returns: List[float] = []
    lengths: List[int] = []
    try:
        for i in range(int(episodes)):
            ep = run_episode(
                env,
                policy,
                device=device,
                gamma=gamma,
                normalize_returns=False,          # 评估只关心原始回报
                max_steps=max_steps,
                seed=int(seed) + i,
                deterministic=deterministic,
            )
            returns.append(float(ep["episode_return"]))
            lengths.append(int(ep["length"]))
    finally:
        if own_env:
            env.close()

    arr = np.asarray(returns, dtype=np.float32) if returns else np.zeros(0, dtype=np.float32)
    return {
        "mean_return": float(arr.mean()) if arr.size else 0.0,
        "std_return": float(arr.std(ddof=0)) if arr.size else 0.0,
        "min_return": float(arr.min()) if arr.size else 0.0,
        "max_return": float(arr.max()) if arr.size else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "episodes": float(len(returns)),
        "n_episodes": float(len(returns)),          # 兼容 report_closed_loop 的字段名
        "deterministic": float(bool(deterministic)),
    }

