"""手写 SAC（Soft Actor-Critic）核心模块 —— W9 Task 1。

在 **Pendulum-v1**（连续动作空间 ``[-2.0, 2.0]``，单回合 200 步）上从零实现
off-policy 最大熵强化学习：经验回放池存 transition → **每个环境步**抽一个
mini-batch 更新 Double-Q → 用重参数化（reparameterized）的**压缩高斯 Actor**
最大化 ``Q − α·log π`` → 由目标熵 ``H̄ = −dim(A)`` 自动调节温度 α。

一句话数学::

    # ① 软 Q 的 Bellman 目标：Double-Q 取 min，下一动作由**当前策略**采样，并减去熵项
    y_t       = r_t + γ·(1−done_t)·[ min_i Q_{θ̄_i}(s', a') − α·log π_φ(a'|s') ],  a' ~ π_φ(·|s')
    # ② 双 Q 回归（目标 y 完全 detach ⇒ 没有"追逐自己"的路径，θ 只出现在损失的一侧）
    J_Q(θ_i)  = E_{t~D}[ ( Q_{θ_i}(s, a) − y_t )² ]
    # ③ 策略损失：重参数化 a = tanh(x), x = μ_φ(s) + σ_φ(s)·ε, ε ~ N(0, I)
    J_π(φ)    = E_{s~D, ε}[ α·log π_φ(a|s) − min_i Q_{θ_i}(s, a) ]
    # ④ 温度自适应（目标熵 H̄ = −dim(A)，Pendulum-v1 即 −1.0）
    J(α)      = E[ −α·( log π_φ(a|s) + H̄ ) ]        # 对 log α 求导，α = exp(log α) 保证 α > 0
    # ⑤ 目标网络软更新（Polyak）
    θ̄_i  ←  τ·θ_i + (1−τ)·θ̄_i

压缩高斯：测度（Jacobian）修正
------------------------------
策略先在 **压缩前**空间采样 ``x = μ + σ·ε ∈ ℝ^A``，再 ``a = tanh(x) ∈ (−1, 1)^A``；
环境的动作范围 ``[low, high]`` 由**仿射映射** ``a_env = a·scale + bias``（``scale=(high−low)/2``、
``bias=(high+low)/2``）到达。密度换元 ⇒ 对数密度必须加上换元的 Jacobian 项::

    log π(a) = log N(x; μ, σ²) − Σ_j log(1 − tanh²(x_j))         # tanh 的 |da/dx|
             = Σ_j [ −½((x_j−μ_j)²/σ_j² + log 2πσ_j²) − log(1 − tanh²(x_j)) ]

工程上用数值稳定的等价形式（``log(1−tanh²x) = 2(log2 − x − softplus(−2x))``，
见 :func:`log_one_minus_tanh_sq`），远离饱和区时与朴素写法逐位一致（单测覆盖）。
**注意**：本模块的 ``log π`` 定义在 **tanh 后的归一化动作空间** ``(−1,1)^A`` 上
（与 SB3 的 ``action_dist = SquashedDiagGaussianDistribution`` 一致）——
因为自动温度的**目标熵是"每个动作维 1 nat"**这种与量纲无关的量，
若把 ``log|scale|`` 也扣掉，α 的平衡点会被环境动作尺度偷偷平移。
ReplayBuffer 因此存的是**归一化动作**（Critic 与 Actor 的 ``log π`` 同空间），
只有 ``env.step`` 拿到仿射映射后的 ``a_env``（:meth:`SquashedGaussianActor.to_env_action`）。

W9 思考题：为什么更新 Actor 时要让梯度穿过 Critic，却不能更新 Critic 的参数？
---------------------------------------------------------------------------
**(1) 为什么必须"穿透"**：``J_π(φ)`` 里唯一告诉策略"往哪走"的信息来自 ``Q``:

    ∂J_π/∂φ = E[ α·∂log π_φ(a|s)/∂φ − ∂Q(s,a)/∂a · ∂a/∂φ ]      # a = tanh(μ_φ(s)+σ_φ(s)ε)

``∂Q/∂a`` 只能通过 Q **网络的激活值**反向传播得到，所以 Actor 损失 backward 时
必须沿着 ``Q(s, a)`` 的前向计算图一路走到动作张量 ``a``，再经重参数化路径回到 ``μ_φ, σ_φ``。
如果写成 ``with torch.no_grad(): q = critic(s, a)``（或对 ``a`` 做 ``detach``），
``∂Q/∂a = 0``，策略只剩下 ``α·log π`` 这一项 —— 它会退化成"把熵顶到最大"的
随机策略（Pendulum 上表现为回报永远停在 −1200 附近）。所以**计算图绝不能断**。

**(2) 为什么不能更新 Critic 的参数**：θ 是"评价者"而不是"行动者"。本轮循环里 θ 已经由
``J_Q`` 更新过（拟合 Bellman 目标）；若 Actor 的反向传播再往 θ 上累积 ``α log π − Q`` 的
梯度，等于让评价函数朝"把 Q 值抬高、让策略显得更好"的方向漂移（reward hacking /
自我欺骗），而目标网络与 α 的自适应都建立在"Q 是（近似）无偏评价者"这个前提上。
极端情况：Q 可以被推到任意大，策略学到的只是"如何让 Q 变大"而不是"如何拿高回报"。

**(3) 代码里怎么切断**（都在 :meth:`SACAgent.update_batch` 与 :meth:`SACAgent.actor_loss`）：

* **切断的是参数、不是计算图**：用 :func:`frozen_parameters` 上下文把 Q 网络参数的
  ``requires_grad`` 临时置 ``False``。反向传播时 PyTorch 不会为这些参数累积 ``.grad``
  （``q1.weight.grad is None``），但**动作张量 a 的梯度照常回传到 Actor**（叶子是
  critic 的**输入**而非参数）—— 这正是"穿透图 + 冻结参数"的组合，单测
  ``test_frozen_critic_still_backprops_to_actor`` 逐条验证。
* **三套独立优化器**：critic / actor / alpha 各一个 Adam，Actor 那一步只
  ``actor_optimizer.step()``，绝不碰 ``critic_optimizer``。
  （若图省事把 Q 与 Actor 塞进**同一个** optimizer 再 step，Actor 这一步的 backward
  就会顺手给 Q 一个"抬高 Q 值"的 Adam 步 —— 单测
  ``test_shared_optimizer_would_corrupt_critic`` 把这个 bug 现场复现作为对照。）
* **目标网络全程 no_grad**：``target_critic`` 参数 ``requires_grad_(False)``，只用
  ``.data`` 做 Polyak 平均 ⇒ Bellman 目标 ``y`` 是真正的常数（再叠一层 ``y.detach()``）。
* **α 的自适应不回传 Actor**：``alpha_loss`` 里 ``log_prob.detach()``，温度更新只改
  ``log_alpha`` 一个标量，否则 α 会"绕过"策略损失去操纵自己的学习信号。

模块内七个可独立测试的部件
--------------------------
1. :class:`SquashedGaussianActor` —— 重参数化的压缩高斯策略（``μ / log σ`` 双头）。
2. :class:`ContinuousCritic` —— 双 Q ``Q_i(s, a)``（输入拼接 ``[s, a]``，输出标量）。
3. :class:`ReplayBuffer` —— 定容经验回放池（预分配 numpy 数组 + 大批量随机采样）。
4. :func:`soft_update` —— Polyak 软更新 ``τ``（纯函数，便于单测）。
5. :class:`SACAgent` —— 集成 Double-Q 损失、Actor 策略损失、α 自适应与三步更新。
6. :func:`build_sac_agent` / :func:`build_sac_optimizers` —— 配置驱动装配。
7. :func:`resolve_target_entropy` —— ``"auto" | float | null`` 的解析（含边界情形）。

与 W8 PPO 的对照
----------------
+---------------+-----------------------------+-----------------------------------+
| 维度          | PPO（W8，on-policy）        | SAC（W9，off-policy）             |
+===============+=============================+===================================+
| 数据          | 当前策略的 rollout，用完即弃 | ReplayBuffer 反复采样（10⁵ 容量） |
+---------------+-----------------------------+-----------------------------------+
| 更新频率      | 1 个 rollout → K 轮遍历      | 每个环境步 → 1 次 mini-batch 更新 |
+---------------+-----------------------------+-----------------------------------+
| 动作分布      | 高斯 + 环境侧 clip          | tanh 压缩高斯（自带 Jacobian 修正）|
+---------------+-----------------------------+-----------------------------------+
| 探索          | 显式 σ / 熵正则             | 最大熵目标 + 自动温度 α           |
+---------------+-----------------------------+-----------------------------------+
| 价值网络      | ``V(s)`` 单头               | ``Q(s,a)`` 双头 + 目标网络        |
+---------------+-----------------------------+-----------------------------------+
| 样本效率      | ~20 万步收敛                | ~2–4 万步收敛（作业预检项）       |
+---------------+-----------------------------+-----------------------------------+

运行方式（无需 pytest，可独立跑单测）::

    单测覆盖 ``src/sac.py`` 的每个数学部件；训练脚本见
    ``experiments/train_sac.py``（热身 → 交替"环境步 / 梯度步"→ 定期独立评估）。

运行方式（无需 pytest，可独立跑单测）::

    python tests/test_sac.py
"""

from __future__ import annotations

import contextlib
import copy
import math
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:  # 兼容"作为包导入"与"直接运行"两种场景
    from .ppo import (                 # 复用 W8 已验证的工具（MLP 构造 / 归一化 / 评估）
        RunningMeanStd,
        _mlp,
        count_parameters,
        env_max_steps,
        evaluate_policy,
        make_env,
    )
except ImportError:  # pragma: no cover
    from ppo import (  # type: ignore
        RunningMeanStd,
        _mlp,
        count_parameters,
        env_max_steps,
        evaluate_policy,
        make_env,
    )

__all__ = [
    "LOG_STD_MIN",
    "LOG_STD_MAX",
    "TANH_EPS",
    "DEFAULT_ACTION_LOW",
    "DEFAULT_ACTION_HIGH",
    "SAC_UPDATE_KEYS",
    "log_one_minus_tanh_sq",
    "squashed_gaussian_log_prob",
    "resolve_target_entropy",
    "frozen_parameters",
    "soft_update",
    "sync_target",
    "SquashedGaussianActor",
    "ContinuousCritic",
    "ReplayBuffer",
    "SACAgent",
    "build_sac_agent",
    "build_sac_optimizers",
    "random_norm_action",
    "grad_norm",
    "RunningMeanStd",
    "evaluate_policy",
    "make_env",
    "env_max_steps",
    "count_parameters",
]


# =============================================================================
# 0. 常量与数学工具
# =============================================================================
LOG_STD_MIN = -20.0          # log σ 下界：σ ≈ 2e-9（与 W8 PPO 保持一致）
LOG_STD_MAX = 2.0            # log σ 上界：σ ≈ 7.39
TANH_EPS = 1e-6              # log(1 − tanh²x) 的数值保护（x 饱和时 1−tanh²x → 0）

DEFAULT_ACTION_LOW = -2.0    # Pendulum-v1 动作下界
DEFAULT_ACTION_HIGH = 2.0    # Pendulum-v1 动作上界

#: :meth:`SACAgent.update_batch` 的诊断量键名（训练脚本按此写 TensorBoard）
SAC_UPDATE_KEYS = (
    "critic_loss", "q1_loss", "q2_loss", "actor_loss", "alpha_loss",
    "alpha", "log_alpha", "target_entropy",
    "q1_mean", "q2_mean", "target_q_mean", "next_q_mean", "td_abs_mean",
    "q_pi_mean", "log_prob_mean", "entropy_mean", "entropy_target_gap",
    "mean_mu_abs", "mean_std", "log_std_mean", "saturation_fraction",
    "reward_mean", "done_fraction", "critic_grad_norm", "actor_grad_norm",
    "alpha_grad_norm", "critic_frozen_ok", "batch_size", "update_calls",
)


def log_one_minus_tanh_sq(x: torch.Tensor) -> torch.Tensor:
    """数值稳定的 ``log(1 − tanh²(x))``（压缩高斯的对数 Jacobian 项）。

    用等价变形

        log(1 − tanh²x) = 2·( log 2 − x − softplus(−2x) )

    避免 ``tanh(x) → ±1`` 时 ``1 − tanh²x`` 下溢到 0 造成 ``log 0 = −inf``；
    ``|x| ≲ 4`` 时与 ``torch.log(1 − tanh(x)² + ε)`` 逐位一致（单测覆盖）。
    """
    return 2.0 * (math.log(2.0) - x - torch.nn.functional.softplus(-2.0 * x))


def squashed_gaussian_log_prob(
    pre_tanh: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    eps: float = TANH_EPS,
) -> torch.Tensor:
    """**手写**压缩高斯对数密度，动作维求和后返回 ``(B, 1)``::

        log π(a) = log N(x; μ, σ²) − Σ_j log(1 − tanh²(x_j)),   a = tanh(x)

    单独抽成纯函数，是为了让"``log π`` 究竟定义在哪个动作空间"在代码里一眼可见
    （见模块 docstring 的"压缩高斯：测度修正"一节）。
    """
    var = std.pow(2)
    log_prob = -0.5 * ((pre_tanh - mean).pow(2) / var + torch.log(2.0 * math.pi * var))
    log_det = -torch.log(1.0 - torch.tanh(pre_tanh).pow(2) + float(eps))
    return (log_prob + log_det).sum(dim=-1, keepdim=True)


def resolve_target_entropy(value: Any, action_dim: int) -> Optional[float]:
    """把配置项 ``target_entropy`` 解析为浮点数；``None`` 表示**固定 α**。

    * ``"auto"`` / ``"-dim_a"`` / ``"heuristic"`` ⇒ ``−action_dim``（SAC 论文启发式，
      Pendulum-v1 即 ``−1.0``：目标熵"每维 1 nat"）；
    * 具体数字（含 ``0``）⇒ 原样使用；
    * ``None`` / ``"none"`` / ``"null"`` / ``"fixed"`` ⇒ ``None``（关闭 α 自适应）。
    """
    if value is None:
        return None
    if isinstance(value, bool):      # 防御：YAML 的 true/false 不应被当成数字
        raise ValueError(f"target_entropy 不支持布尔值: {value!r}")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("", "none", "null", "fixed", "false"):
        return None
    if text in ("auto", "-dim_a", "auto_neg_dim", "heuristic"):
        return -float(action_dim)
    try:
        return float(text)
    except ValueError as exc:  # pragma: no cover - 配置写错时给出明确提示
        raise ValueError(
            f"无法解析 target_entropy={value!r}（可选 auto | 具体浮点数 | none）"
        ) from exc


@contextlib.contextmanager
def frozen_parameters(*modules: Optional[nn.Module]) -> Iterator[None]:
    """**临时**把若干模块参数的 ``requires_grad`` 置 ``False``（退出时原样恢复）。

    这是 W9 思考题的代码落点之一：更新 Actor 时**只冻结参数、不冻结计算图**。
    冻结参数 ⇒ 反向传播不会为 Critic 的权重累积 ``.grad``（也就不会被任何 optimizer
    更新）；计算图仍连通 ⇒ ``∂Q(s,a)/∂a`` 照常回传到 Actor，策略依旧能"看见"价值函数
    给出的方向。对比写法 ``with torch.no_grad()`` 会把整张 Q 的计算图切断
    （``∂Q/∂a = 0``），Actor 就学不到东西了。
    """
    previous: List[Tuple[nn.Parameter, bool]] = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            previous.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, flag in previous:
            param.requires_grad_(flag)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak 软更新 ``θ̄ ← τ·θ + (1−τ)·θ̄``（在 ``.data`` 上原地运算 ⇒ 不建图、不累积梯度）。

    ``τ = 1`` ⇒ 硬拷贝（等价于 :func:`sync_target`）；``τ = 0`` ⇒ 目标网络保持不变。
    """
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau 必须在 [0, 1] 内，收到 {tau}")
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def sync_target(target: nn.Module, source: nn.Module) -> None:
    """硬拷贝目标网络（``τ = 1`` 的特例；也用于 ``target_critic`` 的初始化）。"""
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


# =============================================================================
# 1. 压缩高斯 Actor 与双 Q Critic
# =============================================================================
def _orthogonal_init(
    module: nn.Module,
    gain: float = 1.0,
    bias: float = 0.0,
    last_gain: Optional[float] = None,
) -> nn.Module:
    """对模块内所有 ``nn.Linear`` 做正交初始化（可选给**最后一层**单独的增益）。

    ``gain`` 的常用取值：隐藏层 ``√2``（ReLU 保持方差）；策略头 ``1e-3``
    （初始 ``μ ≈ 0`` ⇒ ``tanh(μ) ≈ 0`` ⇒ 初始动作 ≈ 0，不会一上来就猛打力矩）；
    Q 输出头 ``1.0``。``bias`` 同样被显式赋值（策略的 ``log_std`` 头用它注入初值）。
    """
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    for index, linear in enumerate(linears):
        g = float(last_gain) if (last_gain is not None and index == len(linears) - 1) else float(gain)
        nn.init.orthogonal_(linear.weight, gain=g)
        nn.init.constant_(linear.bias, float(bias))
    return module


class SquashedGaussianActor(nn.Module):
    """重参数化的**压缩高斯策略**：``x = μ_φ(s) + σ_φ(s)·ε``，``a = tanh(x)``。

    与 W8 PPO 的 Actor（状态无关 ``log_std`` + 环境侧 clip）的三点区别：

    1. ``log_std`` 由网络输出（**状态相关**：离目标很远时该多探索，靠近时该收敛），
       并夹紧到 ``[log_std_min, log_std_max]``；``μ / log σ`` 用两个线性头共享主干。
    2. ``tanh`` 压缩 ⇒ 动作**天生**落在 ``(−1, 1)``，不会像 PPO 那样"采样越界再 clip"；
       代价是 ``log π`` 必须加 Jacobian 修正项（:func:`squashed_gaussian_log_prob`）。
    3. 采样走**重参数化** ``μ + σ·ε``（PPO 用 ``log π_old`` 做重要性采样，不需要对动作
       求导；SAC 的策略梯度 ``∂Q/∂a`` 必须穿过采样路径），因此这里显式写出 ``ε`` 的来源，
       并允许传入固定 ``generator`` 便于单测复现。

    ``log π`` 定义在**归一化动作空间** ``(−1,1)^A``（见模块 docstring）；
    喂给环境的 ``a_env = a·scale + bias`` 由 :meth:`to_env_action` 完成。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        log_std_init: float = -0.5,
        action_low: float = DEFAULT_ACTION_LOW,
        action_high: float = DEFAULT_ACTION_HIGH,
        trunk_init_gain: float = 1.4142135,
        head_init_gain: float = 1e-3,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.log_std_init = float(log_std_init)
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        # 主干：复用 W8 的 _mlp（num_layers 个 Linear+激活 + 一个输出层），去掉输出层
        # ⇒ 得到"num_layers 个隐藏层、每层都带激活"的共享主干（SB3 的 net_arch 语义）。
        self.trunk = nn.Sequential(
            *list(_mlp(self.state_dim, self.hidden_dim, self.hidden_dim,
                       self.num_layers, self.activation))[:-1]
        )
        self.mu_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.action_dim)
        _orthogonal_init(self.trunk, gain=trunk_init_gain)
        _orthogonal_init(self.mu_head, gain=head_init_gain)
        _orthogonal_init(self.log_std_head, gain=head_init_gain, bias=self.log_std_init)

        # 归一化动作 (−1,1) → 环境动作 [low, high] 的仿射映射（只用于 env.step）
        span = self.action_high - self.action_low
        self.register_buffer("action_scale", torch.tensor([span / 2.0]))
        self.register_buffer("action_bias", torch.tensor([(self.action_high + self.action_low) / 2.0]))

    # ---------------------------------------------------------------- #
    # 前向 / 分布参数
    # ---------------------------------------------------------------- #
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(μ_φ(s), log σ_φ(s))``，形均 ``(B, action_dim)``（``log σ`` 已夹紧）。"""
        features = self.trunk(state)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), self.log_std_min, self.log_std_max)
        return mu, log_std

    def std(self, state: torch.Tensor) -> torch.Tensor:
        """动作标准差 ``σ_φ(s) = exp(clip(log σ))``，形 ``(B, action_dim)``。"""
        _, log_std = self.forward(state)
        return torch.exp(log_std)

    def to_env_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        """``a_env = a·scale + bias``（仿射到环境动作范围；**不参与** ``log π``）。

        会自动把输入搬到 ``action_scale`` 所在的设备 / dtype（策略在 GPU 上时
        ``env.step`` 接收的是 numpy 动作，回来往往还是 CPU 张量）。
        """
        action_norm = action_norm.to(device=self.action_scale.device, dtype=self.action_scale.dtype)
        return action_norm * self.action_scale + self.action_bias

    def to_norm_action(self, action_env: torch.Tensor) -> torch.Tensor:
        """:meth:`to_env_action` 的逆映射（调试 / 热身期随机动作：``[−2,2]`` → ``[−1,1]``）。"""
        action_env = action_env.to(device=self.action_scale.device, dtype=self.action_scale.dtype)
        return (action_env - self.action_bias) / self.action_scale


    # ---------------------------------------------------------------- #
    # 采样（重参数化，可微）
    # ---------------------------------------------------------------- #
    def sample(
        self,
        state: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """采样（或取均值）动作，返回策略损失与日志需要的全部张量（**保留计算图**）。

        Returns:
            ``pre_tanh``：``x = μ + σ·ε``（重参数化点，``∂x/∂φ`` 存在 ⇒ Actor 可学）；
            ``action_norm``：``tanh(x) ∈ (−1,1)``（**Critic 的输入**：buffer 里存的就是它）；
            ``action_env``：仿射到环境范围的动作（**只喂给 env.step**）；
            ``log_prob``：``(B, 1)``，压缩高斯对数密度（含 Jacobian 修正）；
            ``mean`` / ``mu`` / ``std`` / ``log_std``：诊断量；
            ``saturation``：本 batch 中 ``|a_norm| > 0.99`` 的比例（tanh 饱和诊断）。
        """
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        if deterministic:
            pre_tanh = mu
        else:
            eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
            pre_tanh = mu + std * eps
        action_norm = torch.tanh(pre_tanh)
        log_prob = squashed_gaussian_log_prob(pre_tanh, mu, std)
        return {
            "pre_tanh": pre_tanh,
            "action_norm": action_norm,
            "action_env": self.to_env_action(action_norm),
            "log_prob": log_prob,
            "mean": torch.tanh(mu),                     # 分布的"确定性动作"
            "mu": mu,
            "std": std,
            "log_std": log_std,
            "saturation": (action_norm.abs() > 0.99).float().mean(),
        }

    def action_log_prob(
        self, state: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(action_norm, log_prob)``（SB3 风格接口，即 :meth:`sample` 的两个字段）。"""
        out = self.sample(state, generator=generator)
        return out["action_norm"], out["log_prob"]

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """**与环境交互**用的采样接口（``no_grad``；返回字段与 :meth:`sample` 相同）。

        返回字典里的 ``action_env`` 会被 ``src.ppo.evaluate_policy`` 直接取用，
        因此 W9 的独立评估与 W8 完全共用同一套评估代码（无重复实现、口径一致）。
        """
        return self.sample(state, deterministic=deterministic)


class ContinuousCritic(nn.Module):
    """连续动作 Q 网络（**Double-Q**）：输入拼接 ``[s, a]``，输出标量 ``Q_i(s, a)``。

    * ``n_critics = 2``（默认，Double-Q 标准）：两套**独立初始化**的 MLP，
      Bellman 目标取 ``min(Q1, Q2)`` 抑制过估计；独立初始化 ⇒ 初始 ``Q1 ≠ Q2``
      （单测断言），这正是"两个估计的误差不相关"这一无偏前提的来源。
    * 训练时同时回归两个 Q；Actor 更新只看 ``min(Q1, Q2)``（悲观估计，
      ``use_min_for_actor`` 可关）。
    * **目标网络**（``target_critic``）由 :class:`SACAgent` 持有一份 ``deepcopy``：
      参数 ``requires_grad_(False)``、只用 ``.data`` 做 Polyak 平均。
    """

    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        n_critics: int = 2,
        trunk_init_gain: float = 1.4142135,
        out_init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation = str(activation)
        self.n_critics = max(1, int(n_critics))
        self.input_dim = self.state_dim + self.action_dim

        self.nets = nn.ModuleList(
            [
                _orthogonal_init(
                    _mlp(self.input_dim, 1, self.hidden_dim, self.num_layers, self.activation),
                    gain=trunk_init_gain,
                    last_gain=out_init_gain,
                )
                for _ in range(self.n_critics)
            ]
        )

    @staticmethod
    def _concat(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """把状态与动作拼成 ``(B, state_dim + action_dim)``（含形状 / batch 维校验）。"""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if state.dim() != 2 or action.dim() != 2:
            raise ValueError(f"state / action 必须是二维张量，收到 {state.shape} / {action.shape}")
        if state.shape[0] != action.shape[0]:
            raise ValueError(f"batch 维不一致: state {state.shape[0]} vs action {action.shape[0]}")
        return torch.cat([state, action], dim=-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """返回 ``[Q_1, …, Q_n]``（每个形 ``(B, 1)``）。"""
        inputs = self._concat(state, action)
        return [net(inputs) for net in self.nets]

    def q_values(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        """:meth:`forward` 的语义化别名。"""
        return self.forward(state, action)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只算第一个 Q（形 ``(B, 1)``）。"""
        return self.nets[0](self._concat(state, action))

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """``min_i Q_i(s, a)``（形 ``(B, 1)``；**保持计算图** —— Actor 需要它）。"""
        return torch.min(torch.cat(self.forward(state, action), dim=-1), dim=-1, keepdim=True).values


# =============================================================================
# 2. 经验回放池（off-policy 的"数据心脏"）
# =============================================================================
class ReplayBuffer:
    """定容经验回放池：预分配 numpy 数组 + 环形覆盖 + 大批量随机采样。

    存储字段（与 SAC 的更新公式一一对应）::

        observations (C, S)      next_observations (C, S)
        actions      (C, A)      ← **归一化动作**（tanh 空间 (−1,1)）；
                                  与 Actor 的 log π 同空间、与 Critic 的输入同空间
        rewards      (C, 1)      terminated (C, 1)      truncated (C, 1)

    两个容易踩的坑（本类显式给出保证，单测覆盖）：

    1. **truncated 不等于 done**。Pendulum-v1 没有真终止，只在 200 步处 ``truncated``；
       把它当终止会让 Bellman 目标丢掉 ``γ·V(s_200)`` 这一项、系统性低估价值
       （W8 PPO 的 ``bootstrap_on_truncation`` 消融实测差了约 100 分）。
       因此 :meth:`sample` 返回的 ``done`` 默认 = ``terminated``；
       只有显式关掉 ``bootstrap_on_truncation`` 才会把 ``truncated`` 也算进 ``done``。
    2. **动作空间要和 Critic 一致**。buffer 里必须是策略输出的那个动作（归一化空间），
       否则 ``Q(s, a)`` 学到的输入分布与 Actor 采样出的输入分布不是同一个坐标系 ——
       这是 off-policy 最隐蔽的错配之一。

    容量 10⁵ ≫ 单次训练 4 万步 ⇒ 整个训练过程**不会覆盖**任何早期经验
    （用 ``is_full`` / ``fill_ratio`` 监控这一点）。
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 3,
        action_dim: int = 1,
        device: Union[str, torch.device] = "cpu",
        bootstrap_on_truncation: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity 必须为正，收到 {capacity}")
        self.capacity = capacity
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        self.seed = None if seed is None else int(seed)
        self._generator: Optional[torch.Generator] = (
            None if self.seed is None else torch.Generator().manual_seed(self.seed)
        )

        self.observations = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.next_observations = np.zeros((capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.truncated = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0            # 环形写指针
        self.size = 0           # 当前有效样本数
        self.total_added = 0    # 历史累计写入次数（含已被覆盖的）

    # ---------------------------------------------------------------- #
    # 基本属性
    # ---------------------------------------------------------------- #
    def __len__(self) -> int:
        return int(self.size)

    @property
    def is_full(self) -> bool:
        """是否已经写过 ``capacity`` 条（此后开始覆盖最旧的数据）。"""
        return self.total_added >= self.capacity

    def can_sample(self, batch_size: int = 1) -> bool:
        """是否有足够样本可采样（``size >= batch_size``）。"""
        return self.size >= int(batch_size)

    # ---------------------------------------------------------------- #
    # 写入
    # ---------------------------------------------------------------- #
    @staticmethod
    def _as_row(value: Any, dim: int, dtype: np.dtype = np.float32) -> np.ndarray:
        """把标量 / 序列 / 张量规范化为 ``(dim,)`` 的一行（元素个数必须等于 ``dim``）。"""
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.asarray(arr, dtype=dtype).reshape(-1)
        if arr.size != dim:
            raise ValueError(f"期望 {dim} 维，收到 {arr.size} 维（{value!r}）")
        return arr

    def add(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool = False,
        truncated: bool = False,
    ) -> int:
        """写入一条 transition，返回它所在槽位下标（覆盖最旧数据时下标会复用）。"""
        index = self.ptr
        self.observations[index] = self._as_row(observation, self.state_dim)
        self.next_observations[index] = self._as_row(next_observation, self.state_dim)
        self.actions[index] = self._as_row(action, self.action_dim)
        self.rewards[index, 0] = float(reward)
        self.terminated[index, 0] = float(bool(terminated))
        self.truncated[index, 0] = float(bool(truncated))
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1
        return index

    def add_batch(
        self,
        observations: Any,
        actions: Any,
        rewards: Any,
        next_observations: Any,
        terminated: Any = None,
        truncated: Any = None,
    ) -> int:
        """批量写入（热身期一次灌入多条经验）；返回写入条数。"""
        obs = np.asarray(observations, dtype=np.float32).reshape(-1, self.state_dim)
        next_obs = np.asarray(next_observations, dtype=np.float32).reshape(-1, self.state_dim)
        acts = np.asarray(actions, dtype=np.float32).reshape(-1, self.action_dim)
        rews = np.asarray(rewards, dtype=np.float32).reshape(-1)
        term = (np.zeros_like(rews) if terminated is None
                else np.asarray(terminated, dtype=np.float32).reshape(-1))
        trunc = (np.zeros_like(rews) if truncated is None
                 else np.asarray(truncated, dtype=np.float32).reshape(-1))
        n = int(obs.shape[0])
        if not (next_obs.shape[0] == acts.shape[0] == rews.size == term.size == trunc.size == n):
            raise ValueError("add_batch: 各字段的样本数不一致")
        for i in range(n):
            self.add(obs[i], acts[i], float(rews[i]), next_obs[i], bool(term[i]), bool(trunc[i]))
        return n


    # ---------------------------------------------------------------- #
    # 采样
    # ---------------------------------------------------------------- #
    def done_mask(self, terminated: np.ndarray, truncated: np.ndarray) -> np.ndarray:
        """按 ``bootstrap_on_truncation`` 计算 Bellman 目标的终止掩码 ``done``。

        ``True`` ⇒ 该 transition 是"真终止"，目标里**不含** ``γ·Q(s', a')``。
        """
        terminated = np.asarray(terminated, dtype=np.float32)
        if self.bootstrap_on_truncation:
            return terminated
        return np.maximum(terminated, np.asarray(truncated, dtype=np.float32))

    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """**有放回**均匀随机采样一个 mini-batch，返回放在 ``self.device`` 上的张量字典。

        Returns:
            ``observation / next_observation``：``(B, S)``；
            ``action``：``(B, A)``（归一化动作，Critic 的输入）；
            ``reward``：``(B, 1)``；``done``：``(B, 1)``（Bellman 掩码）；
            ``terminated / truncated``：``(B, 1)``（诊断用）；
            ``index``：``(B,)`` int64 槽位下标（复盘"这条经验被采样过几次"）。
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，收到 {batch_size}")
        if not self.can_sample(batch_size):
            raise ValueError(
                f"回放池样本不足：size={self.size} < batch_size={batch_size}"
                "（检查 warmup_steps 是否 ≥ batch_size）"
            )
        gen = generator if generator is not None else self._generator
        index = torch.randint(0, self.size, (batch_size,), generator=gen).numpy()
        return self.gather(index)

    def gather(self, index: Union[np.ndarray, torch.Tensor, Sequence[int]]) -> Dict[str, torch.Tensor]:
        """按下标取一批样本（:meth:`sample` 的确定性版本，便于单测逐条核对）。"""
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()
        idx = np.asarray(index, dtype=np.int64).reshape(-1)
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self.size):
            raise IndexError(f"下标越界: 取值应落在 [0, {self.size})，收到 [{idx.min()}, {idx.max()}]")

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32, device=self.device)

        terminated = self.terminated[idx]
        truncated = self.truncated[idx]
        done = self.done_mask(terminated, truncated).reshape(-1, 1)
        return {
            "observation": _to_tensor(self.observations[idx]),
            "next_observation": _to_tensor(self.next_observations[idx]),
            "action": _to_tensor(self.actions[idx]),
            "reward": _to_tensor(self.rewards[idx]),
            "done": _to_tensor(done),
            "terminated": _to_tensor(terminated),
            "truncated": _to_tensor(truncated),
            "index": torch.as_tensor(idx, dtype=torch.int64, device=self.device),
        }

    # ---------------------------------------------------------------- #
    # 监控 / 存档
    # ---------------------------------------------------------------- #
    def stats(self) -> Dict[str, float]:
        """回放池诊断（写 TensorBoard：多久填满、奖励尺度、终止比例、动作覆盖率）。"""
        if self.size == 0:
            return {
                "size": 0.0, "capacity": float(self.capacity), "fill_ratio": 0.0,
                "total_added": float(self.total_added), "reward_mean": float("nan"),
                "reward_std": float("nan"), "terminated_fraction": 0.0,
                "truncated_fraction": 0.0, "action_abs_mean": float("nan"),
            }
        n = self.size
        return {
            "size": float(n),
            "capacity": float(self.capacity),
            "fill_ratio": float(n) / float(self.capacity),
            "total_added": float(self.total_added),
            "reward_mean": float(self.rewards[:n].mean()),
            "reward_std": float(self.rewards[:n].std()),
            "terminated_fraction": float(self.terminated[:n].mean()),
            "truncated_fraction": float(self.truncated[:n].mean()),
            "action_abs_mean": float(np.abs(self.actions[:n]).mean()),
        }

    def state_dict(self) -> Dict[str, Any]:
        """导出完整状态（随检查点保存；4 万步 × 10 维 float32 ≈ 6 MB，可接受）。"""
        n = self.size
        return {
            "observations": self.observations[:n].copy(),
            "next_observations": self.next_observations[:n].copy(),
            "actions": self.actions[:n].copy(),
            "rewards": self.rewards[:n].copy(),
            "terminated": self.terminated[:n].copy(),
            "truncated": self.truncated[:n].copy(),
            "ptr": int(self.ptr), "size": int(n), "total_added": int(self.total_added),
            "capacity": int(self.capacity), "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "bootstrap_on_truncation": bool(self.bootstrap_on_truncation),
        }

    def load_state_dict(self, state: Optional[Dict[str, Any]]) -> None:
        """恢复状态（``None`` 时保持当前内容不变）；容量 / 维度不一致时直接报错。"""
        if not state:
            return
        if int(state.get("capacity", self.capacity)) != self.capacity:
            raise ValueError(f"检查点容量 {state['capacity']} 与当前 {self.capacity} 不一致")
        if int(state.get("state_dim", self.state_dim)) != self.state_dim or \
                int(state.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("检查点的 state_dim / action_dim 与当前回放池不一致")
        n = int(state["size"])
        if n > self.capacity:
            raise ValueError(f"检查点样本数 {n} 超过容量 {self.capacity}")
        self.observations[:n] = state["observations"][:n]
        self.next_observations[:n] = state["next_observations"][:n]
        self.actions[:n] = state["actions"][:n]
        self.rewards[:n] = state["rewards"][:n]
        self.terminated[:n] = state["terminated"][:n]
        self.truncated[:n] = state["truncated"][:n]
        self.ptr = int(state["ptr"])
        self.size = n
        self.total_added = int(state.get("total_added", n))

    def clear(self) -> None:
        """清空内容（容量 / 配置不变）。"""
        self.ptr = 0
        self.size = 0
        self.total_added = 0


# =============================================================================
# 3. SAC Agent：Double-Q 损失 + Actor 策略损失 + α 自适应
# =============================================================================
class SACAgent(nn.Module):
    """把 Actor / Double-Q / 目标网络 / 温度 α 组装成一个可训练的智能体。

    ``update_batch`` 的三步（严格按 SAC 论文的更新顺序，单测逐步验证）：

    1. **Critic 步**：``J_Q(θ) = Σ_i MSE(Q_{θ_i}(s,a), y)``，
       其中 ``y`` 在 ``torch.no_grad()`` 里由 **target critic** 与 **当前策略采样出的
       下一动作** 算出 ⇒ ``y`` 是常数，不存在"追逐自己"的路径（DQN 式发散风险被消除）。
    2. **Actor 步**：``J_π(φ) = E[ α·log π_φ(a|s) − min_i Q_{θ_i}(s, a) ]``，
       ``a = tanh(μ_φ(s) + σ_φ(s)·ε)``。这一步在 :func:`frozen_parameters` 下计算 ——
       **计算图穿透 Critic（拿到 ``∂Q/∂a``），但 Critic 的参数被冻结（不累积梯度）**，
       这正是 W9 思考题的答案；三步各用独立优化器，Actor 步绝不 ``critic_optimizer.step()``。
    3. **α 步**：``J(α) = −log α·E[log π + H̄]``（对 ``log α`` 求导，``α = exp(log α)``）。
       ``log_prob`` 已 ``detach`` ⇒ 温度更新只改 ``log_alpha`` 这一个标量，
       不会把梯度偷偷回传给 Actor。

    关于梯度方向的直觉（作业思考题"α 前期大还是后期大"）：
    梯度下降把 ``log α`` 往 ``(log π + H̄)`` 的方向推 ⇒ ``E[log π] > −H̄ = dim(A)``
    时 α 变小（策略已经比目标熵更"随机"，该收敛了），``E[log π] < dim(A)`` 时 α 变大
    （熵不足、探索不够，该加大温度）。因此 α 的轨迹读出来就是"当前策略离目标熵有多远"。

    本实现采用与 SB3 相同的写法：损失里用 ``−log α·(log π + H̄)`` 而**不是**精确的
    ``−α·(log π + H̄)``。二者只差一个正因子 α，等价于把 α 的更新幅度重标定
    （避免 α → 0 时温度更新停滞），是社区通行做法；单测锁定这一约定。
    """

    def __init__(
        self,
        actor: SquashedGaussianActor,
        critic: ContinuousCritic,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 1.0,
        target_entropy: Optional[float] = -1.0,
        use_min_for_actor: bool = True,
        alpha_min: float = 0.0,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha 必须为正，收到 {alpha}")
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau 必须落在 (0, 1] 内，收到 {tau}")
        self.actor = actor
        self.critic = critic
        # 目标网络：深拷贝 + 永久冻结参数 + eval 模式（不参与任何 optimizer）
        self.target_critic = copy.deepcopy(critic)
        for param in self.target_critic.parameters():
            param.requires_grad_(False)
        self.target_critic.eval()

        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)
        self.use_min_for_actor = bool(use_min_for_actor)
        self.alpha_min = float(alpha_min)
        self.automatic_entropy = target_entropy is not None
        self.target_entropy: Optional[float] = (
            None if target_entropy is None else float(target_entropy)
        )
        # 温度参数化在 log 空间（保证 α > 0）；关闭自适应时 requires_grad=False
        self.log_alpha = nn.Parameter(
            torch.tensor(math.log(float(alpha)), dtype=torch.float32),
            requires_grad=self.automatic_entropy,
        )
        # 仅用于诊断（打开时会额外算一次 ∂Q/∂a 的范数，证明计算图确实穿透了 Critic）
        self.compute_action_grad = False
        self._update_calls = 0

    # ---------------------------------------------------------------- #
    # 属性
    # ---------------------------------------------------------------- #
    @property
    def alpha(self) -> torch.Tensor:
        """温度 ``α = exp(log α)``（标量张量；需要数值时用 ``float(agent.alpha)``）。"""
        return self.log_alpha.exp()

    @property
    def state_dim(self) -> int:
        return int(self.actor.state_dim)

    @property
    def action_dim(self) -> int:
        return int(self.actor.action_dim)

    def alpha_parameters(self) -> List[nn.Parameter]:
        """温度参数列表（关闭自适应时为空 ⇒ 不建 optimizer、不更新）。"""
        return [self.log_alpha] if self.automatic_entropy else []

    def trainable_parameters(self) -> List[nn.Parameter]:
        """除目标网络外的全部可训练参数。"""
        return list(self.actor.parameters()) + list(self.critic.parameters()) + self.alpha_parameters()

    # ---------------------------------------------------------------- #
    # 目标网络 / 温度
    # ---------------------------------------------------------------- #
    def soft_update_target(self) -> None:
        """Polyak 软更新目标 Q 网络（每个梯度步后调用一次）。"""
        soft_update(self.target_critic, self.critic, self.tau)

    def sync_target(self) -> None:
        """硬拷贝当前 Q 网络到目标网络（``τ = 1`` 的等价物）。"""
        sync_target(self.target_critic, self.critic)

    def clamp_alpha(self) -> None:
        """把 ``α`` 夹到 ``[alpha_min, ∞)``（``alpha_min <= 0`` 时不做任何事）。"""
        if self.alpha_min and self.alpha_min > 0:
            with torch.no_grad():
                self.log_alpha.data.clamp_(min=math.log(float(self.alpha_min)))


    # ---------------------------------------------------------------- #
    # ⑨ Critic 损失：Double-Q 回归到软 Bellman 目标
    # ---------------------------------------------------------------- #
    def critic_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """``J_Q(θ) = Σ_i MSE(Q_{θ_i}(s,a), y)``，返回 ``(loss, 诊断量)``。

        目标（**全程 no_grad**，故 ``y`` 是常数）::

            a'   ~ π_φ(·|s')                  # 当前策略采样（不是取 max：SAC 与 DDPG 的关键区别）
            y_t  = r_t + γ·(1−done_t)·[ min_i Q_{θ̄_i}(s', a') − α·log π_φ(a'|s') ]
        """
        obs = batch["observation"]
        action = batch["action"]
        reward = batch["reward"] * self.reward_scale
        next_obs = batch["next_observation"]
        done = batch["done"]

        with torch.no_grad():
            next_out = self.actor.sample(next_obs)        # 重参数化采样；no_grad ⇒ 不建图
            next_action, next_log_prob = next_out["action_norm"], next_out["log_prob"]
            next_q = torch.cat(self.target_critic(next_obs, next_action), dim=-1)
            next_q_min = torch.min(next_q, dim=-1, keepdim=True).values
            alpha = self.alpha.detach()                   # α 有自己的损失，不进 Critic 的图
            target_q = reward + self.gamma * (1.0 - done) * (next_q_min - alpha * next_log_prob)

        q_values = self.critic(obs, action)               # [Q_1, …, Q_n]，均为 (B, 1)
        per_critic = [(q - target_q).pow(2).mean() for q in q_values]
        loss = torch.stack(per_critic).sum()              # Σ_i（论文写法；0.5 系数只等比缩放梯度）

        info: Dict[str, float] = {
            "critic_loss": float(loss.detach()),
            "q1_loss": float(per_critic[0].detach()),
            "target_q_mean": float(target_q.mean()),
            "next_q_mean": float(next_q_min.mean()),
            "td_abs_mean": float(torch.abs(torch.stack(q_values).mean(0) - target_q).mean()),
            "reward_mean": float(reward.mean()),
            "done_fraction": float(done.mean()),
            "alpha": float(alpha),
            "q1_mean": float(q_values[0].mean()),
        }
        if len(per_critic) > 1:
            info["q2_loss"] = float(per_critic[1].detach())
            info["q2_mean"] = float(q_values[1].mean())
        return loss, info

    # ---------------------------------------------------------------- #
    # ⑩ Actor 损失：穿透 Critic 的图 + 冻结 Critic 的参数（思考题）
    # ---------------------------------------------------------------- #
    def actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """``J_π(φ) = E[ α·log π_φ(a|s) − min_i Q_{θ_i}(s, a) ]``（``a`` 是重参数化采样）。

        关键实现（W9 思考题）：

        * ``with frozen_parameters(self.critic)``：**只把 Q 网络的参数冻结**，
          ``requires_grad False`` ⇒ 反向传播不会为它累积 ``.grad``；
        * **绝不** ``torch.no_grad()``：``a`` 仍是需要梯度的张量，``∂Q(s,a)/∂a``
          会经 Q 网络的激活值一路回传到 ``μ_φ / σ_φ``；
        * ``α`` 取 ``detach()``：温度不参与策略的优化目标（它有自己的损失）。
        """
        obs = batch["observation"]
        out = self.actor.sample(obs)
        action, log_prob = out["action_norm"], out["log_prob"]

        with frozen_parameters(self.critic):
            q_values = self.critic(obs, action)          # ★ 图连通、参数冻结
            q_pi = (
                torch.min(torch.cat(q_values, dim=-1), dim=-1, keepdim=True).values
                if self.use_min_for_actor else q_values[0]
            )
            alpha = self.alpha.detach()                  # α 不从策略损失里学习
            loss = (alpha * log_prob - q_pi).mean()

        info: Dict[str, Any] = {
            "actor_loss": float(loss.detach()),
            "log_prob_mean": float(log_prob.mean()),
            "entropy_mean": float(-log_prob.mean()),     # 微分熵的样本估计 H ≈ −E[log π]
            "q_pi_mean": float(q_pi.mean()),
            "mean_mu_abs": float(out["mu"].abs().mean()),
            "mean_std": float(out["std"].mean()),
            "log_std_mean": float(out["log_std"].mean()),
            "saturation_fraction": float(out["saturation"]),
            "log_prob": log_prob,                        # 传给 alpha_loss（由它再 detach）
        }
        if self.compute_action_grad:
            # 显式验证"梯度确实穿透了 Critic"：∂q_pi/∂a 的范数（保留图，不影响后续 backward）
            try:
                grad = torch.autograd.grad(q_pi.mean(), action, retain_graph=True)[0]
                info["action_grad_norm"] = float(grad.norm())
            except RuntimeError:
                # Critic 与动作无关（例如常数 Q）⇒ 计算图根本不含 action，按 0 处理
                info["action_grad_norm"] = 0.0
        return loss, info


    # ---------------------------------------------------------------- #
    # ⑪ 温度损失：目标熵 H̄ = −dim(A)
    # ---------------------------------------------------------------- #
    def alpha_loss(self, log_prob: torch.Tensor) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """``J(α) = −log α · E[ log π + H̄ ]``（``log_prob`` 内部 detach，不回传 Actor）。

        返回 ``(loss, 诊断量)``；关闭自适应（``target_entropy is None``）时返回 ``(None, {})``。
        """
        if not self.automatic_entropy:
            return None, {}
        log_prob = log_prob.detach()                     # ★ 只让 log_alpha 吃梯度
        loss = -(self.log_alpha * (log_prob + float(self.target_entropy))).mean()
        info = {
            "alpha_loss": float(loss.detach()),
            "log_alpha": float(self.log_alpha.detach()),
            "alpha": float(self.alpha.detach()),
            "target_entropy": float(self.target_entropy),
            "entropy_target_gap": float((-log_prob).mean()) - float(-self.target_entropy),
        }
        return loss, info

    # ---------------------------------------------------------------- #
    # 与环境交互
    # ---------------------------------------------------------------- #
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        """把状态喂给 Actor 采样一个动作（``no_grad``；返回 ``action_env`` 等字段）。"""
        return self.actor.act(state, deterministic=deterministic)
    # ---------------------------------------------------------------- #
    # ⑫ 一次完整更新：Critic → Actor → α → Polyak
    # ---------------------------------------------------------------- #
    def update_batch(
        self,
        batch: Dict[str, torch.Tensor],
        optimizers: Dict[str, Optional[optim.Optimizer]],
        gradient_clip: float = 0.0,
    ) -> Dict[str, float]:
        """执行一个 mini-batch 的 SAC 更新，返回全部标量诊断量（可直接写 TensorBoard）。

        严格的三步 + 一次目标网络软更新，并显式维持三条不变量：

        1. Actor 步里 **Critic 参数不累积梯度**（``critic_frozen_ok == 1.0``）——
           Critic 的梯度在它自己 step 之后被清零，因此 Actor 的 backward 若污染了
           Critic，``.grad`` 一定会重新出现，这个诊断量就是"思考题断言"的在线证明。
        2. Actor 步里 **计算图必须穿过 Critic**：``actor_grad_norm > 0``
           （若误用 ``torch.no_grad()``，Actor 只剩熵项的梯度，诊断量会显著变小）。
        3. 每一步各用**自己的优化器**：``critic/actor/alpha`` 三个 ``optim`` 互不干扰。

        Args:
            batch: :meth:`ReplayBuffer.sample` 的返回值（已按 device 就绪）。
            optimizers: ``{"critic": …, "actor": …, "alpha": …}``；
                ``alpha`` 为 ``None`` 表示固定温度（不做 α 步）。
            gradient_clip: ``>0`` 时对三组参数分别做全局梯度裁剪（SAC 一般不需要）。
        """
        critic_opt = optimizers.get("critic")
        actor_opt = optimizers.get("actor")
        alpha_opt = optimizers.get("alpha")
        if critic_opt is None or actor_opt is None:
            raise ValueError("optimizers 必须包含 'critic' 与 'actor'（'alpha' 可为 None）")
        logs: Dict[str, float] = {}

        # ---------- ① Critic：Double-Q 回归到（detach 的）软 Bellman 目标 ----------
        critic_opt.zero_grad(set_to_none=True)
        critic_loss, critic_info = self.critic_loss(batch)
        critic_loss.backward()
        logs["critic_grad_norm"] = grad_norm(self.critic.parameters())
        if gradient_clip and gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.critic.parameters(), float(gradient_clip))
        critic_opt.step()
        self.critic.zero_grad(set_to_none=True)      # ★ 清空：让下一步的"是否被污染"可判定
        logs.update(critic_info)

        # ---------- ② Actor：图穿透 Critic、参数冻结（W9 思考题）----------
        actor_opt.zero_grad(set_to_none=True)
        actor_loss, actor_info = self.actor_loss(batch)
        actor_loss.backward()
        logs["actor_grad_norm"] = grad_norm(self.actor.parameters())
        logs["critic_frozen_ok"] = float(all(p.grad is None for p in self.critic.parameters()))
        if gradient_clip and gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.actor.parameters(), float(gradient_clip))
        actor_opt.step()
        logs.update({k: v for k, v in actor_info.items() if k != "log_prob"})

        # ---------- ③ α：目标熵自适应（log_prob 已 detach，只更新 log_alpha）----------
        if alpha_opt is not None and self.automatic_entropy and actor_info.get("log_prob") is not None:
            alpha_opt.zero_grad(set_to_none=True)
            alpha_loss, alpha_info = self.alpha_loss(actor_info["log_prob"])
            alpha_loss.backward()
            logs["alpha_grad_norm"] = grad_norm(self.alpha_parameters())
            alpha_opt.step()
            self.clamp_alpha()
            logs.update(alpha_info)
        else:
            logs.update({
                "alpha_loss": 0.0,
                "alpha": float(self.alpha.detach()),
                "log_alpha": float(self.log_alpha.detach()),
                "target_entropy": float(self.target_entropy) if self.automatic_entropy else float("nan"),
                "entropy_target_gap": float("nan"),
                "alpha_grad_norm": 0.0,
            })

        # ---------- ④ 目标网络 Polyak 软更新 ----------
        self.soft_update_target()
        self._update_calls += 1
        logs["batch_size"] = float(batch["observation"].shape[0])
        logs["update_calls"] = float(self._update_calls)
        return logs

    def diagnostics(self) -> Dict[str, float]:
        """与 batch 无关的静态诊断（写最终报告 / 日志）：α、目标熵、参数量。"""
        return {
            "alpha": float(self.alpha.detach()),
            "log_alpha": float(self.log_alpha.detach()),
            "target_entropy": float(self.target_entropy) if self.automatic_entropy else float("nan"),
            "automatic_entropy": float(self.automatic_entropy),
            "gamma": float(self.gamma),
            "tau": float(self.tau),
            "n_critics": float(self.critic.n_critics),
            "update_calls": float(self._update_calls),
            "actor_params": float(count_parameters(self.actor)),
            "critic_params": float(count_parameters(self.critic)),
            # 目标网络参数被冻结（requires_grad=False），count_parameters 会把它们算成 0，
            # 这里直接按 numel 统计，便于日志里核对"目标网络确实是独立一份参数"。
            "target_critic_params": float(sum(p.numel() for p in self.target_critic.parameters())),
            "target_critic_trainable": 0.0,
        }




def grad_norm(parameters: Iterable[torch.Tensor]) -> float:
    """当前 ``.grad`` 的全局 L2 范数（无梯度的参数按 0 计）。"""
    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum())
    return float(math.sqrt(total))


# =============================================================================
# 4. 装配工具（配置驱动）
# =============================================================================
def build_sac_agent(config: Dict[str, Any], device: Union[str, torch.device] = "cpu") -> SACAgent:
    """按配置构建 :class:`SACAgent`（读取 ``model`` / ``training`` 两段）。

    支持的字段（缺省值即 ``configs/sac_config.yaml`` 的取值）::

        model:     state_dim / action_dim / hidden_dim / critic_hidden_dim / num_layers /
                   activation / log_std_min / log_std_max / log_std_init /
                   action_low / action_high / n_critics / trunk_init_gain / head_init_gain /
                   critic_out_init_gain
        training:  gamma / tau / init_alpha / alpha_min / target_entropy("auto"|float|null) /
                   use_min_for_actor / reward_scale
    """
    model_cfg = config.get("model", config) or {}
    train_cfg = config.get("training", {}) or {}
    device = torch.device(device)

    state_dim = int(model_cfg.get("state_dim", 3))
    action_dim = int(model_cfg.get("action_dim", 1))
    hidden_dim = int(model_cfg.get("hidden_dim", 256))
    num_layers = int(model_cfg.get("num_layers", 2))
    activation = str(model_cfg.get("activation", "relu"))
    trunk_gain = float(model_cfg.get("trunk_init_gain", 1.4142135))
    head_gain = float(model_cfg.get("head_init_gain", 1e-3))
    critic_gain = float(model_cfg.get("critic_out_init_gain", 1.0))

    actor = SquashedGaussianActor(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=hidden_dim, num_layers=num_layers, activation=activation,
        log_std_min=float(model_cfg.get("log_std_min", LOG_STD_MIN)),
        log_std_max=float(model_cfg.get("log_std_max", LOG_STD_MAX)),
        log_std_init=float(model_cfg.get("log_std_init", -0.5)),
        action_low=float(model_cfg.get("action_low", DEFAULT_ACTION_LOW)),
        action_high=float(model_cfg.get("action_high", DEFAULT_ACTION_HIGH)),
        trunk_init_gain=trunk_gain, head_init_gain=head_gain,
    ).to(device)
    critic = ContinuousCritic(
        state_dim=state_dim, action_dim=action_dim,
        hidden_dim=int(model_cfg.get("critic_hidden_dim", hidden_dim)),
        num_layers=num_layers, activation=activation,
        n_critics=int(model_cfg.get("n_critics", 2)),
        trunk_init_gain=trunk_gain, out_init_gain=critic_gain,
    ).to(device)

    return SACAgent(
        actor, critic,
        gamma=float(train_cfg.get("gamma", 0.99)),
        tau=float(train_cfg.get("tau", 0.005)),
        alpha=float(train_cfg.get("init_alpha", 1.0)),
        target_entropy=resolve_target_entropy(train_cfg.get("target_entropy", "auto"), action_dim),
        use_min_for_actor=bool(train_cfg.get("use_min_for_actor", True)),
        alpha_min=float(train_cfg.get("alpha_min", 0.0) or 0.0),
        reward_scale=float(train_cfg.get("reward_scale", 1.0)),
    ).to(device)


def build_sac_optimizers(
    agent: SACAgent,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    lr_alpha: float = 3e-4,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> Dict[str, Optional[optim.Optimizer]]:
    """构建**三套独立** Adam：``critic`` / ``actor`` / ``alpha``（固定温度时 alpha=None）。

    为什么必须分开（而不是一个 Adam 装三组 param_group）：

    * 三者的更新**时机与方向**不同 —— Critic 拟合 Bellman 目标（下降 Q 的 TD 误差）、
      Actor 最大化 ``Q − α log π``（会"抬高" Q 的输入方向）、α 调整温度；
      若共用一个 optimizer 的 ``step()``，Actor 那一步会顺手把 Critic 往错方向推；
    * 拆开后，Actor 步只需保证"Critic 不产生梯度"（:func:`frozen_parameters`），
      两条不变量的职责清晰可测（``test_actor_step_does_not_change_critic`` /
      ``test_shared_optimizer_would_corrupt_critic``）。
    """
    def _adam(params, lr: float) -> Optional[optim.Adam]:
        params = [p for p in params if p.requires_grad]
        if not params:
            return None
        return optim.Adam(params, lr=float(lr), betas=tuple(betas), eps=1e-8,
                          weight_decay=float(weight_decay))

    return {
        "critic": _adam(agent.critic.parameters(), lr_critic),
        "actor": _adam(agent.actor.parameters(), lr_actor),
        "alpha": _adam(agent.alpha_parameters(), lr_alpha) if agent.automatic_entropy else None,
    }


def random_norm_action(env: Any, actor: SquashedGaussianActor) -> np.ndarray:
    """热身期用：采一个**归一化空间**的均匀随机动作（与 buffer 的动作坐标系一致）。

    ``env.action_space.sample()`` 给的是环境空间 ``[low, high]``；这里用
    :meth:`SquashedGaussianActor.to_norm_action` 仿射回 ``(−1, 1)`` 并夹到
    ``±(1 − 1e-6)``（tanh 的开区间），这样热身数据与策略采样数据处于同一坐标系。
    随机源由环境的 ``action_space`` 决定（训练脚本用 ``env.reset(seed=...)`` /
    ``action_space.seed(...)`` 固定，保证可复现）。
    """
    action_env = np.asarray(env.action_space.sample(), dtype=np.float32).reshape(-1)
    with torch.no_grad():
        norm = actor.to_norm_action(torch.as_tensor(action_env)).cpu().numpy()
    return np.clip(norm, -1.0 + 1e-6, 1.0 - 1e-6).astype(np.float32)

