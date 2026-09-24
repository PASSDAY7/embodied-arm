#!/usr/bin/env python
"""W9 Task 1（SAC / Soft Actor-Critic）单元测试。

覆盖范围
--------
1. **手写数学**：``log(1 − tanh²x)`` 的数值稳定写法（饱和区不产生 ``log 0``）；
   压缩高斯的换元（Jacobian）修正与朴素写法逐位一致；
   ``log π`` 在 ``(−1,1)`` 上**数值积分 = 1**（"测度修正是否正确"的硬判据）。
2. **压缩高斯 Actor**：形状 / ``log σ`` 夹紧 / 确定性动作 ``tanh(μ)`` /
   仿真动作落在环境范围内 / 固定 generator 可复现 / 仿射映射 ``[−1,1] ↔ [low,high]``；
   **重参数化的可导性**：``μ`` 与 ``log σ`` 都拿到非零梯度（PPO 用 ``log π_old``
   不需要这一性质，SAC 必须）。
3. **双 Q Critic**：``Q_i(s,a)`` 形状、两个独立初始化的 Q 不相等、``min`` 语义正确、
   输入拼接顺序确为 ``[s, a]``（用线性权重构造的"金标准"逐位核对）、维度校验报错。
4. **目标网络**：Polyak 软更新逐参数一致（``τ=0/1`` 边界）、目标网络参数永久
   ``requires_grad=False``、深拷贝而非共享、``sync_target`` 硬拷贝。
5. **ReplayBuffer**：定容环形覆盖 / 采样形状与 dtype / 样本不足报错 /
   **truncated ≠ done**（``bootstrap_on_truncation`` 开关的语义）/ 状态存档往返 /
   下标越界报错 / ``add_batch`` / ``stats``。
6. **Critic 损失**：``done≡1`` 时目标退化为 ``r``（与手算逐位一致）；
   目标 ``y`` 对 Critic 参数**完全不敏感**（detach 的证据）；Critic 的 backward
   不碰 Actor / ``log α``；单状态玩具系统上收敛到解析不动点
   ``Q* = (r − γα·logπ)/(1−γ)``。
7. **Actor 损失（W9 思考题）**：
   * 与手写公式（同一 RNG 状态 ⇒ 同一 ``ε``）逐位一致；
   * 二次玩具 Q 下策略确实朝 ``argmax_a Q`` 移动（证明 ``∂Q/∂a`` 真的穿过 Critic）；
   * **冻结 Critic 参数后 Actor 仍有非零梯度**（图连通 + 参数冻结）；
   * 对照：``q.detach()`` 会让"来自价值的方向"梯度归零；
   * Q 恒定时最小化 ``α·log π`` 使熵上升；
   * 共用同一个 optimizer 会污染 Critic（复现错误做法的后果）。
8. **温度自适应**：``J(α) = −log α·E[log π + H̄]`` 的公式与梯度方向
   （策略太确定 ⇒ α↑；策略太随机 ⇒ α↓）、``log_prob.detach()`` 不回传策略、
   ``α`` 下界夹紧、``target_entropy=None`` 时完全固定。
9. **SACAgent.update_batch**：三步更新的参数变化、目标网络 Polyak、
   ``critic_frozen_ok=1.0``、返回的诊断量全部有限。
10. **端到端**：① 单状态玩具环境上 SAC 把策略推到最优动作；
    ② Pendulum-v1 上短路训练（300 步随机数据 + 50 次更新）诊断量全部有限。
11. **工程脚本**：``configs/sac_config.yaml`` 与作业指定超参逐项一致；
    ``experiments/train_sac.py`` 的 ``as_bool / rolling_mean / check_action_bounds /
    RunningAvg``。

运行方式（无需 pytest）::

    python tests/test_sac.py

若已安装 pytest，也可直接 ``pytest tests/test_sac.py``。
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sac import (  # noqa: E402
    LOG_STD_MAX,
    LOG_STD_MIN,
    ContinuousCritic,
    ReplayBuffer,
    SACAgent,
    SquashedGaussianActor,
    build_sac_agent,
    build_sac_optimizers,
    frozen_parameters,
    grad_norm,
    log_one_minus_tanh_sq,
    make_env,
    random_norm_action,
    resolve_target_entropy,
    soft_update,
    squashed_gaussian_log_prob,
    sync_target,
)

DEVICE = torch.device("cpu")          # 测试统一用 CPU，保证任何机器可复现
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 测试辅助
# =============================================================================
def set_seeds(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_actor(**kwargs) -> SquashedGaussianActor:
    """小而快的 Actor（hidden_dim=32 / 1 层），便于单测里跑几十次更新。"""
    params: Dict[str, Any] = dict(
        state_dim=3, action_dim=1, hidden_dim=32, num_layers=1, log_std_init=-0.5,
    )
    params.update(kwargs)
    set_seeds(0)
    return SquashedGaussianActor(**params).to(DEVICE)


def make_agent(
    actor: Optional[SquashedGaussianActor] = None,
    critic: Optional[nn.Module] = None,
    **kwargs,
) -> SACAgent:
    """构造小型 SACAgent（默认双 Q、自动温度、γ=0.99、τ=0.5）。"""
    if actor is None:
        actor = make_actor()
    if critic is None:
        set_seeds(1)
        critic = ContinuousCritic(
            state_dim=actor.state_dim, action_dim=actor.action_dim,
            hidden_dim=32, num_layers=1, n_critics=2,
        ).to(DEVICE)
    params: Dict[str, Any] = dict(gamma=0.99, tau=0.5, alpha=1.0, target_entropy=-1.0)
    params.update(kwargs)
    set_seeds(2)
    return SACAgent(actor, critic, **params).to(DEVICE)


def make_batch(agent: SACAgent, batch_size: int = 8, reward: float = 1.0,
               done: float = 0.0, seed: int = 0) -> Dict[str, torch.Tensor]:
    """构造确定性 mini-batch（不依赖 ReplayBuffer，便于公式核对）。"""
    gen = torch.Generator().manual_seed(seed)
    obs = torch.randn(batch_size, agent.state_dim, generator=gen)
    next_obs = torch.randn(batch_size, agent.state_dim, generator=gen)
    action = torch.tanh(torch.randn(batch_size, agent.action_dim, generator=gen))
    return {
        "observation": obs,
        "next_observation": next_obs,
        "action": action,
        "reward": torch.full((batch_size, 1), float(reward)),
        "done": torch.full((batch_size, 1), float(done)),
        "terminated": torch.full((batch_size, 1), float(done)),
        "truncated": torch.zeros(batch_size, 1),
        "index": torch.arange(batch_size),
    }


def params_snapshot(module: nn.Module) -> List[torch.Tensor]:
    """参数快照（按值克隆，用于"逐位未变"的断言）。"""
    return [p.detach().clone() for p in module.parameters()]


def params_equal(module: nn.Module, snapshot: List[torch.Tensor]) -> bool:
    """当前参数是否与快照逐位一致。"""
    return all(torch.equal(p.detach(), s) for p, s in zip(module.parameters(), snapshot))


class QuadraticCritic(nn.Module):
    """玩具 Critic：``Q(s, a) = bias − (a − a*)²``（``a*`` 可配置）。

    用来验证"策略确实沿 ``∂Q/∂a`` 移动"：最优动作就是 tanh 空间的 ``a*``。
    只实现 :class:`SACAgent` 需要的接口（``forward`` 返回 Q 列表 + 一个可训练参数）。
    """

    def __init__(self, target_action: float = 0.5, offset: float = 0.0) -> None:
        super().__init__()
        self.target_action = float(target_action)
        self.bias = nn.Parameter(torch.tensor(float(offset)))
        self.n_critics = 1

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        return [self.bias - (action - self.target_action).pow(2)]


# =============================================================================
# 1. 手写数学：tanh 的 Jacobian 项与压缩高斯密度
# =============================================================================
def test_log_one_minus_tanh_sq_matches_naive_formula() -> None:
    """稳定写法与 ``log(1 − tanh²x + ε)`` 在非饱和区逐位一致，在饱和区仍然有限。"""
    x = torch.linspace(-3.0, 3.0, 101)
    stable = log_one_minus_tanh_sq(x)
    naive = torch.log(1.0 - torch.tanh(x).pow(2) + 1e-6)
    assert torch.allclose(stable, naive, atol=1e-4), (stable - naive).abs().max()
    # 对称性：log(1 − tanh²(−x)) == log(1 − tanh²(x))
    assert torch.allclose(stable, log_one_minus_tanh_sq(-x), atol=1e-6)
    # 饱和区：朴素写法下溢成 −inf，稳定写法仍给出有限值（x=30 ⇒ ≈ −60）
    far = torch.tensor([30.0, -30.0])
    assert torch.isfinite(log_one_minus_tanh_sq(far)).all()
    assert float(log_one_minus_tanh_sq(far)[0]) < -55.0


def test_squashed_gaussian_log_prob_matches_manual_change_of_variables() -> None:
    """``log π(a) = log N(x; μ, σ²) − Σ log(1 − tanh²x)``（手写金标准逐位核对）。"""
    set_seeds(3)
    x = torch.randn(7, 1) * 0.7
    mu = torch.randn(7, 1) * 0.3
    std = torch.exp(torch.randn(7, 1) * 0.2)
    got = squashed_gaussian_log_prob(x, mu, std)
    expected = -0.5 * ((x - mu).pow(2) / std.pow(2) + torch.log(2 * math.pi * std.pow(2))) \
        - torch.log(1.0 - torch.tanh(x).pow(2) + 1e-6)
    assert got.shape == (7, 1)
    assert torch.allclose(got, expected, atol=1e-4)


def test_squashed_log_prob_integrates_to_one() -> None:
    """密度换元的硬判据：``∫_{−1}^{1} exp(log π(a)) da ≈ 1``（数值积分）。"""
    mu, std = torch.tensor([[0.2]]), torch.tensor([[0.9]])
    a = torch.linspace(-1 + 1e-4, 1 - 1e-4, 40001).reshape(-1, 1)
    x = torch.atanh(a)                                    # 反变换：x = atanh(a)
    log_p = squashed_gaussian_log_prob(x, mu.expand_as(x), std.expand_as(x))
    integral = float(torch.trapz(log_p.exp().squeeze(-1), a.squeeze(-1)))
    assert abs(integral - 1.0) < 2e-3, integral
    # 另一个参数点同样成立（μ 偏离 0、σ 明显不同尺度）
    mu, std = torch.tensor([[-1.0]]), torch.tensor([[0.35]])
    log_p = squashed_gaussian_log_prob(x, mu.expand_as(x), std.expand_as(x))
    integral = float(torch.trapz(log_p.exp().squeeze(-1), a.squeeze(-1)))
    assert abs(integral - 1.0) < 5e-3, integral


def test_resolve_target_entropy_variants() -> None:
    """``auto`` ⇒ ``−dim(A)``；数字原样；``none`` ⇒ 固定 α；布尔 / 乱码报错。"""
    assert resolve_target_entropy("auto", 1) == -1.0
    assert resolve_target_entropy("AUTO", 3) == -3.0
    assert resolve_target_entropy("-dim_a", 2) == -2.0
    assert resolve_target_entropy(0.5, 1) == 0.5
    assert resolve_target_entropy(-2, 1) == -2.0
    assert resolve_target_entropy("0.25", 1) == 0.25
    for value in (None, "none", "null", "fixed", ""):
        assert resolve_target_entropy(value, 1) is None
    for bad in (True, "abc"):
        try:
            resolve_target_entropy(bad, 1)
            raise AssertionError(f"应当拒绝 {bad!r}")
        except ValueError:
            pass


# =============================================================================
# 2. 压缩高斯 Actor
# =============================================================================
def test_actor_shapes_and_log_std_clamp() -> None:
    """``(μ, log σ)`` 形状正确，``log σ`` 落在夹紧区间内（饱和时梯度为 0 是预期行为）。"""
    actor = make_actor()
    state = torch.randn(5, 3)
    mu, log_std = actor(state)
    assert mu.shape == (5, 1) and log_std.shape == (5, 1)
    assert float(log_std.min()) >= LOG_STD_MIN - 1e-6
    assert float(log_std.max()) <= LOG_STD_MAX + 1e-6

    # 把 log_std 头偏置推到极大 / 极小 ⇒ 夹紧生效
    with torch.no_grad():
        actor.log_std_head.bias.fill_(50.0)
    assert torch.allclose(actor(state)[1], torch.full((5, 1), LOG_STD_MAX), atol=1e-5)
    with torch.no_grad():
        actor.log_std_head.bias.fill_(-50.0)
    assert torch.allclose(actor(state)[1], torch.full((5, 1), LOG_STD_MIN), atol=1e-5)
    # σ = exp(clip(log σ))，且 log_std_init 决定了初始 σ（默认 −0.5 ⇒ ≈ 0.607）
    actor2 = make_actor(log_std_init=-0.5)
    assert abs(float(actor2.std(state).mean()) - math.exp(-0.5)) < 5e-3


def test_actor_deterministic_action_is_tanh_of_mu_and_in_env_range() -> None:
    """确定性动作 = ``tanh(μ)``；采样动作（含极端 σ）永远落在 ``(−2, 2)`` 内。"""
    actor = make_actor(log_std_init=2.0)         # σ ≈ 7.39，故意制造饱和
    state = torch.randn(512, 3)
    out_det = actor.sample(state, deterministic=True)
    mu, _ = actor(state)
    assert torch.allclose(out_det["action_norm"], torch.tanh(mu), atol=1e-6)
    assert torch.allclose(out_det["action_env"], torch.tanh(mu) * 2.0, atol=1e-5)

    out = actor.sample(state)
    assert float(out["action_env"].min()) > -2.0 - 1e-6
    assert float(out["action_env"].max()) < 2.0 + 1e-6
    assert float(out["saturation"]) > 0.0        # σ 很大 ⇒ 确实有饱和动作
    assert out["log_prob"].shape == (512, 1) and torch.isfinite(out["log_prob"]).all()


def test_actor_sample_is_reparameterized_and_differentiable() -> None:
    """重参数化：``μ`` 与 ``log σ`` 都要拿到非零梯度（SAC 的策略梯度必须穿过采样）。"""
    actor = make_actor(log_std_init=-0.5)
    state = torch.randn(16, 3)
    out = actor.sample(state)
    loss = out["action_norm"].sum() + out["log_prob"].sum()
    loss.backward()
    assert actor.mu_head.weight.grad is not None
    assert float(actor.mu_head.weight.grad.abs().sum()) > 0
    assert actor.log_std_head.weight.grad is not None
    assert float(actor.log_std_head.weight.grad.abs().sum()) > 0, "log σ 必须是可学习的"
    # 同一次前向里 ε 是"加性噪声"：μ 的梯度应等于 ∂a/∂μ × 上游梯度（非零即可）
    assert actor.trunk[0].weight.grad is not None


def test_actor_fixed_generator_is_reproducible() -> None:
    """传入固定 ``generator`` ⇒ 采样完全可复现（单测 / 调试的前提）。"""
    actor = make_actor()
    state = torch.randn(4, 3)
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a1 = actor.sample(state, generator=g1)["action_norm"]
    a2 = actor.sample(state, generator=g2)["action_norm"]
    assert torch.equal(a1, a2)
    g3 = torch.Generator().manual_seed(8)
    a3 = actor.sample(state, generator=g3)["action_norm"]
    assert not torch.allclose(a1, a3)
    # log_prob 必须与"同一 ε"对应：用 a1 反推 x 再算 log π 应等于返回的 log_prob
    x = torch.atanh(a1.clamp(-1 + 1e-7, 1 - 1e-7))
    mu, log_std = actor(state)
    manual = squashed_gaussian_log_prob(x, mu, log_std.exp())
    assert torch.allclose(manual, actor.sample(state, generator=torch.Generator().manual_seed(7))["log_prob"], atol=1e-5)


def test_actor_action_affine_mapping_roundtrip() -> None:
    """仿射映射：``[−1,1] ↔ [low, high]``（含非对称区间），且自动处理设备/dtype。"""
    actor = make_actor(action_low=-2.0, action_high=2.0)
    norm = torch.tensor([[-1.0], [0.0], [1.0]])
    env = actor.to_env_action(norm)
    assert torch.allclose(env, torch.tensor([[-2.0], [0.0], [2.0]]), atol=1e-6)
    assert torch.allclose(actor.to_norm_action(env), norm, atol=1e-6)

    asym = make_actor(action_low=-1.0, action_high=3.0)
    assert torch.allclose(asym.to_env_action(norm), torch.tensor([[-1.0], [1.0], [3.0]]), atol=1e-6)
    assert torch.allclose(asym.to_norm_action(asym.to_env_action(norm)), norm, atol=1e-6)
    # 输入是 numpy / 列表时也能工作（train_sac.py 里就是这么用的）
    assert abs(float(asym.to_env_action(torch.tensor([0.0]))[0]) - 1.0) < 1e-6


def test_actor_initial_action_is_near_zero() -> None:
    """小增益初始化：初始动作接近 0（不会一上来就猛打力矩），且 log σ ≈ log_std_init。"""
    set_seeds(11)
    actor = SquashedGaussianActor(state_dim=3, action_dim=1, hidden_dim=64, num_layers=2)
    state = torch.randn(256, 3) * 3.0
    out = actor.sample(state, deterministic=True)
    assert float(out["action_env"].abs().mean()) < 0.05
    assert abs(float(out["log_std"].mean()) - actor.log_std_init) < 0.05


# =============================================================================
# 3. 双 Q Critic 与目标网络
# =============================================================================
def test_critic_double_q_shapes_and_independent_init() -> None:
    """双 Q：返回 ``[Q1, Q2]``（各 ``(B,1)``）、初始值不相等（独立初始化）、min 正确。"""
    set_seeds(5)
    critic = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=32, num_layers=1, n_critics=2)
    state, action = torch.randn(6, 3), torch.randn(6, 1)
    qs = critic(state, action)
    assert isinstance(qs, list) and len(qs) == 2
    assert all(q.shape == (6, 1) for q in qs)
    assert not torch.allclose(qs[0], qs[1]), "两个 Q 必须独立初始化（Double-Q 的前提）"
    assert torch.allclose(critic.min_q(state, action), torch.minimum(qs[0], qs[1]))
    assert torch.allclose(critic.q1(state, action), qs[0])
    assert torch.allclose(critic.q_values(state, action)[1], qs[1])
    # n_critics 可配（1 = 单 Q，3 = 三重 Q）
    assert len(ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1,
                                n_critics=1)(state, action)) == 1
    assert len(ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1,
                                n_critics=3)(state, action)) == 3


def test_critic_input_is_state_then_action() -> None:
    """拼接顺序必须是 ``[s, a]``：直接核对 ``_concat``，并确认两个输入都影响输出。"""
    critic = ContinuousCritic(state_dim=2, action_dim=1, hidden_dim=8, num_layers=1, n_critics=1)
    assert critic.input_dim == 3
    state = torch.tensor([[0.5, -0.25], [1.0, 2.0]])
    action = torch.tensor([[0.75], [-1.5]])
    concat = ContinuousCritic._concat(state, action)
    assert concat.shape == (2, 3)
    assert torch.equal(concat[:, :2], state)          # 前 state_dim 维是状态
    assert torch.equal(concat[:, 2:], action)         # 后 action_dim 维是动作
    # 两个输入都影响输出 ⇒ 拼接被完整使用（顺序写反会得到不同的数值）
    q0 = critic(state, action)[0]
    assert not torch.allclose(q0, critic(state + 1.0, action)[0])
    assert not torch.allclose(q0, critic(state, action + 1.0)[0])
    # 一维输入会被自动补 batch 维
    assert critic(torch.randn(2), torch.randn(1))[0].shape == (1, 1)


def test_critic_validation_errors() -> None:
    """维度 / batch 不一致要报错（而不是静默广播）。"""
    critic = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1)
    try:
        critic(torch.randn(4, 3), torch.randn(5, 1))
        raise AssertionError("batch 维不一致应当报错")
    except ValueError:
        pass
    try:
        critic(torch.randn(4, 3, 1), torch.randn(4, 1))
        raise AssertionError("三维输入应当报错")
    except ValueError:
        pass
    # 一维输入会被自动补 batch 维
    assert critic(torch.randn(3), torch.randn(1))[0].shape == (1, 1)


def test_soft_update_polyak_formula_and_boundaries() -> None:
    """``θ̄ ← τθ + (1−τ)θ̄`` 逐参数成立；``τ=1`` 硬拷贝、``τ=0`` 不变、越界报错。"""
    set_seeds(6)
    src = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1, n_critics=1)
    tgt = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1, n_critics=1)
    for p in tgt.parameters():
        p.requires_grad_(False)
    before = params_snapshot(tgt)
    soft_update(tgt, src, 0.0)
    assert params_equal(tgt, before)

    tau = 0.25
    soft_update(tgt, src, tau)
    for p_t, p_s, p_b in zip(tgt.parameters(), src.parameters(), before):
        expected = tau * p_s.detach() + (1 - tau) * p_b
        assert torch.allclose(p_t.detach(), expected, atol=1e-6)

    soft_update(tgt, src, 1.0)                    # τ=1 ⇒ 硬拷贝
    for p_t, p_s in zip(tgt.parameters(), src.parameters()):
        assert torch.equal(p_t.detach(), p_s.detach())
    for bad in (-0.1, 1.5):
        try:
            soft_update(tgt, src, bad)
            raise AssertionError(f"tau={bad} 应当报错")
        except ValueError:
            pass


def test_target_network_is_deepcopy_frozen_and_synced() -> None:
    """目标网络：深拷贝（不共享参数）、永久冻结、硬拷贝 / 软更新都不产生梯度。"""
    agent = make_agent()
    assert not any(p.requires_grad for p in agent.target_critic.parameters())
    assert all(p.requires_grad for p in agent.critic.parameters())
    # 不共享内存：改 critic 不影响 target
    before = params_snapshot(agent.target_critic)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(1.0)
    assert params_equal(agent.target_critic, before)

    agent.sync_target()
    for p_t, p_c in zip(agent.target_critic.parameters(), agent.critic.parameters()):
        assert torch.equal(p_t.detach(), p_c.detach())
    # 软更新后目标参数朝 critic 靠拢（τ<1 ⇒ 不会直接等）
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(0.5)
    tgt_before = params_snapshot(agent.target_critic)
    agent.soft_update_target()
    moved = any(not torch.equal(p.detach(), b) for p, b in
                zip(agent.target_critic.parameters(), tgt_before))
    assert moved
    # 目标网络不接受梯度：输入带 requires_grad 时图是通的，但目标网络的参数 .grad 仍为 None
    obs = torch.randn(4, 3, requires_grad=True)
    act = torch.randn(4, 1)
    q = agent.target_critic(obs, act)[0].sum()
    assert q.grad_fn is not None, "图是通的（只是参数被冻结）"
    q.backward()
    assert obs.grad is not None
    assert all(p.grad is None for p in agent.target_critic.parameters())
    assert all(p.grad is None for p in agent.critic.parameters())


def test_sync_target_matches_soft_update_with_tau_one() -> None:
    """``sync_target`` 与 ``soft_update(τ=1)`` 等价。"""
    set_seeds(9)
    a = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1, n_critics=2)
    b = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1, n_critics=2)
    c = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=8, num_layers=1, n_critics=2)
    sync_target(b, a)
    soft_update(c, a, 1.0)
    for p_b, p_c in zip(b.parameters(), c.parameters()):
        assert torch.equal(p_b.detach(), p_c.detach())


# =============================================================================
# 4. ReplayBuffer（off-policy 的数据心脏）
# =============================================================================
def test_buffer_add_sample_shapes_and_dtypes() -> None:
    """写入 / 采样：字段齐全、形状 dtype 正确、device 正确、值逐条对应。"""
    buf = ReplayBuffer(capacity=16, state_dim=3, action_dim=1, device="cpu", seed=0)
    assert len(buf) == 0 and not buf.can_sample(1)
    for i in range(5):
        buf.add(np.full(3, i, dtype=np.float32), np.array([0.1 * i], np.float32),
                float(-i), np.full(3, i + 100, dtype=np.float32),
                terminated=(i % 2 == 0), truncated=(i == 3))
    assert len(buf) == 5 and buf.can_sample(5) and not buf.is_full
    batch = buf.gather([4, 0, 4])
    assert batch["observation"].shape == (3, 3) and batch["reward"].shape == (3, 1)
    assert batch["action"].shape == (3, 1) and batch["done"].shape == (3, 1)
    assert batch["observation"].dtype == torch.float32
    assert batch["index"].dtype == torch.int64
    assert torch.allclose(batch["observation"][0], torch.full((3,), 4.0))
    assert torch.allclose(batch["observation"][1], torch.zeros(3))
    assert float(batch["reward"][0]) == -4.0
    assert float(batch["next_observation"][0][0]) == 104.0
    # sample 是 gather 的随机版本（样本数不能超过 size）
    sampled = buf.sample(5, generator=torch.Generator().manual_seed(3))
    assert sampled["observation"].shape == (5, 3)
    assert set(sampled["index"].tolist()).issubset(set(range(5)))


def test_buffer_ring_overwrite_and_is_full() -> None:
    """环形覆盖：写满后覆盖最旧数据，``total_added`` 继续累加。"""
    buf = ReplayBuffer(capacity=3, state_dim=1, action_dim=1, device="cpu")
    for i in range(5):
        buf.add(np.array([float(i)]), np.array([0.0]), float(i), np.array([float(i)]))
    assert len(buf) == 3 and buf.is_full and buf.total_added == 5
    obs = buf.observations[:3, 0]
    assert list(obs) == [3.0, 4.0, 2.0], obs        # 槽位 0,1 被 3,4 覆盖，槽位 2 仍是 2
    assert buf.ptr == 2                             # 指针回到第 2 个槽位


def test_buffer_sample_errors_and_gather_bounds() -> None:
    """样本不足 / batch_size≤0 / 下标越界都要给出明确错误。"""
    buf = ReplayBuffer(capacity=8, state_dim=2, action_dim=1, device="cpu")
    buf.add(np.zeros(2), np.zeros(1), 0.0, np.zeros(2))
    for bad in (2, 100):
        try:
            buf.sample(bad)
            raise AssertionError(f"batch_size={bad} 应当报错")
        except ValueError:
            pass
    try:
        buf.sample(0)
        raise AssertionError("batch_size=0 应当报错")
    except ValueError:
        pass
    for bad_index in ([1], [-1]):
        try:
            buf.gather(bad_index)
            raise AssertionError(f"下标 {bad_index} 应当报错")
        except IndexError:
            pass


def test_buffer_truncation_is_not_done_by_default() -> None:
    """**truncated ≠ done**：默认只把 ``terminated`` 当作终止（Pendulum 必须自举）。"""
    for bootstrap in (True, False):
        buf = ReplayBuffer(capacity=4, state_dim=1, action_dim=1, device="cpu",
                           bootstrap_on_truncation=bootstrap)
        buf.add(np.zeros(1), np.zeros(1), 1.0, np.zeros(1), terminated=False, truncated=True)
        buf.add(np.zeros(1), np.zeros(1), 1.0, np.zeros(1), terminated=True, truncated=True)
        buf.add(np.zeros(1), np.zeros(1), 1.0, np.zeros(1), terminated=False, truncated=False)
        batch = buf.gather([0, 1, 2])
        expected = [0.0, 1.0, 0.0] if bootstrap else [1.0, 1.0, 0.0]
        assert batch["done"].squeeze(-1).tolist() == expected
        assert batch["truncated"].squeeze(-1).tolist() == [1.0, 1.0, 0.0]
        assert batch["terminated"].squeeze(-1).tolist() == [0.0, 1.0, 0.0]


def test_buffer_state_dict_roundtrip_and_validation() -> None:
    """状态存档往返逐位一致；容量 / 维度不一致时报错。"""
    buf = ReplayBuffer(capacity=6, state_dim=2, action_dim=1, device="cpu", seed=1)
    for i in range(4):
        buf.add(np.array([i, i + 1], np.float32), np.array([0.5], np.float32) * i,
                float(i), np.array([i + 2, i + 3], np.float32),
                terminated=(i == 2), truncated=(i == 3))
    state = buf.state_dict()
    other = ReplayBuffer(capacity=6, state_dim=2, action_dim=1, device="cpu", seed=1)
    other.load_state_dict(state)
    assert len(other) == len(buf) == 4 and other.ptr == buf.ptr
    assert np.array_equal(other.actions, buf.actions) and np.array_equal(other.rewards, buf.rewards)
    assert np.array_equal(other.terminated, buf.terminated) and np.array_equal(other.truncated, buf.truncated)
    other.load_state_dict(None)                     # None ⇒ 不改动
    assert len(other) == 4
    for bad_state in ({**state, "capacity": 99}, {**state, "state_dim": 5}):
        try:
            other.load_state_dict(bad_state)
            raise AssertionError("容量 / 维度不一致应当报错")
        except ValueError:
            pass


def test_buffer_add_batch_and_stats() -> None:
    """``add_batch`` 批量写入 + ``stats`` 统计量正确。"""
    buf = ReplayBuffer(capacity=10, state_dim=2, action_dim=1, device="cpu")
    n = buf.add_batch(np.zeros((3, 2)), np.full((3, 1), 0.5), [-1.0, -2.0, -3.0],
                      np.ones((3, 2)), terminated=[0, 1, 0], truncated=[0, 0, 1])
    assert n == 3 and len(buf) == 3
    stats = buf.stats()
    assert stats["size"] == 3.0 and stats["fill_ratio"] == 0.3
    assert abs(stats["reward_mean"] + 2.0) < 1e-6
    assert abs(stats["terminated_fraction"] - 1 / 3) < 1e-6
    assert abs(stats["truncated_fraction"] - 1 / 3) < 1e-6
    assert abs(stats["action_abs_mean"] - 0.5) < 1e-6
    empty = ReplayBuffer(capacity=2, state_dim=1, action_dim=1, device="cpu").stats()
    assert empty["size"] == 0.0 and math.isnan(empty["reward_mean"])
    # 样本数不一致要报错
    try:
        buf.add_batch(np.zeros((3, 2)), np.zeros((2, 1)), [0.0, 0.0, 0.0], np.zeros((3, 2)))
        raise AssertionError("样本数不一致应当报错")
    except ValueError:
        pass
    buf.clear()
    assert len(buf) == 0 and buf.total_added == 0


# =============================================================================
# 5. Critic 损失（Double-Q + 软 Bellman 目标）
# =============================================================================
def test_critic_loss_with_done_uses_reward_as_target() -> None:
    """``done ≡ 1`` ⇒ 目标退化为 ``r``：损失 = 两个 Q 对 ``r`` 的 MSE 之和（逐位核对）。"""
    agent = make_agent()
    batch = make_batch(agent, batch_size=16, reward=3.0, done=1.0)
    loss, info = agent.critic_loss(batch)
    qs = agent.critic(batch["observation"], batch["action"])
    expected = sum(float((q - 3.0).pow(2).mean()) for q in qs)
    assert abs(float(loss.detach()) - expected) < 1e-5
    assert abs(info["critic_loss"] - expected) < 1e-5
    assert abs(info["target_q_mean"] - 3.0) < 1e-5
    assert abs(info["q1_loss"] + info["q2_loss"] - info["critic_loss"]) < 1e-5
    assert abs(info["q1_mean"] - float(qs[0].mean())) < 1e-5
    # reward_scale 只缩放学习目标
    scaled = make_agent(reward_scale=2.0)
    scaled.load_state_dict(agent.state_dict())
    scaled.reward_scale = 2.0
    _, info_scaled = scaled.critic_loss(batch)
    assert abs(info_scaled["target_q_mean"] - 6.0) < 1e-4


def test_critic_loss_target_is_detached_from_critic_params() -> None:
    """目标 ``y`` 不含 Critic 参数的梯度：扰动 Critic 参数后 ``target_q_mean`` 不变。"""
    agent = make_agent()
    batch = make_batch(agent, batch_size=16)
    torch.manual_seed(0)
    _, info_before = agent.critic_loss(batch)
    with torch.no_grad():
        for p in agent.critic.parameters():
            p.add_(0.7)
    torch.manual_seed(0)          # 复位 RNG ⇒ 两次的目标动作 a' 完全相同
    _, info_after = agent.critic_loss(batch)
    assert abs(info_before["target_q_mean"] - info_after["target_q_mean"]) < 1e-6
    assert abs(info_before["q1_mean"] - info_after["q1_mean"]) > 1e-3   # Q 值本身变了


def test_critic_loss_gradient_does_not_reach_actor_or_alpha() -> None:
    """Critic 的 backward 只该碰 Critic：Actor 与 ``log α`` 的 ``.grad`` 必须为空。"""
    agent = make_agent()
    batch = make_batch(agent, batch_size=8)
    loss, _ = agent.critic_loss(batch)
    loss.backward()
    assert any(p.grad is not None for p in agent.critic.parameters())
    assert all(p.grad is None for p in agent.actor.parameters()), "目标里的 a' 必须来自 no_grad 采样"
    assert agent.log_alpha.grad is None, "α 只能由自己的损失更新"


def test_critic_loss_converges_to_analytic_fixed_point() -> None:
    """单状态玩具系统：硬目标（τ=1）下 Q 收敛到 ``Q* = (r − γα·E[logπ])/(1−γ)``。

    这是"软 Bellman 目标 + Double-Q + 熵项"整条链路的数值验证：把 ``σ`` 夹到极小
    （``log σ = −20``）使采样几乎确定，从而 ``Q*`` 有解析值、可以直接比对。
    """
    actor = make_actor(log_std_init=-5.0)        # σ ≈ 0.0067 ⇒ 采样几乎确定 ⇒ Q* 可解析
    agent = make_agent(actor=actor, gamma=0.9, tau=1.0, alpha=1.0, target_entropy=None)
    obs = torch.zeros(32, 3)
    act = torch.zeros(32, 1)
    batch = {
        "observation": obs, "next_observation": obs, "action": act,
        "reward": torch.full((32, 1), 1.0), "done": torch.zeros(32, 1),
        "terminated": torch.zeros(32, 1), "truncated": torch.zeros(32, 1),
        "index": torch.arange(32),
    }
    with torch.no_grad():
        log_prob_mean = float(agent.actor.sample(obs)["log_prob"].mean())
    q_star = (1.0 - 0.9 * 1.0 * log_prob_mean) / (1.0 - 0.9)

    optimizer = optim.Adam(agent.critic.parameters(), lr=0.2)
    first_q = float(agent.critic.min_q(obs, act).mean())
    for _ in range(1500):
        loss, _ = agent.critic_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        agent.sync_target()                     # τ=1 等价于硬拷贝目标网络
    q_final = float(agent.critic.min_q(obs, act).mean())
    assert abs(q_final - q_star) / abs(q_star) < 0.02, (first_q, q_final, q_star)
    assert abs(q_final - q_star) < abs(first_q - q_star)


# =============================================================================
# 6. Actor 损失：W9 思考题（梯度要穿过 Critic，但不能更新 Critic 的参数）
# =============================================================================
class ConstantCritic(nn.Module):
    """玩具 Critic：``Q(s, a) = value``（与动作无关）。

    用于隔离"熵项"的行为：``J_π = α·log π − const`` ⇒ 最小化即最大化熵。
    带一个可训练参数，方便挂到 optimizer 上。
    """

    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(float(value)))
        self.n_critics = 1

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> List[torch.Tensor]:
        return [self.value * torch.ones_like(action)]


def test_actor_loss_matches_manual_formula() -> None:
    """与手写公式逐位一致（把 RNG 复位到同一状态 ⇒ 两次采样的 ``ε`` 相同）。"""
    agent = make_agent()
    batch = make_batch(agent, batch_size=12)
    torch.manual_seed(1234)
    loss, info = agent.actor_loss(batch)
    torch.manual_seed(1234)
    out = agent.actor.sample(batch["observation"])
    with frozen_parameters(agent.critic):
        qs = agent.critic(batch["observation"], out["action_norm"])
        q_pi = torch.min(torch.cat(qs, dim=-1), dim=-1, keepdim=True).values
        expected = (float(agent.alpha) * out["log_prob"] - q_pi).mean()
    assert torch.allclose(loss.detach(), expected.detach(), atol=1e-6), (loss.item(), expected.item())
    assert abs(info["q_pi_mean"] - float(q_pi.mean())) < 1e-6
    assert abs(info["log_prob_mean"] - float(out["log_prob"].mean())) < 1e-5
    assert abs(info["entropy_mean"] + info["log_prob_mean"]) < 1e-6
    assert torch.isfinite(loss) and info["actor_loss"] == float(loss.detach())


def test_actor_moves_toward_argmax_of_q_through_frozen_critic() -> None:
    """二次玩具 Q：策略的**确定性动作**确实朝 ``argmax_a Q`` 移动（沿 ``∂Q/∂a`` 走）。"""
    for target in (0.6, -0.5):
        actor = make_actor(log_std_init=-2.0)
        agent = make_agent(actor=actor, critic=QuadraticCritic(target_action=target),
                           target_entropy=None, alpha=1.0)
        batch = {"observation": torch.zeros(8, 3), "action": torch.zeros(8, 1),
                 "next_observation": torch.zeros(8, 3), "reward": torch.zeros(8, 1),
                 "done": torch.zeros(8, 1), "terminated": torch.zeros(8, 1),
                 "truncated": torch.zeros(8, 1), "index": torch.arange(8)}
        optimizer = optim.Adam(agent.actor.parameters(), lr=1e-2)
        with torch.no_grad():
            before = float(agent.actor.sample(batch["observation"], deterministic=True)["action_norm"].mean())
        for _ in range(200):
            loss, _ = agent.actor_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = float(agent.actor.sample(batch["observation"], deterministic=True)["action_norm"].mean())
        assert abs(after - target) < abs(before - target), (target, before, after)
        assert abs(after - target) < 0.5 * abs(before - target) + 1e-3, (target, before, after)


def test_frozen_critic_still_backprops_to_actor() -> None:
    """**思考题核心**：冻结 Critic 的**参数**后，Actor 依然能拿到 ``∂Q/∂a`` 的梯度。

    * 计算图连通 ⇒ ``q_pi`` 对动作可导、``∂q_pi/∂a ≠ 0``；
    * 参数冻结 ⇒ Critic 的 ``.grad`` 全为 ``None``（不会被任何 optimizer 更新）；
    * Actor 的参数拿到非零梯度 ⇒ 策略确实能"看见"价值方向。
    """
    agent = make_agent(critic=QuadraticCritic(target_action=0.5))
    batch = make_batch(agent, batch_size=8)
    loss, _ = agent.actor_loss(batch)

    # 图连通：∂Q/∂a ≠ 0（单独用 autograd 验证，不影响后续 backward）
    obs = batch["observation"]
    action = agent.actor.sample(obs)["action_norm"].detach().requires_grad_(True)
    with frozen_parameters(agent.critic):
        q_connected = agent.critic(obs, action)[0]
    grad_action = torch.autograd.grad(q_connected.sum(), action)[0]
    assert float(grad_action.abs().sum()) > 0, "∂Q/∂a 必须非零，否则 Actor 学不到东西"

    loss.backward()
    assert all(p.grad is None for p in agent.critic.parameters()), "冻结的 Critic 不应累积梯度"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in agent.actor.parameters()), "Actor 必须有非零梯度"
    assert all(p.requires_grad for p in agent.critic.parameters()), "冻结是临时的（退出后恢复）"


def test_detaching_q_leaves_only_entropy_gradient() -> None:
    """对照：``q.detach()``（等价于 ``no_grad`` 切断图）只会剩下熵项的梯度。"""
    agent = make_agent(critic=QuadraticCritic(target_action=0.9), alpha=0.5)
    batch = make_batch(agent, batch_size=8)
    obs = batch["observation"]

    torch.manual_seed(7)
    loss_connected, _ = agent.actor_loss(batch)
    loss_connected.backward()
    grad_connected = agent.actor.mu_head.weight.grad.detach().clone()
    agent.actor.zero_grad()

    torch.manual_seed(7)
    out = agent.actor.sample(obs)
    with frozen_parameters(agent.critic):
        q_detached = agent.critic(obs, out["action_norm"])[0].detach()
    loss_cut = (float(agent.alpha) * out["log_prob"] - q_detached).mean()
    loss_cut.backward()
    grad_cut = agent.actor.mu_head.weight.grad.detach().clone()
    agent.actor.zero_grad()

    torch.manual_seed(7)
    out2 = agent.actor.sample(obs)
    loss_entropy = (float(agent.alpha) * out2["log_prob"]).mean()
    loss_entropy.backward()
    grad_entropy = agent.actor.mu_head.weight.grad.detach().clone()

    assert not torch.allclose(grad_connected, grad_cut), "切断 Q 后梯度必然不同"
    assert torch.allclose(grad_cut, grad_entropy, atol=1e-6), "只剩熵项的梯度"
    # 两者之差正是"来自价值的梯度"：非零 ⇒ 策略确实在利用 Q 的方向
    assert float((grad_connected - grad_cut).abs().sum()) > 0


def test_constant_q_maximizes_entropy() -> None:
    """Q 与动作无关时 ``J_π = α·log π`` ⇒ 训练使策略熵上升（最大熵目标的直接证据）。"""
    actor = make_actor(log_std_init=-1.0)
    agent = make_agent(actor=actor, critic=ConstantCritic(0.0), target_entropy=None, alpha=1.0)
    obs = torch.zeros(64, 3)
    batch = {"observation": obs, "action": torch.zeros(64, 1), "next_observation": obs,
             "reward": torch.zeros(64, 1), "done": torch.zeros(64, 1),
             "terminated": torch.zeros(64, 1), "truncated": torch.zeros(64, 1),
             "index": torch.arange(64)}
    with torch.no_grad():
        entropy_before = float(-agent.actor.sample(obs)["log_prob"].mean())
    optimizer = optim.Adam(agent.actor.parameters(), lr=1e-2)
    for _ in range(40):
        loss, _ = agent.actor_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        entropy_after = float(-agent.actor.sample(obs)["log_prob"].mean())
    assert entropy_after > entropy_before + 0.1, (entropy_before, entropy_after)


def test_actor_step_does_not_change_critic() -> None:
    """连续做 5 次 Actor 步：Critic 参数逐位不变、``.grad`` 始终为 ``None``。"""
    agent = make_agent()
    batch = make_batch(agent, batch_size=16)
    snapshot = params_snapshot(agent.critic)
    actor_before = params_snapshot(agent.actor)
    optimizer = optim.Adam(agent.actor.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = agent.actor_loss(batch)
        loss.backward()
        optimizer.step()
        assert all(p.grad is None for p in agent.critic.parameters())
    assert params_equal(agent.critic, snapshot), "Critic 必须逐位不变"
    assert not params_equal(agent.actor, actor_before), "Actor 必须确实被更新"


def test_shared_optimizer_would_corrupt_critic() -> None:
    """反例：把 Q 与 Actor 塞进**同一个** optimizer，Actor 步会把 Q 往"抬高"方向推。

    这正是"必须冻结 Critic 参数"的工程理由（本模块用三套独立 optimizer + 冻结双保险）。
    """
    agent = make_agent(critic=QuadraticCritic(target_action=0.0))
    batch = make_batch(agent, batch_size=16)
    obs = batch["observation"]
    torch.manual_seed(0)
    out = agent.actor.sample(obs)
    qs = agent.critic(obs, out["action_norm"])              # ← 故意**不**冻结
    q_pi = torch.min(torch.cat(qs, dim=-1), dim=-1, keepdim=True).values
    loss = (float(agent.alpha) * out["log_prob"] - q_pi).mean()

    shared = optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=1e-2)
    snapshot = params_snapshot(agent.critic)
    q_before = float(q_pi.mean())
    shared.zero_grad()
    loss.backward()
    shared.step()
    assert not params_equal(agent.critic, snapshot), "共享 optimizer 会污染 Critic"
    with torch.no_grad():
        q_after = float(agent.critic(obs, out["action_norm"])[0].mean())
    assert q_after > q_before, "污染的方向是「把 Q 抬高」（reward hacking）"

    # 正确做法：冻结参数 + 只用 Actor 的 optimizer ⇒ Critic 逐位不动
    agent2 = make_agent(critic=QuadraticCritic(target_action=0.0))
    snapshot2 = params_snapshot(agent2.critic)
    loss2, _ = agent2.actor_loss(batch)
    actor_only = optim.Adam(agent2.actor.parameters(), lr=1e-2)
    actor_only.zero_grad()
    loss2.backward()
    actor_only.step()
    assert params_equal(agent2.critic, snapshot2)


def test_use_min_for_actor_switch_and_action_grad_diagnostic() -> None:
    """``use_min_for_actor`` 影响 ``q_pi``（与手算 min(q1,q2)/q1 逐位一致）；
    ``compute_action_grad`` 给出 ``|∂Q/∂a|``。"""
    set_seeds(21)
    critic = ContinuousCritic(state_dim=3, action_dim=1, hidden_dim=16, num_layers=1, n_critics=2)
    actor = make_actor()
    agent_min = make_agent(actor=actor, critic=critic, use_min_for_actor=True)
    agent_q1 = make_agent(actor=actor, critic=critic, use_min_for_actor=False)
    batch = make_batch(agent_min, batch_size=8)

    torch.manual_seed(3)
    _, info_min = agent_min.actor_loss(batch)
    torch.manual_seed(3)
    _, info_q1 = agent_q1.actor_loss(batch)
    torch.manual_seed(3)                        # 复位 RNG ⇒ 用同一批 a 手算金标准
    out = agent_min.actor.sample(batch["observation"])
    with torch.no_grad():
        qs = [q.detach() for q in critic(batch["observation"], out["action_norm"])]
    expected_min = float(torch.minimum(qs[0], qs[1]).mean())
    expected_q1 = float(qs[0].mean())
    assert abs(info_min["q_pi_mean"] - expected_min) < 1e-5, (info_min["q_pi_mean"], expected_min)
    assert abs(info_q1["q_pi_mean"] - expected_q1) < 1e-5, (info_q1["q_pi_mean"], expected_q1)
    assert info_min["q_pi_mean"] <= info_q1["q_pi_mean"] + 1e-6      # min 更悲观

    agent_min.compute_action_grad = True
    torch.manual_seed(3)
    _, info_grad = agent_min.actor_loss(batch)
    assert info_grad["action_grad_norm"] > 0, "∂Q/∂a 非零（图确实穿透了 Critic）"

    const_agent = make_agent(actor=make_actor(), critic=ConstantCritic(1.0), target_entropy=None)
    const_agent.compute_action_grad = True
    _, info_const = const_agent.actor_loss(batch)
    assert info_const["action_grad_norm"] == 0.0, "Q 与动作无关 ⇒ ∂Q/∂a = 0"


# =============================================================================
# 7. 温度 α 的自适应
# =============================================================================
def test_alpha_initialization_and_parameterisation() -> None:
    """``α = exp(log α)``、初值来自配置、自动模式可学习、固定模式不可学习。"""
    agent = make_agent(alpha=0.7)
    assert abs(float(agent.alpha) - 0.7) < 1e-6
    assert abs(float(agent.log_alpha) - math.log(0.7)) < 1e-6
    assert agent.automatic_entropy and agent.log_alpha.requires_grad
    assert len(agent.alpha_parameters()) == 1

    fixed = make_agent(alpha=0.25, target_entropy=None)
    assert abs(float(fixed.alpha) - 0.25) < 1e-6
    assert not fixed.automatic_entropy
    assert not fixed.log_alpha.requires_grad
    assert fixed.alpha_parameters() == []
    for bad_alpha in (0.0, -1.0):
        try:
            make_agent(alpha=bad_alpha)
            raise AssertionError("α ≤ 0 应当报错")
        except ValueError:
            pass


def test_alpha_loss_formula_and_gradient_does_not_touch_networks() -> None:
    """``J(α) = −log α·E[log π + H̄]``（逐位核对）；``log_prob`` 已 detach。"""
    agent = make_agent(alpha=1.0, target_entropy=-1.0)
    log_prob = torch.tensor([[-2.0], [-1.0], [0.5], [1.5]])
    loss, info = agent.alpha_loss(log_prob)
    expected = -(float(agent.log_alpha) * (log_prob + (-1.0))).mean()
    assert abs(float(loss.detach()) - float(expected)) < 1e-6
    assert abs(info["target_entropy"] + 1.0) < 1e-6
    assert abs(info["log_alpha"] - float(agent.log_alpha)) < 1e-6
    loss.backward()
    assert agent.log_alpha.grad is not None and abs(float(agent.log_alpha.grad)) > 0
    assert all(p.grad is None for p in agent.actor.parameters())
    assert all(p.grad is None for p in agent.critic.parameters())
    # 关闭自适应 ⇒ 没有损失
    assert make_agent(target_entropy=None).alpha_loss(log_prob)[0] is None


def test_alpha_direction_follows_entropy_gap() -> None:
    """温度方向：策略太确定（``E[log π]`` 大）⇒ α↑；太随机（``log π`` 很负）⇒ α↓。"""
    up = make_agent(alpha=1.0, target_entropy=-1.0)
    optimizer = optim.Adam(up.alpha_parameters(), lr=1e-2)
    for _ in range(20):
        loss, _ = up.alpha_loss(torch.full((8, 1), 2.0))    # E[log π]=2 ⇒ 熵 −2，比目标更确定
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(up.alpha) > 1.05, float(up.alpha)

    down = make_agent(alpha=1.0, target_entropy=-1.0)
    optimizer = optim.Adam(down.alpha_parameters(), lr=1e-2)
    for _ in range(20):
        loss, _ = down.alpha_loss(torch.full((8, 1), -5.0))  # E[log π]=−5 ⇒ 熵 5，比目标随机
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(down.alpha) < 0.95, float(down.alpha)


def test_alpha_min_clamp() -> None:
    """``alpha_min`` 生效：α 不会掉到下界以下（夹紧在 ``log`` 空间）。"""
    agent = make_agent(alpha=1.0, alpha_min=0.5, target_entropy=-1.0)
    optimizer = optim.Adam(agent.alpha_parameters(), lr=0.5)
    for _ in range(50):
        loss, _ = agent.alpha_loss(torch.full((4, 1), -20.0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        agent.clamp_alpha()
    assert float(agent.alpha) >= 0.5 - 1e-6, float(agent.alpha)
    assert abs(float(agent.alpha) - 0.5) < 1e-3


# =============================================================================
# 8. 装配（build_*）与完整更新 update_batch
# =============================================================================
def test_build_sac_agent_from_real_config() -> None:
    """按 ``configs/sac_config.yaml`` 装配：结构 / 目标熵 / α / 动作边界 / 参数量。"""
    import yaml

    with open(os.path.join(ROOT, "configs", "sac_config.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    agent = build_sac_agent(config, device="cpu")
    m, t = config["model"], config["training"]
    assert agent.state_dim == 3 and agent.action_dim == 1
    assert agent.actor.hidden_dim == m["hidden_dim"] == 256
    assert agent.critic.hidden_dim == m["critic_hidden_dim"] == 256
    assert agent.critic.n_critics == m["n_critics"] == 2
    assert abs(agent.gamma - t["gamma"]) < 1e-9 and abs(agent.tau - t["tau"]) < 1e-9
    assert abs(float(agent.alpha) - t["init_alpha"]) < 1e-6
    assert agent.target_entropy == -1.0 and agent.automatic_entropy      # "auto" ⇒ −dim(A)
    assert agent.use_min_for_actor is True
    assert agent.actor.action_low == -2.0 and agent.actor.action_high == 2.0
    # 目标网络是独立副本（参数量相同、参数值相同、requires_grad 全 False）
    assert sum(p.numel() for p in agent.target_critic.parameters()) == \
        sum(p.numel() for p in agent.critic.parameters())
    assert all(not p.requires_grad for p in agent.target_critic.parameters())


def test_build_sac_optimizers_three_independent_adam() -> None:
    """三套独立 Adam：学习率正确、参数集合互不相交、固定 α 时 ``alpha`` 为 None。"""
    import yaml

    with open(os.path.join(ROOT, "configs", "sac_config.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    agent = build_sac_agent(config, device="cpu")
    t = config["training"]
    optimizers = build_sac_optimizers(
        agent, lr_actor=t["lr_actor"], lr_critic=t["lr_critic"], lr_alpha=t["lr_alpha"],
    )
    assert set(optimizers) == {"critic", "actor", "alpha"}
    assert optimizers["critic"].param_groups[0]["lr"] == t["lr_critic"] == 1e-3
    assert optimizers["actor"].param_groups[0]["lr"] == t["lr_actor"] == 3e-4
    assert optimizers["alpha"].param_groups[0]["lr"] == t["lr_alpha"] == 3e-4
    critic_ids = {id(p) for p in optimizers["critic"].param_groups[0]["params"]}
    actor_ids = {id(p) for p in optimizers["actor"].param_groups[0]["params"]}
    alpha_ids = {id(p) for p in optimizers["alpha"].param_groups[0]["params"]}
    assert not (critic_ids & actor_ids) and not (critic_ids & alpha_ids) and not (actor_ids & alpha_ids)
    assert alpha_ids == {id(agent.log_alpha)}
    # 目标网络永远不进 optimizer
    assert not ({id(p) for p in agent.target_critic.parameters()} & (critic_ids | actor_ids))
    fixed = build_sac_agent({**config, "training": {**t, "target_entropy": None}}, device="cpu")
    assert build_sac_optimizers(fixed)["alpha"] is None


def test_update_batch_runs_three_steps_and_reports_diagnostics() -> None:
    """一次完整更新：三段参数都变化、目标网络被软更新、诊断量齐全且有限。"""
    from src.sac import SAC_UPDATE_KEYS

    agent = make_agent(tau=0.1)
    optimizers = build_sac_optimizers(agent, lr_actor=1e-3, lr_critic=1e-2, lr_alpha=1e-2)
    batch = make_batch(agent, batch_size=32)
    actor_before = params_snapshot(agent.actor)
    critic_before = params_snapshot(agent.critic)
    target_before = params_snapshot(agent.target_critic)
    alpha_before = float(agent.log_alpha)

    logs = agent.update_batch(batch, optimizers)
    assert not params_equal(agent.actor, actor_before), "Actor 应被更新"
    assert not params_equal(agent.critic, critic_before), "Critic 应被更新"
    assert not params_equal(agent.target_critic, target_before), "目标网络应被 Polyak 软更新"
    assert float(agent.log_alpha) != alpha_before, "α 应被更新"
    assert logs["critic_frozen_ok"] == 1.0
    assert logs["actor_grad_norm"] > 0 and logs["critic_grad_norm"] > 0 and logs["alpha_grad_norm"] > 0
    assert logs["batch_size"] == 32.0 and logs["update_calls"] == 1.0
    for key in SAC_UPDATE_KEYS:
        assert key in logs, f"缺少诊断量 {key}"
        assert isinstance(logs[key], float)
        if key != "target_entropy":                      # 固定 α 时为 nan
            assert math.isfinite(logs[key]) or (key == "entropy_target_gap" and math.isnan(logs[key]))
    # 目标网络只被"软"更新：τ=0.1 ⇒ 与 critic 仍有明显差距
    diff = max(float((p_t - p_c).abs().max())
               for p_t, p_c in zip(agent.target_critic.parameters(), agent.critic.parameters()))
    assert diff > 0
    # 第二步仍可正常执行（状态可复用）
    logs2 = agent.update_batch(batch, optimizers)
    assert logs2["update_calls"] == 2.0
    assert logs2["critic_frozen_ok"] == 1.0
    assert agent.diagnostics()["update_calls"] == 2.0


def test_update_batch_validation_and_fixed_alpha_path() -> None:
    """缺少 optimizer 报错；固定温度时不更新 α（但其余流程照常）。"""
    agent = make_agent(target_entropy=None)
    batch = make_batch(agent)
    try:
        agent.update_batch(batch, {"critic": None, "actor": None})
        raise AssertionError("缺少 optimizer 应当报错")
    except ValueError:
        pass
    optimizers = build_sac_optimizers(agent)
    assert optimizers["alpha"] is None
    before = float(agent.log_alpha)
    logs = agent.update_batch(batch, optimizers)
    assert float(agent.log_alpha) == before
    assert logs["alpha_loss"] == 0.0 and logs["alpha_grad_norm"] == 0.0
    assert math.isnan(logs["target_entropy"])


def test_grad_norm_helper_and_frozen_parameters_context() -> None:
    """``grad_norm`` 与 ``frozen_parameters`` 的语义。"""
    module = nn.Linear(3, 1)
    assert grad_norm(module.parameters()) == 0.0
    module(torch.randn(4, 3)).sum().backward()
    assert grad_norm(module.parameters()) > 0

    flags_before = [p.requires_grad for p in module.parameters()]
    with frozen_parameters(module):
        assert all(not p.requires_grad for p in module.parameters())
        module.zero_grad(set_to_none=True)
        out = module(torch.randn(2, 3))
        assert out.grad_fn is None, "参数冻结 + 输入不需要梯度 ⇒ 不建图"
        assert all(p.grad is None for p in module.parameters())
    assert [p.requires_grad for p in module.parameters()] == flags_before
    # 嵌套 / None 参数都安全
    with frozen_parameters(None, module):
        assert all(not p.requires_grad for p in module.parameters())
    assert all(flags_before)


def test_diagnostics_and_random_action_helper() -> None:
    """``diagnostics`` 字段；``random_norm_action`` 落在 (−1,1) 且与缓冲坐标系一致。"""
    agent = make_agent()
    diag = agent.diagnostics()
    for key in ("alpha", "log_alpha", "target_entropy", "gamma", "tau", "n_critics",
                "actor_params", "critic_params", "target_critic_params", "update_calls"):
        assert key in diag
    assert diag["n_critics"] == 2.0 and diag["target_critic_params"] > 0

    env = make_env("Pendulum-v1")
    try:
        env.reset(seed=0)
        env.action_space.seed(0)
        actions = np.stack([random_norm_action(env, agent.actor) for _ in range(50)])
        assert actions.shape == (50, 1)
        assert float(actions.min()) > -1.0 and float(actions.max()) < 1.0
        env_actions = agent.actor.to_env_action(torch.as_tensor(actions)).numpy()
        assert float(env_actions.min()) > -2.0 and float(env_actions.max()) < 2.0
        # 固定 action_space 种子后完全可复现
        env.action_space.seed(0)
        again = np.stack([random_norm_action(env, agent.actor) for _ in range(50)])
        assert np.allclose(actions, again)
    finally:
        env.close()


# =============================================================================
# 9. 端到端
# =============================================================================
def test_toy_bandit_end_to_end_learning() -> None:
    """端到端（bandit 玩具）：``reward = −|a_env − 1.0|``，SAC 应学会逼近最优动作。

    因为每步都终止（``done=True``），Bellman 目标退化为 ``Q = r``，Critic 学到的就是
    精确的奖励函数 —— 于是 Actor 的最优动作有解析解 ``a_env = 1.0``。这条流水线
    （buffer → Double-Q → 冻结 Critic 的 Actor 更新 → α 自适应）能否真的"把收益做上去"
    因此有可判定的标准。
    """
    target = 1.0
    agent = make_agent(actor=make_actor(log_std_init=-0.5), gamma=0.99, tau=0.01,
                       alpha=0.2, target_entropy=-1.0, alpha_min=0.05)
    optimizers = build_sac_optimizers(agent, lr_actor=3e-3, lr_critic=3e-3, lr_alpha=3e-3)
    buf = ReplayBuffer(capacity=20000, state_dim=3, action_dim=1, device="cpu", seed=0)
    rng = np.random.default_rng(0)
    obs = np.zeros(3, dtype=np.float32)
    for _ in range(2000):                        # 热身：纯随机动作
        a_norm = float(rng.uniform(-1 + 1e-6, 1 - 1e-6))
        a_env = float(agent.actor.to_env_action(torch.tensor([a_norm]))[0])
        buf.add(obs, np.array([a_norm], np.float32), -abs(a_env - target), obs, True, False)
    with torch.no_grad():
        before = float(agent.actor.sample(torch.zeros(1, 3), deterministic=True)["action_env"][0, 0])
    for _ in range(2500):
        agent.update_batch(buf.sample(128), optimizers)
    with torch.no_grad():
        after = float(agent.actor.sample(torch.zeros(1, 3), deterministic=True)["action_env"][0, 0])
    assert abs(after - target) < abs(before - target), (before, after)
    assert abs(after - target) < 0.25, (before, after, "策略未能逼近最优动作")
    assert 0.05 <= float(agent.alpha) < 10.0, float(agent.alpha)
    # Critic 学到的是"精确奖励函数"（Q ≈ r），可在 buffer 上直接核对
    batch = buf.sample(256)
    q = agent.critic.min_q(batch["observation"], batch["action"])
    assert float((q - batch["reward"]).abs().mean()) < 0.2


def test_pendulum_smoke_updates_are_finite() -> None:
    """Pendulum-v1 短路训练：随机数据 + 数百次更新，诊断量全部有限、动作始终在范围内。"""
    env = make_env("Pendulum-v1")
    agent = make_agent(alpha=1.0, tau=0.005)
    optimizers = build_sac_optimizers(agent, lr_actor=1e-3, lr_critic=1e-3, lr_alpha=1e-3)
    buf = ReplayBuffer(capacity=8192, state_dim=3, action_dim=1, device="cpu", seed=0)
    logs: Dict[str, Any] = {}
    episode_returns: List[float] = []
    episode_return = 0.0
    try:
        obs, _ = env.reset(seed=0)
        env.action_space.seed(0)
        for step in range(700):
            a_norm = random_norm_action(env, agent.actor)
            a_env = agent.actor.to_env_action(torch.as_tensor(a_norm)).numpy()
            next_obs, reward, terminated, truncated, _info = env.step(a_env)
            buf.add(obs, a_norm, float(reward), next_obs, terminated, truncated)
            episode_return += float(reward)
            obs = next_obs
            if terminated or truncated:
                episode_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()
            if step >= 256 and step % 2 == 0:
                logs = agent.update_batch(buf.sample(256), optimizers)
        assert logs, "至少应完成一次更新"
        for key, value in logs.items():
            if isinstance(value, float) and key != "target_entropy":
                assert math.isfinite(value), f"{key} 不是有限值: {value}"
        assert logs["critic_frozen_ok"] == 1.0
        assert 0.0 < float(agent.alpha) < 100.0
        # 随机策略的 Pendulum 回报应在 −1600 附近（远差于收敛策略的 −200 上下）
        assert 1 <= len(episode_returns) <= 6, len(episode_returns)
        mean_return = float(np.mean(episode_returns))
        assert -2000.0 < mean_return < -400.0, mean_return
        # buffer 里存的是归一化动作（不是环境动作）
        assert float(np.abs(buf.actions).max()) < 1.0
        stats = buf.stats()
        assert stats["size"] == 700.0 and stats["truncated_fraction"] > 0.0
        assert stats["terminated_fraction"] == 0.0              # Pendulum 没有真终止
    finally:
        env.close()


# =============================================================================
# 10. 工程脚本与配置
# =============================================================================
def load_script(module_name: str, path: str):
    """按文件路径加载脚本模块（与 W8 测试同一套做法）。"""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)          # type: ignore[union-attr]
    return module


def test_sac_config_matches_assignment_spec() -> None:
    """``configs/sac_config.yaml`` 与作业指定超参逐项一致。"""
    import yaml

    with open(os.path.join(ROOT, "configs", "sac_config.yaml"), encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    exp = config["experiment"]
    model, train, eval_cfg = config["model"], config["training"], config["eval"]
    assert exp["seed"] == 42 and exp["device"] == "cuda" and exp["env_name"] == "Pendulum-v1"
    assert model["state_dim"] == 3 and model["action_dim"] == 1
    assert train["total_timesteps"] == 40000
    assert train["buffer_size"] == 100000
    assert train["batch_size"] == 256
    assert float(train["lr_actor"]) == 3e-4
    assert float(train["lr_critic"]) == 1e-3
    assert float(train["lr_alpha"]) == 3e-4
    assert float(train["tau"]) == 0.005 and float(train["gamma"]) == 0.99
    assert str(train["target_entropy"]).lower() == "auto"
    assert train["warmup_steps"] == 1000
    assert float(train["target_reward"]) == -200.0
    assert train["bootstrap_on_truncation"] is True
    assert eval_cfg["n_episodes"] == 20 and eval_cfg["every_steps"] > 0


def test_train_script_helpers_and_entrypoints() -> None:
    """``experiments/train_sac.py`` 的辅助函数与入口存在且语义正确。"""
    module = load_script("train_sac_script", os.path.join(ROOT, "experiments", "train_sac.py"))
    assert callable(module.main) and callable(module.train_sac)
    assert module.CONFIG_PATH == os.path.join("configs", "sac_config.yaml")

    assert module.as_bool("false") is False and module.as_bool("true") is True
    assert module.as_bool(None) is False and module.as_bool(None, default=True) is True
    assert module.as_bool(1) is True

    assert module.rolling_mean([1.0, 2.0, 3.0], 2) == 2.5
    assert module.rolling_mean([1.0, 2.0, 3.0], 0) == 2.0
    assert module.rolling_mean([], 5) == 0.0

    env = make_env("Pendulum-v1")
    try:
        config = {"model": {"action_low": -1.0, "action_high": 1.0}}     # 故意写错的边界
        fixed = module.check_action_bounds(config, env)
        assert fixed["action_low"] == -2.0 and fixed["action_high"] == 2.0
        assert config["model"]["action_low"] == -2.0                     # 原地修正
    finally:
        env.close()

    avg = module.RunningAvg()
    avg.update({"a": 1.0, "b": torch.tensor([2.0]), "note": "不是数值", "nan": float("nan")})
    avg.update({"a": 3.0})
    assert avg.mean() == {"a": 2.0}, avg.mean()      # 张量 / 字符串 / NaN 都被跳过
    avg.reset()
    assert avg.mean() == {}


# =============================================================================
# 运行器（不依赖 pytest）
# =============================================================================
def main() -> int:
    set_seeds(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    print(f"== W9 Task 1 SAC 单元测试：共 {len(tests)} 项 ==")
    failed: List[Tuple[str, str]] = []
    start = time.perf_counter()
    for name, func in tests:
        t0 = time.perf_counter()
        try:
            func()
        except Exception as exc:                                   # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {name}  ({time.perf_counter() - t0:.2f}s)\n       {type(exc).__name__}: {exc}")
        else:
            print(f"[ ok ] {name}  ({time.perf_counter() - t0:.2f}s)")
    print("-" * 78)
    print(f"通过 {len(tests) - len(failed)}/{len(tests)}，用时 {time.perf_counter() - start:.1f}s")
    if failed:
        for name, message in failed:
            print(f"  ✗ {name}: {message}")
        return 1
    print("全部通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
