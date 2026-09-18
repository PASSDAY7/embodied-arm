#!/usr/bin/env python
"""W6 Task 1（REINFORCE / 策略梯度 + Baseline）单元测试。

覆盖范围
--------
1. **回报与优势**：反向递推 ``G_t = r_t + γ·G_{t+1}`` 与朴素 ``O(T²)`` 双重循环逐位一致；
   ``γ=0`` 退化为即时奖励；Return 归一化后零均值、单位标准差；三种基线
   （``none`` / ``batch_mean`` / ``value``）的形态与"基线只减方差不改期望"的机理。
2. **策略网络**：输出 logits 形状、``Categorical`` 概率归一、``log_prob`` 与手写
   ``log_softmax`` 逐点一致、熵落在 ``[0, log A]``、贪心 ``argmax`` 与采样行为、
   采样结果 ``detach``。
3. **损失与负号**：``L = −Σ logπ·Â``（``mean`` / ``sum``）、熵项 ``−β·ΣH`` 的符号、
   优势作为常数权重的 ``detach`` 语义，以及 **思考题 1** 的显式实验
   （``test_minus_sign_is_required_for_ascent``：去掉负号会把"抬高好动作概率"
   变成"抬高坏动作概率"）。
4. **回合采样**：缓存 ``(s, a, r, logπ)`` 与网络输出一致、``terminated`` / ``truncated``
   / ``max_steps`` 三种终止路径、``stack_episodes`` 的批拼接与 ``episode_index``。
5. **一次策略改进**：参数确实被更新、诊断量自洽（``Â`` 零均值单位标准差、
   ``log_prob_drift ≈ 0``）、重算 logπ 与缓存值一致
   （``test_recomputed_log_probs_match_cached``）、联合优化器与两个独立优化器等价
   （``test_joint_optimizer_matches_two_separate_optimizers``）、梯度裁剪、
   Return 归一化只改梯度尺度不改方向（**思考题 2** 的定量证据）。
6. **端到端**：``evaluate_policy`` 可复现且回报 == 存活步数；REINFORCE 在 CartPole-v1
   上 120 个回合内显著优于随机策略（固定种子）。

运行方式（无需 pytest）::

    python tests/test_reinforce.py

若已安装 pytest，也可直接 ``pytest tests/test_reinforce.py``。
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reinforce import (  # noqa: E402
    PolicyNetwork,
    ValueNetwork,
    compute_advantages,
    compute_returns,
    evaluate_policy,
    make_env,
    reinforce_loss,
    run_episode,
    stack_episodes,
    update_policy,
    value_loss,
)

DEVICE = torch.device("cpu")          # 测试统一用 CPU，保证任何机器可复现
LOG2 = float(np.log(2.0))             # 二动作离散策略的熵上界


# =============================================================================
# 0. 测试辅助
# =============================================================================
def _make_episode(policy: PolicyNetwork, n_steps: int = 12, seed: int = 0,
                  gamma: float = 0.99, state_dim: int = 4, action_dim: int = 2) -> Dict[str, Any]:
    """构造一个"和 :func:`run_episode` 输出同构"的离线轨迹（不依赖环境，快且确定）。

    关键点：``log_probs`` / ``entropies`` 由**当前策略**前向得到并 ``detach``，
    与 ``run_episode`` 缓存的语义完全一致（供一致性校验测试使用）。
    """
    gen = torch.Generator().manual_seed(seed)
    states = torch.randn(n_steps, state_dim, generator=gen)
    actions = torch.randint(0, action_dim, (n_steps,), generator=gen)
    rewards = torch.randint(0, 2, (n_steps,), generator=gen).float()      # CartPole 风格 0/1 奖励
    with torch.no_grad():
        dist = policy.distribution(states)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
    ret = compute_returns(rewards, gamma=gamma, normalize=False)
    return {
        "states": states,
        "actions": actions,
        "log_probs": log_probs.detach(),
        "entropies": entropies.detach(),
        "rewards": rewards,
        "returns": ret["raw_returns"],
        "raw_returns": ret["raw_returns"],
        "raw_mean": ret["raw_mean"],
        "raw_std": ret["raw_std"],
        "episode_return": float(rewards.sum()),
        "discounted_return": float(ret["raw_returns"][0]),
        "length": n_steps,
        "terminated": True,
        "truncated": False,
        "gamma": gamma,
    }


class _DummyEnv:
    """极简确定性环境（reward 恒为 1），用于精确检验终止路径分支。"""

    def __init__(self, horizon: int = 1000, terminate: bool = True, state_dim: int = 4) -> None:
        self.horizon = int(horizon)
        self.terminate = bool(terminate)
        self.state_dim = int(state_dim)
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(self.state_dim, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        done = self.t >= self.horizon
        return (
            np.zeros(self.state_dim, dtype=np.float32),
            1.0,
            bool(done and self.terminate),                 # terminated：任务失败/成功
            bool(done and not self.terminate),             # truncated：时间上限
            {},
        )

    def close(self):
        pass


def _flat_grads(module: torch.nn.Module) -> torch.Tensor:
    """把模块所有参数的梯度拼成一维向量（用于逐位比较不同更新路径）。"""
    parts = [p.grad.detach().reshape(-1) for p in module.parameters() if p.grad is not None]
    return torch.cat(parts)


def _update_grad_vector(policy: PolicyNetwork, episodes, **kwargs) -> torch.Tensor:
    """用 ``lr=0`` 的 SGD 跑一次 ``update_policy``：参数不变、梯度留在 ``.grad`` 上。"""
    net = copy.deepcopy(policy)
    opt = optim.SGD(net.parameters(), lr=0.0)          # step() 不改变参数，只借用其 zero_grad
    update_policy(net, episodes, optimizer=opt, **kwargs)
    return _flat_grads(net)


def _expect_value_error(fn, *args, **kwargs):
    """断言调用抛出 ``ValueError``（保持"无需 pytest"的运行方式）。"""
    try:
        fn(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"{fn.__name__} 应当抛出 ValueError")


# =============================================================================
# 1. 回报与优势
# =============================================================================
def test_returns_match_naive_backward_recursion():
    """反向递推 ``G_t = r_t + γ·G_{t+1}`` 与朴素双重循环 ``Σ_k γ^k r_{t+k}`` 逐位一致。"""
    rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    gamma = 0.9
    naive = [sum(rewards[t + k] * gamma ** k for k in range(len(rewards) - t))
             for t in range(len(rewards))]
    raw = compute_returns(rewards, gamma=gamma, normalize=False)["raw_returns"]
    assert np.allclose(raw.numpy(), np.asarray(naive, dtype=np.float32), atol=1e-5)
    # 末步回报 = 最后一步即时奖励（递推起点）
    assert abs(float(raw[-1]) - rewards[-1]) < 1e-6
    # 递推关系逐点自洽：G_t == r_t + γ·G_{t+1}
    for t in range(len(rewards) - 1):
        assert abs(float(raw[t]) - (rewards[t] + gamma * float(raw[t + 1]))) < 1e-5
    # γ < 1 时回报有界：|G_t| ≤ R_max / (1 − γ)
    r_max = max(abs(r) for r in rewards)
    assert float(raw.abs().max()) <= r_max / (1.0 - gamma) + 1e-5


def test_returns_with_gamma_zero_equals_immediate_rewards():
    """``γ = 0`` 退化为"只看眼前"：``G_t == r_t``。"""
    rewards = [3.0, -1.0, 2.0, 0.5]
    raw = compute_returns(rewards, gamma=0.0, normalize=False)["raw_returns"]
    assert np.allclose(raw.numpy(), np.asarray(rewards, dtype=np.float32), atol=1e-6)


def test_returns_normalization_is_zero_mean_unit_std():
    """Return 归一化给出零均值、单位标准差，且对奖励的正向缩放**不变**。"""
    rewards = [1.0] * 20 + [0.0] * 5
    out = compute_returns(rewards, gamma=0.99, normalize=True)
    assert abs(out["returns"].mean()) < 1e-5
    assert abs(float(out["returns"].std(unbiased=False)) - 1.0) < 1e-4
    # 归一化对尺度不变：把奖励放大 7 倍，returns 完全一致（区别只在 raw_std）
    scaled = compute_returns([r * 7.0 for r in rewards], gamma=0.99, normalize=True)
    assert torch.allclose(out["returns"], scaled["returns"], atol=1e-5)
    assert abs(scaled["raw_std"] - 7.0 * out["raw_std"]) < 1e-3
    # 回合内回报全相同（σ=0）时用 ε 兜底，不产生 NaN
    flat = compute_returns([1.0, 1.0, 1.0], gamma=0.0, normalize=True)
    assert bool(torch.isfinite(flat["returns"]).all())
    assert float(flat["returns"].abs().max()) < 1e6


def test_advantage_baseline_forms():
    """三种基线形态：无基线 / 批内均值 / 常量价值；减基线后优势均值为 0。"""
    ret = torch.tensor([10.0, 20.0, 30.0, 40.0])
    none_out = compute_advantages(ret, values=None)
    assert torch.allclose(none_out["advantages"], ret) and none_out["baseline_mean"] == 0.0

    mean_out = compute_advantages(ret, values=ret.mean().expand_as(ret))
    assert abs(mean_out["advantage_mean"]) < 1e-6
    assert torch.allclose(mean_out["advantages"], ret - ret.mean(), atol=1e-6)
    assert abs(mean_out["baseline_mean"] - float(ret.mean())) < 1e-6

    # V ≡ mean(G) 的"价值基线"与批内均值基线等价（常量基线是价值基线的特例）
    const = torch.full_like(ret, float(ret.mean()))
    assert torch.allclose(compute_advantages(ret, values=const)["advantages"],
                          mean_out["advantages"], atol=1e-6)

    norm_out = compute_advantages(ret, values=ret.mean().expand_as(ret), normalize=True)
    assert abs(norm_out["advantage_mean"]) < 1e-5
    assert abs(norm_out["advantage_std"] - 1.0) < 1e-4


def test_value_baseline_reduces_advantage_variance():
    """基线的两个关键性质：**常量平移不改批内方差**、``V_φ(s) ≈ G`` 时方差大幅下降。"""
    gen = torch.Generator().manual_seed(0)
    ret = 20.0 + 10.0 * torch.randn(64, generator=gen)
    zero_b = compute_advantages(ret, values=None)
    mean_b = compute_advantages(ret, values=ret.mean().expand_as(ret))
    near_b = compute_advantages(ret, values=ret + 0.5 * torch.randn(64, generator=gen))

    # 常量基线只是平移 ⇒ 批内方差不变（它的作用在"回合间"，见下一个测试）
    assert abs(mean_b["advantage_std"] - zero_b["advantage_std"]) < 1e-5
    assert abs(mean_b["advantage_mean"]) < 1e-5
    # 真正贴近 G 的 V_φ(s) 才能把优势方差压下去（方差缩减）
    assert near_b["advantage_std"] < 0.2 * zero_b["advantage_std"]


def test_batch_mean_baseline_makes_update_immune_to_reward_offset():
    """批内均值基线让更新对"回报整体加常数"免疫 —— 无偏性（``E[∇logπ·b]=0``）的直接体现。"""
    torch.manual_seed(4)
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    ep = _make_episode(policy, n_steps=12, seed=31)
    shifted = dict(ep)
    shifted["raw_returns"] = ep["raw_returns"] + 100.0          # 整体平移 100

    kw = dict(normalize_returns=False)
    g_mean = _update_grad_vector(policy, [ep], baseline="batch_mean", **kw)
    g_mean_shift = _update_grad_vector(policy, [shifted], baseline="batch_mean", **kw)
    assert torch.allclose(g_mean, g_mean_shift, atol=1e-6), "减均值后平移不应改变梯度"

    g_none = _update_grad_vector(policy, [ep], baseline="none", **kw)
    g_none_shift = _update_grad_vector(policy, [shifted], baseline="none", **kw)
    assert not torch.allclose(g_none, g_none_shift), "无基线时常数偏移会污染梯度尺度与方向"


def test_advantage_length_mismatch_raises():
    """基线长度与回报长度不一致时立刻报错（避免静默广播导致的错误梯度）。"""
    _expect_value_error(compute_advantages, [1.0, 2.0, 3.0],
                        values=[1.0, 2.0])




# =============================================================================
# 2. 策略网络 π_θ(a | s)
# =============================================================================
def _tiny_policy(hidden_dim: int = 16, num_layers: int = 1, seed: int = 0) -> PolicyNetwork:
    torch.manual_seed(seed)
    return PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)


def test_policy_distribution_is_a_valid_categorical():
    """概率归一、``log_prob`` 与手写 ``log_softmax`` 一致、熵落在 ``[0, log A]``。"""
    policy = _tiny_policy()
    states = torch.randn(6, 4)
    logits = policy(states)
    assert tuple(logits.shape) == (6, 2)
    dist = policy.distribution(states)
    probs = dist.probs
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert bool(((probs > 0) & (probs < 1)).all())

    actions = torch.tensor([0, 1, 1, 0, 1, 0])
    manual_lp = torch.log_softmax(logits, dim=-1).gather(1, actions.view(-1, 1)).squeeze(1)
    assert torch.allclose(policy.log_prob(states, actions), manual_lp, atol=1e-6)

    manual_ent = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    assert torch.allclose(policy.entropy(states), manual_ent, atol=1e-6)
    assert bool((policy.entropy(states) >= -1e-6).all())
    assert bool((policy.entropy(states) <= LOG2 + 1e-6).all())


def test_zero_logits_give_uniform_policy_with_max_entropy():
    """零 logits ⇒ 均匀分布，熵取到上界 ``log 2``（策略"一无所知"的状态）。"""
    policy = _tiny_policy()
    for p in policy.parameters():
        torch.nn.init.zeros_(p)
    states = torch.randn(3, 4)
    probs = policy.distribution(states).probs
    assert torch.allclose(probs, torch.full_like(probs, 0.5), atol=1e-6)
    assert torch.allclose(policy.entropy(states), torch.full((3,), LOG2), atol=1e-6)
    assert torch.allclose(policy.log_prob(states, torch.zeros(3, dtype=torch.long)),
                          torch.full((3,), -LOG2), atol=1e-6)


def test_policy_act_deterministic_and_sampling():
    """``deterministic=True`` 等价 ``argmax``；``False`` 时真的在按分布采样。"""
    policy = _tiny_policy()
    state = torch.randn(4)
    greedy = policy.act(state, deterministic=True)
    assert int(greedy["action"]) == int(policy(state.unsqueeze(0)).argmax(dim=-1))
    assert not greedy["log_prob"].requires_grad and not greedy["entropy"].requires_grad

    torch.manual_seed(1)
    draws = torch.cat([policy.act(state, deterministic=False)["action"] for _ in range(200)])
    assert set(draws.tolist()) == {0, 1}, "采样应覆盖两个动作（探索的来源）"
    # 采样频率应大致接近策略概率（宽松上界，只验证"不是常数动作"）
    emp = float(draws.float().mean())
    prob1 = float(policy.distribution(state.unsqueeze(0)).probs[0, 1])
    assert abs(emp - prob1) < 0.15, (emp, prob1)


def test_policy_log_prob_matches_direct_softmax_formula():
    """``logπ(a|s) = logits_a − logsumexp(logits)``（逐点一致，数值稳定）。"""
    policy = _tiny_policy(hidden_dim=32, num_layers=2)
    states = torch.randn(5, 4)
    actions = torch.tensor([1, 0, 1, 1, 0])
    logits = policy(states)
    expected = logits.gather(1, actions.view(-1, 1)).squeeze(1) - torch.logsumexp(logits, dim=-1)
    assert torch.allclose(policy.log_prob(states, actions), expected, atol=1e-6)
    assert bool((policy.log_prob(states, actions) <= 1e-6).all())


def test_value_network_shape_and_mse_loss():
    """Critic 输出 ``(N,)``；``value_loss`` 就是逐元素 MSE（目标不参与求导）。"""
    critic = ValueNetwork(state_dim=4, hidden_dim=16, num_layers=1)
    states = torch.randn(7, 4)
    values = critic.value(states)
    assert tuple(values.shape) == (7,) and tuple(critic(states).shape) == (7, 1)
    targets = torch.randn(7)
    loss = value_loss(values, targets)
    assert abs(float(loss) - float(((values - targets) ** 2).mean())) < 1e-6
    loss.backward()
    assert any(p.grad is not None for p in critic.parameters())
    assert not targets.requires_grad


# =============================================================================
# 3. 损失函数与负号（思考题 1）
# =============================================================================
def test_reinforce_loss_equals_negative_sum_of_log_prob_times_advantage():
    """``L = −Σ logπ·Â``：``reduction`` 决定聚合方式，负号决定优化方向。"""
    lp = torch.tensor([-0.70, -0.30, -1.20])
    adv = torch.tensor([2.0, -1.0, 0.5])
    loss_mean, stats = reinforce_loss(lp, adv)
    assert abs(float(loss_mean) - float(-(lp * adv).mean())) < 1e-6
    assert stats["reduction"] == "mean" and stats["n_steps"] == 3
    assert abs(stats["policy_gradient"] - float(loss_mean)) < 1e-6

    loss_sum, _ = reinforce_loss(lp, adv, reduction="sum")
    assert abs(float(loss_sum) - float(-(lp * adv).sum())) < 1e-6
    assert abs(float(loss_sum) - 3.0 * float(loss_mean)) < 1e-6

    # 熵项：L = −Σ logπ·Â − β·ΣH（最大化熵 ⇒ 损失里是减号）
    ent = torch.tensor([0.5, 0.5, 0.5])
    loss_ent, stats_ent = reinforce_loss(lp, adv, entropies=ent, entropy_coef=0.2)
    assert abs(float(loss_ent) - (float(loss_mean) - 0.2 * 0.5)) < 1e-6
    assert abs(stats_ent["entropy"] - 0.5) < 1e-6
    # β=0 时熵只是诊断量，不进入损失（不影响梯度）
    loss_b0, stats_b0 = reinforce_loss(lp, adv, entropies=ent, entropy_coef=0.0)
    assert abs(float(loss_b0) - float(loss_mean)) < 1e-6
    assert abs(stats_b0["entropy"] - 0.5) < 1e-6
    _expect_value_error(reinforce_loss, lp, adv, reduction="nope")


def test_advantage_enters_loss_as_a_constant_weight():
    """优势在损失里被 ``detach``：梯度只流向 ``logπ``，不流向优势的来源。"""
    lp = torch.tensor([-0.7, -0.3, -1.2], requires_grad=True)
    adv = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    loss, _ = reinforce_loss(lp, adv)
    loss.backward()
    assert adv.grad is None, "优势是常数权重，不应带梯度"
    assert lp.grad is not None
    assert torch.allclose(lp.grad, -adv.detach() / 3.0, atol=1e-6)      # dL/dlogπ = −A/N
    _expect_value_error(reinforce_loss, torch.zeros(3), torch.zeros(4))


def test_minus_sign_is_required_for_ascent():
    """★ 思考题 1：``L = −logπ·A`` 的负号 = 把"最小化"翻译成"最大化期望回报"。

    构造一个单步 Bandit（状态固定、动作 1 的优势 ``A = +1``），只改损失前的符号跑 20 步
    梯度更新：

    * ``L = −logπ·A``（正确写法）：动作 1 的概率被抬高，目标 ``J = logπ·A`` **上升**；
    * ``L = +logπ·A``（丢掉负号）：动作 1 的概率被压低，``J`` **下降** —— 也就是
      "训练去最大化坏动作的概率"，与策略梯度的目标完全相反。
    """
    state = torch.zeros(1, 4)
    action = torch.tensor([1])                  # 我们要抬高概率的"好动作"
    advantage = torch.tensor([1.0])             # A > 0

    def train_with_sign(sign: float, steps: int = 20):
        torch.manual_seed(0)
        policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=16, num_layers=1)
        opt = optim.Adam(policy.parameters(), lr=0.1)
        objective = lambda: float((policy.log_prob(state, action) * advantage).mean())  # noqa: E731
        j_before = objective()
        for _ in range(steps):
            loss = sign * (policy.log_prob(state, action) * advantage).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            prob = float(policy.distribution(state).probs[0, 1])
        return objective(), j_before, prob

    j_ok, j0_ok, p_ok = train_with_sign(-1.0)    # 有负号：梯度上升
    j_bad, j0_bad, p_bad = train_with_sign(+1.0)  # 无负号：梯度下降

    assert p_ok > 0.9, f"A>0 的动作概率应被抬高，实际 {p_ok:.4f}"
    assert j_ok > j0_ok, (j0_ok, j_ok)
    assert p_bad < 0.1, f"去掉负号会压低好动作的概率，实际 {p_bad:.4f}"
    assert j_bad < j0_bad, (j0_bad, j_bad)
    assert j_ok > j_bad


def test_update_policy_moves_log_probs_along_the_advantage_direction():
    """把"负号"放回真实更新路径：更新后 ``Σ_t A_t·Δlogπ_t > 0``，且优势大的动作提升更多。"""
    torch.manual_seed(3)
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    episode = _make_episode(policy, n_steps=10, seed=21)
    states, actions = episode["states"], episode["actions"]
    with torch.no_grad():
        returns = episode["raw_returns"]
        adv = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        logp_before = policy.log_prob(states, actions)

    opt = optim.Adam(policy.parameters(), lr=0.01)
    update_policy(policy, episode, optimizer=opt, baseline="batch_mean", normalize_returns=True)

    with torch.no_grad():
        delta = policy.log_prob(states, actions) - logp_before
    assert float((delta * adv).mean()) > 0.0, "蒙特卡洛目标 Σ Δlogπ·A 应为正（沿 ∇J 上升）"
    good, bad = adv > 0, adv < 0
    if bool(good.any()) and bool(bad.any()):
        assert float(delta[good].mean()) > 0.0, "优势 > 0 的动作其 logπ 应上升"
        assert float(delta[good].mean()) > float(delta[bad].mean()), "优势大的动作提升更多"



# =============================================================================
# 4. 回合采样与批次拼接
# =============================================================================
def test_run_episode_caches_policy_outputs():
    """缓存 ``(s, a, r, logπ, H)`` 与网络输出逐位一致；CartPole 回报 == 存活步数。"""
    torch.manual_seed(0)
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    env = make_env("CartPole-v1")
    try:
        ep = run_episode(env, policy, device=DEVICE, gamma=0.99, normalize_returns=False, seed=1)
    finally:
        env.close()

    t = int(ep["length"])
    assert 0 < t <= 500, t
    assert tuple(ep["states"].shape) == (t, 4)
    for key in ("actions", "log_probs", "entropies", "rewards", "returns", "raw_returns"):
        assert tuple(ep[key].shape) == (t,), (key, ep[key].shape)
    assert ep["states"].dtype == torch.float32 and ep["actions"].dtype == torch.long
    # CartPole 每步奖励 1.0 ⇒ 未折扣总回报 = 回合长度
    assert abs(ep["episode_return"] - float(t)) < 1e-6
    assert abs(float(ep["rewards"].sum()) - float(t)) < 1e-6
    assert abs(ep["discounted_return"] - float(ep["raw_returns"][0])) < 1e-4
    assert torch.allclose(ep["returns"], ep["raw_returns"])              # normalize=False

    with torch.no_grad():
        dist = policy.distribution(ep["states"])
        assert torch.allclose(dist.log_prob(ep["actions"]), ep["log_probs"], atol=1e-5)
        assert torch.allclose(dist.entropy(), ep["entropies"], atol=1e-5)
    assert not ep["log_probs"].requires_grad and not ep["entropies"].requires_grad
    assert int(ep["actions"].min()) >= 0 and int(ep["actions"].max()) <= 1
    assert bool(ep["terminated"]) or bool(ep["truncated"])


def test_episode_terminated_truncated_and_default_limit_paths():
    """三条终止路径都正确：环境 ``terminated`` / 环境 ``truncated`` / ``max_steps`` 兜底。"""
    policy = _tiny_policy(hidden_dim=8, num_layers=1)

    # (a) 环境在 5 步后 terminated（任务失败或成功）
    ep_a = run_episode(_DummyEnv(horizon=5, terminate=True), policy, device=DEVICE, max_steps=50)
    assert ep_a["length"] == 5 and ep_a["terminated"] is True and ep_a["truncated"] is False

    # (b) 环境自己到 1000 步才 truncated，但 max_steps=12 先行兜底
    ep_b = run_episode(_DummyEnv(horizon=1000, terminate=False), policy, device=DEVICE, max_steps=12)
    assert ep_b["length"] == 12 and ep_b["truncated"] is True and ep_b["terminated"] is False
    assert abs(float(ep_b["rewards"].sum()) - 12.0) < 1e-6

    # (c) max_steps=None：回落到 env_max_steps（无 spec 的假环境 → 默认 500）
    ep_c = run_episode(_DummyEnv(horizon=1000, terminate=False), policy, device=DEVICE)
    assert ep_c["length"] == 500 and ep_c["truncated"] is True


def test_stack_episodes_concatenates_and_keeps_episode_index():
    """批拼接：顺序、长度、``episode_index`` 与逐回合回报都保持可追溯。"""
    policy = _tiny_policy()
    ep_a = _make_episode(policy, n_steps=4, seed=1)
    ep_b = _make_episode(policy, n_steps=6, seed=2)
    batch = stack_episodes([ep_a, ep_b])

    assert batch["n_episodes"] == 2 and batch["n_transitions"] == 10
    assert batch["episode_lengths"] == [4, 6]
    assert torch.allclose(batch["states"], torch.cat([ep_a["states"], ep_b["states"]]))
    assert torch.allclose(batch["actions"], torch.cat([ep_a["actions"], ep_b["actions"]]))
    assert batch["episode_index"].tolist() == [0] * 4 + [1] * 6
    assert batch["episode_returns"] == [ep_a["episode_return"], ep_b["episode_return"]]
    # 单条轨迹既能直接传，也能包成列表传
    assert stack_episodes([ep_a])["n_transitions"] == 4
    _expect_value_error(stack_episodes, [])


def test_duplicated_episode_batch_matches_single_episode_gradient():
    """批更新 = 各回合梯度的**按步数加权平均**（同一条轨迹复制两份梯度不变）。"""
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    ep = _make_episode(policy, n_steps=10, seed=5)
    kw = dict(baseline="batch_mean", normalize_returns=False)
    g_single = _update_grad_vector(policy, [ep], **kw)
    g_dup = _update_grad_vector(policy, [ep, ep], **kw)
    assert torch.allclose(g_single, g_dup, atol=1e-6)

    ep1 = _make_episode(policy, n_steps=8, seed=11)
    ep2 = _make_episode(policy, n_steps=5, seed=12)
    kw = dict(baseline="none", normalize_returns=False)
    g1 = _update_grad_vector(policy, [ep1], **kw)
    g2 = _update_grad_vector(policy, [ep2], **kw)
    gb = _update_grad_vector(policy, [ep1, ep2], **kw)
    assert torch.allclose(gb, (8.0 * g1 + 5.0 * g2) / 13.0, atol=1e-6)



# =============================================================================
# 5. 一次策略改进 update_policy
# =============================================================================
def test_update_policy_changes_parameters_and_reports_consistent_stats():
    """更新确实改变参数；诊断量与批数据自洽（``Â`` 零均值单位标准差、``drift ≈ 0``）。"""
    torch.manual_seed(11)
    policy = _tiny_policy(hidden_dim=32, num_layers=2)
    ep = _make_episode(policy, n_steps=20, seed=17)
    before = [p.detach().clone() for p in policy.parameters()]
    opt = optim.Adam(policy.parameters(), lr=1e-2)
    stats = update_policy(policy, ep, optimizer=opt, gamma=0.99,
                          baseline="batch_mean", normalize_returns=True)

    assert any(not torch.allclose(b, p.detach()) for b, p in zip(before, policy.parameters()))
    assert abs(stats["advantage_mean"]) < 1e-5
    assert abs(stats["advantage_std"] - 1.0) < 1e-3
    assert stats["n_steps"] == 20.0 and stats["n_episodes"] == 1.0
    assert abs(stats["baseline_offset"] - float(ep["raw_returns"].mean())) < 1e-4
    assert abs(stats["return_raw_mean"] - float(ep["raw_returns"].mean())) < 1e-4
    assert stats["grad_norm"] > 0.0
    assert 0.0 <= stats["entropy"] <= LOG2 + 1e-6
    assert stats["log_prob_drift"] < 1e-5
    assert stats["critic_loss"] == 0.0 and stats["critic_coef"] == 0.0   # 本批未启用 Critic

    # optimizer=None：只计算、不更新（评估与单元测试用）
    frozen = [p.detach().clone() for p in policy.parameters()]
    update_policy(policy, ep, optimizer=None, baseline="none")
    assert all(torch.allclose(f, p.detach()) for f, p in zip(frozen, policy.parameters()))


def test_recomputed_log_probs_match_cached():
    """★ 采样缓存的 logπ 与更新时重算的 logπ 一致（drift ≈ 1e-7）；更新之后才会漂移。"""
    torch.manual_seed(0)
    policy = _tiny_policy(hidden_dim=32, num_layers=2)
    ep = _make_episode(policy, n_steps=15, seed=3)

    stats = update_policy(policy, ep, optimizer=None, baseline="batch_mean")
    assert stats["log_prob_drift"] < 1e-5, stats["log_prob_drift"]

    # 缓存值与"按 (s, a) 直接前向"逐位一致
    with torch.no_grad():
        direct = policy.log_prob(ep["states"], ep["actions"])
    assert torch.allclose(direct, ep["log_probs"], atol=1e-6)

    # 参数更新之后，同样的 (s, a) 重算出的 logπ 才会与缓存值分离
    opt = optim.Adam(policy.parameters(), lr=0.5)
    for _ in range(3):
        update_policy(policy, ep, optimizer=opt, baseline="batch_mean")
    with torch.no_grad():
        moved = policy.log_prob(ep["states"], ep["actions"])
    assert float((moved - ep["log_probs"]).abs().max()) > 1e-3


def test_joint_optimizer_matches_two_separate_optimizers():
    """★ 一个联合 Adam + 一个联合损失 == 两个独立 Adam + 两个独立损失（逐位一致）。

    成立的原因：优势 ``(G − V(s)).detach()`` 切断了流向 Critic 的梯度，于是
    "策略项只依赖策略参数、Critic 项只依赖 Critic 参数"，而 Adam 是**逐参数**更新的，
    两个参数子集的更新过程完全独立 —— 联合与否只是写法差异。
    """
    torch.manual_seed(7)
    policy_a = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=32, num_layers=2)
    critic_a = ValueNetwork(state_dim=4, hidden_dim=32, num_layers=2)
    policy_b, critic_b = copy.deepcopy(policy_a), copy.deepcopy(critic_a)
    ep = _make_episode(policy_a, n_steps=16, seed=11)
    lr, coef = 0.05, 1.0

    # (a) 联合：与 src.reinforce.update_policy 的实现完全一致
    joint_opt = optim.Adam(list(policy_a.parameters()) + list(critic_a.parameters()), lr=lr)
    stats = update_policy(policy_a, ep, optimizer=joint_opt, critic=critic_a, baseline="value",
                          critic_coef=coef, normalize_returns=True)
    assert stats["critic_loss"] > 0.0 and stats["critic_value_mean"] != 0.0

    # (b) 拆分：教科书式的两个优化器 + 两个损失
    opt_p, opt_c = optim.Adam(policy_b.parameters(), lr=lr), optim.Adam(critic_b.parameters(), lr=lr)
    dist = policy_b.distribution(ep["states"])
    lp, ent = dist.log_prob(ep["actions"]), dist.entropy()
    values = critic_b.value(ep["states"])
    adv = ep["raw_returns"] - values.detach()
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    loss_p, _ = reinforce_loss(lp, adv, entropies=ent)
    loss_c = value_loss(values, ep["raw_returns"])
    opt_p.zero_grad(set_to_none=True)
    opt_c.zero_grad(set_to_none=True)
    loss_p.backward()
    loss_c.backward()
    opt_p.step()
    opt_c.step()

    for pa, pb in zip(policy_a.parameters(), policy_b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), float((pa - pb).abs().max())
    for ca, cb in zip(critic_a.parameters(), critic_b.parameters()):
        assert torch.allclose(ca, cb, atol=1e-6), float((ca - cb).abs().max())



def test_grad_clip_limits_global_gradient_norm():
    """``max_grad_norm`` 生效时参数梯度范数被压到阈值；``grad_norm`` 记录的是裁剪前值。"""
    torch.manual_seed(5)
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    ep = _make_episode(policy, n_steps=16, seed=2)
    big = dict(ep)
    big["raw_returns"] = ep["raw_returns"] * 500.0            # 放大优势 ⇒ 放大梯度
    kw = dict(baseline="none", normalize_returns=False)
    opt = optim.SGD(policy.parameters(), lr=0.0)              # lr=0：只观察梯度，不改参数

    free = update_policy(policy, big, optimizer=opt, **kw)
    assert free["grad_norm"] > 1.0, free["grad_norm"]

    clipped = update_policy(policy, big, optimizer=opt, max_grad_norm=0.5, **kw)
    assert clipped["grad_norm"] > 1.0                         # 裁剪前范数
    norm_after = float(np.sqrt(sum(
        float(p.grad.pow(2).sum()) for p in policy.parameters() if p.grad is not None
    )))
    assert abs(norm_after - 0.5) < 1e-4, norm_after


def test_return_normalization_only_rescales_the_gradient():
    """思考题 2：Return 归一化 ≈ 给梯度乘 ``1/σ(G)``（改尺度、不改方向、对奖励缩放免疫）。"""
    torch.manual_seed(3)
    policy = _tiny_policy(hidden_dim=32, num_layers=2)
    ep = _make_episode(policy, n_steps=25, seed=13)
    kw = dict(baseline="batch_mean")

    g_off = _update_grad_vector(policy, [ep], normalize_returns=False, **kw)
    g_on = _update_grad_vector(policy, [ep], normalize_returns=True, **kw)
    sigma = float(ep["raw_returns"].std(unbiased=False))

    ratio = float(g_off.norm()) / float(g_on.norm())
    assert abs(ratio - sigma) / sigma < 5e-3, (ratio, sigma)
    cos = float(torch.nn.functional.cosine_similarity(g_off, g_on, dim=0))
    assert cos > 0.9999, cos

    # 把回报整体放大 10 倍：不归一化时梯度同比例放大；归一化后**完全不变**
    ep_big = dict(ep)
    ep_big["raw_returns"] = ep["raw_returns"] * 10.0
    g_off_big = _update_grad_vector(policy, [ep_big], normalize_returns=False, **kw)
    g_on_big = _update_grad_vector(policy, [ep_big], normalize_returns=True, **kw)
    assert float(g_off_big.norm()) > 9.0 * float(g_off.norm())
    assert torch.allclose(g_on_big, g_on, atol=1e-6), "归一化应当对奖励尺度不变"


def test_value_baseline_trains_critic_towards_returns():
    """``baseline=value`` 时 Critic 的 MSE 在多步更新中显著下降（它确实在学 V_φ）。"""
    torch.manual_seed(9)
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=32, num_layers=2)
    critic = ValueNetwork(state_dim=4, hidden_dim=32, num_layers=2)
    ep = _make_episode(policy, n_steps=32, seed=23)
    opt = optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=5e-3)
    losses = [
        update_policy(policy, ep, optimizer=opt, critic=critic, baseline="value",
                      normalize_returns=True, critic_coef=0.5)["critic_loss"]
        for _ in range(60)
    ]
    assert losses[-1] < losses[0], (losses[0], losses[-1])
    assert losses[-1] < 0.3 * losses[0], (losses[0], losses[-1])


def test_update_policy_validates_baseline_and_critic():
    """非法基线名 / ``baseline=value`` 缺 Critic 时立刻报错。"""
    policy = _tiny_policy()
    ep = _make_episode(policy, n_steps=5, seed=0)
    _expect_value_error(update_policy, policy, ep, optimizer=None, baseline="nope")
    _expect_value_error(update_policy, policy, ep, optimizer=None, baseline="value")



# =============================================================================
# 6. 端到端
# =============================================================================
def test_evaluate_policy_is_reproducible_and_equals_survived_steps():
    """独立评估可复现；CartPole 的回报 == 存活步数；贪心评估无采样噪声。"""
    policy = _tiny_policy(hidden_dim=16, num_layers=1)
    r1 = evaluate_policy(policy, device=DEVICE, episodes=3, seed=7)
    r2 = evaluate_policy(policy, device=DEVICE, episodes=3, seed=7)
    assert r1["mean_return"] == r2["mean_return"]              # 同一策略 + 同一种子 ⇒ 逐位一致
    assert abs(r1["mean_return"] - r1["mean_length"]) < 1e-3   # 每步奖励 1 ⇒ 回报 = 步数
    assert r1["n_episodes"] == 3.0 and float(r1["deterministic"]) == 1.0
    assert 0.0 <= r1["min_return"] <= r1["mean_return"] <= r1["max_return"] <= 500.0
    # 换一个种子会得到不同的回合（说明评估逐回合播种，而不是复用同一条轨迹）
    r3 = evaluate_policy(policy, device=DEVICE, episodes=5, seed=8)
    assert r3["mean_return"] != r1["mean_return"]
    assert 0.0 < r1["std_return"] < 500.0


def test_short_training_on_cartpole_improves_return():
    """端到端冒烟：REINFORCE 在 CartPole-v1 上 120 个回合内显著优于随机策略。"""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = PolicyNetwork(state_dim=4, action_dim=2, hidden_dim=64, num_layers=2)
    opt = optim.Adam(policy.parameters(), lr=2e-3)
    env = make_env("CartPole-v1")
    returns: List[float] = []
    try:
        for episode in range(1, 121):
            ep = run_episode(env, policy, device=DEVICE, gamma=0.99,
                             normalize_returns=False, seed=seed + episode - 1)
            returns.append(float(ep["episode_return"]))
            update_policy(policy, ep, optimizer=opt, gamma=0.99,
                          baseline="batch_mean", normalize_returns=True)
    finally:
        env.close()

    early = float(np.mean(returns[:10]))       # 近乎随机策略的起点
    late = float(np.mean(returns[-20:]))       # 训练末期的水平
    assert early < 60.0, early
    assert late > 2.5 * early, (early, late)
    assert late > 120.0, (early, late)
    # 训练完成后用贪心策略评估，应明显好于训练期的采样策略
    eval_result = evaluate_policy(policy, device=DEVICE, episodes=5, seed=123)
    assert eval_result["mean_return"] > 100.0, eval_result


# =============================================================================
# 运行器（无需 pytest）
# =============================================================================
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:                     # noqa: BLE001
            failed.append((name, exc))
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("-" * 60)
    print(f"通过 {len(tests) - len(failed)}/{len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

