#!/usr/bin/env python
"""W8 Task 1（连续空间 PPO / PPO-Clip + GAE-λ）单元测试。

覆盖范围
--------
1. **手写数学**：对角高斯 ``log_prob`` / 熵的闭式实现与 ``torch.distributions`` 逐位一致；
   ``σ = exp(clip(log_std))`` 与夹紧行为。
2. **连续动作的截断（思考题）**：
   * env 拿到的是 ``clip(a_raw)``，而 buffer 里存的是 ``a_raw``（两者分离）；
   * 用 ``clip(a_raw)`` 算 ``logπ`` 会产生**与奖励无关的系统性伪梯度**：
     ``μ = ±5, σ = 0.1`` 时 ≈ ∓300（把 ``μ`` 硬拽向动作边界），而正确做法恒为零均值；
   * 若把 ``clip(a_raw)`` 当成动作存进 buffer，PPO 在"参数未变的第 0 个 epoch"
     就会出现 ``r_t ≠ 1``；存 ``a_raw`` 则恒有 ``r_t = 1``。
3. **RolloutBuffer**：定长写入 / 越界报错 / 复位；字段形状与 dtype；
   ``2048/64 = 32`` 个 mini-batch 恰好覆盖每个样本一次（不重不漏）、不同 epoch 划分不同。
4. **GAE-λ**：与**独立推导**的朴素 ``O(T²)`` 展开式逐位一致；``λ=0`` 退化为 TD(0)（``Â=δ``、
   ``Target = r + γV(s')``）；``γ=λ=1`` 退化为蒙特卡洛（``Â_t + V_t = Σ r``）；
   ``done`` 掩码（回合边界不泄漏优势）；``truncated`` 自举开关的差异。
5. **PPO 损失**：``L^CLIP`` 与手写公式一致、``r=1`` 时等于 ``−mean(Â)``；优势被 ``detach``；
   正向优势 + 比值超出上限时**梯度被裁剪分支截断**（不再产生策略梯度）；
   ``L^VF = ½·MSE``（裁剪版 = 悲观下界 + 越界方向梯度归零）；熵项符号
   （``−c₂H`` ⇒ 最小化总损失等价于最大化熵）；优势标准化确实得到零均值单位标准差；
   ``ppo_losses`` 用 ``a_env`` 而非 ``a_raw`` 时代理目标会"自欺地变好"。
6. **优化器**：两个 param group 的单 Adam 与**两个独立 Adam**逐步等价（逐参数比对）。
7. **一次 PPO 更新**：参数确实被更新、诊断量自洽；``lr=0`` 时 ``ratio ≡ 1``（不变量）；
   梯度裁剪让参数步长变小；``target_kl`` 早停；空 buffer 报 ``ValueError``。
8. **状态归一化 / 采样器 / 评估**：``RunningMeanStd`` 与 numpy 逐位一致且可存档往返；
   惰性 reset 的边界自举（回合在边界结束时 ``last_value=0``，否则等于 ``V(s_T)``）；
   ``reward_scale`` 只改 buffer；喂给环境的动作永远在 ``[-2, 2]`` 内；
   ``evaluate_policy`` 可复现、回合长度 = 200、且不更新归一化统计量。
9. **端到端**：① 玩具"二次型单步环境"上 PPO 把 ``μ`` 从 0 推到最优 1.5（证明整条流水线
   确实在做收益上升）；② Pendulum-v1 上短路训练：固定价值目标上 Critic 损失显著下降、
   多轮 rollout 诊断量全部有限。
10. **工程脚本**：``configs/ppo_config.yaml`` 与作业指定超参逐项一致；
   ``experiments/train_ppo.py`` 的 ``as_bool / optional_float / rolling_mean /
   check_action_bounds``；``experiments/export_tb_curves.py`` 的标量读取与 PNG/CSV 导出。

运行方式（无需 pytest）::

    python tests/test_ppo.py

若已安装 pytest，也可直接 ``pytest tests/test_ppo.py``。
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ppo import (  # noqa: E402
    LOG_STD_MAX,
    LOG_STD_MIN,
    Actor,
    ActorCritic,
    Critic,
    RolloutBuffer,
    RolloutCollector,
    RunningMeanStd,
    build_actor_critic,
    build_ppo_optimizer,
    compute_gae,
    entropy_of_normal,
    evaluate_policy,
    gaussian_log_prob,
    make_env,
    policy_loss_clip,
    ppo_losses,
    ppo_update,
    value_loss,
)

DEVICE = torch.device("cpu")          # 测试统一用 CPU，保证任何机器可复现
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# 测试辅助：独立实现的 GAE 朴素展开式（O(T²)，作为递归实现的"金标准"）
# =============================================================================
def naive_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    done: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """按**定义**直接展开 ``Â_t = Σ_k (γλ)^k · δ_{t+k}``（每个 k 都重新累乘掩码）。

    与 ``compute_gae`` 的倒序递推是两条完全不同的代码路径：递推 O(T)，本实现 O(T²)，
    两者逐位一致才说明递推没写错（尤其是 done 掩码与自举项的位置）。
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    d = np.asarray(done, dtype=np.float64).reshape(-1)
    T = r.size
    # δ_t = r_t + γ·V(s_{t+1})·(1−done_t) − V(s_t)，其中 V(s_T) 取 last_value
    deltas = np.zeros(T, dtype=np.float64)
    for t in range(T):
        next_v = v[t + 1] if t + 1 < T else float(last_value)
        deltas[t] = r[t] + gamma * next_v * (1.0 - d[t]) - v[t]
    adv = np.zeros(T, dtype=np.float64)
    for t in range(T):
        acc, alive = 0.0, 1.0
        for k in range(T - t):
            if k > 0:
                alive *= (1.0 - d[t + k - 1])       # 中间出现 done ⇒ 之后的 δ 不再贡献
            if alive == 0.0:
                break
            acc += (gamma * gae_lambda) ** k * deltas[t + k]
        adv[t] = acc
    return adv, adv + v


class QuadraticBanditEnv:
    """玩具环境：**单步回合**，奖励 ``r = −(a − 1.5)²``，动作空间 ``[-2, 2]``。

    最优动作是 ``a* = 1.5``，状态恒为 ``[0]``（状态无关的连续控制问题）。
    用它做端到端学习断言：Pendulum 需要几万步才见效，而这里几百步就能看到 ``μ → 1.5``，
    且因为每步都是真终止（``terminated=True``），GAE 退化为"逐样本的即时奖励回归"，
    数学上非常干净 —— 如果 PPO 的符号 / 目标 / 更新顺序有任何错误，``μ`` 就不会向 1.5 靠。
    """

    def __init__(self, target: float = 1.5) -> None:
        import gymnasium as gym

        self.target = float(target)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.spec = None
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None, options: Any = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        a = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
        reward = -((a - self.target) ** 2)
        return np.zeros(1, dtype=np.float32), float(reward), True, False, {}

    def close(self) -> None:      # 与其他 Gym 环境的接口保持一致
        pass


def load_script_module(name: str, path: str):
    """把 ``experiments/*.py`` 当模块加载（experiments 目录不是包，只能按路径加载）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

# =============================================================================
# 1. 手写数学：高斯 log_prob / 熵 / σ = exp(clip(log_std))
# =============================================================================
def test_handwritten_gaussian_log_prob_matches_torch():
    """手写对角高斯密度与 ``torch.distributions.Normal`` 逐位一致。"""
    torch.manual_seed(0)
    mu = torch.randn(7, 3)
    std = torch.rand(7, 3) + 0.2
    action = mu + torch.randn(7, 3) * 0.5
    mine = gaussian_log_prob(action, mu, std)
    ref = Normal(mu, std).log_prob(action).sum(dim=-1, keepdim=True)
    assert mine.shape == (7, 1)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_handwritten_entropy_matches_torch():
    """手写熵 ``Σ(½(1+log2π) + logσ)`` 与 ``Normal.entropy()`` 一致。"""
    log_std = torch.tensor([[-0.5, 0.3, 1.2]])
    mine = entropy_of_normal(log_std)
    ref = Normal(torch.zeros_like(log_std), torch.exp(log_std)).entropy().sum(dim=-1, keepdim=True)
    assert torch.allclose(mine, ref, atol=1e-6)
    assert mine.shape == (1, 1)


def test_std_is_exp_of_clipped_log_std():
    """``σ = exp(clip(log_std))``：可学习参数 + 上下界夹紧（防熵崩塌 / 全随机）。"""
    actor = Actor(state_dim=3, log_std_init=-0.5)
    assert abs(float(actor.std()[0]) - math.exp(-0.5)) < 1e-6
    assert id(actor.log_std) in {id(p) for p in actor.parameters()}

    with torch.no_grad():
        actor.log_std.fill_(5.0)
    assert abs(float(actor.clipped_log_std()[0]) - LOG_STD_MAX) < 1e-6
    with torch.no_grad():
        actor.log_std.fill_(-50.0)
    assert abs(float(actor.clipped_log_std()[0]) - LOG_STD_MIN) < 1e-6


def test_mean_may_exceed_bounds_but_env_action_is_clipped():
    """``μ(s)`` 允许越界（高斯定义在 ℝ 上），但喂给环境的动作必在 ``[-2, 2]``。"""
    actor = Actor(state_dim=3, action_low=-2.0, action_high=2.0)
    with torch.no_grad():
        actor.mu_net[-1].bias.fill_(5.0)            # 均值远远超出上界
    out = actor.act(torch.zeros(4, 3))
    assert (out["mean"] > 2.0).all()
    assert float(out["action_env"].max()) <= 2.0 + 1e-6
    assert float(out["action_env"].min()) >= -2.0 - 1e-6
    assert torch.allclose(out["action_env"], torch.clamp(out["action_raw"], -2.0, 2.0))


# =============================================================================
# 2. 连续动作截断与 log_prob（W8 思考题）
# =============================================================================
def test_buffer_keeps_raw_action_while_env_gets_clipped_action():
    """env 只执行 ``clip(a_raw)``，但 buffer 必须保留 ``a_raw``（思考题核心结论）。"""
    torch.manual_seed(3)
    env = make_env("Pendulum-v1")
    model = build_actor_critic({"model": {"log_std_init": 1.0}})      # σ≈2.7 ⇒ 必然大量越界
    buffer = RolloutBuffer(capacity=256, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, normalizer=None,
                                 seed=7, normalize_obs=False)
    try:
        info = collector.collect(256)
    finally:
        env.close()

    raw, env_act = collector.last_actions_raw, collector.last_actions_env
    assert raw.shape == (256, 1) and env_act.shape == (256, 1)
    assert (raw.abs() > 2.0).any(), "σ=2.7 时应当出现越界采样，否则本测试失去意义"
    assert float(env_act.min()) >= -2.0 - 1e-6 and float(env_act.max()) <= 2.0 + 1e-6
    assert torch.allclose(env_act, torch.clamp(raw, -2.0, 2.0))
    # buffer 里存的必须是 a_raw（未截断），而不是喂给环境的动作
    assert torch.allclose(buffer.actions_raw[: len(buffer)], raw, atol=1e-5)
    assert not torch.allclose(buffer.actions_raw[: len(buffer)], env_act, atol=1e-5)
    assert info["action_clip_fraction"] > 0.0


def test_clipped_action_log_prob_is_biased():
    """用 ``clip(a)`` 算 logπ 会得到**非零均值伪梯度**；用 ``a_raw`` 才是无偏的（零均值）。"""
    torch.manual_seed(0)
    mu = torch.tensor([[5.0]], requires_grad=True)          # 均值远在边界之外
    std = torch.full((1,), 0.1)
    n = 200000
    a_raw = Normal(mu.detach(), std).sample((n,)).reshape(n, 1)
    a_clip = torch.clamp(a_raw, -2.0, 2.0)

    mu_raw = mu.expand(n, 1)
    grad_raw = torch.autograd.grad(gaussian_log_prob(a_raw, mu_raw, std).mean(), mu)[0]
    mu_clip = mu.expand(n, 1)
    grad_clip = torch.autograd.grad(gaussian_log_prob(a_clip, mu_clip, std).mean(), mu)[0]

    assert abs(float(grad_raw)) < 0.5, float(grad_raw)          # ≈ 0：得分函数零均值（蒙特卡洛误差 ~1/(σ√n)）
    # μ=+5 时概率质量全落在上界之外 ⇒ E[clip(a)] = 2 ⇒ 伪梯度 (2 − 5)/σ² = −300
    # （策略梯度上升沿 +∂ 方向走 ⇒ μ 被硬拽向边界 +2；镜像的 μ=−5 给出 +300 ⇒ 拽向 −2）
    assert -320.0 < float(grad_clip) < -280.0, float(grad_clip)
    mu_neg = torch.tensor([[-5.0]], requires_grad=True)
    a_neg = torch.clamp(Normal(mu_neg.detach(), std).sample((n,)).reshape(n, 1), -2.0, 2.0)
    grad_neg = torch.autograd.grad(
        gaussian_log_prob(a_neg, mu_neg.expand(n, 1), std).mean(), mu_neg)[0]
    assert 280.0 < float(grad_neg) < 320.0, float(grad_neg)
    # 这个"梯度"与奖励无关、只由"越界 ⇒ 贴边"制造，量级比真实梯度大两个数量级以上
    assert abs(float(grad_clip)) > 100.0 * max(abs(float(grad_raw)), 0.1)


def test_ratio_is_one_only_when_raw_action_is_stored():
    """``r_t = 1`` 不变量：动作存 ``a_raw`` 时成立；存 ``clip(a)`` 时会坏掉。"""
    torch.manual_seed(11)
    states = torch.randn(64, 3)

    model = build_actor_critic({"model": {}})
    out = model.act(states)
    old_log_prob = out["log_prob"]                               # 采样时按 a_raw 计算
    new_log_prob = model.log_prob(states, out["action_raw"])      # 参数未变 ⇒ 重算一致
    assert torch.allclose(old_log_prob, new_log_prob, atol=1e-6)
    assert torch.allclose(torch.exp(new_log_prob - old_log_prob), torch.ones_like(old_log_prob), atol=1e-6)

    # σ 很大 ⇒ 大量动作越界：若错误地把 clip(a) 当动作，比值立刻偏离 1
    wide = build_actor_critic({"model": {"log_std_init": 1.5}})   # σ≈4.5
    out_wide = wide.act(states)
    ratio_wrong = torch.exp(wide.log_prob(states, out_wide["action_env"]) - out_wide["log_prob"])
    assert float((ratio_wrong - 1.0).abs().max()) > 0.1
    ratio_right = torch.exp(wide.log_prob(states, out_wide["action_raw"]) - out_wide["log_prob"])
    assert float((ratio_right - 1.0).abs().max()) < 1e-5

# =============================================================================
# 3. RolloutBuffer：定长写入 / 越界保护 / mini-batch 划分
# =============================================================================
def test_rollout_buffer_write_read_and_capacity():
    """写入 / 读取 / 容量保护 / 复位（复位只挪指针，不清空数据）。"""
    buffer = RolloutBuffer(capacity=8, state_dim=3, action_dim=2)
    assert len(buffer) == 0 and not buffer.is_full
    for i in range(8):
        idx = buffer.add(state=[i, i + 1, i + 2], action_raw=[0.1 * i, -0.1 * i], log_prob=-0.5,
                         reward=-1.0 * i, value=0.25, terminated=(i == 7), truncated=False)
        assert idx == i
    assert len(buffer) == 8 and buffer.is_full
    assert buffer.terminated.sum() == 1.0 and buffer.truncated.sum() == 0.0
    assert torch.allclose(buffer.states[3], torch.tensor([3.0, 4.0, 5.0]))
    assert torch.allclose(buffer.rewards[5], torch.tensor([-5.0]))

    try:
        buffer.add([0, 0, 0], [0.0], 0.0, 0.0, 0.0)
        raise AssertionError("缓冲区满时必须抛 IndexError，防止静默覆盖数据")
    except IndexError:
        pass

    buffer.reset()
    assert len(buffer) == 0
    assert float(buffer.states[3].sum()) == 12.0        # 复位不清空历史数据


def test_minibatches_cover_each_sample_exactly_once():
    """``2048/64 = 32`` 批，且每个样本恰好出现一次（不重不漏）。"""
    buffer = RolloutBuffer(capacity=2048, state_dim=3)
    for _ in range(2048):
        buffer.add([0.0, 0.0, 0.0], [0.0], 0.0, 0.0, 0.0)

    batches = buffer.get_minibatches(batch_size=64, generator=torch.Generator().manual_seed(0))
    assert len(batches) == 32
    assert all(b.numel() == 64 for b in batches)
    flat = torch.cat(batches)
    assert flat.numel() == 2048 and torch.unique(flat).numel() == 2048
    assert sorted(flat.tolist()) == list(range(2048))

    # 换一个 generator ⇒ 划分不同（每个 epoch 都重新打乱）
    other = buffer.get_minibatches(batch_size=64, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(flat, torch.cat(other))

    # 尾批处理：100 个样本 / 64 ⇒ [64, 36]；drop_last=True ⇒ [64]
    small = RolloutBuffer(capacity=100, state_dim=3)
    for _ in range(100):
        small.add([0.0, 0.0, 0.0], [0.0], 0.0, 0.0, 0.0)
    sizes = [b.numel() for b in small.get_minibatches(64, generator=torch.Generator().manual_seed(0))]
    sizes_drop = [b.numel() for b in small.get_minibatches(64, generator=torch.Generator().manual_seed(0),
                                                          drop_last=True)]
    assert sizes == [64, 36] and sizes_drop == [64]
    assert RolloutBuffer(capacity=4, state_dim=3).get_minibatches(64) == []

# =============================================================================
# 4. GAE-λ：与朴素 O(T²) 展开式比对 + 各退化情形
# =============================================================================
def test_gae_matches_naive_ot2_expansion():
    """倒序递推（O(T)）与按定义展开的朴素实现（O(T²)）逐位一致。"""
    torch.manual_seed(0)
    T = 40
    rewards = torch.randn(T, 1)
    values = torch.randn(T, 1)
    terminated = torch.zeros(T, 1)
    truncated = torch.zeros(T, 1)
    terminated[17] = 1.0            # 中途一次真终止（掩码生效）
    truncated[29] = 1.0             # 中途一次到点（默认继续自举）
    gamma, lam, last_value = 0.97, 0.90, 0.3

    out = compute_gae(rewards, values, terminated, truncated,
                      gamma=gamma, gae_lambda=lam, last_value=last_value)
    naive_adv, naive_ret = naive_gae(rewards.numpy(), values.numpy(), terminated.numpy().reshape(-1),
                                     gamma=gamma, gae_lambda=lam, last_value=last_value)
    assert torch.allclose(out["advantages"], torch.as_tensor(naive_adv, dtype=torch.float32).reshape(T, 1),
                          atol=1e-5)
    assert torch.allclose(out["returns"], torch.as_tensor(naive_ret, dtype=torch.float32).reshape(T, 1),
                          atol=1e-5)
    # Target_t = Â_t + V(s_t)
    assert torch.allclose(out["returns"], out["advantages"] + values, atol=1e-6)
    # Δ_t = r_t + γV(s_{t+1})·(1 − done_t) − V(s_t)（逐项手算，含终止掩码）
    delta_manual = (rewards[:-1] + gamma * values[1:] * (1.0 - terminated[:-1]) - values[:-1])
    assert torch.allclose(out["deltas"][:-1], delta_manual, atol=1e-5)
    assert abs(float(out["deltas"][17]) - float(rewards[17] - values[17])) < 1e-5   # 终止步不自举


def test_gae_lambda_zero_is_td0():
    """``λ=0`` ⇒ ``Â_t = δ_t``，``Target_t = r_t + γV(s_{t+1})``（一步 TD 目标）。"""
    torch.manual_seed(1)
    T = 6
    rewards, values = torch.randn(T, 1), torch.randn(T, 1)
    out = compute_gae(rewards, values, torch.zeros(T, 1), torch.zeros(T, 1),
                      gamma=0.9, gae_lambda=0.0, last_value=0.5)
    assert torch.allclose(out["advantages"], out["deltas"], atol=1e-6)
    assert abs(float(out["returns"][0] - (rewards[0] + 0.9 * values[1]))) < 1e-5
    assert abs(float(out["returns"][T - 1] - (rewards[T - 1] + 0.9 * 0.5))) < 1e-5


def test_gae_lambda_one_gamma_one_is_monte_carlo():
    """``γ=λ=1``、``V≡0`` ⇒ ``Â_t`` 退化为"从 t 起的奖励后缀和"（完全蒙特卡洛）。"""
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    values = torch.zeros(5, 1)
    out = compute_gae(rewards, values, torch.zeros(5, 1), torch.zeros(5, 1),
                      gamma=1.0, gae_lambda=1.0, last_value=0.0)
    expected = torch.tensor([[15.0], [14.0], [12.0], [9.0], [5.0]])
    assert torch.allclose(out["advantages"], expected, atol=1e-5)
    assert torch.allclose(out["returns"], expected, atol=1e-5)


def test_gae_done_mask_stops_advantage_leakage():
    """``terminated`` 处的优势不得跨回合泄漏（掩码把递推切断）。"""
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    values = torch.zeros(4, 1)
    terminated = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    out = compute_gae(rewards, values, terminated, None, gamma=0.99, gae_lambda=0.95, last_value=0.0)
    adv = out["advantages"].reshape(-1)
    assert abs(float(adv[2]) - 3.0) < 1e-5                     # 终止步只等于自身即时奖励
    assert abs(float(adv[3]) - 4.0) < 1e-5
    assert abs(float(adv[1]) - (2.0 + 0.99 * 0.95 * 3.0)) < 1e-5
    assert abs(float(adv[0]) - (1.0 + 0.99 * 0.95 * float(adv[1]))) < 1e-5


def test_gae_truncation_bootstrap_flag_matters():
    """``truncated`` 是否自举会改变优势（Pendulum 每 200 步到点，这个开关很关键）。"""
    rewards = torch.ones(3, 1)
    values = torch.zeros(3, 1)
    truncated = torch.tensor([[0.0], [0.0], [1.0]])
    boot = compute_gae(rewards, values, torch.zeros(3, 1), truncated,
                       gamma=0.9, gae_lambda=1.0, last_value=10.0, bootstrap_on_truncation=True)
    cut = compute_gae(rewards, values, torch.zeros(3, 1), truncated,
                      gamma=0.9, gae_lambda=1.0, last_value=10.0, bootstrap_on_truncation=False)
    assert abs(float(boot["advantages"][2]) - (1.0 + 0.9 * 10.0)) < 1e-5   # 继续自举 V(s_T)
    assert abs(float(cut["advantages"][2]) - 1.0) < 1e-5                   # 当作终止 ⇒ 掩掉自举
    assert float(boot["advantages"][0]) > float(cut["advantages"][0])
    assert compute_gae(rewards, values, torch.zeros(3, 1), None)["advantages"].shape == (3, 1)
    try:
        compute_gae(rewards, values[:2], torch.zeros(3, 1), None)
        raise AssertionError("形状不一致应当报错")
    except ValueError:
        pass

# =============================================================================
# 5. PPO 损失：L^CLIP / L^VF / 熵 / 优势标准化
# =============================================================================
def test_policy_loss_matches_handwritten_formula():
    """``L^CLIP`` 与手写公式逐项一致，并给出正确的 ``ratio / clip_fraction / approx_kl``。"""
    torch.manual_seed(0)
    old = torch.randn(16, 1) * 0.3
    new = (old + torch.randn(16, 1) * 0.1).requires_grad_(True)
    adv = torch.randn(16, 1)
    eps = 0.2
    loss, stats = policy_loss_clip(new, old, adv, clip_eps=eps)

    ratio = torch.exp(new.detach() - old)
    manual = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
    assert abs(float(loss) - float(manual)) < 1e-6
    assert abs(stats["ratio_mean"] - float(ratio.mean())) < 1e-6
    assert abs(stats["clip_fraction"] - float(((ratio - 1).abs() > eps).float().mean())) < 1e-6
    assert stats["approx_kl"] >= -1e-6                      # k3 估计量 (r−1)−log r ≥ 0


def test_policy_loss_ratio_one_is_negative_advantage_mean():
    """``r ≡ 1``（参数未变）时 ``L^CLIP = −mean(Â)``。"""
    torch.manual_seed(2)
    log_prob = torch.randn(32, 1)
    adv = torch.randn(32, 1)
    loss, _ = policy_loss_clip(log_prob.clone(), log_prob.detach(), adv, clip_eps=0.2)
    assert abs(float(loss) + float(adv.mean())) < 1e-6


def test_policy_loss_gradient_blocked_when_clipped():
    """比值超出裁剪区间时，``min`` 取到常数分支 ⇒ 该样本不再产生策略梯度（信赖域的"刹车"）。"""
    new_log_prob = torch.tensor([[math.log(1.5)]], requires_grad=True)      # r = 1.5 > 1+ε
    loss, stats = policy_loss_clip(new_log_prob, torch.zeros(1, 1), torch.tensor([[1.0]]), clip_eps=0.2)
    assert abs(float(loss) + 1.2) < 1e-6                    # 取到裁剪分支 clip(r)·Â = 1.2
    assert abs(float(torch.autograd.grad(loss, new_log_prob)[0])) < 1e-6
    assert stats["clip_fraction"] == 1.0

    inside = torch.tensor([[0.05]], requires_grad=True)     # 比值 ≈ 1.05，落在区间内
    loss2, _ = policy_loss_clip(inside, torch.zeros(1, 1), torch.tensor([[1.0]]), clip_eps=0.2)
    assert abs(float(torch.autograd.grad(loss2, inside)[0])) > 1e-3

    # 负优势 + 比值低于 1−ε 同样被截断
    neg = torch.tensor([[math.log(0.5)]], requires_grad=True)
    loss3, _ = policy_loss_clip(neg, torch.zeros(1, 1), torch.tensor([[-1.0]]), clip_eps=0.2)
    assert abs(float(torch.autograd.grad(loss3, neg)[0])) < 1e-6


def test_policy_loss_advantage_is_detached():
    """优势是**常数权重**：梯度不回流到 ``advantages``（PPO 不训练优势本身）。"""
    adv = torch.ones(8, 1, requires_grad=True)
    new = torch.zeros(8, 1, requires_grad=True)
    loss, _ = policy_loss_clip(new, torch.zeros(8, 1), adv)
    loss.backward()
    assert adv.grad is None
    assert new.grad is not None and float(new.grad.abs().sum()) > 0.0


def test_advantage_normalization_is_zero_mean_unit_std():
    """mini-batch 内标准化后 ``mean(Â)=0``、``std(Â)=1`` ⇒ ``r≡1`` 时损失为 0。"""
    adv = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
    new = torch.zeros(4, 1, requires_grad=True)
    loss, _ = policy_loss_clip(new, torch.zeros(4, 1), adv, normalize_advantage=True)
    assert abs(float(loss)) < 1e-6
    normalized = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    assert abs(float(normalized.mean())) < 1e-6
    assert abs(float(normalized.std(unbiased=False)) - 1.0) < 1e-6


def test_value_loss_is_half_mse_and_clip_option():
    """``L^VF = ½·mean((V−Target)²)``；``clip_eps`` 让价值也受限 —— 悲观下界 + 越界方向梯度归零。"""
    values = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([[3.0], [0.0]])
    assert abs(float(value_loss(values, targets)) - 0.5 * float(((values - targets) ** 2).mean())) < 1e-9
    assert abs(float(value_loss(values, targets, None, clip_eps=0.2))
               - float(value_loss(values, targets))) < 1e-9            # 没给 V_old ⇒ 退化为把手写公式

    old = torch.zeros(2, 1)
    target = torch.full((2, 1), 1.0)
    near = torch.full((2, 1), 0.1, requires_grad=True)                 # 移动 0.1 < ε ⇒ 未被裁剪
    assert abs(float(value_loss(near, target, old, clip_eps=0.2)) - float(value_loss(near, target))) < 1e-7

    far = torch.full((2, 1), 0.5, requires_grad=True)                  # 移动 0.5 > ε ⇒ 触发裁剪
    clipped_far = value_loss(far, target, old, clip_eps=0.2)
    assert float(clipped_far) > float(value_loss(far, target))         # max(...) ⇒ 悲观下界
    grad_far = torch.autograd.grad(clipped_far, far)[0]
    assert float(grad_far.abs().max()) < 1e-7                          # 该方向不再产生梯度

def test_entropy_term_rewards_exploration():
    """总损失里的 ``−c₂·H``：熵项精确等于 ``−c₂·H``，且梯度会把 ``log_std`` 往上推。"""
    torch.manual_seed(5)
    model = build_actor_critic({"model": {"log_std_init": -1.0}})
    states = torch.randn(32, 3)
    actions, zeros = torch.zeros(32, 1), torch.zeros(32, 1)
    _, no_bonus = ppo_losses(model, states, actions, zeros, zeros, zeros, zeros,
                             value_coef=0.0, entropy_coef=0.0)
    _, with_bonus = ppo_losses(model, states, actions, zeros, zeros, zeros, zeros,
                               value_coef=0.0, entropy_coef=0.5)
    assert abs(no_bonus["total_loss"]) < 1e-6
    assert abs((no_bonus["total_loss"] - with_bonus["total_loss"]) - 0.5 * no_bonus["entropy"]) < 1e-6

    before = float(model.clipped_log_std().mean())
    optimizer = build_ppo_optimizer(model, lr_actor=1e-2, lr_critic=1e-2)
    for _ in range(5):
        loss, _ = ppo_losses(model, states, actions, zeros, zeros, zeros, zeros,
                             value_coef=0.0, entropy_coef=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert float(model.clipped_log_std().mean()) > before + 0.01


def test_ppo_losses_uses_stored_raw_action():
    """``ppo_losses`` 必须用 buffer 里的 ``a_raw``：用截断动作会得到错位的代理目标。"""
    torch.manual_seed(7)
    model = build_actor_critic({"model": {"log_std_init": 1.5}})       # σ≈4.5 ⇒ 大量越界
    states = torch.randn(64, 3)
    out = model.act(states)
    advantages = torch.ones(64, 1)                                     # 用非零优势，比值才进入损失
    zeros = torch.zeros(64, 1)

    loss_raw, stats_raw = ppo_losses(model, states, out["action_raw"], out["log_prob"],
                                     zeros, advantages, zeros, value_coef=0.0,
                                     normalize_advantage=False)
    assert abs(stats_raw["policy_loss"] + 1.0) < 1e-6                  # r≡1 ⇒ L^CLIP = −mean(Â) = −1
    assert abs(stats_raw["ratio_mean"] - 1.0) < 1e-6 and stats_raw["clip_fraction"] == 0.0
    assert abs(float(loss_raw) + 1.0) < 1e-6                           # 关掉价值项后总损失 = 策略损失

    loss_clip, stats_clip = ppo_losses(model, states, out["action_env"], out["log_prob"],
                                       zeros, advantages, zeros, value_coef=0.0,
                                       normalize_advantage=False)
    assert abs(stats_clip["policy_loss"] + 1.0) > 1e-2                 # 用 a_env ⇒ r ≠ 1，目标错位
    assert abs(stats_clip["ratio_mean"] - 1.0) > 0.05                  # 比值系统性偏离 1
    assert float(loss_clip) < float(loss_raw)                          # 错位目标反而"显得更好"（自欺）


def test_explained_variance_extremes():
    """``explained_variance``：V 完全等于目标时为 1，与"常数预测"持平时为 0。"""
    from src.ppo import explained_variance

    targets = torch.randn(64, 1)
    assert abs(explained_variance(targets.clone(), targets) - 1.0) < 1e-5
    assert abs(explained_variance(torch.full_like(targets, float(targets.mean())), targets)) < 1e-5


# =============================================================================
# 6. 优化器：单 Adam 双 param group ≡ 两个独立 Adam
# =============================================================================
def test_two_param_groups_equiv_two_separate_adam():
    """两个 param group 的单 Adam 与两个独立 Adam **逐步逐参数等价**（Adam 是逐参数更新）。"""
    import copy

    torch.manual_seed(0)
    joint_model = build_actor_critic({"model": {}})
    sep_model = copy.deepcopy(joint_model)
    joint_opt = build_ppo_optimizer(joint_model, lr_actor=3e-4, lr_critic=1e-3)
    actor_opt = optim.Adam(sep_model.actor_parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
    critic_opt = optim.Adam(sep_model.critic_parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    states = torch.randn(32, 3)
    actions = torch.randn(32, 1)
    advantages = torch.randn(32, 1)
    old_log_probs = torch.zeros(32, 1)
    targets = torch.randn(32, 1)
    old_values = torch.zeros(32, 1)

    for _ in range(5):
        loss_joint, _ = ppo_losses(joint_model, states, actions, old_log_probs, old_values,
                                   advantages, targets, normalize_advantage=False)
        joint_opt.zero_grad(set_to_none=True)
        loss_joint.backward()
        joint_opt.step()

        loss_sep, _ = ppo_losses(sep_model, states, actions, old_log_probs, old_values,
                                 advantages, targets, normalize_advantage=False)
        actor_opt.zero_grad(set_to_none=True)
        critic_opt.zero_grad(set_to_none=True)
        loss_sep.backward()
        actor_opt.step()
        critic_opt.step()

    for (name_a, param_a), (name_b, param_b) in zip(joint_model.named_parameters(),
                                                    sep_model.named_parameters()):
        assert name_a == name_b
        assert torch.allclose(param_a, param_b, atol=1e-6), name_a

# =============================================================================
# 7. ppo_update：不变量、参数更新、梯度裁剪、KL 早停
# =============================================================================
def collect_buffer(model, steps: int = 256, seed: int = 7):
    """用给定模型在 Pendulum 上采样一个 rollout（顺带把 GAE 算好）。"""
    env = make_env("Pendulum-v1")
    buffer = RolloutBuffer(capacity=steps, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, normalizer=None,
                                 seed=seed, normalize_obs=False)
    try:
        collector.collect(steps)
    finally:
        env.close()
    return buffer


def test_ppo_update_with_zero_lr_keeps_ratio_one():
    """``lr=0`` ⇒ 参数不动 ⇒ 每个 mini-batch 的 ``r ≡ 1``、``L^CLIP = −mean(Â)``、``KL = 0``。"""
    torch.manual_seed(0)
    model = build_actor_critic({"model": {}})
    buffer = collect_buffer(model, 256)
    optimizer = build_ppo_optimizer(model, lr_actor=0.0, lr_critic=0.0)
    stats = ppo_update(model, optimizer, buffer, update_epochs=3, batch_size=64,
                       max_grad_norm=0.0, normalize_advantage=False)
    assert abs(stats["ratio_mean"] - 1.0) < 1e-5
    assert stats["clip_fraction"] == 0.0
    assert abs(stats["approx_kl"]) < 1e-6
    assert abs(stats["policy_loss"] + stats["advantage_mean"]) < 1e-3
    assert stats["epochs_run"] == 3.0 and stats["n_minibatches"] == 12.0      # 3 轮 × 4 批


def test_ppo_update_changes_params_and_reports_finite_stats():
    """一次真实更新：参数确实被改动，且所有诊断量有限、键齐全。"""
    torch.manual_seed(0)
    model = build_actor_critic({"model": {}})
    buffer = collect_buffer(model, 512)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = build_ppo_optimizer(model, lr_actor=1e-3, lr_critic=1e-3)
    stats = ppo_update(model, optimizer, buffer, clip_eps=0.2, update_epochs=2, batch_size=128,
                       value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5)

    changed = [k for k, v in before.items() if not torch.allclose(v, model.state_dict()[k])]
    assert changed, "更新后应当有参数发生变化"
    assert any(name.startswith("actor.") for name in changed)
    assert any(name.startswith("critic.") for name in changed)
    for key in ("policy_loss", "value_loss", "entropy", "total_loss", "ratio_mean", "ratio_std",
                "clip_fraction", "approx_kl", "grad_norm", "explained_variance", "log_std_mean",
                "value_abs_error_mean", "advantage_mean", "advantage_std", "advantage_abs_mean",
                "target_mean", "n_steps", "n_minibatches", "epochs_run"):
        assert key in stats, key
        assert math.isfinite(float(stats[key])), key
    assert stats["n_steps"] == 512.0 and stats["epochs_run"] == 2.0 and stats["n_minibatches"] == 8.0


def test_gradient_clipping_shrinks_parameter_step():
    """梯度裁剪确实起作用：裁剪越狠，参数移动越少。"""
    import copy

    torch.manual_seed(0)
    base = build_actor_critic({"model": {}})
    buffer = collect_buffer(base, 256)

    steps = {}
    for tag, max_norm in (("off", 0.0), ("on", 0.05)):
        model = copy.deepcopy(base)
        optimizer = build_ppo_optimizer(model, lr_actor=1e-3, lr_critic=1e-3)
        ppo_update(model, optimizer, buffer, update_epochs=1, batch_size=256,
                   max_grad_norm=max_norm, normalize_advantage=False)
        distance = sum(float((p - q).pow(2).sum()) for p, q in
                       zip(model.parameters(), base.parameters()))
        steps[tag] = math.sqrt(distance)
    assert steps["on"] < steps["off"], steps


def test_ppo_update_kl_early_stop():
    """``target_kl`` 触发时：只跑 1 个 epoch 就停，并在诊断里标记。"""
    torch.manual_seed(0)
    model = build_actor_critic({"model": {}})
    buffer = collect_buffer(model, 256)
    optimizer = build_ppo_optimizer(model, lr_actor=1e-3, lr_critic=1e-3)
    stats = ppo_update(model, optimizer, buffer, update_epochs=10, batch_size=64,
                       max_grad_norm=0.5, target_kl=-1.0)
    assert stats["epochs_run"] == 1.0 and stats["kl_early_stop"] == 1.0
    try:
        ppo_update(model, optimizer, RolloutBuffer(capacity=8, state_dim=3), update_epochs=1)
        raise AssertionError("空 buffer 应当报错")
    except ValueError:
        pass

# =============================================================================
# 8. 状态归一化 / 采样器 / 独立评估
# =============================================================================
def test_running_mean_std_matches_numpy_and_roundtrips():
    """在线统计量与 numpy 全量统计一致；单样本 update 与批量 update 等价；可序列化。"""
    rng = np.random.RandomState(0)
    data = rng.randn(500, 3) * np.array([1.0, 2.0, 5.0]) + 0.5
    rms = RunningMeanStd(3)
    for start in range(0, 500, 50):
        rms.update(data[start:start + 50])
    assert abs(rms.count - (500 + 1e-4)) < 1e-3
    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-6)

    normalized = rms.normalize_np(data)
    assert abs(float(normalized.mean())) < 1e-4
    assert abs(float(normalized.std()) - 1.0) < 1e-2

    one_by_one, batched = RunningMeanStd(3), RunningMeanStd(3)
    for row in data[:100]:
        one_by_one.update(row)
    batched.update(data[:100])
    assert np.allclose(one_by_one.mean, batched.mean, atol=1e-8)
    assert np.allclose(one_by_one.var, batched.var, atol=1e-8)

    clone = RunningMeanStd(3)
    clone.load_state_dict(rms.state_dict())
    assert np.allclose(clone.normalize_np(data), normalized)
    clone.load_state_dict(None)                       # None 不应改动任何东西
    assert np.allclose(clone.normalize_np(data), normalized)

    extreme = rms.normalize_np(np.array([1e6, 1e6, 1e6]))
    assert np.all(np.abs(extreme) <= rms.clip + 1e-6)  # 极端值被截断
    torch_version = rms.normalize_torch(torch.as_tensor(data[:10], dtype=torch.float32))
    assert np.allclose(torch_version.numpy(), rms.normalize_np(data[:10]), atol=1e-5)


def test_collector_bootstraps_from_true_next_state():
    """rollout 未结束回合时，边界自举值必须等于 ``V(s_T)``（而不是新回合初始状态的价值）。"""
    torch.manual_seed(0)
    env = make_env("Pendulum-v1")
    model = build_actor_critic({"model": {}})
    buffer = RolloutBuffer(capacity=64, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, seed=1, normalize_obs=False)
    try:
        info = collector.collect(64)                 # 64 < 200 ⇒ 回合不会结束
    finally:
        env.close()

    assert info["n_episodes"] == 0 and collector._obs is not None
    with torch.no_grad():
        expected = float(model.value(torch.as_tensor(collector._obs, dtype=torch.float32).unsqueeze(0)))
    assert abs(info["last_value"] - expected) < 1e-5
    assert float(info["gae"]["done"].sum()) == 0.0                 # 无真终止
    expected_delta = float(buffer.rewards[63]) + 0.99 * expected - float(buffer.values[63])
    assert abs(float(info["gae"]["deltas"][63]) - expected_delta) < 1e-4
    # GAE 已写回 buffer，Target = Â + V
    assert torch.allclose(buffer.returns[:64], buffer.advantages[:64] + buffer.values[:64], atol=1e-5)


def test_collector_zero_bootstrap_when_episode_ends_at_boundary():
    """回合恰好在 rollout 边界结束（Pendulum 200 步）⇒ 自举值取 0，优势不再跨回合累加。"""
    torch.manual_seed(0)
    env = make_env("Pendulum-v1")
    model = build_actor_critic({"model": {}})
    buffer = RolloutBuffer(capacity=200, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, seed=0, normalize_obs=False)
    try:
        info = collector.collect(200)
    finally:
        env.close()

    assert info["n_episodes"] == 1 and info["episode_lengths"] == [200]
    assert info["last_value"] == 0.0
    assert collector._obs is None                                   # 下一步前才会 reset（惰性）
    expected_delta = float(buffer.rewards[199]) - float(buffer.values[199])
    assert abs(float(info["gae"]["deltas"][199]) - expected_delta) < 1e-4


def test_collector_reward_scale_only_affects_buffer():
    """``reward_scale`` 只改学习目标（buffer 奖励），不改日志里的原始回合回报。"""
    torch.manual_seed(0)
    env = make_env("Pendulum-v1")
    model = build_actor_critic({"model": {}})
    buffer = RolloutBuffer(capacity=200, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, seed=0,
                                 normalize_obs=False, reward_scale=0.1)
    try:
        info = collector.collect(200)
    finally:
        env.close()
    assert len(info["episode_returns"]) == 1
    raw_return = info["episode_returns"][0]
    assert -2000.0 < raw_return < 0.0
    assert abs(float(buffer.rewards[:200].sum()) - 0.1 * raw_return) < 1e-3


def test_evaluate_policy_pendulum_validity_and_reproducibility():
    """评估：固定种子可复现、回合长度恒为 200、随机策略回报在 −1200 ~ −1600 量级。"""
    torch.manual_seed(0)
    model = build_actor_critic({"model": {}})
    first = evaluate_policy(model, device=DEVICE, episodes=3, seed=123)
    second = evaluate_policy(model, device=DEVICE, episodes=3, seed=123)
    assert first["mean_return"] == second["mean_return"]
    assert first["n_episodes"] == 3.0 and first["episodes"] == 3.0
    assert first["mean_length"] == 200.0
    assert -2000.0 < first["mean_return"] < 0.0
    assert first["min_return"] <= first["mean_return"] <= first["max_return"]
    assert first["deterministic"] == 1.0
    assert evaluate_policy(model, device=DEVICE, episodes=3, seed=7)["mean_return"] != first["mean_return"]
    stochastic = evaluate_policy(model, device=DEVICE, episodes=3, seed=123, deterministic=False)
    assert stochastic["mean_return"] != first["mean_return"]

    # 评估只读归一化统计量（不会因为评估而更新分布）
    normalizer = RunningMeanStd(3)
    normalizer.update(np.zeros((10, 3)))
    count_before = normalizer.count
    evaluate_policy(model, device=DEVICE, episodes=1, seed=1, normalizer=normalizer)
    assert normalizer.count == count_before
    assert math.isfinite(evaluate_policy(model, device=DEVICE, episodes=1, seed=1,
                                         normalizer=normalizer)["mean_return"])

# =============================================================================
# 9. 端到端：玩具环境学到最优动作 + Pendulum 短路训练不崩
# =============================================================================
def test_ppo_solves_quadratic_bandit():
    """端到端①：``r = −(a−1.5)²`` 的单步环境上，PPO 应把 ``μ`` 从 0 推到最优 1.5。

    这是对整条流水线（采样 → a_raw 存 buffer → GAE → Clip 损失 → 双学习率 Adam）
    最强的符号/逻辑检查：任何一处写反，``μ`` 都会朝错误方向跑或原地不动。
    """
    torch.manual_seed(0)
    env = QuadraticBanditEnv(target=1.5)
    model = build_actor_critic({"model": {"state_dim": 1, "action_dim": 1, "hidden_dim": 32,
                                          "log_std_init": 0.0, "action_low": -2.0, "action_high": 2.0}})
    buffer = RolloutBuffer(capacity=64, state_dim=1, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, seed=0,
                                 normalize_obs=False, gamma=1.0, gae_lambda=0.95)
    optimizer = build_ppo_optimizer(model, lr_actor=3e-3, lr_critic=3e-3)

    zero = torch.zeros(1, 1)
    mu_before = float(model.mu(zero))
    returns: List[float] = []
    try:
        for _ in range(200):
            info = collector.collect(64)                     # 64 个单步回合
            returns.extend(info["episode_returns"])
            ppo_update(model, optimizer, buffer, clip_eps=0.2, update_epochs=4, batch_size=64,
                       value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5, normalize_advantage=True)
    finally:
        env.close()

    mu_after = float(model.mu(zero))
    assert abs(mu_before) < 0.3, mu_before
    assert abs(mu_after - 1.5) < 0.4, mu_after                      # 学到最优动作
    early = float(np.mean(returns[:100]))
    late = float(np.mean(returns[-100:]))
    assert late > early + 1.0, (early, late)                        # 回报从 ≈ −4 升到 ≈ 0
    # 采样给环境的动作始终被截断到动作空间内，且等于 clip(a_raw)
    assert torch.all(collector.last_actions_env >= -2.0)
    assert torch.all(collector.last_actions_env <= 2.0)
    assert torch.allclose(collector.last_actions_env, collector.last_actions_raw.clamp(-2.0, 2.0))


def test_pendulum_short_training_is_stable_and_critic_learns():
    """端到端②：Pendulum-v1 上采样 512 步并做多轮更新 ——
    ① Critic 对**同一批价值目标**的回归损失显著下降（价值目标在更新中是常量 ⇒ 等价于监督回归）；
    ② 连续多个 rollout 的训练诊断全部有限、回合回报始终在合理区间。"""
    torch.manual_seed(0)
    env = make_env("Pendulum-v1")
    model = build_actor_critic({"model": {}})
    buffer = RolloutBuffer(capacity=512, state_dim=3, action_dim=1)
    collector = RolloutCollector(env, model, buffer, device=DEVICE, seed=0, normalize_obs=True,
                                 normalizer=RunningMeanStd(3))
    optimizer = build_ppo_optimizer(model, lr_actor=3e-4, lr_critic=1e-3)
    try:
        info = collector.collect(512)
        with torch.no_grad():
            value_before = float(value_loss(model.value(buffer.states[:512]), buffer.returns[:512]))
        stats = ppo_update(model, optimizer, buffer, clip_eps=0.2, update_epochs=20, batch_size=64,
                           value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5)
        with torch.no_grad():
            value_after = float(value_loss(model.value(buffer.states[:512]), buffer.returns[:512]))
        assert value_after < 0.6 * value_before, (value_before, value_after)   # Critic 确实在学
        assert stats["n_minibatches"] == 160.0 and stats["n_steps"] == 512.0
        # 再跑两个 rollout：诊断量有限、裁剪比例在 [0,1]、回合回报为负
        for _ in range(2):
            latest = collector.collect(512)
            stats = ppo_update(model, optimizer, buffer, clip_eps=0.2, update_epochs=2, batch_size=128,
                               value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5)
            for key in ("policy_loss", "value_loss", "approx_kl", "grad_norm", "total_loss", "entropy"):
                assert math.isfinite(float(stats[key])), key
            assert 0.0 <= stats["clip_fraction"] <= 1.0
            assert stats["log_std_mean"] < LOG_STD_MAX + 1e-6
            assert latest["std_mean"] > 0.0 and latest["n_steps"] == 512     # collect() 返回 σ
            assert all(-2000.0 < r < 0.0 for r in latest["episode_returns"])
    finally:
        env.close()

    assert collector.normalizer is not None and collector.normalizer.count > 512
    # 归一化统计量把 θ̇（尺度 ~8）与 cos/sin（尺度 ~1）拉到同一量级
    assert 0.5 < float(collector.normalizer.std.min()) < 2.0
    assert 1.0 < float(collector.normalizer.std.max()) < 12.0

# =============================================================================
# 10. 工程脚本：配置一致性 + train_ppo 工具函数 + TensorBoard 曲线导出
# =============================================================================
def test_config_file_matches_assignment_spec():
    """``configs/ppo_config.yaml`` 的指定字段必须与作业要求逐项一致（防止手改错超参）。"""
    import yaml

    with open(os.path.join(ROOT, "configs", "ppo_config.yaml"), "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    assert config["experiment"]["seed"] == 42
    assert config["experiment"]["env_name"] == "Pendulum-v1"
    assert config["model"]["state_dim"] == 3 and config["model"]["action_dim"] == 1
    assert config["model"]["hidden_dim"] == 64
    training = config["training"]
    assert training["total_timesteps"] == 200000 and training["rollout_steps"] == 2048
    assert training["gamma"] == 0.99 and training["gae_lambda"] == 0.95
    assert training["clip_eps"] == 0.2 and training["update_epochs"] == 10
    assert training["batch_size"] == 64
    assert abs(training["lr_actor"] - 3e-4) < 1e-12 and abs(training["lr_critic"] - 1e-3) < 1e-12
    assert training["max_grad_norm"] == 0.5 and training["target_reward"] == -200.0
    # 数据利用率：一个 rollout 恰好切成整数个 mini-batch（不浪费样本）
    assert training["rollout_steps"] % training["batch_size"] == 0


def test_train_ppo_script_helpers():
    """``train_ppo.py`` 的配置解析 / 动作边界对齐等辅助函数。"""
    module = load_script_module("train_ppo_script", os.path.join(ROOT, "experiments", "train_ppo.py"))

    assert module.as_bool("false") is False and module.as_bool("False") is False
    assert module.as_bool("true") is True and module.as_bool("on") is True
    assert module.as_bool(None) is False and module.as_bool(1) is True
    assert module.as_bool(None, default=True) is True
    assert module.as_bool("???", default=True) is True          # 无法识别 ⇒ 回退到默认值

    assert module.optional_float(None) is None and module.optional_float("null") is None
    assert module.optional_float("") is None and module.optional_float(0.2) == 0.2

    assert module.rolling_mean([1.0, 2.0, 3.0], 2) == 2.5
    assert module.rolling_mean([1.0, 2.0, 3.0], 0) == 2.0        # window<=0 ⇒ 全部数据
    assert module.rolling_mean([], 5) == 0.0

    env = make_env("Pendulum-v1")
    try:
        config = {"model": {"action_low": -1.0, "action_high": 1.0}}     # 故意写错的边界
        fixed = module.check_action_bounds(config, env)
        assert fixed["action_low"] == -2.0 and fixed["action_high"] == 2.0
        assert config["model"]["action_low"] == -2.0                      # 原地修正

        from src.ppo import env_max_steps

        assert env_max_steps(env, default=123) == 200                     # Pendulum 单回合上限
    finally:
        env.close()


def test_export_tb_curves_writes_png_and_csv():
    """``export_tb_curves.py``：写一次假的事件文件，验证 PNG / CSV / 摘要全链路。"""
    from torch.utils.tensorboard import SummaryWriter

    module = load_script_module("export_tb_curves_script",
                               os.path.join(ROOT, "experiments", "export_tb_curves.py"))
    tmp = tempfile.mkdtemp(prefix="tb_export_")
    try:
        run_dir = os.path.join(tmp, "ppo_fake_run")
        writer = SummaryWriter(log_dir=run_dir)
        for i in range(10):
            writer.add_scalar("train/episode_return", -1200.0 + 110.0 * i, 2048 * i)
            writer.add_scalar("train/rolling_return", -1150.0 + 100.0 * i, 2048 * i)
            writer.add_scalar("eval/mean_return", -1100.0 + 95.0 * i, 2048 * i)
            writer.add_scalar("train/action_clip_fraction", 0.2 - 0.01 * i, 2048 * i)
        writer.close()

        info = module.export_curves(run_dir, out_dir=os.path.join(tmp, "figs"), prefix="unit")
        assert os.path.isfile(info["png"]) and os.path.getsize(info["png"]) > 5000
        assert os.path.isfile(info["csv"])
        assert info["n_tags"] == 4 and info["n_rows"] == 40
        assert "eval/mean_return" in info["plotted"]
        with open(info["csv"], "r", encoding="utf-8") as fh:
            rows = fh.read().strip().splitlines()
        assert rows[0] == "tag,step,value" and len(rows) == 41

        assert module.resolve_run_dir(tmp) == run_dir            # 父目录 ⇒ 自动定位唯一运行目录
        summary = "\n".join(module.summarize(module.load_scalars(run_dir), module.DEFAULT_SUMMARY_TAGS))
        assert "eval/mean_return" in summary and "train/action_clip_fraction" in summary
        try:
            module.resolve_run_dir(os.path.join(tmp, "not_exist"))
            raise AssertionError("不存在的目录应当报错")
        except FileNotFoundError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# 运行器（不依赖 pytest）
# =============================================================================
def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    print(f"== W8 Task 1 PPO 单元测试：共 {len(tests)} 项 ==")
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
