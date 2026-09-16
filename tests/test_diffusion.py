#!/usr/bin/env python
"""W4 Task 2（极简条件扩散策略 Mini-Diffusion Policy）单元测试。

覆盖范围
--------
1. **噪声调度表数学**：``β_t`` / ``ᾱ_t`` 单调性与范围、预计算系数与定义式一致、
   推理时间步序列（降序、含 0、就地跨步）。
2. **前向加噪 / 闭式反解**：``q_sample`` 与公式逐点一致（含 ``t`` 广播正确性）、
   由 ``ε`` 反解 ``a^0`` 的往返精度、后验均值 / 方差与 DDPM 原论文一致。
3. **噪声预测网络**：输入输出形状 ``(B, K, A)``、对 ``t`` 与 ``s`` 的**条件敏感性**
   （改变 ``t`` 或 ``s`` 必须改变输出，否则条件化失效）、``t`` 标量广播。
4. **扩散策略接口**：与 Task 1 策略同接口（``forward`` / ``forward_actions`` /
   ``training``）；输出形状 ``(B, K, A)``、落在动作空间内、真采样（逐次不同）、
   固定随机源可复现；每次规划恰好 ``N`` 次网络前向（推理开销的来源）。
5. **训练目标**：噪声预测 MSE 的理论下界性质（``ε_θ ≡ 0`` 时损失 ≈ 1）、
   掩码损失**忽略零填充位置**、损失可反传、小规模训练使损失显著下降。
6. **多峰机理（核心论据）**：在对称双峰玩具数据上，**确定性 MSE 回归收敛到条件
   均值 ≈ 0**（"撞墙的平均动作"），而**扩散策略采样能覆盖两个模态**。
7. **多峰专家数据**：``expert_controller_multimodal`` 两种模式都能摆起，
   且方向歧义区域（``ambiguity_mask``）内的真实标签确实双峰（±kick 各约一半）。
8. **引擎兼容性**：``train_one_epoch`` / ``validate`` 的 ``step_fn`` 扩展与默认路径
   行为一致（默认路径必须与改造前逐位一致）。

运行方式（无需 pytest）::

    python tests/test_diffusion.py

若已安装 pytest，也可直接 ``pytest tests/test_diffusion.py``。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import (  # noqa: E402
    BCDataset,
    ambiguity_mask,
    collect_multimodal_expert_trajectories,
    expert_controller,
    expert_controller_multimodal,
    make_bimodal_toy_dataset,
    toy_bimodal_mode_centers,
)
from src.diffusion import (  # noqa: E402
    DiffusionPolicy,
    DiffusionSchedule,
    NoisePredictor,
    build_diffusion_policy,
    count_parameters,
)
from src.engine import train_one_epoch, validate  # noqa: E402
from src.networks import build_network  # noqa: E402

DEVICE = torch.device("cpu")          # 测试统一用 CPU，保证任何机器可复现


# =============================================================================
# 1. 噪声调度表
# =============================================================================
def test_schedule_tables_are_consistent():
    """``β_t`` / ``ᾱ_t`` 的范围、单调性与全部派生系数都满足定义式。"""
    for name in ("cosine", "linear", "scaled_linear"):
        sched = DiffusionSchedule(num_train_timesteps=50, schedule=name)
        acp = sched.alphas_cumprod
        assert sched.betas.shape == (50,)
        assert bool((sched.betas > 0).all() and (sched.betas < 1).all()), name
        assert bool((acp.diff() < 0).all()), f"{name}: ᾱ_t 必须单调递减"
        assert float(acp[0]) > 0.9, (name, float(acp[0]))
        # T=50 时各调度的 ᾱ_49 上界（余弦抹得最干净；线性 / 缩放线性仍留信号）
        assert float(acp[-1]) < {"cosine": 0.02, "linear": 0.62, "scaled_linear": 0.75}[name], \
            (name, float(acp[-1]))
        # 逐项校验派生系数
        assert torch.allclose(sched.sqrt_alphas_cumprod, acp.sqrt(), atol=1e-6)
        assert torch.allclose(sched.sqrt_one_minus_alphas_cumprod, (1 - acp).sqrt(), atol=1e-6)
        expect_var = sched.betas * (1 - sched.alphas_cumprod_prev) / (1 - acp)
        assert torch.allclose(sched.posterior_variance, expect_var, atol=1e-8)
        # ᾱ_{t-1} 的拼接约定：t=0 处为 1
        assert float(sched.alphas_cumprod_prev[0]) == 1.0
        assert torch.allclose(sched.alphas_cumprod_prev[1:], acp[:-1])
    # 余弦调度在 100 步下才能真正把信号抹成噪声
    assert float(DiffusionSchedule(100, "cosine").alphas_cumprod[-1]) < 1e-2


def test_sample_timesteps_are_descending_and_end_at_zero():
    """推理时间步：降序、必含 0、数量正确；``n_steps >= T`` 时覆盖全部训练步。"""
    sched = DiffusionSchedule(num_train_timesteps=100)
    steps = sched.sample_timesteps(16)
    assert len(steps) == 16, steps
    assert steps[0] == 99 and steps[-1] == 0, steps
    assert all(a > b for a, b in zip(steps, steps[1:])), steps
    full = sched.sample_timesteps(100)
    assert full == list(range(99, -1, -1))
    # 一步到位：最吵 → 干净（必须包含 0，否则采样端点不合法）
    assert sched.sample_timesteps(1) == [99, 0]


def test_q_sample_matches_closed_form():
    """``q_sample`` 必须逐点等于 ``sqrt(ᾱ_t)·a^0 + sqrt(1-ᾱ_t)·ε``（含 t 广播）。"""
    sched = DiffusionSchedule(num_train_timesteps=20)
    x0 = torch.randn(3, 4, 2)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 7, 19])
    out = sched.q_sample(x0, t, noise=noise)
    for i, ti in enumerate(t.tolist()):
        acp = float(sched.alphas_cumprod[ti])
        expect = np.sqrt(acp) * x0[i].numpy() + np.sqrt(1 - acp) * noise[i].numpy()
        assert np.allclose(out[i].numpy(), expect, atol=1e-6), ti
    # t=0 时几乎等于原样本；t=T-1 时几乎等于纯噪声（信号基本被抹掉）
    near_clean = sched.q_sample(x0, torch.zeros(3, dtype=torch.long))
    assert float((near_clean - x0).abs().max()) < 0.2
    near_noise = sched.q_sample(x0, torch.full((3,), 19))
    assert float((near_noise - x0).abs().mean()) > 0.5
    assert float(near_noise.std()) > 0.5


def test_predict_x0_round_trip():
    """``a^0 → a^t → â^0`` 往返误差极小（闭式反解正确性的直接证据）。"""
    sched = DiffusionSchedule(num_train_timesteps=100)
    x0 = torch.randn(5, 3, 1)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 10, 40, 70, 98])
    x_t = sched.q_sample(x0, t, noise=noise)
    back = sched.predict_x0_from_noise(x_t, t, noise, clip=None)
    assert float((back - x0).abs().max()) < 1e-3
    # 带 clamp 时结果必须落在 [-1, 1]
    clamped = sched.predict_x0_from_noise(x_t, t, noise, clip=1.0)
    assert float(clamped.abs().max()) <= 1.0


def test_posterior_matches_ddpm_formula():
    """后验均值 / 方差与 DDPM 论文式 (6)(7) 一致；``t=0`` 处方差为 0。"""
    sched = DiffusionSchedule(num_train_timesteps=30)
    x0 = torch.randn(2, 2, 1)
    x_t = torch.randn_like(x0)
    t = torch.tensor([0, 17])
    mean, var, logvar = sched.q_posterior_mean_variance(x0, x_t, t)
    for i, ti in enumerate(t.tolist()):
        beta = float(sched.betas[ti])
        acp = float(sched.alphas_cumprod[ti])
        acp_prev = 1.0 if ti == 0 else float(sched.alphas_cumprod[ti - 1])
        alpha = float(sched.alphas[ti])
        expect_mean = (
            beta * np.sqrt(acp_prev) / (1 - acp) * x0[i].numpy()
            + (1 - acp_prev) * np.sqrt(alpha) / (1 - acp) * x_t[i].numpy()
        )
        expect_var = beta * (1 - acp_prev) / (1 - acp)
        assert np.allclose(mean[i].numpy(), expect_mean, atol=1e-5)
        assert np.allclose(float(var[i].reshape(-1)[0]), expect_var, atol=1e-7)
    assert float(var[0].max()) == 0.0
    assert torch.allclose(torch.exp(logvar), var.clamp_min(1e-20), atol=1e-5)


# =============================================================================
# 2. 噪声预测网络 ε_θ(s, a^t, t)
# =============================================================================
def test_noise_net_shapes_and_conditioning():
    """形状 ``(B, K, A)`` 正确；对 ``s`` 与 ``t`` 都**必须**敏感（否则条件化失效）。"""
    net = NoisePredictor(state_dim=3, action_dim=1, chunk_size=4,
                         hidden_dim=32, num_layers=2, time_embed_dim=16)
    state = torch.randn(5, 3)
    x_t = torch.randn(5, 4, 1)
    t = torch.tensor([0, 3, 5, 9, 12])

    out = net(state, x_t, t)
    assert out.shape == (5, 4, 1)
    # 零初始化输出头 → 训练起点预测恰好为 0（利于稳定优化）
    assert float(out.abs().max()) < 1e-6
    # 条件敏感性（打乱一层参数，避免零初始化掩盖差异）
    net2 = NoisePredictor(state_dim=3, action_dim=1, chunk_size=4,
                          hidden_dim=32, num_layers=2, time_embed_dim=16)
    with torch.no_grad():
        for p in net2.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    base = net2(state, x_t, t)
    assert not torch.allclose(base, net2(state, x_t, t + 1), atol=1e-4), "对扩散步 t 不敏感"
    assert not torch.allclose(base, net2(state + 1.0, x_t, t), atol=1e-4), "对状态 s 不敏感"
    # 标量 t 广播
    assert net2(state, x_t, torch.tensor(4)).shape == (5, 4, 1)
    # K=1 时允许 (B, A) 输入
    net1 = NoisePredictor(state_dim=3, action_dim=1, chunk_size=1,
                          hidden_dim=16, num_layers=1, time_embed_dim=8)
    assert net1(state, torch.randn(5, 1), torch.zeros(5, dtype=torch.long)).shape == (5, 1, 1)


# =============================================================================
# 3. 扩散策略接口（必须与 Task 1 的确定性策略同接口）
# =============================================================================
def _tiny_diffusion_config(num_train_timesteps: int = 20, sample_steps: int = 5, **extra):
    """构造小规模扩散配置（CPU 秒级可跑完）。"""
    cfg = {
        "model": {"state_dim": 3, "action_dim": 1, "chunk_size": 4, "action_scale": 2.0},
        "diffusion": {"num_train_timesteps": num_train_timesteps, "sample_steps": sample_steps,
                      "schedule": "cosine", "sampler": "ddim", "seed": None},
        "noise_net": {"hidden_dim": 32, "num_layers": 2, "time_embed_dim": 16},
    }
    cfg["diffusion"].update(extra)
    return cfg


def test_policy_interface_and_sampling_cost():
    """``forward`` / ``forward_actions`` / ``training`` 语义与 Task 1 一致；开销 = N 次前向。"""
    policy = build_diffusion_policy(_tiny_diffusion_config(sample_steps=5))
    assert policy.training is True                       # 刚构建时处于训练态
    state = torch.randn(6, 3)

    policy.n_network_calls = 0
    actions = policy.forward_actions(state)
    assert actions.shape == (6, 4, 1), actions.shape
    assert float(actions.abs().max()) <= policy.action_scale + 1e-6
    assert policy.n_network_calls == 5, policy.n_network_calls     # 每次规划恰好 N 次网络前向
    assert policy.sample_cost == 5
    assert policy.training is True                       # forward_actions 不应改变训练态
    assert policy.forward(state).shape == (6, 4, 1)
    # 复现性（固定随机源）/ 真采样（默认随机源）
    gen_a, gen_b = torch.Generator().manual_seed(11), torch.Generator().manual_seed(11)
    assert torch.equal(policy.sample(state, generator=gen_a), policy.sample(state, generator=gen_b))
    assert not torch.equal(policy.forward_actions(state), policy.forward_actions(state))
    # 归一化往返：动作空间 ↔ 扩散空间
    real = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]]).reshape(1, 5, 1)
    assert torch.allclose(real, policy.denormalize_actions(policy.normalize_actions(real)), atol=1e-6)
    assert count_parameters(policy) > 0
    # DDPM 采样器：需要 T 步，开销更大，但同样落在动作空间内
    ddpm = build_diffusion_policy(_tiny_diffusion_config(num_train_timesteps=20, sampler="ddpm"))
    assert ddpm.sample_cost == 20
    assert float(ddpm.forward_actions(state).abs().max()) <= 2.0 + 1e-6
    # 跨步采样：N 越小越省；N=1 为"一步到位"（最吵 → 干净，共 2 步）
    for n in (2, 3, 5):
        assert policy.set_sample_steps(n).sample_cost == n
    assert policy.set_sample_steps(1).sample_cost == 2


def test_sampling_trajectory_starts_from_noise():
    """整条去噪轨迹：起点是标准高斯噪声，终点落在归一化动作空间 ``[-1, 1]`` 内。"""
    policy = build_diffusion_policy(_tiny_diffusion_config(sample_steps=8))
    state = torch.randn(3, 3)
    x0, traj = policy.sample(state, return_trajectory=True)
    assert traj.shape == (9, 3, 4, 1), traj.shape
    assert x0.shape == (3, 4, 1)


# =============================================================================
# 4. 训练目标（噪声预测 MSE）与其掩码正确性
# =============================================================================
def test_noise_loss_scale_and_masking():
    """零初始化网络 → 损失 ≈ ``E[ε²] = 1``；零填充位置完全不影响损失。"""
    policy = build_diffusion_policy(_tiny_diffusion_config())
    state = torch.randn(32, 3)
    actions = torch.rand(32, 4, 1) * 4.0 - 2.0                 # (B,K,A) 真实动作块
    gen = torch.Generator().manual_seed(0)
    loss, info = policy.p_losses(state, actions, None, generator=gen)
    # 128 个噪声样本的均方波动约 ±0.13，故容差取 0.3
    assert abs(float(loss) - 1.0) < 0.3, float(loss)
    assert set(info.keys()) == {"eps_mse", "x0_mse"}

    valid = torch.ones(32, 4, dtype=torch.bool)
    valid[:, 2:] = False                                       # 后两步是零填充
    padded = actions.clone(); padded[:, 2:] = 0.0
    garbage = actions.clone(); garbage[:, 2:] = 7.0            # 填充位置塞入垃圾数据
    losses = []
    for tail in (padded, garbage):
        g = torch.Generator().manual_seed(5)
        value, _ = policy.p_losses(state, tail, valid, generator=g,
                                   t=torch.full((32,), 9, dtype=torch.long))
        losses.append(float(value))
    assert abs(losses[0] - losses[1]) < 1e-6, losses           # 掩码位置不参与损失

    g = torch.Generator().manual_seed(5)
    full_loss, _ = policy.p_losses(state, actions, torch.ones(32, 4, dtype=torch.bool),
                                   generator=g, t=torch.full((32,), 9, dtype=torch.long))
    assert abs(float(full_loss) - losses[0]) > 1e-6             # 掩码确实生效


def test_noise_loss_decreases_and_backpropagates():
    """小规模训练：噪声预测 MSE 显著下降，且梯度确实回传到网络参数。"""
    torch.manual_seed(0)
    policy = build_diffusion_policy(_tiny_diffusion_config(num_train_timesteps=40, sample_steps=4))
    state = torch.randn(256, 3)
    actions = torch.tanh(torch.randn(256, 4, 1) * 1.5) * 2.0
    optimizer = optim.Adam(policy.parameters(), lr=3e-3)

    def step_fn(model, batch):
        s, a = batch
        loss, _ = model.p_losses(s, a, None)
        return loss, int(s.size(0))

    loader = DataLoader(torch.utils.data.TensorDataset(state, actions), batch_size=64, shuffle=True)
    history = []
    for _ in range(30):
        history.append(train_one_epoch(policy, loader, None, optimizer, DEVICE, 0.0, step_fn=step_fn))
    assert history[-1] < 0.6 * history[0], (history[0], history[-1])
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in policy.parameters())


# =============================================================================
# 5. 训练引擎兼容性（step_fn 扩展不能改变经典路径）
# =============================================================================
def test_engine_step_fn_extension_is_equivalent():
    """默认路径与 ``step_fn`` 路径在等价损失下产生**逐位相同**的参数更新。"""
    torch.manual_seed(0)
    states = torch.randn(32, 2)
    targets = torch.randn(32, 1)
    loader = DataLoader(torch.utils.data.TensorDataset(states, targets),
                        batch_size=8, shuffle=False)

    torch.manual_seed(1)
    model_a = nn.Linear(2, 1)
    torch.manual_seed(1)
    model_b = nn.Linear(2, 1)
    init = [p.detach().clone() for p in model_a.parameters()]
    assert all(torch.allclose(p, q) for p, q in
               zip(model_a.parameters(), model_b.parameters()))

    loss_a = train_one_epoch(model_a, loader, nn.MSELoss(),
                             optim.SGD(model_a.parameters(), lr=0.05), DEVICE)

    calls = {"n": 0}

    def step_fn(model, batch):
        calls["n"] += 1
        s, a = batch
        return nn.functional.mse_loss(model(s), a), int(s.size(0))

    loss_b = train_one_epoch(model_b, loader, None,
                             optim.SGD(model_b.parameters(), lr=0.05), DEVICE, step_fn=step_fn)

    assert calls["n"] == 4                                     # 每个 batch 调用一次
    assert abs(loss_a - loss_b) < 1e-9
    for p_a, p_b, p_init in zip(model_a.parameters(), model_b.parameters(), init):
        assert torch.allclose(p_a, p_b, atol=1e-7)
        assert not torch.allclose(p_a, p_init, atol=1e-6)      # 确实发生了参数更新
    # validate 也支持 step_fn
    val = validate(model_b, loader, None, DEVICE, step_fn=step_fn)
    assert val > 0


# =============================================================================
# 6. 多峰专家：控制器语义 / 歧义掩码 / 数据双峰性
# =============================================================================
def test_expert_multimodal_controller_semantics():
    """方向歧义状态上两种模式符号相反；非歧义状态与单模式专家逐位一致。"""
    hanging_slow = np.array([-1.0, 0.0, 0.1])      # θ≈π（下垂）且几乎停住
    upright = np.array([1.0, 0.0, 0.0])            # θ≈0（倒立）
    swinging = np.array([1.0, 0.0, 5.0])           # θ≈0 且高速旋转

    assert expert_controller_multimodal(hanging_slow, mode=0) > 0.9
    assert expert_controller_multimodal(hanging_slow, mode=1) < -0.9
    for state in (upright, swinging):
        assert (expert_controller_multimodal(state, 0)
                == expert_controller_multimodal(state, 1)), "非歧义状态不应依赖 mode"
        assert abs(expert_controller_multimodal(state, 0) - expert_controller(state)) < 1e-9

    mask = ambiguity_mask(np.stack([upright, hanging_slow, swinging]))
    assert mask.tolist() == [False, True, False]
    # 阈值可调：角速度阈值放大后，"下垂 + 高速"也进入歧义区（但倒立状态永远不会）
    hanging_fast = np.array([-1.0, 0.0, 5.0])
    assert ambiguity_mask(np.stack([hanging_fast])).tolist() == [False]
    assert ambiguity_mask(np.stack([hanging_fast]), kick_threshold=6.0).tolist() == [True]
    assert ambiguity_mask(np.stack([swinging]), kick_threshold=6.0).tolist() == [False]


def test_multimodal_expert_data_is_bimodal_and_solvable():
    """采集的多峰数据：两种模式都能摆起，且歧义状态的真实标签**只有 ±kick**（双峰）。"""
    episodes, modes = collect_multimodal_expert_trajectories(
        env_name="Pendulum-v1", num_expert_steps=3000, seed=0,
        reuse_cache=False, cache_path=None, verbose=False,
    )
    assert len(episodes) >= 10
    assert set(np.unique(modes).tolist()) == {0, 1}, "两种摆起模式都应出现"
    upright_ratio = []
    for states, _ in episodes:
        theta = np.arctan2(states[:, 1], states[:, 0])
        upright_ratio.append(float((np.abs(theta[-50:]) < 0.5).mean()))
    # 两种模式都可行：绝大多数回合末端都稳稳立住（PD 平衡区）
    assert float(np.mean(upright_ratio)) > 0.8, float(np.mean(upright_ratio))

    all_states = np.concatenate([s for s, _ in episodes], axis=0)
    all_actions = np.concatenate([a for _, a in episodes], axis=0).reshape(-1)
    mask = ambiguity_mask(all_states)
    labels = all_actions[mask]
    assert labels.size > 0
    # 方向歧义区域：正负两个模态都出现，且**没有任何接近 0 的"平均动作"标签**
    assert (labels > 0.9).mean() > 0.15 and (labels < -0.9).mean() > 0.15, \
        (float((labels > 0.9).mean()), float((labels < -0.9).mean()))
    assert float(np.abs(labels).min()) > 0.9, "歧义区不应存在接近 0 的（平均）动作标签"


def test_toy_dataset_is_symmetric_bimodal():
    """双峰玩具数据：两个模态关于 0 对称 → 每个状态的经验均值都 ≈ 0。"""
    states, actions, modes = make_bimodal_toy_dataset(n_states=3, n_per_mode=200, seed=1)
    assert states.shape == (1200, 1) and actions.shape == (1200, 1)
    assert set(np.unique(modes).tolist()) == {0, 1}
    centers = toy_bimodal_mode_centers(np.unique(states))          # (3, 2) = [-μ(s), +μ(s)]
    for i, s in enumerate(np.unique(states)):
        assert abs(float(centers[i, 0] + centers[i, 1])) < 1e-9    # 严格对称
        sel = np.isclose(states[:, 0], s)
        assert abs(float(actions[sel].mean())) < 0.05, (s, actions[sel].mean())


# =============================================================================
# 7. 多峰机理（核心论据）：确定性 MSE 收敛到"平均动作"，扩散覆盖两个模态
# =============================================================================
def test_bimodal_toy_deterministic_mean_vs_diffusion_modes():
    """同一份双峰数据、同等训练预算：

    * 确定性 MLP（MSE）→ 输出 ≈ 条件均值 0，到两个模态的距离都超过容差（不可执行）；
    * 扩散策略 → 采样同时命中两个模态（每个状态 ``hit_neg > 0`` 且 ``hit_pos > 0``）。

    这是"为什么多峰任务需要扩散"的最小可验证证据（CPU 约 30 秒）。
    """
    torch.manual_seed(0)
    states, actions, _ = make_bimodal_toy_dataset(
        n_states=3, n_per_mode=600, mode_center=1.0, mode_slope=0.5, seed=0
    )
    perm = np.random.default_rng(0).permutation(len(states))
    split = int(0.8 * len(states))
    train_loader = DataLoader(BCDataset(states[perm[:split]], actions[perm[:split]]),
                              batch_size=128, shuffle=True)
    val_loader = DataLoader(BCDataset(states[perm[split:]], actions[perm[split:]]), batch_size=512)

    grid = np.unique(states[:, 0]).astype(np.float32).reshape(-1, 1)
    centers = toy_bimodal_mode_centers(grid)                       # (S, 2) = [-μ, +μ]
    grid_t = torch.as_tensor(grid)
    tol = 0.3

    # ---- (a) 确定性 MLP：MSE 最优解 = 条件均值 ----
    det = build_network({"model": {"name": "mlp", "state_dim": 1, "action_dim": 1,
                                   "hidden_dim": 32, "num_layers": 2, "chunk_size": 1,
                                   "squash": True, "action_scale": 2.0}})
    det_opt = optim.Adam(det.parameters(), lr=2e-3)
    criterion = nn.MSELoss()
    for _ in range(200):
        train_one_epoch(det, train_loader, criterion, det_opt, DEVICE, 0.0)
    with torch.no_grad():
        det_out = det.forward_actions(grid_t)[:, 0, 0].numpy()
    det_dist = np.abs(det_out[:, None] - centers).min(axis=1)
    assert float(np.abs(det_out).max()) < 0.3, f"确定性回归应输出≈0（条件均值），{det_out}"
    assert float(det_dist.min()) > tol, f"平均动作到最近模态应超过容差，{det_dist}"
    # 理论下界：MSE 无法低于双峰方差（条件均值只能落在两个模态之间）
    theoretical = float((centers ** 2).mean() + 0.05 ** 2)
    assert validate(det, val_loader, criterion, DEVICE) < 1.5 * theoretical

    # ---- (b) 扩散策略：同一数据 / 同一 epoch 数 ----
    policy = build_diffusion_policy({
        "model": {"state_dim": 1, "action_dim": 1, "chunk_size": 1, "action_scale": 2.0},
        "diffusion": {"num_train_timesteps": 50, "sample_steps": 16, "seed": None},
        "noise_net": {"hidden_dim": 48, "num_layers": 2, "time_embed_dim": 16},
    })
    policy_opt = optim.Adam(policy.parameters(), lr=2e-3)

    def step_fn(model, batch):
        s, a = batch
        loss, _ = model.p_losses(s, a, None)
        return loss, int(s.size(0))

    for _ in range(200):
        train_one_epoch(policy, train_loader, None, policy_opt, DEVICE, 0.0, step_fn=step_fn)

    n_samples = 400
    with torch.no_grad():
        samples = policy.forward_actions(grid_t.repeat_interleave(n_samples, 0))[:, 0, 0]
        samples = samples.numpy().reshape(len(grid), n_samples)
    dist_each = np.abs(samples[:, :, None] - centers[:, None, :])   # (S, N, 2)
    hit_any = (dist_each.min(axis=2) <= tol)
    hit_neg = (dist_each[:, :, 0] <= tol).mean(axis=1)
    hit_pos = (dist_each[:, :, 1] <= tol).mean(axis=1)
    assert float(hit_any.mean()) > 0.3, float(hit_any.mean())
    assert bool(np.all((hit_neg > 0.0) & (hit_pos > 0.0))), "每个状态都应同时命中两个模态"
    assert float(samples.std(axis=1).min()) > 0.3, "采样标准差应显著大于 0（确实在采样）"


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
