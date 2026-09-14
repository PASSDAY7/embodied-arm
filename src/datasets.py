"""数据加载与增强模块。

包含三部分：
1. ``expert_controller``   —— 倒立摆专家控制器（能量摆起 + PD 平衡）
2. ``collect_expert_data`` —— 采集专家轨迹（支持本地缓存）并按比例切分
3. ``BCDataset`` / ``build_dataloaders`` —— 行为克隆数据集与增强、DataLoader 构建
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:  # gymnasium 为可选依赖，仅数据采集时需要
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None


# =============================================================================
# 1. 专家控制器（能量摆起 + PD 平衡）
# =============================================================================
def expert_controller(state: np.ndarray) -> float:
    """倒立摆专家控制器。

    * 偏离倒立较远时：能量塑形控制，将摆从下垂位置摆起；
    * 接近倒立位置时：PD 反馈控制，保持平衡。

    Args:
        state: ``[cosθ, sinθ, θ_dot]``

    Returns:
        力矩 τ，范围 ``[-2.0, 2.0]``。
    """
    cos_theta, sin_theta, theta_dot = state
    theta = np.arctan2(sin_theta, cos_theta)  # 还原角度 θ ∈ [-π, π]

    # 倒立摆系统参数（与 Pendulum-v1 一致）
    g, mass, length = 10.0, 1.0, 1.0
    inertia = mass * length ** 2
    e_target = mass * g * length  # 倒立位置总机械能

    kinetic = 0.5 * inertia * theta_dot ** 2
    potential = mass * g * length * cos_theta
    e_total = kinetic + potential

    if abs(theta) < 0.5:  # 接近倒立，PD 平衡控制
        kp, kd = 12.0, 1.5
        tau = -kp * theta - kd * theta_dot
    else:  # 能量塑形摆起
        k_energy = 2.5
        tau = k_energy * theta_dot * np.sign(e_target - e_total)

    return float(np.clip(tau, -2.0, 2.0))


# =============================================================================
# 2. 专家数据采集与缓存
# =============================================================================
def _collect_raw(
    env_name: str = "Pendulum-v1",
    num_expert_steps: Optional[int] = None,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """与环境交互采集专家轨迹，返回 (states, actions)。

    两种采集模式（``num_expert_steps`` 优先）：
    * ``num_expert_steps`` 给定：连续采集回合直到步数达到该值，并截断到精确步数；
    * 否则按 ``n_episodes`` 采集固定回合数。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法采集专家数据。请先 `pip install gymnasium`。")

    env = gym.make(env_name)
    states_list, actions_list = [], []
    ep = 0

    while True:
        if num_expert_steps is not None:
            if len(states_list) >= num_expert_steps:
                break
        elif ep >= n_episodes:
            break

        state, _ = env.reset(seed=None if seed is None else seed + ep)
        done = False
        while not done:
            action = expert_controller(state)
            states_list.append(state)
            actions_list.append(action)
            state, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            done = terminated or truncated
        ep += 1

    env.close()

    if num_expert_steps is not None:
        states_list = states_list[:num_expert_steps]
        actions_list = actions_list[:num_expert_steps]

    states = np.asarray(states_list, dtype=np.float32)                       # (N, 3)
    actions = np.asarray(actions_list, dtype=np.float32).reshape(-1, 1)      # (N, 1)
    return states, actions


def collect_expert_data(
    env_name: str = "Pendulum-v1",
    num_expert_steps: Optional[int] = 10000,
    train_ratio: float = 0.8,
    n_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    reuse_cache: bool = True,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """采集（或从缓存加载）专家数据，并按 ``train_ratio`` 切分训练 / 验证集。

    Args:
        env_name: Gymnasium 环境名。
        num_expert_steps: 目标专家数据总步数（推荐，按步数采集并精确截断）。
        train_ratio: 训练集占比，其余作为验证集。
        n_episodes: 若未指定 ``num_expert_steps``，则按固定回合数采集。
        seed: 随机种子。
        reuse_cache: 若缓存存在是否复用。
        cache_path: 缓存文件路径（.npz）。
        verbose: 是否打印过程信息。

    Returns:
        ``(X_train, y_train, X_val, y_val)``
    """
    states = actions = None

    if reuse_cache and cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        states, actions = data["states"], data["actions"]
        if verbose:
            print(f"[data] 从缓存加载专家数据: {cache_path} ({len(states)} 步)")
    else:
        if verbose:
            if num_expert_steps is not None:
                print(f"[data] 采集专家数据: {env_name}, 目标 {num_expert_steps} 步 ...")
            else:
                print(f"[data] 采集专家数据: {env_name}, {n_episodes} 回合 ...")
        states, actions = _collect_raw(
            env_name=env_name,
            num_expert_steps=num_expert_steps,
            n_episodes=n_episodes if n_episodes is not None else 50,
            seed=seed,
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, states=states, actions=actions)
            if verbose:
                print(f"[data] 专家数据已缓存至: {cache_path}")

    if verbose:
        print(f"[data] 专家数据收集完成：共 {len(states)} 步")

    # 打乱并按 train_ratio 切分
    train_ratio = float(np.clip(train_ratio, 0.0, 1.0))
    rng = np.random.default_rng(seed if seed is not None else 42)
    indices = rng.permutation(len(states))
    split_idx = int(train_ratio * len(states))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    return (
        states[train_idx], actions[train_idx],
        states[val_idx], actions[val_idx],
    )


# =============================================================================
# 3. 数据集与增强
# =============================================================================
class BCDataset(Dataset):
    """行为克隆数据集，支持对状态 / 动作施加高斯噪声进行数据增强。"""

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        augment: bool = False,
        state_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.augment = augment
        self.state_noise_std = float(state_noise_std)
        self.action_noise_std = float(action_noise_std)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self.states[idx]
        action = self.actions[idx]

        if self.augment:
            if self.state_noise_std > 0:
                noise = torch.randn_like(state) * self.state_noise_std
                state = state + noise
                # 保持单位圆约束: cos²+sin² ≈ 1（若状态为 [cos, sin, ...]）
                if state.numel() >= 2:
                    norm = torch.sqrt(state[0] ** 2 + state[1] ** 2).clamp_min(1e-6)
                    state = state.clone()
                    state[0] = state[0] / norm
                    state[1] = state[1] / norm
            if self.action_noise_std > 0:
                action = action + torch.randn_like(action) * self.action_noise_std

        return state, action


def build_dataloaders(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 0,
    augment: bool = False,
    state_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """构建训练 / 验证 DataLoader。

    训练集可选增强，验证集始终不做任何变换。
    """
    train_dataset = BCDataset(
        X_train, y_train,
        augment=augment,
        state_noise_std=state_noise_std,
        action_noise_std=action_noise_std,
        seed=seed,
    )
    val_dataset = BCDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


# =============================================================================
# 4. 闭环评估
# =============================================================================
def evaluate_closed_loop(
    policy,
    device: torch.device,
    env_name: str = "Pendulum-v1",
    n_episodes: int = 20,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """将策略放入环境闭环运行，返回平均回报等统计量。"""
    if gym is None:
        raise ImportError("未安装 gymnasium，无法进行闭环评估。")

    env = gym.make(env_name, render_mode="human" if render else None)
    episode_returns = []

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            state, _ = env.reset(seed=None if seed is None else seed + ep)
            ep_return = 0.0
            done = False
            while not done:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(state_t).squeeze(0).cpu().numpy()
                state, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
            episode_returns.append(float(ep_return))

    env.close()
    if was_training:
        policy.train()

    returns = np.asarray(episode_returns, dtype=np.float64)
    return {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }
