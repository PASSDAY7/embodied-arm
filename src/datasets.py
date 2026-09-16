"""数据加载与增强模块。

包含五部分：
1. ``expert_controller``            —— 倒立摆专家控制器（能量摆起 + PD 平衡）
1.5 ``expert_controller_multimodal`` —— **多峰**专家（起步方向由 mode 决定，W4 Task 2 用）
2. ``collect_expert_data``          —— 采集专家轨迹（支持本地缓存）并按比例切分
2.5 ``collect_expert_trajectories`` —— **保留回合边界**的轨迹采集（动作块切分前提）
2.6 ``collect_multimodal_expert_trajectories`` —— 多峰专家轨迹采集（扩散策略训练前提）
3. ``BCDataset`` / ``build_dataloaders`` —— 行为克隆数据集与增强、DataLoader 构建
4. ``ChunkingBCDataset`` / ``build_chunking_dataloaders`` —— 动作块行为克隆（W4 Task 1）
4.5 ``make_bimodal_toy_dataset``   —— 双峰玩具任务（多峰机理的最小可验证例子）
5. ``evaluate_closed_loop``         —— 闭环评估（单步 / 动作块 + EMA 平滑执行）
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

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


def expert_controller_multimodal(
    state: np.ndarray,
    mode: int = 0,
    kick_threshold: float = 1.0,
    hang_margin: float = 0.5,
    kick_torque: float = 2.0,
) -> float:
    """**多峰**专家控制器：同一状态存在两条同样可行的摆起分支（W4 Task 2 专用）。

    与 :func:`expert_controller` 的唯一区别是"起步阶段的方向歧义"：当摆**停在下垂
    位置附近**（``|θ| > π - hang_margin``）且**角速度很小**（``|θ̇| < kick_threshold``）
    时，能量泵控制律 ``τ = k·θ̇·sign(ΔE)`` 退化到 ``τ ≈ 0``，"往左摆"与"往右摆"
    两条路线都成立 —— 此时方向由 ``mode`` 决定：

    * ``mode = 0`` → ``τ = +kick_torque``（逆时针摆起）
    * ``mode = 1`` → ``τ = -kick_torque``（顺时针摆起）

    其余情形与单模式专家完全一致（倒立附近 PD 平衡、其余状态能量塑形），因此：
    **同一个状态在两种模式下都可能出现，且对应动作符号相反** —— 这正是条件分布
    ``p(a | s)`` 的**多峰（bimodal）**来源，也是确定性 MSE 回归"取均值 → 输出 ≈ 0 →
    卡在下垂位置"的直接根源。

    Args:
        state: ``[cosθ, sinθ, θ̇]``。
        mode: 摆起方向模式（``0`` / ``1``）。
        kick_threshold: 判定"几乎停住"的角速度阈值。
        hang_margin: 距离下垂位置的角度窗口（弧度），窗口内且低速才触发方向选择。
        kick_torque: 起步力矩幅值（Pendulum 动作上限为 2.0）。

    Returns:
        力矩 τ，范围 ``[-2.0, 2.0]``。
    """
    cos_theta, sin_theta, theta_dot = state
    theta = float(np.arctan2(sin_theta, cos_theta))       # 还原角度 θ ∈ [-π, π]

    # 倒立摆系统参数（与 Pendulum-v1 一致）
    g, mass, length = 10.0, 1.0, 1.0
    inertia = mass * length ** 2
    e_target = mass * g * length                          # 倒立位置总机械能
    e_total = 0.5 * inertia * theta_dot ** 2 + mass * g * length * cos_theta

    if abs(theta) < 0.5:                                  # 接近倒立 → PD 平衡（无歧义）
        tau = -12.0 * theta - 1.5 * theta_dot
    elif abs(theta) > np.pi - hang_margin and abs(theta_dot) < kick_threshold:
        tau = float(kick_torque) if int(mode) % 2 == 0 else -float(kick_torque)
    else:                                                 # 能量塑形摆起（无歧义）
        tau = 2.5 * theta_dot * np.sign(e_target - e_total)

    return float(np.clip(tau, -2.0, 2.0))


def ambiguity_mask(
    states: np.ndarray,
    kick_threshold: float = 1.0,
    hang_margin: float = 0.5,
    near_upright: float = 0.5,
) -> np.ndarray:
    """标记 :func:`expert_controller_multimodal` 的**方向歧义**状态（多峰区域）。

    判定条件（与控制器内部分支完全一致）::

        |θ| ≥ near_upright  且  |θ| > π - hang_margin  且  |θ̇| < kick_threshold

    这些状态就是 ``p(a | s)`` 出现双峰的地方；扩散策略的"多峰捕获"分析、以及
    确定性 MSE 策略"输出平均动作"的诊断都以该区域为对象。

    Args:
        states: ``(N, 3)`` 状态数组 ``[cosθ, sinθ, θ̇]``。

    Returns:
        ``(N,)`` bool 掩码。
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] < 3:
        raise ValueError(f"states 形状应为 (N, 3)，当前 {states.shape}")
    theta = np.arctan2(states[:, 1], states[:, 0])
    return (
        (np.abs(theta) >= float(near_upright))
        & (np.abs(theta) > np.pi - float(hang_margin))
        & (np.abs(states[:, 2]) < float(kick_threshold))
    )


# =============================================================================
# 2. 专家数据采集与缓存
# =============================================================================
def _collect_episodes(
    env_name: str = "Pendulum-v1",
    num_expert_steps: Optional[int] = None,
    n_episodes: int = 50,
    seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """按**回合**采集专家轨迹，返回 ``[(states_i, actions_i), ...]``。

    与 ``_collect_raw`` 的唯一区别：保留**回合边界**，且回合内状态/动作保持时间
    顺序（不拆分、不打乱）——这是动作块（action chunk）滑窗切分的必要前提。

    两种采集模式（``num_expert_steps`` 优先）：
    * ``num_expert_steps`` 给定：连续采集回合直到步数达到该值，最后一条轨迹截断到精确步数；
    * 否则按 ``n_episodes`` 采集固定回合数。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法采集专家数据。请先 `pip install gymnasium`。")

    env = gym.make(env_name)
    episodes: List[Tuple[np.ndarray, np.ndarray]] = []
    total, ep = 0, 0

    while True:
        if num_expert_steps is not None:
            if total >= num_expert_steps:
                break
        elif ep >= n_episodes:
            break

        state, _ = env.reset(seed=None if seed is None else seed + ep)
        states_list, actions_list = [], []
        done = False
        while not done:
            action = expert_controller(state)
            states_list.append(state)
            actions_list.append(action)
            state, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            done = terminated or truncated
        ep += 1
        total += len(states_list)
        episodes.append((
            np.asarray(states_list, dtype=np.float32),                       # (T, state_dim)
            np.asarray(actions_list, dtype=np.float32).reshape(-1, 1),       # (T, 1)
        ))

    env.close()

    if num_expert_steps is not None and episodes:
        # 截断最后一条轨迹，使总步数精确等于 num_expert_steps
        overflow = total - int(num_expert_steps)
        if overflow > 0:
            states_last, actions_last = episodes[-1]
            episodes[-1] = (states_last[: len(states_last) - overflow],
                            actions_last[: len(actions_last) - overflow])

    return episodes


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

    本函数是 ``_collect_episodes`` 的**扁平化**视图（所有回合首尾相接），
    与旧版实现输出完全一致（逐位相同），供单步 BC 流程使用。
    """
    episodes = _collect_episodes(
        env_name=env_name,
        num_expert_steps=num_expert_steps,
        n_episodes=n_episodes,
        seed=seed,
    )
    states = np.concatenate([s for s, _ in episodes], axis=0)                # (N, 3)
    actions = np.concatenate([a for _, a in episodes], axis=0)               # (N, 1)
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


def collect_expert_trajectories(
    env_name: str = "Pendulum-v1",
    num_expert_steps: int = 10000,
    seed: Optional[int] = None,
    reuse_cache: bool = True,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """采集（或从缓存加载）**保留回合边界**的专家轨迹，供动作块行为克隆使用。

    缓存格式（``.npz``，与单步 BC 的扁平缓存相互独立）::

        states:          (N, state_dim)  所有回合首尾相接
        actions:         (N, action_dim)
        episode_lengths: (E,)            每条轨迹的步数（用于还原回合边界）

    Args:
        env_name: Gymnasium 环境名。
        num_expert_steps: 目标专家总步数（最后一条轨迹会被截断到精确步数）。
        seed: 随机种子（决定回合初始状态序列）。
        reuse_cache: 缓存存在时是否复用。
        cache_path: 缓存文件路径（.npz）。
        verbose: 是否打印过程信息。

    Returns:
        ``[(states_i, actions_i), ...]``，每条轨迹内状态/动作时间连续，
        窗口切分时**不会跨越轨迹边界**。
    """
    if reuse_cache and cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        states, actions = data["states"], data["actions"]
        lengths = data["episode_lengths"]
        if verbose:
            print(f"[data] 从缓存加载专家轨迹: {cache_path} "
                  f"({len(lengths)} 条轨迹 / {len(states)} 步)")
    else:
        episodes = _collect_episodes(
            env_name=env_name,
            num_expert_steps=num_expert_steps,
            seed=seed,
        )
        states = np.concatenate([s for s, _ in episodes], axis=0)
        actions = np.concatenate([a for _, a in episodes], axis=0)
        lengths = np.asarray([len(s) for s, _ in episodes], dtype=np.int64)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(
                cache_path, states=states, actions=actions, episode_lengths=lengths
            )
            if verbose:
                print(f"[data] 专家轨迹已缓存至: {cache_path}")

    if verbose:
        print(f"[data] 专家轨迹收集完成：{len(lengths)} 条轨迹，共 {len(states)} 步")

    # 按回合边界还原为轨迹列表（窗口切分时严禁跨界）
    episodes, offset = [], 0
    for length in lengths:
        length = int(length)
        episodes.append((states[offset: offset + length], actions[offset: offset + length]))
        offset += length
    return episodes


def _collect_episodes_multimodal(
    env_name: str = "Pendulum-v1",
    num_expert_steps: int = 10000,
    seed: Optional[int] = None,
    mode_policy: str = "random",
    expert_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """按回合采集**多峰专家**轨迹，每个回合随机/交替选择一种摆起模式。

    Args:
        mode_policy: ``random``（按种子随机，推荐：模式与回合初始状态独立）
            | ``alternate``（奇偶回合交替，便于人工核对）。
        expert_kwargs: 透传给 :func:`expert_controller_multimodal` 的参数
            （``kick_threshold`` / ``hang_margin`` / ``kick_torque``）。

    Returns:
        ``(episodes, modes)``：``modes[i]`` 为第 ``i`` 条轨迹所用的模式编号。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法采集专家数据。请先 `pip install gymnasium`。")

    kwargs = dict(expert_kwargs or {})
    rng = np.random.default_rng(seed if seed is not None else 0)
    policy = str(mode_policy).lower()

    env = gym.make(env_name)
    episodes: List[Tuple[np.ndarray, np.ndarray]] = []
    modes: List[int] = []
    total, ep = 0, 0

    while total < int(num_expert_steps):
        if policy == "alternate":
            mode = ep % 2
        elif policy == "random":
            mode = int(rng.integers(2))
        else:
            raise ValueError(f"未知 mode_policy '{mode_policy}'，可选: random | alternate")

        state, _ = env.reset(seed=None if seed is None else seed + ep)
        states_list, actions_list = [], []
        done = False
        while not done:
            action = expert_controller_multimodal(state, mode=mode, **kwargs)
            states_list.append(state)
            actions_list.append(action)
            state, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            done = terminated or truncated
        ep += 1
        total += len(states_list)
        modes.append(int(mode))
        episodes.append((
            np.asarray(states_list, dtype=np.float32),
            np.asarray(actions_list, dtype=np.float32).reshape(-1, 1),
        ))

    env.close()

    # 截断最后一条轨迹，使总步数精确等于 num_expert_steps（与单模式采集一致）
    overflow = total - int(num_expert_steps)
    if overflow > 0 and episodes:
        states_last, actions_last = episodes[-1]
        episodes[-1] = (states_last[: len(states_last) - overflow],
                        actions_last[: len(actions_last) - overflow])

    return episodes, np.asarray(modes, dtype=np.int64)


def collect_multimodal_expert_trajectories(
    env_name: str = "Pendulum-v1",
    num_expert_steps: int = 10000,
    seed: Optional[int] = None,
    reuse_cache: bool = True,
    cache_path: Optional[str] = None,
    mode_policy: str = "random",
    expert_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """采集（或从缓存加载）**多峰专家**轨迹，供扩散策略训练。

    缓存格式（``.npz``，与单模式缓存相互独立）::

        states:          (N, state_dim)
        actions:         (N, action_dim)
        episode_lengths: (E,)
        modes:           (E,)            每个回合的摆起模式（0/1）

    Returns:
        ``(episodes, modes)``；轨迹按回合边界还原，窗口切分不会跨回合。
    """
    if reuse_cache and cache_path and os.path.isfile(cache_path):
        data = np.load(cache_path)
        states, actions = data["states"], data["actions"]
        lengths, modes = data["episode_lengths"], data["modes"]
        if verbose:
            print(f"[data] 从缓存加载多峰专家轨迹: {cache_path} "
                  f"({len(lengths)} 条轨迹 / {len(states)} 步 / 模式分布 "
                  f"{np.bincount(modes, minlength=2).tolist()})")
    else:
        episodes, modes = _collect_episodes_multimodal(
            env_name=env_name,
            num_expert_steps=num_expert_steps,
            seed=seed,
            mode_policy=mode_policy,
            expert_kwargs=expert_kwargs,
        )
        states = np.concatenate([s for s, _ in episodes], axis=0)
        actions = np.concatenate([a for _, a in episodes], axis=0)
        lengths = np.asarray([len(s) for s, _ in episodes], dtype=np.int64)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(
                cache_path, states=states, actions=actions,
                episode_lengths=lengths, modes=modes,
            )
            if verbose:
                print(f"[data] 多峰专家轨迹已缓存至: {cache_path}")

    if verbose:
        print(f"[data] 多峰专家轨迹收集完成：{len(lengths)} 条轨迹，共 {len(states)} 步")

    episodes, offset = [], 0
    for length in lengths:
        length = int(length)
        episodes.append((states[offset: offset + length], actions[offset: offset + length]))
        offset += length
    return episodes, np.asarray(modes, dtype=np.int64)



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
# 4. 动作块（Action Chunking）数据集 —— W4
# =============================================================================
def build_action_chunks(
    actions: np.ndarray,
    chunk_size: int,
    episode_lengths: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """把动作序列滑窗切分为 ``K`` 步动作块，轨迹末端不足 ``K`` 步时**零填充**。

    对每条轨迹的每个时刻 ``t``（``0 <= t < T``）生成一个窗口::

        chunk[t] = [a_t, a_{t+1}, ..., a_{t+K-1}]，其中越界位置补 0

    窗口**绝不跨越轨迹边界**：若 ``a_t`` 属于第 ``i`` 条轨迹，则窗口只从第
    ``i`` 条轨迹内部取值，不足部分补零（等价于该轨迹视为已终止）。

    Args:
        actions: ``(N, action_dim)`` 动作序列（多条轨迹首尾相接）。
        chunk_size: 动作块长度 ``K``。
        episode_lengths: 每条轨迹的步数；``None`` 表示视为单条长轨迹。

    Returns:
        ``(chunks, valid)``：

        * ``chunks`` —— ``(N, K, action_dim)`` float32 动作块；
        * ``valid``  —— ``(N, K)`` bool 掩码，``False`` 处为**零填充**。
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    n_steps, action_dim = actions.shape
    k = max(1, int(chunk_size))

    chunks = np.zeros((n_steps, k, action_dim), dtype=np.float32)
    valid = np.zeros((n_steps, k), dtype=bool)

    lengths = list(episode_lengths) if episode_lengths is not None else [n_steps]
    offset = 0
    for length in lengths:
        length = int(length)
        if length <= 0:
            continue
        segment = actions[offset: offset + length]
        for t in range(length):
            available = min(k, length - t)                 # 本窗口真实可用的步数
            chunks[offset + t, :available] = segment[t: t + available]
            valid[offset + t, :available] = True
        offset += length

    return chunks, valid


class ChunkingBCDataset(Dataset):
    """动作块行为克隆数据集（Action Chunking BC）。

    每个样本::

        输入 : s_t                      形状 (state_dim,)
        标签 : [a_t, ..., a_{t+K-1}]     形状 (K, action_dim)

    动作块由 :func:`build_action_chunks` 一次性预计算（数据量很小，比每个 epoch
    现场切窗更快），增强逻辑与 :class:`BCDataset` **完全一致**，从而单步 BC 与
    动作块 BC 的数据管线严格可比。

    注：``chunk[0]``（推理时真正被执行的第一个动作）的标签永远来自真实专家动作
    （``t < T`` 恒成立），零填充只影响块尾，因此不会污染策略的即时响应。
    """

    def __init__(
        self,
        states: np.ndarray,
        action_chunks: np.ndarray,
        valid: Optional[np.ndarray] = None,
        augment: bool = False,
        state_noise_std: float = 0.0,
        action_noise_std: float = 0.0,
        seed: int = 0,
        return_valid: bool = False,
    ) -> None:
        self.states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        self.action_chunks = torch.as_tensor(np.asarray(action_chunks), dtype=torch.float32)
        self.valid = None if valid is None else np.asarray(valid, dtype=bool)
        self.chunk_size = int(self.action_chunks.shape[1])
        self.augment = augment
        self.state_noise_std = float(state_noise_std)
        self.action_noise_std = float(action_noise_std)
        self._rng = random.Random(seed)
        # ``True`` 时额外返回 ``(K,)`` 有效掩码，供扩散策略的掩码损失使用（W4 Task 2）
        self.return_valid = bool(return_valid)

    def __len__(self) -> int:
        return self.states.shape[0]

    @property
    def padding_ratio(self) -> float:
        """零填充位置占比（仅用于监控，不参与损失计算）。"""
        if self.valid is None or self.valid.size == 0:
            return 0.0
        return float(1.0 - self.valid.mean())

    def __getitem__(self, idx: int):
        state = self.states[idx]
        chunk = self.action_chunks[idx]

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
                chunk = chunk + torch.randn_like(chunk) * self.action_noise_std

        if not self.return_valid:
            return state, chunk
        valid = (torch.ones(self.chunk_size, dtype=torch.bool) if self.valid is None
                 else torch.as_tensor(self.valid[idx], dtype=torch.bool))
        return state, chunk, valid


def build_chunking_dataloaders(
    episodes: List[Tuple[np.ndarray, np.ndarray]],
    chunk_size: int = 8,
    train_ratio: float = 0.8,
    batch_size: int = 256,
    num_workers: int = 0,
    augment: bool = False,
    state_noise_std: float = 0.0,
    action_noise_std: float = 0.0,
    seed: int = 0,
    return_valid: bool = False,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    """由专家轨迹构建动作块训练 / 验证 DataLoader。

    切分方式与单步 BC **完全对齐**（同为"按状态样本随机 ``train_ratio`` 切分"）：
    先按时间顺序生成全部 ``N`` 个 ``(s_t, chunk_t)`` 样本，再用
    ``numpy.random.default_rng(seed).permutation(N)`` 打乱切分。因此单步 BC 与
    动作块 BC 的训练/验证**状态集合逐样本一致**，唯一差别是标签由 ``a_t`` 变为
    ``[a_t, ..., a_{t+K-1}]`` —— 对比实验因此干净可控。

    Args:
        return_valid: ``True`` 时 batch 为 ``(state, chunk, valid)`` 三元组
            （``valid`` 形状 ``(B, K)``），供扩散策略的掩码损失使用。

    Returns:
        ``(train_loader, val_loader, info)``，``info`` 含样本数与零填充比例。
    """
    states = np.concatenate([s for s, _ in episodes], axis=0)
    actions = np.concatenate([a for _, a in episodes], axis=0)
    lengths = [int(len(s)) for s, _ in episodes]

    chunks, valid = build_action_chunks(actions, chunk_size, episode_lengths=lengths)
    n_samples = states.shape[0]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    split_idx = int(np.clip(train_ratio, 0.0, 1.0) * n_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_dataset = ChunkingBCDataset(
        states[train_idx], chunks[train_idx], valid[train_idx],
        augment=augment,
        state_noise_std=state_noise_std,
        action_noise_std=action_noise_std,
        seed=seed,
        return_valid=return_valid,
    )
    val_dataset = ChunkingBCDataset(
        states[val_idx], chunks[val_idx], valid[val_idx], augment=False,
        return_valid=return_valid,
    )

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

    info = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_trajectories": int(len(episodes)),
        "chunk_size": int(chunk_size),
        "padding_ratio": float(1.0 - valid.mean()) if valid.size else 0.0,
        "train_padding_ratio": train_dataset.padding_ratio,
    }
    return train_loader, val_loader, info


# =============================================================================
# 4.5 双峰玩具任务（多峰机理的最小可验证例子）—— W4 Task 2
# =============================================================================
def toy_bimodal_mode_centers(
    states: np.ndarray, mode_center: float = 1.0, mode_slope: float = 0.5
) -> np.ndarray:
    """双峰玩具任务的两个**模态中心** ``μ(s) = mode_center + mode_slope·s``。

    Args:
        states: ``(N, 1)`` 或 ``(N,)`` 状态。

    Returns:
        ``(N, 2)``：``[:, 0] = -μ(s)``、``[:, 1] = +μ(s)``（两个模态中心）。
    """
    s = np.asarray(states, dtype=np.float64).reshape(-1)
    mu = float(mode_center) + float(mode_slope) * s
    return np.stack([-mu, mu], axis=1)


def make_bimodal_toy_dataset(
    n_states: int = 9,
    n_per_mode: int = 2000,
    mode_center: float = 1.0,
    mode_slope: float = 0.5,
    mode_noise_std: float = 0.05,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造一维**双峰**条件分布 ``p(a | s)`` 的最小可验证数据集。

    数据生成：状态 ``s`` 取 ``[-1, 1]`` 上的等距网格；对每个状态分别从两个模态
    各采 ``n_per_mode`` 个样本::

        a | s  ~  0.5 · N(-μ(s), σ²)  +  0.5 · N(+μ(s), σ²),   μ(s) = 1.0 + 0.5·s

    由于两个模态**关于 0 对称**，任何确定性回归器的最小 MSE 解都是 ``E[a|s] = 0``；
    而 ``a = 0`` 既不是模态 A 也不是模态 B，是"两个可行动作之间的空地" —— 这正是
    复杂操作中"确定性 MSE 学出撞墙的平均动作"的最小复现。扩散策略则可以从
    ``p(a|s)`` 中采样，按比例命中两个模态。

    Args:
        n_states: 状态网格点数。
        n_per_mode: 每个状态、每个模态的样本数。
        mode_noise_std: 模态内部的高斯噪声（0 = 完全双点分布，对比最尖锐）。
        seed: 随机种子。

    Returns:
        ``(states (N, 1), actions (N, 1), mode_ids (N,))``；``mode_ids ∈ {0, 1}``。
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1.0, 1.0, int(n_states), dtype=np.float32).reshape(-1, 1)
    centers = toy_bimodal_mode_centers(grid, mode_center, mode_slope)

    states, actions, modes = [], [], []
    for i, s in enumerate(grid):
        for mode in (0, 1):
            if mode == 0:
                base = np.full((n_per_mode, 1), centers[i, 0], dtype=np.float32)
            else:
                base = np.full((n_per_mode, 1), centers[i, 1], dtype=np.float32)
            if mode_noise_std > 0:
                base = base + rng.normal(0.0, mode_noise_std, size=base.shape).astype(np.float32)
            states.append(np.repeat(s.reshape(1, 1), n_per_mode, axis=0))
            actions.append(base.astype(np.float32))
            modes.append(np.full(n_per_mode, mode, dtype=np.int64))

    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)
    modes = np.concatenate(modes, axis=0)
    perm = rng.permutation(len(states))
    return states[perm], actions[perm], modes[perm]


# =============================================================================
# 5. 闭环评估
# =============================================================================
def _rollout_closed_loop(
    policy,
    device: torch.device,
    env_name: str = "Pendulum-v1",
    n_episodes: int = 20,
    chunk_size: int = 1,
    execution: str = "chunk",
    ema_alpha: float = 0.5,
    replan_every: int = 0,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """通用闭环 rollout：单步策略与动作块策略共用同一套执行逻辑。

    执行调度::

        每 `execute_steps` 步做一次前向推理 → 得到未来 K 步动作块 → 依次执行

    * ``execution="chunk"``：**开环**执行整块预测（一次推理打出 K 步动作）；
    * ``execution="ema"``  ：把新预测动作块与上一次规划在**同一绝对时刻**上做指数移动
      平均（EMA，ACT temporal ensembling 的简化形式）再执行，抑制块与块之间的动作跳变。
      注意：只有 ``execute_steps < K``（相邻规划在时间上**有重叠**）时平滑才会生效；
      若 ``execute_steps == K``（默认，无重叠），EMA 与开环执行**逐位一致**。

    当 ``chunk_size == 1`` 且 ``execution="chunk"`` 时，本函数与"一步一步前向"
    的传统单步 BC **完全等价**（同一时刻、同一动作、同一数值）。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法进行闭环评估。")

    k = max(1, int(chunk_size))
    # 每次前向执行多少步：默认 = K（即"每隔 K 步推理一次"），且不允许超过 K
    execute_steps = k if int(replan_every) <= 0 else min(int(replan_every), k)
    execution = str(execution).lower()
    if execution not in ("chunk", "ema"):
        raise ValueError(f"未知执行模式 '{execution}'，可选: chunk | ema")
    alpha = float(np.clip(ema_alpha, 0.0, 1.0))

    env = gym.make(env_name, render_mode="human" if render else None)
    episode_returns = []
    n_forward = 0

    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            state, _ = env.reset(seed=None if seed is None else seed + ep)
            ep_return = 0.0
            done = False
            plan = None                          # (K, action_dim) 当前规划（chunk 模式）
            future: Dict[int, np.ndarray] = {}   # 绝对时刻 → 待执行动作（ema 模式时序缓冲）
            abs_step = 0                         # 回合内绝对步数，EMA 对齐的唯一基准
            needs_replan = True                  # 回合开始立刻推理一次
            steps_since_replan = 0
            while not done:
                if needs_replan or steps_since_replan >= execute_steps:
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    # 一次前向 → 未来 K 步动作块 (K, action_dim)
                    chunk = policy.forward_actions(state_t)[0].cpu().numpy()
                    n_forward += 1
                    if execution == "ema":
                        # 按**绝对时刻**对齐写入缓冲：新规划覆盖 [t, t+K)，
                        # 与上一次规划在重叠时刻做 EMA（ACT temporal ensembling）。
                        # 注意：若按块内下标对齐（错误做法），等价于把 4 步前的动作
                        # 混进当前动作，会严重破坏控制。
                        for j in range(k):
                            idx = abs_step + j
                            future[idx] = (
                                alpha * chunk[j] + (1.0 - alpha) * future[idx]
                                if idx in future else chunk[j]
                            )
                    else:
                        plan = chunk                # 开环：整块直接执行
                    needs_replan = False
                    steps_since_replan = 0
                if execution == "ema":
                    action = future.pop(abs_step)   # 取出该时刻经平滑后的动作
                else:
                    action = plan[steps_since_replan]   # (action_dim,)
                state, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
                abs_step += 1
                steps_since_replan += 1
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
        "episode_returns": [float(r) for r in returns],
        "n_episodes": int(n_episodes),
        "chunk_size": k,
        "execution": execution,
        "ema_alpha": alpha,
        "execute_steps": int(execute_steps),
        "n_forward_passes": int(n_forward),
        "forward_per_episode": float(n_forward) / max(1, n_episodes),
    }


def evaluate_closed_loop(
    policy,
    device: torch.device,
    env_name: str = "Pendulum-v1",
    n_episodes: int = 20,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """单步 BC 的闭环评估：每一步都做一次前向推理（``K = 1``）。"""
    return _rollout_closed_loop(
        policy,
        device=device,
        env_name=env_name,
        n_episodes=n_episodes,
        chunk_size=1,
        execution="chunk",
        render=render,
        seed=seed,
    )


def evaluate_closed_loop_chunking(
    policy,
    device: torch.device,
    env_name: str = "Pendulum-v1",
    n_episodes: int = 20,
    chunk_size: int = 8,
    execution: str = "chunk",
    ema_alpha: float = 0.5,
    replan_every: int = 0,
    render: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """动作块 BC 的闭环评估：每隔 ``K`` 步推理一次，一次打出 ``K`` 步动作。

    Args:
        chunk_size: 动作块长度 ``K``（与网络 ``model.chunk_size`` 一致）。
        execution: ``chunk`` 开环整块执行 | ``ema`` EMA 平滑执行。
        ema_alpha: EMA 中新预测的权重（``1.0`` 等价于 ``chunk``）。
        replan_every: 每多少步重新推理一次；``0`` 表示等于 ``chunk_size``
            （即"每隔 K 步执行一次前向推理"）。

    Returns:
        含平均 / 标准差 / 极值回报、逐回合回报与推理次数统计的字典。
    """
    return _rollout_closed_loop(
        policy,
        device=device,
        env_name=env_name,
        n_episodes=n_episodes,
        chunk_size=chunk_size,
        execution=execution,
        ema_alpha=ema_alpha,
        replan_every=replan_every,
        render=render,
        seed=seed,
    )

