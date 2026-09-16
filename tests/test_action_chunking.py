#!/usr/bin/env python
"""W4 Task 1（动作块行为克隆）单元测试。

覆盖范围
--------
1. **动作块切分 / 末端零填充**：``build_action_chunks`` 的形状、滑窗内容、
   零填充位置与 ``valid`` 掩码；窗口**不跨轨迹边界**。
2. **网络输出头维度**：``K * action_dim``；``forward`` / ``forward_actions`` 形状。
3. **K=1 向后兼容**：不配置 ``chunk_size`` 时与被改造前的单步 BC 完全等价
   （输出形状、维度、``forward_actions`` 语义）。
4. **数据对齐**：动作块数据集的 ``chunk[:, 0]`` 恒等于单步 BC 的标签 ``a_t``；
   两种 DataLoader 的训练 / 验证**状态集合逐样本一致**（对比公平性的前提）。
5. **闭环执行调度**：``K=1`` 退化为"每步推理一次"且与单步评估逐回合一致；
   ``K>1`` 时前向次数 ≈ 回合步数 / K（每隔 K 步推理一次）。

运行方式（无需 pytest）::

    python tests/test_action_chunking.py

若已安装 pytest，也可直接 ``pytest tests/test_action_chunking.py``。
"""

from __future__ import annotations

import os
import sys
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.datasets as datasets  # noqa: E402
from src.datasets import (  # noqa: E402
    BCDataset,
    ChunkingBCDataset,
    build_action_chunks,
    build_chunking_dataloaders,
    build_dataloaders,
    collect_expert_trajectories,
    evaluate_closed_loop,
    evaluate_closed_loop_chunking,
)
from src.networks import MLPPolicy, build_network  # noqa: E402


# =============================================================================
# 1. 动作块切分 / 末端零填充
# =============================================================================
def test_build_action_chunks_shape_and_content():
    """滑窗内容逐点正确：``chunk[t, j] == a_{t+j}``（同一轨迹内）。"""
    actions = np.arange(10, dtype=np.float32).reshape(-1, 1)
    chunks, valid = build_action_chunks(actions, 3, episode_lengths=[10])

    assert chunks.shape == (10, 3, 1), chunks.shape
    assert valid.shape == (10, 3)
    for t in range(10):
        for j in range(3):
            if t + j < 10:
                assert chunks[t, j, 0] == actions[t + j, 0], (t, j)
                assert valid[t, j]


def test_build_action_chunks_tail_zero_padding():
    """轨迹末端不足 K 步 → 零填充，且 ``valid`` 掩码精确标记填充位置。"""
    actions = np.arange(5, dtype=np.float32).reshape(-1, 1)      # 单条轨迹 5 步，K=3
    chunks, valid = build_action_chunks(actions, 3, episode_lengths=[5])

    expected_valid = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool)
    assert np.array_equal(valid, expected_valid), valid
    assert chunks[3, 2, 0] == 0.0 and chunks[4, 1, 0] == 0.0 and chunks[4, 2, 0] == 0.0
    # 真实位置仍等于专家动作
    assert chunks[4, 0, 0] == actions[4, 0]


def test_build_action_chunks_does_not_cross_episode_boundary():
    """窗口严禁跨回合边界：上一条轨迹末尾不能"借"下一条轨迹开头的动作。"""
    actions = np.arange(8, dtype=np.float32).reshape(-1, 1)
    chunks, valid = build_action_chunks(actions, 3, episode_lengths=[5, 3])

    # 第 1 条轨迹的最后一个窗口（t=4）只有 1 个真实动作，其余补零
    assert chunks[4, 0, 0] == 4.0
    assert chunks[4, 1, 0] == 0.0 and chunks[4, 2, 0] == 0.0
    assert not valid[4, 1] and not valid[4, 2]
    # 第 2 条轨迹（索引 5,6,7）内部窗口正常
    assert chunks[5, 0, 0] == 5.0 and chunks[5, 1, 0] == 6.0 and chunks[5, 2, 0] == 7.0
    assert valid[5].all()


def test_build_action_chunks_multi_dim_and_k1():
    """多动作维度正确；``K=1`` 时退化为单步标签 ``(N, 1, A)``。"""
    actions = np.random.randn(6, 2).astype(np.float32)
    chunks, valid = build_action_chunks(actions, 1, episode_lengths=[6])
    assert chunks.shape == (6, 1, 2)
    assert valid.all()
    assert np.allclose(chunks[:, 0, :], actions)

    chunks2, _ = build_action_chunks(actions, 2, episode_lengths=[3, 3])
    assert chunks2.shape == (6, 2, 2)
    assert np.allclose(chunks2[0], actions[0:2])
    assert np.allclose(chunks2[3], actions[3:5])          # 第二回合首窗口


# =============================================================================
# 2. 网络输出头维度
# =============================================================================
def _model_config(chunk_size=1):
    return {
        "model": {
            "state_dim": 3, "action_dim": 1, "hidden_dim": 64, "num_layers": 2,
            "name": "mlp", "action_scale": 2.0, "squash": True, "chunk_size": chunk_size,
        }
    }


def test_network_output_head_dimension():
    """输出头维度 = K * action_dim，前向形状为 ``(batch, K, action_dim)``。"""
    for k in (1, 4, 8):
        net = build_network(_model_config(k))
        assert net.chunk_size == k
        assert net.output_dim == k * 1, (k, net.output_dim)

        s = torch.randn(5, 3)
        out = net(s)
        expected = (5, 1) if k == 1 else (5, k, 1)
        assert tuple(out.shape) == expected, (k, out.shape)

        block = net.forward_actions(s)
        assert tuple(block.shape) == (5, k, 1), (k, block.shape)
        if k == 1:
            assert torch.equal(block, out.unsqueeze(1))


def test_k1_backward_compatible_with_single_step_bc():
    """未配置 ``chunk_size`` 时（默认 1）与被改造前的单步 BC 完全等价。"""
    legacy_cfg = _model_config(1)["model"]
    legacy_cfg.pop("chunk_size")                       # 模拟 W3 的旧配置

    torch.manual_seed(0)
    legacy = MLPPolicy(3, 1, 64, num_layers=2, action_scale=2.0, squash=True)
    torch.manual_seed(0)
    new = build_network(_model_config(1))

    assert getattr(legacy, "chunk_size", 1) == 1
    assert new.chunk_size == 1 and new.output_dim == 1

    s = torch.randn(7, 3)
    assert tuple(new(s).shape) == (7, 1)                            # 单步输出，未多一维
    assert torch.equal(new(s), legacy(s))                           # 参数与输出逐位一致


def test_gru_and_cnn_heads_respect_chunk_size():
    """其它网络（CNN / GRU）同样遵循 ``K * action_dim`` 输出头。"""
    cfg = _model_config(4)
    cfg["model"].update({
        "name": "gru", "gru_hidden": 32, "gru_layers": 1,
        "conv_channels": [16, 32], "kernel_size": 3,
    })
    gru = build_network(cfg)
    assert gru.output_dim == 4 and tuple(gru.forward_actions(torch.randn(3, 3)).shape) == (3, 4, 1)

    cfg["model"].update({"name": "cnn", "state_dim": 12})            # CNN 需要图像输入
    cnn = build_network(cfg)
    assert cnn.output_dim == 4
    assert tuple(cnn.forward_actions(torch.randn(2, 12)).shape) == (2, 4, 1)


# =============================================================================
# 3. 数据集对齐（对比公平性）
# =============================================================================
def _synthetic_episodes(n_episodes=4, length=9, state_dim=3, action_dim=1, seed=0):
    rng = np.random.default_rng(seed)
    return [
        (rng.normal(size=(length, state_dim)).astype(np.float32),
         rng.normal(size=(length, action_dim)).astype(np.float32))
        for _ in range(n_episodes)
    ]


def test_chunk_first_action_equals_single_step_label():
    """``chunk[:, 0]`` 恒等于单步 BC 的标签 ``a_t`` —— 即时响应标签从不被零填充污染。"""
    episodes = _synthetic_episodes()
    states = np.concatenate([s for s, _ in episodes])
    actions = np.concatenate([a for _, a in episodes])
    k = 5

    chunks, _ = build_action_chunks(actions, k, episode_lengths=[len(s) for s, _ in episodes])
    ds = ChunkingBCDataset(states, chunks, augment=False)
    bc = BCDataset(states, actions, augment=False)

    assert len(ds) == len(bc) == len(states)
    for i in range(len(states)):
        s_chunk, chunk = ds[i]
        s_bc, a_t = bc[i]
        assert torch.equal(s_chunk, s_bc)                 # 输入状态一致
        assert chunk.shape == (k, 1)
        assert torch.equal(chunk[0].reshape(-1), a_t)     # 第一个动作 == 单步标签


def test_dataloader_split_uses_identical_state_samples():
    """两种切分方式（单步 BC 手工切分 vs 动作块切分）训练/验证样本逐条对齐。"""
    episodes = _synthetic_episodes()
    states = np.concatenate([s for s, _ in episodes])
    actions = np.concatenate([a for _, a in episodes])
    train_ratio, seed = 0.75, 42

    # 单步 BC 的切分方式（与 collect_expert_data / train_bc.py 完全一致）
    idx = np.random.default_rng(seed).permutation(len(states))
    split = int(train_ratio * len(states))
    train_bc = BCDataset(states[idx[:split]], actions[idx[:split]], augment=False)
    val_bc = BCDataset(states[idx[split:]], actions[idx[split:]], augment=False)

    train_chunk, val_chunk, info = build_chunking_dataloaders(
        episodes, chunk_size=4, train_ratio=train_ratio, batch_size=8, seed=seed,
    )
    assert info["n_train"] == split and info["n_val"] == len(states) - split

    def _rows(loader):
        return torch.cat([s for s, _ in loader], dim=0).numpy()

    val_bc_rows = np.stack([val_bc[i][0].numpy() for i in range(len(val_bc))])
    train_bc_rows = np.stack([train_bc[i][0].numpy() for i in range(len(train_bc))])

    # 验证集：DataLoader 不打乱 → 顺序也应逐样本一致
    assert np.allclose(_rows(val_chunk), val_bc_rows)
    # 训练集：DataLoader shuffle=True → 比较排序后的样本集合
    assert (sorted(map(tuple, np.round(_rows(train_chunk), 6)))
            == sorted(map(tuple, np.round(train_bc_rows, 6))))


def test_dataloader_split_alignment_via_canonical_api():
    """用公开 API 复核：``build_dataloaders`` 与 ``build_chunking_dataloaders`` 状态一致。"""
    episodes = _synthetic_episodes(n_episodes=3, length=8)
    states = np.concatenate([s for s, _ in episodes])
    actions = np.concatenate([a for _, a in episodes])
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(states))
    split = int(0.5 * len(states))

    bc_loader, _ = build_dataloaders(
        states[idx[:split]], actions[idx[:split]], states[idx[split:]], actions[idx[split:]],
        batch_size=8, seed=7,
    )
    chunk_loader, _, info = build_chunking_dataloaders(
        episodes, chunk_size=3, train_ratio=0.5, batch_size=8, seed=7,
    )

    def _sorted_states(loader):
        rows = torch.cat([s for s, _ in loader], dim=0).numpy()
        return sorted(map(tuple, np.round(rows, 6)))

    assert info["n_train"] == split
    assert _sorted_states(bc_loader) == _sorted_states(chunk_loader)


# =============================================================================
# 4. 闭环执行调度（每隔 K 步推理一次）
# =============================================================================
def _tiny_policy(chunk_size: int, seed: int = 0):
    torch.manual_seed(seed)
    return build_network(_model_config(chunk_size))


def test_closed_loop_k1_equals_single_step_evaluation():
    """``K=1`` 的动作块评估与单步 BC 评估逐回合一致（同一调度语义）。"""
    policy = _tiny_policy(1, seed=1)
    kw = dict(device=torch.device("cpu"), env_name="Pendulum-v1", n_episodes=3, seed=123)

    single = evaluate_closed_loop(policy, **kw)
    chunked = evaluate_closed_loop_chunking(policy, chunk_size=1, execution="chunk", **kw)

    assert np.allclose(single["episode_returns"], chunked["episode_returns"])
    assert single["mean_return"] == chunked["mean_return"]
    # 每一步都推理一次 → 前向次数 = 回合步数（与单步 BC 完全一致）
    assert chunked["n_forward_passes"] == single["n_forward_passes"]
    assert chunked["execute_steps"] == 1
    assert chunked["forward_per_episode"] >= 1.0


def test_closed_loop_chunking_reduces_forward_calls():
    """``K=8`` 开环执行时每回合前向次数 ≈ 回合步数 / 8（Pendulum 为 200 步）。"""
    policy = _tiny_policy(8, seed=2)
    result = evaluate_closed_loop_chunking(
        policy, device=torch.device("cpu"), env_name="Pendulum-v1",
        n_episodes=2, chunk_size=8, execution="chunk", replan_every=0, seed=7,
    )
    assert result["execute_steps"] == 8                 # 0 → 默认等于 K
    assert 24 <= result["forward_per_episode"] <= 26    # ceil(200/8) = 25
    assert result["forward_per_episode"] < 30           # 显著低于单步 BC 的 ~200 次


def test_closed_loop_replan_every_controls_forward_calls():
    """``replan_every`` 精确控制重规划间隔：越大 → 前向次数越少。"""
    policy = _tiny_policy(4, seed=3)
    kw = dict(device=torch.device("cpu"), env_name="Pendulum-v1", n_episodes=1, chunk_size=4, seed=5)

    every_4 = evaluate_closed_loop_chunking(policy, execution="chunk", replan_every=4, **kw)
    every_2 = evaluate_closed_loop_chunking(policy, execution="chunk", replan_every=2, **kw)
    every_1 = evaluate_closed_loop_chunking(policy, execution="chunk", replan_every=1, **kw)

    assert every_4["execute_steps"] == 4 and every_2["execute_steps"] == 2
    assert every_4["n_forward_passes"] < every_2["n_forward_passes"] < every_1["n_forward_passes"]


class _RecordingEnv:
    """最小环境替身：记录执行的动作，固定 ``horizon`` 步后终止（隔离环境动力学）。"""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.actions = []

    def reset(self, seed=None):                                   # noqa: D102
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):                                       # noqa: D102
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        done = len(self.actions) >= self.horizon
        return np.zeros(3, dtype=np.float32), 0.0, done, False, {}

    def close(self):                                              # noqa: D102
        pass


class _ScriptedChunkPolicy:
    """脚本策略：第 ``i`` 次推理返回动作块 ``10*i + [0, 1, ..., K-1]``（可解析验证）。"""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.training = True
        self.calls = 0

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def forward_actions(self, state):
        values = self.calls * 10 + np.arange(self.chunk_size, dtype=np.float32)
        self.calls += 1
        return torch.as_tensor(values, dtype=torch.float32).reshape(1, self.chunk_size, 1)


def _execute_chunked(execution, chunk_size, replan_every, ema_alpha, horizon):
    """用替身环境跑一次闭环 rollout，返回实际执行的动作序列。"""
    env = _RecordingEnv(horizon)
    policy = _ScriptedChunkPolicy(chunk_size)
    with mock.patch.object(datasets.gym, "make", lambda *a, **kw: env):
        datasets._rollout_closed_loop(
            policy, device=torch.device("cpu"), env_name="Pendulum-v1", n_episodes=1,
            chunk_size=chunk_size, execution=execution, ema_alpha=ema_alpha,
            replan_every=replan_every,
        )
    return np.concatenate(env.actions)


def test_closed_loop_ema_blends_by_absolute_time():
    """EMA 必须按**绝对时刻**对齐（只平滑时间重叠处），而非按块内下标对齐。

    K=4、每 2 步重规划时，规划 ``i`` 的动作块为 ``10i + [0,1,2,3]``：

    * 时刻 2 的动作 = ``0.5 * 规划1[0] + 0.5 * 规划0[2] = 6``（时刻对齐，两值同为 t=2 的预测）
    * 若按块内下标对齐（错误做法）会得到 ``0.5*规划1[0] + 0.5*规划0[0] = 5``
    """
    executed = _execute_chunked("ema", chunk_size=4, replan_every=2, ema_alpha=0.5, horizon=8)
    assert np.allclose(executed, [0, 1, 6, 7, 16, 17, 26, 27]), executed


def test_closed_loop_ema_alpha1_equals_chunk_execution():
    """``ema_alpha=1`` 时 EMA 退化为开环执行（平滑叠加本身不引入偏差）。"""
    chunk = _execute_chunked("chunk", chunk_size=4, replan_every=2, ema_alpha=0.5, horizon=8)
    ema = _execute_chunked("ema", chunk_size=4, replan_every=2, ema_alpha=1.0, horizon=8)
    assert np.allclose(chunk, ema), (chunk, ema)


def test_closed_loop_ema_uses_same_schedule_as_chunk():
    """EMA 不改变推理调度（前向次数与开环一致）；无重叠时与开环执行逐位相同。"""
    policy = _tiny_policy(4, seed=4)
    kw = dict(device=torch.device("cpu"), env_name="Pendulum-v1", n_episodes=1, chunk_size=4, seed=9)

    chunk = evaluate_closed_loop_chunking(policy, execution="chunk", **kw)
    ema = evaluate_closed_loop_chunking(policy, execution="ema", ema_alpha=0.5, **kw)

    assert chunk["n_forward_passes"] == ema["n_forward_passes"]
    assert ema["execution"] == "ema" and ema["ema_alpha"] == 0.5
    # replan_every = K（无时间重叠）→ 平滑无对象，EMA 与开环执行完全一致
    assert chunk["episode_returns"] == ema["episode_returns"]


# =============================================================================
# 5. 专家数据缓存一致性（单步 BC 与动作块 BC 使用同一批专家数据）
# =============================================================================
def test_resolve_chunk_size_consistency():
    """``chunking.chunk_size`` 与 ``model.chunk_size`` 的一致性校验（K 单点定义）。"""
    from experiments.train_chunking_bc import resolve_chunk_size

    assert resolve_chunk_size({"model": {"chunk_size": 8}, "chunking": {"chunk_size": None}}) == 8
    assert resolve_chunk_size({"model": {"chunk_size": 4}, "chunking": {}}) == 4
    assert resolve_chunk_size({"model": {}, "chunking": {}}) == 1          # 默认单步 BC

    try:
        resolve_chunk_size({"model": {"chunk_size": 8}, "chunking": {"chunk_size": 4}})
    except ValueError as exc:
        assert "不一致" in str(exc)
    else:
        raise AssertionError("chunking.chunk_size 与 model.chunk_size 不一致时应抛出 ValueError")


def test_trajectory_cache_matches_flat_cache():
    """轨迹级缓存与单步 BC 的扁平缓存内容一致（同一批专家数据，仅多存回合长度）。"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "expert_pendulum")
    flat_path = os.path.join(data_dir, "expert_pendulum.npz")
    traj_path = os.path.join(data_dir, "expert_pendulum_traj.npz")
    if not (os.path.isfile(flat_path) and os.path.isfile(traj_path)):
        print("[skip] 缺少缓存文件（需先运行训练脚本生成）:", flat_path, traj_path)
        return

    flat = np.load(flat_path)
    traj = np.load(traj_path)
    assert np.array_equal(flat["states"], traj["states"])
    assert np.array_equal(flat["actions"], traj["actions"])
    lengths = traj["episode_lengths"]
    assert int(lengths.sum()) == len(traj["states"])

    episodes = collect_expert_trajectories(
        env_name="Pendulum-v1", num_expert_steps=10000, seed=42,
        reuse_cache=True, cache_path=traj_path, verbose=False,
    )
    assert len(episodes) == len(lengths)
    assert all(len(s) == int(n) for (s, _), n in zip(episodes, lengths))


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


