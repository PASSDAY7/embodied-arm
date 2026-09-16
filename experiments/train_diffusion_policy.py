#!/usr/bin/env python
"""极简条件扩散策略（Mini-Diffusion Policy）训练与评估脚本 —— W4 Task 2。

任务
----
把 Task 1 的**确定性**动作块回归（MLP + MSE，一次前向输出唯一动作块）升级为基于
DDPM 的**概率式**动作块生成器 ``p(a | s)``，并量化它在**多峰动作分布**上的收益。

四步数学（对应 ``src/diffusion.py``）::

    ① 前向加噪   a^t = sqrt(ᾱ_t)·a^0 + sqrt(1 - ᾱ_t)·ε
    ② 网络预测   ε_θ(s, a^t, t)            ← 状态 s 与扩散步 t 双条件化
    ③ 训练损失   E_{t,ε}[ ||ε - ε_θ(s, a^t, t)||² ]（掩码掉动作块的零填充位置）
    ④ 闭环采样   a^N ~ N(0, I) → N 步反向去噪 → 干净动作块 → 执行

为什么需要它：多峰动作分布
--------------------------
Pendulum 的**多峰专家**（``expert_controller_multimodal``）在"停在下垂位置附近、
角速度近 0"这一关键状态上，既可能向左摆起也可能向右摆起 —— 同一个状态 ``s``
对应两个符号相反的可行动作 ``±2``。此时：

* 确定性 MSE 回归的最小解是**条件均值** ``E[a|s] ≈ 0`` —— 一个"既非左也非右"的
  无效动作（通俗说就是"撞墙的平均动作"）；在**状态完全无法区分模态**的极端情形下
  （见下面的双峰玩具任务），回归的输出精确等于 0；
* 扩散策略建模的是**整个分布** ``p(a|s)``，每次规划从分布中**采样一个模态**，
  因此两个分支都能被捕获并执行。

但本工程的实测结论必须诚实记录（详见 ``docs/w4_task2_diffusion_policy.md``）：
Pendulum 里"歧义状态"只占 0.95%，且相邻状态的 ω 符号已能区分方向，所以确定性回归
可以绕开均值，**闭环回报反而更好**（-181.7 vs -331.8，代价是扩散每次规划慢 132.8 倍、
块间模式翻转率 49.2%）。扩散真正的优势出现在"状态完全无法区分模态"的任务上
（双峰玩具任务：回归 0% vs 扩散 98.5% 命中率，双峰覆盖 100%）。

本脚本的对照实验（六层证据）
----------------------------
1. **未见过歧义状态探针**（``multimodal_probe`` + ``sample_ambiguous_states``）：
   用 seed 10000+ 独立生成状态（不在训练集里），每状态采样 256 次，统计"落到最近模态的
   比例 / 平均幅值 / 采样标准差 / 是否两侧都采到"；确定性策略同一状态只输出一个值（std = 0）。
2. **闭环回报**：同一份多峰数据、同一训练预算下，"确定性 MLP 动作块" vs "扩散策略"
   各跑 20 个回合并对比平均回报（含失败回合计数、逐回合回报与显著性判据）。
3. **推理开销与 N 消融**：单次规划的墙钟耗时、每回合网络前向次数、"去噪步数 N vs 闭环
   回报"，以及"重规划间隔"消融，直观展示"效果-开销"权衡。
4. **一致性诊断**（``analyze_plan_consistency``）：相邻两次规划之间"块首动作符号翻转率"，
   用来量化"每次规划独立抽噪声"带来的模式抖动。
5. **稀疏数据消融**（``run_sparse_data_ablation``）：只用 16% 轨迹时两种策略的对照，
   用来检验"数据越少 ⇒ 均值陷阱越明显"（实测：假设不成立，扩散的样本效率并不更高）。
6. **双峰玩具任务**（``run_bimodal_toy_demo``）：1 维状态 / 1 维动作、两峰位置由状态解析
   给出的最小例子（``a ~ N(±μ(s), σ²)``），是"扩散能建模多峰、MSE 不能"的干净证据。

工程实现
--------
``--config/--set/--resume`` 解析、运行目录 / 日志 / TensorBoard、epoch 循环、优雅
停机与检查点等全部复用 ``src/engine.py``（扩散的噪声预测损失通过 ``train_step_fn`` /
``val_step_fn`` 注入），闭环评估复用 ``src/datasets.py`` 的 ``evaluate_closed_loop_chunking``
（与 Task 1 同一套执行调度），因此两套实验的训练与评估协议严格可比。

用法示例::

    python experiments/train_diffusion_policy.py                              # 默认（DDIM 16 步，K=8）
    python experiments/train_diffusion_policy.py --set diffusion.sample_steps=10
    python experiments/train_diffusion_policy.py --set diffusion.sampler=ddpm   # 随机 ancestral 采样
    python experiments/train_diffusion_policy.py --set model.chunk_size=4       # 换动作块长度
    python experiments/train_diffusion_policy.py --device cpu --no-eval         # 只训练不评估
    python experiments/train_diffusion_policy.py --resume                       # 断点续训
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # gymnasium 为可选依赖，仅需要闭环采样探针 / 一致性统计时使用
    import gymnasium as gym
except ImportError:  # pragma: no cover
    gym = None

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 允许脚本以 `python experiments/train_diffusion_policy.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import (  # noqa: E402
    BCDataset,
    ambiguity_mask,
    build_chunking_dataloaders,
    collect_multimodal_expert_trajectories,
    evaluate_closed_loop_chunking,
    make_bimodal_toy_dataset,
    toy_bimodal_mode_centers,
)
from src.diffusion import build_diffusion_policy, count_parameters  # noqa: E402
from src.engine import (  # noqa: E402
    apply_overrides,
    build_scheduler,
    finish_run,
    load_experiment_config,
    parse_train_args,
    prepare_run,
    report_closed_loop,
    restore_from_checkpoint,
    train_loop,
    train_one_epoch,
    validate,
)
from src.networks import build_network  # noqa: E402
from src.utils import Checkpointer  # noqa: E402

# 与 Task 1 复用**同一份** K 解析逻辑（两级配置一致性校验），避免两份实现漂移
from experiments.train_chunking_bc import resolve_chunk_size  # noqa: E402

CONFIG_PATH = os.path.join("configs", "diffusion_config.yaml")
TITLE = "极简条件扩散策略（Mini-Diffusion Policy）实验"
SCRIPT_PATH = "experiments/train_diffusion_policy.py"


# =============================================================================
# 1. 损失注入：把"噪声预测 MSE"接进通用训练引擎
# =============================================================================
def make_diffusion_step_fn(policy, device: torch.device):
    """构造 ``step_fn(model, batch) -> (loss, n)``，供 ``engine.train_loop`` 使用。

    batch 为 ``(state, chunk, valid)``（``build_chunking_dataloaders(return_valid=True)``）。
    每个 batch **现场**采样扩散步 ``t ~ U{0, T-1}`` 与噪声 ``ε``：这正是 DDPM 简化
    目标的蒙特卡洛估计（一次采样即一个无偏样本，无需多步展开）。
    """

    def step_fn(model, batch) -> Tuple[torch.Tensor, int]:
        state, chunk, valid = batch
        state = state.to(device, non_blocking=True)
        chunk = chunk.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        loss, _ = model.p_losses(state, chunk, valid)
        return loss, int(state.size(0))

    return step_fn


@torch.no_grad()
def evaluate_noise_loss(policy, loader, device: torch.device) -> Dict[str, float]:
    """在整份验证集上评估噪声预测 MSE 与 ``x0``（动作块）重建 MSE。

    与训练损失的随机性不同，这里对每个样本重复 ``n_repeats`` 次采样 ``t`` 取平均，
    因此指标更稳定、可用于 TensorBoard 上报。
    """
    policy.eval()
    eps_sum, x0_sum, n = 0.0, 0.0, 0
    for state, chunk, valid in loader:
        state = state.to(device); chunk = chunk.to(device); valid = valid.to(device)
        loss, info = policy.p_losses(state, chunk, valid)
        eps_sum += loss.item() * state.size(0)
        x0_sum += float(info["x0_mse"]) * state.size(0)
        n += int(state.size(0))
    return {"noise_mse": eps_sum / max(1, n), "x0_mse": x0_sum / max(1, n)}



# =============================================================================
# 2. 确定性对照：同一份多峰数据上的 MLP 动作块 BC（Task 1 的配置与损失）
# =============================================================================
def train_deterministic_baseline(
    train_loader, val_loader, ctx, epochs: Optional[int] = None,
    tag: str = "baseline_loss", log_prefix: str = "[对照-确定性]",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """在**多峰数据**上训练 Task 1 式的确定性动作块策略（MLP + 逐元素 MSE）。

    与 Task 1 逐项对齐：同一 ``build_network`` MLP（输出头 ``K * action_dim``）、
    同一 MSE 损失、同一 Adam 超参、同一 epoch 数、同一数据切分（同一 ``seed``）——
    因此"确定性均值"与"扩散采样"的差距只来自**输出分布 vs 条件均值**这一个变量。

    Args:
        epochs: 覆盖 ``ctx.epochs``（稀疏数据消融会用到更少的 epoch）。

    Returns:
        ``(model, stats)``；``model`` 已加载验证集最优权重。
    """
    train_cfg = ctx.train_cfg
    model = build_network(ctx.config).to(ctx.device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    log_interval = int(train_cfg.get("log_interval", 20))

    total_epochs = int(epochs) if epochs is not None else int(ctx.epochs)
    best_val, best_state = float("inf"), None
    ctx.logger.info("%s 在同一份多峰数据上训练 MLP 动作块策略（%d epoch）...", log_prefix, total_epochs)
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, ctx.device, max_grad_norm
        )
        val_loss = validate(model, val_loader, criterion, ctx.device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ctx.tb.add_scalars(tag, {"train": train_loss, "val": val_loss}, epoch)
        if epoch % log_interval == 0 or epoch == 1 or epoch == total_epochs:
            ctx.logger.info(
                "%s Epoch [%d/%d] | Train MSE: %.6f | Val MSE: %.6f | Best: %.6f",
                log_prefix, epoch, total_epochs, train_loss, val_loss, best_val,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    stats = {
        "best_val_mse": best_val,
        "epochs": total_epochs,
        "n_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    ctx.logger.info("%s 训练完成 | 最佳验证集 MSE: %.6f | 参数量: %d",
                    log_prefix, best_val, stats["n_params"])
    return model, stats


def train_diffusion_model(
    ctx, config, train_loader, val_loader, epochs: Optional[int] = None,
    tag: str = "diffusion_loss", log_prefix: str = "[扩散]",
):
    """训练一个新的扩散策略（供稀疏数据消融等需要独立训练的地方复用）。

    与主流程使用**完全相同**的损失注入方式（``step_fn`` + 噪声预测 MSE）与优化器设置，
    只是不写检查点、不做优雅停机（消融实验的定位是"同预算下的对照"，不是产出权重）。
    """
    train_cfg = ctx.train_cfg
    policy = build_diffusion_policy(config).to(ctx.device)
    policy.schedule.to(ctx.device)
    optimizer = optim.Adam(
        policy.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    step_fn = make_diffusion_step_fn(policy, ctx.device)
    total_epochs = int(epochs) if epochs is not None else int(ctx.epochs)
    log_interval = int(train_cfg.get("log_interval", 20))

    best_val, best_state = float("inf"), None
    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(policy, train_loader, None, optimizer, ctx.device,
                                     float(train_cfg.get("max_grad_norm", 0.0)), step_fn=step_fn)
        val_loss = validate(policy, val_loader, None, ctx.device, step_fn=step_fn)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        ctx.tb.add_scalars(tag, {"train": train_loss, "val": val_loss}, epoch)
        if epoch % log_interval == 0 or epoch == 1 or epoch == total_epochs:
            ctx.logger.info("%s Epoch [%d/%d] | Train noise-MSE: %.6f | Val noise-MSE: %.6f | Best: %.6f",
                            log_prefix, epoch, total_epochs, train_loss, val_loss, best_val)
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.schedule.to(ctx.device)
    return policy, {"best_val_noise_mse": best_val, "epochs": total_epochs,
                    "n_params": count_parameters(policy)}


# =============================================================================
# 3. 推理开销基准（单次规划耗时）
# =============================================================================
def benchmark_plan_latency(
    policy, device: torch.device, state_dim: int = 3,
    repeats: int = 300, warmup: int = 50,
) -> Dict[str, float]:
    """测量闭环中最真实的工况：**单环境、单状态**下一次规划的墙钟耗时。"""
    state = torch.zeros(1, state_dim, device=device)
    policy.eval()
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            policy.forward_actions(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(int(repeats)):
            policy.forward_actions(state)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    ms = 1000.0 * elapsed / max(1, int(repeats))
    return {"ms_per_plan": ms, "plans_per_second": 1000.0 / max(ms, 1e-9), "repeats": int(repeats)}


# =============================================================================
# 4. 多峰探针：同一状态下"分布采样" vs "确定性输出"
# =============================================================================
@torch.no_grad()
def multimodal_probe(
    policy,
    states: np.ndarray,
    mode_centers: Tuple[float, float] = (-2.0, 2.0),
    n_samples: int = 256,
    hit_tol: float = 0.3,
    device: torch.device = torch.device("cpu"),
    stochastic: bool = True,
) -> Dict[str, Any]:
    """在**方向歧义**状态上探测策略输出的分布形态。

    指标（对每个状态跨 ``n_samples`` 次采样统计）::

        mean_abs_action    平均 |a|：确定性均值策略 ≈ 0（落到两个模态之间的"空地"），
                           扩散策略 ≈ 模态幅值（始终给出可行动作）
        mean_std_action    采样标准差：确定性 = 0；扩散 > 0 说明真的在采样
        hit_rate           落到**任一**模态中心 ``hit_tol`` 邻域的比例
        hit_rate_pos/neg   分别命中正/负向模态的比例（双峰才会两者都 > 0）
        both_modes_rate    同时命中两个模态的状态比例（"多峰被捕获"的判定）
        nearest_mode_dist  到最近模态中心的平均距离（越小越"可执行"）

    Args:
        stochastic: ``True`` 时对 ``(状态 × 采样)`` 展开的大 batch 一次采样（各样本独立）；
            ``False`` 时对同一状态重复前向并把同一输出复制 ``n_samples`` 份（确定性策略）。
    """
    states = np.asarray(states, dtype=np.float32)
    n_states = int(states.shape[0])
    if n_states == 0:
        return {"summary": {"n_states": 0}, "per_state": {}}

    state_t = torch.as_tensor(states, device=device)
    centers = np.asarray(mode_centers, dtype=np.float64)

    if stochastic:
        repeated = state_t.repeat_interleave(int(n_samples), dim=0)
        actions = policy.forward_actions(repeated)[:, 0, 0]
        actions = actions.detach().cpu().numpy().reshape(n_states, int(n_samples))
    else:
        base = policy.forward_actions(state_t)[:, 0, 0].detach().cpu().numpy().reshape(-1, 1)
        actions = np.repeat(base, int(n_samples), axis=1)

    dist = np.abs(actions[:, :, None] - centers[None, None, :])     # (S, N, 2)
    hit_each = dist <= float(hit_tol)
    per_state = {
        "mean_abs_action": np.abs(actions).mean(axis=1).tolist(),
        "mean_std_action": actions.std(axis=1).tolist(),
        "hit_rate": hit_each.any(axis=2).mean(axis=1).tolist(),
        "hit_rate_neg": hit_each[:, :, 0].mean(axis=1).tolist(),
        "hit_rate_pos": hit_each[:, :, 1].mean(axis=1).tolist(),
        "nearest_mode_dist": dist.min(axis=2).mean(axis=1).tolist(),
    }
    summary = {
        "n_states": n_states,
        "n_samples": int(n_samples),
        "mode_centers": [float(c) for c in centers],
        "hit_tol": float(hit_tol),
        "mean_abs_action": float(np.mean(per_state["mean_abs_action"])),
        "mean_std_action": float(np.mean(per_state["mean_std_action"])),
        "hit_rate": float(np.mean(per_state["hit_rate"])),
        "nearest_mode_dist": float(np.mean(per_state["nearest_mode_dist"])),
        "both_modes_rate": float(np.mean([
            (pos > 0.0 and neg > 0.0)
            for pos, neg in zip(per_state["hit_rate_pos"], per_state["hit_rate_neg"])
        ])),
    }
    return {
        "summary": summary,
        "per_state": per_state,
        "states": states.tolist(),
        "action_samples": actions.tolist(),
    }


def sample_ambiguous_states(
    env_name: str = "Pendulum-v1",
    n_states: int = 8,
    seed: int = 10_000,
    max_tries: int = 4_000,
    kick_threshold: float = 1.0,
    hang_margin: float = 0.5,
) -> np.ndarray:
    """从**训练时未见过**的回合初始状态里挑出"方向歧义"状态（用于多峰探针）。

    为什么不能用训练集里的歧义状态做探针：训练样本中的歧义状态各自对应**某一次
    具体演示**（要么左摆、要么右摆），高容量回归器可以"记住"每个样本而绕开取均值；
    而闭环评估时面对的是**全新回合的初始状态**，那里没有任何记忆可依赖。因此探针
    必须用未见过的状态才能公平地反映"泛化到新状态"时的行为。

    Args:
        seed: 起始随机种子；调用方应取远大于训练种子（如 10000+）以免与训练数据重合。
        max_tries: 最多尝试多少个初始状态。

    Returns:
        ``(n, 3)`` 状态数组（可能少于 ``n_states``，取决于环境采到多少歧义状态）。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法采样探针状态。")

    env = gym.make(env_name)
    found: List[np.ndarray] = []
    try:
        for k in range(int(max_tries)):
            if len(found) >= int(n_states):
                break
            state, _ = env.reset(seed=int(seed) + k)
            state = np.asarray(state, dtype=np.float64)
            if bool(ambiguity_mask(state.reshape(1, -1), kick_threshold=kick_threshold,
                                   hang_margin=hang_margin)[0]):
                found.append(state.astype(np.float32))
    finally:
        env.close()
    return np.asarray(found, dtype=np.float32) if found else np.zeros((0, 3), dtype=np.float32)


# =============================================================================
# 5. 闭环评估与数据侧参照
# =============================================================================
def run_closed_loop(
    policy, ctx, chunk_size: int, execution: str, replan_every: int,
    label: str, tb_tag: str,
) -> Dict[str, Any]:
    """复用 Task 1 的执行调度做闭环评估（20 回合，相同初始状态种子）。

    ``n_forward_passes`` 是**规划次数**（每次规划 = 一次 ``forward_actions``）；
    扩散策略每次规划内部还有 ``N`` 次噪声网络前向，对比时应同时看 ``sample_cost``。
    """
    result = evaluate_closed_loop_chunking(
        policy,
        device=ctx.device,
        env_name=ctx.env_name,
        n_episodes=ctx.n_eval_episodes,
        chunk_size=chunk_size,
        execution=execution,
        ema_alpha=float(ctx.config.get("inference", {}).get("ema_alpha", 0.5)),
        replan_every=replan_every,
        render=bool(ctx.eval_cfg.get("render", False)),
        seed=ctx.seed,
    )
    report_closed_loop(ctx, result, tag=tb_tag, label=label)
    n_net = int(result["n_forward_passes"]) * int(getattr(policy, "sample_cost", 1))
    ctx.logger.info(
        "推理开销   | 规划次数: %d（每回合 %.1f 次）| 网络前向: %d 次 / 每回合 %.1f 次",
        result["n_forward_passes"], result["forward_per_episode"], n_net,
        n_net / max(1, int(result["n_episodes"])),
    )
    result["n_network_forwards"] = n_net
    result["mean_return_plus"] = float(result["mean_return"] - result["std_return"])
    result["failed_episodes"] = int(sum(1 for r in result["episode_returns"] if r < -800.0))
    return result


def analyze_plan_consistency(
    policy, ctx, chunk_size: int, n_episodes: int = 10, seed_offset: int = 500,
) -> Dict[str, Any]:
    """统计**多峰采样带来的时序一致性代价**：相邻两次规划的"方向"是否跳变。

    在欠驱动系统（倒立摆）里，"往哪边摆"一旦确定就必须在一整个摆动周期内保持一致；
    概率式策略每次规划都重新采样，理论上可能在两个模态之间反复横跳，把能量互相抵消。
    本诊断统计每回合相邻规划之间**块首动作符号的翻转次数**，用来量化这一风险。

    Returns:
        含 ``flips_per_episode`` / ``flip_rate``（翻转次数 / 规划次数）的字典。
    """
    if gym is None:
        raise ImportError("未安装 gymnasium，无法统计规划一致性。")

    env = gym.make(ctx.env_name)
    execute_steps = max(1, int(chunk_size))
    flips_per_episode: List[int] = []
    plans_per_episode: List[int] = []
    was_training = policy.training
    policy.eval()
    try:
        with torch.no_grad():
            for ep in range(int(n_episodes)):
                state, _ = env.reset(seed=ctx.seed + seed_offset + ep)
                done = False
                signs: List[float] = []
                steps_since_plan = execute_steps  # 回合开始立刻规划
                while not done:
                    if steps_since_plan >= execute_steps:
                        state_t = torch.as_tensor(state, dtype=torch.float32,
                                                  device=ctx.device).unsqueeze(0)
                        action = float(policy.forward_actions(state_t)[0, 0, 0].item())
                        signs.append(np.sign(action) if abs(action) > 1e-6 else 0.0)
                        steps_since_plan = 0
                    state, _, terminated, truncated, _ = env.step(np.array([float(action)],
                                                                          dtype=np.float32))
                    done = terminated or truncated
                    steps_since_plan += 1
                flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
                flips_per_episode.append(flips)
                plans_per_episode.append(len(signs))
    finally:
        env.close()
        if was_training:
            policy.train()

    flips = np.asarray(flips_per_episode, dtype=np.float64)
    plans = np.asarray(plans_per_episode, dtype=np.float64)
    return {
        "n_episodes": int(n_episodes),
        "flips_per_episode": float(flips.mean()),
        "plans_per_episode": float(plans.mean()),
        "flip_rate": float(flips.sum() / max(1.0, plans.sum())),
    }


# =============================================================================
# 6. 双峰玩具任务：多峰机理的最小可验证例子（1 维状态 / 1 维动作）
# =============================================================================
def run_bimodal_toy_demo(ctx, toy_cfg: Dict[str, Any], diff_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """在解析可控的双峰玩具数据上直接对比"确定性均值"与"扩散采样"。

    数据集:: ``a | s ~ 0.5·N(-μ(s), σ²) + 0.5·N(+μ(s), σ²)``（``μ(s) = 1 + 0.5·s``，
    两个模态关于 0 严格对称）。因此：

    * 确定性 MSE 的最优解逐点就是 **条件均值 ``E[a|s] = 0``**，而 ``a = 0`` 到两个
      模态的距离都等于 ``μ(s) ≈ 1.0~1.5`` —— 这就是"撞墙的平均动作"；
    * 扩散策略建模 ``p(a|s)``，采样应≈50/50 落在两个模态附近。

    返回可直接写入 JSON 的统计字典（同时上报 TensorBoard）。
    """
    device = ctx.device
    n_states = int(toy_cfg.get("n_states", 9))
    states, actions, _ = make_bimodal_toy_dataset(
        n_states=n_states,
        n_per_mode=int(toy_cfg.get("n_per_mode", 2000)),
        mode_center=float(toy_cfg.get("mode_center", 1.0)),
        mode_slope=float(toy_cfg.get("mode_slope", 0.5)),
        mode_noise_std=float(toy_cfg.get("mode_noise_std", 0.05)),
        seed=ctx.seed,
    )
    rng = np.random.default_rng(ctx.seed)
    perm = rng.permutation(len(states))
    split = int(0.8 * len(states))
    train_ds = BCDataset(states[perm[:split]], actions[perm[:split]])
    val_ds = BCDataset(states[perm[split:]], actions[perm[split:]])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)

    epochs = int(toy_cfg.get("epochs", 200))
    lr = float(ctx.train_cfg.get("lr", 1e-3))
    log_every = max(1, epochs // 4)

    # ---- (a) 确定性 MLP：MSE 回归 → 条件均值 ----
    det_cfg = {"model": {"name": "mlp", "state_dim": 1, "action_dim": 1, "hidden_dim": 128,
                         "num_layers": 2, "chunk_size": 1, "squash": True, "action_scale": 2.0}}
    det = build_network(det_cfg).to(device)
    det_opt = optim.Adam(det.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(det, train_loader, criterion, det_opt, device, 0.0)
        va = validate(det, val_loader, criterion, device)
        ctx.tb.add_scalars("toy/deterministic_mse", {"train": tr, "val": va}, epoch)
        if epoch % log_every == 0 or epoch == 1:
            ctx.logger.info("[玩具-确定性] Epoch [%d/%d] | Train MSE: %.6f | Val MSE: %.6f",
                            epoch, epochs, tr, va)

    # ---- (b) 扩散策略：同一份数据、同一 epoch 数、同一学习率 ----
    toy_diff_cfg = {
        "model": {"state_dim": 1, "action_dim": 1, "chunk_size": 1, "action_scale": 2.0},
        "diffusion": {
            "num_train_timesteps": int(diff_cfg.get("num_train_timesteps", 100)),
            "schedule": diff_cfg.get("schedule", "cosine"),
            "sample_steps": int(diff_cfg.get("sample_steps", 16)),
            "sampler": diff_cfg.get("sampler", "ddim"),
            "eta": 0.0, "clip_x0": 1.0, "seed": None,
        },
        "noise_net": {"hidden_dim": 128, "num_layers": 3, "time_embed_dim": 32},
    }
    toy_policy = build_diffusion_policy(toy_diff_cfg).to(device)
    toy_policy.schedule.to(device)

    def toy_noise_step(model, batch):
        state, action = batch
        state = state.to(device)
        action = action.to(device)
        loss, _ = model.p_losses(state, action, None)
        return loss, int(state.size(0))

    toy_opt = optim.Adam(toy_policy.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(toy_policy, train_loader, None, toy_opt, device, 0.0,
                             step_fn=toy_noise_step)
        va = validate(toy_policy, val_loader, None, device, step_fn=toy_noise_step)
        ctx.tb.add_scalars("toy/diffusion_noise_mse", {"train": tr, "val": va}, epoch)
        if epoch % log_every == 0 or epoch == 1:
            ctx.logger.info("[玩具-扩散] Epoch [%d/%d] | Train noise-MSE: %.6f | Val noise-MSE: %.6f",
                            epoch, epochs, tr, va)

    # ---- (c) 逐状态对比：模态中心来自解析式 μ(s) ----
    grid = np.linspace(-1.0, 1.0, n_states, dtype=np.float32).reshape(-1, 1)
    centers = toy_bimodal_mode_centers(
        grid, float(toy_cfg.get("mode_center", 1.0)), float(toy_cfg.get("mode_slope", 0.5))
    )
    hit_tol = float(toy_cfg.get("hit_tol", 0.3))
    n_samples = int(toy_cfg.get("n_samples", 512))
    grid_t = torch.as_tensor(grid, device=device)
    with torch.no_grad():
        det_out = det.forward_actions(grid_t)[:, 0, 0].cpu().numpy()
        samples = toy_policy.forward_actions(grid_t.repeat_interleave(n_samples, 0))[:, 0, 0]
        samples = samples.cpu().numpy().reshape(len(grid), n_samples)

    det_dist = np.abs(det_out[:, None] - centers).min(axis=1)
    dist_each = np.abs(samples[:, :, None] - centers[:, None, :])   # (S, N, 2)
    sample_dist = dist_each.min(axis=2)
    hit_neg = (dist_each[:, :, 0] <= hit_tol).mean(axis=1)
    hit_pos = (dist_each[:, :, 1] <= hit_tol).mean(axis=1)
    summary = {
        "n_states": int(n_states),
        "n_samples_per_state": n_samples,
        "hit_tol": hit_tol,
        "mode_centers": centers.tolist(),
        "deterministic": {
            "val_mse": float(validate(det, val_loader, criterion, device)),
            "n_params": int(sum(p.numel() for p in det.parameters() if p.requires_grad)),
            "actions": det_out.tolist(),
            "mean_abs_action": float(np.abs(det_out).mean()),
            "nearest_mode_dist": float(det_dist.mean()),
            "hit_rate": float((det_dist <= hit_tol).mean()),
        },
        "diffusion": {
            "val_noise_mse": float(validate(toy_policy, val_loader, None, device,
                                            step_fn=toy_noise_step)),
            "n_params": count_parameters(toy_policy),
            "mean_abs_action": float(np.abs(samples).mean()),
            "mean_std_action": float(samples.std(axis=1).mean()),
            "nearest_mode_dist": float(sample_dist.mean()),
            "hit_rate": float((sample_dist <= hit_tol).mean()),
            "hit_rate_neg": float(hit_neg.mean()),
            "hit_rate_pos": float(hit_pos.mean()),
            "both_modes_rate": float(np.mean((hit_neg > 0) & (hit_pos > 0))),
            "per_state_abs_mean": np.abs(samples).mean(axis=1).tolist(),
        },
    }
    for tag, value in [
        ("toy/deterministic_nearest_mode_dist", summary["deterministic"]["nearest_mode_dist"]),
        ("toy/deterministic_hit_rate", summary["deterministic"]["hit_rate"]),
        ("toy/diffusion_nearest_mode_dist", summary["diffusion"]["nearest_mode_dist"]),
        ("toy/diffusion_hit_rate", summary["diffusion"]["hit_rate"]),
        ("toy/diffusion_both_modes_rate", summary["diffusion"]["both_modes_rate"]),
    ]:
        ctx.tb.add_scalar(tag, value, 0)

    ctx.logger.info(
        "[玩具] 双峰任务（μ(s)=%.1f+%.1f·s）| 确定性 MSE: 平均|a|=%.3f, 到最近模态 %.3f, 命中率 %.1f%% "
        "| 扩散: 平均|a|=%.3f, 到最近模态 %.3f, 命中率 %.1f%%（负峰 %.1f%% / 正峰 %.1f%%）, 双峰覆盖 %.1f%%",
        float(toy_cfg.get("mode_center", 1.0)), float(toy_cfg.get("mode_slope", 0.5)),
        summary["deterministic"]["mean_abs_action"], summary["deterministic"]["nearest_mode_dist"],
        100.0 * summary["deterministic"]["hit_rate"],
        summary["diffusion"]["mean_abs_action"], summary["diffusion"]["nearest_mode_dist"],
        100.0 * summary["diffusion"]["hit_rate"],
        100.0 * summary["diffusion"]["hit_rate_neg"], 100.0 * summary["diffusion"]["hit_rate_pos"],
        100.0 * summary["diffusion"]["both_modes_rate"],
    )
    return summary


def dataset_ambiguity_reference(episodes, expert_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """数据侧参照：**方向歧义**状态上的真实标签分布（证明 ``p(a|s)`` 真的双峰）。

    * ``frac_neg_kick`` / ``frac_pos_kick``：真实标签落在 ``±kick_torque`` 的比例；
    * ``frac_near_zero``：真实标签接近 0 的比例（专家**从不**输出这种"平均动作"）；
    * ``ambiguous_ratio``：歧义状态占全部样本的比例（关键状态往往只占极小比例，
      却决定整个回合能否起步 —— 这正是"均值回归"最致命的地方）。
    """
    states = np.concatenate([s for s, _ in episodes], axis=0)
    actions = np.concatenate([a for _, a in episodes], axis=0).reshape(-1)
    kick = float(expert_cfg.get("kick_torque", 2.0))
    mask = ambiguity_mask(
        states,
        kick_threshold=float(expert_cfg.get("kick_threshold", 1.0)),
        hang_margin=float(expert_cfg.get("hang_margin", 0.5)),
    )
    labels = actions[mask]
    n_amb = int(labels.size)
    return {
        "n_steps": int(states.shape[0]),
        "ambiguous_steps": n_amb,
        "ambiguous_ratio": float(mask.mean()),
        "n_pos_kick": int((labels > kick * 0.5).sum()),
        "n_neg_kick": int((labels < -kick * 0.5).sum()),
        "frac_pos_kick": float((labels > kick * 0.5).mean()) if n_amb else 0.0,
        "frac_neg_kick": float((labels < -kick * 0.5).mean()) if n_amb else 0.0,
        "frac_near_zero": float((np.abs(labels) < 0.5).mean()) if n_amb else 0.0,
        "probe_states": states[mask][:64].tolist(),
    }


def run_sparse_data_ablation(
    ctx, episodes, config, chunk_size: int, sparse_cfg: Dict[str, Any],
    expert_cfg: Dict[str, Any], probe_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """稀疏演示数据消融：**数据少 ⇒ 均值陷阱现形**。

    动机（Task 2 的核心机理）：数据充足时，高容量回归器可能为**每一条演示**分别记住
    "这一条是往左、那一条是往右"，从而绕开"取均值"；数据变少后，同一个状态邻域里
    只剩互相矛盾的标签，回归器只能输出两者的中间值 —— 一个**不可行**的动作。此时
    扩散策略仍在采样 ``p(a|s)``，因此应当明显更稳。

    做法：取前 ``data_fraction`` 比例的轨迹（同一子集同时喂给两种策略），用更少的 epoch
    训练"确定性 MLP 动作块"与"扩散策略"，然后做 20 回合闭环评估 + 未见过歧义状态探针。
    """
    fraction = float(sparse_cfg.get("data_fraction", 0.16))
    n_sub = max(1, int(round(len(episodes) * fraction)))
    epochs = int(sparse_cfg.get("epochs", 120))
    sub_episodes = episodes[:n_sub]
    n_steps = int(sum(len(s) for s, _ in sub_episodes))
    ctx.logger.info(
        "[消融] 稀疏数据：%d / %d 条轨迹（%d 步，约 %.0f%% 数据），两种策略各训练 %d epoch",
        n_sub, len(episodes), n_steps, 100.0 * fraction, epochs,
    )

    chunk_size = int(chunk_size)
    train_loader, val_loader, info = build_chunking_dataloaders(
        sub_episodes, chunk_size=chunk_size,
        train_ratio=float(ctx.data_cfg.get("train_ratio", 0.8)),
        batch_size=int(ctx.train_cfg.get("batch_size", 64)),
        num_workers=int(ctx.train_cfg.get("num_workers", 0)),
        augment=False, seed=ctx.seed, return_valid=True,
    )
    base_train_loader, base_val_loader, _ = build_chunking_dataloaders(
        sub_episodes, chunk_size=chunk_size,
        train_ratio=float(ctx.data_cfg.get("train_ratio", 0.8)),
        batch_size=int(ctx.train_cfg.get("batch_size", 64)),
        num_workers=int(ctx.train_cfg.get("num_workers", 0)),
        augment=False, seed=ctx.seed, return_valid=False,
    )
    ctx.logger.info("[消融] 稀疏数据训练样本 %d | 验证样本 %d", info["n_train"], info["n_val"])

    det_model, det_stats = train_deterministic_baseline(
        base_train_loader, base_val_loader, ctx, epochs=epochs,
        tag="ablation/sparse_baseline_mse", log_prefix="[消融-确定性]",
    )
    policy, policy_stats = train_diffusion_model(
        ctx, config, train_loader, val_loader, epochs=epochs,
        tag="ablation/sparse_diffusion_mse", log_prefix="[消融-扩散]",
    )

    execution = str((config.get("inference", {}) or {}).get("execution", "chunk")).lower()
    result: Dict[str, Any] = {
        "data_fraction": fraction, "n_trajectories": n_sub, "n_steps": n_steps,
        "epochs": epochs, "deterministic_train": det_stats, "diffusion_train": policy_stats,
    }
    result["closed_loop"] = {
        "diffusion": run_closed_loop(policy, ctx, chunk_size, execution, 0,
                                     label="闭环评估（消融｜稀疏数据｜扩散策略）",
                                     tb_tag="ablation/sparse_mean_return_diffusion"),
        "deterministic": run_closed_loop(det_model, ctx, chunk_size, execution, 0,
                                         label="闭环评估（消融｜稀疏数据｜确定性 MLP）",
                                         tb_tag="ablation/sparse_mean_return_deterministic"),
    }
    probe_states = sample_ambiguous_states(
        env_name=ctx.env_name, n_states=int(probe_cfg.get("n_states", 8)),
        seed=int(probe_cfg.get("state_seed", 10_000)),
        kick_threshold=float(expert_cfg.get("kick_threshold", 1.0)),
        hang_margin=float(expert_cfg.get("hang_margin", 0.5)),
    )
    if probe_states.shape[0] > 0:
        kick = float(expert_cfg.get("kick_torque", 2.0))
        result["probe"] = {
            "diffusion": multimodal_probe(
                policy, probe_states, mode_centers=(-kick, kick),
                n_samples=int(probe_cfg.get("n_samples", 256)),
                hit_tol=float(probe_cfg.get("hit_tol", 0.3)),
                device=ctx.device, stochastic=True)["summary"],
            "deterministic": multimodal_probe(
                det_model, probe_states, mode_centers=(-kick, kick),
                n_samples=int(probe_cfg.get("n_samples", 256)),
                hit_tol=float(probe_cfg.get("hit_tol", 0.3)),
                device=ctx.device, stochastic=False)["summary"],
        }
        ctx.logger.info(
            "[消融] 未见过歧义状态探针 | 扩散: 平均|a|=%.3f, 到最近模态 %.3f, 命中率 %.1f%% | "
            "确定性: 平均|a|=%.3f, 到最近模态 %.3f, 命中率 %.1f%%",
            result["probe"]["diffusion"]["mean_abs_action"],
            result["probe"]["diffusion"]["nearest_mode_dist"],
            100.0 * result["probe"]["diffusion"]["hit_rate"],
            result["probe"]["deterministic"]["mean_abs_action"],
            result["probe"]["deterministic"]["nearest_mode_dist"],
            100.0 * result["probe"]["deterministic"]["hit_rate"],
        )
    return result


# =============================================================================
# 7. 结论汇总（把"判断"与"数字"一起写进报告，避免只贴数字不下结论）
# =============================================================================
def compare_policies(report: Dict[str, Any]) -> Dict[str, Any]:
    """比较扩散与确定性对照的闭环回报，给出**统计上是否真的不同**的判断。

    判据：均值差 ``|Δ|`` 需大于两者回合间标准差均值，才认为"确有差异"；否则视为
    同一水平（与中文读本 §7"判断两结果是否真的不同"的口径一致）。
    """
    closed = report.get("closed_loop", {})
    if "diffusion" not in closed or "deterministic" not in closed:
        return {}
    d, b = closed["diffusion"], closed["deterministic"]
    delta = float(d["mean_return"] - b["mean_return"])
    pooled = 0.5 * float(d["std_return"] + b["std_return"])
    significant = bool(abs(delta) > pooled)
    if delta > 0 and significant:
        verdict = "扩散显著更优"
    elif delta > 0:
        verdict = "扩散更优，但均值差未超过回合间波动"
    elif significant:
        verdict = "确定性显著更优（本设定下多峰的收益不抵采样代价）"
    else:
        verdict = "两者相当"
    return {
        "mean_return_diff": delta,
        "pooled_std": pooled,
        "significant": significant,
        "significantly_better": bool(delta > 0 and significant),
        "failed_episodes_diffusion": int(d["failed_episodes"]),
        "failed_episodes_deterministic": int(b["failed_episodes"]),
        "conclusion": (
            f"扩散 {d['mean_return']:.1f} vs 确定性 {b['mean_return']:.1f}"
            f"（Δ={delta:+.1f}, 回合间 std 均值 {pooled:.1f}）→ {verdict}"
        ),
    }



# =============================================================================
# 8. 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("极简条件扩散策略（Mini-Diffusion Policy）训练脚本（配置驱动）", CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    model_cfg = config.get("model", {})
    diff_cfg = config.get("diffusion", {}) or {}
    infer_cfg = config.get("inference", {}) or {}
    probe_cfg = config.get("probe", {}) or {}
    expert_cfg = (config.get("data", {}) or {}).get("expert", {}) or {}

    chunk_size = resolve_chunk_size(config)                  # 动作块长度 K（与 Task 1 同一解析逻辑）
    action_dim = int(model_cfg.get("action_dim", 1))

    ctx = prepare_run(
        args, config, title=TITLE, logger_name="diffusion_policy",
        extra_info=[
            f"动作块长度 : {chunk_size}  (K * action_dim = {chunk_size * action_dim} 维输出)",
            f"扩散过程   : T={int(diff_cfg.get('num_train_timesteps', 100))} 步加噪 | "
            f"调度表={diff_cfg.get('schedule', 'cosine')} | 推理 N={int(diff_cfg.get('sample_steps', 16))} 步"
            f" | 采样器={str(diff_cfg.get('sampler', 'ddim')).upper()}",
            f"多峰专家   : mode_policy={(config.get('data', {}) or {}).get('mode_policy', 'random')} | "
            f"kick_threshold={expert_cfg.get('kick_threshold', 1.0)} | "
            f"hang_margin={expert_cfg.get('hang_margin', 0.5)} | "
            f"kick_torque={expert_cfg.get('kick_torque', 2.0)}",
        ],
    )
    logger, device, seed = ctx.logger, ctx.device, ctx.seed
    train_cfg, data_cfg, aug_cfg = ctx.train_cfg, ctx.data_cfg, ctx.aug_cfg

    # -------- 1/6 多峰专家轨迹 --------
    logger.info("[1/6] 准备**多峰**专家轨迹（同一起步状态存在 ±两个可行摆起方向）...")
    cache_path = os.path.join(
        ctx.paths.get("data_dir", "data/expert_pendulum"),
        data_cfg.get("cache_file", "expert_pendulum_multimodal_traj.npz"),
    )
    episodes, modes = collect_multimodal_expert_trajectories(
        env_name=ctx.env_name,
        num_expert_steps=int(data_cfg.get("num_expert_steps", 10000)),
        seed=seed,
        reuse_cache=bool(data_cfg.get("reuse_cache", True)),
        cache_path=cache_path,
        mode_policy=str(data_cfg.get("mode_policy", "random")),
        expert_kwargs=expert_cfg,
    )
    lengths = [len(s) for s, _ in episodes]
    logger.info(
        "多峰专家轨迹 %d 条 / 共 %d 步 | 每条长度 %d ~ %d | 模式分布 %s",
        len(episodes), int(sum(lengths)), min(lengths), max(lengths),
        np.bincount(modes, minlength=2).tolist(),
    )
    reference = dataset_ambiguity_reference(episodes, expert_cfg)
    logger.info(
        "方向歧义状态 %d / %d = %.2f%% | 真实标签: +kick %.1f%% / -kick %.1f%% / |a|<0.5 %.1f%%"
        "（p(a|s) 在该区域确实双峰）",
        reference["ambiguous_steps"], reference["n_steps"], 100.0 * reference["ambiguous_ratio"],
        100.0 * reference["frac_pos_kick"], 100.0 * reference["frac_neg_kick"],
        100.0 * reference["frac_near_zero"],
    )



    # -------- 2/6 动作块 DataLoader（扩散需 valid 掩码；确定性对照走 Task 1 的两元组路径）--------
    loader_kwargs = dict(
        chunk_size=chunk_size,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        augment=bool(aug_cfg.get("enabled", False)),
        state_noise_std=float(aug_cfg.get("state_noise_std", 0.0)),
        action_noise_std=float(aug_cfg.get("action_noise_std", 0.0)),
        seed=seed,
    )
    train_loader, val_loader, info = build_chunking_dataloaders(
        episodes, return_valid=True, **loader_kwargs
    )
    base_train_loader, base_val_loader, _ = build_chunking_dataloaders(
        episodes, return_valid=False, **loader_kwargs
    )
    logger.info(
        "[2/6] 训练样本 %d | 验证样本 %d | 动作块标签 (K=%d, action_dim=%d) | "
        "零填充 %.2f%%（训练集 %.2f%%）",
        info["n_train"], info["n_val"], chunk_size, action_dim,
        100.0 * info["padding_ratio"], 100.0 * info["train_padding_ratio"],
    )
    ctx.tb.add_scalar("chunk/padding_ratio", info["padding_ratio"], 0)

    # -------- 3/6 扩散策略（噪声预测网络 + 调度表）+ 优化器 / 检查点 --------
    policy = build_diffusion_policy(config).to(device)
    policy.schedule.to(device)                    # 调度表不是 nn.Module，需显式搬到设备
    if int(policy.chunk_size) != chunk_size:
        raise ValueError(f"扩散策略动作块 K={policy.chunk_size} 与解析出的 K={chunk_size} 不一致")
    n_params = count_parameters(policy)
    logger.info("[3/6] 噪声预测网络 ε_θ(s, a^t, t) | 可训练参数量: %d | 每次规划网络前向 %d 次",
                n_params, policy.sample_cost)
    ctx.tb.add_scalar("model/n_params", n_params, 0)

    optimizer = optim.Adam(
        policy.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}))
    checkpointer = ctx.build_checkpointer()
    step_fn = make_diffusion_step_fn(policy, device)
    start_epoch = restore_from_checkpoint(ctx, policy, optimizer, scheduler, checkpointer)

    # -------- 4/6 训练（噪声预测 MSE；引擎 / 优雅停机 / 检查点全部复用 Task 1 的实现）--------
    stats = train_loop(
        policy, train_loader, val_loader, None, optimizer, scheduler, checkpointer, ctx,
        start_epoch=start_epoch,
        resume_hint=f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}",
        loss_label="noise-MSE",
        train_step_fn=step_fn, val_step_fn=step_fn,
    )
    interrupted = bool(stats["interrupted"])
    logger.info("训练结束，用时 %.1fs | 已完成 epoch: %d | 最佳验证集 noise-MSE: %.6f",
                stats["train_time"], stats["last_epoch"], stats["best_metric"])
    ctx.tb.add_scalar("train_time_sec", stats["train_time"], 0)

    report: Dict[str, Any] = {
        "experiment": ctx.experiment_name,
        "run_dir": ctx.run_dir,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "epochs": int(ctx.epochs),
        "seed": int(seed),
        "diffusion": {
            "num_train_timesteps": int(diff_cfg.get("num_train_timesteps", 100)),
            "schedule": str(diff_cfg.get("schedule", "cosine")),
            "sample_steps": int(diff_cfg.get("sample_steps", 16)),
            "sampler": str(diff_cfg.get("sampler", "ddim")),
            "eta": float(diff_cfg.get("eta", 0.0)),
            "n_params": n_params,
        },
        "reference": reference,
        "train_stats": dict(stats),
    }


    # -------- 5/6 加载最优权重 + 多峰探针 + 闭环评估 + 推理开销 + 消融 --------
    if interrupted:
        logger.info("[5/6] 因安全中断退出，跳过评估（--resume 续训完成后再评估）")
    elif ctx.no_eval:
        logger.info("[5/6] 已跳过评估 (--no-eval)")
    elif not os.path.isfile(checkpointer.best_path):
        logger.warning("[5/6] 未找到最优权重 %s，跳过评估", checkpointer.best_path)
    else:
        logger.info("[5/6] 加载最优权重并开始评估 ...")
        Checkpointer.load(checkpointer.best_path, policy, map_location=device)
        policy.schedule.to(device)

        # ---- (a) 扩散损失与"动作块重建误差" ----
        val_metrics = evaluate_noise_loss(policy, val_loader, device)
        report["val_metrics"] = val_metrics
        logger.info("验证集 | 噪声预测 MSE: %.6f | 动作块重建 MSE(x0): %.6f",
                    val_metrics["noise_mse"], val_metrics["x0_mse"])
        ctx.tb.add_scalar("eval/val_noise_mse", val_metrics["noise_mse"], 0)
        ctx.tb.add_scalar("eval/val_x0_mse", val_metrics["x0_mse"], 0)

        # ---- (b) 确定性对照策略（同一份数据 / 同一训练预算）----
        baseline_model, baseline_stats = None, {}
        if bool(infer_cfg.get("baseline", True)):
            baseline_model, baseline_stats = train_deterministic_baseline(
                base_train_loader, base_val_loader, ctx
            )
            report["deterministic_baseline"] = baseline_stats

        # ---- (c) 多峰探针：**训练未见过**的歧义初始状态上的输出分布 ----
        if bool(probe_cfg.get("enabled", True)):
            n_probe = int(probe_cfg.get("n_states", 8))
            probe_states = sample_ambiguous_states(
                env_name=ctx.env_name, n_states=n_probe,
                seed=int(probe_cfg.get("state_seed", 10_000)),
                kick_threshold=float(expert_cfg.get("kick_threshold", 1.0)),
                hang_margin=float(expert_cfg.get("hang_margin", 0.5)),
            )
            if probe_states.shape[0] == 0:                     # 退化保护：用训练集里的歧义状态
                probe_states = np.asarray(reference["probe_states"][:n_probe], dtype=np.float32)
            kick = float(expert_cfg.get("kick_torque", 2.0))
            centers = (-kick, kick)
            report["probe"] = {
                "probe_states": probe_states.tolist(),
                "diffusion": multimodal_probe(
                    policy, probe_states, mode_centers=centers,
                    n_samples=int(probe_cfg.get("n_samples", 256)),
                    hit_tol=float(probe_cfg.get("hit_tol", 0.3)),
                    device=device, stochastic=True,
                ),
                "deterministic": None,
            }
            if baseline_model is not None:
                report["probe"]["deterministic"] = multimodal_probe(
                    baseline_model, probe_states, mode_centers=centers,
                    n_samples=int(probe_cfg.get("n_samples", 256)),
                    hit_tol=float(probe_cfg.get("hit_tol", 0.3)),
                    device=device, stochastic=False,
                )
            d_sum = report["probe"]["diffusion"]["summary"]
            logger.info(
                "多峰探针（%d 个**未见过的**歧义初始状态 × %d 次采样）| 扩散: 平均|a|=%.3f | "
                "采样 std=%.3f | 命中任一模态 %.1f%% | 到最近模态 %.3f",
                d_sum["n_states"], d_sum["n_samples"], d_sum["mean_abs_action"],
                d_sum["mean_std_action"], 100.0 * d_sum["hit_rate"], d_sum["nearest_mode_dist"],
            )
            if report["probe"]["deterministic"] is not None:
                b_sum = report["probe"]["deterministic"]["summary"]
                logger.info(
                    "多峰探针（确定性对照）| 平均|a|=%.3f | 采样 std=%.3f | 命中任一模态 %.1f%% | "
                    "到最近模态 %.3f（= 与两个可行动作的平均距离）",
                    b_sum["mean_abs_action"], b_sum["mean_std_action"],
                    100.0 * b_sum["hit_rate"], b_sum["nearest_mode_dist"],
                )
                ctx.tb.add_scalar("probe/diffusion_hit_rate", 100.0 * d_sum["hit_rate"], 0)
                ctx.tb.add_scalar("probe/deterministic_hit_rate", 100.0 * b_sum["hit_rate"], 0)
                ctx.tb.add_scalar("probe/diffusion_nearest_mode_dist", d_sum["nearest_mode_dist"], 0)
                ctx.tb.add_scalar("probe/deterministic_nearest_mode_dist",
                                  b_sum["nearest_mode_dist"], 0)

        # ---- (d) 闭环评估（与 Task 1 同一执行调度）+ 推理开销基准 ----
        execution = str(infer_cfg.get("execution", "chunk")).lower()
        replan_every = int(infer_cfg.get("replan_every", 0) or 0)
        horizon = replan_every or chunk_size
        report["closed_loop"], report["latency"], report["ablation"] = {}, {}, []
        report["closed_loop"]["diffusion"] = run_closed_loop(
            policy, ctx, chunk_size, execution, replan_every,
            label=f"闭环评估（扩散 {str(diff_cfg.get('sampler', 'ddim')).upper()} "
                  f"N={int(diff_cfg.get('sample_steps', 16))}，每 {horizon} 步规划一次）",
            tb_tag="eval/mean_return",
        )
        report["latency"]["diffusion"] = benchmark_plan_latency(
            policy, device, state_dim=int(model_cfg.get("state_dim", 3)),
            repeats=int(infer_cfg.get("timing_repeats", 300)),
        )
        if baseline_model is not None:
            report["closed_loop"]["deterministic"] = run_closed_loop(
                baseline_model, ctx, chunk_size, execution, replan_every,
                label=f"闭环评估（确定性 MLP 动作块对照，每 {horizon} 步规划一次）",
                tb_tag="eval/mean_return_deterministic",
            )
            report["latency"]["deterministic"] = benchmark_plan_latency(
                baseline_model, device, state_dim=int(model_cfg.get("state_dim", 3)),
                repeats=int(infer_cfg.get("timing_repeats", 300)),
            )
        report["compare"] = compare_policies(report)

        # ---- (d2) 规划一致性：多峰采样的"时序一致性代价" ----
        if bool(probe_cfg.get("consistency", True)):
            report["consistency"] = {
                "diffusion": analyze_plan_consistency(policy, ctx, chunk_size),
                "deterministic": (analyze_plan_consistency(baseline_model, ctx, chunk_size)
                                  if baseline_model is not None else None),
            }
            c_d = report["consistency"]["diffusion"]
            logger.info(
                "规划一致性（%d 回合）| 扩散: 每回合块首动作符号翻转 %.1f 次 / %.0f 次规划"
                "（翻转率 %.1f%%）",
                c_d["n_episodes"], c_d["flips_per_episode"], c_d["plans_per_episode"],
                100.0 * c_d["flip_rate"],
            )
            if report["consistency"]["deterministic"] is not None:
                c_b = report["consistency"]["deterministic"]
                logger.info(
                    "规划一致性（确定性对照）| 每回合翻转 %.1f 次 / %.0f 次规划（翻转率 %.1f%%）",
                    c_b["flips_per_episode"], c_b["plans_per_episode"], 100.0 * c_b["flip_rate"],
                )
                ctx.tb.add_scalar("consistency/diffusion_flip_rate",
                                  100.0 * c_d["flip_rate"], 0)
                ctx.tb.add_scalar("consistency/deterministic_flip_rate",
                                  100.0 * c_b["flip_rate"], 0)

        # ---- (e) 消融：推理去噪步数 N 与重规划间隔 ----
        compare_cfg = infer_cfg.get("compare", {}) or {}
        for n_steps in list(compare_cfg.get("sample_steps", []) or []):
            n_steps = int(n_steps)
            policy.set_sample_steps(n_steps)
            res = run_closed_loop(
                policy, ctx, chunk_size, execution, replan_every,
                label=f"闭环评估（扩散 N={n_steps} 去噪步，每 {horizon} 步规划一次）",
                tb_tag=f"eval/mean_return_steps{n_steps}",
            )
            report["ablation"].append({
                "type": "sample_steps", "sample_steps": n_steps,
                "mean_return": res["mean_return"], "std_return": res["std_return"],
                "plan_per_episode": res["forward_per_episode"],
                "network_forward_per_episode": res["n_network_forwards"] / max(1, res["n_episodes"]),
            })
        policy.set_sample_steps(int(diff_cfg.get("sample_steps", 16)))
        for plan_every in list(compare_cfg.get("replan_every", []) or []):
            plan_every = int(plan_every)
            if plan_every == replan_every:
                continue
            res = run_closed_loop(
                policy, ctx, chunk_size, execution, plan_every,
                label=f"闭环评估（扩散，每 {plan_every or chunk_size} 步重新规划）",
                tb_tag=f"eval/mean_return_replan{plan_every or chunk_size}",
            )
            report["ablation"].append({
                "type": "replan_every", "replan_every": plan_every,
                "mean_return": res["mean_return"], "std_return": res["std_return"],
                "plan_per_episode": res["forward_per_episode"],
                "network_forward_per_episode": res["n_network_forwards"] / max(1, res["n_episodes"]),
            })

        latency = report["latency"]
        if "deterministic" in latency:
            logger.info(
                "推理耗时   | 确定性 MLP: %.3f ms/次规划 | 扩散(N=%d): %.3f ms/次规划 | 慢 %.1f 倍",
                latency["deterministic"]["ms_per_plan"], int(diff_cfg.get("sample_steps", 16)),
                latency["diffusion"]["ms_per_plan"],
                latency["diffusion"]["ms_per_plan"] / max(1e-9, latency["deterministic"]["ms_per_plan"]),
            )
        else:
            logger.info("推理耗时   | 扩散(N=%d): %.3f ms/次规划",
                        int(diff_cfg.get("sample_steps", 16)), latency["diffusion"]["ms_per_plan"])

    # -------- 6/6 稀疏数据消融 + 双峰玩具任务（多峰机理的两组可验证证据）--------
    sparse_cfg = (config.get("ablation", {}) or {}).get("sparse", {}) or {}
    if bool(sparse_cfg.get("enabled", False)) and not interrupted and not ctx.no_eval:
        logger.info("[6/6] 稀疏数据消融：数据少 ⇒ 均值陷阱现形 ...")
        report["ablation_sparse"] = run_sparse_data_ablation(
            ctx, episodes, config, chunk_size, sparse_cfg, expert_cfg, probe_cfg
        )

    toy_cfg = probe_cfg.get("toy", {}) or {}
    if bool(toy_cfg.get("enabled", False)) and not interrupted:
        logger.info("[6/6] 双峰玩具任务：多峰机理的最小可验证例子（1 维状态 / 1 维动作）...")
        report["toy"] = run_bimodal_toy_demo(ctx, toy_cfg, diff_cfg)

    report_path = os.path.join(ctx.run_dir, "diffusion_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    logger.info("评估明细（含逐回合回报 / 探针样本 / 消融）已写入: %s", report_path)

    extra_lines: List[str] = []
    closed = report.get("closed_loop", {})
    for key, label in (("diffusion", "扩散策略"), ("deterministic", "确定性 MLP")):
        if key in closed:
            extra_lines.append(
                f"闭环平均回报({label:<10}) : {closed[key]['mean_return']:.2f} ± "
                f"{closed[key]['std_return']:.2f}  [失败回合 {closed[key]['failed_episodes']}/"
                f"{closed[key]['n_episodes']}]"
            )
    if report.get("compare"):
        extra_lines.append(f"对比结论   : {report['compare']['conclusion']}")
    if report.get("toy"):
        extra_lines.append(
            "双峰玩具   : 确定性到最近模态 "
            f"{report['toy']['deterministic']['nearest_mode_dist']:.2f} -> 扩散 "
            f"{report['toy']['diffusion']['nearest_mode_dist']:.3f}（命中率 "
            f"{100.0 * report['toy']['diffusion']['hit_rate']:.1f}%）"
        )

    finish_run(
        ctx, checkpointer, stats,
        eval_result=closed.get("diffusion"),
        extra_lines=extra_lines,
        resume_command=f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}",
        metric_label="noise-MSE",
    )


if __name__ == "__main__":
    main()

