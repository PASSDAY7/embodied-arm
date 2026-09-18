#!/usr/bin/env python
"""PPO（PPO-Clip + GAE-λ）训练脚本 —— W8 Task 1。

任务
----
在 **Pendulum-v1**（连续动作空间 ``[-2.0, 2.0]``）上从零手写 PPO：状态 ``[cos θ, sin θ, θ̇]``，
动作为单维力矩；随机策略回报约 −1200 ~ −1600，收敛策略应稳定在 −200 以上。

算法流程（一次迭代 = 采样 + 多轮小批量更新）
-------------------------------------------
1. **Rollout**：用 ``π_θold`` 与环境交互 ``rollout_steps=2048`` 步，缓存
   ``(s, a_raw, logπ_old, r, terminated, truncated, V_old)``（:class:`src.ppo.RolloutCollector`）；
2. **GAE-λ**：从 ``T−1`` 倒序递推 ``δ_t = r_t + γV(s_{t+1}) − V(s_t)``、
   ``Â_t = δ_t + γλ·Â_{t+1}``，并令 ``Target_t = Â_t + V(s_t)``；
3. **更新**：把 2048 步打乱切成 32 个 mini-batch，重复 ``update_epochs=10`` 轮，
   最小化 ``L^CLIP + c₁·L^VF − c₂·H``（:func:`src.ppo.ppo_update`）；
4. **评估 / 存盘**：每 ``eval.every_rollouts`` 个 rollout 做一次独立评估（均值动作、
   固定种子）；评估平均回报 ≥ ``training.target_reward``（默认 −200）即提前结束。

思考题（连续动作空间的截断与 log_prob）
------------------------------------
采样得到 ``a_raw ~ N(μ, σ²)`` 后必须 ``a_env = clip(a_raw, −2, 2)`` 才能喂给环境，
但 ``logπ`` 一定要算在 **``a_raw``** 上，不能用 ``a_env``：

* ``a_env`` 不服从该高斯（``±2`` 处还有概率质量），用它算密度既不是"执行动作的密度"，
  也破坏"积分变量与密度一致"这一策略梯度定理的前提；
* 高斯的得分函数本来是零均值的（``E[(a−μ)/σ²] = 0``），换成 ``clip(a)`` 后期望变成
  ``(E[clip(a)] − μ)/σ²`` —— 一旦 ``μ`` 越过边界就产生**系统性伪梯度**，把 ``μ`` 越推越远；
* PPO 更新时要重算 ``logπ_θ(a_t)``：若缓存的是 ``a_env``，第 0 个 epoch 的比值
  ``r_t ≠ 1``（新旧 logπ 算在**不同动作点**），裁剪区间失去基准；缓存 ``a_raw`` 则恒有 ``r_t = 1``。

实现上：``RolloutBuffer.actions_raw`` 存 ``a_raw``，只有 ``env.step`` 拿到 ``clip`` 后的值
（见 ``tests/test_ppo.py`` 的 ``test_env_action_is_clipped_but_buffer_keeps_raw`` /
``test_clipped_action_log_prob_is_biased`` / ``test_ratio_is_one_before_any_update``）。
日志里的 ``train/action_clip_fraction`` 实时显示"有多少比例的采样越界"，用于评估截断的影响面。

工程实现
--------
``--config/--set/--resume`` 解析、运行目录 / 日志 / TensorBoard、优雅停机与 Checkpointer
全部复用 ``src/engine.py``；算法核心（ActorCritic / RolloutBuffer / GAE / PPO 损失 /
采样器）在 ``src/ppo.py``，与 W3/W4/W6 共享同一套工程骨架。额外把 **状态归一化统计量**、
**累计交互步数**、**环境重置计数** 一起写进检查点 ``extras``，因此 ``--resume`` 能续上
学习率、归一化分布与可复现的初始状态序列。

用法示例::

    python experiments/train_ppo.py                                      # 默认配置（seed 42 / cuda）
    python experiments/train_ppo.py --device cpu                         # 无 GPU 时
    python experiments/train_ppo.py --set training.total_timesteps=40000  # 快速验证
    python experiments/train_ppo.py --set model.normalize_obs=false      # 消融：关掉状态归一化
    python experiments/train_ppo.py --set training.bootstrap_on_truncation=false  # 消融：把 truncated 当终止
    python experiments/train_ppo.py --set model.shared_backbone=true     # 共享主干双头
    python experiments/train_ppo.py --resume                             # 断点续训（自动找最新检查点）
    python experiments/export_tb_curves.py --logdir logs/ppo_pendulum_*  # 导出 TensorBoard 曲线（PNG/CSV）

注意 ``--set`` 的写法（与 W6 一致）::

    * 一次 ``--set`` 后可跟多个 ``KEY=VALUE``（重复写 ``--set`` 只有最后一次生效）；
    * 布尔值 ``false`` / ``False`` 都能识别（``cast_value`` 走 ``ast.literal_eval``，
      本脚本再用 :func:`as_bool` 兜住小写字符串的坑）。
"""
from __future__ import annotations

import math
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# 允许脚本以 `python experiments/train_ppo.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import (  # noqa: E402
    apply_overrides,
    finish_run,
    load_experiment_config,
    parse_train_args,
    prepare_run,
    report_closed_loop,
    restore_from_checkpoint,
)
from src.ppo import (  # noqa: E402
    ActorCritic,
    RolloutBuffer,
    RolloutCollector,
    RunningMeanStd,
    build_actor_critic,
    build_ppo_optimizer,
    count_parameters,
    env_max_steps,
    evaluate_policy,
    make_env,
    ppo_update,
)
from src.utils import Checkpointer  # noqa: E402

CONFIG_PATH = os.path.join("configs", "ppo_config.yaml")
TITLE = "PPO（PPO-Clip + GAE-λ）实验"
SCRIPT_PATH = "experiments/train_ppo.py"


# =============================================================================
# 工具
# =============================================================================
def rolling_mean(values: List[float], window: int) -> float:
    """最近 ``window`` 个回合的平均回报（数据不足时按现有数据平均）。"""
    if not values:
        return 0.0
    return float(np.mean(values[-int(window):] if int(window) > 0 else values))


def as_bool(value: Any, default: bool = False) -> bool:
    """把配置里的布尔值规范化为 ``bool``（PyYAML 小写 ``false`` 会退化成字符串 ``"false"``）。"""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("false", "0", "no", "off", "none", ""):
            return False
        if lowered in ("true", "1", "yes", "on"):
            return True
        return bool(default)
    if value is None:
        return bool(default)
    return bool(value)


def optional_float(value: Any) -> Optional[float]:
    """把 ``null`` / ``none`` / 空串解析为 ``None``，其余转 ``float``（用于 ``value_clip_eps`` 等）。"""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "none", "null", "false"):
        return None
    return float(value)


def check_action_bounds(config: Dict[str, Any], env) -> Dict[str, Any]:
    """用**环境真实动作范围**校验 / 覆盖 ``model.action_low/high``。

    配置写错边界（例如把 Pendulum 写成 ``[-1, 1]``）不会报错、只会让策略默默变差，
    所以这里显式对齐并打印一条警告 —— 这类"静默错误"是连续控制实验最常见的坑之一。
    """
    model_cfg = config.setdefault("model", {})
    low = float(np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)[0])
    high = float(np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)[0])
    for key, truth in (("action_low", low), ("action_high", high)):
        cfg_value = model_cfg.get(key)
        if cfg_value is None or abs(float(cfg_value) - truth) > 1e-6:
            model_cfg[key] = truth
    return model_cfg

# =============================================================================
# 训练主循环（采样 → GAE → 多轮 mini-batch 更新 → 评估）
# =============================================================================
def train_ppo(
    ctx,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    normalizer: Optional[RunningMeanStd],
    checkpointer: Checkpointer,
    start_rollout: int = 1,
    steps_done: int = 0,
    resets_done: int = 0,
) -> Dict[str, Any]:
    """PPO 主循环，返回训练统计（供 :func:`src.engine.finish_run` 使用）。

    Args:
        ctx: :func:`src.engine.prepare_run` 装配的运行上下文。
        model / optimizer / buffer / normalizer: 由 ``main`` 构建（续训时已恢复状态）。
        checkpointer: 检查点管理器（``mode=max`` 时按**独立评估回报**刷新最优）。
        start_rollout: 起始 rollout 编号（续训 = 已完成 rollout 数 + 1）。
        steps_done: 续训前已累计的交互步数（用于 TensorBoard 横轴与检查点累计量）。
        resets_done: 续训前环境已重置次数（保证初始状态序列可复现地接续）。

    Returns:
        ``{"best_metric", "last_epoch", "interrupted", "early_stopped", "train_time",
           "steps_total", "resets_total", "rolling_return", "history", "last_stats",
           "last_eval", "eval_history"}``。
    """
    logger, tb, device = ctx.logger, ctx.tb, ctx.device
    train_cfg, eval_cfg = ctx.train_cfg, ctx.eval_cfg

    # ---- 超参（全部来自配置，便于消融）----
    total_timesteps = int(train_cfg.get("total_timesteps", 200000))
    rollout_steps = int(train_cfg.get("rollout_steps", 2048))
    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    clip_eps = float(train_cfg.get("clip_eps", 0.2))
    update_epochs = int(train_cfg.get("update_epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 64))
    value_coef = float(train_cfg.get("value_coef", 0.5))
    entropy_coef = float(train_cfg.get("entropy_coef", 0.0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    normalize_advantage = as_bool(train_cfg.get("normalize_advantage", True), True)
    bootstrap_on_truncation = as_bool(train_cfg.get("bootstrap_on_truncation", True), True)
    reward_scale = float(train_cfg.get("reward_scale", 1.0))
    value_clip_eps = optional_float(train_cfg.get("value_clip_eps"))
    target_kl = optional_float(train_cfg.get("target_kl"))
    drop_last = as_bool(train_cfg.get("drop_last_batch", False), False)
    log_interval = max(1, int(train_cfg.get("log_interval", 1)))
    target_reward = float(train_cfg.get("target_reward", -200.0))
    target_window = max(1, int(train_cfg.get("target_window", 100)))

    eval_interval = max(1, int(eval_cfg.get("every_rollouts", 5)))
    eval_episodes = ctx.n_eval_episodes
    eval_deterministic = as_bool(eval_cfg.get("deterministic", True), True)
    eval_seed = int(eval_cfg.get("seed", 12345))

    remaining_steps = max(0, total_timesteps - int(steps_done))
    n_rollouts = int(math.ceil(remaining_steps / rollout_steps)) if remaining_steps > 0 else 0
    last_rollout = start_rollout + max(0, n_rollouts - 1)

    env = make_env(ctx.env_name)
    collector = RolloutCollector(
        env, model, buffer, device=device, normalizer=normalizer,
        gamma=gamma, gae_lambda=gae_lambda, reward_scale=reward_scale,
        bootstrap_on_truncation=bootstrap_on_truncation, seed=ctx.seed,
        normalize_obs=normalizer is not None, n_resets=resets_done,
    )

    history: List[float] = []                 # 逐回合的**原始**回报（不受 reward_scale 影响）
    eval_history: List[Dict[str, float]] = []
    steps_total = int(steps_done)
    last_stats: Dict[str, float] = {}
    last_eval: Dict[str, float] = {}
    interrupted = False
    early_stopped = False
    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        if stop_requested["flag"]:
            logger.warning("再次收到信号 %s，强制退出（当前 rollout 进度将丢失）", signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        stop_requested["flag"] = True
        logger.warning("收到信号 %s：将在当前 rollout 结束后保存检查点并安全退出", signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handle_signal)
        except (ValueError, OSError, AttributeError):  # pragma: no cover - 非主线程等场景
            pass

    def _checkpoint(rollout_id: int, metric: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        """写 last（更优时写 best）；归一化统计量 / 累计步数 / 重置计数一起进 extras。"""
        extras = dict(extra)
        extras.update({
            "run_dir": ctx.run_dir, "seed": ctx.seed,
            "steps_total": int(steps_total), "collector_resets": int(collector.n_resets),
        })
        if normalizer is not None:
            extras["obs_normalizer_state"] = normalizer.state_dict()
        return checkpointer.step(model, optimizer, None, epoch=rollout_id, metric=float(metric),
                                 config=ctx.config, extras=extras)

    start_time = time.time()
    rollout = start_rollout - 1
    rolling = 0.0
    logger.info(
        "[2/3] 开始训练：rollout %d → %d（每个 %d 步）| 总预算 %d 步 | 已交互 %d 步",
        start_rollout, last_rollout, rollout_steps, total_timesteps, steps_total,
    )
    logger.info(
        "      超参数: γ=%.3f | λ=%.3f | ε=%.2f | K=%d | batch=%d | lr=(%.1e, %.1e) | "
        "clip_grad=%.2f | 优势标准化=%s | truncated 自举=%s",
        gamma, gae_lambda, clip_eps, update_epochs, batch_size,
        optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], max_grad_norm,
        "on" if normalize_advantage else "off", "on" if bootstrap_on_truncation else "off",
    )

    try:
        if n_rollouts == 0:
            logger.warning("总预算 %d 步已用完（已交互 %d 步），跳过训练", total_timesteps, steps_total)
        for i in range(n_rollouts):
            rollout = start_rollout + i
            steps_this = int(min(rollout_steps, total_timesteps - steps_total))
            t0 = time.time()

            # ---- 1) 采样一个 rollout（含 GAE 与截断/终止的分别处理）----
            roll = collector.collect(steps_this)
            steps_total += int(roll["n_steps"])
            history.extend(roll["episode_returns"])
            rolling = rolling_mean(history, target_window)
            bstats = buffer.stats()

            # ---- 2) 监控（横轴统一用"累计交互步数"，与 RL 论文曲线一致）----
            tb.add_scalar("train/steps_total", float(steps_total), steps_total)
            tb.add_scalar("train/rolling_return", rolling, steps_total)
            tb.add_scalar("train/action_clip_fraction", roll["action_clip_fraction"], steps_total)
            tb.add_scalar("train/last_value", roll["last_value"], steps_total)
            tb.add_scalar("train/std_mean", roll["std_mean"], steps_total)
            for ret in roll["episode_returns"]:
                tb.add_scalar("train/episode_return", float(ret), steps_total)
            if roll["episode_returns"]:
                tb.add_scalar("train/rollout_return_mean", float(np.mean(roll["episode_returns"])), steps_total)
            for key in ("reward_mean", "value_mean", "advantage_mean", "advantage_std",
                        "truncated_count", "terminated_count"):
                tb.add_scalar(f"buffer/{key}", bstats[key], steps_total)

            # ---- 3) PPO 多轮小批量更新 ----
            last_stats = ppo_update(
                model, optimizer, buffer,
                clip_eps=clip_eps, update_epochs=update_epochs, batch_size=batch_size,
                value_coef=value_coef, entropy_coef=entropy_coef, max_grad_norm=max_grad_norm,
                normalize_advantage=normalize_advantage, value_clip_eps=value_clip_eps,
                target_kl=target_kl, drop_last=drop_last,
            )
            rollout_time = time.time() - t0
            last_stats.update({
                "episode_return_mean": float(np.mean(roll["episode_returns"])) if roll["episode_returns"] else float("nan"),
                "steps_total": float(steps_total),
                "rollout_time": float(rollout_time),
                "std_mean": float(roll["std_mean"]),                 # 当前策略的 σ=exp(log_std)
                "action_clip_fraction": float(roll["action_clip_fraction"]),
            })
            for key in ("policy_loss", "value_loss", "total_loss", "entropy", "std_mean", "log_std_mean",
                        "approx_kl", "clip_fraction", "ratio_mean", "ratio_std", "grad_norm",
                        "explained_variance", "value_mean", "epochs_run", "kl_early_stop"):
                if key in last_stats:
                    tb.add_scalar(f"train/{key}", float(last_stats[key]), steps_total)

            if rollout % log_interval == 0 or i == 0 or rollout == last_rollout:
                logger.info(
                    "Rollout [%d/%d] | 步数 %6d | 回合回报均值 %8.2f | 滑动(%d) %8.2f | "
                    "策略损失 %7.4f | 价值损失 %9.3f | 熵 %.3f | KL %.5f | clip占比 %.3f | "
                    "梯度范数 %6.2f | 用时 %.1fs",
                    rollout, last_rollout, steps_total,
                    last_stats["episode_return_mean"], target_window, rolling,
                    last_stats["policy_loss"], last_stats["value_loss"], last_stats["entropy"],
                    last_stats["approx_kl"], last_stats["clip_fraction"], last_stats["grad_norm"],
                    rollout_time,
                )

            # ---- 4) 独立评估 + 存盘（模型选择用"评估回报"，不用训练回报）----
            if rollout % eval_interval == 0 or rollout == last_rollout:
                eval_result = evaluate_policy(
                    model, env_name=ctx.env_name, device=device, episodes=eval_episodes,
                    seed=eval_seed, gamma=gamma, deterministic=eval_deterministic,
                    normalizer=normalizer,
                )
                last_eval = eval_result
                eval_history.append(dict(eval_result))
                report_closed_loop(
                    ctx, eval_result, step=steps_total,
                    label=f"独立评估（{'均值动作' if eval_deterministic else '采样'}，{eval_episodes} 回合）",
                    extra_tags={
                        "eval/std_return": eval_result["std_return"],
                        "eval/min_return": eval_result["min_return"],
                        "eval/max_return": eval_result["max_return"],
                        "eval/mean_length": eval_result["mean_length"],
                        "eval/rolling_train_return": rolling,
                    },
                )
                save_info = _checkpoint(rollout, eval_result["mean_return"], {
                    "eval_return": eval_result["mean_return"], "rolling_return": rolling,
                })
                logger.info(
                    "检查点     | %s%s", save_info["last"],
                    "  *best" if save_info["is_best"] else f"  (历史最优 {save_info['best_metric']:.2f})",
                )
                if eval_result["mean_return"] >= target_reward:
                    early_stopped = True
                    logger.info(
                        "★ 评估平均回报 %.2f ≥ 目标 %.1f：任务已解决，提前结束训练"
                        "（rollout %d/%d，累计交互 %d 步）",
                        eval_result["mean_return"], target_reward, rollout, last_rollout, steps_total,
                    )
                    break

            # ---- 5) 按训练滑动均值提前停止（与评估判据互为交叉验证）----
            if rolling >= target_reward and len(history) >= target_window:
                early_stopped = True
                logger.info(
                    "★ 最近 %d 个训练回合平均回报 %.2f ≥ 目标 %.1f：提前结束训练"
                    "（rollout %d/%d，累计交互 %d 步）",
                    target_window, rolling, target_reward, rollout, last_rollout, steps_total,
                )
                break

            # ---- 6) 优雅停机（SIGINT / SIGTERM）----
            if stop_requested["flag"]:
                interrupted = True
                _checkpoint(rollout, rolling, {"rolling_return": rolling, "interrupted": True})
                logger.warning(
                    "已保存检查点并于 rollout %d 安全退出（累计 %d 步）；继续训练可执行：\n"
                    "  python %s --config %s --resume %s",
                    rollout, steps_total, SCRIPT_PATH, ctx.args.config, checkpointer.last_path,
                )
                break
    except Exception:
        # 异常也最大限度保留成果：先写应急检查点再向上抛
        _checkpoint(rollout, rolling_mean(history, target_window) if history else 0.0,
                    {"emergency": True, "rolling_return": rolling})
        logger.error("训练异常，已写应急检查点: %s", checkpointer.last_path)
        raise
    finally:
        env.close()

    train_time = time.time() - start_time
    stats: Dict[str, Any] = {
        "best_metric": float(checkpointer.best_metric) if checkpointer.best_metric is not None else 0.0,
        "last_epoch": int(rollout),
        "interrupted": bool(interrupted),
        "early_stopped": bool(early_stopped),
        "train_time": float(train_time),
        "steps_total": int(steps_total),
        "resets_total": int(collector.n_resets),
        "rolling_return": rolling_mean(history, target_window),
        "n_episodes": len(history),
        "history": history,
        "last_stats": last_stats,
        "last_eval": last_eval,
        "eval_history": eval_history,
    }
    logger.info(
        "训练结束，用时 %.1fs | 完成 rollout: %d | 累计交互步数: %d | 完成回合: %d | "
        "最近 %d 回合平均回报: %.2f",
        train_time, max(0, stats["last_epoch"] - start_rollout + 1), steps_total, len(history),
        min(target_window, len(history)), stats["rolling_return"],
    )
    return stats




# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("PPO（PPO-Clip + GAE-λ）训练脚本（配置驱动）", CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    train_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    eval_cfg = config.setdefault("eval", {})
    # 规范化布尔字段：`--set training.normalize_advantage=false`（小写）也能被正确识别
    model_cfg["normalize_obs"] = as_bool(model_cfg.get("normalize_obs", True), True)
    model_cfg["shared_backbone"] = as_bool(model_cfg.get("shared_backbone", False), False)
    train_cfg["normalize_advantage"] = as_bool(train_cfg.get("normalize_advantage", True), True)
    train_cfg["bootstrap_on_truncation"] = as_bool(train_cfg.get("bootstrap_on_truncation", True), True)
    eval_cfg["deterministic"] = as_bool(eval_cfg.get("deterministic", True), True)
    eval_cfg["render"] = as_bool(eval_cfg.get("render", False), False)

    lr_actor = float(train_cfg.get("lr_actor", 3e-4))
    lr_critic = float(train_cfg.get("lr_critic", 1e-3))
    target_reward = float(train_cfg.get("target_reward", -200.0))

    # -------- 基础设置：设备 / 种子 / 运行目录 / 日志 / TensorBoard / 续训解析 --------
    ctx = prepare_run(
        args, config, title=TITLE, logger_name="ppo",
        extra_info=[
            "算法       : PPO-Clip + GAE-λ（on-policy，数据由策略自己采样，无专家数据）",
            f"超参数     : γ={train_cfg.get('gamma', 0.99)} | λ={train_cfg.get('gae_lambda', 0.95)}"
            f" | ε={train_cfg.get('clip_eps', 0.2)} | K={train_cfg.get('update_epochs', 10)}"
            f" | batch={train_cfg.get('batch_size', 64)} | rollout={train_cfg.get('rollout_steps', 2048)}",
            f"学习率     : actor={lr_actor:.1e} | critic={lr_critic:.1e}"
            f" | max_grad_norm={train_cfg.get('max_grad_norm', 0.5)}"
            f" | 目标回报={target_reward}（评估平均回报）",
        ],
    )
    logger, device = ctx.logger, ctx.device

    # -------- 1. 环境 / 网络 / 优化器 / 缓存 --------
    probe = make_env(ctx.env_name)
    check_action_bounds(config, probe)
    logger.info(
        "[1/3] 环境: %s | 状态空间 %s | 动作空间 %s | 单回合上限 %d 步",
        ctx.env_name, probe.observation_space, probe.action_space, env_max_steps(probe),
    )
    probe.close()

    model = build_actor_critic(config).to(device)
    if model.shared_backbone:
        logger.info(
            "Actor-Critic: 共享主干 MLP(%d → %d×%d) + 双头 | 可训练参数: %d",
            model.state_dim, model.hidden_dim, max(1, model.num_layers - 1), count_parameters(model),
        )
    else:
        logger.info(
            "Actor 网络 : MLP(%d → %d×%d → %d) | 可训练参数: %d",
            model.state_dim, model.num_layers, model.hidden_dim, model.action_dim,
            count_parameters(model.actor),
        )
        logger.info(
            "Critic 网络: MLP(%d → %d×%d → 1) | 可训练参数: %d",
            model.state_dim, model.num_layers, model.hidden_dim, count_parameters(model.critic),
        )
    logger.info(
        "动作分布   : N(μ(s), σ²)，σ=exp(log_std)（可学习，夹紧到 [%.1f, %.1f]）| log_std 初值 %.2f"
        " | 动作截断 [%.2f, %.2f] | 状态归一化: %s",
        model.log_std_min, model.log_std_max, float(model.log_std_parameter().mean()),
        model.action_low, model.action_high,
        "on" if model_cfg["normalize_obs"] else "off",
    )


    optimizer = build_ppo_optimizer(
        model, lr_actor=lr_actor, lr_critic=lr_critic,
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    logger.info(
        "优化器     : Adam(param_groups=2, lr_actor=%.1e, lr_critic=%.1e, weight_decay=%.1e)"
        " | 策略参数张量 %d 个 / 价值参数张量 %d 个（逐参数更新 ⇒ 等价于两个独立 Adam）",
        lr_actor, lr_critic, float(train_cfg.get("weight_decay", 0.0)),
        len(optimizer.param_groups[0]["params"]), len(optimizer.param_groups[1]["params"]),
    )

    normalizer = RunningMeanStd(int(model_cfg.get("state_dim", 3))) if model_cfg["normalize_obs"] else None
    rollout_steps = int(train_cfg.get("rollout_steps", 2048))
    buffer = RolloutBuffer(
        capacity=rollout_steps, state_dim=int(model_cfg.get("state_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 1)), device=device,
    )
    logger.info(
        "Rollout缓存: capacity=%d 步 | state_dim=%d | action_dim=%d | device=%s | GAE: truncated 自举=%s",
        buffer.capacity, buffer.state_dim, buffer.action_dim, device,
        "on" if train_cfg["bootstrap_on_truncation"] else "off（把 200 步到点当终止，有偏）",
    )
    checkpointer = ctx.build_checkpointer()

    # -------- 2. 断点续训：恢复网络 / 优化器 / rollout 号 / 最优指标 / 随机源 --------
    start_rollout = restore_from_checkpoint(ctx, model, optimizer, None, checkpointer)
    steps_done, resets_done = 0, 0
    if ctx.resume_payload is not None:
        resume_extra = ctx.resume_payload.get("extra") or {}
        steps_done = int(resume_extra.get("steps_total") or 0)     # 累计交互预算跨续训保持
        resets_done = int(resume_extra.get("collector_resets") or 0)
        if normalizer is not None and resume_extra.get("obs_normalizer_state") is not None:
            normalizer.load_state_dict(resume_extra["obs_normalizer_state"])
            logger.info(
                "已恢复状态归一化统计量: count=%.0f | mean=%s | std=%s",
                normalizer.count, np.round(normalizer.mean, 3), np.round(normalizer.std, 3),
            )
        logger.info("已恢复累计交互步数: %d | 环境重置计数: %d", steps_done, resets_done)

    # -------- 3. 训练（采样 → GAE → 多轮 mini-batch 更新 → 评估）--------
    stats = train_ppo(
        ctx, model, optimizer, buffer, normalizer, checkpointer,
        start_rollout=start_rollout, steps_done=steps_done, resets_done=resets_done,
    )

    # -------- 4. 用"最优权重"做最终评估 --------
    eval_result: Optional[Dict[str, Any]] = None
    render = eval_cfg["render"]
    if ctx.no_eval:
        logger.info("[3/3] 已跳过最终评估 (--no-eval)")
    elif stats["interrupted"]:
        logger.info("[3/3] 因安全中断退出，跳过最终评估（--resume 续训完成后再评估）")
    else:
        if os.path.isfile(checkpointer.best_path):
            logger.info("[3/3] 加载最优权重并做最终评估: %s", checkpointer.best_path)
            best_payload = Checkpointer.load(checkpointer.best_path, model, map_location=device) or {}
            best_extra = best_payload.get("extra") or {}
            # 归一化统计量也要跟着最优权重走，否则"训练末期的分布"与"最优权重"不匹配
            if normalizer is not None and best_extra.get("obs_normalizer_state") is not None:
                normalizer.load_state_dict(best_extra["obs_normalizer_state"])
        else:
            logger.warning("[3/3] 未找到最优权重，使用当前权重评估")
        render_env = make_env(ctx.env_name, render=True) if render else None
        eval_result = evaluate_policy(
            model, env_name=ctx.env_name, device=device, episodes=ctx.n_eval_episodes,
            seed=int(eval_cfg.get("seed", 12345)), gamma=float(train_cfg.get("gamma", 0.99)),
            deterministic=eval_cfg["deterministic"], normalizer=normalizer, env=render_env,
        )
        report_closed_loop(
            ctx, eval_result, tag="eval/final_mean_return",
            label=f"最终评估（最优权重，{ctx.n_eval_episodes} 回合）",
            step=stats["steps_total"],
            extra_tags={
                "eval/final_std_return": eval_result["std_return"],
                "eval/final_min_return": eval_result["min_return"],
                "eval/final_max_return": eval_result["max_return"],
            },
        )
        if render_env is not None:
            render_env.close()

    # -------- 5. 总结（含思考题相关的诊断回显）--------
    solved = stats["early_stopped"] or (
        eval_result is not None and eval_result["mean_return"] >= target_reward
    )
    st = stats.get("last_stats") or {}
    extra_lines = [
        f"已完成 rollout : {stats['last_epoch']}（累计与环境交互 {stats['steps_total']} 步）",
        f"训练滑动平均   : {stats['rolling_return']:.2f}（最近 {min(100, max(0, int(train_cfg.get('target_window', 100))))}"
        f"/{int(train_cfg.get('target_window', 100))} 个训练回合）",
        f"目标回报       : {target_reward:.1f} | {'已达成 ✓' if solved else '未达成（可增加 total_timesteps）'}",
    ]
    if st:
        extra_lines.append(
            f"末次更新诊断   : 策略损失 {st.get('policy_loss', 0.0):.4f} | 价值损失 {st.get('value_loss', 0.0):.2f}"
            f" | 熵 {st.get('entropy', 0.0):.4f} | σ {st.get('std_mean', 0.0):.3f}"
            f" | approx_kl {st.get('approx_kl', 0.0):.5f} | clip 占比 {st.get('clip_fraction', 0.0):.3f}"
            f" | 梯度范数 {st.get('grad_norm', 0.0):.2f} | 价值解释度 {st.get('explained_variance', 0.0):.3f}"
        )
        extra_lines.append(
            f"动作截断诊断   : 采样越界比例 {st.get('action_clip_fraction', 0.0):.3f}"
            f"（a_raw 超出 [%.1f, %.1f] 的步占比；越界步的 logπ 仍按 a_raw 计算）"
            % (model.action_low, model.action_high)
        )
    if normalizer is not None:
        extra_lines.append(
            "状态归一化     : count=%d | mean=%s | std=%s"
            % (normalizer.count, np.round(normalizer.mean, 3), np.round(normalizer.std, 3))
        )
    extra_lines.append(
        f"曲线导出       : python experiments/export_tb_curves.py --logdir {ctx.run_dir}"
    )
    finish_run(
        ctx, checkpointer, stats,
        eval_result=eval_result,
        extra_lines=extra_lines,
        resume_command=(
            f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}"
        ),
        metric_label="评估回报",
    )


if __name__ == "__main__":
    main()
