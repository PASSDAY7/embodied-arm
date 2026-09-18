#!/usr/bin/env python
"""REINFORCE（策略梯度 + Baseline）训练脚本 —— W6 Task 1。

任务
----
在 CartPole-v1 上**从零手写**第一个"不需要专家数据"的强化学习算法：策略不再模仿
专家动作，而是自己与环境交互、按回报的符号与大小调整动作概率。

算法（策略梯度定理的蒙特卡洛实现）
--------------------------------
1. **采样**：用当前策略 ``π_θ`` 跑一条完整回合，缓存 ``(s_t, a_t, logπ_θ(a_t|s_t), r_t)``；
2. **回报**：从回合末尾反向递推折扣回报 ``G_t = r_t + γ·G_{t+1}``，并做 Return 归一化
   ``Â_t = (G_t − μ_G) / (σ_G + ε)``（``algorithm.normalize_returns=true``）；
3. **损失**：``L(θ) = − Σ_t logπ_θ(a_t|s_t) · Â_t``（对步数取均值）；
4. **更新**：``optimizer.step()`` 沿 ``−∇L`` 走一步 ⇒ 等价于沿 ``∇J`` 的**上升**方向。

思考题 1：损失函数前为什么要加负号？
----------------------------------
策略梯度定理给出的是"**最大化**期望回报"的方向（``∇_θ J(θ) = E[∇logπ·A]``），
而 ``torch.optim`` 里的 SGD / Adam **只会最小化**它拿到的目标（``θ ← θ − lr·∇L``）。
把目标写成 ``L = −Σ logπ·A`` 之后，梯度下降方向就是 ``−∇J``，即
"``J`` 的上升方向"：``A_t > 0`` 的动作其 ``logπ`` 被抬高（概率变大），
``A_t < 0`` 的动作其 ``logπ`` 被压低（概率变小）。若不加负号，训练会朝"最大化坏动作
概率"的方向跑 —— 单元测试 ``test_minus_sign_is_required_for_ascent`` 用一个刻意构造的
单步例子把这个符号错误显式量化出来。

思考题 2：如果去掉 Return 归一化会发生什么？
------------------------------------------
``G_t`` 的量纲是"奖励总量"，CartPole 里它可以到 100（γ=0.99 时 ``G_0 ≈ 95``），
于是策略梯度的步长被回报尺度直接放大：前期回合短、``σ_G`` 小，后期回合长、``σ_G`` 大，
同一个 ``lr`` 在训练过程中的**有效步长漂移**可达一个数量级；而 ``Σ_t logπ·A`` 还会把
"回合内所有步的梯度"按回报大小叠加，使不同回合的更新强度失衡。
Return 归一化把批内回报拉到零均值、单位标准差后：

* **尺度稳定**：梯度尺度与"当前策略能拿到多少分"解耦，``lr`` 不需要随训练阶段重调；
* **自带最简基线**（减均值）⇒ 方差下降、无偏性仍成立（单元测试对无偏性有数值验证）；
* **代价是丢掉绝对分值信息**：当"批内所有回合表现都好"时，归一化后仍会把最好的回合
  与最差的回合拉开同样距离，回报的**绝对提升**不再直接反映在梯度幅度上
  （对策略梯度而言这通常是无害的，因为策略更新的方向才是关键，绝对水平由 ``J`` 度量）。
消融开关：``--set algorithm.normalize_returns=false``（只减均值不除标准差），
``--set algorithm.baseline=none algorithm.normalize_returns=false``（纯 ``Σ logπ·G``）。

工程实现
--------
``--config/--set/--resume`` 解析、运行目录 / 日志 / TensorBoard 装配、优雅停机与
检查点全部复用 ``src/engine.py``；算法核心（策略网络、回报递推、优势、损失、回合采样、
独立评估）在 ``src/reinforce.py``，与 W3/W4 的 BC 脚本共享同一套工程骨架。

与 BC 的关键差异
----------------
* **没有 ``data`` 段**：训练数据由策略自己产生（on-policy），"epoch" 换成"回合"；
* **没有 ``val_mse``**：模型选择用**独立评估回合的平均回报**（``eval.every_episodes``
  个回合评估一次，贪心策略 + 固定种子）；
* **提前停止**：最近 ``training.target_window``（默认 100）个回合的平均回报
  ≥ ``training.target_reward``（默认 475）即认定任务解决并退出。

用法示例::

    python experiments/train_reinforce.py                                  # 默认配置（seed 42 / cuda）
    python experiments/train_reinforce.py --device cpu                     # 无 GPU 时
    python experiments/train_reinforce.py --set algorithm.baseline=value    # 进阶基线（小 Critic）
    python experiments/train_reinforce.py --set algorithm.normalize_returns=false   # 思考题 2 的消融
    python experiments/train_reinforce.py --set algorithm.baseline=none algorithm.normalize_returns=false
    python experiments/train_reinforce.py --set training.max_episodes=200   # 快速验证
    python experiments/train_reinforce.py --resume                          # 断点续训

注意 ``--set`` 的写法::

    * 一次 ``--set`` 后可以跟多个 ``KEY=VALUE``（重复写 ``--set`` 只有最后一次生效）；
    * 布尔值写成 ``false`` / ``true`` / ``False`` / ``True`` 都能被正确识别
      （``src.engine.cast_value`` 走 ``ast.literal_eval``，本脚本再用 :func:`as_bool`
      兜住小写字符串的坑）。
"""

from __future__ import annotations

import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch.optim as optim

# 允许脚本以 `python experiments/train_reinforce.py` 直接运行
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
from src.reinforce import (  # noqa: E402
    build_policy,
    build_value_network,
    count_parameters,
    env_max_steps,
    evaluate_policy,
    make_env,
    run_episode,
    update_policy,
)
from src.utils import Checkpointer  # noqa: E402

CONFIG_PATH = os.path.join("configs", "reinforce_config.yaml")
TITLE = "REINFORCE（策略梯度 + Baseline）实验"
SCRIPT_PATH = "experiments/train_reinforce.py"


# =============================================================================
# 工具
# =============================================================================
def rolling_mean(values: List[float], window: int) -> float:
    """最近 ``window`` 个回合的平均回报（数据不足时按现有数据平均）。"""
    if not values:
        return 0.0
    return float(np.mean(values[-int(window):] if int(window) > 0 else values))


def as_bool(value: Any, default: bool = False) -> bool:
    """把配置里的布尔值规范化为 ``bool``。

    ``src.engine.cast_value`` 用 ``ast.literal_eval`` 解析 ``--set``，因此 PyYAML 常用的
    小写 ``false`` 会退化成**字符串** ``"false"``（而字符串是 truthy 的）。这里显式识别
    ``true/false/1/0/yes/no/on/off``，让 ``--set algorithm.normalize_returns=false``
    （小写）与 ``=False``（Python 字面量）行为一致。
    """
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


def build_optimizer(policy, critic, train_cfg: Dict[str, Any]) -> optim.Adam:
    """构建 Adam：``baseline=value`` 时把 Critic 参数一起交给**同一个**优化器。

    策略项与 Critic 项各自只依赖互不相交的参数子集，Adam 逐参数更新，
    因此"一个联合优化器"与"两个独立优化器"完全等价（见 ``src/reinforce.py`` 模块说明），
    好处是检查点里只需要保存一份优化器状态。
    """
    params = list(policy.parameters())
    if critic is not None:
        params += list(critic.parameters())
    return optim.Adam(
        params,
        lr=float(train_cfg.get("lr", 2e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


# =============================================================================
# 回合制训练循环（on-policy：采样 → 更新 → 采样 ...）
# =============================================================================
def train_reinforce(
    ctx,
    policy,
    optimizer: optim.Optimizer,
    critic,
    checkpointer: Checkpointer,
    start_episode: int = 1,
    steps_done: int = 0,
) -> Dict[str, Any]:
    """REINFORCE 主循环，返回本轮训练统计（供 :func:`src.engine.finish_run` 使用）。

    每个回合：采样一条轨迹 → 累积 ``update_batch_episodes`` 条 → 做一次策略改进；
    每 ``eval.every_episodes`` 个回合：独立评估（贪心策略）→ 写 ``last`` / ``best`` 检查点。

    Args:
        ctx: :func:`src.engine.prepare_run` 装配的运行上下文。
        policy: 策略网络 ``π_θ``。
        optimizer: Adam（``baseline=value`` 时同时持有 Critic 参数）。
        critic: 价值网络或 ``None``。
        checkpointer: 检查点管理器（``mode=max`` 时按评估回报刷新最优）。
        start_episode: 起始回合（断点续训为"已完成回合 + 1"）。
        steps_done: 续训前已累计的交互步数（仅用于日志/TensorBoard 的累计量）。

    Returns:
        ``{"best_metric", "last_epoch", "interrupted", "early_stopped", "train_time",
           "last_episode_return", "rolling_return", "steps_total", "history", "last_stats"}``。
    """
    logger, tb, device = ctx.logger, ctx.tb, ctx.device
    train_cfg, eval_cfg = ctx.train_cfg, ctx.eval_cfg
    algo_cfg = ctx.config.get("algorithm", {}) or {}

    gamma = float(train_cfg.get("gamma", 0.99))
    max_episodes = int(train_cfg.get("max_episodes", 600))
    log_interval = max(1, int(train_cfg.get("log_interval", 20)))
    batch_episodes = max(1, int(train_cfg.get("update_batch_episodes", 1)))
    target_reward = float(train_cfg.get("target_reward", 475.0))
    target_window = max(1, int(train_cfg.get("target_window", 100)))
    eval_interval = max(1, int(eval_cfg.get("every_episodes", log_interval)))
    eval_episodes = ctx.n_eval_episodes
    eval_deterministic = as_bool(eval_cfg.get("deterministic", True), True)
    eval_seed = int(eval_cfg.get("seed", 12345))

    # 算法超参（全部来自配置，便于消融）
    baseline = str(algo_cfg.get("baseline", "batch_mean"))
    normalize_returns = as_bool(algo_cfg.get("normalize_returns", True), True)
    normalize_advantages = as_bool(algo_cfg.get("normalize_advantages", False), False)
    entropy_coef = float(algo_cfg.get("entropy_coef", 0.0))
    critic_coef = float(algo_cfg.get("critic_coef", 0.5))
    reduction = str(algo_cfg.get("reduction", "mean"))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))

    env = make_env(ctx.env_name)
    history: List[float] = []            # 逐回合的未折扣回报（= 存活步数）
    pending: List[Dict[str, Any]] = []   # 尚未用于更新的轨迹
    steps_total = int(steps_done)        # 含续训前的历史累计步数
    last_stats: Dict[str, Any] = {}
    interrupted = False
    early_stopped = False
    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        if stop_requested["flag"]:
            logger.warning("再次收到信号 %s，强制退出（当前回合进度将丢失）", signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        stop_requested["flag"] = True
        logger.warning("收到信号 %s：将在当前回合结束后保存检查点并安全退出", signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handle_signal)
        except (ValueError, OSError, AttributeError):  # pragma: no cover - 非主线程等场景
            pass

    def _checkpoint(ep: int, metric: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        """写 last（必要时写 best）；``critic_state_dict`` 一并放进 extras 以支持续训。"""
        extras = dict(extra)
        extras.update({"run_dir": ctx.run_dir, "seed": ctx.seed})
        if critic is not None:
            extras["critic_state_dict"] = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
        return checkpointer.step(
            policy, optimizer, None,
            epoch=ep, metric=float(metric), config=ctx.config, extras=extras,
        )

    start_time = time.time()
    episode = start_episode - 1
    rolling = 0.0
    logger.info(
        "[2/3] 开始训练：episode %d → %d | 基线: %s | Return 归一化: %s | 批量回合数: %d",
        start_episode, max_episodes, baseline, "on" if normalize_returns else "off", batch_episodes,
    )

    try:
        for episode in range(start_episode, max_episodes + 1):
            # ---- 1) 采样一条完整轨迹（on-policy 数据）----
            #     每个回合用 seed+episode-1 播种 ⇒ 回合起始状态可复现；
            #     动作采样依赖全局 torch RNG（续训时由检查点里的 RNG 状态接管）。
            ep = run_episode(
                env, policy, device=device, gamma=gamma, normalize_returns=False,
                seed=ctx.seed + episode - 1, deterministic=False,
            )
            history.append(float(ep["episode_return"]))
            steps_total += int(ep["length"])
            pending.append(ep)
            rolling = rolling_mean(history, target_window)

            # ---- 2) 攒够 batch_episodes 条后做一次策略改进 ----
            if len(pending) >= batch_episodes:
                last_stats = update_policy(
                    policy, pending,
                    optimizer=optimizer, critic=critic, device=device, gamma=gamma,
                    normalize_returns=normalize_returns, baseline=baseline,
                    normalize_advantages=normalize_advantages,
                    critic_coef=critic_coef, entropy_coef=entropy_coef,
                    max_grad_norm=max_grad_norm, reduction=reduction,
                )
                pending = []
                tb.add_scalar("train/policy_loss", last_stats["policy_loss"], episode)
                tb.add_scalar("train/advantage_abs_mean", last_stats["advantage_abs_mean"], episode)
                tb.add_scalar("train/advantage_raw_std", last_stats["advantage_raw_std"], episode)
                tb.add_scalar("train/entropy", last_stats["entropy"], episode)
                tb.add_scalar("train/grad_norm", last_stats["grad_norm"], episode)
                tb.add_scalar("train/return_raw_mean", last_stats["return_raw_mean"], episode)
                tb.add_scalar("train/log_prob_drift", last_stats["log_prob_drift"], episode)
                if baseline == "value":
                    tb.add_scalar("train/critic_loss", last_stats["critic_loss"], episode)

            # ---- 3) 监控曲线 ----
            tb.add_scalar("train/episode_return", history[-1], episode)
            tb.add_scalar("train/episode_length", float(ep["length"]), episode)
            tb.add_scalar("train/rolling_return", rolling, episode)
            tb.add_scalar("train/steps_total", float(steps_total), episode)
            tb.add_scalar("train/return_raw_std_episode", float(ep["raw_std"]), episode)

            if episode % log_interval == 0 or episode == start_episode:
                logger.info(
                    "Episode [%d/%d] | 回报: %6.1f | 滑动均值(%d): %6.2f | 策略损失: %8.4f | "
                    "熵: %.3f | 优势均值: %7.4f | 梯度范数: %.3f | 累计步数: %d",
                    episode, max_episodes, history[-1], target_window, rolling,
                    last_stats.get("policy_loss", 0.0), last_stats.get("entropy", 0.0),
                    last_stats.get("advantage_raw_mean", 0.0), last_stats.get("grad_norm", 0.0),
                    steps_total,
                )

            # ---- 4) 独立评估 + 存盘（模型选择用的是"评估回报"，不是训练回报）----
            if episode % eval_interval == 0 or episode == max_episodes:
                eval_result = evaluate_policy(
                    policy, env_name=ctx.env_name, device=device, episodes=eval_episodes,
                    seed=eval_seed, gamma=gamma, deterministic=eval_deterministic,
                )
                report_closed_loop(
                    ctx, eval_result, label=f"独立评估（贪心，{eval_episodes} 回合）", step=episode,
                    extra_tags={
                        "eval/std_return": eval_result["std_return"],
                        "eval/min_return": eval_result["min_return"],
                        "eval/max_return": eval_result["max_return"],
                        "eval/rolling_train_return": rolling,
                    },
                )
                save_info = _checkpoint(episode, eval_result["mean_return"], {
                    "episode_return": history[-1], "rolling_return": rolling,
                    "steps_total": steps_total,
                })
                logger.info(
                    "检查点     | %s%s", save_info["last"],
                    "  *best" if save_info["is_best"] else f"  (历史最优 {save_info['best_metric']:.2f})",
                )
                if eval_result["mean_return"] >= target_reward and len(history) >= target_window:
                    early_stopped = True
                    logger.info(
                        "★ 评估平均回报 %.2f ≥ 目标 %.1f：任务已解决，提前结束训练（episode %d/%d）",
                        eval_result["mean_return"], target_reward, episode, max_episodes,
                    )
                    break

            # ---- 5) 提前停止（按训练回合的滑动均值）----
            if rolling >= target_reward and len(history) >= target_window:
                early_stopped = True
                logger.info(
                    "★ 最近 %d 个训练回合平均回报 %.2f ≥ 目标 %.1f：提前结束训练（episode %d/%d）",
                    target_window, rolling, target_reward, episode, max_episodes,
                )
                break

            # ---- 6) 优雅停机（SIGINT / SIGTERM）----
            if stop_requested["flag"]:
                interrupted = True
                _checkpoint(episode, rolling, {
                    "episode_return": history[-1], "rolling_return": rolling,
                    "steps_total": steps_total, "interrupted": True,
                })
                logger.warning(
                    "已保存检查点并于 episode %d 安全退出；继续训练可执行：\n"
                    "  python %s --config %s --resume %s",
                    episode, SCRIPT_PATH, ctx.args.config, checkpointer.last_path,
                )
                break
    except Exception:
        # 异常也要最大限度保留成果：先写应急检查点再向上抛
        _checkpoint(
            episode,
            rolling_mean(history, target_window) if history else 0.0,
            {"episode_return": history[-1] if history else 0.0,
             "steps_total": steps_total, "emergency": True},
        )
        logger.error("训练异常，已写应急检查点: %s", checkpointer.last_path)
        raise
    finally:
        env.close()

    train_time = time.time() - start_time
    stats: Dict[str, Any] = {
        "best_metric": float(checkpointer.best_metric) if checkpointer.best_metric is not None else 0.0,
        "last_epoch": int(episode),
        "interrupted": bool(interrupted),
        "early_stopped": bool(early_stopped),
        "train_time": float(train_time),
        "last_episode_return": float(history[-1]) if history else 0.0,
        "rolling_return": rolling_mean(history, target_window),
        "steps_total": int(steps_total),
        "history": history,
        "last_stats": last_stats,
    }
    logger.info(
        "训练结束，用时 %.1fs | 已完成回合: %d | 累计交互步数: %d | 最近 %d 回合平均回报: %.2f",
        train_time, stats["last_epoch"], steps_total,
        min(target_window, len(history)), stats["rolling_return"],
    )
    return stats


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("REINFORCE（策略梯度 + Baseline）训练脚本（配置驱动）", CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    train_cfg = config.get("training", {}) or {}
    algo_cfg = config.setdefault("algorithm", {})
    eval_cfg = config.setdefault("eval", {})
    # 规范化布尔字段：`--set algorithm.normalize_returns=false`（小写）也能被正确识别
    algo_cfg["normalize_returns"] = as_bool(algo_cfg.get("normalize_returns", True), True)
    algo_cfg["normalize_advantages"] = as_bool(algo_cfg.get("normalize_advantages", False), False)
    eval_cfg["deterministic"] = as_bool(eval_cfg.get("deterministic", True), True)
    eval_cfg["render"] = as_bool(eval_cfg.get("render", False), False)
    baseline = str(algo_cfg.get("baseline", "batch_mean"))
    normalize_returns = algo_cfg["normalize_returns"]
    target_reward = float(train_cfg.get("target_reward", 475.0))

    # -------- 基础设置：设备 / 种子 / 运行目录 / 日志 / TensorBoard / 续训解析 --------
    ctx = prepare_run(
        args, config, title=TITLE, logger_name="reinforce",
        extra_info=[
            "算法       : REINFORCE（蒙特卡洛策略梯度；训练数据由策略自己采样，无专家数据）",
            f"基线       : {baseline} | Return 归一化: {'on' if normalize_returns else 'off'}"
            f" | 熵奖励系数: {algo_cfg.get('entropy_coef', 0.0)}",
            f"超参数     : γ={train_cfg.get('gamma', 0.99)} | lr={train_cfg.get('lr', 2e-3)}"
            f" | max_episodes={train_cfg.get('max_episodes', 600)} | 目标回报={target_reward}",
        ],
    )
    logger, device, seed = ctx.logger, ctx.device, ctx.seed

    # -------- 1. 环境 / 网络 / 优化器 / 检查点 --------
    env = make_env(ctx.env_name)
    logger.info(
        "[1/3] 环境: %s | 状态空间 %s | 动作空间 %s | 单回合上限 %d 步",
        ctx.env_name, env.observation_space, env.action_space, env_max_steps(env),
    )
    env.close()

    policy = build_policy(config).to(device)
    critic = build_value_network(config).to(device) if baseline == "value" else None
    logger.info(
        "策略网络   : MLP(%d → %d × %d → %d) | 可训练参数: %d",
        policy.state_dim, policy.num_layers, policy.hidden_dim, policy.action_dim,
        count_parameters(policy),
    )
    if critic is not None:
        logger.info(
            "价值网络   : MLP(%d → %d × %d → 1) | 可训练参数: %d",
            critic.state_dim, critic.num_layers, critic.hidden_dim, count_parameters(critic),
        )

    optimizer = build_optimizer(policy, critic, train_cfg)
    logger.info(
        "优化器     : Adam(lr=%.4g, weight_decay=%.4g) | 参数张量 %d 个（策略%s）",
        float(train_cfg.get("lr", 2e-3)), float(train_cfg.get("weight_decay", 0.0)),
        len(optimizer.param_groups[0]["params"]), " + Critic 共用一个" if critic is not None else "",
    )
    checkpointer = ctx.build_checkpointer()

    # -------- 2. 断点续训：恢复策略 / 优化器 / 回合号 / 最优指标 / 随机源 --------
    start_episode = restore_from_checkpoint(ctx, policy, optimizer, None, checkpointer)
    prev_steps = 0
    if ctx.resume_payload is not None:
        resume_extra = ctx.resume_payload.get("extra") or {}
        prev_steps = int(resume_extra.get("steps_total") or 0)      # 累计交互步数跨续训保持
        if critic is not None:
            critic_state = resume_extra.get("critic_state_dict")
            if critic_state is not None:
                critic.load_state_dict(critic_state)
                logger.info("已恢复价值网络权重（检查点 extras.critic_state_dict）")

    # -------- 3. 回合制训练 --------
    train_stats = train_reinforce(
        ctx, policy, optimizer, critic, checkpointer,
        start_episode=start_episode, steps_done=prev_steps,
    )

    # -------- 4. 用"最优权重"做最终评估 --------
    eval_result: Optional[Dict[str, Any]] = None
    render = bool(eval_cfg.get("render", False))
    if ctx.no_eval:
        logger.info("[3/3] 已跳过最终评估 (--no-eval)")
    elif train_stats["interrupted"]:
        logger.info("[3/3] 因安全中断退出，跳过最终评估（--resume 续训完成后再评估）")
    else:
        if os.path.isfile(checkpointer.best_path):
            logger.info("[3/3] 加载最优权重并做最终评估: %s", checkpointer.best_path)
            Checkpointer.load(checkpointer.best_path, policy, map_location=device)
        else:
            logger.warning("[3/3] 未找到最优权重，使用当前权重评估")
        render_env = make_env(ctx.env_name, render=True) if render else None
        eval_result = evaluate_policy(
            policy,
            env_name=ctx.env_name,
            device=device,
            episodes=ctx.n_eval_episodes,
            seed=int(eval_cfg.get("seed", 12345)),
            gamma=float(train_cfg.get("gamma", 0.99)),
            deterministic=bool(eval_cfg.get("deterministic", True)),
            env=render_env,
        )
        report_closed_loop(
            ctx, eval_result, tag="eval/final_mean_return",
            label=f"最终评估（最优权重，{ctx.n_eval_episodes} 回合）",
            step=train_stats["last_epoch"],
            extra_tags={
                "eval/final_std_return": eval_result["std_return"],
                "eval/final_min_return": eval_result["min_return"],
                "eval/final_max_return": eval_result["max_return"],
            },
        )
        if render_env is not None:
            render_env.close()

    # -------- 5. 总结（含两个思考题相关的算法配置回显）--------
    solved = train_stats["early_stopped"] or (
        eval_result is not None and eval_result["mean_return"] >= target_reward
    )
    extra_lines = [
        f"已完成回合     : {train_stats['last_epoch']}（累计与环境交互 {train_stats['steps_total']} 步）",
        f"训练滑动平均   : {train_stats['rolling_return']:.2f}"
        f"（最近 {int(train_cfg.get('target_window', 100))} 个训练回合）",
        f"目标回报       : {target_reward:.1f} | {'已达成 ✓' if solved else '未达成（可增加 max_episodes）'}",
        f"算法配置       : baseline={baseline} | normalize_returns={normalize_returns}"
        f" | entropy_coef={algo_cfg.get('entropy_coef', 0.0)}",
    ]
    if train_stats.get("last_stats"):
        st = train_stats["last_stats"]
        extra_lines.append(
            f"末次更新诊断   : 回合回报均值 {st['return_raw_mean']:.2f}"
            f" | |优势| 均值 {st['advantage_abs_mean']:.4f}"
            f" | 熵 {st['entropy']:.4f} | 梯度范数 {st['grad_norm']:.4f}"
        )
    finish_run(
        ctx, checkpointer, train_stats,
        eval_result=eval_result,
        extra_lines=extra_lines,
        resume_command=(
            f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}"
        ),
        metric_label="评估回报",
    )


if __name__ == "__main__":
    main()


