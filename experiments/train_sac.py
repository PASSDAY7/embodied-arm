#!/usr/bin/env python
"""SAC（Soft Actor-Critic）训练脚本 —— W9 Task 1。

任务
----
在 **Pendulum-v1**（连续动作空间 ``[-2, 2]``，单回合 200 步）上训练手写 SAC：
``total_timesteps=40000``、``buffer_size=100000``、``batch_size=256``、
``lr=(3e-4, 1e-3, 3e-4)``、``tau=0.005``、``gamma=0.99``、
``target_entropy="auto"``（= −dim(A) = −1.0）、``warmup_steps=1000``。

算法流程（off-policy：**每个环境步都更新一次**）::

    热身期（前 warmup_steps=1000 步）：动作均匀随机，只填 ReplayBuffer，不更新
    ────────────────────────────────────────────────────────────────────────
    之后每个环境步：
      ① 用 π_φ(·|s) 采样 a_norm = tanh(μ+σε) → a_env = a_norm·scale + bias，env.step
      ② buffer.add(s, a_norm, r, s', terminated, truncated)   # 存**归一化动作**
      ③ 抽 1 个 mini-batch：Critic 步（Double-Q）→ Actor 步（冻结 Critic）→ α 步 → Polyak
      ④ 每 eval.every_steps 步：独立评估（20 回合、固定种子、均值动作）→ 存盘 / 判优

监控指标（TensorBoard，横轴统一为**累计环境交互步数**）
----------------------------------------------------
``train/q1_loss`` / ``train/q2_loss``（双 Q 的 TD 误差）、``train/actor_loss``（策略损失）、
``train/alpha_loss``、``train/alpha`` 与 ``train/log_alpha``（温度的**自适应轨迹**）、
``train/entropy_mean``（样本估计的微分熵）、``train/entropy_target_gap``、
``train/target_q_mean``（软 Bellman 目标尺度）、``train/critic_grad_norm`` /
``train/actor_grad_norm`` / ``train/alpha_grad_norm``、``train/critic_frozen_ok``
（Actor 步是否真的**没有**给 Critic 留下梯度 —— 思考题的在线断言）、
``buffer/*``（回放池填充率 / 奖励尺度 / 截断比例 / 动作覆盖率）、
``eval/*``（闭环评估回报）。

样本效率预检（作业要求"记录它在多少步突破 −200 分"）
----------------------------------------------
评估每 ``eval.every_steps=2000`` 步做一次，脚本记录**第一次** ``mean_return ≥ −200``
的步数：写进 TensorBoard ``eval/first_solved_step`` 并在最终汇总里打印，
可与 W8 PPO（194 560 步、独立评估 −199.91）直接对比量级。

思考题：为什么更新 Actor 时梯度要穿过 Critic，但不能更新 Critic 的参数？
-------------------------------------------------------------------
完整推导见 ``src/sac.py`` 模块 docstring；工程落点在本脚本依赖的三处：

1. ``SACAgent.actor_loss`` 用 ``frozen_parameters(critic)`` 冻结 **参数**（``requires_grad=False``）
   而**不**用 ``torch.no_grad()`` —— 图连通（``∂Q/∂a`` 能回传到 ``μ/σ``）、参数不积梯度；
2. ``build_sac_optimizers`` 建**三套独立 Adam**（critic / actor / alpha），
   Actor 步只 ``actor_optimizer.step()``，Critic 绝不会被"抬高 Q 值"的方向污染；
3. ``target_critic`` 参数永久 ``requires_grad_(False)`` + ``torch.no_grad()`` 计算 Bellman 目标，
   且 α 的损失里 ``log_prob.detach()``。

用法示例::

    python experiments/train_sac.py                                      # 默认配置（seed 42 / cuda）
    python experiments/train_sac.py --device cpu                         # 无 GPU 时
    python experiments/train_sac.py --set training.total_timesteps=5000  # 快速冒烟
    python experiments/train_sac.py --set training.target_entropy=null   # 消融：固定 α
    python experiments/train_sac.py --set training.bootstrap_on_truncation=false  # 消融：200 步当终止
    python experiments/train_sac.py --resume                             # 断点续训（自动找最新）
    python experiments/export_tb_curves.py --logdir logs/sac_pendulum_* --prefix w9_sac  # 导出曲线
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

# 允许脚本以 `python experiments/train_sac.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import (  # noqa: E402
    apply_overrides,
    finish_run,
    load_experiment_config,
    parse_train_args,
    prepare_run,
    report_closed_loop,
)
from src.sac import (  # noqa: E402
    ReplayBuffer,
    RunningMeanStd,
    build_sac_agent,
    build_sac_optimizers,
    count_parameters,
    env_max_steps,
    evaluate_policy,
    make_env,
    random_norm_action,
)
from src.utils import Checkpointer, restore_rng_state  # noqa: E402

CONFIG_PATH = os.path.join("configs", "sac_config.yaml")
TITLE = "SAC（Soft Actor-Critic，off-policy 最大熵 RL）实验"
SCRIPT_PATH = "experiments/train_sac.py"

# =============================================================================
# 工具（与 W8 train_ppo.py 保持同一套语义）
# =============================================================================
def rolling_mean(values: List[float], window: int) -> float:
    """最近 ``window`` 个回合的平均回报（数据不足时按现有数据平均）。"""
    if not values:
        return 0.0
    return float(np.mean(values[-int(window):] if int(window) > 0 else values))


def as_bool(value: Any, default: bool = False) -> bool:
    """把配置里的布尔值规范化为 ``bool``（PyYAML 小写 ``false`` 可能退化成字符串）。"""
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


def check_action_bounds(config: Dict[str, Any], env) -> Dict[str, Any]:
    """用**环境真实动作范围**校验 / 覆盖 ``model.action_low/high``（防止静默的错误边界）。"""
    model_cfg = config.setdefault("model", {})
    low = float(np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)[0])
    high = float(np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)[0])
    for key, truth in (("action_low", low), ("action_high", high)):
        cfg_value = model_cfg.get(key)
        if cfg_value is None or abs(float(cfg_value) - truth) > 1e-6:
            model_cfg[key] = truth
    return model_cfg


class RunningAvg:
    """区间内诊断量的均值累加器（在线聚合 → 每个日志点输出一个标量）。"""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, logs: Dict[str, Any]) -> None:
        for key, value in logs.items():
            if isinstance(value, torch.Tensor) or not isinstance(value, (int, float, np.floating)):
                continue                       # 跳过 log_prob 这类张量 / 非数值字段
            value = float(value)
            if math.isnan(value):
                continue
            self.sums[key] = self.sums.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: v / max(1, self.counts[k]) for k, v in self.sums.items()}

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()


# =============================================================================
# 训练主循环（热身 → 环境步 / 梯度步交替 → 定期评估）
# =============================================================================
def train_sac(
    ctx,
    agent,
    buffer,
    optimizers: Dict[str, Optional[torch.optim.Optimizer]],
    normalizer: Optional[RunningMeanStd],
    checkpointer: Checkpointer,
    steps_done: int = 0,
    updates_done: int = 0,
    episodes_done: int = 0,
    history: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """SAC 主循环，返回训练统计（供 :func:`src.engine.finish_run` 使用）。

    Args:
        ctx: :func:`src.engine.prepare_run` 装配的运行上下文。
        agent: :class:`src.sac.SACAgent`（续训时已恢复权重 / 温度 / 目标网络）。
        buffer: :class:`src.sac.ReplayBuffer`（续训时已恢复内容）。
        optimizers: ``{"critic", "actor", "alpha"}`` 三套独立 Adam。
        normalizer: 可选的在线状态归一化器（``model.normalize_obs=true`` 时非空）。
        checkpointer: ``mode=max`` 时按**独立评估回报**刷新最优权重。
        steps_done / updates_done / episodes_done: 续训前的累计量。
        history: 续训前的训练回合回报历史（用于滑动均值与提前停止判据）。

    Returns:
        ``{"best_metric", "last_epoch", "interrupted", "early_stopped", "train_time",
           "steps_total", "updates_total", "episodes_total", "rolling_return",
           "first_solved_step", "history", "eval_history", "alpha_history"}``。
    """
    logger, tb, device = ctx.logger, ctx.tb, ctx.device
    train_cfg = ctx.train_cfg
    eval_cfg = ctx.eval_cfg

    total_timesteps = int(train_cfg.get("total_timesteps", 40000))
    batch_size = int(train_cfg.get("batch_size", 256))
    warmup_steps = int(train_cfg.get("warmup_steps", 1000))
    train_freq = max(1, int(train_cfg.get("train_freq", 1)))
    gradient_steps = max(1, int(train_cfg.get("gradient_steps", 1)))
    gamma = float(train_cfg.get("gamma", 0.99))
    gradient_clip = float(train_cfg.get("max_grad_norm", 0.0) or 0.0)
    log_interval = max(1, int(train_cfg.get("log_interval", 1000)))
    target_reward = float(train_cfg.get("target_reward", -200.0))
    target_window = max(1, int(train_cfg.get("target_window", 100)))
    bootstrap_on_truncation = as_bool(train_cfg.get("bootstrap_on_truncation", True), True)

    eval_interval = max(1, int(eval_cfg.get("every_steps", 2000)))
    eval_episodes = ctx.n_eval_episodes
    eval_deterministic = as_bool(eval_cfg.get("deterministic", True), True)
    eval_seed = int(eval_cfg.get("seed", 12345))
    ckpt_interval = int((train_cfg.get("checkpoint", {}) or {}).get("every_steps", 5000) or 0)

    env = make_env(ctx.env_name)
    obs, _info = env.reset(seed=ctx.seed + int(episodes_done))
    try:
        env.action_space.seed(ctx.seed + int(episodes_done))
    except AttributeError:  # pragma: no cover - 极少数环境不支持
        pass
    max_steps = env_max_steps(env)
    episode_return, episode_len = 0.0, 0

    history = list(history or [])
    eval_history: List[Dict[str, float]] = []
    alpha_history: List[Dict[str, float]] = []
    steps_total = int(steps_done)
    updates_total = int(updates_done)
    episodes_total = int(episodes_done)
    first_solved_step: Optional[int] = None
    last_eval: Dict[str, float] = {}
    running = RunningAvg()
    interrupted = False
    early_stopped = False
    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        if stop_requested["flag"]:
            logger.warning("再次收到信号 %s，强制退出（当前进度可能丢失）", signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        stop_requested["flag"] = True
        logger.warning("收到信号 %s：将在当前环境步结束后保存检查点并安全退出", signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handle_signal)
        except (ValueError, OSError, AttributeError):  # pragma: no cover - 非主线程等场景
            pass

    def _checkpoint(step: int, metric: float, extra: Dict[str, Any]) -> Dict[str, Any]:
        """写 last（更优时写 best）：回放池 / 归一化统计量 / 三套优化器一起进 extras。"""
        extras = dict(extra)
        extras.update({
            "run_dir": ctx.run_dir, "seed": ctx.seed,
            "steps_total": int(steps_total), "updates_total": int(updates_total),
            "episodes_total": int(episodes_total),
            "alpha": float(agent.alpha.detach()), "log_alpha": float(agent.log_alpha.detach()),
            "first_solved_step": -1 if first_solved_step is None else int(first_solved_step),
            "episode_history": history[-max(target_window, 200):],
            "replay_buffer_state": buffer.state_dict(),
            "optimizer_states": {
                name: (opt.state_dict() if opt is not None else None)
                for name, opt in optimizers.items()
            },
        })
        if normalizer is not None:
            extras["obs_normalizer_state"] = normalizer.state_dict()
        return checkpointer.step(agent, None, None, epoch=step, metric=float(metric),
                                 config=ctx.config, extras=extras)

    start_time = time.time()
    rolling = rolling_mean(history, target_window)
    logger.info(
        "[2/3] 开始训练：环境步 %d → %d | 热身 %d 步 | 每 %d 步更新 %d 次 mini-batch(%d) | "
        "回放池容量 %d（当前 %d 条）",
        steps_total + 1, total_timesteps, warmup_steps, train_freq, gradient_steps, batch_size,
        buffer.capacity, buffer.size,
    )
    logger.info(
        "      超参数: γ=%.3f | τ=%.3f | α_init=%.3f | 目标熵=%s | 双Q=%d | "
        "lr=(actor %.1e, critic %.1e, alpha %.1e) | truncated 自举=%s",
        gamma, agent.tau, float(agent.alpha.detach()), agent.target_entropy,
        agent.critic.n_critics, optimizers["actor"].param_groups[0]["lr"],
        optimizers["critic"].param_groups[0]["lr"],
        (optimizers["alpha"].param_groups[0]["lr"] if optimizers["alpha"] is not None else 0.0),
        "是" if bootstrap_on_truncation else "否（把 200 步当终止，有偏）",
    )

    # ---------------------------------------------------------------- #
    # 主循环
    # ---------------------------------------------------------------- #
    try:
        remaining = max(0, total_timesteps - steps_total)
        if remaining == 0:
            logger.warning("总预算 %d 步已用完（已交互 %d 步），跳过训练", total_timesteps, steps_total)
        for _ in range(remaining):
            steps_total += 1
            warmup = steps_total <= warmup_steps

            # ---- ① 状态归一化（buffer 里存的就是 Critic 看到的东西）----
            if normalizer is not None:
                normalizer.update(obs)
                state_np = normalizer.normalize_np(obs)
            else:
                state_np = np.asarray(obs, dtype=np.float32)

            # ---- ② 选动作：热身期均匀随机，之后由 π_φ(·|s) 采样 ----
            if warmup:
                action_norm = random_norm_action(env, agent.actor)
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    out = agent.act(state_t)
                action_norm = out["action_norm"][0].cpu().numpy().astype(np.float32)
            action_env = agent.actor.to_env_action(torch.as_tensor(action_norm)).cpu().numpy()

            # ---- ③ 与环境交互：a_env 只进 env.step，buffer 存归一化动作 ----
            next_obs, reward, terminated, truncated, _info = env.step(action_env)
            next_state_np = (normalizer.normalize_np(next_obs) if normalizer is not None
                             else np.asarray(next_obs, dtype=np.float32))
            buffer.add(state_np, action_norm, float(reward), next_state_np, terminated, truncated)
            episode_return += float(reward)
            episode_len += 1

            # ---- ④ 回合结束：记录（原始）回报并重置环境 ----
            if terminated or truncated or episode_len >= max_steps:
                history.append(episode_return)
                episodes_total += 1
                rolling = rolling_mean(history, target_window)
                tb.add_scalar("train/episode_return", episode_return, steps_total)
                tb.add_scalar("train/episode_length", float(episode_len), steps_total)
                tb.add_scalar("train/rolling_return", rolling, steps_total)
                obs, _info = env.reset(seed=ctx.seed + episodes_total)
                episode_return, episode_len = 0.0, 0
            else:
                obs = next_obs

            # ---- ⑤ 梯度步：off-policy 每步一次 mini-batch 更新 ----
            if (not warmup) and steps_total % train_freq == 0 and buffer.can_sample(batch_size):
                for _ in range(gradient_steps):
                    batch = buffer.sample(batch_size)
                    running.update(agent.update_batch(batch, optimizers, gradient_clip))
                    updates_total += 1

            # ---- ⑥ 训练监控（横轴 = 累计环境交互步数，与 RL 论文曲线一致）----
            if steps_total % log_interval == 0 or steps_total == total_timesteps:
                stats = running.mean()
                running.reset()
                tb.add_scalar("train/steps_total", float(steps_total), steps_total)
                tb.add_scalar("train/updates_total", float(updates_total), steps_total)
                tb.add_scalar("train/alpha", float(agent.alpha.detach()), steps_total)
                tb.add_scalar("train/log_alpha", float(agent.log_alpha.detach()), steps_total)
                if agent.target_entropy is not None:
                    tb.add_scalar("train/target_entropy", float(agent.target_entropy), steps_total)
                for key in ("q1_loss", "q2_loss", "critic_loss", "actor_loss", "alpha_loss",
                            "target_q_mean", "next_q_mean", "td_abs_mean", "reward_mean",
                            "done_fraction", "log_prob_mean", "entropy_mean", "entropy_target_gap",
                            "mean_mu_abs", "mean_std", "log_std_mean", "saturation_fraction",
                            "critic_grad_norm", "actor_grad_norm", "alpha_grad_norm",
                            "critic_frozen_ok"):
                    if key in stats:
                        tb.add_scalar(f"train/{key}", float(stats[key]), steps_total)
                bstats = buffer.stats()
                for key in ("size", "fill_ratio", "reward_mean", "reward_std",
                            "terminated_fraction", "truncated_fraction", "action_abs_mean"):
                    if key in bstats and not math.isnan(float(bstats[key])):
                        tb.add_scalar(f"buffer/{key}", float(bstats[key]), steps_total)
                if stats:
                    q_loss_text = (
                        f"{stats.get('q1_loss', float('nan')):.4f}/{stats.get('q2_loss', float('nan')):.4f}"
                        if agent.critic.n_critics > 1 else
                        f"{stats.get('q1_loss', float('nan')):.4f}（单Q）"
                    )
                    logger.info(
                        "Step %6d | 更新 %6d | 回合 %4d | 滑动(%d) %8.2f | Q损失 %s | "
                        "策略损失 %8.4f | α %7.4f | 熵 %7.4f | σ %6.3f | "
                        "目标Q %8.3f | 梯度(c/a/α) %.2f/%.2f/%.2f | 缓冲 %.0f 条",
                        steps_total, updates_total, len(history), target_window, rolling,
                        q_loss_text,
                        stats.get("actor_loss", float("nan")), float(agent.alpha.detach()),
                        stats.get("entropy_mean", float("nan")), stats.get("mean_std", float("nan")),
                        stats.get("target_q_mean", float("nan")),
                        stats.get("critic_grad_norm", 0.0), stats.get("actor_grad_norm", 0.0),
                        stats.get("alpha_grad_norm", 0.0), bstats["size"],
                    )

            # ---- ⑦ 独立评估 + 存盘（模型选择用"评估回报"，不用训练回报）----
            if steps_total % eval_interval == 0 or steps_total == total_timesteps:
                eval_result = evaluate_policy(
                    agent.actor, env_name=ctx.env_name, device=device, episodes=eval_episodes,
                    seed=eval_seed, gamma=gamma, deterministic=eval_deterministic,
                    normalizer=normalizer,
                )
                last_eval = eval_result
                eval_history.append(dict(eval_result))
                # 当前策略的熵 / σ / |μ|（在"当前训练状态"上蒙特卡洛估计，用于 α 轨迹解读）
                with torch.no_grad():
                    probe = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
                    outs = agent.actor.sample(probe.expand(64, -1))
                    cur_entropy = float(-outs["log_prob"].mean())
                    cur_std = float(outs["std"].mean())
                    cur_mu_abs = float(outs["mu"].abs().mean())
                alpha_history.append({
                    "step": float(steps_total), "alpha": float(agent.alpha.detach()),
                    "log_alpha": float(agent.log_alpha.detach()), "entropy": cur_entropy,
                    "mean_std": cur_std, "mean_mu_abs": cur_mu_abs,
                    "eval_return": float(eval_result["mean_return"]),
                })
                tb.add_scalar("eval/alpha", float(agent.alpha.detach()), steps_total)
                tb.add_scalar("eval/entropy_mean", cur_entropy, steps_total)
                tb.add_scalar("eval/mean_std", cur_std, steps_total)
                tb.add_scalar("eval/mean_mu_abs", cur_mu_abs, steps_total)
                if first_solved_step is None and eval_result["mean_return"] >= target_reward:
                    first_solved_step = steps_total
                    tb.add_scalar("eval/first_solved_step", float(steps_total), steps_total)
                    logger.info(
                        "★ 样本效率：首次突破目标回报 %.1f @ %d 步（%d 次梯度更新）"
                        " | 评估平均 %.2f | α=%.4f | 熵=%.3f",
                        target_reward, steps_total, updates_total,
                        eval_result["mean_return"], float(agent.alpha.detach()), cur_entropy,
                    )
                report_closed_loop(
                    ctx, eval_result, step=steps_total,
                    label=f"独立评估（{'均值动作' if eval_deterministic else '采样'}，{eval_episodes} 回合）",
                    extra_tags={
                        "eval/std_return": eval_result["std_return"],
                        "eval/min_return": eval_result["min_return"],
                        "eval/max_return": eval_result["max_return"],
                        "eval/mean_length": eval_result["mean_length"],
                        "eval/rolling_train_return": rolling,
                        "eval/alpha": float(agent.alpha.detach()),
                        "eval/updates_total": float(updates_total),
                    },
                )
                save_info = _checkpoint(steps_total, eval_result["mean_return"], {
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
                        "（累计 %d 步 / %d 次梯度更新）",
                        eval_result["mean_return"], target_reward, steps_total, updates_total,
                    )
                    break
            elif ckpt_interval and steps_total % ckpt_interval == 0:
                # ---- ⑦' 非评估点的周期性存盘（重启后最多丢 ckpt_interval 步）----
                _checkpoint(steps_total, rolling, {"rolling_return": rolling})

            # ---- ⑧ 按训练滑动均值提前停止（与评估判据互为交叉验证）----
            if rolling >= target_reward and len(history) >= target_window:
                early_stopped = True
                logger.info(
                    "★ 最近 %d 个训练回合平均回报 %.2f ≥ 目标 %.1f：提前结束训练"
                    "（累计 %d 步 / %d 次梯度更新）",
                    target_window, rolling, target_reward, steps_total, updates_total,
                )
                break

            # ---- ⑨ 优雅停机（SIGINT / SIGTERM）----
            if stop_requested["flag"]:
                interrupted = True
                _checkpoint(steps_total, rolling, {"rolling_return": rolling, "interrupted": True})
                logger.warning(
                    "已保存检查点并于 step %d 安全退出（累计 %d 步 / %d 次更新）；继续训练可执行：\n"
                    "  python %s --config %s --resume %s",
                    steps_total, steps_total, updates_total,
                    SCRIPT_PATH, ctx.args.config, checkpointer.last_path,
                )
                break
    except Exception:
        # 异常也最大限度保留成果：先写应急检查点再向上抛
        _checkpoint(steps_total, rolling, {"emergency": True, "rolling_return": rolling})
        logger.error("训练异常，已写应急检查点: %s", checkpointer.last_path)
        raise
    finally:
        env.close()

    train_time = time.time() - start_time
    stats_out: Dict[str, Any] = {
        "best_metric": float(checkpointer.best_metric) if checkpointer.best_metric is not None else 0.0,
        "last_epoch": int(steps_total),
        "interrupted": bool(interrupted),
        "early_stopped": bool(early_stopped),
        "train_time": float(train_time),
        "steps_total": int(steps_total),
        "updates_total": int(updates_total),
        "episodes_total": int(episodes_total),
        "n_episodes": len(history),
        "rolling_return": rolling_mean(history, target_window),
        "first_solved_step": first_solved_step,
        "history": history,
        "last_eval": last_eval,
        "eval_history": eval_history,
        "alpha_history": alpha_history,
        "alpha": float(agent.alpha.detach()),
        "log_alpha": float(agent.log_alpha.detach()),
        "buffer_stats": buffer.stats(),
    }
    logger.info(
        "训练结束，用时 %.1fs | 累计交互步数: %d | 梯度更新: %d 次 | 完成回合: %d | "
        "最近 %d 回合平均回报: %.2f | α: %.4f",
        train_time, steps_total, updates_total, episodes_total,
        min(target_window, len(history)), stats_out["rolling_return"], stats_out["alpha"],
    )
    return stats_out

# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("SAC（Soft Actor-Critic）训练脚本（配置驱动）", CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    train_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    eval_cfg = config.setdefault("eval", {})
    # 规范化布尔字段：`--set training.bootstrap_on_truncation=false`（小写）也能识别
    model_cfg["normalize_obs"] = as_bool(model_cfg.get("normalize_obs", False), False)
    train_cfg["bootstrap_on_truncation"] = as_bool(train_cfg.get("bootstrap_on_truncation", True), True)
    train_cfg["use_min_for_actor"] = as_bool(train_cfg.get("use_min_for_actor", True), True)
    eval_cfg["deterministic"] = as_bool(eval_cfg.get("deterministic", True), True)
    eval_cfg["render"] = as_bool(eval_cfg.get("render", False), False)

    lr_actor = float(train_cfg.get("lr_actor", 3e-4))
    lr_critic = float(train_cfg.get("lr_critic", 1e-3))
    lr_alpha = float(train_cfg.get("lr_alpha", 3e-4))
    target_reward = float(train_cfg.get("target_reward", -200.0))

    # -------- 基础设置：设备 / 种子 / 运行目录 / 日志 / TensorBoard / 续训解析 --------
    ctx = prepare_run(
        args, config, title=TITLE, logger_name="sac",
        extra_info=[
            "算法       : SAC（off-policy 最大熵 RL；经验回放 + Double-Q + 自动温度）",
            f"超参数     : γ={train_cfg.get('gamma', 0.99)} | τ={train_cfg.get('tau', 0.005)}"
            f" | α_init={train_cfg.get('init_alpha', 1.0)}"
            f" | 目标熵={train_cfg.get('target_entropy', 'auto')}（auto ⇒ −dim(A)）"
            f" | batch={train_cfg.get('batch_size', 256)}"
            f" | buffer={train_cfg.get('buffer_size', 100000)}",
            f"学习率     : actor={lr_actor:.1e} | critic={lr_critic:.1e} | alpha={lr_alpha:.1e}"
            f" | 热身={train_cfg.get('warmup_steps', 1000)} 步 | 目标回报={target_reward}",
        ],
    )
    logger, device = ctx.logger, ctx.device

    # -------- 1. 环境 / 网络 / 优化器 / 回放池 --------
    probe = make_env(ctx.env_name)
    check_action_bounds(config, probe)
    logger.info(
        "[1/3] 环境: %s | 状态空间 %s | 动作空间 %s | 单回合上限 %d 步",
        ctx.env_name, probe.observation_space, probe.action_space, env_max_steps(probe),
    )
    probe.close()

    agent = build_sac_agent(config, device=device)
    optimizers = build_sac_optimizers(
        agent, lr_actor=lr_actor, lr_critic=lr_critic, lr_alpha=lr_alpha,
        weight_decay=float(train_cfg.get("weight_decay", 0.0) or 0.0),
    )
    logger.info(
        "Actor 网络 : MLP(%d → %d×%d) + 双头（μ / log σ）| tanh 压缩 | 可训练参数: %d",
        agent.state_dim, agent.actor.num_layers, agent.actor.hidden_dim, count_parameters(agent.actor),
    )
    logger.info(
        "Critic 网络: %d × MLP([%d+%d] → %d×%d → 1) | 可训练参数: %d | Actor 用 min(Q1,Q2): %s",
        agent.critic.n_critics, agent.state_dim, agent.action_dim,
        agent.critic.num_layers, agent.critic.hidden_dim, count_parameters(agent.critic),
        "是" if agent.use_min_for_actor else "否（只用 Q1）",
    )
    logger.info(
        "目标网络   : 深拷贝 + 永久 requires_grad=False | Polyak τ=%.4f（每步软更新）| 参数 %d 个",
        agent.tau, sum(p.numel() for p in agent.target_critic.parameters()),
    )
    logger.info(
        "温度 α     : 初值 %.3f（log α=%.3f，%s）| 目标熵 %s | α 下界 %s",
        float(agent.alpha.detach()), float(agent.log_alpha.detach()),
        "可学习" if agent.automatic_entropy else "固定（target_entropy=none）",
        ("%.3f" % agent.target_entropy) if agent.automatic_entropy else "—",
        ("%.4f" % agent.alpha_min) if agent.alpha_min > 0 else "无",
    )
    logger.info(
        "优化器     : 三套独立 Adam（critic %.1e / actor %.1e / alpha %s）"
        " ⇒ Actor 步的梯度不会更新 Critic（思考题）",
        lr_critic, lr_actor, ("%.1e" % lr_alpha) if optimizers["alpha"] is not None else "—",
    )

    normalizer = (RunningMeanStd(int(model_cfg.get("state_dim", 3)))
                  if model_cfg["normalize_obs"] else None)
    buffer = ReplayBuffer(
        capacity=int(train_cfg.get("buffer_size", 100000)),
        state_dim=int(model_cfg.get("state_dim", 3)),
        action_dim=int(model_cfg.get("action_dim", 1)),
        device=device,
        bootstrap_on_truncation=train_cfg["bootstrap_on_truncation"],
        seed=ctx.seed,
    )
    logger.info(
        "回放池     : capacity=%d | state_dim=%d | action_dim=%d（**归一化动作**）| device=%s | "
        "truncated %s",
        buffer.capacity, buffer.state_dim, buffer.action_dim, device,
        "继续自举（正确）" if buffer.bootstrap_on_truncation else "当终止（有偏）",
    )
    checkpointer = ctx.build_checkpointer()

    # -------- 2. 断点续训：恢复权重 / 温度 / 三套优化器 / 回放池 / 累计计数 / 随机源 --------
    steps_done, updates_done, episodes_done, history = 0, 0, 0, []
    if ctx.resume_payload is None:
        logger.info("从头开始训练（未指定 --resume）；前 %d 步为纯随机热身",
                    int(train_cfg.get("warmup_steps", 1000)))
    else:
        payload = ctx.resume_payload
        extra = payload.get("extra") or {}
        agent.load_state_dict(payload["model_state_dict"])
        opt_states = extra.get("optimizer_states") or {}
        for name, opt in optimizers.items():
            state = opt_states.get(name)
            if opt is not None and state:
                opt.load_state_dict(state)
        buffer.load_state_dict(extra.get("replay_buffer_state"))
        if normalizer is not None:
            normalizer.load_state_dict(extra.get("obs_normalizer_state"))
        restore_rng_state(payload.get("rng_state"))
        steps_done = int(extra.get("steps_total") or 0)
        updates_done = int(extra.get("updates_total") or 0)
        episodes_done = int(extra.get("episodes_total") or 0)
        history = list(extra.get("episode_history") or [])
        prev_best = payload.get("best_val_loss", payload.get("metric"))
        if prev_best is not None:
            checkpointer.best_metric = float(prev_best)
        checkpointer.best_epoch = payload.get("best_epoch")
        logger.info(
            "已恢复检查点: %s | 累计交互 %d 步 | 梯度更新 %d 次 | 完成回合 %d | 回放池 %d 条 | "
            "α=%.4f | 历史最优评估 %s @ step %s",
            ctx.resume_path, steps_done, updates_done, episodes_done, buffer.size,
            float(agent.alpha.detach()),
            ("%.2f" % checkpointer.best_metric) if checkpointer.best_metric is not None else "—",
            checkpointer.best_epoch,
        )
        if normalizer is not None and normalizer.count > 0:
            logger.info("已恢复状态归一化统计量: count=%.0f | mean=%s | std=%s",
                        normalizer.count, np.round(normalizer.mean, 3), np.round(normalizer.std, 3))

    # -------- 3. 训练（热身 → 交替 环境步 / 梯度步 → 定期评估）--------
    stats = train_sac(ctx, agent, buffer, optimizers, normalizer, checkpointer,
                      steps_done=steps_done, updates_done=updates_done,
                      episodes_done=episodes_done, history=history)

    # -------- 4. 用"最优权重"做最终评估 --------
    eval_result = None
    render = eval_cfg["render"]
    if ctx.no_eval:
        logger.info("[3/3] 已跳过最终评估 (--no-eval)")
    elif stats["interrupted"]:
        logger.info("[3/3] 因安全中断退出，跳过最终评估（--resume 续训完成后再评估）")
    else:
        if os.path.isfile(checkpointer.best_path):
            logger.info("[3/3] 加载最优权重并做最终评估: %s", checkpointer.best_path)
            best_payload = Checkpointer.load(checkpointer.best_path, agent, map_location=device) or {}
            best_extra = best_payload.get("extra") or {}
            # 归一化统计量也要跟着最优权重走，否则"训练末期的分布"与"最优权重"不匹配
            if normalizer is not None and best_extra.get("obs_normalizer_state") is not None:
                normalizer.load_state_dict(best_extra["obs_normalizer_state"])
        else:
            logger.warning("[3/3] 未找到最优权重，使用当前权重评估")
        render_env = make_env(ctx.env_name, render=True) if render else None
        eval_result = evaluate_policy(
            agent.actor, env_name=ctx.env_name, device=device, episodes=ctx.n_eval_episodes,
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
    target_window = max(1, int(train_cfg.get("target_window", 100)))
    first_solved = stats.get("first_solved_step")
    extra_lines = [
        f"已完成环境步 : {stats['steps_total']}（梯度更新 {stats['updates_total']} 次，完成回合 "
        f"{stats['episodes_total']}，用时 {stats['train_time']:.1f}s）",
        f"训练滑动平均 : {stats['rolling_return']:.2f}（最近 {min(target_window, stats['n_episodes'])}"
        f"/{target_window} 个训练回合）",
        f"目标回报     : {target_reward:.1f} | {'已达成 ✓' if solved else '未达成（可增加 total_timesteps）'}",
        "样本效率     : " + (
            f"首次评估平均 ≥ {target_reward:.1f} @ {first_solved} 步（约 {stats['updates_total']} 次梯度更新）"
            f" | 对照 W8 PPO：194560 步才到 −199.91（约 4.9 倍步数）"
            if first_solved else "本次训练未突破目标回报"
        ),
    ]
    alpha_hist = stats.get("alpha_history") or []
    if alpha_hist:
        alphas = [h["alpha"] for h in alpha_hist]
        entropies = [h["entropy"] for h in alpha_hist]
        last = alpha_hist[-1]
        extra_lines.append(
            f"α 自适应轨迹 : 1.000 → {last['alpha']:.4f}（最小 {min(alphas):.4f} / 最大 {max(alphas):.4f}）"
            f" | 策略熵 {entropies[0]:.3f} → {last['entropy']:.3f}"
            f"（目标 |H̄| = {abs(agent.target_entropy):.1f} = dim(A)）"
        )
        extra_lines.append(
            f"末态策略诊断 : σ {last['mean_std']:.4f} | |μ| {last['mean_mu_abs']:.4f}"
            f" | 评估回报 {last['eval_return']:.2f}"
        )
    bstats = stats.get("buffer_stats") or {}
    extra_lines.append(
        "回放池末态   : %d/%d 条（填充率 %.3f）| 奖励均值 %.2f | terminated/truncated %.3f/%.3f"
        % (int(bstats.get("size", 0)), int(bstats.get("capacity", 0)), bstats.get("fill_ratio", 0.0),
           bstats.get("reward_mean", float("nan")), bstats.get("terminated_fraction", 0.0),
           bstats.get("truncated_fraction", 0.0))
    )
    extra_lines.append(
        "思考题诊断   : Actor 步只冻结 Critic 的“参数”（frozen_parameters）、不切断计算图"
        " ⇒ train/critic_frozen_ok 恒为 1.0，且 train/actor_grad_norm > 0"
    )
    if normalizer is not None:
        extra_lines.append(
            "状态归一化   : count=%d | mean=%s | std=%s"
            % (normalizer.count, np.round(normalizer.mean, 3), np.round(normalizer.std, 3))
        )
    extra_lines.append(
        f"曲线导出     : python experiments/export_tb_curves.py --logdir {ctx.run_dir} --prefix w9_sac"
    )
    finish_run(
        ctx, checkpointer, stats, eval_result=eval_result, extra_lines=extra_lines,
        resume_command=f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}",
        metric_label="评估回报",
    )


if __name__ == "__main__":
    main()
