#!/usr/bin/env python
"""动作块行为克隆（Action Chunking BC）训练脚本 —— W4 Task 1。

任务
----
在已跑通的单步 BC 流水线（``experiments/train_bc.py``）基础上实现动作块行为克隆：

1. **动作块标签**：对专家轨迹做滑窗切分，标签由 ``a_t`` 变为
   ``[a_t, a_{t+1}, ..., a_{t+K-1}]``（形状 ``(K, action_dim)``）；
   窗口**不跨回合边界**，轨迹末端不足 ``K`` 步的部分**零填充**；
2. **网络输出头**：维度由 ``action_dim`` 变为 ``K * action_dim``（``model.chunk_size = K``）；
3. **闭环执行**：每隔 ``K`` 步推理一次，一次前向打出 ``K`` 步动作块并**开环执行**
   （``execution="chunk"``）；也可选用 ``execution="ema"`` 在相邻动作块之间做指数
   移动平均以抑制块间跳变，或把 ``replan_every`` 设为 1 退化为"每步重规划、只执行
   块首动作"（receding horizon）。

与单步 BC 的可比性
------------------
* 数据切分方式逐样本对齐：同为"按状态样本随机 ``train_ratio`` 切分"，
  ``numpy.random.default_rng(seed).permutation(N)``，且训练集/验证集的**状态**集合
  完全一致，唯一差别是标签由 ``a_t`` 变成 ``K`` 步动作块；
* 网络结构、优化器、学习率、epoch 数、数据增强、随机种子均与单步 BC 相同
  （配置通过 ``defaults: bc_config.yaml`` 继承），因此闭环回报的差异只来自
  "动作块建模 + 每隔 K 步推理"这一变量。

工程实现
--------
``--config/--set/--resume`` 解析、运行目录 / 日志 / TensorBoard 装配、epoch 训练
循环、优雅停机与检查点等通用逻辑全部复用 ``src/engine.py``，与单步 BC 共享同一套
训练引擎。

用法示例::

    python experiments/train_chunking_bc.py                                   # K=8（默认配置）
    python experiments/train_chunking_bc.py --set model.chunk_size=4          # K=4
    python experiments/train_chunking_bc.py --set inference.execution=ema     # 块间 EMA 平滑
    python experiments/train_chunking_bc.py --device cpu --no-eval            # 只训练不评估
    python experiments/train_chunking_bc.py --resume                          # 断点续训
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import torch.nn as nn
import torch.optim as optim

# 允许脚本以 `python experiments/train_chunking_bc.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import (  # noqa: E402
    build_chunking_dataloaders,
    collect_expert_trajectories,
    evaluate_closed_loop_chunking,
)
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
)
from src.networks import build_network  # noqa: E402
from src.utils import Checkpointer  # noqa: E402

CONFIG_PATH = os.path.join("configs", "chunking_bc_config.yaml")
TITLE = "动作块行为克隆（Action Chunking BC）实验"
SCRIPT_PATH = "experiments/train_chunking_bc.py"


def resolve_chunk_size(config) -> int:
    """解析动作块长度 ``K``：``chunking.chunk_size`` 优先，否则跟随 ``model.chunk_size``。

    两级配置必须一致（否则网络输出头与标签形状不匹配），不一致时直接报错。
    """
    model_k = max(1, int(config.get("model", {}).get("chunk_size", 1) or 1))
    data_k = int(config.get("chunking", {}).get("chunk_size") or model_k)
    if data_k != model_k:
        raise ValueError(
            f"chunking.chunk_size ({data_k}) 与 model.chunk_size ({model_k}) 不一致；"
            "请把 chunking.chunk_size 设为 null（跟随 model.chunk_size）或二者取相同值。"
        )
    return model_k


def evaluate_chunked_policies(model, ctx, chunk_size: int, infer_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """在最优权重上做动作块闭环评估，返回 ``{模式名: 评估结果}``。

    模式：

    * 主模式 —— ``inference.execution`` 指定（``chunk`` 开环整块 / ``ema`` 块间平滑），
      每 ``K`` 步（或 ``inference.replan_every``）推理一次；
    * 对照模式 —— ``inference.compare_executions=true`` 时额外评估 ``ema`` 与
      ``receding``（每步重规划、只执行块首动作），用于说明"开环执行 vs 重规划"的影响。
    """
    execution = str(infer_cfg.get("execution", "chunk")).lower()
    ema_alpha = float(infer_cfg.get("ema_alpha", 0.5))
    replan_every = int(infer_cfg.get("replan_every", 0) or 0)
    render = bool(ctx.eval_cfg.get("render", False))

    modes: Dict[str, Tuple[str, int]] = {execution: (execution, replan_every)}
    if bool(infer_cfg.get("compare_executions", False)):
        # EMA 平滑只有在相邻规划**时间重叠**时才生效 → 其重规划间隔取 K//2（可配置）
        ema_replan = infer_cfg.get("ema_replan_every")
        ema_replan = max(1, int(ema_replan)) if ema_replan else max(1, chunk_size // 2)
        modes.setdefault("ema", ("ema", ema_replan))
        modes.setdefault("receding", ("chunk", int(infer_cfg.get("receding_replan_every", 1) or 1)))

    results: Dict[str, Dict[str, Any]] = {}
    for name, (mode_exec, mode_replan) in modes.items():
        result = evaluate_closed_loop_chunking(
            model,
            device=ctx.device,
            env_name=ctx.env_name,
            n_episodes=ctx.n_eval_episodes,
            chunk_size=chunk_size,
            execution=mode_exec,
            ema_alpha=ema_alpha,
            replan_every=mode_replan,
            render=render,
            seed=ctx.seed,
        )
        results[name] = result
        report_closed_loop(
            ctx, result,
            tag="eval/mean_return" if name == execution else f"eval/mean_return_{name}",
            label=f"闭环评估（{name}，每 {result['execute_steps']} 步推理一次）",
        )
        ctx.logger.info(
            "推理开销   | 前向次数: %d | 每回合平均: %.1f 次（单步 BC 约 200 次）",
            result["n_forward_passes"], result["forward_per_episode"],
        )
        ctx.tb.add_scalar(f"eval/forward_per_episode_{name}", result["forward_per_episode"], 0)
    return results


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("动作块行为克隆（Action Chunking BC）训练脚本（配置驱动）", CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    model_cfg = config.get("model", {})
    infer_cfg = config.get("inference", {})

    chunk_size = resolve_chunk_size(config)                       # 动作块长度 K
    action_dim = int(model_cfg.get("action_dim", 1))

    # -------- 基础设置：设备 / 种子 / 运行目录 / 日志 / TensorBoard / 续训解析 --------
    ctx = prepare_run(
        args, config, title=TITLE, logger_name="chunking_bc",
        extra_info=[
            f"动作块长度 : {chunk_size}  (K * action_dim = {chunk_size * action_dim} 维输出头)",
            f"执行方式   : {infer_cfg.get('execution', 'chunk')} "
            f"(replan_every={infer_cfg.get('replan_every', 0) or chunk_size}, "
            f"ema_alpha={infer_cfg.get('ema_alpha', 0.5)})",
        ],
    )
    logger, device, seed = ctx.logger, ctx.device, ctx.seed
    train_cfg, data_cfg, aug_cfg = ctx.train_cfg, ctx.data_cfg, ctx.aug_cfg

    # -------- 1. 采集 / 加载保留回合边界的专家轨迹 --------
    logger.info("[1/3] 准备专家轨迹（保留回合边界，供滑窗切分）...")
    cache_path = os.path.join(
        ctx.paths.get("data_dir", "data/expert_pendulum"),
        data_cfg.get("cache_file", "expert_pendulum_traj.npz"),
    )
    episodes = collect_expert_trajectories(
        env_name=ctx.env_name,
        num_expert_steps=int(data_cfg.get("num_expert_steps", 10000)),
        seed=seed,
        reuse_cache=bool(data_cfg.get("reuse_cache", True)),
        cache_path=cache_path,
    )
    lengths = [len(s) for s, _ in episodes]
    logger.info(
        "专家轨迹 %d 条 / 共 %d 步 | 每条长度 %d ~ %d",
        len(episodes), int(sum(lengths)), min(lengths), max(lengths),
    )

    # -------- 2. 滑窗切分 K 步动作块标签 + 构建 DataLoader --------
    train_loader, val_loader, info = build_chunking_dataloaders(
        episodes,
        chunk_size=chunk_size,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        augment=bool(aug_cfg.get("enabled", False)),
        state_noise_std=float(aug_cfg.get("state_noise_std", 0.0)),
        action_noise_std=float(aug_cfg.get("action_noise_std", 0.0)),
        seed=seed,
    )
    logger.info(
        "训练样本 %d | 验证样本 %d（切分方式与单步 BC 逐样本对齐）",
        info["n_train"], info["n_val"],
    )
    logger.info(
        "动作块标签形状: (K=%d, action_dim=%d) | 零填充比例: %.2f%%（训练集 %.2f%%）",
        chunk_size, action_dim,
        100.0 * info["padding_ratio"], 100.0 * info["train_padding_ratio"],
    )
    ctx.tb.add_scalar("chunk/chunk_size", chunk_size, 0)
    ctx.tb.add_scalar("chunk/padding_ratio", info["padding_ratio"], 0)

    # -------- 3. 构建网络 / 优化器 / 检查点 --------
    model = build_network(config).to(device)                     # 输出头 = K * action_dim
    if int(getattr(model, "output_dim", 0)) != chunk_size * action_dim:
        raise ValueError(
            f"网络输出维度 {getattr(model, 'output_dim', None)} 与 K*action_dim "
            f"({chunk_size}*{action_dim}) 不一致，请检查 model.chunk_size 配置。"
        )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "可训练参数量: %d（输出头 %d 维 = %d x %d）",
        n_params, model.output_dim, chunk_size, action_dim,
    )

    criterion = nn.MSELoss()                    # 逐元素 MSE：与单步 BC 同一优化目标
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}))
    checkpointer = ctx.build_checkpointer()

    # -------- 3.5 断点续训：恢复模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源 --------
    start_epoch = restore_from_checkpoint(ctx, model, optimizer, scheduler, checkpointer)

    # -------- 4. 训练循环（与单步 BC 共用同一套引擎）--------
    stats = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, checkpointer, ctx,
        start_epoch=start_epoch,
        resume_hint=f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}",
    )
    interrupted = bool(stats["interrupted"])

    logger.info(
        "训练结束，用时 %.1fs | 已完成 epoch: %d | 最佳验证集 MSE: %.6f",
        stats["train_time"], stats["last_epoch"], stats["best_metric"],
    )
    logger.info("最优权重: %s", checkpointer.best_path)
    logger.info("最近权重: %s", checkpointer.last_path)
    logger.info("%s", checkpointer.summary())
    ctx.tb.add_scalar("train_time_sec", stats["train_time"], 0)

    # -------- 5. 加载最优权重并闭环评估（每隔 K 步推理一次）--------
    eval_results: Dict[str, Dict[str, Any]] = {}
    primary_mode = str(infer_cfg.get("execution", "chunk")).lower()
    if ctx.no_eval:
        logger.info("[3/3] 已跳过闭环评估 (--no-eval)")
    elif interrupted:
        logger.info("[3/3] 因安全中断退出，跳过闭环评估（--resume 续训完成后再评估）")
    elif not os.path.isfile(checkpointer.best_path):
        logger.warning("[3/3] 未找到最优权重 %s，跳过闭环评估", checkpointer.best_path)
    else:
        logger.info("[3/3] 加载最优权重并进行闭环评估（动作块执行，K=%d）...", chunk_size)
        Checkpointer.load(checkpointer.best_path, model, map_location=device)
        eval_results = evaluate_chunked_policies(
            model, ctx, chunk_size=chunk_size, infer_cfg=infer_cfg
        )

    # -------- 总结 --------
    extra_lines = [
        f"闭环平均回报({mode:<8}) : {result['mean_return']:.2f}"
        f"  [每回合前向 {result['forward_per_episode']:.1f} 次]"
        for mode, result in eval_results.items()
    ]
    if eval_results:
        extra_lines.append("单步 BC 对照：见 experiments/train_bc.py 的运行日志（同等训练设置）")
    finish_run(
        ctx, checkpointer, stats,
        eval_result=eval_results.get(primary_mode),
        extra_lines=extra_lines,
        resume_command=f"python {SCRIPT_PATH} --config {args.config} --resume {checkpointer.last_path}",
    )


if __name__ == "__main__":
    main()

