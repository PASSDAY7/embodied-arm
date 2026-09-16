#!/usr/bin/env python
"""行为克隆（Behavior Cloning）训练脚本 —— 工程化版本（配置驱动）。

配置读取
--------
使用 **PyYAML** 读取 ``configs/bc_config.yaml``（如未安装：``pip install PyYAML``）。
配置结构::

    experiment:  { name, seed, device, env_name }
    model:       { state_dim, action_dim, hidden_dim, ... }
    training:    { batch_size, epochs, lr, max_grad_norm, ... }
    data:        { num_expert_steps, train_ratio, ... }

其中 ``experiment.name`` 会被用于：
* **TensorBoard** 日志目录命名：``{log_dir}/{name}_{timestamp}/``
* **Checkpointer** 权重文件命名：``{name}_best.pt``（最优）与 ``{name}_last.pt``（每 epoch）

断点续训 / 异常恢复
-------------------
``Checkpointer`` 会把 **模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源状态**
一并原子写入检查点，并在文件尾部附 sha256 摘要做完整性校验，因此：

* 每个 epoch 都会覆盖写入 ``{name}_last.pt``，进程被 kill / 断电后可从最近完成的
  epoch 继续；由于随机源被一并恢复，续训结果与"不中断"在数值上完全一致；
* 捕获到 SIGINT / SIGTERM 时会在当前 epoch 结束后存盘并安全退出；
* 训练中抛出异常时会先做一次应急存盘再抛出，最大限度保留已有成果。

工程实现说明
------------
本脚本只负责"单步 BC"特有的部分（专家数据加载、网络构建、闭环评估）；
``--config / --set / --resume`` 解析、运行目录 / 日志 / TensorBoard 装配、epoch
训练循环、优雅停机与检查点等**通用工程逻辑集中在** ``src/engine.py``。
动作块 BC（``experiments/train_chunking_bc.py``）复用同一套引擎，保证两者的
训练数学与工程语义完全一致（唯一变量是标签形状与推理方式）。

用法示例::

    python experiments/train_bc.py --config configs/bc_config.yaml
    python experiments/train_bc.py --config configs/bc_config.yaml --set training.epochs=50
    python experiments/train_bc.py --config configs/bc_config.yaml --set model.name=gru
    python experiments/train_bc.py --config configs/bc_config.yaml --device cpu --no-eval

    # 断点续训：自动寻找最新检查点 / 指定文件 / 指定目录
    python experiments/train_bc.py --resume
    python experiments/train_bc.py --resume logs/checkpoints/xxx/bc_pendulum_last.pt
    python experiments/train_bc.py --resume logs/checkpoints --set training.epochs=200
"""

from __future__ import annotations

import os
import sys

import torch.nn as nn
import torch.optim as optim

# 允许脚本以 `python experiments/train_bc.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import build_dataloaders, collect_expert_data, evaluate_closed_loop  # noqa: E402
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

CONFIG_PATH = os.path.join("configs", "bc_config.yaml")
TITLE = "行为克隆（Behavior Cloning）实验"


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_train_args("行为克隆训练脚本（配置驱动）", default_config=CONFIG_PATH)
    config = apply_overrides(load_experiment_config(args.config), args.overrides)

    # -------- 基础设置：设备 / 种子 / 运行目录 / 日志 / TensorBoard / 续训解析 --------
    ctx = prepare_run(args, config, title=TITLE, logger_name="bc")
    logger, device, seed = ctx.logger, ctx.device, ctx.seed
    train_cfg, data_cfg, aug_cfg = ctx.train_cfg, ctx.data_cfg, ctx.aug_cfg

    # -------- 1. 收集 / 加载专家数据 --------
    logger.info("[1/3] 准备专家数据 ...")
    cache_path = os.path.join(
        ctx.paths.get("data_dir", "data/expert_pendulum"),
        data_cfg.get("cache_file", "expert_pendulum.npz"),
    )
    X_train, y_train, X_val, y_val = collect_expert_data(
        env_name=ctx.env_name,
        num_expert_steps=int(data_cfg.get("num_expert_steps", 10000)),
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        seed=seed,
        reuse_cache=bool(data_cfg.get("reuse_cache", True)),
        cache_path=cache_path,
    )
    logger.info("训练样本 %d | 验证样本 %d", len(X_train), len(X_val))

    # -------- 2. 构建 DataLoader / 网络 / 优化器 --------
    train_loader, val_loader = build_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        augment=bool(aug_cfg.get("enabled", False)),
        state_noise_std=float(aug_cfg.get("state_noise_std", 0.0)),
        action_noise_std=float(aug_cfg.get("action_noise_std", 0.0)),
        seed=seed,
    )

    model = build_network(config).to(device)          # chunk_size 默认 1 → 单步 BC
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("可训练参数量: %d", n_params)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", {}))
    # Checkpointer 使用配置中的 name 命名权重文件：{name}_best.pt / {name}_last.pt
    checkpointer = ctx.build_checkpointer()

    # -------- 2.5 断点续训：恢复模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源 --------
    start_epoch = restore_from_checkpoint(ctx, model, optimizer, scheduler, checkpointer)

    # -------- 3. 训练循环（支持优雅中断 / 异常恢复）--------
    stats = train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler, checkpointer, ctx,
        start_epoch=start_epoch,
        resume_hint=(
            f"python experiments/train_bc.py --config {args.config} --resume {checkpointer.last_path}"
        ),
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

    # -------- 4. 加载最优权重并闭环评估 --------
    eval_result = None
    if ctx.no_eval:
        logger.info("[3/3] 已跳过闭环评估 (--no-eval)")
    elif interrupted:
        logger.info("[3/3] 因安全中断退出，跳过闭环评估（--resume 续训完成后再评估）")
    elif not os.path.isfile(checkpointer.best_path):
        logger.warning("[3/3] 未找到最优权重 %s，跳过闭环评估", checkpointer.best_path)
    else:
        logger.info("[3/3] 加载最优权重并进行闭环评估 ...")
        Checkpointer.load(checkpointer.best_path, model, map_location=device)
        eval_result = evaluate_closed_loop(           # 单步 BC：每一步都推理一次（K=1）
            model,
            device=device,
            env_name=ctx.env_name,
            n_episodes=ctx.n_eval_episodes,
            render=bool(ctx.eval_cfg.get("render", False)),
            seed=seed,
        )
        report_closed_loop(ctx, eval_result, tag="eval/mean_return", label="闭环评估")

    # -------- 总结 --------
    finish_run(
        ctx, checkpointer, stats,
        eval_result=eval_result,
        resume_command=(
            f"python experiments/train_bc.py --config {args.config} --resume {checkpointer.last_path}"
        ),
    )


if __name__ == "__main__":
    main()
