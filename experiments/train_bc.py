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
------------------
``Checkpointer`` 会把 **模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源状态**
一并原子写入检查点，并在文件尾部附 sha256 摘要做完整性校验，因此：

* 每个 epoch 都会覆盖写入 ``{name}_last.pt``，进程被 kill / 断电后可从最近完成的
  epoch 继续；由于随机源被一并恢复，续训结果与"不中断"在数值上完全一致；
* 捕获到 SIGINT / SIGTERM 时会在当前 epoch 结束后存盘并安全退出；
* 训练中抛出异常时会先做一次应急存盘再抛出，最大限度保留已有成果。

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

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# 允许脚本以 `python experiments/train_bc.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402  PyYAML：读取 YAML 配置文件

from src.datasets import build_dataloaders, collect_expert_data, evaluate_closed_loop  # noqa: E402
from src.networks import build_network  # noqa: E402
from src.utils import (  # noqa: E402
    Checkpointer,
    TensorBoardLogger,
    get_logger,
    load_checkpoint_file,
    resolve_device,
    set_seed,
)


# =============================================================================
# 命令行参数 & 配置读取
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="行为克隆训练脚本（配置驱动）")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "bc_config.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="命令行覆盖配置项，例如 --set training.epochs=50 model.name=gru",
    )
    parser.add_argument("--device", type=str, default=None, help="覆盖 experiment.device: auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 experiment.seed")
    parser.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help="断点续训：检查点文件 / 目录 / auto(默认，自动找最新)。不传该参数则从头训练",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=None,
        help="每 N 个 epoch 额外归档一份快照（0=关闭），默认取 training.checkpoint.snapshot_every",
    )
    parser.add_argument("--no-tb", action="store_true", help="禁用 TensorBoard")
    parser.add_argument("--no-eval", action="store_true", help="跳过闭环评估")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """使用 PyYAML 读取 YAML 配置文件，返回字典。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _cast(value: str) -> Any:
    """把命令行字符串转换为 python 字面量。"""
    try:
        import ast

        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """按 ``a.b.c`` 路径写入嵌套字典。"""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"无效的覆盖项 '{item}'，应为 KEY=VALUE 格式")
        key, value = item.split("=", 1)
        _set_nested(config, key.strip(), _cast(value.strip()))
    return config


# =============================================================================
# 训练 / 验证
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 0.0,
) -> float:
    """训练一个 epoch，返回平均 MSE。"""
    model.train()
    loss_sum, n_samples = 0.0, 0
    for batch_s, batch_a in loader:
        batch_s = batch_s.to(device, non_blocking=True)
        batch_a = batch_a.to(device, non_blocking=True)

        pred = model(batch_s)
        loss = criterion(pred, batch_a)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_sum += loss.item() * batch_s.size(0)
        n_samples += batch_s.size(0)
    return loss_sum / max(1, n_samples)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """在验证集上评估，返回平均 MSE。"""
    model.eval()
    loss_sum, n_samples = 0.0, 0
    for batch_s, batch_a in loader:
        batch_s = batch_s.to(device, non_blocking=True)
        batch_a = batch_a.to(device, non_blocking=True)
        pred = model(batch_s)
        loss = criterion(pred, batch_a)
        loss_sum += loss.item() * batch_s.size(0)
        n_samples += batch_s.size(0)
    return loss_sum / max(1, n_samples)


def build_scheduler(optimizer: optim.Optimizer, sched_cfg: Dict[str, Any]):
    """按配置构建学习率调度器，未指定则返回 None。"""
    name = str(sched_cfg.get("name", "none")).lower()
    if name in ("none", ""):
        return None
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sched_cfg.get("T_max", 100)))
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 30)),
            gamma=float(sched_cfg.get("gamma", 0.5)),
        )
    raise ValueError(f"未知调度器 '{name}'，可选: none | cosine | step")


# =============================================================================
# 主流程
# =============================================================================
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.overrides)

    # -------- 读取配置（新 schema）--------
    exp_cfg = config.get("experiment", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    aug_cfg = config.get("augmentation", {})
    paths = config.get("paths", {})
    eval_cfg = config.get("eval", {})

    # 命令行优先级覆盖
    if args.device:
        exp_cfg["device"] = args.device
    if args.seed is not None:
        exp_cfg["seed"] = args.seed

    experiment_name = str(exp_cfg.get("name", "bc_pendulum"))  # 用于 TensorBoard / Checkpointer 命名
    seed = int(exp_cfg.get("seed", 42))
    env_name = str(exp_cfg.get("env_name", "Pendulum-v1"))
    device = resolve_device(str(exp_cfg.get("device", "auto")))

    # -------- 基础设置 --------
    set_seed(seed)

    # -------- 断点续训：解析 --resume 目标并预读检查点（用于复用原运行目录）--------
    ckpt_cfg = train_cfg.get("checkpoint", {}) or {}
    log_root = paths.get("log_dir", "logs")
    ckpt_root = paths.get("checkpoint_dir", os.path.join(log_root, "checkpoints"))
    monitor = str(ckpt_cfg.get("monitor", "val_mse"))
    monitor_mode = str(ckpt_cfg.get("mode", "min"))
    snapshot_every = (
        args.snapshot_every if args.snapshot_every is not None
        else int(ckpt_cfg.get("snapshot_every", 0))
    )
    keep_last_n = int(ckpt_cfg.get("keep_last_n", 2))

    resume_target = args.resume if args.resume is not None else paths.get("resume")
    resume_path = Checkpointer.resolve_path(resume_target, checkpoint_dir=ckpt_root)
    resume_payload = load_checkpoint_file(resume_path, map_location="cpu") if resume_path else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_run_dir = ((resume_payload or {}).get("extra") or {}).get("run_dir")
    if resume_payload is not None and saved_run_dir and os.path.isdir(saved_run_dir):
        # 续训：复用原运行的 TensorBoard 目录，曲线 / 日志保持连续
        run_dir = saved_run_dir
        checkpoint_dir = os.path.dirname(resume_path)
    else:
        run_dir = os.path.join(log_root, f"{experiment_name}_{timestamp}")      # TensorBoard 目录含实验名
        checkpoint_dir = os.path.join(
            ckpt_root, f"{experiment_name}_{timestamp}"                         # Checkpointer 目录含实验名
        )
    os.makedirs(run_dir, exist_ok=True)

    logger = get_logger("bc", log_file=os.path.join(run_dir, "train.log"))
    tb = TensorBoardLogger(run_dir, enabled=not args.no_tb)                     # 用实验名所在目录初始化 TensorBoard

    logger.info("=" * 60)
    logger.info("行为克隆（Behavior Cloning）实验：%s", env_name)
    logger.info("=" * 60)
    logger.info("配置文件   : %s", args.config)
    logger.info("实验名称   : %s", experiment_name)
    logger.info("运行目录   : %s", run_dir)
    logger.info("检查点目录 : %s", checkpoint_dir)
    logger.info("设备       : %s", device)
    logger.info("随机种子   : %d", seed)
    logger.info("网络结构   : %s", config.get("model", {}).get("name", "mlp"))
    if resume_path:
        logger.info("断点续训   : %s", resume_path)
    tb.add_text("config", "```yaml\n" + _config_to_text(config) + "\n```", 0)

    # -------- 1. 收集 / 加载专家数据 --------
    logger.info("[1/3] 准备专家数据 ...")
    cache_path = os.path.join(
        paths.get("data_dir", "data/expert_pendulum"),
        data_cfg.get("cache_file", "expert_pendulum.npz"),
    )
    X_train, y_train, X_val, y_val = collect_expert_data(
        env_name=env_name,
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

    model = build_network(config).to(device)
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
    checkpointer = Checkpointer(
        checkpoint_dir,
        filename=f"{experiment_name}_best.pt",
        monitor=monitor,
        mode=monitor_mode,
        snapshot_every=snapshot_every,
        keep_last_n=keep_last_n,
    )

    # -------- 2.5 断点续训：恢复模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源 --------
    start_epoch = 1
    if resume_payload is not None:
        ckpt = Checkpointer.load(
            path=resume_path,
            model=model,
            map_location=device,
            optimizer=optimizer,
            scheduler=scheduler,
            restore_rng=True,            # 恢复随机源 → 与"不中断"的训练轨迹完全一致
            payload=resume_payload,      # 复用已读出的内容，避免重复读盘
        )
        start_epoch = int(ckpt.get("epoch") or 0) + 1
        prev_best = ckpt.get("best_val_loss", ckpt.get("metric"))
        if prev_best is not None:
            checkpointer.best_metric = float(prev_best)
        checkpointer.best_epoch = ckpt.get("best_epoch")
        logger.info(
            "已恢复检查点 | 已完成 epoch: %s | 历史最优 %s: %.6f @ epoch %s | 将从 epoch %d 继续",
            ckpt.get("epoch"), monitor, checkpointer.best_metric,
            checkpointer.best_epoch, start_epoch,
        )
        tb.add_text(
            "resume",
            f"resumed from `{resume_path}`\n\n"
            f"- finished_epoch: {ckpt.get('epoch')}\n"
            f"- best_{monitor}: {checkpointer.best_metric:.6f} @ epoch {checkpointer.best_epoch}\n"
            f"- checkpoint_created_at: {ckpt.get('created_at')}\n",
            start_epoch,
        )
    else:
        logger.info("从头开始训练（未指定 --resume）")

    # -------- 3. 训练循环（支持优雅中断 / 异常恢复）--------
    epochs = int(train_cfg.get("epochs", 100))
    log_interval = int(train_cfg.get("log_interval", 20))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0))
    if start_epoch > epochs:
        logger.warning("[2/3] 检查点已完成 %d 个 epoch（目标 %d），无需继续训练", start_epoch - 1, epochs)
    else:
        logger.info("[2/3] 开始训练：epoch %d → %d", start_epoch, epochs)

    # 信号处理：首次 SIGINT/SIGTERM 在当前 epoch 结束后存盘退出，再次收到则强制退出
    stop_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        if stop_requested["flag"]:
            logger.warning("再次收到信号 %s，强制退出（当前 epoch 进度将丢失）", signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        stop_requested["flag"] = True
        logger.warning("收到信号 %s：将在当前 epoch 结束后保存检查点并安全退出", signum)

    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _handle_signal)
        except (ValueError, OSError, AttributeError):   # pragma: no cover - 非主线程等场景
            pass

    last_epoch = start_epoch - 1
    last_val_mse = None
    best_val_mse = checkpointer.best_metric
    interrupted = False
    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_mse = train_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)
            val_mse = validate(model, val_loader, criterion, device)

            if scheduler is not None:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            # 每个 epoch 都写 last（异常恢复用）；刷新最优时再写 best
            save_info = checkpointer.step(
                model, optimizer, scheduler,
                epoch=epoch, metric=val_mse, config=config,
                extras={"run_dir": run_dir, "seed": seed},
            )
            best_val_mse = save_info["best_metric"]
            last_epoch, last_val_mse = epoch, val_mse

            tb.add_scalars("loss", {"train": train_mse, "val": val_mse}, epoch)
            tb.add_scalar("lr", lr, epoch)
            tb.add_scalar("best_val_mse", best_val_mse, epoch)

            if epoch % log_interval == 0 or epoch == start_epoch or epoch == epochs or save_info["is_best"]:
                logger.info(
                    "Epoch [%d/%d] | Train MSE: %.6f | Val MSE: %.6f | Best: %.6f | LR: %.2e%s",
                    epoch, epochs, train_mse, val_mse, best_val_mse, lr,
                    "  *best" if save_info["is_best"] else "",
                )

            if stop_requested["flag"]:
                interrupted = True
                logger.warning(
                    "已保存检查点并于 epoch %d 安全退出；继续训练可执行：\n"
                    "  python experiments/train_bc.py --config %s --resume %s",
                    epoch, args.config, checkpointer.last_path,
                )
                break
    except BaseException as exc:            # 异常兜底：尽最大努力保留已有成果
        logger.error("训练过程中出现异常 (%s: %s)，正在保存应急检查点 ...", type(exc).__name__, exc)
        try:
            checkpointer.save_last(
                model, optimizer, scheduler, epoch=last_epoch, metric=last_val_mse, config=config,
                extras={"run_dir": run_dir, "seed": seed, "reason": f"crash:{type(exc).__name__}"},
            )
            logger.error("应急检查点已保存: %s", checkpointer.last_path)
        except Exception as save_exc:       # pragma: no cover
            logger.error("应急检查点保存失败: %s", save_exc)
        raise
    finally:
        tb.flush()                          # 无论走哪条路径，先把监控数据落盘

    train_time = time.time() - start_time
    logger.info(
        "训练结束，用时 %.1fs | 已完成 epoch: %d | 最佳验证集 MSE: %.6f",
        train_time, last_epoch, best_val_mse,
    )
    logger.info("最优权重: %s", checkpointer.best_path)
    logger.info("最近权重: %s", checkpointer.last_path)
    logger.info("%s", checkpointer.summary())
    tb.add_scalar("train_time_sec", train_time, 0)

    # -------- 4. 加载最优权重并闭环评估 --------
    eval_result = None
    if args.no_eval:
        logger.info("[3/3] 已跳过闭环评估 (--no-eval)")
    elif interrupted:
        logger.info("[3/3] 因安全中断退出，跳过闭环评估（--resume 续训完成后再评估）")
    elif not os.path.isfile(checkpointer.best_path):
        logger.warning("[3/3] 未找到最优权重 %s，跳过闭环评估", checkpointer.best_path)
    else:
        logger.info("[3/3] 加载最优权重并进行闭环评估 ...")
        Checkpointer.load(checkpointer.best_path, model, map_location=device)
        eval_result = evaluate_closed_loop(
            model,
            device=device,
            env_name=env_name,
            n_episodes=int(eval_cfg.get("n_episodes", 20)),
            render=bool(eval_cfg.get("render", False)),
            seed=seed,
        )
        tb.add_scalar("eval/mean_return", eval_result["mean_return"], 0)
        logger.info(
            "闭环评估 | 回合数: %d | 平均回报: %.2f ± %.2f (min %.2f, max %.2f)",
            int(eval_cfg.get("n_episodes", 20)),
            eval_result["mean_return"], eval_result["std_return"],
            eval_result["min_return"], eval_result["max_return"],
        )
    # -------- 总结 --------
    logger.info("=" * 60)
    logger.info("最终实验结果")
    logger.info("=" * 60)
    logger.info("最佳验证集 MSE : %.6f (epoch %s)", best_val_mse, checkpointer.best_epoch)
    logger.info("已完成 epoch   : %d", last_epoch)
    if eval_result is not None:
        logger.info("闭环平均回报   : %.2f", eval_result["mean_return"])
    logger.info("TensorBoard    : tensorboard --logdir %s", log_root)
    logger.info("最优权重       : %s", checkpointer.best_path)
    logger.info("最近权重       : %s", checkpointer.last_path)
    if interrupted:
        logger.info(
            "断点续训       : python experiments/train_bc.py --config %s --resume %s",
            args.config, checkpointer.last_path,
        )

    tb.flush()
    tb.close()


def _config_to_text(config: Dict[str, Any]) -> str:
    """把配置渲染成 YAML 文本，用于写入 TensorBoard。"""
    try:
        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    except Exception:  # pragma: no cover
        return str(config)


if __name__ == "__main__":
    main()
