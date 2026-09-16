"""训练引擎（train engine）：单步 BC 与动作块 BC 共用的训练基础设施。

把与具体任务**无关**的工程逻辑集中在这里，保证不同实验脚本使用**完全相同**的
训练数学与工程语义 —— 这既消除了代码重复，也让「单步 BC vs 动作块 BC」的对比
公平可控（唯一变量只有标签与推理方式）：

* 命令行解析（``--config / --set / --device / --seed / --resume / ...``）
* 配置读取与 ``a.b.c=value`` 形式的命令行覆盖
* 训练 / 验证循环、学习率调度器
* 运行目录、TensorBoard、日志、``Checkpointer`` 的装配（:func:`prepare_run`）
* 支持 ``SIGINT``/``SIGTERM`` 优雅停机、异常应急存盘的 epoch 循环（:func:`train_loop`）
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:  # 兼容"作为包导入"与"脚本直接运行"两种场景
    from .utils import (
        Checkpointer,
        TensorBoardLogger,
        get_logger,
        load_checkpoint_file,
        load_config,
        resolve_device,
        set_seed,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        Checkpointer,
        TensorBoardLogger,
        get_logger,
        load_checkpoint_file,
        load_config,
        resolve_device,
        set_seed,
    )


# =============================================================================
# 1. 命令行参数与配置覆盖
# =============================================================================
def build_arg_parser(
    description: str = "行为克隆训练脚本（配置驱动）",
    default_config: str = os.path.join("configs", "bc_config.yaml"),
) -> argparse.ArgumentParser:
    """构建训练脚本通用命令行参数。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=default_config, help="YAML 配置文件路径")
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
    return parser


def parse_train_args(
    description: str = "行为克隆训练脚本（配置驱动）",
    default_config: str = os.path.join("configs", "bc_config.yaml"),
) -> argparse.Namespace:
    """解析训练脚本命令行参数。"""
    return build_arg_parser(description, default_config).parse_args()


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """读取 YAML 配置（支持 ``defaults`` 继承）。"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    return load_config(config_path)


def cast_value(value: str) -> Any:
    """把命令行字符串转换为 python 字面量。"""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """按 ``a.b.c`` 路径写入嵌套字典。"""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """应用 ``--set key=value`` 形式的命令行覆盖。"""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"无效的覆盖项 '{item}'，应为 KEY=VALUE 格式")
        key, value = item.split("=", 1)
        set_nested(config, key.strip(), cast_value(value.strip()))
    return config


def config_to_text(config: Dict[str, Any]) -> str:
    """把配置渲染成 YAML 文本，用于写入 TensorBoard。"""
    import yaml

    try:
        return yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    except Exception:  # pragma: no cover
        return str(config)



# =============================================================================
# 2. 训练 / 验证 / 调度器
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 0.0,
    step_fn: Optional[Callable[[nn.Module, Any], Tuple[torch.Tensor, int]]] = None,
) -> float:
    """训练一个 epoch，返回平均损失（MSE）。

    单步 BC 的标签为 ``(batch, action_dim)``，动作块 BC 的标签为
    ``(batch, K, action_dim)``；二者共用本函数（``MSELoss`` 逐元素求均值，
    与形状无关），保证两种任务的训练数学严格一致。

    Args:
        step_fn: **可选的自定义单步损失**，签名 ``step_fn(model, batch) -> (loss, n)``
            （``n`` 为 batch 内样本数）。用于损失形式不再是 ``criterion(model(s), a)``
            的任务 —— W4 Task 2 的扩散策略即在此处现场采样扩散步 ``t`` 与噪声 ``ε``，
            计算噪声预测 MSE。``None`` 时走默认的"前向 + criterion"路径（与改造前逐位一致）。
    """
    model.train()
    loss_sum, n_samples = 0.0, 0
    for batch in loader:
        if step_fn is None:
            batch_s, batch_a = batch
            batch_s = batch_s.to(device, non_blocking=True)
            batch_a = batch_a.to(device, non_blocking=True)
            loss = criterion(model(batch_s), batch_a)
            n_batch = int(batch_s.size(0))
        else:
            loss, n_batch = step_fn(model, batch)
            n_batch = int(n_batch)

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_sum += loss.item() * n_batch
        n_samples += n_batch
    return loss_sum / max(1, n_samples)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    step_fn: Optional[Callable[[nn.Module, Any], Tuple[torch.Tensor, int]]] = None,
) -> float:
    """在验证集上评估，返回平均损失（MSE）。

    ``step_fn`` 语义与 :func:`train_one_epoch` 完全相同（扩散策略的验证损失同为
    噪声预测 MSE，同样在现场随机采样 ``t``；样本量足够大时其期望值可与训练损失直接比较）。
    """
    model.eval()
    loss_sum, n_samples = 0.0, 0
    for batch in loader:
        if step_fn is None:
            batch_s, batch_a = batch
            batch_s = batch_s.to(device, non_blocking=True)
            batch_a = batch_a.to(device, non_blocking=True)
            loss = criterion(model(batch_s), batch_a)
            n_batch = int(batch_s.size(0))
        else:
            loss, n_batch = step_fn(model, batch)
            n_batch = int(n_batch)
        loss_sum += loss.item() * n_batch
        n_samples += n_batch
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
# 3. 运行上下文装配
# =============================================================================
@dataclass
class RunContext:
    """一次训练运行的上下文：配置、路径、日志与监控句柄。

    由 :func:`prepare_run` 装配，供 :func:`train_loop` 与实验脚本共用，
    避免每个脚本重复实现"目录 / 日志 / TensorBoard / 续训解析"等样板代码。
    """

    args: argparse.Namespace
    config: Dict[str, Any]
    exp_cfg: Dict[str, Any]
    train_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    aug_cfg: Dict[str, Any]
    eval_cfg: Dict[str, Any]
    paths: Dict[str, Any]
    experiment_name: str
    seed: int
    env_name: str
    device: torch.device
    monitor: str
    monitor_mode: str
    snapshot_every: int
    keep_last_n: int
    resume_target: Any
    resume_path: Optional[str]
    resume_payload: Optional[Dict[str, Any]]
    log_root: str
    ckpt_root: str
    run_dir: str
    checkpoint_dir: str
    logger: logging.Logger
    tb: TensorBoardLogger
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def epochs(self) -> int:
        return int(self.train_cfg.get("epochs", 100))

    @property
    def no_eval(self) -> bool:
        return bool(getattr(self.args, "no_eval", False))

    @property
    def n_eval_episodes(self) -> int:
        return int(self.eval_cfg.get("n_episodes", 20))

    def build_checkpointer(self) -> Checkpointer:
        """构建 ``Checkpointer``：``{name}_best.pt``（最优）/ ``{name}_last.pt``（每 epoch）。"""
        return Checkpointer(
            self.checkpoint_dir,
            filename=f"{self.experiment_name}_best.pt",
            monitor=self.monitor,
            mode=self.monitor_mode,
            snapshot_every=self.snapshot_every,
            keep_last_n=self.keep_last_n,
        )


def prepare_run(
    args: argparse.Namespace,
    config: Dict[str, Any],
    title: str = "行为克隆（Behavior Cloning）实验",
    logger_name: str = "bc",
    extra_info: Optional[Sequence[str]] = None,
) -> RunContext:
    """装配运行上下文：设备 / 种子 / 目录 / 日志 / TensorBoard / 续训目标。

    Args:
        args: 命令行参数（:func:`parse_train_args` 的返回值）。
        config: 已应用命令行覆盖的完整配置字典。
        title: 日志标题（如"行为克隆（Behavior Cloning）实验"）。
        logger_name: logger 名称（同名 logger 复用已有 handler）。
        extra_info: 额外打印的日志行（如"动作块长度 : 8"）。

    Note:
        本函数会调用 ``set_seed(seed)`` 固定全局随机源；续训时会**复用原运行的
        目录**（``run_dir`` / ``checkpoint_dir``），使 TensorBoard 曲线与日志连续。
    """
    train_cfg = config.get("training", {})
    eval_cfg = config.get("eval", {})
    paths = config.get("paths", {})

    # 命令行优先级覆盖
    if args.device:
        exp_cfg = config.setdefault("experiment", {})
        exp_cfg["device"] = args.device
    if args.seed is not None:
        config.setdefault("experiment", {})["seed"] = args.seed

    exp_cfg = config.get("experiment", {})
    experiment_name = str(exp_cfg.get("name", "bc_pendulum"))   # 用于 TensorBoard / Checkpointer 命名
    seed = int(exp_cfg.get("seed", 42))
    env_name = str(exp_cfg.get("env_name", "Pendulum-v1"))
    device = resolve_device(str(exp_cfg.get("device", "auto")))

    # -------- 基础设置 --------
    set_seed(seed)

    # -------- 断点续训：解析 --resume 目标并预读检查点（用于复用原运行目录）--------
    ckpt_cfg = train_cfg.get("checkpoint", {}) or {}
    log_root = str(paths.get("log_dir", "logs"))
    ckpt_root = str(paths.get("checkpoint_dir", os.path.join(log_root, "checkpoints")))
    monitor = str(ckpt_cfg.get("monitor", "val_mse"))
    monitor_mode = str(ckpt_cfg.get("mode", "min"))
    snapshot_every = (
        args.snapshot_every if getattr(args, "snapshot_every", None) is not None
        else int(ckpt_cfg.get("snapshot_every", 0))
    )
    keep_last_n = int(ckpt_cfg.get("keep_last_n", 2))

    resume_target = args.resume if getattr(args, "resume", None) is not None else paths.get("resume")
    resume_path = Checkpointer.resolve_path(resume_target, checkpoint_dir=ckpt_root)
    resume_payload = load_checkpoint_file(resume_path, map_location="cpu") if resume_path else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_run_dir = ((resume_payload or {}).get("extra") or {}).get("run_dir")
    if resume_payload is not None and saved_run_dir and os.path.isdir(saved_run_dir):
        # 续训：复用原运行的 TensorBoard 目录，曲线 / 日志保持连续
        run_dir = str(saved_run_dir)
        checkpoint_dir = os.path.dirname(resume_path)
    else:
        run_dir = os.path.join(log_root, f"{experiment_name}_{timestamp}")           # TensorBoard 目录含实验名
        checkpoint_dir = os.path.join(ckpt_root, f"{experiment_name}_{timestamp}")   # Checkpointer 目录含实验名
    os.makedirs(run_dir, exist_ok=True)

    logger = get_logger(logger_name, log_file=os.path.join(run_dir, "train.log"))
    tb = TensorBoardLogger(run_dir, enabled=not getattr(args, "no_tb", False))

    logger.info("=" * 60)
    logger.info("%s：%s", title, env_name)
    logger.info("=" * 60)
    logger.info("配置文件   : %s", args.config)
    logger.info("实验名称   : %s", experiment_name)
    logger.info("运行目录   : %s", run_dir)
    logger.info("检查点目录 : %s", checkpoint_dir)
    logger.info("设备       : %s", device)
    logger.info("随机种子   : %d", seed)
    logger.info("网络结构   : %s", config.get("model", {}).get("name", "mlp"))
    for line in (extra_info or []):
        logger.info("%s", line)
    if resume_path:
        logger.info("断点续训   : %s", resume_path)
    tb.add_text("config", "```yaml\n" + config_to_text(config) + "\n```", 0)

    return RunContext(
        args=args, config=config,
        exp_cfg=exp_cfg, train_cfg=train_cfg,
        data_cfg=config.get("data", {}), aug_cfg=config.get("augmentation", {}),
        eval_cfg=eval_cfg, paths=paths,
        experiment_name=experiment_name, seed=seed, env_name=env_name, device=device,
        monitor=monitor, monitor_mode=monitor_mode,
        snapshot_every=snapshot_every, keep_last_n=keep_last_n,
        resume_target=resume_target, resume_path=resume_path, resume_payload=resume_payload,
        log_root=log_root, ckpt_root=ckpt_root, run_dir=run_dir, checkpoint_dir=checkpoint_dir,
        logger=logger, tb=tb,
    )


# =============================================================================
# 4. 训练循环（优雅停机 + 应急存盘）
# =============================================================================
def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpointer: Checkpointer,
    ctx: RunContext,
    start_epoch: int = 1,
    resume_hint: Optional[str] = None,
    loss_label: str = "MSE",
    train_step_fn: Optional[Callable[[nn.Module, Any], Tuple[torch.Tensor, int]]] = None,
    val_step_fn: Optional[Callable[[nn.Module, Any], Tuple[torch.Tensor, int]]] = None,
) -> Dict[str, Any]:
    """执行 epoch 训练循环，返回本轮训练统计。

    工程语义（与 W3 单步 BC 完全一致）：

    * 每个 epoch 都写 ``last`` 检查点（便于异常恢复），刷新最优时再写 ``best``；
    * 首次收到 ``SIGINT`` / ``SIGTERM``：当前 epoch 结束后存盘并安全退出，打印续训命令；
      再次收到则强制退出（不丢已有检查点）；
    * 训练中抛异常时先写应急检查点再向上抛出，最大限度保留已有成果。

    Returns:
        含 ``last_epoch`` / ``best_metric`` / ``last_val_metric`` / ``interrupted``
        / ``train_time`` / ``epochs`` 的统计字典。

    Args:
        train_step_fn / val_step_fn: 可选的自定义单步损失（``step_fn(model, batch)
            -> (loss, n)``），透传给 :func:`train_one_epoch` / :func:`validate`。
            ``None`` 时与"前向 + criterion"的经典 BC 训练**逐位一致**；扩散策略
            （W4 Task 2）用它替换成"现场采样 ``t`` / ``ε`` 的噪声预测 MSE"。
    """
    logger, tb = ctx.logger, ctx.tb
    epochs = ctx.epochs
    log_interval = int(ctx.train_cfg.get("log_interval", 20))
    max_grad_norm = float(ctx.train_cfg.get("max_grad_norm", 0.0))

    if start_epoch > epochs:
        logger.warning("[2/3] 检查点已完成 %d 个 epoch（目标 %d），无需继续训练", start_epoch - 1, epochs)
    else:
        logger.info("[2/3] 开始训练：epoch %d → %d", start_epoch, epochs)

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
        except (ValueError, OSError, AttributeError):  # pragma: no cover - 非主线程等场景
            pass

    last_epoch = start_epoch - 1
    last_val_metric: Optional[float] = None
    best_metric = checkpointer.best_metric
    interrupted = False
    start_time = time.time()
    hint = resume_hint or f"--config {ctx.args.config} --resume {checkpointer.last_path}"

    try:
        for epoch in range(start_epoch, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, ctx.device, max_grad_norm,
                step_fn=train_step_fn,
            )
            val_loss = validate(model, val_loader, criterion, ctx.device, step_fn=val_step_fn)

            if scheduler is not None:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            save_info = checkpointer.step(
                model, optimizer, scheduler,
                epoch=epoch, metric=val_loss, config=ctx.config,
                extras={"run_dir": ctx.run_dir, "seed": ctx.seed},
            )
            best_metric = save_info["best_metric"]
            last_epoch, last_val_metric = epoch, val_loss

            tb.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            tb.add_scalar("lr", lr, epoch)
            tb.add_scalar(f"best_{ctx.monitor}", best_metric, epoch)

            if epoch % log_interval == 0 or epoch == start_epoch or epoch == epochs or save_info["is_best"]:
                logger.info(
                    "Epoch [%d/%d] | Train %s: %.6f | Val %s: %.6f | Best: %.6f | LR: %.2e%s",
                    epoch, epochs, loss_label, train_loss, loss_label, val_loss, best_metric, lr,
                    "  *best" if save_info["is_best"] else "",
                )

            if stop_requested["flag"]:
                interrupted = True
                logger.warning("已保存检查点并于 epoch %d 安全退出；继续训练可执行：\n  %s", epoch, hint)
                break
    except BaseException as exc:
        logger.error("训练过程中出现异常 (%s: %s)，正在保存应急检查点 ...", type(exc).__name__, exc)
        try:
            checkpointer.save_last(
                model, optimizer, scheduler, epoch=last_epoch, metric=last_val_metric,
                config=ctx.config,
                extras={"run_dir": ctx.run_dir, "seed": ctx.seed, "reason": f"crash:{type(exc).__name__}"},
            )
            logger.error("应急检查点已保存: %s", checkpointer.last_path)
        except Exception as save_exc:  # pragma: no cover
            logger.error("应急检查点保存失败: %s", save_exc)
        raise
    finally:
        tb.flush()

    return {
        "last_epoch": last_epoch,
        "best_metric": best_metric,
        "last_val_metric": last_val_metric,
        "interrupted": interrupted,
        "train_time": time.time() - start_time,
        "epochs": epochs,
    }


# =============================================================================
# 5. 续训恢复 / 结果汇总
# =============================================================================
def restore_from_checkpoint(
    ctx: RunContext,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    checkpointer: Checkpointer,
) -> int:
    """断点续训：恢复模型 / 优化器 / 调度器 / epoch / 最优指标 / 随机源。

    随机源也一并恢复（``restore_rng=True``），因此"中断→续训"与"不中断"的
    训练轨迹完全一致，便于复现。

    Returns:
        ``start_epoch``：下一个待训练 epoch（从头训练时为 1）。
    """
    if ctx.resume_payload is None:
        ctx.logger.info("从头开始训练（未指定 --resume）")
        return 1

    ckpt = Checkpointer.load(
        path=ctx.resume_path,
        model=model,
        map_location=ctx.device,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,             # 恢复随机源 → 与"不中断"的训练轨迹完全一致
        payload=ctx.resume_payload,   # 复用已读出的内容，避免重复读盘
    )
    start_epoch = int(ckpt.get("epoch") or 0) + 1
    prev_best = ckpt.get("best_val_loss", ckpt.get("metric"))
    if prev_best is not None:
        checkpointer.best_metric = float(prev_best)
    checkpointer.best_epoch = ckpt.get("best_epoch")
    ctx.logger.info(
        "已恢复检查点 | 已完成 epoch: %s | 历史最优 %s: %.6f @ epoch %s | 将从 epoch %d 继续",
        ckpt.get("epoch"), ctx.monitor, checkpointer.best_metric,
        checkpointer.best_epoch, start_epoch,
    )
    ctx.tb.add_text(
        "resume",
        f"resumed from `{ctx.resume_path}`\n\n"
        f"- finished_epoch: {ckpt.get('epoch')}\n"
        f"- best_{ctx.monitor}: {checkpointer.best_metric:.6f} @ epoch {checkpointer.best_epoch}\n"
        f"- checkpoint_created_at: {ckpt.get('created_at')}\n",
        start_epoch,
    )
    return start_epoch


def report_closed_loop(
    ctx: RunContext,
    result: Dict[str, Any],
    tag: str = "eval/mean_return",
    label: str = "闭环评估",
    step: int = 0,
    extra_tags: Optional[Dict[str, float]] = None,
) -> None:
    """把闭环评估结果写入 TensorBoard 并打印统一格式日志。"""
    ctx.tb.add_scalar(tag, float(result["mean_return"]), step)
    for name, value in (extra_tags or {}).items():
        ctx.tb.add_scalar(name, float(value), step)
    ctx.logger.info(
        "%s | 回合数: %d | 平均回报: %.2f ± %.2f (min %.2f, max %.2f)",
        label, int(result.get("n_episodes", 0)),
        result["mean_return"], result["std_return"],
        result["min_return"], result["max_return"],
    )


def finish_run(
    ctx: RunContext,
    checkpointer: Checkpointer,
    stats: Dict[str, Any],
    eval_result: Optional[Dict[str, Any]] = None,
    extra_lines: Optional[Sequence[str]] = None,
    resume_command: Optional[str] = None,
    close_tb: bool = True,
    metric_label: str = "MSE",
) -> None:
    """打印最终结果汇总并落盘 / 关闭 TensorBoard。

    Args:
        metric_label: 被监控指标的名称（默认 ``MSE``；扩散策略传 ``noise-MSE``）。
    """
    logger = ctx.logger
    logger.info("=" * 60)
    logger.info("最终实验结果")
    logger.info("=" * 60)
    logger.info("最佳验证集 %-8s: %.6f (epoch %s)", metric_label, stats["best_metric"],
                checkpointer.best_epoch)
    logger.info("已完成 epoch   : %d", stats["last_epoch"])
    if eval_result is not None:
        logger.info("闭环平均回报   : %.2f", eval_result["mean_return"])
    for line in (extra_lines or []):
        logger.info("%s", line)
    logger.info("TensorBoard    : tensorboard --logdir %s", ctx.log_root)
    logger.info("最优权重       : %s", checkpointer.best_path)
    logger.info("最近权重       : %s", checkpointer.last_path)
    if stats.get("interrupted") and resume_command:
        logger.info("断点续训       : %s", resume_command)

    ctx.tb.flush()
    if close_tb:
        ctx.tb.close()




