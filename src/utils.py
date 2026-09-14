"""工程工具模块：配置解析、日志、随机种子、TensorBoard 监控与模型检查点。"""

from __future__ import annotations

import glob
import hashlib
import io
import logging
import os
import random
import struct
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


# =============================================================================
# 1. 配置解析
# =============================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 加载配置，支持 ``defaults`` 继承其它配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "defaults" in config:
        base_path = config.pop("defaults")
        base_path = os.path.join(os.path.dirname(config_path), base_path)
        base = load_config(base_path)
        config = deep_merge(base, config)

    return config


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，``override`` 优先。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_device(device_cfg: str = "auto") -> torch.device:
    """解析设备配置：auto 优先 CUDA。"""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def set_seed(seed: int) -> None:
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 允许 cuDNN 基准测试加速，但关闭非确定性算法以可复现优先
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =============================================================================
# 1.5 随机源状态（断点续训做"比特级无缝衔接"的关键）
# =============================================================================
def capture_rng_state() -> Dict[str, Any]:
    """抓取所有随机源状态：python / numpy / torch(CPU) / torch(CUDA)。"""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """把任意形式的 RNG 状态规范化为 **CPU** 上的 uint8 ByteTensor。

    ``torch.set_rng_state`` / ``torch.cuda.set_rng_state`` 都要求状态张量是
    CPU ByteTensor（CUDA 默认生成器本身是 CPU 侧对象，传入 CUDA 张量会被
    拒绝），因此这里统一转成 CPU 上的 uint8 并拷贝一份。
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
    else:
        tensor = torch.frombuffer(bytearray(value), dtype=torch.uint8)
    return tensor.to(device="cpu", dtype=torch.uint8).clone()


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """把 ``capture_rng_state`` 抓取的状态原样写回随机源。

    恢复后，后续的 shuffle / 数据增强噪声 / 权重初始化都会与中断前
    完全一致，从而做到"比特级无缝衔接"。
    """
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch.cuda.is_available():
        count = min(len(cuda_states), torch.cuda.device_count())
        restored = [_as_cpu_byte_tensor(s) for s in cuda_states[:count]]
        if restored:
            torch.cuda.set_rng_state_all(restored)


# =============================================================================
# 2. 日志
# =============================================================================
def get_logger(name: str = "bc", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """创建同时输出到控制台（和可选文件）的 logger。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # 避免重复添加 handler
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# 3. TensorBoard 监控
# =============================================================================
class TensorBoardLogger:
    """封装 SummaryWriter，提供标量 / 直方图 / 文本记录，并优雅降级。"""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.log_dir = log_dir
        self.enabled = enabled
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self.writer = SummaryWriter(log_dir=log_dir)
            except Exception as exc:  # pragma: no cover
                self.enabled = False
                print(f"[tb] TensorBoard 不可用，已禁用: {exc}")

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_scalars(main_tag, values, step)

    def add_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.enabled and self.writer is not None:
            self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        if self.enabled and self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# =============================================================================
# 4. 模型检查点（工业级：原子写入 + 完整性校验 + 完整训练状态 + 断点续训）
# =============================================================================
CHECKPOINT_FORMAT = "bc-checkpoint"
CHECKPOINT_VERSION = 2

_MAGIC = b"BCKPT1\n\x00"      # 8 字节魔数，用于识别自定义容器格式
_DIGEST_LEN = 64              # sha256 十六进制摘要长度
_LEN_FMT = "<Q"               # payload 长度（小端 uint64）


class CheckpointIntegrityError(RuntimeError):
    """检查点文件损坏、被截断或摘要校验失败。"""


def _torch_load(source: Any, map_location: Any = "cpu") -> Any:
    """跨版本兼容的 ``torch.load``。

    必须显式 ``weights_only=False``：检查点里除张量外还保存了 RNG 状态
    （python tuple / numpy 数组）与配置字典。
    """
    try:
        return torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - torch < 1.13 无 weights_only 参数
        return torch.load(source, map_location=map_location)


def _to_bytes(payload: Dict[str, Any]) -> bytes:
    """把 payload 序列化成字节流。"""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def atomic_save(payload: Dict[str, Any], path: str) -> str:
    """把 ``payload`` **原子地**写入 ``path``，并在尾部附加 sha256 摘要。

    写入流程：临时文件 → fsync → ``os.replace`` 原子替换 → 目录 fsync。
    任何时刻断电 / 被 ``kill -9`` 都只会留下"旧的完整文件"或"新的完整文件"，
    绝不会出现半截损坏的检查点。

    Returns:
        写入的文件路径。
    """
    raw = _to_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest().encode("ascii")

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_ckpt_", suffix=".part", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_MAGIC)
            handle.write(struct.pack(_LEN_FMT, len(raw)))
            handle.write(raw)
            handle.write(digest)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)          # POSIX 原子替换
        tmp_path = None
        try:                                # 让 rename 本身也落盘
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:  # pragma: no cover - 某些文件系统不支持对目录 fsync
            pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return path


def load_checkpoint_file(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """读取检查点文件，校验完整性后返回 payload。

    * 自定义容器格式：校验 magic / 长度 / sha256 摘要，损坏则抛
      :class:`CheckpointIntegrityError`；
    * 旧格式（裸 ``torch.save`` 文件）：自动兼容，直接反序列化。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"检查点文件不存在: {path}")

    with open(path, "rb") as handle:
        raw = handle.read()

    if not raw.startswith(_MAGIC):                  # 兼容历史裸 torch.save 文件
        return _torch_load(io.BytesIO(raw), map_location=map_location)

    header_end = len(_MAGIC) + struct.calcsize(_LEN_FMT)
    if len(raw) < header_end + _DIGEST_LEN:
        raise CheckpointIntegrityError(f"检查点文件被截断: {path}")
    (length,) = struct.unpack(_LEN_FMT, raw[len(_MAGIC):header_end])

    body = raw[header_end:header_end + length]
    stored = raw[header_end + length:header_end + length + _DIGEST_LEN]
    if len(body) != length or len(stored) != _DIGEST_LEN:
        raise CheckpointIntegrityError(f"检查点文件被截断: {path}")

    actual = hashlib.sha256(body).hexdigest().encode("ascii")
    if actual != stored:
        raise CheckpointIntegrityError(
            f"检查点校验失败（文件已损坏）: {path}\n"
            f"  expected sha256 = {stored.decode()}\n"
            f"  actual   sha256 = {actual.decode()}"
        )
    return _torch_load(io.BytesIO(body), map_location=map_location)


class Checkpointer:
    """工业级检查点管理器（支持断点续训与异常恢复）。

    特性:
      * **原子写入**：临时文件 + fsync + ``os.replace``，杜绝半截文件；
      * **完整性校验**：文件尾部附 sha256 摘要，加载时校验；
      * **完整训练状态**：``model`` / ``optimizer`` / ``scheduler`` /
        ``epoch`` / ``best_val_loss`` / **RNG 状态**，可做比特级续训；
      * **双轨保存**：``{name}_best.pt``（历史最优）与 ``{name}_last.pt``
        （每个 epoch 覆盖，用于中断后续训）；
      * **快照轮转**：``snapshot_every`` 每 N 个 epoch 归档一份，只保留
        ``keep_last_n`` 份。
    """

    def __init__(
        self,
        checkpoint_dir: str,
        name: str = "checkpoint",
        filename: Optional[str] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        snapshot_every: int = 0,
        keep_last_n: int = 2,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        if filename:                       # 向后兼容旧接口：filename 即 best 文件名
            self.best_filename = filename
            stem = os.path.splitext(filename)[0]
            if stem.endswith("_best"):
                stem = stem[: -len("_best")]
            self.name = stem or name
        else:
            self.name = name
            self.best_filename = f"{name}_best.pt"
        self.last_filename = f"{self.name}_last.pt"
        self.best_path = os.path.join(checkpoint_dir, self.best_filename)
        self.last_path = os.path.join(checkpoint_dir, self.last_filename)

        self.monitor = monitor
        self.mode = mode if mode in ("min", "max") else "min"
        self.snapshot_every = max(0, int(snapshot_every))
        self.keep_last_n = max(0, int(keep_last_n))
        sentinel = float("inf") if self.mode == "min" else -float("inf")
        self.best_metric = sentinel
        self.best_epoch: Optional[int] = None
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    # 内部工具
    # ---------------------------------------------------------------- #
    def _is_improvement(self, metric: Optional[float]) -> bool:
        """判断 ``metric`` 是否刷新了历史最优。"""
        if metric is None:
            return False
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def summary(self) -> str:
        """可读的检查点状态摘要，便于日志输出。"""
        return (
            f"Checkpointer(name={self.name}, monitor={self.monitor}({self.mode}), "
            f"best={self.best_metric:.6f}@{self.best_epoch}, dir={self.checkpoint_dir})"
        )

    def _build_payload(self, model, optimizer=None, scheduler=None, epoch=None,
                       metric=None, config=None, extras=None) -> Dict[str, Any]:
        """组装包含"完整训练状态 + 随机源状态"的 payload。"""
        payload: Dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "name": self.name,
            "epoch": epoch,                     # 已完成（finished）的最后一个 epoch
            "metric": metric,                   # 本次评估指标
            "monitor": self.monitor,
            "best_val_loss": self.best_metric,  # 历史最优，续训判优的基准
            "best_epoch": self.best_epoch,
            "model_state_dict": model.state_dict(),
            "rng_state": capture_rng_state(),   # 比特级无缝衔接的关键
            "config": config,
            "extra": dict(extras or {}),
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        return payload

    # ---------------------------------------------------------------- #
    # 保存
    # ---------------------------------------------------------------- #
    def save(self, model, optimizer=None, scheduler=None, epoch=None, metric=None,
             config=None, extras=None) -> str:
        """保存"最优"检查点 ``{name}_best.pt``，并更新内部最优记录。"""
        if metric is not None:
            self.best_metric = metric
            self.best_epoch = epoch
        payload = self._build_payload(model, optimizer, scheduler, epoch, metric, config, extras)
        return atomic_save(payload, self.best_path)

    def save_best(self, *args, **kwargs) -> str:
        """``save`` 的语义化别名。"""
        return self.save(*args, **kwargs)

    def save_last(self, model, optimizer=None, scheduler=None, epoch=None, metric=None,
                  config=None, extras=None) -> str:
        """保存"最后一次"检查点 ``{name}_last.pt``（断点续训 / 崩溃恢复用）。"""
        merged = dict(extras or {})
        merged.setdefault("kind", "last")
        payload = self._build_payload(model, optimizer, scheduler, epoch, metric, config, merged)
        path = atomic_save(payload, self.last_path)
        if self.snapshot_every and epoch is not None and int(epoch) % self.snapshot_every == 0:
            self._archive_snapshot(payload, int(epoch))
        return path

    def step(self, model, optimizer=None, scheduler=None, epoch=None, metric=None,
             config=None, extras=None) -> Dict[str, Any]:
        """每个 epoch 调用一次：总是写 ``last``，指标更优时再写 ``best``。

        Returns:
            ``{"last": 路径, "best": 路径或 None, "is_best": bool, "best_metric": 最优值}``
        """
        is_best = self._is_improvement(metric)
        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
        merged = dict(extras or {})
        merged.update({"is_best": is_best, "best_val_loss": self.best_metric})

        last_path = self.save_last(model, optimizer, scheduler, epoch, metric, config, merged)
        best_path = None
        if is_best:
            best_path = self.save(model, optimizer, scheduler, epoch, metric, config,
                                  dict(merged, kind="best"))
        return {"last": last_path, "best": best_path, "is_best": is_best, "best_metric": self.best_metric}

    def _archive_snapshot(self, payload: Dict[str, Any], epoch: int) -> str:
        """按 ``snapshot_every`` 归档快照，只保留最新 ``keep_last_n`` 份。"""
        path = os.path.join(self.checkpoint_dir, f"{self.name}_epoch{int(epoch):04d}.pt")
        atomic_save(payload, path)
        snapshots = sorted(glob.glob(os.path.join(self.checkpoint_dir, f"{self.name}_epoch*.pt")))
        while len(snapshots) > self.keep_last_n:
            os.remove(snapshots.pop(0))
        return path

    def maybe_save_best(self, model, metric, optimizer=None, scheduler=None,
                        epoch=None, config=None) -> bool:
        """兼容旧接口：指标更优时保存 best，返回是否已保存。"""
        if self._is_improvement(metric):
            self.save(model, optimizer, scheduler, epoch, metric, config)
            return True
        return False

    # ---------------------------------------------------------------- #
    # 查找 / 解析检查点路径
    # ---------------------------------------------------------------- #
    @staticmethod
    def find_latest(root: str, name: Optional[str] = None) -> Optional[str]:
        """在 ``root``（可递归）中查找最新的检查点文件。

        优先级：``*_last.pt`` > 其它（``*_best.pt`` / 快照），同级按修改时间取最新。
        """
        if os.path.isfile(root):
            return root
        if not os.path.isdir(root):
            return None

        candidates: List[str] = []
        for dirpath, _dirs, files in os.walk(root):
            for filename in files:
                if not filename.endswith((".pt", ".pth")):
                    continue
                if name and not filename.startswith(f"{name}_"):
                    continue
                candidates.append(os.path.join(dirpath, filename))
        if not candidates:
            return None

        def sort_key(path: str) -> Tuple[int, float]:
            base = os.path.basename(path)
            priority = 0 if base.endswith(("_last.pt", "_last.pth")) else 1
            return (priority, -os.path.getmtime(path))     # 优先级升序、时间降序

        return min(candidates, key=sort_key)

    @staticmethod
    def resolve_path(target: Optional[str], checkpoint_dir: Optional[str] = None) -> Optional[str]:
        """把 ``--resume`` 的取值解析为具体文件路径。

        支持三种形式：
        * 具体文件：``logs/checkpoints/xxx/bc_pendulum_best.pt``；
        * 目录：``logs/checkpoints``（在其下递归查找最新检查点）；
        * ``auto`` / ``latest`` / ``true``：在 ``checkpoint_dir`` 下自动查找。
        """
        if target is None:
            return None
        target = str(target).strip()
        if target.lower() in ("", "none", "null", "false", "~"):
            return None

        if target.lower() in ("auto", "latest", "true"):
            root = checkpoint_dir or os.path.join("logs", "checkpoints")
            found = Checkpointer.find_latest(root)
            if found is None:
                raise FileNotFoundError(f"--resume {target}: 在 '{root}' 下未找到任何检查点")
            return found

        if os.path.isfile(target):
            return target
        if os.path.isdir(target):
            found = Checkpointer.find_latest(target)
            if found is None:
                raise FileNotFoundError(f"--resume: 目录 '{target}' 下没有检查点文件")
            return found
        for suffix in (".pt", ".pth"):
            if os.path.isfile(target + suffix):
                return target + suffix
        raise FileNotFoundError(f"--resume: 找不到检查点 '{target}'（可为文件路径、目录，或 auto）")

    # ---------------------------------------------------------------- #
    # 加载
    # ---------------------------------------------------------------- #
    @staticmethod
    def load(path: Optional[str] = None, model=None, map_location: Any = "cpu",
             optimizer=None, scheduler=None, strict: bool = True,
             restore_rng: bool = False, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """加载检查点，按需恢复到 ``model`` / ``optimizer`` / ``scheduler`` / 随机源。

        Args:
            path: 检查点文件路径（若已给定 ``payload`` 则可省略，避免重复读盘）。
            model: 目标网络，加载 ``model_state_dict``。
            map_location: 张量重映射设备。
            optimizer / scheduler: 给定则恢复其状态（断点续训必需）。
            strict: 传给 ``load_state_dict`` 的 strict 参数。
            restore_rng: 是否恢复随机源状态（比特级无缝续训必须为 True）。
            payload: 已读取的检查点内容。

        Returns:
            检查点 payload（含 ``epoch`` / ``best_val_loss`` / ``rng_state`` 等）。
        """
        if payload is None:
            if path is None:
                raise ValueError("必须提供 path 或 payload 之一")
            payload = load_checkpoint_file(path, map_location=map_location)

        if model is not None:
            state = payload.get("model_state_dict", payload)
            model.load_state_dict(state, strict=strict)
        if optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        if scheduler is not None and payload.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if restore_rng:
            restore_rng_state(payload.get("rng_state"))
        return payload
