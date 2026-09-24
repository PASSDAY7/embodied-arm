#!/usr/bin/env python
"""TensorBoard 曲线导出工具（W8 Task 1 验收：导出 TensorBoard 曲线）。

把一次运行的 ``logs/<exp>_<timestamp>/events.out.tfevents.*`` 读出来，生成：

* ``<out>/<prefix>_curves.png`` —— 多面板训练曲线（学习曲线 / 损失 / 熵与 σ / KL 与裁剪 /
  梯度范数 / 价值解释度 / 动作截断比例 …），横轴统一为**累计交互步数**；
* ``<out>/<prefix>_scalars.csv`` —— 全部标量的长表 ``tag, step, value``（可直接进 Excel / pandas）；
* 终端打印每个关键 tag 的"首个 / 最后 / 最优"值摘要表。

用法::

    python experiments/export_tb_curves.py --logdir logs/ppo_pendulum_20260918_142705
    python experiments/export_tb_curves.py --logdir logs --out-dir docs/figures --prefix w8_ppo
    python experiments/export_tb_curves.py --logdir logs/<run> --all       # 每个 tag 一个子图

依赖：``tensorboard``（EventAccumulator 读事件文件）+ ``matplotlib``（``Agg`` 后端渲染 PNG，
无需显示器）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 每个"面板" = 标题 + 若干 (tag, 图例) + 可选参考线
PANEL_SPECS: Tuple[Tuple[str, Tuple[Tuple[str, str], ...], Optional[float]], ...] = (
    ("学习曲线（回合回报）", (
        ("train/episode_return", "训练回合回报"),
        ("train/rolling_return", "滑动平均（100 回合）"),
    ), None),
    ("独立评估回报", (
        ("eval/mean_return", "评估平均回报"),
        ("eval/final_mean_return", "最终评估"),
    ), -200.0),
    ("策略损失 L^CLIP（mini-batch 均值）", (("train/policy_loss", "L^CLIP"),), None),
    ("价值损失 L^VF", (("train/value_loss", "L^VF"),), None),
    ("熵与动作标准差", (
        ("train/entropy", "熵 H"),
        ("train/std_mean", "σ = exp(log_std)"),
    ), None),
    ("KL 与裁剪比例（信任域诊断）", (
        ("train/approx_kl", "approx_kl (k3)"),
        ("train/clip_fraction", "clip 占比"),
    ), None),
    ("梯度范数（裁剪前）", (("train/grad_norm", "grad_norm"),), None),
    ("价值解释度与单步奖励", (
        ("train/explained_variance", "explained_variance"),
        ("buffer/reward_mean", "单步奖励均值"),
    ), None),
    ("动作截断比例（思考题诊断）", (("train/action_clip_fraction", "a_raw 越界比例"),), None),
    # ---- W9 Task 1: SAC（off-policy 最大熵 RL）的监控面板 ----
    ("SAC：双 Q 损失（TD 误差）", (
        ("train/q1_loss", "Q1 损失"),
        ("train/q2_loss", "Q2 损失"),
    ), None),
    ("SAC：策略损失 J_π = E[α logπ − min Q]", (("train/actor_loss", "J_π"),), None),
    ("SAC：温度 α 的自适应轨迹", (
        ("train/alpha", "α = exp(log α)"),
        ("eval/alpha", "评估点 α"),
        ("train/log_alpha", "log α"),
    ), None),
    ("SAC：温度损失 J(α)", (("train/alpha_loss", "J(α)"),), None),
    ("SAC：熵与目标熵（H̄ = −dim A）", (
        ("train/entropy_mean", "策略熵 H ≈ −E[log π]"),
        ("eval/entropy_mean", "评估点熵"),
        ("train/entropy_target_gap", "H − |H̄|"),
    ), None),
    ("SAC：软 Bellman 目标与 Q 尺度", (
        ("train/target_q_mean", "y = r + γ(minQ̄ − α logπ)"),
        ("train/next_q_mean", "min Q̄(s', a')"),
        ("train/td_abs_mean", "|Q − y|"),
    ), None),
    ("SAC：三段梯度范数（c/a/α）", (
        ("train/critic_grad_norm", "Critic"),
        ("train/actor_grad_norm", "Actor"),
        ("train/alpha_grad_norm", "α"),
    ), None),
    ("SAC：思考题诊断（Actor 步冻结 Critic）", (
        ("train/critic_frozen_ok", "critic_frozen_ok（应恒为 1）"),
    ), None),
    ("SAC：动作分布紧凑度", (
        ("train/mean_std", "σ（压缩前空间）"),
        ("train/saturation_fraction", "tanh 饱和比例"),
        ("train/mean_mu_abs", "|μ|"),
    ), None),
    ("SAC：回放池（off-policy 数据）", (
        ("buffer/size", "样本数"),
        ("buffer/reward_mean", "单步奖励均值"),
        ("buffer/truncated_fraction", "truncated 比例"),
    ), None),
)


def resolve_run_dir(logdir: str) -> str:
    """把 ``--logdir`` 解析为具体运行目录（支持通配符 / 父目录 / 直接给运行目录）。"""
    def _has_events(path: str) -> bool:
        return bool(glob.glob(os.path.join(path, "events.out.tfevents.*")))

    if os.path.isdir(logdir) and _has_events(logdir):
        return logdir
    candidates = [
        d for d in glob.glob(os.path.join(logdir, "*"))
        if os.path.isdir(d) and (_has_events(d) or any(
            _has_events(sub) for sub in glob.glob(os.path.join(d, "*")) if os.path.isdir(sub)
        ))
    ]
    if not candidates:
        raise FileNotFoundError(f"在 '{logdir}' 下没找到 TensorBoard 事件文件（events.out.tfevents.*）")
    if len(candidates) > 1:
        candidates.sort(key=os.path.getmtime)
        print(f"[warn] 找到 {len(candidates)} 个运行目录，取最新的: {os.path.basename(candidates[-1])}")
    return candidates[-1]

def load_scalars(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """读取目录下**所有**事件文件的标量，返回 ``{tag: [(step, value), ...]}``（按 step 排序）。"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 tensorboard 才能读取事件文件（pip install tensorboard）") from exc

    merged: Dict[str, Dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})     # 0 = 不限制保留点数
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            bucket = merged.setdefault(tag, {})
            for event in acc.Scalars(tag):
                bucket[int(event.step)] = float(event.value)           # 同 step 后写覆盖先写
    return {tag: sorted(bucket.items()) for tag, bucket in merged.items()}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """滑动平均（``window<=1`` 或数据不足时原样返回）。"""
    if window is None or window <= 1 or values.size < window:
        return values
    kernel = np.ones(int(window)) / float(window)
    return np.convolve(values, kernel, mode="valid")


def plot_curves(
    scalars: Dict[str, List[Tuple[int, float]]],
    out_path: str,
    variant: str = "ppo",
    smooth_window: int = 1,
    all_tags: bool = False,
) -> List[str]:
    """绘制多面板曲线图并保存为 PNG，返回实际画出的 tag 列表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ([(tag, ((tag, tag),), None) for tag in sorted(scalars)] if all_tags else list(PANEL_SPECS))
    panels = [p for p in panels if any(tag in scalars for tag, _ in p[1])]
    if not panels:
        raise ValueError("没有任何可绘制的标量（与预期的 tag 不匹配）")

    ncols = 3 if len(panels) > 4 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.1 * nrows), squeeze=False)
    drawn: List[str] = []
    for idx, (title, series, ref) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        for tag, label in series:
            if tag not in scalars:
                continue
            steps = np.asarray([s for s, _ in scalars[tag]], dtype=np.float64)
            values = np.asarray([v for _, v in scalars[tag]], dtype=np.float64)
            if smooth_window > 1 and values.size > smooth_window:
                ax.plot(steps[smooth_window - 1:], _smooth(values, smooth_window), lw=1.6, label=label)
            else:
                ax.plot(steps, values, lw=1.3, label=label)
            drawn.append(tag)
        if ref is not None:
            ax.axhline(ref, color="crimson", ls="--", lw=1.0, label=f"目标 {ref:g}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("累计交互步数 (timesteps)", fontsize=8)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle(f"{variant}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return drawn

def write_csv(scalars: Dict[str, List[Tuple[int, float]]], out_path: str) -> int:
    """把全部标量写成长表 ``tag, step, value``，返回写入行数。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "step", "value"])
        for tag in sorted(scalars):
            for step, value in scalars[tag]:
                writer.writerow([tag, step, f"{value:.10g}"])
                rows += 1
    return rows


def summarize(scalars: Dict[str, List[Tuple[int, float]]], keys: Sequence[str]) -> List[str]:
    """生成关键标量的"首值 / 末值 / 最优值"摘要行（终端表格）。"""
    lines = [f"{'tag':<34}{'首个':>12}{'最后':>12}{'最优(max)':>12}{'步数':>10}"]
    for tag in keys:
        if tag not in scalars:
            continue
        steps = [s for s, _ in scalars[tag]]
        values = [v for _, v in scalars[tag]]
        lines.append(f"{tag:<34}{values[0]:>12.3f}{values[-1]:>12.3f}{max(values):>12.3f}{steps[-1]:>10d}")
    return lines


def export_curves(
    logdir: str,
    out_dir: str = "curves",
    prefix: Optional[str] = None,
    smooth_window: int = 1,
    all_tags: bool = False,
) -> Dict[str, Any]:
    """一站式导出：解析目录 → 读标量 → 画 PNG → 写 CSV，返回产物路径与统计。"""
    run_dir = resolve_run_dir(logdir)
    prefix = prefix or os.path.basename(run_dir.rstrip("/"))
    scalars = load_scalars(run_dir)
    png_path = os.path.join(out_dir, f"{prefix}_curves.png")
    csv_path = os.path.join(out_dir, f"{prefix}_scalars.csv")
    tags = plot_curves(scalars, png_path, variant=prefix, smooth_window=smooth_window, all_tags=all_tags)
    rows = write_csv(scalars, csv_path)
    return {
        "run_dir": run_dir, "png": png_path, "csv": csv_path,
        "n_tags": len(scalars), "n_rows": rows, "plotted": tags, "scalars": scalars,
    }


DEFAULT_SUMMARY_TAGS: Tuple[str, ...] = (
    "train/episode_return", "train/rolling_return", "eval/mean_return", "eval/final_mean_return",
    "train/policy_loss", "train/value_loss", "train/approx_kl", "train/clip_fraction",
    "train/grad_norm", "train/std_mean", "train/action_clip_fraction", "buffer/reward_mean",
    "train/explained_variance",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 TensorBoard 曲线导出为 PNG + CSV")
    parser.add_argument("--logdir", type=str, default="logs",
                        help="运行目录 / 父目录 / 通配符（默认 logs，自动取最新的运行）")
    parser.add_argument("--out-dir", type=str, default="curves", help="输出目录（默认 curves）")
    parser.add_argument("--prefix", type=str, default=None, help="输出文件名前缀（默认取运行目录名）")
    parser.add_argument("--smooth", type=int, default=1, help="滑动平均窗口（1 = 不平滑）")
    parser.add_argument("--all", action="store_true", help="为每个 tag 单独画一张子图")
    args = parser.parse_args()

    info = export_curves(args.logdir, args.out_dir, args.prefix, args.smooth, args.all)
    print(f"[ok] 运行目录: {info['run_dir']}")
    print(f"[ok] 曲线图  : {info['png']}（{len(info['plotted'])} 条曲线）")
    print(f"[ok] 标量 CSV: {info['csv']}（{info['n_tags']} 个 tag / {info['n_rows']} 行）")
    print("-" * 78)
    for line in summarize(info["scalars"], DEFAULT_SUMMARY_TAGS):
        print(line)


if __name__ == "__main__":
    main()
