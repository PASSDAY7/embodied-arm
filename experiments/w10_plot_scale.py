#!/usr/bin/env python
"""W10 Task 3 出图：64 路 Franka 力矩控制回路的吞吐与环境数扩展性。

读 ``experiments/w10_arm_torque_control.py --out_json ...`` 写出的结果 JSON（纯标准库可解析的
dict，不需要 GPU / Isaac Sim），画三张图：

1. **SPS（steps/s，横轴环境数取对数）** —— 每步墙钟由什么决定；
2. **env-steps/s** —— 环境数翻倍时吞吐是否线性增长；
3. **单步开销归因**（来自 ``--ablate``）—— 张量通路 / ``write_data_to_sim`` / ``sim.step`` /
   ``scene.update`` 各占多少毫秒。

同时把扫描结果导成 ``<out>/w10_torque_scale.csv``，并往 stdout 打一张汇总表。

用法::

    python experiments/w10_plot_scale.py                       # 用仓库里的默认路径
    python experiments/w10_plot_scale.py --scale-json /tmp/w10_scale_1.json ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # 无显示器（headless）也能存 PNG
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "docs", "results")
FIGURES_DIR = os.path.join(REPO_ROOT, "docs", "figures")

# 图里要写中文，先把可用的 CJK 字体挑出来（不同机器上装的字体不一样）。
CJK_CANDIDATES = (
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Source Han Sans SC", "WenQuanYi Zen Hei", "Droid Sans Fallback",
    "Microsoft YaHei", "SimHei",
)


def setup_cjk_font() -> str | None:
    """挑一个已安装的 CJK 字体设为默认，返回字体名；都没有则返回 None（会退化成方框）。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免缺字形
            return name
    return None



def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_scale(paths: list[str]) -> list[dict]:
    """读入所有扫描结果，按 num_envs 升序排好（同一个 num_envs 只留第一个）。"""
    seen: dict[int, dict] = {}
    for path in paths:
        data = load_json(path)
        n = int(data["num_envs"])
        if n not in seen:
            data["_path"] = path
            seen[n] = data
    return [seen[n] for n in sorted(seen)]


def plot_scale(runs: list[dict], canonical: dict | None, out_png: str) -> None:
    ns = [r["num_envs"] for r in runs]
    sps = [r["sps_mean"] for r in runs]
    env_sps = [r["env_steps_per_s"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # --- (1) SPS：应该基本是水平线（每步固定开销主导）---
    ax = axes[0]
    ax.semilogx(ns, sps, "o-", color="tab:blue", lw=2, ms=7, label="实测 SPS")
    mean_sps = sum(sps) / len(sps)
    ax.axhline(mean_sps, color="tab:red", ls="--", lw=1.5, label=f"均值 {mean_sps:.0f} steps/s")
    for x, y in zip(ns, sps):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xlabel("并行环境数 num_envs（log）")
    ax.set_ylabel("SPS [steps/s]")
    ax.set_title("(1) 循环吞吐 vs 环境数\nSPS 几乎不变 ⇒ 每步被固定开销支配")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # --- (2) env-steps/s：应该线性增长（log-log 上斜率 1）---
    ax = axes[1]
    ax.loglog(ns, env_sps, "s-", color="tab:green", lw=2, ms=7, label="实测 env-steps/s")
    ref = [env_sps[0] * n / ns[0] for n in ns]  # 以最小环境数为基准的理想线性外推
    ax.loglog(ns, ref, ":", color="tab:gray", lw=1.5, label="理想线性（斜率 1）")
    for x, y in zip(ns, env_sps):
        ax.annotate(f"{y:,.0f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9)
    ax.set_xlabel("并行环境数 num_envs（log）")
    ax.set_ylabel("吞吐 [env-steps/s]")
    ax.set_title("(2) 仿真吞吐 vs 环境数\n256 路仍 ≈ 线性，没有撞到显存/算力墙")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # --- (3) 单步开销归因（堆叠柱）---
    ax = axes[2]
    if canonical is not None and canonical.get("ablation_ms_per_step"):
        ab = canonical["ablation_ms_per_step"]
        parts = [
            ("张量通路（compute+写命令）", ab["cost_tensor_path"], "tab:olive"),
            ("write_data_to_sim", ab["cost_write_data_to_sim"], "tab:orange"),
            ("sim.step(render=False)", ab["cost_sim_step"], "tab:red"),
            ("scene.update(dt)", ab["cost_scene_update"], "tab:purple"),
        ]
        bottom = 0.0
        for name, val, color in parts:
            ax.bar([0], [val], bottom=[bottom], color=color, width=0.5,
                   label=f"{name} {val:.3f} ms")
            bottom += val
        ax.bar([0], [bottom], color="none", edgecolor="k", width=0.5)
        ax.text(0, bottom + 0.06, f"合计 {bottom:.2f} ms/步", ha="center", fontsize=10)
        ax.set_ylim(0, bottom * 1.3)
        ax.set_title(f"(3) 单步开销归因（{canonical['num_envs']} envs）\n"
                     f"张量通路只占 {100 * parts[0][1] / bottom:.0f}%")
    else:
        ax.text(0.5, 0.5, "缺少 --ablate 结果", ha="center", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_ylabel("ms / 步")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("W10 Task 3：Isaac Lab 3.0 零拷贝张量力矩控制回路（RTX 2080 Ti, 11 GB）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=140)
    print(f"[OUT] {out_png}")


def export_csv(runs: list[dict], out_csv: str) -> None:
    cols = [
        "num_envs",
        "sps_mean",
        "sps_min",
        "sps_max",
        "ms_per_step",
        "env_steps_per_s",
        "host_bench_us_per_step",
        "vram_process_gib",
        "free_vram_after_gib",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in runs:
            writer.writerow([r.get(c, "") for c in cols])
    print(f"[OUT] {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description="W10 Task 3 出图：吞吐 / 扩展性 / 开销归因。")
    parser.add_argument("--scale-json", nargs="*", default=None,
                        help="环境数扫描的结果 JSON；默认用 docs/results/w10_scale_*.json。")
    parser.add_argument("--canonical", default=os.path.join(RESULTS_DIR, "w10_torque_64env.json"),
                        help="含 --ablate 结果的主结果 JSON。")
    parser.add_argument("--out-dir", default=FIGURES_DIR, help="PNG / CSV 输出目录。")
    parser.add_argument("--prefix", default="w10_torque_scale", help="输出文件名前缀。")
    args = parser.parse_args()

    paths = args.scale_json or sorted(glob.glob(os.path.join(RESULTS_DIR, "w10_scale_*.json")))
    if not paths:
        print(f"[ERR] 找不到扫描结果 JSON（缺 {RESULTS_DIR}/w10_scale_*.json）")
        return 1
    font = setup_cjk_font()
    print(f"[INFO] 中文字体：{font}" if font else "[WARN] 未找到 CJK 字体，图中中文会变成方框")
    runs = collect_scale(paths)
    canonical = load_json(args.canonical) if args.canonical and os.path.exists(args.canonical) else None

    os.makedirs(args.out_dir, exist_ok=True)
    plot_scale(runs, canonical, os.path.join(args.out_dir, f"{args.prefix}.png"))
    export_csv(runs, os.path.join(args.out_dir, f"{args.prefix}.csv"))

    header = f"{'num_envs':>9} {'SPS':>8} {'ms/step':>8} {'env-steps/s':>12} {'进程显存[GiB]':>14}"
    print("\n" + header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['num_envs']:>9} {r['sps_mean']:>8.1f} {r['ms_per_step']:>8.2f} "
              f"{r['env_steps_per_s']:>12,.0f} {r['vram_process_gib']:>14.2f}")
    if canonical is not None and canonical.get("ablation_ms_per_step"):
        ab = canonical["ablation_ms_per_step"]
        print(f"\n[归因 @ {canonical['num_envs']} envs, {ab['steps_per_variant']} 次/档]")
        for k in ("cost_tensor_path", "cost_write_data_to_sim", "cost_sim_step", "cost_scene_update"):
            print(f"  {k:26s} {ab[k]:.3f} ms/步")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

