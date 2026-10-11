#!/usr/bin/env python
"""W16：论文级矢量图表 + LaTeX 表格自动生成（只读 CSV/JSON，**不需要 Isaac Sim**）。

输入（全部由 ``experiments/w15_paper_benchmarks.py`` 落盘）：

========================================  ============================================================
文件                                       用途
========================================  ============================================================
``docs/results/w15_benchmarks.csv``       论文大表（每行 = 一个控制器在一个场景下的六项指标）
``docs/results/w15_per_env.csv``          逐 env 指标（Figure 2 箱线图的 64 个样本/控制器）
``docs/results/w15_baselines.json``       含时域轨迹（Figure 1 的 2 s 放大图）+ 全部自检信息
========================================  ============================================================

输出（``docs/figures/`` 与 ``docs/results/``）：

* ``fig1_tracking_zoom.pdf/.png``  时域轨迹放大图：``q_d(t)`` vs ``q(t)`` 的贴合细节 + 末端误差曲线；
* ``fig2_ee_error_boxplot.pdf/.png`` 四种控制器末端笛卡尔误差的逐 env 箱线图（64 环境）；
* ``fig3_stress_sensitivity.pdf/.png`` 超载 / 延迟敏感性折线图（横轴自变量，纵轴末端 RMSE）；
* ``w15_tables.tex`` + 终端输出：Table 1（四控制器对比）与 Table 2（消融）的 LaTeX 源码。

排版约定（IEEE Transactions 双栏）：图宽 3.5 in（单栏）、字号 8 pt、``fonttype=42``（PDF 里嵌真字体、
可被 IEEE PDF-eXpress 通过）、300 dpi PNG；坐标轴不写中文（投稿用英文），中文说明留在终端与本文件注释里。

用法::

    python experiments/export_paper_figures.py                       # 用默认路径
    python experiments/export_paper_figures.py --only fig2,tex       # 只出某几件
    python experiments/export_paper_figures.py --style report --dpi 200
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------- 常量 / 样式
LABELS = {
    "pd_dr": "PD",
    "pd_ff_dr": "PD + FF",
    "pure_rl": "Pure RL",
    "residual_sac": "Residual SAC (Ours)",
}
COLORS = {"pd_dr": "#7f7f7f", "pd_ff_dr": "#1f77b4", "pure_rl": "#2ca02c", "residual_sac": "#d62728"}
MARKERS = {"pd_dr": "o", "pd_ff_dr": "s", "pure_rl": "^", "residual_sac": "D"}
ORDER = ("pd_dr", "pd_ff_dr", "pure_rl", "residual_sac")
DATASETS = {"payload": ("Payload [kg]", "payload"), "latency": ("Action delay [steps]", "latency"),
            "friction": ("Friction / damping multiplier", "friction")}


def _style(style: str) -> None:
    """IEEE 单栏/双栏排版参数（``report`` 只是为了在屏幕上看得清）。"""
    single = style == "ieee"
    plt.rcParams.update({
        "font.size": 8 if single else 11,
        "axes.labelsize": 8 if single else 11,
        "axes.titlesize": 8 if single else 11,
        "legend.fontsize": 7 if single else 10,
        "xtick.labelsize": 7 if single else 10,
        "ytick.labelsize": 7 if single else 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.0,
        "legend.frameon": False,
        "figure.autolayout": False,
        "pdf.fonttype": 42,      # TrueType 嵌入（IEEE PDF-eXpress 要求）
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _single_col(style: str) -> tuple[float, float]:
    return (3.5, 2.6) if style == "ieee" else (7.0, 4.6)


def _save(fig, figures_dir: str, stem: str, formats: list[str], dpi: int) -> list[str]:
    os.makedirs(figures_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(figures_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=dpi, format=fmt)
        paths.append(path)
    plt.close(fig)
    for p in paths:
        print(f"[OUT] 图已写出：{p}")
    return paths


# ---------------------------------------------------------------- 读数据
def load_rows(path: str) -> list[dict]:
    """读论文大表 CSV（列值都是字符串，用 :func:`_f` 转 float）。"""
    if not os.path.isfile(path):
        raise SystemExit(f"[IN] 找不到 {path}；先跑 experiments/w15_paper_benchmarks.py --stage baselines")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"[IN] 大表：{path}（{len(rows)} 行）")
    return rows


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, "") or default)
    except (TypeError, ValueError):
        return float(default)


def pick(rows: list[dict], stage: str | None = None, scenario: str | None = None,
         controller: str | None = None) -> list[dict]:
    """按 stage / scenario / controller 过滤（``None`` = 不过滤）。"""
    out = []
    for r in rows:
        if stage is not None and str(r.get("stage", "")) != stage:
            continue
        if scenario is not None and str(r.get("scenario", "")) != scenario:
            continue
        if controller is not None and str(r.get("controller", "")) != controller:
            continue
        out.append(r)
    return out


def per_env(row: dict) -> list[float]:
    """大表行里的 ``per_env_ee_rmse_cm``（``;`` 分隔的 64 个数）。"""
    return [float(v) for v in str(row.get("per_env_ee_rmse_cm", "")).split(";") if v.strip()]


def controller_order(rows: list[dict]) -> list[str]:
    """出现过的控制器名（按 :data:`ORDER` 排序，未知的排后面）。"""
    names = {str(r.get("controller", "")) for r in rows}
    return [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)


def label_of(name: str) -> str:
    """控制器显示名（消融行形如 ``residual_sac[H=1]`` 也能认）。"""
    if name in LABELS:
        return LABELS[name]
    if name.startswith("residual_sac[") and name.endswith("]"):
        return f"Residual SAC {name[len('residual_sac['):-1]}"
    return name


def color_of(name: str) -> str:
    base = name.split("[")[0]
    return COLORS.get(base, "#9467bd")


def marker_of(name: str) -> str:
    return MARKERS.get(name.split("[")[0], "v")
# ---------------------------------------------------------------- Figure 1
def fig1_tracking_zoom(traces_json: str, figures_dir: str, formats: list[str], dpi: int,
                       style: str, window_s: float = 2.0) -> list[str]:
    """Figure 1：时域轨迹放大图（默认前 2 s，100 Hz ⇒ 200 个点）。

    上：**误差最大的那个关节**的 ``q_d(t)``（黑虚线）与四个控制器实际 ``q(t)`` ——
    相位滞后 / 幅值衰减在 2 s 的窗口里一眼可见；
    下：末端笛卡尔误差 [cm] 随时间的曲线（谁把误差压在 1 cm 以下同样一眼可见）。
    """
    if not os.path.isfile(traces_json):
        print(f"[SKIP] Figure 1：缺 {traces_json}（跑 --stage baselines 就有了）")
        return []
    with open(traces_json, encoding="utf-8") as fh:
        data = json.load(fh)
    blk = (data.get("stages") or {}).get("baselines") or {}
    traces, results = blk.get("traces") or {}, blk.get("results") or {}
    if not traces:
        print("[SKIP] Figure 1：JSON 里没有 traces（--trace_windows 0？）")
        return []

    joint_idx, worst = 0, -1.0
    for res in results.values():
        pj = (res.get("metrics", {}).get("joint_rmse_rad", {}) or {}).get("per_joint_mean_over_envs") or []
        if pj:
            j = int(np.argmax(pj))
            if float(pj[j]) > worst:
                worst, joint_idx = float(pj[j]), j
    jnames = ((list(results.values())[0].get("metrics", {}) if results else {})
              .get("joint_rmse_rad", {}).get("joint_names") or [])
    jlabel = jnames[joint_idx] if joint_idx < len(jnames) else f"joint{joint_idx + 1}"

    w, h = _single_col(style)
    fig, axes = plt.subplots(2, 1, figsize=(w, h * 1.75 if style == "ieee" else h * 1.6), sharex=True)
    first = traces.get(ORDER[0]) or traces[list(traces)[0]]
    t_ref = np.asarray(first["t_s"], dtype=float)
    keep = np.ones_like(t_ref, dtype=bool) if not window_s else (t_ref <= float(window_s))
    qd = np.asarray(first["q_d_rad"], dtype=float)[:, joint_idx]
    axes[0].plot(t_ref[keep], qd[keep], "k--", lw=1.0, label=r"$q_d$ (reference)")
    for name in ORDER:
        if name not in traces:
            continue
        d = traces[name]
        t = np.asarray(d["t_s"], dtype=float)
        k = np.ones_like(t, dtype=bool) if not window_s else (t <= float(window_s))
        axes[0].plot(t[k], np.asarray(d["q_rad"], dtype=float)[:, joint_idx][k],
                     color=color_of(name), label=label_of(name))
        axes[1].plot(t[k], np.asarray(d["ee_cm"], dtype=float)[k], color=color_of(name), label=label_of(name))
    axes[0].set_ylabel(f"{jlabel} [rad]")
    axes[0].legend(ncol=2, loc="best")
    axes[1].set_ylabel("End-effector error [cm]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(ncol=2, loc="best")
    print(f"[FIG1] 关节取 {jlabel}（{ORDER[0]} 误差最大的那个关节）；窗口 {window_s:g} s")
    return _save(fig, figures_dir, "fig1_tracking_zoom", formats, dpi)


# ---------------------------------------------------------------- Figure 2
def fig2_boxplot(rows: list[dict], figures_dir: str, formats: list[str], dpi: int,
                 style: str, per_env_csv: str = "", log_axis: str = "auto") -> list[str]:
    """Figure 2：末端笛卡尔误差的逐 env 箱线图（每个箱子 64 个样本 = 64 个并行环境）。

    为什么要箱线图而不是只报均值：失配是按 env 独立采样的（0–3 kg 负载、±20% 连杆质量、
    0.1–1.0 N·m 摩擦），**分布尾部的环境**才是应力所在；箱线图把中位数/四分位/极值一次说清。

    纵轴默认 **auto**：当最差与最好控制器的中位数相差 > 20× 时自动切对数轴 —— 否则纯 RL 那一列
    会把另外三列压成一条线（差值本身就是结论的一部分，但压成一条线就什么都看不出来了）。
    """
    base = pick(rows, stage="baselines")
    if not base:
        print("[SKIP] Figure 2：大表里没有 stage=baselines 的行")
        return []
    labels, data_names, data = [], [], []
    for name in controller_order(base):
        row = next((r for r in base if r["controller"] == name), None)
        vals = per_env(row or {})
        if not vals and per_env_csv:
            vals = _per_env_from_csv(per_env_csv, name)
        if vals:
            labels.append(label_of(name))
            data_names.append(name)
            data.append(vals)
    if not data:
        print("[SKIP] Figure 2：没有逐 env 数据（per_env_ee_rmse_cm 列或 --per_env_csv）")
        return []
    medians = [float(np.median(v)) for v in data]
    use_log = (log_axis == "on") or (log_axis == "auto" and min(medians) > 0
                                     and max(medians) / min(medians) > 20.0)
    fig = plt.figure(figsize=_single_col(style))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, showmeans=True, patch_artist=True, widths=0.55,
                    meanprops={"marker": "+", "markeredgecolor": "k", "markersize": 4})
    for patch, name in zip(bp["boxes"], data_names):
        patch.set_facecolor(color_of(name))
        patch.set_alpha(0.45)
        patch.set_edgecolor(color_of(name))
    for med in bp["medians"]:
        med.set_color("k")
        med.set_linewidth(1.0)
    for i, vals in enumerate(data, start=1):
        med = float(np.median(vals))
        ax.annotate(f"{med:.2f}", (i, med), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=6, color="k")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("End-effector RMSE [cm]" + ("  (log scale)" if use_log else ""))
    if use_log:
        ax.set_yscale("log")
        ax.grid(True, which="minor", alpha=0.15, linewidth=0.3)
    print("[FIG2] 箱线图样本数 = " + ", ".join(f"{label_of(n)}: {len(v)}" for n, v in zip(data_names, data))
          + f" | 纵轴 = {'log' if use_log else 'linear'}"
          + f"（中位数 {min(medians):.2f}–{max(medians):.2f} cm，跨 {max(medians)/max(min(medians),1e-9):.1f}×）")
    return _save(fig, figures_dir, "fig2_ee_error_boxplot", formats, dpi)


def _per_env_from_csv(path: str, controller: str) -> list[float]:
    """从 ``w15_per_env.csv`` 里取某个控制器在 baselines 场景下的逐 env 末端 RMSE。"""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [_f(r, "ee_rmse_cm") for r in rows
            if str(r.get("stage")) == "baselines" and str(r.get("controller")) == controller]
# ---------------------------------------------------------------- Figure 3
def fig3_stress(rows: list[dict], figures_dir: str, formats: list[str], dpi: int,
                style: str) -> list[str]:
    """Figure 3：压测敏感性折线图（横轴 = 自变量，纵轴 = 末端 RMSE [cm]）。

    * 载荷轴：训练分布只到 3 kg，4 / 5 kg 是**分布外**——图上用虚线标出训练边界；
    * 延迟轴：0→3 步（0→30 ms @100 Hz）的动作下发延迟；
    * 摩擦轴：damping/静摩擦 ×1.0→×3.0（传动磨损 / 润滑失效的等效）。
    """
    axes_specs = [(a, DATASETS[a][0]) for a in ("payload", "latency", "friction") if pick(rows, stage=a)]
    if not axes_specs:
        print("[SKIP] Figure 3：大表里没有 payload / latency / friction 任一赛程的行")
        return []
    w, h = _single_col(style)
    fig, axes = plt.subplots(1, len(axes_specs), figsize=(w * len(axes_specs) if style == "ieee"
                                                          else 6.0 * len(axes_specs), h), squeeze=False)
    for ax, (axis, xlabel) in zip(axes[0], axes_specs):
        group = pick(rows, stage=axis)
        for name in controller_order(group):
            pts = sorted((_f(r, "sweep_value"), _f(r, "ee_rmse_cm"), per_env(r))
                         for r in group if r["controller"] == name)
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            ax.plot(x, y, marker=marker_of(name), color=color_of(name), label=label_of(name))
            for xi, _yi, vals in pts:                      # 逐 env 的 IQR 带（只在样本够时画）
                if len(vals) >= 8:
                    lo, hi = np.percentile(vals, [25, 75])
                    ax.fill_between([xi - 0.02, xi + 0.02], [lo, lo], [hi, hi],
                                    color=color_of(name), alpha=0.18, lw=0)
        if axis == "payload":
            ax.axvline(3.0, ls=":", color="k", lw=0.8)
            ax.text(3.02, ax.get_ylim()[1], " training limit", fontsize=6, va="top", ha="left")
            ax.axvspan(3.0, ax.get_xlim()[1], color="k", alpha=0.05, lw=0)
        if axis == "friction":
            ax.axvline(1.0, ls=":", color="k", lw=0.8)
            ax.text(1.02, ax.get_ylim()[1], " nominal", fontsize=6, va="top", ha="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("End-effector RMSE [cm]")
        ax.legend(loc="best")
    print(f"[FIG3] 画了 {len(axes_specs)} 条轴：" + ", ".join(a for a, _ in axes_specs))
    return _save(fig, figures_dir, "fig3_stress_sensitivity", formats, dpi)
# ---------------------------------------------------------------- LaTeX 表格
def _tex_header(caption: str, label: str) -> list[str]:
    """统一的 booktabs 表头（列 = 论文要求的六项指标）。"""
    return [
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Controller & $\\mathrm{RMSE}_q$ [rad] & $\\mathrm{RMSE}_{ee}$ [cm] & $e_{\\max}$ [cm] & "
        "$\\overline{\\|\\tau\\|_2}$ [N$\\cdot$m] & Jerk [N$^2\\cdot$m$^2$] & Sat. ratio \\\\",
        "    \\midrule",
    ]


def _tex_row(name: str, row: dict, bold: bool = False) -> str:
    """一行数据（``bold`` 用于 Ours 那行）。"""
    cells = [f"{_f(row, 'joint_rmse_rad'):.6f}", f"{_f(row, 'ee_rmse_cm'):.4f}",
             f"{_f(row, 'ee_max_cm'):.3f}", f"{_f(row, 'mean_tau_norm_nm'):.3f}",
             f"{_f(row, 'jerk_nm2'):.3f}", f"{_f(row, 'sat_step_frac') * 100:.2f}\\%"]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
        name = f"\\textbf{{{name}}}"
    return "    " + name + " & " + " & ".join(cells) + " \\\\"


def tex_table1(rows: list[dict]) -> str:
    """Table 1：四个控制器在训练失配分布下的全周期指标（论文正文那张大表）。"""
    base = pick(rows, stage="baselines")
    if not base:
        return "% Table 1：大表里没有 stage=baselines 的行（先跑 --stage baselines）"
    names = controller_order(base)
    lines = [
        "% " + "=" * 76,
        "% Table 1：四控制器 Baseline 横向对比（experiments/export_paper_figures.py 自动生成）",
        "% 统一条件：同一条参考轨迹 / 同一组 DR 采样 / 同一套指标代码；越小越好",
        "% 需要 \\usepackage{booktabs}",
        "% " + "=" * 76,
        "\\begin{table*}[t]",
    ]
    caption = ("Full-episode tracking performance of the four controllers under the identical mismatch "
               "distribution (unknown payload $0$--$3$\\,kg, link mass $\\pm20\\%$, joint damping "
               "$0.5$--$2.0$\\,N$\\cdot$m$\\cdot$s/rad, static friction $0.1$--$1.0$\\,N$\\cdot$m; "
               "64 parallel environments, 500 steps at 100\\,Hz). "
               "$e_{\\max}$ is the worst instantaneous end-effector error over all environments and "
               "time steps; Jerk $=\\frac{1}{T}\\sum_t\\|\\tau_t-\\tau_{t-1}\\|_2^2$; "
               "Sat.\\ ratio is the fraction of steps with any joint at its torque limit.")
    lines += _tex_header(caption, "tab:baseline")
    for name in names:
        row = next((r for r in base if r["controller"] == name), None)
        if row is None:
            continue
        lines.append(_tex_row(label_of(name), row, bold=(name == "residual_sac")))
    ref = next((r for r in base if r["controller"] == "pd_ff_dr"), None)
    ours = next((r for r in base if r["controller"] == "residual_sac"), None)
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    if ref is not None and ours is not None:
        d_ee = (1.0 - _f(ours, "ee_rmse_cm") / _f(ref, "ee_rmse_cm")) * 100.0
        d_j = (1.0 - _f(ours, "joint_rmse_rad") / _f(ref, "joint_rmse_rad")) * 100.0
        lines += [
            "  \\vspace{2pt}",
            "  \\begin{minipage}{\\linewidth}\\footnotesize",
            f"  Relative to PD+FF: end-effector RMSE $-${d_ee:.1f}\\%, joint RMSE $-${d_j:.1f}\\%. "
            "All controllers share the same reference trajectory, the same domain-randomisation "
            "realisation and the same metric implementation.",
            "  \\end{minipage}",
        ]
    lines += ["\\end{table*}", ""]
    return "\n".join(lines)


def tex_table2(rows: list[dict]) -> str:
    """Table 2：消融实验（历史长度 H / 残差限幅 / jerk 惩罚项）。"""
    abl = pick(rows, stage="ablation")
    if not abl:
        return "% Table 2：大表里没有 stage=ablation 的行（先跑 --stage ablation，训练较贵）"
    dims: list[str] = []
    for r in abl:
        d = str(r.get("sweep", ""))
        if d and d not in dims:
            dims.append(d)
    lines = [
        "% " + "=" * 76,
        "% Table 2：消融实验（每个变体训练到同一环境步预算后评估）",
        "% 需要 \\usepackage{booktabs}",
        "% " + "=" * 76,
        "\\begin{table*}[t]",
    ]
    caption = ("Ablation study. Every variant is trained with the identical budget, reward and "
               "domain-randomisation distribution, and evaluated with the same protocol as "
               "Table~\\ref{tab:baseline}. $H$: number of stacked observation frames; "
               "$|\\Delta\\tau|$: residual torque limit; $w_j$: jerk penalty weight.")
    lines += _tex_header(caption, "tab:ablation")
    for d in dims:
        lines.append(f"    \\multicolumn{{7}}{{l}}{{\\textit{{{d}}}}} \\\\")
        for name in controller_order(pick(rows, stage="ablation")):
            row = next((r for r in abl if r["controller"] == name and str(r.get("sweep")) == d), None)
            if row is None:
                continue
            variant = str(row.get("scenario", "")).split(":", 1)[-1]
            lines.append(_tex_row(variant, row, bold=variant.endswith("(Ours)")))
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)
# ---------------------------------------------------------------- main
def main() -> int:
    """命令行入口：读 W15 落盘的大表/JSON → 出 Figure 1/2/3 + LaTeX 表 1/2。"""
    parser = argparse.ArgumentParser(
        description="W15 → 论文级矢量图 + LaTeX 表格（只读 CSV/JSON，不需要 Isaac Sim）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    res_dir = os.path.join(_REPO, "docs", "results")
    parser.add_argument("--table_csv", type=str, default=os.path.join(res_dir, "w15_benchmarks.csv"),
                        help="W15 落盘的论文大表 CSV。")
    parser.add_argument("--per_env_csv", type=str, default=os.path.join(res_dir, "w15_per_env.csv"),
                        help="逐 env 指标 CSV（大表里 per_env_ee_rmse_cm 列为空时兜底）。")
    parser.add_argument("--baselines_json", type=str, default=os.path.join(res_dir, "w15_baselines.json"),
                        help="--stage baselines 的 JSON（Figure 1 的时域轨迹在这里面）。")
    parser.add_argument("--figures_dir", type=str, default=os.path.join(_REPO, "docs", "figures"),
                        help="图输出目录。")
    parser.add_argument("--out_tex", type=str, default=os.path.join(res_dir, "w15_tables.tex"),
                        help="LaTeX 表格输出路径（同时打印到终端）。")
    parser.add_argument("--only", type=str, default="all", help="逗号分隔：fig1,fig2,fig3,tex,all。")
    parser.add_argument("--style", choices=["ieee", "report"], default="ieee",
                        help="ieee=单栏 3.5 in / 8 pt；report=放大版（屏幕上看）。")
    parser.add_argument("--dpi", type=int, default=300, help="PNG 分辨率（矢量 PDF 不受影响）。")
    parser.add_argument("--formats", type=str, default="pdf,png", help="输出格式（逗号分隔）。")
    parser.add_argument("--window_s", type=float, default=2.0, help="Figure 1 的时域放大窗口 [s]。")
    parser.add_argument("--fig2_log", choices=["auto", "on", "off"], default="auto",
                        help="Figure 2 纵轴是否用对数刻度（auto：中位数跨 >20× 时自动切）。")
    args = parser.parse_args()

    _style(args.style)
    formats = [s.strip() for s in str(args.formats).split(",") if s.strip()]
    want = {s.strip() for s in str(args.only).split(",") if s.strip()}

    def _want(tag: str) -> bool:
        return "all" in want or tag in want

    rows = load_rows(args.table_csv)
    made: list[str] = []
    if _want("fig1"):
        made += fig1_tracking_zoom(args.baselines_json, args.figures_dir, formats, args.dpi,
                                   args.style, args.window_s)
    if _want("fig2"):
        made += fig2_boxplot(rows, args.figures_dir, formats, args.dpi, args.style, args.per_env_csv,
                             args.fig2_log)
    if _want("fig3"):
        made += fig3_stress(rows, args.figures_dir, formats, args.dpi, args.style)
    if _want("tex"):
        tex = tex_table1(rows) + "\n" + tex_table2(rows)
        os.makedirs(os.path.dirname(os.path.abspath(args.out_tex)) or ".", exist_ok=True)
        with open(args.out_tex, "w", encoding="utf-8") as fh:
            fh.write(tex)
        print(f"[OUT] LaTeX 表格：{args.out_tex}")
        print("\n" + "=" * 94 + "\n[LaTeX] 直接粘进论文（需要 \\usepackage{booktabs}）\n" + "=" * 94)
        print(tex)
    print("\n[SUMMARY] 本次产出：")
    for p in made:
        print(f"  - {p}")
    if not made and not _want("tex"):
        print("  （什么都没出：--only 里没给有效的 tag？）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
