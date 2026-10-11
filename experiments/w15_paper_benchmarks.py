#!/usr/bin/env python
"""W15：论文级三维实验矩阵 —— 四控制器统一 Baseline 对比（Table 1）+ 极端抗扰压测（Figure 3）+ 消融（Table 2）。

.. code-block:: text

    统一评测口径（同一个 env、同一套指标代码，四个控制器只换"力矩怎么算"）
    ============================================================================
      pd_dr         τ = Kp(q_d−q) + Kd(q̇_d−q̇)                        （无模型先验，纯反馈）
      pd_ff_dr      τ = Ĝ_nom(q) + Kp(q_d−q) + Kd(q̇_d−q̇)              （工业界最常见：CAD 标称前馈）
      pure_rl       τ = clip(Δτ_full, ±τ_lim)，Δτ_full = scale ⊙ tanh(π_φ(o))，scale = τ_lim
                    （切断经典控制器：τ_base ≡ 0，SAC 从零输出**全部**关节力矩）
      residual_sac  τ = clip(Ĝ_nom(q) + Kp(q_d−q) + Kd(q̇_d−q̇) + Δτ, ±τ_lim)，|Δτ| ≤ 1.5 N·m
                    （Ours：双层残差，基座兜底 + 上层补偿动态残差）
    ============================================================================

为什么要有这个脚本（而不是把评测塞进 W13）：W13 的每一次评估只回答"我的残差策略比基座好多少"，
它跑的是**训练用的那一条 DR 采样**。论文实验矩阵要回答的是三个不同的问题：

1. **同一个失配分布下，四种控制器谁好、好多少**（Table 1）——
   因此必须把"力矩怎么算"这一个变量独立出来，DR 采样、参考轨迹、增益、指标代码全部共用；
2. **失配超出训练分布以后，谁先崩**（Figure 3）——载荷 0→5 kg（训练只到 3 kg）、
   摩擦/阻尼 ×1.0→×3.0、动作下发延迟 0→3 步（0→30 ms）三个方向分别压测；
3. **框架里每个组件是不是必要的**（Table 2）——历史帧长 H、残差动作限幅、jerk 惩罚项逐项消融。

一、指标口径（与 W12/W13 同源，逐行对齐，不另起炉灶）
-----------------------------------------------------
.. list-table::
   :header-rows: 1

   * - 指标
     - 定义
     - 落盘键
   * - 关节跟踪 RMSE
     - ``sqrt(mean_t (q_j(t) − q_{d,j}(t))²)``，先逐 env 逐关节、再跨 env 取均值 [rad]
     - ``joint_rmse_rad``
   * - 末端笛卡尔 RMSE
     - ``sqrt(mean_t ‖p(q(t)) − p(q_d(t))‖²)`` [cm]（FK 自校验必须先通过）
     - ``ee_rmse_cm``
   * - 最大瞬时末端误差
     - ``max_{t,e} ‖p(q)−p(q_d)‖`` [cm]（最坏时刻 × 最坏 env，不接受"平均值掩盖峰值"）
     - ``ee_max_cm``
   * - 平均力矩范数
     - ``mean_t ‖τ_t‖₂`` [N·m]（执行器能耗口径）
     - ``mean_tau_norm_nm``
   * - 控制抖动
     - ``mean_t ‖τ_t − τ_{t−1}‖²`` [N·m²]（τ_{−1} = 0，与 W12/W13 奖励里那一项同定义）
     - ``jerk_nm2``
   * - 饱和步占比
     - 任一关节 ``|τ| ≥ τ_lim − 1e-9`` 的步数比例（**执行器安全性**）
     - ``sat_step_frac``

**为什么要重写一遍评测循环（而不是复用 ``run_policy_episode``）**：那两份循环的累加器只统计了关节/末端/奖励，
没有"平均力矩范数、抖动、逐 env 末端 RMSE（箱线图要用）"。本脚本把这六个指标放进**同一份**累加器，
四个控制器共用 ⇒ 控制器之间不可能因为"口径不同"而产生虚假差异。代价是必须证明"新循环 ≡ 老循环"：
``--verify_w12_path`` 会用 W12 的 ``run_episode`` 再跑一遍 pd_dr / pd_ff_dr，把两份结果逐字段对账
（一致判据：相对偏差 < 1e-9；不通过就直接在 JSON 里标红，见 :func:`verify_unified_vs_w12`）。

二、压测怎么"钉死"除自变量以外的失配
------------------------------------
:class:`DynamicsRandomizer` 只是一个"按 seed 从区间采样 → 写进 PhysX"的注入器，所以压测点 =
**换一组区间 + 同一个 seed + 重新注入**。本脚本的默认约定（``--stress_fix`` 可覆盖）：

======  ==========================================  =================================================
压测轴   自变量                                           其余失配
======  ==========================================  =================================================
载荷     ``payload_mass = (m, m)``，m ∈ 0,1.5,3,4,5 kg   连杆 ×U(0.8,1.2) + 摩擦/阻尼 = 训练区间（照原样）
摩擦     ``damping/friction ×mult``，mult ∈ 1,1.5,2,3     载荷钉死 1.5 kg（训练区间中点）+ 连杆 ×U(0.8,1.2)
延迟     ``--action_delay`` ∈ 0,1,2,3 步（10 ms/步）      全部失配 = 训练分布（就是训练时那一套）
======  ==========================================  =================================================

延时的物理含义：控制器算出 ``τ_t`` 后**不是立刻下发**，而是经过一条长度 N 的 FIFO 延迟线
（`N` 步前的力矩此刻才到达关节）。延迟线在 episode 开始时用首步力矩**预填**（等价于"这条链路
实验开始前就已经在跑"），因此不会引入额外的起步瞬态；当 ``N = 0`` 时行为与 W13 逐位相同。

三、四个控制器的公平性约定（写进论文方法节前必须交代）
------------------------------------------------------
1. **奖励完全一致**（含 ``--w_ee 50000`` 的末端项与 ``--w_mag 0.05`` 的残差幅值正则）：
   对比的是**架构**（基座在不在、动作空间多大），不是奖励工程；
2. **训练预算一致**（``--total_steps`` 默认 450000 环境步、64 路并行、同一随机种子）；
3. **观测一致**（同一个 105 维 H=3 时序观测；pure_rl 也照给，不给它"额外照顾"）；
4. **训练用的失配分布一致**（同一条 DR 采样）；
5. 评估时策略一律**确定性**（``tanh(μ)``）、``torch.no_grad()``、不写回放池、不更新归一化统计量。

用法::

    # ① Table 1：四控制器横向对比（pure_rl 没有权重时会先自动训练一轮）
    bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py \\
        --stage baselines --out_json docs/results/w15_table1_baselines.json

    # ② Figure 3（载荷轴）：0/1.5/3.0/4.0/5.0 kg，其中 4/5 kg 是分布外
    bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py \\
        --stage payload --payload_list 0,1.5,3,4,5

    # ③ 摩擦退化 / 动作延迟
    bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage friction --friction_mult 1,1.5,2,3
    bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage latency --latency_steps 0,1,2,3

    # ④ 只跑压测的某一个点做冒烟（不训练、加载现成权重）
    bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py \\
        --stage payload --payload_list 4 --num_envs 16 --steps 100 --no_train_pure_rl

    # ⑤ 出图 + LaTeX 表（不需要 Isaac Sim，纯 numpy/matplotlib）
    python experiments/export_paper_figures.py

自检（每次运行都会做，结果落在 JSON 的 ``self_checks``）：
    * ``fk``：末端误差的度量基础（``FK(q)`` vs PhysX，阈值 ``--fk_tolerance``）；
    * ``verify_w12_path``：新统一评测循环 vs W12 ``run_episode``（pd_dr / pd_ff_dr 各一遍）；
    * ``dr_readback``：每个压测点的注入值从 PhysX 回读（证明"压测真的加压了"）；
    * ``policy_meta``：加载的策略权重里带的训练元信息（步数 / 奖励权重），防止拿错权重。
"""

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 181-207 行] 头部 import + isaaclab 导入守卫
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:  # pragma: no cover - 只在环境没配好时触发
    raise SystemExit(
        "找不到 isaaclab：请用仓库里的环境固化脚本启动（它会激活 py312 并挂好 PYTHONPATH）：\n"
        "    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py --num_envs 64\n"
        "或者自己拼环境（等价做法）：\n"
        "    export PYTHONPATH=$(ls -d /data/zzh/IsaacLab/source/*/ | tr '\\n' ':')\n"
        "    HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python -u "
        f"{os.path.abspath(__file__)} --num_envs 64\n"
    ) from exc

##
# ============================================================================
# W15 命令行：规模/轨迹/观测/奖励/DR/基座控制器/SAC 超参 **与 W13 同名同默认**
# ----------------------------------------------------------------------------
# 下面这一组 add_argument 是从 w13_residual_sac_train.py 的 _make_parser() 里
# **逐字切过来**的（行号见本文件末尾的生成说明），保证"W15 跑的是同一个环境"
# 不是口头承诺；W15 自己只新增 §赛程与产物 那一组开关。
# ============================================================================
##
def _make_parser() -> argparse.ArgumentParser:
    """W15 的全部命令行参数（环境部分与 W13/W12 同名同默认，方便横向对账）。"""
    parser = argparse.ArgumentParser(
        description="W15：论文级实验矩阵 —— 四控制器 Baseline 对比 / 极端抗扰压测 / 消融 + 统一评测落盘。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- 规模与积分（与 W12 同名）----
    parser.add_argument("--num_envs", type=int, default=64, help="并行环境数（同时是并行采样的批大小）。")
    parser.add_argument("--steps", type=int, default=500, help="单条 episode 的步数（也是评估步数）。")
    parser.add_argument("--dt", type=float, default=1.0 / 100.0, help="物理步长 [s]，默认 100 Hz。")
    parser.add_argument("--env_spacing", type=float, default=2.5, help="环境间距 [m]。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（DR 采样 / 轨迹采样 / SAC 采样）。")
    parser.add_argument("--no_gravity", action="store_true", help="关闭重力（默认开，与 W12 一致）。")
    parser.add_argument("--no_ground", action="store_true", help="不生成地面（默认生成）。")
    parser.add_argument("--init_at_reference", action="store_true", default=True,
                        help="每段 episode 从 q_d(0)/q̇_d(0) 起步（默认开，去掉起步瞬态）。")
    parser.add_argument("--no_init_at_reference", action="store_false", dest="init_at_reference",
                        help="改为从默认位形起步。")

    # ---- 参考轨迹（与 W12 同名同默认）----
    parser.add_argument("--traj_amp", type=float, nargs=2, default=(0.15, 0.35), metavar=("MIN", "MAX"),
                        help="逐关节参考幅值 A_i ~ U(MIN, MAX) [rad]（再按关节限位 clamp）。")
    parser.add_argument("--traj_freq", type=float, nargs=2, default=(0.2, 0.6), metavar=("MIN", "MAX"),
                        help="逐关节参考频率 f_i ~ U(MIN, MAX) [Hz]。")
    parser.add_argument("--traj_center", type=float, nargs=7, default=(0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.0),
                        metavar="Q0", help="参考轨迹中心 q0（7 个臂关节）[rad]。")
    parser.add_argument("--traj_seed", type=int, default=-1, help="轨迹采样种子，-1 表示跟随 --seed。")
    parser.add_argument("--harmonics", type=int, default=1, help="每个关节叠加的正弦项数 K。")
    parser.add_argument("--traj_margin", type=float, default=0.8,
                        help="幅值相对关节限位余量的安全比例（0.8 = 至少留 20%% 余量）。")

    # ---- 观测 / 奖励（与 W12 同名同默认）----
    parser.add_argument("--history", type=int, default=3, help="历史观测帧数 H（任务书要求 3）。")
    parser.add_argument("--history_warmup", choices=["repeat", "zeros"], default="repeat",
                        help="历史没填满时的填充策略。")
    parser.add_argument("--wp", type=float, default=10.0, help="位置误差权重 w_p。")
    parser.add_argument("--wv", type=float, default=0.1, help="速度误差权重 w_v。")
    parser.add_argument("--wtau", type=float, default=0.001, help="执行力矩幅值权重 w_τ。")
    parser.add_argument("--wj", type=float, default=0.01, help="抖动（jerk）权重 w_j。")
    parser.add_argument("--w_ee", type=float, default=0.0,
                        help="**任务书四项之外**的末端位置奖励权重 w_ee·‖p(q)−p(q_d)‖²（默认 0 = 只用任务书四项）。"
                             "为什么要它（可选）：任务书奖励是**关节空间各向同性**的和方，而验收指标是**末端位置**；"
                             "实测 j7（绕法兰轴旋转、几乎不移动手部原点）占了误差平方和的约 60%%，于是关节奖励的改进"
                             "（pos −21%%、vel −5%%）在末端指标上被抵消（1.3279 vs 1.3278 cm 持平）。设 --w_ee 20000 "
                             "把验收指标本身写进奖励（基线处该项 ≈3.5/步，与 τ 项同量级）。")
    parser.add_argument("--w_mag", type=float, default=0.05,
                        help="**任务书四项之外**的残差幅值正则 w_mag·‖Δτ‖²（默认 0.05，设 0 回到任务书四项）。"
                             "为什么要它：残差结构里 ‖τ_exec‖² 项会**奖励抵消基座力矩**（最省力的降 ‖τ‖ 方式是让基座空转），"
                             "实测 w_mag=0 时 |Δτ| 从 0.002 一路漂到 3.5 N·m、末端误差 1.33 → 2.6 cm。")
    parser.add_argument("--jerk_on", choices=["residual", "exec"], default="residual",
                        help="jerk 项惩罚谁：residual=Δτ_t−Δτ_{t−1}（任务书口径，默认）；exec=τ_exec 差分（W12 口径）。")

    # ---- 被控关节 / FK（与 W12 同名同默认）----
    parser.add_argument("--joint_expr", type=str, default=r"panda_joint[1-7]", help="被控关节正则。")
    parser.add_argument("--ee_body", type=str, default="panda_hand", help="末端连杆名。")
    parser.add_argument("--urdf", type=str, default="", help="Franka URDF 路径（默认自动找）。")
    parser.add_argument("--no_fk_check", action="store_true", help="跳过 FK vs PhysX 自校验。")
    parser.add_argument("--fk_tolerance", type=float, default=1e-4, help="FK 自校验通过阈值 [m]。")

    # ---- W11 域随机化（与 W12 同名同默认）----
    parser.add_argument("--no_randomization", action="store_true", help="完全不注入失配（消融）。")
    parser.add_argument("--payload_mass", type=float, nargs=2, default=(0.0, 3.0), metavar=("MIN", "MAX"),
                        help="末端未知负载质量范围 [kg]。")
    parser.add_argument("--payload_body", type=str, default="panda_hand", help="挂载负载的连杆名。")
    parser.add_argument("--payload_com", type=float, nargs=3, default=(0.0, 0.0, 0.05), metavar=("X", "Y", "Z"),
                        help="负载质心（载荷连杆系）[m]。")
    parser.add_argument("--payload_mode", choices=["parallel_axis", "mass"], default="parallel_axis",
                        help="负载惯量处理方式。")
    parser.add_argument("--link_mass_scale", type=float, nargs=2, default=(0.8, 1.2), metavar=("LO", "HI"),
                        help="连杆质量比例扰动范围。")
    parser.add_argument("--joint_damping", type=float, nargs=2, default=(0.5, 2.0), metavar=("LO", "HI"),
                        help="关节阻尼范围 [N·m·s/rad]。")
    parser.add_argument("--joint_friction", type=float, nargs=2, default=(0.1, 1.0), metavar=("LO", "HI"),
                        help="关节静摩擦力矩范围 [N·m]。")

    # ---- 底层基座控制器（与 W12 同名同默认）----
    parser.add_argument("--gains", choices=["mweighted", "auto", "manual"], default="mweighted",
                        help="PD 增益来源：mweighted=Kp=ωn²·M̂(q0) 矩阵增益（默认）；auto=对角；manual=--kp/--kd。")
    parser.add_argument("--wn", type=float, default=30.0, help="目标闭环带宽 ωn [rad/s]。")
    parser.add_argument("--zeta", type=float, default=1.0, help="阻尼比 ζ。")
    parser.add_argument("--kp", type=float, nargs="+", default=(100.0,), help="[--gains manual] 比例增益。")
    parser.add_argument("--kd", type=float, nargs="+", default=(10.0,), help="[--gains manual] 微分增益。")
    parser.add_argument("--pd_sign", choices=["stabilizing", "literal"], default="stabilizing",
                        help="PD 符号口径：stabilizing=标准（默认）；literal=任务书字面写法（正反馈，仅供对照）。")
    parser.add_argument("--ff", choices=["nominal", "oracle"], default="nominal",
                        help="重力前馈来源：nominal=标称模型（诚实，默认）；oracle=读 PhysX（知道负载，上限参照）。")
    parser.add_argument("--base_mode", choices=["pd_ff", "pd"], default="pd_ff",
                        help="基座是否叠加重力前馈：pd_ff=Ĝ+PD（默认，任务书口径）；pd=纯 PD。")

    # ---- 残差动作空间 ----
    parser.add_argument("--res_limit", type=float, default=1.5,
                        help="残差力矩硬上限 |Δτ| ≤ res_limit [N·m]（tanh 动作头缩放的天花板）。"
                             "任务书写的是 ±15；默认 1.5 是因为**要补的量**只有这个量级"
                             "（W12 量到：关节静摩擦 0.1–1.0 N·m、阻尼 ≤2 N·m·s/rad）；"
                             "--res_limit 15 --res_scale_mode uniform 可回到任务书字面配置（实测会漂移，见 docstring §四）。")
    parser.add_argument("--res_scale_mode", choices=["kp", "uniform"], default="kp",
                        help="逐关节缩放口径：kp=scale_j=min(res_limit, deflection·Kp_j)（默认，按关节刚度归一化，"
                             "让满量程对应各关节**相同的静态偏转**）；uniform=所有关节都用 res_limit（任务书口径）。")
    parser.add_argument("--res_deflection_rad", type=float, default=0.2,
                        help="kp 口径下的满量程静态偏转 [rad]（Δτ_j/Kp_j）；0.2 ⇒ j7（Kp=2.60）的残差上限 ≈0.52 N·m。")

    # ---- SAC 训练超参（默认 = W9 的 Pendulum 配置 + 残差场景的三处改动）----
    parser.add_argument("--total_steps", type=int, default=500000,
                        help="训练总预算（**累计环境步数** = 交互帧数 × num_envs）。")
    parser.add_argument("--buffer_size", type=int, default=300000, help="回放池容量（条数）。")
    parser.add_argument("--buffer", choices=["gpu", "host"], default="gpu",
                        help="回放池实现：gpu=显存整块环形写入（默认）；host=W9 的 numpy 版（对照）。")
    parser.add_argument("--batch_size", type=int, default=256, help="SAC mini-batch 大小。")
    parser.add_argument("--warmup_steps", type=int, default=20000,
                        help="热身期**环境步数**（零残差 + 小幅随机，不做梯度更新）。")
    parser.add_argument("--warmup_res_scale", type=float, default=1.0,
                        help="热身期随机残差幅值 [N·m]（Δτ = U(−s, s)）——围绕基座做**局部**探索，"
                             "不要用 ±15 那种全局乱甩（见 docstring §四）。")
    parser.add_argument("--train_freq", type=int, default=1, help="每多少个环境步做一次梯度更新。")
    parser.add_argument("--gradient_steps", type=int, default=2, help="每次更新做几个 mini-batch（UTD）。")
    parser.add_argument("--actor_delay_updates", type=int, default=1000,
                        help="Critic 热身：前 N 次更新把 Actor 学习率压到 0（只学 Q），之后再放开 —— "
                             "避免 Actor 在 Q 还没学到价值尺度时就朝'乐观的 Q'乱跑。")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="策略学习率。")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="双 Q 学习率。")
    parser.add_argument("--lr_alpha", type=float, default=3e-4, help="温度 α 学习率。")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子。")
    parser.add_argument("--tau", type=float, default=0.005, help="目标网络 Polyak 系数。")
    parser.add_argument("--alpha_init", type=float, default=0.02, help="温度 α 初值（固定温度时就是最终值）。")
    parser.add_argument("--target_entropy", type=str, default="fixed",
                        help="目标熵：fixed=固定 α（默认 —— 残差 RL 要**近乎确定性**的策略）；"
                             "auto=−dim(A)=−7；也可给具体浮点数（如 -1.0）做自适应对照。"
                             "为什么默认固定：自适应控制器会把 α 一路推高（实测 30 万步里 0.05 → 0.10），"
                             "探索噪声越大 ⇒ 残差越粗 ⇒ 末端误差越差（实测曲线 1.33 → 2.6 cm 与 α/|Δτ| 同步上涨）。")
    parser.add_argument("--hidden", type=int, default=256, help="Actor 隐藏层宽度。")
    parser.add_argument("--critic_hidden", type=int, default=256, help="Critic 隐藏层宽度。")
    parser.add_argument("--layers", type=int, default=2, help="隐藏层数。")
    parser.add_argument("--activation", type=str, default="relu", help="激活函数。")
    parser.add_argument("--log_std_min", type=float, default=-5.0, help="log σ 下界（σ ≥ e^-5 ≈ 6.7e-3）。")
    parser.add_argument("--log_std_max", type=float, default=2.0, help="log σ 上界。")
    parser.add_argument("--log_std_init", type=float, default=-0.5,
                        help="log σ 头初值（默认 -0.5 ⇒ σ≈0.61；配 ±1.5 N·m 的动作范围 ⇒ 采样覆盖**整个**动作区间，"
                             "Critic 没有外推区；这是残差 RL 防‘漂到数据外’的关键一课）。")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="写进 Bellman 目标的奖励缩放（1.0 = 原样）。任务书的残差 jerk 口径下单步奖励 ≈ −2.5 "
                             "（W12 量过：换成 τ_exec 口径才是 −22）⇒ 价值尺度 ≈ −250，本来就在合理区间，不用缩放；"
                             "想复现\"不缩放会拖慢 Critic\"的坑可以设 --jerk_on exec 再看这个开关。")
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="全局梯度裁剪（<=0 关闭）。")
    parser.add_argument("--normalize_obs", action="store_true", default=True, help="观测归一化（默认开）。")
    parser.add_argument("--no_normalize_obs", action="store_false", dest="normalize_obs", help="关闭观测归一化。")
    parser.add_argument("--bootstrap_on_truncation", action="store_true", default=True,
                        help="到 500 步截断时继续自举（默认开；周期任务没有真终止）。")
    parser.add_argument("--no_bootstrap_on_truncation", action="store_false", dest="bootstrap_on_truncation",
                        help="把 500 步截断当成真终止（消融，Bellman 目标有偏）。")

    # ================== [W15 新增] 赛程 / 压测 / 消融 / 产物 ==================
    parser.add_argument("--stage", type=str, default="baselines",
                        choices=["baselines", "payload", "friction", "latency", "ablation", "all"],
                        help="跑哪一段赛程：baselines=Table 1（四控制器横向对比）；payload/friction/latency=Figure 3 的三条压测轴；"
                             "ablation=Table 2；all=依次全跑（ablation 默认关，见 --do_ablation）。")
    parser.add_argument("--do_ablation", action="store_true",
                        help="--stage all 时把 ablation 也排进去（默认不排：9 个变体 × 训练预算 ≈ 1.5 h）。")
    parser.add_argument("--controllers", type=str, default="pd_dr,pd_ff_dr,pure_rl,residual_sac",
                        help="--stage baselines 要测的控制器（逗号分隔）。")
    parser.add_argument("--stress_controllers", type=str, default="pd_dr,pd_ff_dr,residual_sac",
                        help="压测轴（payload/friction/latency）上跑哪些控制器（任务书要求 pd_dr / pd_ff_dr / Ours）。")

    # ---- 残差策略权重（Ours）----
    parser.add_argument("--residual_policy_which", choices=["best_ee", "final"], default="best_ee",
                        help="残差策略用哪份权重：best_ee=按末端 RMSE 最好那次（默认，压测用）；final=训练末尾。")
    parser.add_argument("--policy", type=str, default="",
                        help="直接指定残差策略权重路径（默认 logs/w13_residual_sac_ee50k/policy_<which>.pt）。")

    # ---- 纯 RL（Controller 3）----
    parser.add_argument("--pure_rl_policy", type=str, default="",
                        help="纯 RL 策略权重路径（默认 logs/w15_pure_rl/policy_final.pt）。")
    parser.add_argument("--pure_rl_steps", type=int, default=450000,
                        help="纯 RL 的训练预算（环境步；与残差策略**同预算**才谈得上公平对比）。")
    parser.add_argument("--train_pure_rl", action="store_true", default=True,
                        help="纯 RL 权重不存在时自动训练一轮（默认开）。")
    parser.add_argument("--no_train_pure_rl", action="store_false", dest="train_pure_rl",
                        help="纯 RL 权重缺失时直接报错退出（冒烟 / 只想跑基线时用）。")
    parser.add_argument("--pure_rl_action_bound", type=float, default=-1.0,
                        help="纯 RL 的 Actor 动作范围 ±b [N·m]（<0 = 自动取 τ_lim 的最大值 87 N·m）；"
                             "逐关节的**实际**缩放在 -1/1 内按 τ_lim 归一化（见 w15_action_scale）。")
    parser.add_argument("--pure_rl_warmup_frac", type=float, default=0.2,
                        help="纯 RL 热身期随机力矩 = frac × max(τ_lim)[N·m]（默认 0.2 ≈ 17.4；见 train_policy 里的口径说明）。")

    # ---- 压测轴参数 ----
    parser.add_argument("--payload_list", type=str, default="0,1.5,3,4,5",
                        help="载荷压测梯度 [kg]（训练分布只到 3 kg，4/5 是分布外 OOD）。")
    parser.add_argument("--friction_mult", type=str, default="1,1.5,2,3",
                        help="摩擦/阻尼退化倍数（damping 与静摩擦区间同时 ×mult）。")
    parser.add_argument("--latency_steps", type=str, default="0,1,2,3",
                        help="动作下发延迟梯度 [步]（1 步 = 10 ms @100 Hz）。")
    parser.add_argument("--action_delay", type=int, default=0,
                        help="baselines/ablation 赛程用的固定延迟 [步]（0 = 与 W13 逐位相同）。")
    parser.add_argument("--stress_fix", type=str, default="",
                        help="覆盖压测的“钉死”规则，形如 ``payload=1.5,link_mass=-,damping=-``；"
                             "``name=-`` 表示该项保持训练区间（不钉死），不写则用下方默认规则。")
    parser.add_argument("--stress_keep_spread", action="store_true",
                        help="压测时**不**把自变量塌成单点（保留训练区间采样，看分布下的均值/方差）。")

    # ---- 消融（Table 2）----
    parser.add_argument("--ablation_dims", type=str, default="history,res_limit,jerk",
                        help="要跑哪几组消融：history（H=1/3/5）、res_limit（0.5/1.5/5.0 N·m）、jerk（w_j=0）。")
    parser.add_argument("--ablation_steps", type=int, default=0,
                        help="消融变体的训练预算（环境步；0 = 用 --total_steps，保证与主结果同预算）。")
    parser.add_argument("--ablation_reuse_main", action="store_true", default=True,
                        help="H=3 & res_limit=1.5 的格子直接复用主结果权重（默认开，省一轮训练）。")
    parser.add_argument("--no_ablation_reuse_main", action="store_false", dest="ablation_reuse_main",
                        help="消融的每个格子都从零训练（更严格，但多花一轮训练）。")

    # ---- 自检 / 落盘 ----
    parser.add_argument("--verify_w12_path", action="store_true", default=True,
                        help="用 W12 的 run_episode 再跑一遍 pd_dr/pd_ff_dr，与新统一评测循环逐字段对账（默认开）。")
    parser.add_argument("--no_verify_w12_path", action="store_false", dest="verify_w12_path",
                        help="跳过上面这项对账（省 2 次评估的时间）。")
    parser.add_argument("--trace_windows", type=float, default=2.0,
                        help="Figure 1 的时域放大窗口长度 [s]（0 = 不记录轨迹）。")
    parser.add_argument("--trace_env", type=int, default=0, help="轨迹记录哪一台 env（0..num_envs-1）。")
    parser.add_argument("--trace_controllers", type=str, default="pd_dr,pd_ff_dr,residual_sac",
                        help="记录时域轨迹的控制器（Figure 1 用）。")
    parser.add_argument("--run_name", type=str, default="w15_paper_benchmarks", help="日志 / 权重子目录名。")
    parser.add_argument("--eval_every", type=int, default=25000,
                        help="训练时每多少个环境步做一次确定性评估（0=只在最后；纯 RL / 消融训练用）。")
    parser.add_argument("--log_every", type=int, default=5000,
                        help="训练时每多少个环境步打印一次训练日志。")
    parser.add_argument("--trace_every", type=int, default=0,
                        help="W12 路径评估时每多少步打印一次在线指标（0=不打印）。")
    parser.add_argument("--out_json", type=str, default="", help="结果 JSON 落盘路径（默认 docs/results/w15_<stage>.json）。")
    parser.add_argument("--table_csv", type=str, default="",
                        help="论文大表 CSV（默认 docs/results/w15_benchmarks.csv；多赛程**累积**写入，按 (stage,scenario,controller) 去重）。")
    parser.add_argument("--per_env_csv", type=str, default="",
                        help="逐 env 指标 CSV（默认 docs/results/w15_per_env.csv；箱线图用）。")
    parser.add_argument("--append_csv", action="store_true", default=True,
                        help="把本次结果并入已有 CSV（默认开；关掉则覆盖）。")
    parser.add_argument("--no_append_csv", action="store_false", dest="append_csv", help="覆盖写 CSV。")
    parser.add_argument("--max_wall_s", type=float, default=0.0, help="墙钟上限 [s]（>0 时到点安全收尾并照常落盘）。")
    parser.add_argument("--figures_dir", type=str, default="",
                        help="出图目录（默认 docs/figures）。W15 自己不出图，这个路径只写进 JSON，"
                             "给 experiments/export_paper_figures.py 用。")

    # AppLauncher 的标准参数（--device / --headless / --livestream ...）必须挂在解析器上，
    # 否则 `AppLauncher(args_cli)` 拿不到 device/headless（W12/W13 里是同一行调用）。
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _make_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 397-418 行] Kit 起来之后的仿真/算法库 import
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
##
# Kit 起来之后再导入仿真 / 算法相关的库
##
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import clone, configclass  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402

from src.sac import (  # noqa: E402  W9 手写 SAC：网络 / 损失 / 回放池 / 装配工具，一行都没重写
    ReplayBuffer as HostReplayBuffer,
    RunningMeanStd,
    build_sac_agent,
    build_sac_optimizers,
)

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 420-1881 行] W12 引擎（cfg/FK/DR/观测/奖励/基座控制器/MDP/run_episode/scenario）
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
##
# ============================================================================
# [拷贝自 W12] experiments/w12_trajectory_tracking_env.py 的逐字内联拷贝
# ----------------------------------------------------------------------------
# 为什么不 import：W12 脚本在**模块级**就 `AppLauncher(args_cli)` 启动 Kit
# （它的 docstring"与 W11 的关系"一节里写明了 W11→W12 也是这么处理的），
# `import w12_trajectory_tracking_env` 会在导入期再启动一次 Kit。
# 逐字拷贝的收益：① 指标口径（关节 RMSE / 末端 cm / 饱和率）与 W12 同源；
#               ② 可以拿 W12 落盘的 JSON 做数值对账（本脚本训练前会做，见 §5 对账）。
# ============================================================================
##

##
# ---- 资产配置：纯力矩 Franka（k=d=0，把 damping/friction 腾给 DR） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 295-319 行] ----
##
def make_franka_torque_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot", gravity: bool = True) -> ArticulationCfg:
    """把 Franka Panda 改成“纯力矩”配置（与 W10/W11 完全一致，便于三个脚本横向对账）。

    * ``stiffness=damping=0``：``FRANKA_PANDA_CFG`` 默认隐式驱动器（k=80, d=4），3.0 的隐式驱动器算的是
      ``τ = k(q*−q) + d(q̇*−q̇) + τ_ff``。不清零的话，我们写进去的 PD 力矩会额外叠加一个未知 PD 项，
      而且 W11 注入的 **DOF damping 会被速度环掩盖**。清零后 ``τ_applied == τ_cmd``（仅受 effort_limit 限幅）。
    * 保留 ``armature=1e-3``：转子惯量，纯力矩模式下让关节数值更稳。
    * 重力**默认打开**（W11 相反）：本任务关心“未知负载会不会让基线跟踪变差”，而负载的作用方式主要是
      静态重力矩 —— 关掉重力，“挂 3 kg 负载”在慢速轨迹上几乎看不见。
    """
    cfg = clone(FRANKA_PANDA_CFG)
    cfg.prim_path = prim_path
    cfg.actuators = {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"], joint_effort_limit=87.0, stiffness=0.0, damping=0.0, armature=1e-3
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"], joint_effort_limit=12.0, stiffness=0.0, damping=0.0, armature=1e-3
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"], joint_effort_limit=200.0, stiffness=0.0, damping=0.0
        ),
    }
    cfg.spawn.rigid_props.disable_gravity = not gravity
    return cfg

##
# ---- 场景：灯光 + N 台 Franka（可选地面） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 322-334 行] ----
##
@configclass
class FrankaTrackingSceneCfg(InteractiveSceneCfg):
    """灯光 + N 台 Franka（每台挂在 ``/World/envs/env_i/Robot``），地面可用 ``--no_ground`` 关掉。"""

    if not args_cli.no_ground:  # configclass 在类定义期求值，所以这里可以直接用 if
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = make_franka_torque_cfg(gravity=not args_cli.no_gravity)

##
# ---- 名字 → 关节下标（官方 API，失败退回正则） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 340-349 行] ----
##
def _resolve_joint_ids(robot: Articulation, expr: str) -> tuple[list[int], list[str]]:
    try:
        ids, names = robot.find_joints(expr)
        return [int(i) for i in ids], list(names)
    except Exception:  # noqa: BLE001 - 版本差异时退回手工匹配
        import re

        pat = re.compile(expr)
        names = [n for n in robot.joint_names if pat.fullmatch(n)]
        return [robot.joint_names.index(n) for n in names], names

##
# ---- 名字 → 连杆下标 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 352-357 行] ----
##
def _resolve_body_id(robot: Articulation, name: str) -> int:
    try:
        ids, _ = robot.find_bodies(name)
        return int(ids[0])
    except Exception:  # noqa: BLE001
        return robot.body_names.index(name)

##
# ---- 张量 → {mean,max,min} ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 360-368 行] ----
##
def _stat(x: torch.Tensor) -> dict:
    """张量 -> ``{min, mean, max, std}``（搬回 host 标量，仅供打印/落盘）。"""
    x = x.reshape(-1).to(dtype=torch.float32)
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std().item()),
    }

##
# ---- 统计量 → 一行文本 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 371-375 行] ----
##
def _fmt_stat(d: dict | None, unit: str = "") -> str:
    """把 :func:`_stat` 的 dict 打成一行字。"""
    if d is None:
        return "-"
    return f"min {d['min']:.4g} / mean {d['mean']:.4g} / max {d['max']:.4g} / std {d['std']:.4g}{unit}"

##
# ---- 手动增益 → 张量 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 378-385 行] ----
##
def _gain_tensor(vals, num_joints: int, device: str) -> torch.Tensor:
    """把 ``--kp/--kd``（1 个值或 J 个值）变成 ``(1, J)`` 的增益张量。"""
    v = torch.tensor(list(vals), dtype=torch.float32, device=device)
    if v.numel() == 1:
        v = v.expand(num_joints)
    if v.numel() != num_joints:
        raise SystemExit(f"增益必须是 1 个值或 {num_joints} 个值，收到 {v.numel()} 个")
    return v.reshape(1, num_joints)

##
# ---- 增益 → 简报 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 388-393 行] ----
##
def _gain_brief(g: torch.Tensor) -> list[float]:
    """增益的简报（矩阵增益取对角元的跨 env 均值），供日志/落盘用。"""
    if g.dim() == 3:
        J = g.shape[1]
        return [round(float(v), 3) for v in g[:, range(J), range(J)].mean(0).tolist()]
    return [round(float(v), 3) for v in g.reshape(-1).tolist()]

##
# ---- 从 M(q0) 自动定 PD 增益（矩阵/对角两种口径） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 396-430 行] ----
##
def auto_gains(robot: Articulation, joint_ids: list[int], wn: float, zeta: float, device: str, mweighted: bool = False):
    """从 PhysX 的广义质量矩阵 ``M(q)`` 自动定 PD 增益。

    **两种口径**（``--gains auto`` / ``--gains mweighted``）：

    ============  ============================  ==================================================
    口径           增益                           为什么
    ============  ============================  ==================================================
    ``auto``       ``kp_j = I_jj·ωn²``（对角）    简单、最像"教科书 PD"；但 M(q) 不是对角阵，
                   ``kd_j = 2ζωn·I_jj``           闭环稳定性由 ``M⁻¹Kp``/``M⁻¹Kd`` 的**特征值**决定，
                                                  对角增益在 100 Hz 下可用带宽被压到 ωn ≲ 6 rad/s
    ``mweighted``  ``Kp = ωn²·M̂``（矩阵）        用标称惯量矩阵把误差"映射成力矩"（计算力矩法的 PD 部分），
                   ``Kd = 2ζωn·M̂``              此时 ``M⁻¹Kp = ωn²I``：各模态解耦，带宽可以开到 ωn ≳ 30 rad/s
    ============  ============================  ==================================================

    为什么必须自动定（不能手填一组 kp/kd）：Franka 的关节等效惯量跨两个数量级 ——
    肩关节 ~1 kg·m²、手腕关节只有 ~1e-2 kg·m²。同一组标量增益用在两者上，要么肩关节软到扛不住重力，
    要么手腕在 100 Hz 下**数值发散**（本脚本实测：``--gains auto --wn 10`` 时
    ``λmax(M⁻¹Kd)·dt = 3.13 > 2``，关节直接在力矩上限之间颤振）。

    ``M̂`` 取**注入前**读到的标称 ``M(q0)``（控制器自己以为的模型）—— 注入失配后这个模型就错了，
    于是"模型失配"会直接体现在跟踪误差上，这正是要量化的东西。

    Returns:
        ``(Kp, Kd, i_diag)``：``mweighted=True`` 时 ``Kp/Kd`` 是 ``(E,J,J)`` 矩阵，否则是 ``(1,J)``。
    """
    Mm = robot.data.mass_matrix.torch.to(dtype=torch.float32)
    nb = int(getattr(robot, "num_base_dofs", 0))
    M_j = Mm[:, nb:, nb:][:, joint_ids][:, :, joint_ids]                  # (E, J, J)
    i_diag = M_j.diagonal(dim1=-2, dim2=-1).mean(dim=0)                   # (J,) 跨 env 平均
    if mweighted:
        return (float(wn) ** 2) * M_j, (2.0 * float(zeta) * float(wn)) * M_j, i_diag
    kp = (float(wn) ** 2) * i_diag
    kd = 2.0 * float(zeta) * float(wn) * i_diag
    return kp.reshape(1, -1).to(device), kd.reshape(1, -1).to(device), i_diag

##
# ---- 多变量离散稳定裕度 λmax(M⁻¹Kp)·dt²、λmax(M⁻¹Kd)·dt ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 433-450 行] ----
##
def stability_margins(robot: Articulation, joint_ids: list[int], Kp, Kd, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """多变量离散稳定裕度：``λmax(M⁻¹Kp)·dt²`` 与 ``λmax(M⁻¹Kd)·dt``（稳定分别要 < 4 与 < 2）。

    单关节条件（``ωn·dt<2``、``kd·dt/I<2``）只是**必要**条件；M(q) 有强耦合时，
    真正决定稳定的是 ``M⁻¹Kp`` / ``M⁻¹Kd`` 的特征值 —— 脚本把这两个数直接算出来，
    免得"手填一组看着合理的 kp/kd，实际在力矩上限之间颤振"（日志里表现为 sat 高 + jerk 大）。
    """
    Mm = robot.data.mass_matrix.torch.to(dtype=torch.float32)
    nb = int(getattr(robot, "num_base_dofs", 0))
    M_j = Mm[:, nb:, nb:][:, joint_ids][:, :, joint_ids]
    M_inv = torch.linalg.inv(M_j)
    E, J = M_inv.shape[0], M_inv.shape[1]
    eye = torch.eye(J, device=M_inv.device).unsqueeze(0).expand(E, J, J)
    Kp_full = Kp if Kp.dim() == 3 else Kp.reshape(1, J, 1) * eye
    Kd_full = Kd if Kd.dim() == 3 else Kd.reshape(1, J, 1) * eye
    lam_p = torch.linalg.eigvals(M_inv @ Kp_full).abs().amax(dim=-1) * dt * dt
    lam_d = torch.linalg.eigvals(M_inv @ Kd_full).abs().amax(dim=-1) * dt
    return lam_p, lam_d

##
# ---- 模块 1：复合正弦参考轨迹（解析导数） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 454-602 行] ----
##
class TrajectoryGenerator:
    """关节空间复合正弦参考轨迹，解析给出位置 / 速度 / 加速度。

    .. code-block::

        q_{d,i}(t) = q_{0,i} + Σ_{h=1..K} A_{i,h} · sin(2π f_{i,h} t + φ_{i,h})
        q̇_{d,i}(t) = Σ_h A_{i,h} ω_{i,h} · cos(...)          ω = 2π f
        q̈_{d,i}(t) = − Σ_h A_{i,h} ω_{i,h}² · sin(...)

    三个设计要点：

    1. **逐关节独立采样** ``(A_i, f_i, φ_i)``：7 个关节的频率互不相同 → 这是“多轴连续插补运动”的核心，
       不会退化成整臂同相位摆动（那会让所有关节的耦合误差同相叠加，指标更好看但不真实）。
    2. **K=1 时严格等于任务书公式**；``--harmonics K`` 只是额外叠加 2f,3f,… 且幅值按 ``1/h`` 衰减，
       用来制造更“宽频”的参考（对未建模摩擦的激励更强）。
    3. **幅值按关节限位 clamp**（``--traj_margin``）：保证 ``q0 ± Σ_h A_h`` 离限位至少还有 20% 余量，
       否则参考轨迹自己撞限位，跟踪 RMSE 就变成“限位饱和”而不是“控制质量”。
       限位检查由 :meth:`check_limits` 在整个 episode 网格上实测（不是估的）。

    **所有 env 共享同一条参考轨迹**（同一份 A/f/φ）：这样跨 env 的误差差异只能来自动力学差异
    （W11 注入的失配），这正是要量化的东西。
    """

    def __init__(
        self,
        num_envs: int,
        arm_joint_names: list[str],
        joint_limits: torch.Tensor,          # (J, 2) [lo, hi]
        center,                              # (J,) 或 None（None → 取限位中点）
        amp_range: tuple[float, float],
        freq_range: tuple[float, float],
        harmonics: int,
        margin: float,
        seed: int,
        device: str,
    ) -> None:
        self.device = device
        self.num_envs = int(num_envs)
        self.joint_names = list(arm_joint_names)
        self.J = len(self.joint_names)
        self.harmonics = max(1, int(harmonics))
        # 采样全程在 CPU 上做（固定种子 → 逐位可复现），最后再搬到 device
        lim = joint_limits.detach().to(dtype=torch.float32, device="cpu")
        lo, hi = lim[:, 0], lim[:, 1]

        # ---- 轨迹中心 q0：默认取限位中点（关节 2/4/6 的限位不对称，中点更安全）----
        if center is None:
            q0 = 0.5 * (lo + hi)
        else:
            q0 = torch.tensor(list(center), dtype=torch.float32)
            if q0.numel() != self.J:
                raise SystemExit(f"--traj_center 需要 {self.J} 个值，收到 {q0.numel()} 个")
        q0 = torch.clamp(q0, min=lo + 1e-3, max=hi - 1e-3)

        # ---- 逐关节采样：幅值 / 频率 / 相位（CPU 固定种子 → 可复现）----
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

        def _u(rng_range, shape):
            a, b = float(rng_range[0]), float(rng_range[1])
            return torch.rand(shape, generator=gen, dtype=torch.float32) * (b - a) + a

        amp0 = _u(amp_range, (self.J,))
        freq0 = _u(freq_range, (self.J,))
        phase0 = torch.rand(self.J, generator=gen, dtype=torch.float32) * (2.0 * math.pi)

        # ---- 幅值安全 clamp：Σ_h A_h ≤ margin · min(q0−lo, hi−q0) ----
        room = torch.minimum(q0 - lo, hi - q0)
        hsum = sum(1.0 / h for h in range(1, self.harmonics + 1))
        amp_cap = (float(margin) * room / hsum).clamp(min=0.0)
        amp_clamped = torch.minimum(amp0, amp_cap)
        # ---- 展开成 K 个谐波（第 h 项：幅值 A/h、频率 h·f、相位 φ + (h−1)·0.7）----
        A, om, ph = [], [], []
        for h in range(1, self.harmonics + 1):
            A.append(amp_clamped / h)
            om.append(2.0 * math.pi * freq0 * h)
            ph.append(phase0 + (h - 1) * 0.7)
        # (K, 1, J) —— 中间维留给未来“每 env 不同轨迹”，现在恒为 1（所有 env 同一条参考）
        self.A = torch.stack(A).reshape(self.harmonics, 1, self.J).to(device)
        self.omega = torch.stack(om).reshape(self.harmonics, 1, self.J).to(device)
        self.phase = torch.stack(ph).reshape(self.harmonics, 1, self.J).to(device)
        self.Aw = self.A * self.omega
        self.Aw2 = self.A * self.omega * self.omega
        self.q0 = q0.reshape(1, self.J).to(device)
        self.freq = freq0.tolist()
        self.amp = amp_clamped.tolist()
        self.amp_raw = amp0.tolist()
        self.phase_rad = phase0.tolist()
        self.limits = lim.to(device)

        # ---- 预分配显存（compute 里全是原地 op，无 host 往返）----
        E = self.num_envs
        self._arg = torch.empty((self.harmonics, 1, self.J), device=device)
        self._sin = torch.empty_like(self._arg)
        self._cos = torch.empty_like(self._arg)
        self._tmp = torch.empty_like(self._arg)
        self._q = torch.empty((E, self.J), device=device)
        self._qd = torch.empty_like(self._q)
        self._qdd = torch.empty_like(self._q)

    # ---------------- 求值 ----------------
    def compute(self, t: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """原地算出 ``q_d / q̇_d / q̈_d``（形状 ``(E, J)``）；返回的是**同一批缓冲**，别跨步保存。"""
        self._arg.copy_(self.phase)
        self._arg.add_(self.omega * t)
        torch.sin(self._arg, out=self._sin)
        torch.cos(self._arg, out=self._cos)

        self._tmp.copy_(self._sin).mul_(self.A)
        self._q.copy_(self._tmp.sum(dim=0)).add_(self.q0)          # (1,J) → 广播到 (E,J)

        self._tmp.copy_(self._cos).mul_(self.Aw)
        self._qd.copy_(self._tmp.sum(dim=0))                       # (1,J) → (E,J)

        self._tmp.copy_(self._sin).mul_(self.Aw2)
        self._qdd.copy_(self._tmp.sum(dim=0)).neg_()
        return self._q, self._qd, self._qdd

    def check_limits(self, steps: int, dt: float) -> dict:
        """在整个 episode 网格上实测参考轨迹是否越限（证明“幅值 clamp 有效”）。"""
        worst_q, worst_v, worst_a, margin_min = 0.0, 0.0, 0.0, 1e9
        for k in range(int(steps)):
            q, qd, qdd = self.compute(k * dt)
            worst_q = max(worst_q, float(q[0].abs().max()))
            worst_v = max(worst_v, float(qd[0].abs().max()))
            worst_a = max(worst_a, float(qdd[0].abs().max()))
            margin = torch.minimum(q[0] - self.limits[:, 0], self.limits[:, 1] - q[0])
            margin_min = min(margin_min, float(margin.min()))
        self.compute(0.0)
        return {
            "max_abs_qd_rad": worst_q,
            "max_abs_qd_dot_rad_s": worst_v,
            "max_abs_qd_ddot_rad_s2": worst_a,
            "min_margin_to_joint_limit_rad": margin_min,
            "inside_limits": bool(margin_min > 0.0),
        }

    def summary(self) -> dict:
        """逐关节参数表（报告里直接引用）+ 幅值 clamp 前后的对比。"""
        return {
            "formula": "q_d,i(t) = q0,i + sum_h A_i,h * sin(2*pi*f_i,h*t + phi_i,h)",
            "joint_names": self.joint_names,
            "harmonics": self.harmonics,
            "center_q0_rad": [round(float(v), 6) for v in self.q0[0].tolist()],
            "amp_rad_sampled": [round(float(v), 6) for v in self.amp_raw],
            "amp_rad_clamped": [round(float(v), 6) for v in self.amp],
            "freq_hz": [round(float(v), 6) for v in self.freq],
            "phase_rad": [round(float(v), 6) for v in self.phase_rad],
            "amp_clamp_ratio": [round(c / max(r, 1e-9), 4) for c, r in zip(self.amp, self.amp_raw)],
        }

##
# ---- 内置 Franka 运动链常量（URDF 找不到时的兜底） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 609-619 行] ----
##
_PANDA_CHAIN_BUILTIN: tuple = (
    ("panda_joint1", "panda_link0", "panda_link1", (0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint2", "panda_link1", "panda_link2", (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint3", "panda_link2", "panda_link3", (0.0, -0.316, 0.0), (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint4", "panda_link3", "panda_link4", (0.0825, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint5", "panda_link4", "panda_link5", (-0.0825, 0.384, 0.0), (-math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint6", "panda_link5", "panda_link6", (0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint7", "panda_link6", "panda_link7", (0.088, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), (0.0, 0.0, 1.0), "revolute"),
    ("panda_joint8", "panda_link7", "panda_link8", (0.0, 0.0, 0.107), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), "fixed"),
    ("panda_hand_joint", "panda_link8", "panda_hand", (0.0, 0.0, 0.0), (0.0, 0.0, -math.pi / 4), (0.0, 0.0, 1.0), "fixed"),
)

##
# ---- 内置链的字段名 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 621-621 行] ----
##
_PANDA_CHAIN_KEYS = ("name", "parent", "child", "xyz", "rpy", "axis", "type")

##
# ---- URDF 向量解析 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 624-628 行] ----
##
def _vec3(text, default) -> list[float]:
    """URDF 属性的 ``"x y z"`` → ``[x, y, z]``（缺失时用 default）。"""
    if text is None:
        return [float(v) for v in default]
    return [float(v) for v in text.split()]

##
# ---- 在 isaaclab 包里找 lula_franka_gen.urdf ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 631-642 行] ----
##
def find_franka_urdf() -> str | None:
    """在 isaaclab 包里找 ``lula_franka_gen.urdf``（Isaac Lab 自己的 Franka 运动学模型）。"""
    try:
        import isaaclab

        hits = glob.glob(os.path.join(os.path.dirname(isaaclab.__file__), "**", "lula_franka_gen.urdf"), recursive=True)
        for h in sorted(hits):
            if os.path.isfile(h):
                return h
    except Exception:  # noqa: BLE001
        pass
    return None

##
# ---- 批量 FK + 重力矩（末端误差与标称前馈的度量基础） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 644-876 行] ----
##
class PandaFK:
    """在 ``panda_link0 → ee_body`` 这条链上做**批量 FK**（纯 torch，全部在显存里完成）。

    约定（URDF 标准，与 PhysX 一致）：

    .. code-block::

        T_child_world = T_parent_world · T_origin · R(axis, q)

    ``T_origin`` 是关节在**父连杆系**下的固定变换，关节转角绕**子连杆系**里的 ``axis`` 旋转。

    为什么要自己写：Isaac Lab 3.0 的 PhysX 后端没有“给定关节角求连杆位姿”的接口
    （``get_link_transforms`` 只能拿当前状态）。而跟踪误差必须在 ``q`` 与 ``q_d`` 两个不同关节配置上
    求末端位置，所以需要一个**离线可求值**的 FK。

    为什么可信：:meth:`check` 每一步都拿 ``FK(q_measured)`` 与 PhysX 自己算的 ``body_link_pos_w``
    逐连杆对比，残差与 FK 来源一起落盘。残差 ≈ 1e-7 m 说明这个 FK 就是仿真里的那条链，
    而不是“看起来差不多的另一条链”。
    """

    def __init__(self, robot: Articulation, device: str, ee_body: str = "panda_hand", urdf_path: str | None = None):
        self.device = device
        self.ee_body = ee_body
        # ---- 取链：优先 URDF；解析不出来就退回内置常量（来源会打印/落盘，不静默）----
        chain = [dict(zip(_PANDA_CHAIN_KEYS, row)) for row in _PANDA_CHAIN_BUILTIN]
        self.source = "builtin_constants"
        path = urdf_path or find_franka_urdf()
        if path and os.path.isfile(path):
            try:
                parsed = self._parse_urdf(path, base_link="panda_link0", target=ee_body)
            except Exception as exc:  # noqa: BLE001
                parsed = None
                print(f"[FK][WARN] 解析 {path} 失败（{type(exc).__name__}: {exc}），退回内置常量")
            if parsed:
                chain, self.source = parsed, path
        # 截断到末端连杆
        hits = [i for i, s in enumerate(chain) if s["child"] == ee_body]
        if hits:
            chain = chain[: hits[0] + 1]
        elif chain[-1]["child"] != ee_body:
            print(f"[FK][WARN] 链末端是 {chain[-1]['child']}，与 --ee_body {ee_body} 不一致，仍按链末端计算")

        self.chain = chain
        self.num_steps = len(chain)
        names_dof, names_body = list(robot.joint_names), list(robot.body_names)
        self.dof_of_step = [names_dof.index(s["name"]) if s["name"] in names_dof else -1 for s in chain]
        self.body_of_step = [names_body.index(s["child"]) if s["child"] in names_body else -1 for s in chain]
        self.is_revolute = [s["type"] != "fixed" and self.dof_of_step[k] >= 0 for k, s in enumerate(chain)]
        self.joint_of_step = [s["name"] for s in chain]
        self.link_of_step = [s["child"] for s in chain]

        # ---- 预计算 origin 齐次矩阵 (L,4,4) 与关节轴 ----
        self.T_origin = torch.stack([self._origin_matrix(s["xyz"], s["rpy"]) for s in chain]).to(device)
        self.axis = torch.tensor([s["axis"] for s in chain], dtype=torch.float32, device=device)

    # ---------------- 解析 ----------------
    @staticmethod
    def _parse_urdf(path: str, base_link: str, target: str) -> list[dict] | None:
        """解析 URDF，返回 ``base_link → target`` 的唯一链（找不到返回 None）。"""
        root = ET.parse(path).getroot()
        by_parent: dict[str, list[dict]] = {}
        for j in root.findall("joint"):
            p, c = j.find("parent"), j.find("child")
            if p is None or c is None:
                continue
            origin, axis_el = j.find("origin"), j.find("axis")
            by_parent.setdefault(p.get("link"), []).append(
                dict(
                    name=j.get("name"),
                    parent=p.get("link"),
                    child=c.get("link"),
                    xyz=_vec3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                    rpy=_vec3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                    axis=_vec3(axis_el.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
                    type=(j.get("type") or "fixed"),
                )
            )

        def dfs(link: str, path_: list[dict], seen: frozenset) -> list[dict] | None:
            if link == target:
                return path_
            if link in seen:
                return None
            seen = seen | {link}
            for j in by_parent.get(link, []):
                res = dfs(j["child"], path_ + [j], seen)
                if res is not None:
                    return res
            return None

        return dfs(base_link, [], frozenset())

    @staticmethod
    def _origin_matrix(xyz, rpy) -> torch.Tensor:
        """URDF ``origin(xyz, rpy)`` → 4×4 齐次矩阵（rpy 是 XYZ 外旋约定，与 ``quat_from_euler_xyz`` 一致）。"""
        rpy_t = torch.tensor(rpy, dtype=torch.float32)
        quat = math_utils.quat_from_euler_xyz(rpy_t[0], rpy_t[1], rpy_t[2])  # (4,) xyzw
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = math_utils.matrix_from_quat(quat)
        T[:3, 3] = torch.tensor(xyz, dtype=torch.float32)
        return T

    @staticmethod
    def _rot_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rodrigues：绕单位轴 ``axis (3,)`` 转 ``angle (E,)`` → ``(E,4,4)``。"""
        x, y, z = axis[0], axis[1], axis[2]
        c, s = torch.cos(angle), torch.sin(angle)
        C = 1.0 - c
        rows = (
            (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
        )
        R = torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)  # (E,3,3)
        T = torch.zeros((angle.shape[0], 4, 4), device=axis.device, dtype=torch.float32)
        T[:, :3, :3] = R
        T[:, 3, 3] = 1.0
        return T

    # ---------------- 求值 ----------------
    def forward(self, q_dof, base_pos, base_quat, all_links: bool = False) -> torch.Tensor:
        """批量 FK。

        Args:
            q_dof: ``(E, num_joints)`` 全部关节角 [rad]。
            base_pos / base_quat: ``(E,3)`` / ``(E,4)``（xyzw）—— 根连杆 ``panda_link0`` 的**世界位姿**
                （直接从仿真读，因此 FK 结果也在世界系，可与 ``body_link_pos_w`` 直接对账）。
            all_links: True → 返回 ``(E, L, 3)`` 全部链上连杆位置；False → 只返回末端 ``(E, 3)``。

        Returns:
            ``(E, L, 3)`` 或 ``(E, 3)``。
        """
        T = torch.zeros((base_pos.shape[0], 4, 4), device=self.device, dtype=torch.float32)
        T[:, :3, :3] = math_utils.matrix_from_quat(base_quat)
        T[:, :3, 3] = base_pos
        T[:, 3, 3] = 1.0
        pos = (
            torch.empty((base_pos.shape[0], self.num_steps, 3), device=self.device, dtype=torch.float32)
            if all_links
            else None
        )
        for k in range(self.num_steps):
            T = T @ self.T_origin[k]
            if self.is_revolute[k]:
                T = T @ self._rot_axis(self.axis[k], q_dof[:, self.dof_of_step[k]])
            if all_links:
                pos[:, k, :] = T[:, :3, 3]
        return pos if all_links else T[:, :3, 3]

    # ---------------- 自校验（FK vs PhysX 真值） ----------------
    def check(self, q_dof, base_pos, base_quat, measured_pos_w) -> dict:
        """把 ``FK(q_measured)`` 与 PhysX 的 ``body_link_pos_w`` 逐连杆对比（每步调用，代价可忽略）。"""
        pos = self.forward(q_dof, base_pos, base_quat, all_links=True)
        per_link, worst_name, worst = {}, "-", 0.0
        for k, b in enumerate(self.body_of_step):
            if b < 0:
                continue
            err = float((pos[:, k] - measured_pos_w[:, b]).norm(dim=-1).max())
            per_link[self.link_of_step[k]] = round(err, 9)
            if err > worst:
                worst, worst_name = err, self.link_of_step[k]
        return {"source": self.source, "max_err_m": worst, "worst_link": worst_name, "per_link_max_err_m": per_link}

    def describe(self) -> dict:
        """链的构成（报告里用来交代“末端误差是按哪条链算的”）。"""
        return {
            "source": self.source,
            "ee_body": self.ee_body,
            "chain_joint_names": self.joint_of_step,
            "chain_link_names": self.link_of_step,
            "num_steps": self.num_steps,
        }

    # ---------------- 重力矩（虚拟功） ----------------
    def gravity_torque(self, q_dof, base_pos, base_quat, masses, coms, g: float = 9.81) -> torch.Tensor:
        """用同一条链几何 + 给定的连杆质量/质心算重力负载矩 ``G(q)``（形状 ``(E, K)``，K=链上转动关节数）。

        原理（虚拟功 / 势能对关节角求导）：

        .. code-block::

            U(q) = Σ_i m_i · g · z_com,i(q)
            G_j(q) = ∂U/∂q_j = Σ_i m_i · g · (a_j × (p_com,i − o_j))_z

        ``G(q)`` 就是**保持静止需要施加的力矩**（EOM ``M q̈ + C q̇ + G = τ`` 里的重力项，符号与 PhysX 的
        ``gravity_compensation_forces`` 一致）。物理量是“势能对关节角求导”，所以 ``∂U ∂z`` 的符号
        必须在链上逐项核对 —— 脚本用 PhysX 的同名量做对账（同一模型下应逐关节吻合到 ~0.1 N·m）。

        其中 ``a_j`` 是关节 j 的**世界系**转轴、``o_j`` 是它在该时刻的枢轴点、``p_com,i`` 是连杆 i 质心的
        世界位置；``j`` 取遍连杆 i 的**祖先关节**（链式结构下就是 ``j ≤ i``）。

        ``masses (E,B)`` / ``coms (E,B,3)`` 由调用方给（本脚本给的是**注入前**读到的标称值），
        所以这个重力模型**不知道** DR 注入了什么负载 —— 正是我们想要的“控制器自己标称模型的前馈”。
        ``τ_g`` 与 PhysX 的 ``gravity_compensation_forces`` 在同一模型下应当逐关节吻合（脚本会实测）。
        """
        E = base_pos.shape[0]
        T = torch.zeros((E, 4, 4), device=self.device, dtype=torch.float32)
        T[:, :3, :3] = math_utils.matrix_from_quat(base_quat)
        T[:, :3, 3] = base_pos
        T[:, 3, 3] = 1.0

        pivots, axes, pcom, mass_of = [], [], [], []
        rev_idx = []                     # chain 步号 -> 在 τ_g 里的列号
        for k in range(self.num_steps):
            T = T @ self.T_origin[k]
            if self.is_revolute[k]:
                pivots.append(T[:, :3, 3].clone())
                axes.append(T[:, :3, :3] @ self.axis[k])
                rev_idx.append(k)
                T = T @ self._rot_axis(self.axis[k], q_dof[:, self.dof_of_step[k]])
            b = self.body_of_step[k]
            if b >= 0:                    # 这一段的子连杆在 USD 里是有质量的刚体
                c = coms[:, b]
                pcom.append(T[:, :3, 3] + (T[:, :3, :3] @ c.unsqueeze(-1)).squeeze(-1))
                mass_of.append(masses[:, b])
        if not pivots or not pcom:
            return torch.zeros((E, max(1, len(rev_idx))), device=self.device, dtype=torch.float32)

        piv = torch.stack(pivots, dim=1)                      # (E, K, 3)
        ax = torch.stack(axes, dim=1)                         # (E, K, 3)
        pc = torch.stack(pcom, dim=1)                         # (E, L, 3)
        m = torch.stack(mass_of, dim=1)                       # (E, L)
        K, L = piv.shape[1], pc.shape[1]
        # r[j, i] = p_com,i − o_j
        r = pc.unsqueeze(1) - piv.unsqueeze(2)                # (E, K, L, 3)
        az = ax.unsqueeze(2)                                  # (E, K, 1, 3)
        cross_z = az[..., 0] * r[..., 1] - az[..., 1] * r[..., 0]      # (a × r)_z  (E,K,L)
        # 关节 j 只影响链上位于它后面的连杆（mask）
        step_of_rev = torch.tensor(rev_idx, device=self.device)
        bodies_step = torch.tensor([k for k in range(self.num_steps) if self.body_of_step[k] >= 0], device=self.device)
        mask = (step_of_rev.unsqueeze(1) <= bodies_step.unsqueeze(0)).to(torch.float32)   # (K, L)
        tau = (m.unsqueeze(1) * mask.unsqueeze(0) * cross_z).sum(dim=-1) * float(g)       # (E, K) = G(q)
        return tau

##
# ---- W11 域随机化注入器（未知负载 / 摩擦 / 阻尼 / 连杆质量） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 880-1088 行] ----
##
class DynamicsRandomizer:
    """按 env 采样“失配参数”，并通过张量接口写进 PhysX 的**模型属性**。

    ==================  =====================================================  ==========================
    物理量               本脚本调用                                              最终落到 PhysX
    ==================  =====================================================  ==========================
    连杆质量             ``set_masses_index(masses=(E,B))``                      ``set_masses``
    连杆惯量             ``set_inertias_index(inertias=(E,B,9))``               ``set_inertias``
    连杆质心             ``set_coms_index(coms=(E,B,7))``                        ``set_coms``
    关节阻尼             ``write_joint_damping_to_sim_index(damping=(E,J))``    ``set_dof_dampings``
    关节静/动摩擦         ``write_joint_friction_coefficient_to_sim_index(...)``  ``set_dof_friction_properties``
    ==================  =====================================================  ==========================

    三个物理约定（与 W11 一致，写进论文方法节前必须交代清楚）：

    1. **负载合成用平行轴定理**：``m = m_h + m_p``、``c = (m_h c_h + m_p p_p)/m``、
       ``I = I_h + m_h·PA(c_h − c) + m_p·PA(p_p − c)``，``PA(d) = ‖d‖²E₃ − ddᵀ``。
       平行轴项半正定 → 合成惯量一定正定，不会写出非法负惯量。
    2. **连杆质量扰动同时按比例缩放惯量**（``I ← I·s``）：保持回转半径不变；只改质量不动惯量会得到
       “更重但更转不动”的非物理刚体。
    3. **这里写的是物理层 DOF 属性，不是 actuator 的 k/d**：本脚本 actuator 层 ``k=d=0``，
       所以注入的阻尼/摩擦不会被 PD 项掩盖。

    与 W11 的差异（唯一一处）：新增 :meth:`restore_nominal`，能把注入的失配**原样撤回**到标称值，
    于是同一个进程里就能跑“标称 vs 注入”的对照，不必起两次仿真（省掉两次 Kit 启动 ≈ 1 分钟）。
    """

    def __init__(
        self,
        robot: Articulation,
        num_envs: int,
        device: str,
        enabled: bool,
        seed: int,
        joint_expr: str,
        payload_body: str,
        payload_com,
        payload_mode: str,
        ranges: dict,
    ) -> None:
        self.robot, self.device, self.enabled, self.num_envs = robot, device, bool(enabled), int(num_envs)
        self.num_bodies = robot.num_bodies
        self.joint_ids, self.joint_names = _resolve_joint_ids(robot, joint_expr)
        self.hand_id = _resolve_body_id(robot, payload_body)
        self.payload_com = torch.tensor(list(payload_com), dtype=torch.float32, device=device)
        self.payload_mode = payload_mode
        self.ranges = ranges
        # CPU 采样 + 固定种子：同一 --seed 在任何设备上都得到同一组失配参数
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.com_path = "off"
        self.payload_mass = self.mass_scale = self.damping = self.friction = None
        self.nominal_mass = self.nominal_inertia = self.nominal_com = None

    def _uniform(self, shape, rng_range) -> torch.Tensor:
        lo, hi = float(rng_range[0]), float(rng_range[1])
        x = torch.rand(shape, generator=self.gen, dtype=torch.float32)
        return (x * (hi - lo) + lo).to(self.device)

    def sample(self) -> None:
        E, B = self.num_envs, self.num_bodies
        self.mass_scale = self._uniform((E, B), self.ranges["link_mass_scale"])
        self.payload_mass = self._uniform((E,), self.ranges["payload_mass"])
        self.damping = self._uniform((E, len(self.joint_ids)), self.ranges["joint_damping"])
        self.friction = self._uniform((E, len(self.joint_ids)), self.ranges["joint_friction"])

    # ---------------- 注入 ----------------
    def apply(self) -> None:
        """把所有失配一次性写进 PhysX（**必须在 ``sim.reset()`` 之后调用**，否则会被初始化覆盖）。

        采样只发生一次（同一 seed → 同一组参数）；重复调用 ``apply`` 是幂等的（用于
        ``restore_nominal`` 之后重新注入，参数保持不变）。
        """
        if not self.enabled:
            return
        if self.payload_mass is None:
            self.sample()
        robot, E, B, h = self.robot, self.num_envs, self.num_bodies, self.hand_id

        # 注入前的标称值（USD 里的值；同时也是 restore_nominal 的还原目标）
        m0 = robot.data.body_mass.torch.to(dtype=torch.float32).clone()
        I0 = robot.data.body_inertia.torch.to(dtype=torch.float32).clone()
        com0 = robot.data.body_com_pose_b.torch.to(dtype=torch.float32).clone()
        self.nominal_mass, self.nominal_inertia, self.nominal_com = m0, I0, com0
        self.nominal_hand_mass = float(m0[0, h])
        self.nominal_hand_com = com0[:, h, :3].clone()

        # (1) 连杆质量 ×U(LO,HI)：惯量同比例缩放（回转半径不变）
        m = m0 * self.mass_scale
        I = I0 * self.mass_scale.unsqueeze(-1)
        com = com0.clone()

        # (2) 末端未知负载：质量 + 惯量（+ 质心）
        mp = self.payload_mass
        m_h = m[:, h].clone()
        m[:, h] = m_h + mp
        I_h = I[:, h].reshape(E, 3, 3)
        if self.payload_mode == "mass":
            I[:, h] = (I_h * (m[:, h] / m_h)[:, None, None]).reshape(E, 9)
        else:
            I_new, c_new = self._combine_payload(
                I_h, com[:, h, :3].clone(), m_h, mp, self.payload_com.unsqueeze(0).expand(E, 3)
            )
            I[:, h] = I_new.reshape(E, 9)
            com[:, h, :3] = c_new

        robot.set_masses_index(masses=m.contiguous())
        robot.set_inertias_index(inertias=I.contiguous())
        if self.payload_mode == "parallel_axis":
            path = self._write_coms(com.contiguous())
            self.com_path = path or "failed -> fallback_mass"
            if path is None:
                I[:, h] = (I_h * (m[:, h] / m_h)[:, None, None]).reshape(E, 9)
                robot.set_inertias_index(inertias=I.contiguous())
                print("[WARN] set_coms 两条路径都不可用：负载质心未参与动力学，已退化为 payload_mode=mass")
        else:
            self.com_path = "off (mass mode)"

        # (3) 关节阻尼 + 静/动摩擦（物理层 DOF 属性）
        jid = torch.tensor(self.joint_ids, dtype=torch.int32, device=self.device)
        robot.write_joint_damping_to_sim_index(damping=self.damping, joint_ids=jid)
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.friction, joint_dynamic_friction_coeff=0.5 * self.friction, joint_ids=jid
        )
        print(
            f"[DR] 注入完成：关节 {self.joint_names}（{len(self.joint_ids)} 个），负载连杆 = body_id {h}"
            f"，质心写入路径 = {self.com_path}"
        )

    # ---------------- 惯量/质心合成（平行轴定理） ----------------
    @staticmethod
    def _parallel_axis(mass: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """``PA(d) = ‖d‖²·E₃ − d·dᵀ``，形状 ``(E,3,3)``。"""
        eye = torch.eye(3, device=d.device, dtype=d.dtype).expand(d.shape[0], 3, 3)
        return (d * d).sum(dim=-1)[:, None, None] * eye - d.unsqueeze(-1) * d.unsqueeze(-2)

    def _combine_payload(self, I_h, c_h, m_h, m_p, p_p):
        """把“点质量负载”并进手部连杆：返回合成惯量 ``(E,3,3)`` 与合成质心 ``(E,3)``。"""
        m = (m_h + m_p).clamp(min=1e-6)
        c = (m_h[:, None] * c_h + m_p[:, None] * p_p) / m[:, None]
        I = I_h + m_h[:, None, None] * self._parallel_axis(m_h, c_h - c) + m_p[:, None, None] * self._parallel_axis(
            m_p, p_p - c
        )
        return I, c

    def _write_coms(self, com_flat) -> str | None:
        """写质心：优先 ``set_coms_index``（会同步用户侧缓存），失败退回 ``root_view.set_coms``。"""
        try:
            self.robot.set_coms_index(coms=com_flat)
            return "set_coms_index"
        except Exception as exc:  # noqa: BLE001 - 版本/包装差异，退回直写
            first = f"{type(exc).__name__}: {exc}"
        try:
            cpu_env_ids = wp.array(list(range(int(com_flat.shape[0]))), dtype=wp.int32, device="cpu")
            self.robot.root_view.set_coms(
                wp.from_torch(com_flat.cpu().contiguous(), dtype=wp.float32), indices=cpu_env_ids
            )
            return f"root_view.set_coms（set_coms_index 不可用：{first}）"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] set_coms_index 失败（{first}）；root_view.set_coms 也失败（{type(exc).__name__}: {exc}）")
            return None

    # ---------------- 撤回（W12 新增：同一进程里跑“标称”对照） ----------------
    def restore_nominal(self) -> None:
        """把质量/惯量/质心写回标称值、把阻尼与摩擦清零 → 回到“理想无损刚体”。"""
        if self.nominal_mass is None:
            return
        self.robot.set_masses_index(masses=self.nominal_mass.contiguous())
        self.robot.set_inertias_index(inertias=self.nominal_inertia.contiguous())
        if self.payload_mode == "parallel_axis":
            self._write_coms(self.nominal_com.contiguous())
        jid = torch.tensor(self.joint_ids, dtype=torch.int32, device=self.device)
        zero = torch.zeros((self.num_envs, len(self.joint_ids)), dtype=torch.float32, device=self.device)
        self.robot.write_joint_damping_to_sim_index(damping=zero, joint_ids=jid)
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=zero, joint_dynamic_friction_coeff=zero, joint_ids=jid
        )
        print("[DR] 已撤回注入：质量/惯量/质心回到标称，关节阻尼与摩擦归零（“标称动力学”对照组用）")

    # ---------------- 回读与统计 ----------------
    def readback(self) -> dict:
        """从 PhysX 回读模型属性（证明“真的写进去了”，不是只写了个变量）。"""
        robot, h = self.robot, self.hand_id
        m = robot.data.body_mass.torch.to(dtype=torch.float32)
        I = robot.data.body_inertia.torch.to(dtype=torch.float32).reshape(self.num_envs, self.num_bodies, 3, 3)
        com = robot.data.body_com_pose_b.torch.to(dtype=torch.float32)
        nominal_com = getattr(self, "nominal_hand_com", None)
        return {
            "hand_mass_kg": _stat(m[:, h]),
            "hand_inertia_min_eig": _stat(torch.linalg.eigvalsh(I[:, h]).min(dim=-1).values),
            "hand_com_shift_m": _stat((com[:, h, :3] - nominal_com).norm(dim=-1)) if nominal_com is not None else None,
            "joint_damping_nms_per_rad": _stat(robot.data.joint_damping.torch.to(dtype=torch.float32)[:, self.joint_ids]),
            "joint_friction_static_nm": _stat(
                robot.data.joint_friction_coeff.torch.to(dtype=torch.float32)[:, self.joint_ids]
            ),
            "joint_friction_dynamic_nm": _stat(
                robot.data.joint_dynamic_friction_coeff.torch.to(dtype=torch.float32)[:, self.joint_ids]
            ),
            "hand_mass_nominal_kg": getattr(self, "nominal_hand_mass", None),
        }

    def sampled_summary(self) -> dict:
        if not self.enabled or self.payload_mass is None:
            return {}
        return {
            "payload_mass_kg": _stat(self.payload_mass),
            "link_mass_scale": _stat(self.mass_scale),
            "joint_damping_nms_per_rad": _stat(self.damping),
            "joint_friction_static_nm": _stat(self.friction),
        }

##
# ---- 模块 5：单帧 s_t = [e_q, ė_q, q_d, q̇_d, τ_{t−1}] + H 帧历史 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1092-1170 行] ----
##
class ObservationBuilder:
    """标准 MDP 观测：单帧 ``s_t = [e_q, ė_q, q_d, q̇_d, τ_{t−1}]``，外加 H 帧历史拼接。

    维度切分（J = 臂关节数 = 7，H = 历史帧数 = 3）：

    ======  ======================  ==========  ====================================================
    切片     含义                     维度         说明
    ======  ======================  ==========  ====================================================
    0:J      ``e_q = q − q_d``       J           位置跟踪误差 [rad]
    J:2J     ``ė_q = q̇ − q̇_d``      J           速度跟踪误差 [rad/s]
    2J:3J    ``q_d``                 J           期望关节位置（告诉策略“现在该在哪一段”）
    3J:4J    ``q̇_d``                J           期望关节速度（前馈信息，没有它策略得自己差分）
    4J:5J    ``τ_{t−1}``             J           上一步力矩（隐式推断未建模摩擦/外载 + 抑制抖动）
    ======  ======================  ==========  ====================================================

    历史拼接顺序 **旧 → 新**：``obs = [s_{t−H+1}, …, s_{t−1}, s_t]``（最后一帧是当前帧）。

    warm-up（episode 开始时历史还没填满）默认 ``repeat``：用首帧重复填满。为什么不用零：
    零帧会被策略误读成“误差为 0 的完美状态”，而 ``repeat`` 对应“一直在原地不动”的自然语义。

    实现上是 **H 个预分配缓冲的环形队列**：稳态下每步只有一次 ``copy_`` + 一次 ``cat``，没有累积分配。
    """

    KEYS = ("e_q", "e_dot_q", "q_d", "qd_d", "tau_prev")

    def __init__(self, num_envs: int, num_joints: int, history: int, warmup: str, device: str) -> None:
        self.num_envs, self.J = int(num_envs), int(num_joints)
        self.H = max(1, int(history))
        self.warmup = warmup
        self.frame_dim = 5 * self.J
        self.obs_dim = self.H * self.frame_dim
        self.device = device
        self._buf = [torch.zeros((self.num_envs, self.frame_dim), device=device) for _ in range(self.H)]
        self._frame = torch.zeros((self.num_envs, self.frame_dim), device=device)
        self._obs = torch.zeros((self.num_envs, self.obs_dim), device=device)
        self._cursor = 0

    def frame(self, q, qd, q_d, qd_d, tau_prev) -> torch.Tensor:
        """拼出当前单帧观测（写进预分配缓冲，返回的是**同一个**缓冲，别跨步保存）。"""
        J, f = self.J, self._frame
        torch.sub(q, q_d, out=f[:, 0:J])            # e_q
        torch.sub(qd, qd_d, out=f[:, J : 2 * J])    # ė_q
        f[:, 2 * J : 3 * J].copy_(q_d)              # q_d
        f[:, 3 * J : 4 * J].copy_(qd_d)             # q̇_d
        f[:, 4 * J : 5 * J].copy_(tau_prev)         # τ_{t−1}
        return f

    def push(self, frame: torch.Tensor) -> None:
        """把当前帧压进环形缓冲（覆盖最旧的一帧）。"""
        self._buf[self._cursor].copy_(frame)
        self._cursor = (self._cursor + 1) % self.H

    def reset(self, first_frame: torch.Tensor) -> None:
        """episode 开始：按 warm-up 策略初始化历史缓冲。"""
        if self.warmup == "repeat":
            for b in self._buf:
                b.copy_(first_frame)
        else:
            for b in self._buf:
                b.zero_()
        self._cursor = 0

    def observation(self) -> torch.Tensor:
        """拼出 H 帧历史观测（旧 → 新），形状 ``(E, H·5J)``。"""
        order = [(self._cursor + i) % self.H for i in range(self.H)]
        self._obs = torch.cat([self._buf[i] for i in order], dim=-1)
        return self._obs

    def describe(self) -> dict:
        J = self.J
        return {
            "frame_layout": {k: [i * J, (i + 1) * J] for i, k in enumerate(self.KEYS)},
            "frame_dim": self.frame_dim,
            "history": self.H,
            "history_order": "oldest -> newest（拼接后最后一帧是当前帧）",
            "warmup": self.warmup,
            "obs_dim": self.obs_dim,
            "formula": "s_t = [e_q, e_dot_q, q_d, qd_d, tau_prev]",
        }

##
# ---- 模块 6：四项复合奖励（W12 口径：jerk 项用 τ_exec） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1174-1212 行] ----
##
class MultiObjectiveReward:
    """``r_t = − w_p‖e_q‖² − w_v‖ė_q‖² − w_τ‖τ_t‖² − w_j‖τ_t − τ_{t−1}‖²``。

    为什么把四项**分别**统计，而不是只报一个总和：

    * ``pos``：任务指标本身。只优化它 → 力矩抖振、能耗爆炸（真机上会磨减速器）；
    * ``vel``：参考速度大时惩罚“相位滞后”（纯位置项对滞后不敏感）；
    * ``tau``：压制“大扭矩硬怼”，让策略学会利用动力学而不是和它对抗；
    * ``jerk``：力矩变化率，压制抖振，直接对应执行器磨损/噪声。

    训练时只看总回报，四个权重里任何一个跑偏都看不出来；分开统计后，日志里能一眼分辨
    “这一轮是精度涨了，还是只把力矩压小了”。

    落盘里的 ``terms`` 是**被减掉的那一项的量值**（都是正数），符号由公式里的负号显式给出。
    """

    KEYS = ("pos", "vel", "tau", "jerk")

    def __init__(self, weights: dict, device: str) -> None:
        self.device = device
        self.w = {k: float(weights[k]) for k in self.KEYS}

    def compute(self, e_q: torch.Tensor, e_dot_q: torch.Tensor, tau: torch.Tensor, tau_prev: torch.Tensor):
        """返回 ``(reward (E,), terms {name: (E,)})``。"""
        pos = self.w["pos"] * (e_q * e_q).sum(dim=-1)
        vel = self.w["vel"] * (e_dot_q * e_dot_q).sum(dim=-1)
        mag = self.w["tau"] * (tau * tau).sum(dim=-1)
        dtau = tau - tau_prev
        jerk = self.w["jerk"] * (dtau * dtau).sum(dim=-1)
        terms = {"pos": pos, "vel": vel, "tau": mag, "jerk": jerk}
        reward = -(pos + vel + mag + jerk)
        return reward, terms

    def describe(self) -> dict:
        return {
            "weights": dict(self.w),
            "formula": "r = -wp||e_q||^2 - wv||e_dot_q||^2 - wtau||tau||^2 - wj||tau_t - tau_{t-1}||^2",
            "note": "落盘 terms 是各项被减掉的量值（正数），reward = -sum(terms)",
        }

##
# ---- 模块 7：开环 PD(+前馈) 基线控制器（含符号口径与限幅） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1217-1316 行] ----
##
class BaselineController:
    """任务书要求的开环基线：``τ_base = K_p·e_q + K_d·ė_q``（可选叠加重力补偿前馈）。

    **符号约定（必须交代清楚，否则“PD”会变成正反馈）**

    任务书给的观测定义是 ``e_q = q − q_d``，同时写 ``τ_base = K_p e_q + K_d ė_q``。
    把 ``e_q = q − q_d`` 代进去，这个字面写法等价于 ``τ = K_p(q − q_d) + K_d(q̇ − q̇_d)``：
    关节掉到目标下方时 ``e_q < 0``，它给出**负**力矩，把关节推得更低 —— **正反馈**。
    本脚本实测过这一点（``--pd_sign literal`` 复现）：静态参考 + 关重力时误差恒为 0 看不出来，
    一旦打开重力，误差 1 s 内涨到 4 rad、98% 的步都在力矩上限（详见报告 §4）。

    所以默认走**标准 PD 口径**（``--pd_sign stabilizing``）：

    .. code-block::

        τ = K_p·(q_d − q) + K_d·(q̇_d − q̇) = −(K_p·e_q + K_d·ė_q)

    两者只差一个负号，但一个收敛、一个发散。奖励里的 ``‖e_q‖²`` 与符号无关，观测也照原样给 ``e_q``，
    所以这个修正只影响“控制器怎么用这个误差”，不影响观测/奖励的定义。

    另外三个把关细节：

    1. **力矩按关节物理上限 clamp**（``joint_effort_limits``）：否则“算出来的力矩”和“真机给得出来的力矩”
       不是一回事，饱和步数会被隐藏。饱和步数与饱和元素占比都会统计并落盘。
    2. **只控制臂关节**：手指关节命令恒 0（本任务不做抓取），但命令缓冲的形状必须是**全部 9 个关节**
       （``set_effort_index(full_data=True)``），所以每步写的是拼好的 ``(E, 9)``。
    3. **重力前馈的符号**：PhysX 的 ``gravity_compensation_forces`` 定义满足 ``M q̈ + C q̇ + g = τ_input``，
       即 ``τ_input = g(q)`` 时 ``q̈ = 0``（Isaac Lab 自带测试
       ``test_get_gravity_compensation_forces_static_equilibrium`` 就是钉这个契约的），所以前馈**直接相加**。
    """

    def __init__(
        self,
        kp: torch.Tensor,
        kd: torch.Tensor,
        num_envs: int,
        num_joints: int,
        device: str,
        mode: str = "pd",
        effort_limits: torch.Tensor | None = None,
        sign: str = "stabilizing",
        ff_source: str = "nominal",
    ) -> None:
        self.kp, self.kd = kp, kd
        self.mode, self.sign, self.ff_source = mode, sign, ff_source
        self.num_envs, self.num_joints = int(num_envs), int(num_joints)
        self.device = device
        self.tau = torch.zeros((self.num_envs, self.num_joints), device=device, dtype=torch.float32)
        self._tau_col = torch.zeros((self.num_envs, self.num_joints, 1), device=device, dtype=torch.float32)
        self.limits = effort_limits if effort_limits is not None else None
        # 统计量（留在显存里累加，最后一次性取回 host，避免每步同步）
        self.sat_steps_any = torch.zeros((), device=device)
        self.sat_entries = torch.zeros((), device=device)
        self.sat_per_joint = torch.zeros((self.num_joints,), device=device)
        self.abs_tau_per_joint = torch.zeros((self.num_joints,), device=device)
        self.steps = 0

    def compute(self, e_q: torch.Tensor, e_dot_q: torch.Tensor, gravity_ff: torch.Tensor | None = None) -> torch.Tensor:
        """算出 ``τ``（原地写进 :attr:`tau`）；``mode == 'zero'`` 时返回全零。"""
        self.steps += 1
        if self.mode == "zero":
            self.tau.zero_()
            return self.tau
        # 先统一算 β = Kp·e_q + Kd·ė_q（对角增益或矩阵增益）
        if self.kp.dim() == 3:
            torch.add(
                torch.bmm(self.kp, e_q.unsqueeze(-1)),
                torch.bmm(self.kd, e_dot_q.unsqueeze(-1)),
                out=self._tau_col,
            )
            self.tau.copy_(self._tau_col.squeeze(-1))
        else:
            self.tau.copy_(e_q).mul_(self.kp).addcmul_(e_dot_q, self.kd)
        if self.sign != "literal":
            self.tau.neg_()          # 标准 PD 口径：τ = −(Kp·e_q + Kd·ė_q)
        # （sign == "literal" 时保留 +β，即任务书的字面写法 —— 正反馈，只用于对照复现）
        if self.mode == "pd_ff" and gravity_ff is not None:
            self.tau.add_(gravity_ff)
        if self.limits is not None:
            over = self.tau.abs() > self.limits
            self.sat_steps_any += over.any(dim=-1).float().sum()
            self.sat_entries += over.float().sum()
            self.sat_per_joint += over.float().sum(dim=0)
            self.tau.clamp_(min=-self.limits, max=self.limits)
        self.abs_tau_per_joint += self.tau.abs().mean(dim=0)
        return self.tau

    def stats(self) -> dict:
        denom = max(1, self.steps * self.num_envs)
        return {
            "mode": self.mode,
            "sign": self.sign,
            "ff_source": self.ff_source,
            "sat_step_frac_any_joint": float(self.sat_steps_any.item()) / denom,
            "sat_entry_frac": float(self.sat_entries.item()) / max(1, self.steps * self.num_envs * self.num_joints),
            "sat_frac_per_joint": [round(float(v) / max(1, self.steps * self.num_envs), 4) for v in self.sat_per_joint],
            "mean_abs_tau_per_joint": [
                round(float(v) / max(1, self.steps), 3) for v in self.abs_tau_per_joint
            ],
        }

##
# ---- 模块 8：轨迹跟踪 MDP（reset / step / FK 自校验） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1320-1523 行] ----
##
class TrajectoryTrackingMDP:
    """机械臂高精度轨迹跟踪 MDP：``obs, reward, terminated, truncated, info = env.step(tau_arm)``。

    每一步的时序（写清楚是为了让 Day 4 的 RL 训练能对齐）：

    1. ``t`` 时刻算参考 ``q_d(t), q̇_d(t), q̈_d(t)``（所有 env 共享同一条参考）；
    2. 用**当前测得**状态拼单帧 ``s_t`` 并压进历史缓冲 → ``obs_t``（H 帧，旧 → 新）；
    3. 外部策略给 ``τ_t``（本脚本用 :class:`BaselineController`，Day 4 换成 RL 策略）；
    4. 写命令 → 推一步物理 → ``scene.update`` → 得到 ``t+dt`` 时刻状态；
    5. 奖励用 ``t+dt`` 的误差 + 本步力矩：
       ``r_t = −w_p‖e_q(t+dt)‖² − w_v‖ė_q(t+dt)‖² − w_τ‖τ_t‖² − w_j‖τ_t − τ_{t−1}‖²``。

    另外每步做一次 **FK 自校验**（``FK(q_measured)`` vs ``body_link_pos_w``）并累积最大残差；
    ``info`` 里给出当步的 ``e_q``、``ė_q``、``ee_err_m`` 与奖励分项，外部统计直接复用。

    ``terminated`` 恒为 False（周期性的非终止任务）；``truncated`` 在 ``t ≥ steps·dt`` 时为 True。
    轨迹跟踪里没有“失败终止”这一说：误差大就体现在奖励和指标上，而不是提前结束 episode。
    """

    def __init__(
        self,
        scene: InteractiveScene,
        sim: SimulationContext,
        robot: Articulation,
        traj: TrajectoryGenerator,
        obs_builder: ObservationBuilder,
        reward_fn: MultiObjectiveReward,
        fk: PandaFK,
        joint_ids: list[int],
        dt: float,
        steps: int,
        fk_check: bool = True,
        init_at_reference: bool = True,
    ) -> None:
        self.scene, self.sim, self.robot = scene, sim, robot
        self.traj, self.obs_builder, self.reward_fn, self.fk = traj, obs_builder, reward_fn, fk
        self.joint_ids = list(joint_ids)
        self.J = len(self.joint_ids)
        self.J_all = int(robot.num_joints)
        self.E = int(traj.num_envs)
        self.dt, self.steps = float(dt), int(steps)
        self.fk_check = bool(fk_check)
        self.init_at_reference = bool(init_at_reference)
        dev = self.traj.device
        self._q_full_tmp = torch.zeros((self.E, self.J_all), device=dev)
        self._e = torch.zeros((self.E, self.J), device=dev)
        self._edot = torch.zeros_like(self._e)
        self._tau_cmd = torch.zeros((self.E, self.J_all), device=dev)   # 全关节命令（手指恒 0）
        self._tau_last = torch.zeros((self.E, self.J), device=dev)      # τ_{t−1}
        self.nominal_masses = None
        self.nominal_coms = None
        self.step_count = 0
        self.fk_err_max = 0.0
        self._fk_report: dict | None = None

    # ---------------- 内部工具 ----------------
    def _state(self):
        """当前臂关节位置 / 速度 ``(E, J)``。"""
        return (
            self.robot.data.joint_pos.torch[:, self.joint_ids],
            self.robot.data.joint_vel.torch[:, self.joint_ids],
        )

    def _write_tau(self, tau_arm: torch.Tensor) -> None:
        """把臂关节力矩拼进 ``(E, J_all)`` 命令缓冲（warp kernel，无 host 往返）。"""
        self._tau_cmd.zero_()
        self._tau_cmd[:, self.joint_ids] = tau_arm
        self.robot.actuators.target_command.set_effort_index(value=self._tau_cmd, full_data=True)

    def _base_pose(self):
        """根连杆（``panda_link0``）的世界位姿 ``(E,3) / (E,4)``。"""
        return (
            self.robot.data.body_link_pos_w.torch[:, 0, :],
            self.robot.data.body_link_quat_w.torch[:, 0, :],
        )

    def _fk_check_once(self) -> dict:
        """FK 自校验：把 ``FK(q_measured)`` 与 PhysX 真值对比（结果累积到 ``fk_err_max``）。"""
        q_full = self.robot.data.joint_pos.torch
        base_pos, base_quat = self._base_pose()
        rep = self.fk.check(q_full, base_pos, base_quat, self.robot.data.body_link_pos_w.torch)
        self.fk_err_max = max(self.fk_err_max, float(rep["max_err_m"]))
        self._fk_report = rep
        return rep

    def ee_position(self, q_arm: torch.Tensor) -> torch.Tensor:
        """用 FK 求给定臂关节角下的末端位置 ``(E,3)``（世界系，基座位姿取当前测量值）。"""
        q_full = self._q_full_tmp
        q_full.copy_(self.robot.data.joint_pos.torch)
        q_full[:, self.joint_ids] = q_arm
        base_pos, base_quat = self._base_pose()
        return self.fk.forward(q_full, base_pos, base_quat)

    def zero_torque_warmup_step(self) -> None:
        """零力矩推一步物理。

        用途：PhysX 的广义质量矩阵 ``M(q)`` 是在 ``simulate()`` 里算的，刚写完模型属性（注入失配）但还没
        推过物理时，读 ``M`` 可能拿到上一次的缓存值。这个"空推一步"只为了让 ``M`` 反映新模型；
        调用方紧接着会 ``reset()``，所以这一步不会污染任何 episode 指标。
        """
        self._tau_cmd.zero_()
        self.robot.actuators.target_command.set_effort_index(value=self._tau_cmd, full_data=True)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)

    def set_nominal_model(self, masses: torch.Tensor, coms: torch.Tensor) -> None:
        """存下**注入前**读到的标称连杆模型（质量 ``(E,B)`` + 质心 ``(E,B,3)``），供标称重力前馈使用。"""
        self.nominal_masses, self.nominal_coms = masses, coms
    def gravity_ff(self, source: str = "nominal", g: float = 9.81) -> torch.Tensor | None:
        """重力补偿前馈 ``(E, J)``。

        * ``nominal``（默认）：用**注入前**的标称质量/质心 + 同一条 FK 链自己算 —— 控制器不知道负载，
          所以负载越大前馈就越不准，这正是要量化的**模型失配**；
        * ``oracle``：直接读 PhysX 的 ``gravity_compensation_forces``（PhysX 按**注入后**的模型算，
          等于“知道真实负载”），只作为**上限参照**。
        """
        if source == "oracle":
            g_all = self.robot.data.gravity_compensation_forces.torch
            return g_all[:, self.robot.num_base_dofs :][:, self.joint_ids]
        if self.nominal_masses is None:
            return None
        base_pos, base_quat = self._base_pose()
        q_full = self._q_full_tmp
        q_full.copy_(self.robot.data.joint_pos.torch)
        return self.fk.gravity_torque(q_full, base_pos, base_quat, self.nominal_masses, self.nominal_coms, g=g)

    def tracking_error(self, t: float):
        """``t`` 时刻的跟踪误差 ``(e_q, ė_q, q_d, q̇_d)``（从**当前测量状态**算，供控制器使用）。

        .. warning::
            后两项 ``q_d / q̇_d`` 是 :class:`TrajectoryGenerator` 的**内部缓冲**，
            下一次调用 ``compute()`` 就会被覆盖 —— 只允许在“本步内”使用，不要跨步保存。
        """
        q_d, qd_d, _ = self.traj.compute(t)
        q, qd = self._state()
        return torch.sub(q, q_d), torch.sub(qd, qd_d), q_d, qd_d

    # ---------------- reset / step ----------------
    def reset(self) -> torch.Tensor:
        """复位：关节状态 → 参考初值（或默认位形），清命令缓冲与历史，返回首帧观测。"""
        q_d, qd_d, _ = self.traj.compute(0.0)
        robot = self.robot
        q_full = robot.data.default_joint_pos.torch.clone()
        qd_full = torch.zeros_like(q_full)
        if self.init_at_reference:
            q_full[:, self.joint_ids] = q_d[:, : self.J]
            qd_full[:, self.joint_ids] = qd_d
        robot.write_joint_state_to_sim_index(position=q_full, velocity=qd_full)
        self.scene.reset()                      # 清 actuator 命令缓冲（不动关节位置）
        self.scene.write_data_to_sim()
        self.scene.update(self.dt)

        self.step_count = 0
        self._tau_last.zero_()
        self.fk_err_max = 0.0
        if self.fk_check:
            self._fk_check_once()
        q, qd = self._state()
        self.obs_builder.reset(self.obs_builder.frame(q, qd, q_d, qd_d, self._tau_last))
        return self.obs_builder.observation()

    def step(self, tau_arm: torch.Tensor):
        """走一步：观测（t）→ 写力矩（t）→ 物理步进 → 奖励（t+dt）。"""
        t = self.step_count * self.dt
        q_d, qd_d, _ = self.traj.compute(t)
        q, qd = self._state()
        torch.sub(q, q_d, out=self._e)
        torch.sub(qd, qd_d, out=self._edot)

        # (1) 观测（含历史）+ 末端误差：都用“写力矩之前”的测量状态
        self.obs_builder.push(self.obs_builder.frame(q, qd, q_d, qd_d, self._tau_last))
        obs = self.obs_builder.observation()
        if self.fk_check:
            self._fk_check_once()
        ee_err = (self.ee_position(q) - self.ee_position(q_d)).norm(dim=-1)

        # (2) 写力矩 + 步进
        self._write_tau(tau_arm)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.step_count += 1

        # (3) 奖励：用 t+dt 的状态 + 本步力矩（jerk 项用 τ_{t−1}）
        q_d2, qd_d2, _ = self.traj.compute(self.step_count * self.dt)
        q2, qd2 = self._state()
        reward, terms = self.reward_fn.compute(q2 - q_d2, qd2 - qd_d2, tau_arm, self._tau_last)
        self._tau_last.copy_(tau_arm)

        info = {"t": t, "e_q": q2 - q_d2, "e_dot_q": qd2 - qd_d2, "ee_err_m": ee_err, "reward_terms": terms}
        return obs, reward, False, self.step_count >= self.steps, info

    def fk_status(self, tolerance: float) -> dict:
        """FK 自校验的总结（``valid`` 决定本轮报告里的末端误差能不能信）。"""
        return {
            "enabled": self.fk_check,
            "source": self.fk.source,
            "max_err_m": self.fk_err_max,
            "tolerance_m": float(tolerance),
            "valid": bool((not self.fk_check) or self.fk_err_max <= float(tolerance)),
            "last_per_link_max_err_m": (self._fk_report or {}).get("per_link_max_err_m", {}),
            "worst_link": (self._fk_report or {}).get("worst_link", "-"),
        }

##
# ---- 模块 9：跨 env 归约 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1527-1534 行] ----
##
def _reduce(t: torch.Tensor) -> dict:
    """跨 env 归约（``(E,)`` 或 ``(E,J)`` → 逐关节 mean/max 或整体统计）。"""
    if t.dim() == 1:
        return {"mean_over_envs": float(t.mean()), "max_over_envs": float(t.max()), "min_over_envs": float(t.min())}
    return {
        "per_joint_mean_over_envs": [float(v) for v in t.mean(dim=0)],
        "per_joint_max_over_envs": [float(v) for v in t.max(dim=0).values],
    }

##
# ---- 模块 9：W12 的 episode 指标累加（baseline 用的就是这一份） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1537-1646 行] ----
##
def run_episode(env: TrajectoryTrackingMDP, controller: BaselineController, tag: str, fk_tol: float,
                eval_skip: int = 0, trace_every: int = 50, joint_names: list[str] | None = None) -> dict:
    """用给定控制器跑一条 episode，返回指标。

    口径先说清楚（否则 RMSE 是笔糊涂账）：

    * 记录的是**每步 ``step()`` 之后**（即 ``t+dt`` 采样时刻）的状态误差，与奖励口径一致；
    * ``joint_rmse`` = ``sqrt(mean_t (q_j(t) − q_{d,j}(t))²)``，先对每个 env 的每个关节算，
      再报「跨 env 的 mean / max」与「逐关节的 mean / max」；
    * 末端笛卡尔误差是从 **FK 自校验通过**的两个位姿算出来的 ``‖p(q) − p(q_d)‖``，单位统一转 cm；
    * ``eval_skip`` 可跳过前 N 步（默认 0 = 全 episode 计入，因为轨迹从 ``q_d(0)`` 起步，没有起步瞬态）。

    所有累加都在显存里进行，只在收尾时取回 host —— 统计本身不拖慢 rollout（吞吐会单独报）。
    """
    E, J, dev = env.E, env.J, env.traj.device
    obs = env.reset()                                            # noqa: F841 - 训练时会用到，这里只驱动时序
    acc_sq, acc_abs = torch.zeros((E, J), device=dev), torch.zeros((E, J), device=dev)
    acc_sq_v = torch.zeros((E, J), device=dev)
    acc_sq_ee, acc_ee = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
    acc_ee_max = torch.zeros(E, device=dev)
    acc_ret = torch.zeros(E, device=dev)
    acc_terms = {k: torch.zeros(E, device=dev) for k in env.reward_fn.KEYS}
    n, trace = 0, []

    t0 = time.perf_counter()
    for k in range(env.steps):
        e, edot, _, _ = env.tracking_error(k * env.dt)
        gravity_ff = None
        if controller.mode == "pd_ff":
            gravity_ff = env.gravity_ff(controller.ff_source)
        tau = controller.compute(e, edot, gravity_ff)

        obs, rew, terminated, truncated, info = env.step(tau)     # noqa: F841
        if k >= eval_skip:
            e2, e2d, ee = info["e_q"], info["e_dot_q"], info["ee_err_m"]
            acc_sq.addcmul_(e2, e2)
            acc_abs.add_(e2.abs())
            acc_sq_v.addcmul_(e2d, e2d)
            acc_sq_ee.addcmul_(ee, ee)
            acc_ee.add_(ee)
            acc_ee_max = torch.maximum(acc_ee_max, ee)
            acc_ret.add_(rew)
            for key, val in info["reward_terms"].items():
                acc_terms[key].add_(val)
            n += 1
        if trace_every > 0 and (k + 1) % trace_every == 0:
            trace.append(
                {
                    "t_s": round(k * env.dt, 4),
                    "joint_rms_rad": float(info["e_q"].pow(2).sum(-1).sqrt().mean()),
                    "ee_cm": float(info["ee_err_m"].mean() * 100.0),
                    "reward_mean": float(rew.mean()),
                }
            )
            print(
                f"  [{tag}] t={k * env.dt:6.3f}s  joint RMS={trace[-1]['joint_rms_rad']:.4f} rad"
                f"  ee={trace[-1]['ee_cm']:.3f} cm  reward={trace[-1]['reward_mean']:.3f}"
            )
    wall = time.perf_counter() - t0

    n = max(1, n)
    rmse_env = torch.sqrt(acc_sq / n)                    # (E,J) 每 env 每关节 RMSE [rad]
    per_env = torch.sqrt(acc_sq.sum(-1) / n)             # (E,)   每 env “全关节 RMSE” [rad]
    mae_env = acc_abs / n
    vel_env = torch.sqrt(acc_sq_v / n)
    ee_rmse_env = torch.sqrt(acc_sq_ee / n)              # (E,) [m]
    ee_mae_env = acc_ee / n
    result = {
        "tag": tag,
        "num_envs": E,
        "steps_counted": n,
        "eval_skip": int(eval_skip),
        "joint_rmse_rad": {
            "mean_over_envs": float(rmse_env.mean()),
            "max_over_envs": float(rmse_env.max()),
            "worst_env_joint": float(per_env.max()),
            "per_joint_mean_over_envs": [float(v) for v in rmse_env.mean(dim=0)],
            "per_joint_max_over_envs": [float(v) for v in rmse_env.max(dim=0).values],
            "joint_names": joint_names,
        },
        "joint_rmse_per_env_rad": _reduce(per_env),
        "joint_mae_rad": _reduce(mae_env),
        "joint_rmse_velocity_rad_s": _reduce(vel_env),
        "cartesian_error_cm": {
            "rmse_mean_over_envs": float(ee_rmse_env.mean() * 100.0),
            "rmse_max_over_envs": float(ee_rmse_env.max() * 100.0),
            "mae_mean_over_envs": float(ee_mae_env.mean() * 100.0),
            "max_over_envs_and_steps": float(acc_ee_max.max() * 100.0),
            "per_env_rmse_cm": [round(float(v) * 100.0, 6) for v in ee_rmse_env],
        },
        "reward": {
            "mean_return_per_env": float(acc_ret.mean()),
            "mean_per_step_per_env": float((acc_ret / n).mean()),
            "terms_per_step_mean": {k: float((v / n).mean()) for k, v in acc_terms.items()},
        },
        "controller": controller.stats(),
        "fk": env.fk_status(fk_tol),
        "wall_time_s": round(wall, 3),
        "env_steps_per_s": round(env.steps * E / max(wall, 1e-9), 1),
        "trace": trace,
    }
    print(
        f"[RESULT] {tag:9s} 关节 RMSE = {result['joint_rmse_rad']['mean_over_envs']:.6f} rad"
        f"（最差 env/关节 {result['joint_rmse_rad']['worst_env_joint']:.6f} rad）"
        f" | 末端 RMSE = {result['cartesian_error_cm']['rmse_mean_over_envs']:.4f} cm"
        f"（最大 {result['cartesian_error_cm']['max_over_envs_and_steps']:.4f} cm）"
        f" | 回报/步 = {result['reward']['mean_per_step_per_env']:.4f}"
        f" | 吞吐 {result['env_steps_per_s']:.0f} env-steps/s"
    )
    return result

##
# ---- 模块 9：scenario 名 →（控制器模式, 是否注入失配, 描述） ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1651-1657 行] ----
##
_SCENARIOS: dict[str, tuple[str, bool, str]] = {
    "pd_nodr": ("pd", False, "PD 基线 + 标称动力学（不注入失配）"),
    "pd_ff_nodr": ("pd_ff", False, "PD + 标称模型重力前馈 + 标称动力学（“前馈+PID”的理想上限）"),
    "pd_dr": ("pd", True, "PD 基线 + 注入失配 —— ★ 任务书要求的基线追踪测试"),
    "pd_ff_dr": ("pd_ff", True, "PD + 重力前馈 + 注入失配（前馈来源见 --ff，默认标称模型）"),
    "zero_dr": ("zero", True, "零力矩（完全不控制）+ 注入失配 —— 误差上界"),
}

##
# ---- 模块 9：DR 采样值 / PhysX 回读值打印 ----
# [拷贝自 W12 · w12_trajectory_tracking_env.py 第 1660-1682 行] ----
##
def _print_dr(info: dict) -> None:
    """打印采样值 + PhysX 回读值（“参数真的写进去了”的证据链）。"""
    sm, rb = info.get("sampled", {}), info.get("readback", {})
    if sm:
        print("[DR] 采样（每 env 独立）：")
        print(f"[DR]   负载质量 [kg]            : {_fmt_stat(sm['payload_mass_kg'])}")
        print(f"[DR]   连杆质量比例             : {_fmt_stat(sm['link_mass_scale'])}")
        print(f"[DR]   关节 damping [N·m·s/rad] : {_fmt_stat(sm['joint_damping_nms_per_rad'])}")
        print(f"[DR]   关节静摩擦 [N·m]         : {_fmt_stat(sm['joint_friction_static_nm'])}")
    if rb:
        nom = rb.get("hand_mass_nominal_kg")
        print("[DR] PhysX 回读（证明写进去了）：")
        print(
            f"[DR]   手部质量 [kg]            : {_fmt_stat(rb['hand_mass_kg'])}"
            + (f"（标称 {nom:.4g}）" if nom is not None else "")
        )
        print(f"[DR]   手部惯量最小特征值       : {_fmt_stat(rb['hand_inertia_min_eig'])}（>0 表示正定）")
        if rb.get("hand_com_shift_m") is not None:
            print(f"[DR]   手部质心位移 [m]         : {_fmt_stat(rb['hand_com_shift_m'])}")
        print(f"[DR]   关节 damping [N·m·s/rad] : {_fmt_stat(rb['joint_damping_nms_per_rad'])}")
        print(f"[DR]   关节静摩擦 [N·m]         : {_fmt_stat(rb['joint_friction_static_nm'])}")
        print(f"[DR]   关节动摩擦 [N·m]         : {_fmt_stat(rb['joint_friction_dynamic_nm'])}")
        print(f"[DR]   质心写入路径             : {info.get('com_write_path')}")


##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 1883-2327 行] W13 残差件（ResidualReward/Stats/回放池/ResidualPolicy/动作缩放/观测环）
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
# ============================================================================
# [W13/W14 新增] 残差口径的奖励 / 统计 / 显存回放池 / 执行层 / 训练循环 / 评估 / 落盘
# ============================================================================
##
class ResidualReward:
    """双层结构下的复合奖励（任务书四项 + 一个残差幅值正则，见模块 docstring §二）::

        r_t = −w_p‖e_q‖² − w_v‖ė_q‖² − w_τ‖τ_exec‖² − w_j‖u_t − u_{t−1}‖² − w_mag‖Δτ_t‖²

    ``u`` 由 ``jerk_on`` 决定：

    * ``residual``（默认，任务书口径）：``u = Δτ`` —— 只惩罚**策略自己**造成的抖动；
    * ``exec``（W12 口径）：``u = τ_exec`` —— 惩罚执行力矩的抖动（含 ``τ_base`` 自身随 ``q̈_d`` 的抖动）。

    最后一项 ``−w_mag‖Δτ‖²``（``--w_mag``，默认 0.05；**设 0 就严格回到任务书四项**）是残差结构的
    必需品，理由是一个真实的激励漏洞：``‖τ_exec‖²`` 项在双层结构里会**奖励抵消基座力矩** ——
    降低 ``‖τ_base + Δτ‖`` 最省力的办法就是让 ``Δτ ≈ −τ_base``（把基座"关掉"），而位置项
    （默认 w_p=10、量级 ~0.4/步）远压不住它（τ 项在基线处 ~1.9/步）。实测（``--w_mag 0``）：
    ``|Δτ|`` 从 0.002 N·m 单调漂到 3.5 N·m，关节 RMSE 0.052 → 0.09 rad、末端 1.33 → 2.6 cm。
    ``w_mag‖Δτ‖²`` 直接把"残差越大越贵"写进奖励（1 N·m 各关节 ≈ 0.35/步），把解锚在 Δτ≈0 附近
    —— 这正是"基座已经不错、只补差量"的残差 RL 该有的先验。

    三个分项（原始平方量）也会一起返回（``raw``），这样两种 jerk 口径可以同时统计、直接对账。
    """

    KEYS = ("pos", "vel", "tau", "jerk", "mag", "ee")

    def __init__(self, weights: dict, jerk_on: str = "residual") -> None:
        self.w = {k: float(weights.get(k, 0.0)) for k in self.KEYS}
        self.jerk_on = str(jerk_on)
        if self.jerk_on not in ("residual", "exec"):
            raise ValueError(f"jerk_on 只能是 residual / exec，收到 {jerk_on!r}")

    def compute(self, e_q, e_dot_q, tau_exec, delta, delta_prev, tau_exec_prev, ee_err_m=None):
        """算一个 batch 的奖励（``ee_err_m`` 给的是 ``‖p(q)−p(q_d)‖`` [m]，仅 ``--w_ee≠0`` 时用）。

        Returns:
            ``(reward (E,), terms dict[str,(E,)], raw dict[str,(E,)])``；
            ``terms`` 是加权后的各项（都 ≤ 0），``raw`` 是未加权的平方量（供两种口径对账）。
        """
        raw = {
            "pos": e_q.pow(2).sum(dim=-1),
            "vel": e_dot_q.pow(2).sum(dim=-1),
            "tau": tau_exec.pow(2).sum(dim=-1),
            "jerk_residual": (delta - delta_prev).pow(2).sum(dim=-1),
            "jerk_exec": (tau_exec - tau_exec_prev).pow(2).sum(dim=-1),
            "mag": delta.pow(2).sum(dim=-1),
            "ee": (ee_err_m.pow(2) if ee_err_m is not None else e_q.new_zeros(e_q.shape[0])),
        }
        jerk_key = "jerk_residual" if self.jerk_on == "residual" else "jerk_exec"
        terms = {
            "pos": -self.w["pos"] * raw["pos"],
            "vel": -self.w["vel"] * raw["vel"],
            "tau": -self.w["tau"] * raw["tau"],
            "jerk": -self.w["jerk"] * raw[jerk_key],
            "mag": -self.w["mag"] * raw["mag"],
            "ee": -self.w["ee"] * raw["ee"],
        }
        reward = (terms["pos"] + terms["vel"] + terms["tau"] + terms["jerk"]
                  + terms["mag"] + terms["ee"])
        return reward, terms, raw

    def describe(self) -> dict:
        return {
            "formula": ("r = -wp‖e_q‖² - wv‖ė_q‖² - wτ‖τ_exec‖² - wj‖u_t - u_{t-1}‖²"
                        " - w_mag‖Δτ‖² - w_ee‖p(q)-p(q_d)‖²"),
            "weights": dict(self.w),
            "jerk_on": self.jerk_on,
            "jerk_variable": "Δτ (residual)" if self.jerk_on == "residual" else "τ_exec",
            "w_mag_note": "w_mag=0 时对应任务书四项（残差结构下会漂到 |Δτ|≈3.5 N·m，见 docstring §二）",
            "w_ee_note": "w_ee=0（默认）时**不含**末端项；≠0 时把验收指标本身写进奖励",
        }


class ResidualStats:
    """残差 / 饱和 / 幅值的在线统计（全在显存里累加，收尾一次性取回 host，不拖慢 rollout）。"""

    def __init__(self, num_envs: int, num_joints: int, scale: torch.Tensor, limits: torch.Tensor, device: str) -> None:
        self.E, self.J = int(num_envs), int(num_joints)
        self.scale = scale
        self.limits = limits
        self.device = device
        self.reset()

    def reset(self) -> None:
        z = lambda: torch.zeros((), device=self.device)
        self.steps = 0
        self.delta_abs = torch.zeros((self.J,), device=self.device)      # Σ_t Σ_e |Δτ|
        self.tau_abs = torch.zeros((self.J,), device=self.device)        # Σ_t Σ_e |τ_exec|
        self.base_abs = torch.zeros((self.J,), device=self.device)       # Σ_t Σ_e |τ_base|
        self.sat_steps = z()          # 有任一关节撞限幅的步数
        self.sat_entries = z()        # 撞限幅的元素个数
        self.tanh_sat_entries = z()   # |a_norm| > 0.99 的元素个数（tanh 饱和）
        self.entries = z()            # 元素总数
        self.delta_max = 0.0

    def update(self, delta: torch.Tensor, a_norm: torch.Tensor, tau_exec: torch.Tensor, tau_base: torch.Tensor) -> None:
        self.steps += 1
        self.delta_abs += delta.abs().mean(dim=0)
        self.tau_abs += tau_exec.abs().mean(dim=0)
        self.base_abs += tau_base.abs().mean(dim=0)
        over = tau_exec.abs() >= (self.limits - 1e-9)
        self.sat_entries += over.sum()
        self.sat_steps += (over.any(dim=-1)).sum()
        self.tanh_sat_entries += (a_norm.abs() > 0.99).sum()
        self.entries += delta.numel()
        self.delta_max = max(self.delta_max, float(delta.abs().max()))

    def snapshot(self) -> dict:
        """与 W12 ``BaselineController.stats()`` 的键名保持一致（``sat_step_frac_any_joint`` 等）。"""
        denom_env = max(1, self.steps * self.E)
        return {
            "steps": int(self.steps),
            "action_scale_nm": [round(float(v), 4) for v in self.scale.tolist()],
            "res_limit_nm": round(float(self.scale.max()), 4),
            "mean_abs_delta_tau_per_joint": [round(float(v) / max(1, self.steps), 4) for v in self.delta_abs],
            "mean_abs_tau_exec_per_joint": [round(float(v) / max(1, self.steps), 4) for v in self.tau_abs],
            "mean_abs_tau_base_per_joint": [round(float(v) / max(1, self.steps), 4) for v in self.base_abs],
            "max_abs_delta_tau": round(float(self.delta_max), 4),
            "mean_abs_delta_tau": round(float(self.delta_abs.mean()) / max(1, self.steps), 4),
            "sat_step_frac_any_joint": float(self.sat_steps) / denom_env,
            "sat_entry_frac": float(self.sat_entries) / max(1, int(self.entries)),
            "tanh_saturation_frac": float(self.tanh_sat_entries) / max(1, int(self.entries)),
        }


class GpuReplayBuffer:
    """显存回放池：W9 ``src.sac.ReplayBuffer`` 的 GPU 等价实现（字段名与语义**逐条对齐**）。

    与 W9 host 版的三点差异（都是被"64 路并行、每步一条 transition/env"逼出来的）：

    1. 存储是 **GPU 张量** ⇒ 采样后不需要 H2D 拷贝（W9 版每个 batch 要搬 ``256×(2×105+7+3)`` 个 float）；
    2. 写入按**整块环形**进行：一步 64 env 一次 ``index_copy_``，而不是 64 次 ``add()`` 的 host 往返
       （64 路 × 几千步 = 几十万次 numpy 行赋值，纯 Python 开销比仿真本身还大）；
    3. 采样用 ``torch.randint(..., device=self.device)``（W9 版是 numpy RNG）。

    必须与 W9 一致的部分（否则 off-policy 的 Bellman 目标就不是同一个东西）：
    字段名 / 含义、环形 FIFO 覆盖、``done`` 只含 ``terminated``（``bootstrap_on_truncation=True``）、
    ``sample()`` 返回的键名、``stats()`` 的键名。``--buffer host`` 可切回 W9 原版做对照。
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str = "cuda:0",
        bootstrap_on_truncation: bool = True,
        seed: int | None = None,
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity 必须为正，收到 {capacity}")
        self.capacity = int(capacity)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.device = torch.device(device)
        self.bootstrap_on_truncation = bool(bootstrap_on_truncation)
        f32 = torch.float32
        self.observations = torch.zeros((self.capacity, self.state_dim), dtype=f32, device=self.device)
        self.next_observations = torch.zeros((self.capacity, self.state_dim), dtype=f32, device=self.device)
        self.actions = torch.zeros((self.capacity, self.action_dim), dtype=f32, device=self.device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=f32, device=self.device)
        self.terminated = torch.zeros((self.capacity, 1), dtype=f32, device=self.device)
        self.truncated = torch.zeros((self.capacity, 1), dtype=f32, device=self.device)
        self.ptr, self.size, self.total_added = 0, 0, 0
        self._gen = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))

    # ---------------- 基本属性（与 W9 同名） ----------------
    def __len__(self) -> int:
        return int(self.size)

    @property
    def is_full(self) -> bool:
        return self.total_added >= self.capacity

    def can_sample(self, batch_size: int = 1) -> bool:
        return self.size >= int(batch_size)

    # ---------------- 写入 ----------------
    def add_batch(self, observations, actions, rewards, next_observations, terminated=None, truncated=None) -> int:
        """整块写入 ``n`` 条 transition（环形覆盖最旧的数据）；返回写入条数。

        下标用 ``(arange(n) + ptr) % capacity`` 一次算出来 —— 当 ``capacity`` 是 ``n`` 的整数倍时
        （训练的典型情形：容量 300k、每步 64 条）等价于两段连续写入，没有 host 往返。
        """
        def _t(x, cols):
            if x is None:
                return None
            return torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, cols)

        obs = _t(observations, self.state_dim)
        n = int(obs.shape[0])
        if n == 0:
            return 0
        nobs = _t(next_observations, self.state_dim)
        act = _t(actions, self.action_dim)
        rew = _t(rewards, 1)
        term = _t(terminated, 1)
        trunc = _t(truncated, 1)
        if term is None:
            term = torch.zeros((n, 1), device=self.device)
        if trunc is None:
            trunc = torch.zeros((n, 1), device=self.device)
        if not (nobs.shape[0] == act.shape[0] == rew.shape[0] == term.shape[0] == trunc.shape[0] == n):
            raise ValueError("add_batch: 各字段的样本数不一致")
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.observations.index_copy_(0, idx, obs)
        self.next_observations.index_copy_(0, idx, nobs)
        self.actions.index_copy_(0, idx, act)
        self.rewards.index_copy_(0, idx, rew)
        self.terminated.index_copy_(0, idx, term)
        self.truncated.index_copy_(0, idx, trunc)
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n
        return n

    # ---------------- 采样 ----------------
    def done_mask(self, terminated, truncated):
        """``bootstrap_on_truncation=True`` 时 ``done = terminated``（W9 的同一约定）。"""
        terminated = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        if self.bootstrap_on_truncation:
            return terminated
        return torch.maximum(terminated, torch.as_tensor(truncated, dtype=torch.float32, device=self.device))

    def sample(self, batch_size: int, generator=None) -> dict:
        """**有放回**均匀采样一个 mini-batch（返回的张量已经在 ``self.device`` 上）。"""
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正，收到 {batch_size}")
        if not self.can_sample(batch_size):
            raise ValueError(f"回放池样本不足：size={self.size} < batch_size={batch_size}（检查 warmup_steps）")
        gen = generator if generator is not None else self._gen
        idx = torch.randint(0, self.size, (batch_size,), device=self.device, generator=gen, dtype=torch.int64)
        return self.gather(idx)

    def gather(self, index) -> dict:
        """按下标取一批样本（``sample`` 的确定性版本，便于逐条核对）。"""
        idx = torch.as_tensor(index, dtype=torch.int64, device=self.device).reshape(-1)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.size):
            raise IndexError(f"下标越界: 取值应落在 [0, {self.size})，收到 [{int(idx.min())}, {int(idx.max())}]")
        terminated, truncated = self.terminated[idx], self.truncated[idx]
        done = self.done_mask(terminated, truncated).reshape(-1, 1)
        return {
            "observation": self.observations[idx],
            "next_observation": self.next_observations[idx],
            "action": self.actions[idx],
            "reward": self.rewards[idx],
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "index": idx,
        }

    # ---------------- 监控 ----------------
    def stats(self) -> dict:
        """与 W9 ``ReplayBuffer.stats()`` 同名同义的诊断量。"""
        if self.size == 0:
            return {
                "size": 0.0, "capacity": float(self.capacity), "fill_ratio": 0.0, "total_added": 0.0,
                "reward_mean": float("nan"), "reward_std": float("nan"), "terminated_fraction": 0.0,
                "truncated_fraction": 0.0, "action_abs_mean": float("nan"),
            }
        n = self.size
        return {
            "size": float(n),
            "capacity": float(self.capacity),
            "fill_ratio": float(n) / float(self.capacity),
            "total_added": float(self.total_added),
            "reward_mean": float(self.rewards[:n].mean()),
            "reward_std": float(self.rewards[:n].std()),
            "terminated_fraction": float(self.terminated[:n].mean()),
            "truncated_fraction": float(self.truncated[:n].mean()),
            "action_abs_mean": float(self.actions[:n].abs().mean()),
        }


class HostBufferAdapter(HostReplayBuffer):
    """``--buffer host`` 用的适配器：把 W9 的 numpy 回放池包一层，让它接受 torch 张量入参。

    除了 ``.cpu().numpy()`` 这一层转置，**不改变 W9 的任何行为** —— 于是这个开关本身就是
    "显存版 vs host 版"的语义对照（两者的 ``sample()`` / ``stats()`` 键与语义必须一致）。
    """

    def add_batch(self, observations, actions, rewards, next_observations, terminated=None, truncated=None) -> int:
        def _np(x):
            return None if x is None else (x.detach().cpu().numpy() if torch.is_tensor(x) else x)

        return super().add_batch(
            _np(observations), _np(actions), _np(rewards), _np(next_observations), _np(terminated), _np(truncated)
        )


def make_replay_buffer(kind: str, capacity: int, state_dim: int, action_dim: int, device: str,
                       bootstrap_on_truncation: bool, seed):
    """按 ``--buffer {gpu,host}`` 建回放池（两者字段与采样语义一致，见各自 docstring）。"""
    if str(kind) == "host":
        return HostBufferAdapter(capacity=capacity, state_dim=state_dim, action_dim=action_dim, device=device,
                                 bootstrap_on_truncation=bootstrap_on_truncation, seed=seed)
    return GpuReplayBuffer(capacity=capacity, state_dim=state_dim, action_dim=action_dim, device=device,
                           bootstrap_on_truncation=bootstrap_on_truncation, seed=seed)


class ResidualPolicy:
    """「基座 τ_base + 策略残差 Δτ」的执行层（**训练与评估共用这一份代码**，保证两侧口径一致）。

    每一步做三件事::

        ① 基座：τ_base = BaselineController.compute(e_q, ė_q, ĝ)      # W12 的类，自带力矩限幅
        ② 残差：Δτ = a_norm ⊙ scale，a_norm = tanh(μ(+σε)) ∈ (−1,1)^J
        ③ 执行：τ_exec = clip(τ_base + Δτ, ±τ_lim)，并累加 |Δτ| / 饱和 / tanh 饱和统计

    ``scale`` 是**逐关节**的动作缩放（``--res_scale_mode``）：``uniform`` 时等于 ``res_limit``；
    ``kp``（默认）时 ``scale_j = min(res_limit, deflection·Kp_j)`` —— 满量程对应各关节**相同的静态偏转**。
    为什么后者更合理：同一个 1 N·m 在 j1（Kp=568）只偏 1.8 mrad、在 j7（Kp=2.60）偏 0.38 rad（差 200 倍），
    而 tanh 动作空间里的探索对每个关节是"等幅"的 ⇒ 不做归一化时，探索力矩在 j7 上是破坏性的
    （实测 σ≈0.22、res_limit=5 ⇒ j7 的探索力矩可达 1.1 N·m ≈ 0.42 rad 偏转，比该关节的基线误差 0.17 rad 还大）。

    两个刻意的设计：

    * ``zero_residual=True`` 时 ``Δτ ≡ 0``，于是 ``τ_exec`` 恒等于 W12 基线控制器的输出 ——
      用来做"我的评估循环 ≡ W12 的 ``run_episode``"的对账（见 :func:`zero_residual_equivalence`）；
    * ``commit()`` 必须在**算完奖励之后**调用：奖励的 jerk 项要用 ``Δτ_{t−1}`` 与 ``τ_exec,t−1``。
    """

    def __init__(self, env, base, reward: ResidualReward, scale: torch.Tensor, limits: torch.Tensor, device: str,
                 deterministic: bool = True, zero_residual: bool = False) -> None:
        self.env, self.base, self.reward = env, base, reward
        self.scale = scale.to(device=device, dtype=torch.float32).reshape(-1)
        self.limits, self.device = limits, device
        self.E, self.J = env.E, env.J
        self.delta = torch.zeros((self.E, self.J), device=device)
        self.delta_prev = torch.zeros_like(self.delta)
        self.tau_prev = torch.zeros_like(self.delta)      # τ_exec,t−1（--jerk_on exec 口径用）
        self.a_norm = torch.zeros_like(self.delta)
        self.deterministic = bool(deterministic)
        self.zero_residual = bool(zero_residual)
        self.stats = ResidualStats(self.E, self.J, self.scale, limits, device)

    def reset_prev(self) -> None:
        """episode 开始：清 τ_{t−1} / Δτ_{t−1}（与 ``env.reset()`` 里 ``_tau_last.zero_()`` 对齐）。"""
        self.delta.zero_()
        self.a_norm.zero_()
        self.delta_prev.zero_()
        self.tau_prev.zero_()

    def compose(self, tau_base: torch.Tensor) -> torch.Tensor:
        """``τ_exec = clip(τ_base + Δτ, ±τ_lim)``（Δτ 已在 :attr:`delta` 里）。"""
        return torch.clamp(torch.add(tau_base, self.delta), min=-self.limits, max=self.limits)

    def warmup_action(self, scale_nm: float) -> torch.Tensor:
        """热身期动作：``Δτ = clip(U(−s, s), ±scale_j)``（按逐关节 scale 归一化后写进 a_norm/delta）。"""
        u = torch.rand((self.E, self.J), device=self.device).mul_(2.0).sub_(1.0)
        torch.clamp(torch.div(u.mul_(float(scale_nm)), self.scale), -1.0, 1.0, out=self.a_norm)
        torch.mul(self.a_norm, self.scale, out=self.delta)
        return self.a_norm

    def act(self, agent, obs_n: torch.Tensor, e_q, e_dot_q, gravity_ff=None):
        """基座 + 残差 → ``τ_exec``（返回 ``(tau_exec, delta)``；``delta`` 是内部缓冲，视为只读）。"""
        tau_base = self.base.compute(e_q, e_dot_q, gravity_ff)
        if self.zero_residual or agent is None:
            self.a_norm.zero_()
        else:
            out = agent.act(obs_n, deterministic=self.deterministic)
            self.a_norm.copy_(out["action_norm"].detach())
        torch.mul(self.a_norm, self.scale, out=self.delta)
        tau_exec = self.compose(tau_base)
        self.stats.update(self.delta, self.a_norm, tau_exec, tau_base)
        return tau_exec, self.delta

    def commit(self, tau_exec: torch.Tensor) -> None:
        """把本步的 ``Δτ`` / ``τ_exec`` 记为"上一步"（**必须在算完奖励之后**调用）。"""
        self.delta_prev.copy_(self.delta)
        self.tau_prev.copy_(tau_exec)


def residual_action_scale(mode: str, res_limit: float, deflection_rad: float, kp: torch.Tensor,
                          J: int, device: str) -> torch.Tensor:
    """逐关节的残留力矩缩放 ``(J,)``（见 :class:`ResidualPolicy` 的 docstring）。

    * ``uniform``：``scale_j = res_limit``（任务书口径：所有关节同一个 N·m 上限）；
    * ``kp``：``scale_j = min(res_limit, deflection_rad · Kp_j)`` —— 满量程 = 相同静态偏转。
    """
    cap = torch.full((J,), float(res_limit), device=device, dtype=torch.float32)
    if str(mode) == "uniform":
        return cap
    kp_vec = kp[:, range(J), range(J)].mean(dim=0) if kp.dim() == 3 else kp.reshape(-1)
    return torch.clamp(float(deflection_rad) * kp_vec.to(device=device, dtype=torch.float32), max=float(res_limit))


def normalize_obs(normalizer, obs_raw: torch.Tensor) -> torch.Tensor:
    """观测归一化（``--no_normalize_obs`` 时原样返回；调用方只读返回的张量）。"""
    if normalizer is None:
        return obs_raw
    return normalizer.normalize_torch(obs_raw)


class PolicyObsRing:
    """策略侧观测环（H 帧历史）：**在选动作之前**把当前帧压进去，保证观测与"当前状态"严格对齐。

    .. warning::
       这是本脚本与 W12 ``env.step()`` 返回的 obs 的**唯一**差别，也是 RL 的正确性所在：
       W12 的 ``step()`` 在"写力矩之前"压入并返回观测，所以它返回的是**本步所作用的状态**的观测；
       对读 ``tracking_error(t)`` 的控制器没问题（控制器拿的永远是当前测量值），
       但对 RL 来说 ``next_obs`` 必须是**动作执行之后**那个状态的观测，否则 buffer 里的
       ``(O_t, a_t, r_t, O_{t+1})`` 就不是标准 MDP 四元组（next_obs 会滞后一步、且与 r 不对应）。

    本类复用 W12 的 :class:`ObservationBuilder`（同一个类、同一个布局、同样的 warm-up 策略），
    只把 push 的时机挪到 ``env.step()`` 之前。**自检**：本类给出的 obs 与 ``env.step()`` 返回的 obs
    应当**逐位相同**（同一帧值、同一 push 顺序，只是读取时机不同），训练循环每步核对并统计最大偏差。
    """

    def __init__(self, env, history: int | None = None, warmup: str | None = None) -> None:
        self.env = env
        self.builder = ObservationBuilder(
            env.E, env.J,
            int(env.obs_builder.H if history is None else history),
            str(env.obs_builder.warmup if warmup is None else warmup),
            env.traj.device,
        )

    def reset(self, tau_prev=None) -> None:
        """与 ``env.reset()`` 同步：用"复位后状态"的首帧按 warm-up 策略填满历史。"""
        env = self.env
        q, qd = env._state()
        q_d, qd_d, _ = env.traj.compute(0.0)
        zero = torch.zeros((env.E, env.J), device=env.traj.device)
        self.builder.reset(self.builder.frame(q, qd, q_d, qd_d, zero if tau_prev is None else tau_prev))

    def observation(self) -> torch.Tensor:
        """当前 H 帧观测（旧 → 新，最后一帧是最近一次 push 的状态）。"""
        return self.builder.observation()

    def measure_push(self, tau_prev):
        """用**当前测量状态**压一帧，返回 ``(obs, e_q, ė_q)``。

        ``e_q / ė_q`` 是新建张量（``torch.sub`` 的结果），可以安全跨过后面的 ``env.step()``；
        ``obs`` 里的 ``q_d / q̇_d`` 取自 ``t = env.step_count · dt`` 的参考（与 W12 同一处）。
        """
        env = self.env
        q, qd = env._state()
        q_d, qd_d, _ = env.traj.compute(env.step_count * env.dt)
        e_q = torch.sub(q, q_d)
        e_dot_q = torch.sub(qd, qd_d)
        self.builder.push(self.builder.frame(q, qd, q_d, qd_d, tau_prev))
        return self.builder.observation(), e_q, e_dot_q

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 2330-2512 行] run_policy_episode / evaluate / run_baseline_scenario
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
def run_policy_episode(env, policy: ResidualPolicy, agent, normalizer, tag: str, fk_tol: float,
                       joint_names: list[str] | None = None, steps: int | None = None,
                       eval_skip: int = 0, trace_every: int = 0) -> dict:
    """用「基座 + 残差策略」跑一条 episode；指标口径**逐行对齐** W12 ``run_episode``。

    口径（与 W12 一致，否则两组数字没法放同一张表）：

    * 关节 RMSE / 速度 RMSE / 奖励用 ``step()`` **之后**（t+dt 采样时刻）的状态；
    * 末端笛卡尔误差取 ``info["ee_err_m"]``（W12 里它在写成力矩**之前**算，即 t 时刻的位姿差 ——
      与关节误差差一个步长。这是 W12 的口径，本函数**照抄不改**，否则两边的数就对不上）；
    * 奖励同时记两种：残差奖励（本脚本用的）与 env 原生的 ``info["reward_terms"]``（W12 口径）。

    返回的 dict 与 W12 的 scenario 结果同 schema（关节 RMSE / 末端 cm / 奖励分项 / 饱和率 / FK 状态），
    因此 :func:`zero_residual_equivalence` 可以把它与 W12 基线的数**逐个字段**比。
    """
    E, J, dev = env.E, env.J, env.traj.device
    steps = int(env.steps if steps is None else steps)
    env.reset()
    ring = PolicyObsRing(env)
    policy.reset_prev()
    policy.stats.reset()
    ring.reset()
    obs_check_max = 0.0
    obs_raw, ring_e_q, ring_e_dot_q = ring.measure_push(policy.tau_prev)      # τ_{−1} = 0

    acc_sq = torch.zeros((E, J), device=dev)
    acc_abs = torch.zeros((E, J), device=dev)
    acc_sq_v = torch.zeros((E, J), device=dev)
    acc_sq_ee, acc_ee = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
    acc_ee_max = torch.zeros(E, device=dev)
    acc_ret, acc_ret_w12 = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
    acc_terms = {k: torch.zeros(E, device=dev) for k in ResidualReward.KEYS}
    acc_w12_terms = {k: torch.zeros(E, device=dev) for k in env.reward_fn.KEYS}
    acc_raw = {k: torch.zeros(E, device=dev) for k in
               ("pos", "vel", "tau", "jerk_residual", "jerk_exec", "mag", "ee")}
    n, trace = 0, []

    t0 = time.perf_counter()
    for k in range(steps):
        e_q, e_dot_q = ring_e_q, ring_e_dot_q
        gravity_ff = env.gravity_ff(policy.base.ff_source) if policy.base.mode == "pd_ff" else None
        tau_exec, _delta = policy.act(agent, normalize_obs(normalizer, obs_raw), e_q, e_dot_q, gravity_ff)
        obs_env, rew_env, _term, trunc, info = env.step(tau_exec)
        obs_check_max = max(obs_check_max, float((obs_raw - obs_env).abs().max()))
        reward, terms, raw = policy.reward.compute(
            info["e_q"], info["e_dot_q"], tau_exec, policy.delta, policy.delta_prev, policy.tau_prev,
            ee_err_m=info["ee_err_m"],
        )
        if k >= int(eval_skip):
            e2, e2d, ee = info["e_q"], info["e_dot_q"], info["ee_err_m"]
            acc_sq.addcmul_(e2, e2)
            acc_abs.add_(e2.abs())
            acc_sq_v.addcmul_(e2d, e2d)
            acc_sq_ee.addcmul_(ee, ee)
            acc_ee.add_(ee)
            acc_ee_max = torch.maximum(acc_ee_max, ee)
            acc_ret.add_(reward)
            acc_ret_w12.add_(rew_env)
            for key, val in terms.items():
                acc_terms[key].add_(val)
            for key, val in info["reward_terms"].items():
                acc_w12_terms[key].add_(val)
            for key, val in raw.items():
                acc_raw[key].add_(val)
            n += 1
        if trace_every > 0 and (k + 1) % int(trace_every) == 0:
            trace.append({
                "t_s": round(k * env.dt, 4),
                "joint_rms_rad": float(info["e_q"].pow(2).sum(-1).sqrt().mean()),
                "ee_cm": float(info["ee_err_m"].mean() * 100.0),
                "reward_mean": float(reward.mean()),
            })
            print(f"  [{tag}] t={k * env.dt:6.3f}s  joint RMS={trace[-1]['joint_rms_rad']:.4f} rad"
                  f"  ee={trace[-1]['ee_cm']:.3f} cm  reward={trace[-1]['reward_mean']:.3f}")
        policy.commit(tau_exec)
        # 下一步的观测：用**物理步之后**的状态压帧（截断时也先取，再 reset —— next_obs 必须是
        # "动作执行后那个状态"的观测，这样 (O_t, a_t, r_t, O_{t+1}) 才是标准 MDP 四元组）
        obs_raw, ring_e_q, ring_e_dot_q = ring.measure_push(tau_exec)
        if bool(trunc):
            env.reset()
            ring.reset()
            policy.reset_prev()
            obs_raw = ring.observation()
            _, ring_e_q, ring_e_dot_q = ring.measure_push(policy.tau_prev)
    wall = time.perf_counter() - t0
    n = max(1, n)

    rmse_env = torch.sqrt(acc_sq / n)
    per_env = torch.sqrt(acc_sq.sum(-1) / n)
    mae_env = acc_abs / n
    vel_env = torch.sqrt(acc_sq_v / n)
    ee_rmse_env = torch.sqrt(acc_sq_ee / n)
    ee_mae_env = acc_ee / n
    result = {
        "tag": tag,
        "num_envs": E,
        "steps_counted": n,
        "eval_skip": int(eval_skip),
        "joint_rmse_rad": {
            "mean_over_envs": float(rmse_env.mean()),
            "max_over_envs": float(rmse_env.max()),
            "worst_env_joint": float(per_env.max()),
            "per_joint_mean_over_envs": [float(v) for v in rmse_env.mean(dim=0)],
            "per_joint_max_over_envs": [float(v) for v in rmse_env.max(dim=0).values],
            "joint_names": joint_names,
        },
        "joint_rmse_per_env_rad": _reduce(per_env),
        "joint_mae_rad": _reduce(mae_env),
        "joint_rmse_velocity_rad_s": _reduce(vel_env),
        "cartesian_error_cm": {
            "rmse_mean_over_envs": float(ee_rmse_env.mean() * 100.0),
            "rmse_max_over_envs": float(ee_rmse_env.max() * 100.0),
            "mae_mean_over_envs": float(ee_mae_env.mean() * 100.0),
            "max_over_envs_and_steps": float(acc_ee_max.max() * 100.0),
            "per_env_rmse_cm": [round(float(v) * 100.0, 6) for v in ee_rmse_env],
        },
        "reward": {
            "mean_return_per_env": float(acc_ret.mean()),
            "mean_per_step_per_env": float((acc_ret / n).mean()),
            "terms_per_step_mean": {k: float((v / n).mean()) for k, v in acc_terms.items()},
            "raw_squared_per_step_mean": {k: float((v / n).mean()) for k, v in acc_raw.items()},
            "w12_reward_terms_per_step_mean": {k: float((v / n).mean()) for k, v in acc_w12_terms.items()},
            "w12_reward_per_step_env_native": float((acc_ret_w12 / n).mean()),
        },
        "residual": policy.stats.snapshot(),
        "obs_ring_vs_env_max_abs": obs_check_max,
        "fk": env.fk_status(fk_tol),
        "wall_time_s": round(wall, 3),
        "env_steps_per_s": round(steps * E / max(wall, 1e-9), 1),
        "trace": trace,
    }
    print(f"[RESULT] {tag:9s} 关节 RMSE = {result['joint_rmse_rad']['mean_over_envs']:.6f} rad"
          f"（最差 env/关节 {result['joint_rmse_rad']['worst_env_joint']:.6f} rad）"
          f" | 末端 RMSE = {result['cartesian_error_cm']['rmse_mean_over_envs']:.4f} cm"
          f"（最大 {result['cartesian_error_cm']['max_over_envs_and_steps']:.4f} cm）"
          f" | 回报/步 = {result['reward']['mean_per_step_per_env']:.4f}"
          f" | |Δτ| = {result['residual']['mean_abs_delta_tau']:.3f} N·m"
          f" | 饱和步占比 = {result['residual']['sat_step_frac_any_joint']:.3f}"
          f" | 吞吐 {result['env_steps_per_s']:.0f} env-steps/s")
    return result


def evaluate(env, policy: ResidualPolicy, agent, normalizer, tag: str, fk_tol: float,
             joint_names: list[str] | None = None, steps: int | None = None, trace_every: int = 0) -> dict:
    """确定性（``tanh(μ)``）闭环评估：临时把策略切到 deterministic，跑完再切回采样模式。"""
    keep = policy.deterministic
    policy.deterministic = True
    try:
        return run_policy_episode(env, policy, agent, normalizer, tag, fk_tol, joint_names,
                                  steps=steps, trace_every=trace_every)
    finally:
        policy.deterministic = keep


##
# ============================================================================
# baseline 复用 / 与 W12 数值对账 / 零残差等价性自检
# ============================================================================
##
def run_baseline_scenario(env, rng, name: str, args, kp, kd, tau_limits, joint_names, dr_state: dict) -> dict:
    """跑一条 W12 的 scenario（用 W12 的 ``BaselineController`` + ``run_episode``，一行没改）。

    ``dr_state`` 是个可变小字典（``{"on": bool|None}``），用来避免重复注入/撤回失配
    （与 W12 ``_run_scenarios`` 的做法一致：在同一进程里切换动力学，排除重启仿真的差异）。
    """
    if name not in _SCENARIOS:
        raise SystemExit(f"未知 baseline scenario {name!r}；可选：{list(_SCENARIOS)}")
    mode, want_dr, desc = _SCENARIOS[name]
    if rng.enabled and want_dr != dr_state.get("on"):
        t_sw = time.perf_counter()
        if want_dr:
            rng.apply()
        else:
            rng.restore_nominal()
        dr_state["on"] = want_dr
        env.zero_torque_warmup_step()
        print(f"[DR] 切换动力学 -> {'注入失配' if want_dr else '标称动力学'}（{time.perf_counter() - t_sw:.3f} s）")
    ctrl = BaselineController(kp, kd, env.E, env.J, env.traj.device, mode=mode, effort_limits=tau_limits,
                              sign=args.pd_sign, ff_source=args.ff)
    print(f"\n--- [{name}] {desc}")
    print(f"--- 控制器 = {mode}（增益 = {_gain_brief(kp)}），动力学 = "
          f"{'注入失配' if want_dr else '标称'}，跑 {env.steps} 步 × {env.E} env")
    return run_episode(env, ctrl, name, args.fk_tolerance, 0, args.trace_every, joint_names)

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 2617-2791 行] train_sac / _set_lr / _print_train_log
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
def train_sac(env, policy, agent, optimizers, buffer, normalizer, args, tau_limits, eval_cb, history) -> dict:
    """SAC 主循环（Day 5）：64 路并行采样 → 显存回放池 → 每步一次更新 → 周期性确定性评估。

    与 W9（Pendulum）主循环的三处差异，都是残差结构决定的：

    1. 观测归一化在线更新（``--normalize_obs``）：**buffer 里存归一化后的观测**（W9 的约定），
       评估时冻结统计量；
    2. 热身期的动作是"零残差 + 小幅均匀随机"（``--warmup_res_scale``），
       而不是 W9 那种"整个动作空间均匀随机"（那在残差结构里等于 ±15 N·m 乱甩）；
    3. 每步同时算两种 jerk 口径的奖励，只有 ``--jerk_on`` 选中的那个进 buffer
       （另一种只用于统计对账）。
    """
    dev = policy.delta.device
    E, J = env.E, env.J
    total = int(args.total_steps)
    warmup_steps = int(args.warmup_steps)
    batch_size = int(args.batch_size)
    train_freq = max(1, int(args.train_freq))
    gradient_steps = max(1, int(args.gradient_steps))
    eval_every = int(args.eval_every)
    rollout_cap = int(math.ceil(total / E))
    warmup_rollout = int(math.ceil(warmup_steps / E))
    eval_rollout = max(1, int(math.ceil(eval_every / E))) if eval_every > 0 else 0
    log_rollout = max(1, int(math.ceil(int(args.log_every) / E)))
    zero_env = torch.zeros(E, device=dev)
    actor_delay = max(0, int(getattr(args, "actor_delay_updates", 0)))
    actor_lr = float(optimizers["actor"].param_groups[0]["lr"])

    print(f"[TRAIN] 预算 {total} 环境步 = {rollout_cap} 次 rollout（每次 {E} env）| "
          f"热身 {warmup_steps} 环境步 ≈ {warmup_rollout} rollout（±{args.warmup_res_scale:g} N·m 随机残差）| "
          f"每 {train_freq} 步更新 {gradient_steps}×batch({batch_size}) | buffer={args.buffer}({buffer.capacity}) | "
          f"α0={args.alpha_init:g} 目标熵={args.target_entropy} | 归一化={'开' if normalizer else '关'} | "
          f"jerk 口径={args.jerk_on} | 奖励缩放={args.reward_scale:g}")
    if actor_delay:
        _set_lr(optimizers["actor"], 0.0)
        print(f"[TRAIN] Critic 热身：前 {actor_delay} 次更新把 Actor 学习率压到 0（只学 Q），"
              f"之后放开 {actor_lr:g}（见 docstring §四）")

    env.reset()
    ring = PolicyObsRing(env)
    policy.reset_prev()
    policy.stats.reset()
    ring.reset()
    obs_raw, e_q, e_dot_q = ring.measure_push(policy.tau_prev)

    win = {"ret": 0.0, "ret_w12": 0.0, "eq": 0.0, "ee": 0.0, "dtau": 0.0, "tau": 0.0, "sat": 0.0, "n": 0}
    obs_check = {"max_abs": 0.0, "steps": 0}
    logs_last: dict = {}
    updates = episodes = step = rollout = 0
    stopped_early = False
    t0 = time.perf_counter()

    while step < total:
        if float(args.max_wall_s) > 0 and (time.perf_counter() - t0) > float(args.max_wall_s):
            stopped_early = True
            print(f"[TRAIN] 到达墙钟上限 {args.max_wall_s:g} s：提前收尾"
                  f"（已交互 {step} 环境步 / {updates} 次更新）")
            break
        warmup = step < warmup_steps
        obs_n = normalize_obs(normalizer, obs_raw)
        gravity_ff = env.gravity_ff(policy.base.ff_source) if policy.base.mode == "pd_ff" else None
        if warmup:
            tau_base = policy.base.compute(e_q, e_dot_q, gravity_ff)
            a_norm = policy.warmup_action(float(args.warmup_res_scale))
            tau_exec = policy.compose(tau_base)
            policy.stats.update(policy.delta, a_norm, tau_exec, tau_base)
        else:
            tau_exec, a_norm = policy.act(agent, obs_n, e_q, e_dot_q, gravity_ff)

        obs_env, rew_env, _term, trunc, info = env.step(tau_exec)
        reward, _terms, _raw = policy.reward.compute(
            info["e_q"], info["e_dot_q"], tau_exec, policy.delta, policy.delta_prev, policy.tau_prev,
            ee_err_m=info["ee_err_m"],
        )
        if rollout < 50:      # 只在前 50 步做"观测环 vs env.step 返回的 obs"逐位核对
            obs_check["max_abs"] = max(obs_check["max_abs"], float((obs_raw - obs_env).abs().max()))
            obs_check["steps"] += 1
        policy.commit(tau_exec)
        obs_next_raw, nxt_e_q, nxt_e_dot_q = ring.measure_push(tau_exec)
        buffer.add_batch(
            obs_n, a_norm, reward, normalize_obs(normalizer, obs_next_raw),
            zero_env, torch.full((E,), float(bool(trunc)), device=dev),
        )
        if normalizer is not None:
            normalizer.update(obs_raw)        # 用**原始**观测更新统计量（评估时冻结）

        win["ret"] += float(reward.mean())
        win["ret_w12"] += float(rew_env.mean())
        win["eq"] += float(info["e_q"].pow(2).sum(-1).mean().sqrt())
        win["ee"] += float(info["ee_err_m"].mean() * 100.0)
        win["dtau"] += float(a_norm.abs().mean())
        win["tau"] += float(tau_exec.abs().mean())
        win["sat"] += float((tau_exec.abs() >= (tau_limits - 1e-9)).any(dim=-1).float().mean())
        win["n"] += 1
        step += E
        rollout += 1
        obs_raw, e_q, e_dot_q = obs_next_raw, nxt_e_q, nxt_e_dot_q
        if bool(trunc):
            episodes += 1
            env.reset()
            ring.reset()
            policy.reset_prev()
            obs_raw, e_q, e_dot_q = ring.measure_push(policy.tau_prev)

        if (not warmup) and buffer.can_sample(batch_size) and (rollout % train_freq == 0):
            for _ in range(gradient_steps):
                logs_last = agent.update_batch(buffer.sample(batch_size), optimizers, float(args.max_grad_norm))
                updates += 1
                if actor_delay and updates == actor_delay:      # Critic 热身结束：放开 Actor
                    _set_lr(optimizers["actor"], actor_lr)
                    print(f"[TRAIN] Critic 热身结束（{actor_delay} 次更新）：Actor 学习率放开到 {actor_lr:g}")

        if rollout % log_rollout == 0:
            _print_train_log(step, total, rollout, t0, win, logs_last, buffer, agent, warmup, obs_check,
                             actor_frozen=bool(actor_delay and updates < actor_delay))
            win = {k: 0.0 for k in win}
        if eval_cb is not None and eval_rollout and (rollout % eval_rollout == 0):
            eval_cb(step, updates, logs_last)
            # 评估在**同一个 env** 上跑了一整条 episode（确定性、不进 buffer），训练轨迹因此被打断：
            # 重新 reset 并重开观测环，保证后面存进 buffer 的 (O_t, a_t, r_t, O_{t+1}) 仍然自洽。
            env.reset()
            ring.reset()
            policy.reset_prev()
            obs_raw, e_q, e_dot_q = ring.measure_push(policy.tau_prev)

    wall = time.perf_counter() - t0
    summary = {
        "env_steps": step,
        "rollout_steps": rollout,
        "updates": updates,
        "episodes_completed": episodes,
        "wall_s": round(wall, 3),
        "env_steps_per_s": round(step / max(wall, 1e-9), 1),
        "updates_per_s": round(updates / max(wall, 1e-9), 1),
        "warmup_steps": warmup_steps,
        "warmup_res_scale": float(args.warmup_res_scale),
        "target_steps": total,
        "stopped_early": bool(stopped_early),
        "obs_ring_vs_env_max_abs": obs_check["max_abs"],
        "obs_check_steps": obs_check["steps"],
        "alpha_final": float(agent.alpha.detach()),
        "log_alpha_final": float(agent.log_alpha.detach()),
        "target_entropy": (None if agent.target_entropy is None else float(agent.target_entropy)),
        "buffer_stats": buffer.stats(),
        "last_update_logs": logs_last,
        "residual_stats_training": policy.stats.snapshot(),
        "history_points": len(history),
    }
    print(f"[TRAIN] 结束：{summary['env_steps']} 环境步 / {updates} 次更新 / {wall:.1f} s"
          f"（{summary['env_steps_per_s']:.0f} env-steps/s，{summary['updates_per_s']:.0f} updates/s）| "
          f"回放池 {buffer.size}/{buffer.capacity} | α = {summary['alpha_final']:.4g}")
    return summary


def _set_lr(optimizer, lr: float) -> None:
    """改某个 Adam 的学习率（用于 "Critic 先热身、Actor 后跟上" —— 不动 src/sac.py 一行）。"""
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _print_train_log(step, total, rollout, t0, win, logs_last, buffer, agent, warmup, obs_check,
                     actor_frozen: bool = False) -> None:
    """一行训练日志：回报 / 误差 / 残差幅值 / α / 熵差 / 损失 / 回放池 / 吞吐。"""
    n = max(1, int(win["n"]))
    el = max(time.perf_counter() - t0, 1e-9)
    phase = "热身" if warmup else ("Critic 热身" if actor_frozen else "训练")
    print(f"[STEP {step:>7d}/{total} | rollout {rollout:>5d}] {phase} | "
          f"回报/步 {win['ret'] / n:+.4f}（env 口径 {win['ret_w12'] / n:+.4f}）| "
          f"‖e_q‖ RMS {win['eq'] / n:.4f} rad | 末端 {win['ee'] / n:.3f} cm | "
          f"|Δτ| {win['dtau'] / n:.3f} N·m | |τ| {win['tau'] / n:.2f} N·m | 饱和 {win['sat'] / n:.3f} | "
          f"α {float(agent.alpha.detach()):.4g} | 熵差 {logs_last.get('entropy_target_gap', float('nan')):+.3f} | "
          f"Q {logs_last.get('q1_mean', float('nan')):+.3f} | critic {logs_last.get('critic_loss', float('nan')):.3g} | "
          f"actor {logs_last.get('actor_loss', float('nan')):+.3f} | buffer {buffer.size / buffer.capacity * 100:.1f}% | "
          f"{step / el:.0f} env-steps/s"
          + (f" | obsΔ {obs_check['max_abs']:.1e}" if obs_check["max_abs"] else ""))

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 2798-2848 行] save_policy / load_policy / write_csv
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
def save_policy(path: str, agent, normalizer, meta: dict) -> str:
    """只存 Actor（部署需要的就是它）+ 归一化统计量 + 元信息（供 ``--eval_only`` 复现评估）。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "target_critic": agent.target_critic.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "normalizer": (normalizer.state_dict() if normalizer is not None else None),
        "meta": meta,
    }, path)
    print(f"[OUT] 策略已保存：{path}")
    return path


def load_policy(path: str, agent, normalizer=None, device: str = "cuda:0") -> dict:
    """加载权重（``--eval_only`` 用）。缺 key 时给出明确报错，不静默继续。"""
    ckpt = torch.load(os.path.abspath(path), map_location=device, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    if "critic" in ckpt:
        agent.critic.load_state_dict(ckpt["critic"])
    if "target_critic" in ckpt:
        agent.target_critic.load_state_dict(ckpt["target_critic"])
    if "log_alpha" in ckpt:
        with torch.no_grad():
            agent.log_alpha.copy_(torch.as_tensor(ckpt["log_alpha"], device=agent.log_alpha.device))
    if normalizer is not None and ckpt.get("normalizer"):
        normalizer.load_state_dict(ckpt["normalizer"])
    print(f"[IN] 已加载策略权重：{os.path.abspath(path)}（meta: {ckpt.get('meta')}）")
    return ckpt


def write_csv(path: str, history: list[dict]) -> str:
    """把训练/评估曲线导成 CSV（列 = history 的键并集，缺的补空）。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not history:
        return path
    cols: list[str] = []
    for row in history:
        for key in row:
            if key not in cols:
                cols.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"[OUT] 训练曲线 CSV 已写入 {path}（{len(history)} 行 × {len(cols)} 列）")
    return path

##
# ============================================================================
# [拷贝自 W13 · w13_residual_sac_train.py 第 3445-3453 行] _jsonable
# 逐字拷贝（含注释）；两侧要同步改动时必须同时改，否则"同一个环境"这个前提就不成立了。
# ============================================================================
##
def _jsonable(obj):
    """把 argparse 的 Namespace 转成可 JSON 化的纯 Python 结构（不认识的对象转字符串，不静默丢字段）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    return str(obj)

##
# ============================================================================
# [W15 新增 · 模块 A] 动作延迟线 / 统一指标累加器 / 统一评测循环 / 轨迹记录
# ============================================================================
##
CONTROLLER_ORDER = ("pd_dr", "pd_ff_dr", "pure_rl", "residual_sac")
CONTROLLER_DESC: dict[str, str] = {
    "pd_dr": "Controller 1：纯 PD（无模型先验，纯反馈）",
    "pd_ff_dr": "Controller 2：标称前馈 + PD（工业界常见做法）",
    "pure_rl": "Controller 3：纯 RL（τ_base ≡ 0，SAC 直接输出全部关节力矩）",
    "residual_sac": "Controller 4：双层残差 RL（Ours：标称前馈+PD 兜底 + SAC 补偿动态残差）",
}
PAPER_METRICS = (
    "joint_rmse_rad", "ee_rmse_cm", "ee_max_cm", "mean_tau_norm_nm", "jerk_nm2", "sat_step_frac",
)


class NominalModel:
    """"第一次注入之前"的连杆模型快照（质量 / 惯量 / 质心），用于每次注入前**复位**。

    为什么必须复位（W15 踩到的坑，写清楚免得后人再踩）：``DynamicsRandomizer.apply()`` 的实现
    每次都把**当前** PhysX 状态读进来当 nominal（见它开头的 ``m0/I0/com0`` 那三行）再乘本轮的
    采样系数 ⇒ **同一个进程里连续注入是累积的**：连杆质量被乘两次、未知负载被叠两次。
    W12/W13 一个进程只注入一次（``apply()`` 一次），所以从没暴露；W15 的压测要在同一进程里
    连续注入十几个点（载荷 5 点 + 摩擦 4 点 + 延迟 4 点 + 每段赛程的起点），不复位就会得到
    "越压越重"的假曲线 —— 实测同一组采样参数连注入三次，手部质量均值 2.173 → 3.681 → 5.224 kg。

    ``restore()`` 把这三个量写回快照值；阻尼/摩擦不用复位（``apply()`` 是绝对写入，不是读改写）。
    """

    def __init__(self, robot: Articulation, hand_body: str = "panda_hand") -> None:
        self.mass = robot.data.body_mass.torch.to(dtype=torch.float32).clone()
        self.inertia = robot.data.body_inertia.torch.to(dtype=torch.float32).clone()
        self.com = robot.data.body_com_pose_b.torch.to(dtype=torch.float32).clone()
        hid = _resolve_body_id(robot, hand_body)
        self.hand_body, self.hand_mass_kg = hand_body, float(self.mass[0, hid])
        self.hand_com_z = float(self.com[0, hid, 2])

    def restore(self, robot: Articulation, rng: DynamicsRandomizer | None = None) -> str:
        """写回快照（质量/惯量/质心），返回质心写入路径（便于在日志里确认走得通）。"""
        robot.set_masses_index(masses=self.mass.contiguous())
        robot.set_inertias_index(inertias=self.inertia.contiguous())
        path = None
        if rng is not None:
            path = rng._write_coms(self.com.contiguous())        # 与 apply() 同一条路径
        if path is None:
            try:
                robot.set_coms_index(coms=self.com.contiguous())
                path = "set_coms_index"
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] 标称质心复位失败（{type(exc).__name__}: {exc}）")
                path = "failed"
        return path

    def describe(self) -> dict:
        return {"hand_body": self.hand_body, "hand_mass_kg_nominal": round(self.hand_mass_kg, 6),
                "note": "注入前的 USD 标称值；每次注入前用它复位，避免 apply() 的累积效应"}


class ActionDelayBuffer:
    """动作下发延迟（Latency / Phase Lag）：本步算出的 ``τ_t`` 经过 ``steps`` 步才到达关节。

    物理含义就是"控制器 → 驱动器"这条链路上的离散延迟（1 步 = ``--dt`` = 10 ms @100 Hz）。
    实现是长度 ``steps`` 的 FIFO：``push(τ_t)`` 写进队尾、返回队首（``steps`` 步之前写进去的那个）。
    ``steps = 0`` 时 ``push`` 原样返回 ⇒ 与 W12/W13 的评测逐位相同。

    ``prime()`` 用首步力矩**预填**整条延迟线：等价于"这条链路在实验开始之前就已经在跑"。
    否则前 ``steps`` 步下发的会是 0 力矩（带载下 30 ms 的自由下落），那部分瞬态不属于控制性能。
    预填要在算出手矩之后、第一步 ``step`` 之前做，所以主循环里是"第一次 push 时自动预填"。
    """

    def __init__(self, num_envs: int, num_joints: int, steps: int, device: str, dt: float = 0.01) -> None:
        self.E, self.J = int(num_envs), int(num_joints)
        self.steps = max(0, int(steps))
        self.dt, self.device = float(dt), device
        self._slots = [torch.zeros((self.E, self.J), device=device) for _ in range(max(1, self.steps))]
        self._out = torch.zeros((self.E, self.J), device=device)
        self.cursor = 0
        self.primed = False

    @property
    def delay_s(self) -> float:
        return self.steps * self.dt

    def reset(self) -> None:
        for s in self._slots:
            s.zero_()
        self._out.zero_()
        self.cursor = 0
        self.primed = False

    def prime(self, tau: torch.Tensor) -> None:
        """用 ``tau`` 预填所有槽位（只在第一次 push 时调用，见类 docstring）。"""
        for s in self._slots:
            s.copy_(tau)
        self.primed = True

    def push(self, tau: torch.Tensor) -> torch.Tensor:
        """写入本步力矩，返回**本步真正下发**的力矩（``steps = 0`` 时就是 ``tau`` 本身）。"""
        if self.steps == 0:
            return tau
        if not self.primed:
            self.prime(tau)
        slot = self._slots[self.cursor]
        self._out.copy_(slot)          # 队首 = steps 步之前写进去的
        slot.copy_(tau)                # 本步写进队尾
        self.cursor = (self.cursor + 1) % self.steps
        return self._out

    def describe(self) -> dict:
        return {"steps": self.steps, "delay_ms": round(self.delay_s * 1000.0, 3),
                "note": "延迟线首步预填（等价于链路实验前已在跑）"}
class UnifiedMetrics:
    """四个控制器共用的指标累加器（公式逐行对齐 W12 ``run_episode`` / W13 ``run_policy_episode``）。

    .. list-table::
       :header-rows: 1

       * - 内部累加器
         - 公式
         - 对应 W12/W13 的哪一行
       * - ``acc_sq`` / ``acc_sq_v``
         - ``Σ_t (q−q_d)²`` / ``Σ_t (q̇−q̇_d)²``
         - ``run_episode`` 的 ``acc_sq.addcmul_(e2, e2)``
       * - ``acc_sq_ee`` / ``acc_ee`` / ``acc_ee_max``
         - ``Σ_t ‖Δp‖²`` / ``Σ_t ‖Δp‖`` / ``max_t ‖Δp‖``
         - 同上（``ee_err_m`` 取 ``info`` 里那个，即写成力矩**之前**的时刻 —— W12 口径）
       * - ``acc_tau_norm`` / ``acc_tau_abs_j``
         - ``Σ_t ‖τ‖₂`` / ``Σ_t mean_e |τ_j|``
         - ``BaselineController.stats()`` 的 ``abs_tau_per_joint``（逐关节均值同口径）
       * - ``acc_jerk``
         - ``Σ_t ‖τ_t − τ_{t−1}‖²``（``τ_{−1} = 0``）
         - ``MultiObjectiveReward`` 里那一项 jerk 的定义
       * - ``sat_steps`` / ``sat_entries``
         - 撞限幅的步数 / 元素数
         - ``BaselineController.stats()`` 的 ``sat_step_frac_any_joint``

    **τ 一律取"真正下发到关节的那个力矩"**（延迟压测下 = 延迟线的输出），否则延迟的影响会被统计掩盖。
    """

    def __init__(self, num_envs: int, num_joints: int, tau_limits: torch.Tensor, device: str,
                 joint_names: list[str] | None = None) -> None:
        self.E, self.J = int(num_envs), int(num_joints)
        lim = tau_limits.to(device=device, dtype=torch.float32)
        # 允许传 (E,J) 或 (J,)：统一压成 (J,)，饱和判据 ``|τ| >= limits`` 靠广播对每个 env 都成立
        self.limits = lim.reshape(-1, lim.shape[-1])[0] if lim.dim() > 1 else lim
        self.device = device
        self.joint_names = joint_names
        self.reset()

    def reset(self) -> None:
        dev, E, J = self.device, self.E, self.J
        self.acc_sq = torch.zeros((E, J), device=dev)
        self.acc_abs = torch.zeros((E, J), device=dev)
        self.acc_sq_v = torch.zeros((E, J), device=dev)
        self.acc_sq_ee, self.acc_ee = torch.zeros(E, device=dev), torch.zeros(E, device=dev)
        self.acc_ee_max = torch.zeros(E, device=dev)
        self.acc_tau_norm = torch.zeros(E, device=dev)
        self.acc_tau_norm_max = torch.zeros(E, device=dev)
        self.acc_tau_abs_j = torch.zeros(J, device=dev)
        self.acc_jerk = torch.zeros(E, device=dev)
        self.acc_jerk_max = torch.zeros(E, device=dev)
        self.sat_steps = torch.zeros((), device=dev)
        self.sat_entries = torch.zeros((), device=dev)
        self.entries = torch.zeros((), device=dev)
        self.acc_ret = torch.zeros(E, device=dev)
        self.acc_env_terms: dict[str, torch.Tensor] = {}
        self.steps = 0
        self.tau_prev = torch.zeros((E, J), device=dev)

    def update(self, e_q, e_dot_q, ee_err_m, tau_applied, env_terms=None, env_reward=None) -> None:
        """累加一步（**力矩用真正下发的那个**；``env_terms`` 是 env 原生四项奖励，四控制器同口径）。"""
        self.steps += 1
        self.acc_sq.addcmul_(e_q, e_q)
        self.acc_abs.add_(e_q.abs())
        self.acc_sq_v.addcmul_(e_dot_q, e_dot_q)
        self.acc_sq_ee.addcmul_(ee_err_m, ee_err_m)
        self.acc_ee.add_(ee_err_m)
        self.acc_ee_max = torch.maximum(self.acc_ee_max, ee_err_m)
        norm = tau_applied.norm(dim=-1)
        self.acc_tau_norm.add_(norm)
        self.acc_tau_norm_max = torch.maximum(self.acc_tau_norm_max, norm)
        self.acc_tau_abs_j += tau_applied.abs().mean(dim=0)
        jerk = (tau_applied - self.tau_prev).pow(2).sum(dim=-1)
        self.acc_jerk.add_(jerk)
        self.acc_jerk_max = torch.maximum(self.acc_jerk_max, jerk)
        self.tau_prev.copy_(tau_applied)
        over = tau_applied.abs() >= (self.limits - 1e-9)
        self.sat_entries += over.sum()
        self.sat_steps += over.any(dim=-1).sum()
        self.entries += float(tau_applied.numel())
        if env_reward is not None:
            self.acc_ret.add_(env_reward)
        if env_terms is not None:
            for key, val in env_terms.items():
                self.acc_env_terms.setdefault(key, torch.zeros(self.E, device=self.device)).add_(val)
    def snapshot(self) -> dict:
        """收尾一次性取回 host：论文表格要的那 6 个指标 + 逐 env / 逐关节细节。"""
        n = max(1, self.steps)
        rmse_env = torch.sqrt(self.acc_sq / n)
        per_env = torch.sqrt(self.acc_sq.sum(-1) / n)
        ee_rmse_env = torch.sqrt(self.acc_sq_ee / n)
        tau_abs_j = [float(v) / n for v in self.acc_tau_abs_j]
        snap = {
            "num_envs": self.E,
            "steps_counted": int(self.steps),
            "joint_rmse_rad": {
                "mean_over_envs": float(rmse_env.mean()),
                "max_over_envs": float(rmse_env.max()),
                "worst_env_joint": float(per_env.max()),
                "per_joint_mean_over_envs": [float(v) for v in rmse_env.mean(dim=0)],
                "per_joint_max_over_envs": [float(v) for v in rmse_env.max(dim=0).values],
                "per_env_rmse_rad": [round(float(v), 6) for v in per_env],
                "joint_names": self.joint_names,
            },
            "cartesian_error_cm": {
                "rmse_mean_over_envs": float(ee_rmse_env.mean() * 100.0),
                "rmse_max_over_envs": float(ee_rmse_env.max() * 100.0),
                "mae_mean_over_envs": float((self.acc_ee / n).mean() * 100.0),
                "max_over_envs_and_steps": float(self.acc_ee_max.max() * 100.0),
                "per_env_rmse_cm": [round(float(v) * 100.0, 6) for v in ee_rmse_env],
            },
            "effort": {
                "mean_tau_norm_nm": float((self.acc_tau_norm / n).mean()),
                "mean_tau_norm_max_over_envs_nm": float((self.acc_tau_norm / n).max()),
                "peak_tau_norm_nm": float(self.acc_tau_norm_max.max()),
                "mean_abs_tau_per_joint": [round(v, 4) for v in tau_abs_j],
                "mean_abs_tau_overall": round(float(sum(tau_abs_j) / max(1, self.J)), 4),
            },
            "smoothness": {
                "jerk_nm2": float((self.acc_jerk / n).mean()),
                "jerk_nm2_max_over_envs": float((self.acc_jerk / n).max()),
                "jerk_rms_nm": float(((self.acc_jerk / n).mean()).sqrt()),
                "note": "jerk = mean_t‖τ_t−τ_{t−1}‖²（τ 取真正下发的力矩，τ_{−1}=0）",
            },
            "safety": {
                "sat_step_frac": float(self.sat_steps) / max(1, self.steps * self.E),
                "sat_entry_frac": float(self.sat_entries) / max(1.0, float(self.entries)),
                "tau_limits_nm": [round(float(v), 3) for v in self.limits.tolist()],
            },
            "env_reward_w12_convention": {
                "mean_per_step_per_env": float((self.acc_ret / n).mean()),
                "terms_per_step_mean": {k: float((v / n).mean()) for k, v in self.acc_env_terms.items()},
            },
        }
        snap["paper_row"] = paper_row(snap)
        return snap


def paper_row(snap: dict) -> dict:
    """从 :meth:`UnifiedMetrics.snapshot` 里抽出论文表格那一行（Table 1/2/3 的列就是它）。"""
    row = {
        "joint_rmse_rad": snap["joint_rmse_rad"]["mean_over_envs"],
        "ee_rmse_cm": snap["cartesian_error_cm"]["rmse_mean_over_envs"],
        "ee_max_cm": snap["cartesian_error_cm"]["max_over_envs_and_steps"],
        "mean_tau_norm_nm": snap["effort"]["mean_tau_norm_nm"],
        "jerk_nm2": snap["smoothness"]["jerk_nm2"],
        "sat_step_frac": snap["safety"]["sat_step_frac"],
        "joint_rmse_worst_env_rad": snap["joint_rmse_rad"]["worst_env_joint"],
        "ee_rmse_max_over_envs_cm": snap["cartesian_error_cm"]["rmse_max_over_envs"],
        "mean_abs_tau_overall_nm": snap["effort"]["mean_abs_tau_overall"],
        "task_reward_per_step": snap["env_reward_w12_convention"]["mean_per_step_per_env"],
    }
    return row


class EpisodeTrace:
    """时间域轨迹记录（Figure 1 用）：只记一台 env 的 ``q_d / q / ‖Δp‖``，默认前 2 s（200 步）。

    为什么只记一台 env：Figure 1 要的是"曲线贴合细节"，一台代表环境就够；64 条曲线叠在一起
    反而看不清相位滞后。记录量 = ``3×steps`` 个标量 + ``14×steps`` 个关节角，落进 JSON 也很轻。
    """

    def __init__(self, num_joints: int, steps: int, env_idx: int = 0, device: str = "cuda:0") -> None:
        self.J, self.steps, self.env_idx = int(num_joints), int(steps), int(env_idx)
        self.device = device
        self.t = torch.zeros(steps, device=device)
        self.q_d = torch.zeros((steps, self.J), device=device)
        self.q = torch.zeros((steps, self.J), device=device)
        self.ee = torch.zeros(steps, device=device)

    def record(self, k: int, t: float, q: torch.Tensor, q_d: torch.Tensor, ee_err_m: torch.Tensor) -> None:
        self.t[k] = float(t)
        self.q_d[k].copy_(q_d[self.env_idx])
        self.q[k].copy_(q[self.env_idx])
        self.ee[k] = ee_err_m[self.env_idx]

    def as_dict(self) -> dict:
        return {
            "env_idx": self.env_idx,
            "steps": self.steps,
            "t_s": [round(float(v), 5) for v in self.t],
            "ee_cm": [round(float(v) * 100.0, 5) for v in self.ee],
            "q_d_rad": [[round(float(v), 6) for v in row] for row in self.q_d],
            "q_rad": [[round(float(v), 6) for v in row] for row in self.q],
        }
class ControllerBundle:
    """一个"控制器"运行所需的一切：基座（经典控制器）+ 可选策略 + 动作延迟线。

    四个控制器之间的差别**只在这里**：``base`` 的模式（pd / pd_ff / zero）、有没有 ``policy``、
    以及策略的动作缩放 ``scale``。评测循环、指标累加器、参考轨迹、DR 采样全部共用 ⇒
    "控制器之间不可能因为口径不同而产生虚假差异"这句话是可验证的（见 :func:`verify_unified_vs_w12`）。
    """

    def __init__(self, name: str, base: BaselineController, limits: torch.Tensor, device: str,
                 policy: ResidualPolicy | None = None, agent=None, normalizer=None,
                 delay_steps: int = 0, dt: float = 0.01) -> None:
        self.name = name
        self.base, self.limits, self.device = base, limits, device
        self.policy, self.agent, self.normalizer = policy, agent, normalizer
        self.delay = ActionDelayBuffer(base.num_envs, base.num_joints, delay_steps, device, dt)

    @property
    def is_rl(self) -> bool:
        return self.policy is not None

    def describe(self) -> dict:
        """落盘用的控制器自述（论文方法节要能一眼看出这个数是怎么来的）。"""
        return {
            "name": self.name,
            "description": CONTROLLER_DESC.get(self.name, ""),
            "base_mode": self.base.mode,
            "ff_source": (self.base.ff_source if self.base.mode == "pd_ff" else None),
            "is_rl": bool(self.is_rl),
            "action_scale_nm": ([round(float(v), 4) for v in self.policy.scale.tolist()] if self.is_rl else None),
            "action_space": ("全关节力矩 ±τ_lim（scale = τ_lim）" if self.name == "pure_rl"
                             else ("残差力矩（逐关节 scale 见 action_scale_nm）" if self.is_rl
                                   else "—（无策略，纯经典控制器）")),
            "delay": self.delay.describe(),
        }


def w15_action_scale(name: str, args, kp: torch.Tensor, limits: torch.Tensor, device: str) -> torch.Tensor:
    """逐关节动作缩放 ``(J,)``。

    * ``pure_rl``：``scale_j = τ_lim,j``（**必须**逐关节用硬件限幅，而不是一刀切 ±15 或 ±87：
      j1–j4 是 87 N·m、j5–j7 只有 12 N·m，用一个大数会让手腕的动作空间被限幅削掉一半，
      tanh 探索在手腕上等于"永远撞限"）；
    * ``residual_sac``：与 W13 完全同口径（``residual_action_scale``，默认 ``--res_scale_mode kp``）。

    另外 :data:`CONTROLLER_ORDER` 之外的调用点（例如将来的"限幅消融"脚本）可以直接用 ``"limits"``：
    此时残余也用逐关节 τ_lim 作动作缩放。本脚本的命令行开关 ``--res_scale_mode`` 与 W13 逐字同源
    （只有 ``kp`` / ``uniform`` 两个选项），所以那条分支在 W15 里不通过 CLI 暴露。
    """
    if name == "pure_rl" or str(getattr(args, "res_scale_mode", "kp")) == "limits":
        return limits.to(device=device, dtype=torch.float32).clone()
    return residual_action_scale(args.res_scale_mode, args.res_limit, args.res_deflection_rad,
                                 kp, int(limits.numel()), device)
def run_unified_episode(env, bundle: ControllerBundle, tag: str, fk_tol: float,
                        joint_names: list[str] | None = None, steps: int | None = None,
                        trace: EpisodeTrace | None = None) -> dict:
    """四个控制器共用的评测循环（确定性闭环、``no_grad``、不写回放池、不更新归一化统计量）。

    时序（与 W12 ``run_episode`` / W13 ``run_policy_episode`` 对齐，逐步核对过）：

    1. ``t`` 时刻：经典控制器读 ``env.tracking_error(t)``；RL 控制器用观测环在**写力矩之前**压一帧
       （:class:`PolicyObsRing`，保证 ``(O_t, a_t, r_t, O_{t+1})`` 是标准 MDP 四元组）；
    2. 算 ``τ_cmd``（基座 / 基座+残差 / 纯力矩）→ 过延迟线得到 ``τ_applied``；
    3. ``env.step(τ_applied)`` → 推一步物理 → ``t+dt`` 的状态；
    4. 指标用 ``info`` 里的 ``e_q / ė_q / ‖Δp‖``（t 时刻的位姿差，W12 口径）与**下发的** ``τ_applied``；
    5. RL：``policy.commit(τ_applied)`` 再为下一步压帧（与 W13 一致）。
    """
    E, J, dev = env.E, env.J, env.traj.device
    steps = int(env.steps if steps is None else steps)
    m = UnifiedMetrics(E, J, bundle.limits, dev, joint_names)
    env.reset()
    bundle.delay.reset()
    ring = None
    obs_raw = e_q = e_dot_q = None
    if bundle.is_rl:
        ring = PolicyObsRing(env)
        bundle.policy.reset_prev()
        bundle.policy.stats.reset()
        ring.reset()
        obs_raw, e_q, e_dot_q = ring.measure_push(bundle.policy.tau_prev)

    t0 = time.perf_counter()
    for k in range(steps):
        if bundle.is_rl:
            e, edot = e_q, e_dot_q
        else:
            e, edot, _, _ = env.tracking_error(k * env.dt)
        gravity_ff = env.gravity_ff(bundle.base.ff_source) if bundle.base.mode == "pd_ff" else None
        if bundle.is_rl:
            tau_cmd, _delta = bundle.policy.act(bundle.agent, normalize_obs(bundle.normalizer, obs_raw),
                                                e, edot, gravity_ff)
        else:
            tau_cmd = bundle.base.compute(e, edot, gravity_ff)
        tau_applied = bundle.delay.push(tau_cmd)
        obs_env, rew_env, _term, trunc, info = env.step(tau_applied)      # noqa: F841
        m.update(info["e_q"], info["e_dot_q"], info["ee_err_m"], tau_applied,
                 env_terms=info["reward_terms"], env_reward=rew_env)
        if trace is not None and k < trace.steps:
            q2, _qd2 = env._state()
            q_d2, _qdd2, _ = env.traj.compute((k + 1) * env.dt)
            ee2 = (env.ee_position(q2) - env.ee_position(q_d2)).norm(dim=-1)
            trace.record(k, (k + 1) * env.dt, q2, q_d2, ee2)
        if bundle.is_rl:
            bundle.policy.commit(tau_applied)
            obs_raw, e_q, e_dot_q = ring.measure_push(tau_applied)
            if bool(trunc):        # 与 W13 一致：截断时重开（这一步不进任何统计）
                env.reset()
                ring.reset()
                bundle.policy.reset_prev()
                obs_raw, e_q, e_dot_q = ring.measure_push(bundle.policy.tau_prev)
    wall = time.perf_counter() - t0

    snap = m.snapshot()
    out = {
        "tag": tag,
        "controller": bundle.name,
        "desc": CONTROLLER_DESC.get(bundle.name, ""),
        "num_envs": E,
        "steps": steps,
        "controller_meta": bundle.describe(),
        "metrics": snap,
        "paper_row": snap["paper_row"],
        "residual_stats": (bundle.policy.stats.snapshot() if bundle.is_rl else None),
        "base_controller_stats": bundle.base.stats(),
        "fk": env.fk_status(fk_tol),
        "wall_time_s": round(wall, 3),
        "env_steps_per_s": round(steps * E / max(wall, 1e-9), 1),
        "trace": (trace.as_dict() if trace is not None else None),
    }
    print(f"[RESULT] {tag:24s} {bundle.name:13s} 关节 RMSE = {snap['joint_rmse_rad']['mean_over_envs']:.6f} rad"
          f" | 末端 RMSE = {snap['cartesian_error_cm']['rmse_mean_over_envs']:.4f} cm"
          f"（峰值 {snap['cartesian_error_cm']['max_over_envs_and_steps']:.3f}）"
          f" | ‖τ‖ = {snap['effort']['mean_tau_norm_nm']:.3f} N·m"
          f" | jerk = {snap['smoothness']['jerk_nm2']:.3f}"
          f" | 饱和 {snap['safety']['sat_step_frac']:.3f}"
          f" | {out['env_steps_per_s']:.0f} env-steps/s")
    return out
def verify_unified_vs_w12(w12_path: dict[str, dict], unified: dict[str, dict], tol: float = 1e-9) -> dict:
    """把 W12 ``run_episode``（老路径）与 :func:`run_unified_episode`（新统一循环）逐字段对账。

    两者跑的是**同一动力学、同一控制器、同一轨迹**，唯一差别是"谁来累加指标"。
    一致判据：相对偏差 ``<= tol``（默认 1e-9；两个循环的累加顺序相同，理论上应当逐位相同）。
    这份对账是"Table 1 里 pd_dr / pd_ff_dr 两行仍然等于 W12 报告里那两个数"的**硬证据**。
    """
    pairs = (("joint_rmse_rad", "mean_over_envs"), ("cartesian_error_cm", "rmse_mean_over_envs"),
             ("cartesian_error_cm", "max_over_envs_and_steps"))
    out = {"available": bool(w12_path), "tol": float(tol), "compared": {}, "match": None}
    worst = 0.0
    for name, res in (w12_path or {}).items():
        now = unified.get(name)
        if now is None:
            continue
        entry = {}
        for key, sub in pairs:
            a = now["metrics"][key][sub]
            b = res[key][sub]
            rel = abs(a - b) / max(abs(b), 1e-12)
            worst = max(worst, rel)
            entry[f"{key}.{sub}"] = {"unified": a, "w12": b, "rel_diff": rel}
        a = now["metrics"]["env_reward_w12_convention"]["mean_per_step_per_env"]
        b = res["reward"]["mean_per_step_per_env"]
        entry["reward_per_step"] = {"unified": a, "w12": b, "rel_diff": abs(a - b) / max(abs(b), 1e-12)}
        worst = max(worst, entry["reward_per_step"]["rel_diff"])
        a = now["metrics"]["safety"]["sat_step_frac"]
        b = res["controller"]["sat_step_frac_any_joint"]
        entry["sat_step_frac"] = {"unified": a, "w12": b, "abs_diff": abs(a - b)}
        worst = max(worst, abs(a - b))
        out["compared"][name] = entry
    if out["compared"]:
        out["max_rel_diff"] = worst
        out["match"] = bool(worst <= tol)
        for name, entry in out["compared"].items():
            print(f"[CHECK] 统一循环 vs W12 run_episode（{name}）：关节 RMSE "
                  f"{entry['joint_rmse_rad.mean_over_envs']['unified']:.9f} vs "
                  f"{entry['joint_rmse_rad.mean_over_envs']['w12']:.9f}"
                  f"（rel {entry['joint_rmse_rad.mean_over_envs']['rel_diff']:.2e}）| 末端 "
                  f"{entry['cartesian_error_cm.rmse_mean_over_envs']['unified']:.6f} vs "
                  f"{entry['cartesian_error_cm.rmse_mean_over_envs']['w12']:.6f} cm")
        print(f"[CHECK] 最大相对偏差 {worst:.2e}（阈值 {tol:g}）-> "
              f"{'通过：新循环与 W12 口径逐位一致' if out['match'] else '不通过：两套指标代码有分歧，必须先查清'}")
    else:
        print("[CHECK] 没有可对账的 W12 路径结果（--no_verify_w12_path 或该 scenario 不在本次赛程里）")
    return out

##
# ============================================================================
# [W15 新增 · 模块 B] DR 压测点 / 控制器装配（Rig）/ 纯 RL 与消融变体的训练 / 三个赛程
# ============================================================================
##
DR_KEYS = ("payload_mass", "link_mass_scale", "joint_damping", "joint_friction")
# 压测时"钉死"用的中点（载荷取训练区间中点 1.5 kg；连杆取标称 1.0×）
DR_MID = {"payload_mass": 1.5, "link_mass_scale": 1.0}
_DR_ALIAS = {"payload": "payload_mass", "payload_mass": "payload_mass", "mass": "payload_mass",
             "link_mass": "link_mass_scale", "link_mass_scale": "link_mass_scale", "link": "link_mass_scale",
             "damping": "joint_damping", "joint_damping": "joint_damping",
             "friction": "joint_friction", "joint_friction": "joint_friction"}


def base_dr_ranges(args) -> dict:
    """训练分布的那一组 DR 区间（直接来自命令行，与 W13 构造 ``DynamicsRandomizer`` 时逐字同源）。"""
    return {
        "payload_mass": tuple(float(v) for v in args.payload_mass),
        "link_mass_scale": tuple(float(v) for v in args.link_mass_scale),
        "joint_damping": tuple(float(v) for v in args.joint_damping),
        "joint_friction": tuple(float(v) for v in args.joint_friction),
    }


def stress_ranges(args, sweep: str, value: float | None) -> dict:
    """算出某个压测点的 DR 区间（除自变量外的失配按默认规则"钉死"或保留训练分布）。

    默认规则（``--stress_fix`` 可逐项覆盖）：

    ======  ================================  ===========================================
    压测轴   自变量                                其余失配
    ======  ================================  ===========================================
    载荷     ``payload_mass = (m, m)``          连杆 ×U(0.8,1.2) + damping/friction = 训练区间
    摩擦     damping/friction 训练区间 ×mult     载荷钉死 1.5 kg（训练区间中点）+ 连杆 ×U(0.8,1.2)
    延迟     ``--action_delay``（不碰 DR）        全部 = 训练区间（就是训练时那一套失配）
    ======  ================================  ===========================================

    ``--stress_fix`` 的写法：``payload=2.5,link_mass=1.0,damping=2,friction=-``；
    **载荷/连杆给绝对值**（kg / 倍率），**damping/friction 给相对训练区间的乘数**，``-`` 表示保持训练区间。

    为什么压测点也要"同一个 seed 重新采样"：这样任意两个控制器、任意两次运行在**同一点**上
    拿到的是**逐位相同**的物理参数，曲线上的差异只可能来自控制器本身。
    """
    base = base_dr_ranges(args)
    ranges = {k: tuple(base[k]) for k in DR_KEYS}
    note: dict[str, str] = {k: "训练区间（照原样）" for k in DR_KEYS}
    if sweep == "payload" and value is not None:
        if args.stress_keep_spread:
            ranges["payload_mass"] = (0.0, float(value))
            note["payload_mass"] = f"区间 [0, {float(value):g}] kg（保留分布）"
        else:
            ranges["payload_mass"] = (float(value), float(value))
            note["payload_mass"] = f"钉死 {float(value):g} kg（本轴自变量）"
    elif sweep == "friction" and value is not None:
        m = float(value)
        ranges["joint_damping"] = tuple(v * m for v in base["joint_damping"])
        ranges["joint_friction"] = tuple(v * m for v in base["joint_friction"])
        ranges["payload_mass"] = (DR_MID["payload_mass"], DR_MID["payload_mass"])
        note["payload_mass"] = f"钉死 {DR_MID['payload_mass']:g} kg（排除载荷干扰）"
        note["joint_damping"] = f"训练区间 ×{m:g}"
        note["joint_friction"] = f"训练区间 ×{m:g}"
    for item in str(getattr(args, "stress_fix", "") or "").split(","):
        if not item.strip():
            continue
        name, _, val = item.partition("=")
        key = _DR_ALIAS.get(name.strip().lower())
        if key is None:
            raise SystemExit(f"--stress_fix 里的名字 {name!r} 不认得；可用：{sorted(set(_DR_ALIAS))}")
        txt = val.strip().lower()
        if txt in ("", "-", "keep", "train", "none"):
            ranges[key], note[key] = tuple(base[key]), "保持训练区间"
            continue
        try:
            fixed = float(txt)
        except ValueError as exc:
            raise SystemExit(f"--stress_fix 的 {item!r} 值 {val!r} 既不是数字也不是 '-'") from exc
        if key in ("joint_damping", "joint_friction"):
            ranges[key], note[key] = tuple(v * fixed for v in base[key]), f"训练区间 ×{fixed:g}"
        else:
            ranges[key], note[key] = (fixed, fixed), f"钉死 {fixed:g}"
    return {"sweep": sweep, "sweep_value": value, "ranges": ranges, "note": note}
def inject_dr(env, rng: DynamicsRandomizer, ranges: dict, seed: int, label: str = "",
              nominal: NominalModel | None = None) -> dict:
    """把一组 DR 区间写进 PhysX（同 seed 重新采样 ⇒ 参数确定），返回"真的写进去了"的证据链。

    ``nominal`` 非空时**先复位**连杆质量/惯量/质心（见 :class:`NominalModel` 的 docstring：
    ``DynamicsRandomizer.apply()`` 在同一进程里连续调用会累积），再注入。
    """
    out = {"label": label, "ranges": {k: [float(x) for x in v] for k, v in ranges.items()}, "applied": False}
    if not rng.enabled:
        print(f"[DR] {label}: --no_randomization 生效，本次不注入失配（只能当消融看）")
        return out
    if nominal is not None:
        out["nominal_restore_com_path"] = nominal.restore(env.robot, rng)
    rng.ranges = {k: tuple(float(x) for x in v) for k, v in ranges.items()}
    rng.gen = torch.Generator(device="cpu").manual_seed(int(seed))
    rng.sample()
    rng.apply()
    env.zero_torque_warmup_step()          # 让 PhysX 的 M(q) 反映新模型（与 W13 同一顺序）
    out.update(applied=True, sampled=rng.sampled_summary(), readback=rng.readback(),
               com_write_path=rng.com_path)
    print(f"[DR] {label}: 区间 " + " | ".join(f"{k}={list(v)}" for k, v in out["ranges"].items()))
    print(f"[DR] {label}: 采样负载 {_fmt_stat(out['sampled'].get('payload_mass_kg'))} kg | "
          f"damping {_fmt_stat(out['sampled'].get('joint_damping_nms_per_rad'))} | "
          f"静摩擦 {_fmt_stat(out['sampled'].get('joint_friction_static_nm'))} N·m")
    print(f"[DR] {label}: 回读手部质量 {_fmt_stat(out['readback'].get('hand_mass_kg'))} kg"
          f"（标称 {out['readback'].get('hand_mass_nominal_kg')}）| 质心写入路径 {out['com_write_path']}")
    return out


def build_sac_agent_for(args, obs_dim: int, J: int, device: str, action_bound: float):
    """按 W13 的 ``sac_cfg`` 结构建 SAC（网络 / 损失 / 更新全部来自 W9 ``src/sac.py``，一行没改）。"""
    te_raw = str(args.target_entropy).strip().lower()
    if te_raw in ("fixed", "none", "null", "off", "false", ""):
        target_entropy = None                     # → 固定 α
    elif te_raw in ("auto", "-dim_a", "heuristic"):
        target_entropy = "auto"
    else:
        target_entropy = float(te_raw)
    cfg = {
        "model": {
            "state_dim": int(obs_dim), "action_dim": int(J),
            "hidden_dim": args.hidden, "critic_hidden_dim": args.critic_hidden,
            "num_layers": args.layers, "activation": args.activation,
            "log_std_min": args.log_std_min, "log_std_max": args.log_std_max,
            "log_std_init": args.log_std_init,
            "action_low": -float(action_bound), "action_high": float(action_bound),
            "n_critics": 2,
        },
        "training": {
            "gamma": args.gamma, "tau": args.tau, "init_alpha": args.alpha_init,
            "target_entropy": target_entropy, "reward_scale": args.reward_scale,
        },
    }
    agent = build_sac_agent(cfg, device=device)
    optimizers = build_sac_optimizers(agent, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                                      lr_alpha=args.lr_alpha)
    return agent, optimizers, cfg


class Rig:
    """把"环境 + 增益 + 权重"打包：按名字造出四个控制器之一，四者共用同一个 env 与 DR 状态。

    为什么要有这一层：四个控制器的差别必须**只剩**"力矩怎么算"。共用一个 env 实例、一份 DR 注入器、
    一组 PD 增益、一条参考轨迹，就连"重新采样失配"这件事也只发生在 :func:`inject_dr` 里。
    """

    def __init__(self, env, rng: DynamicsRandomizer, kp, kd, tau_limits, args, device: str,
                 joint_names: list[str], nominal: NominalModel | None = None) -> None:
        self.env, self.rng = env, rng
        self.nominal = nominal
        # τ_lim 统一存成 (J,)：BaselineController / ResidualPolicy 里的比较与 clamp 都靠广播，
        # 但 w15_action_scale 需要的是"关节数"（numel），所以不能留 (E,J)。
        self.tau_limits = tau_limits.reshape(-1, tau_limits.shape[-1])[0].clone()
        self.kp, self.kd = kp, kd
        self.args, self.device, self.joint_names = args, device, joint_names
        self.agent_res, self.norm_res, self.agent_res_meta = None, None, {}
        self.agent_pure, self.norm_pure, self.agent_pure_meta = None, None, {}
        self.reward = ResidualReward(
            {"pos": args.wp, "vel": args.wv, "tau": args.wtau, "jerk": args.wj,
             "mag": args.w_mag, "ee": args.w_ee}, args.jerk_on)

    # ---- 奖励 / 权重 ----
    def inject(self, ranges: dict, seed: int | None = None, label: str = "") -> dict:
        """注入一组 DR 区间（**先复位标称模型**，避免 ``apply()`` 的累积效应，见 :class:`NominalModel`）。"""
        return inject_dr(self.env, self.rng, ranges, self.args.seed if seed is None else seed, label,
                         nominal=self.nominal)

    def make_reward(self, args) -> ResidualReward:
        return ResidualReward({"pos": args.wp, "vel": args.wv, "tau": args.wtau, "jerk": args.wj,
                               "mag": args.w_mag, "ee": args.w_ee}, args.jerk_on)

    def _new_base(self, mode: str, args) -> BaselineController:
        return BaselineController(self.kp, self.kd, self.env.E, self.env.J, self.env.traj.device,
                                  mode=mode, effort_limits=self.tau_limits, sign=args.pd_sign,
                                  ff_source=args.ff)

    def bundle(self, name: str, delay_steps: int = 0, args=None,
               allow_missing_agent: bool = False) -> ControllerBundle:
        """按名字造一个 :class:`ControllerBundle`（``args`` 可覆盖，用于消融变体的动作缩放/权重）。

        ``allow_missing_agent=True`` 只给 :func:`train_policy` 用：训练开始时权重还不存在，
        但训练循环自己会把 ``agent`` 显式传给 ``policy.act``，所以那条路径不需要预加载的权重；
        评估路径一律走默认值 —— 权重缺失必须**报错**，不能静默退化成"零残差"。
        """
        args = args or self.args
        dev, env = self.env.traj.device, self.env
        if name in ("pd_dr", "pd_ff_dr"):
            mode = "pd" if name == "pd_dr" else "pd_ff"
            return ControllerBundle(name, self._new_base(mode, args), self.tau_limits, dev,
                                    delay_steps=delay_steps, dt=env.dt)
        if name == "pure_rl":
            if self.agent_pure is None and not allow_missing_agent:
                raise SystemExit("[IN] 纯 RL 控制器还没有权重：先训练（--train_pure_rl）或用 --pure_rl_policy 指定")
            base = self._new_base("zero", args)          # ★ 切断经典控制器：τ_base ≡ 0
            policy = ResidualPolicy(env, base, self.make_reward(args),
                                    w15_action_scale("pure_rl", args, self.kp, self.tau_limits, dev),
                                    self.tau_limits, dev, deterministic=True)
            return ControllerBundle(name, base, self.tau_limits, dev, policy=policy, agent=self.agent_pure,
                                    normalizer=self.norm_pure, delay_steps=delay_steps, dt=env.dt)
        if name == "residual_sac":
            if self.agent_res is None and not allow_missing_agent:
                raise SystemExit("[IN] 残差控制器还没有权重：检查 --policy / --residual_policy_which 指向的 .pt")
            base = self._new_base("pd_ff", args)         # Ours：标称前馈 + PD 兜底
            policy = ResidualPolicy(env, base, self.make_reward(args),
                                    w15_action_scale("residual_sac", args, self.kp, self.tau_limits, dev),
                                    self.tau_limits, dev, deterministic=True)
            return ControllerBundle(name, base, self.tau_limits, dev, policy=policy, agent=self.agent_res,
                                    normalizer=self.norm_res, delay_steps=delay_steps, dt=env.dt)
        raise SystemExit(f"未知控制器 {name!r}；可用：{list(CONTROLLER_ORDER)}")
def parse_names(text: str) -> list[str]:
    """把 ``--controllers a,b,c`` 解析成有序去重的名字列表（按 :data:`CONTROLLER_ORDER` 排序输出）。"""
    want = [s.strip() for s in str(text).split(",") if s.strip()]
    bad = [n for n in want if n not in CONTROLLER_ORDER]
    if bad:
        raise SystemExit(f"未知控制器 {bad}；可用：{list(CONTROLLER_ORDER)}")
    return [n for n in CONTROLLER_ORDER if n in want]


def scenario_label(axis: str, value: float) -> str:
    """压测点的场景名（进 CSV/图例）。"""
    if axis == "payload":
        return f"payload_{float(value):g}kg"
    if axis == "friction":
        return f"friction_{float(value):g}x"
    if axis == "latency":
        return f"latency_{int(round(float(value)))}step"
    return str(axis)


def load_into_rig(rig: Rig, kind: str, path: str, device: str, args=None) -> dict:
    """把一份权重装进 rig（``kind`` ∈ {residual_sac, pure_rl}）：建 agent + 归一化统计量 + 加载权重。

    归一化统计量必须**跟着权重一起**加载（``save_policy`` 里存了 ``normalizer.state_dict()``）：
    观测归一化是训练时在线估的，评估时用它自己的那份，缺了就会得到完全不同的观测尺度。
    """
    args = args or rig.args
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise SystemExit(f"[IN] 找不到策略权重：{path}\n"
                         f"     先训练一轮（--train_pure_rl）或把 --{kind}_policy 指到正确路径。")
    obs_dim, J = rig.env.obs_builder.obs_dim, rig.env.J
    bound = float(args.res_limit)
    if kind == "pure_rl":
        b = float(args.pure_rl_action_bound)
        bound = b if b > 0 else float(rig.tau_limits.max())
    agent, _opts, cfg = build_sac_agent_for(args, obs_dim, J, device, bound)
    normalizer = RunningMeanStd(obs_dim) if args.normalize_obs else None
    ckpt = load_policy(path, agent, normalizer, device)
    out = {"agent": agent, "normalizer": normalizer, "meta": ckpt.get("meta") or {}, "cfg": cfg,
           "path": path, "action_bound": bound, "obs_dim": obs_dim}
    if kind == "pure_rl":
        rig.agent_pure, rig.norm_pure, rig.agent_pure_meta = agent, normalizer, out["meta"]
    else:
        rig.agent_res, rig.norm_res, rig.agent_res_meta = agent, normalizer, out["meta"]
    return out


def train_policy(rig: Rig, kind: str, steps: int, device: str, args, out_dir: str, tag: str) -> dict:
    """训练一条策略（纯 RL / 残差 / 消融变体），返回 summary + 训练曲线 + best/final 权重路径。

    与 W13 的训练循环**同一份代码**（``train_sac``）：64 路并行采样 → 显存回放池 → 每步更新 →
    周期性确定性评估 → 按末端 RMSE 存一份 ``policy_best_ee.pt``。差别只有三点：
    ① 评估用本脚本的统一循环（四控制器同一份指标代码）；② 奖励/动作缩放按传入的 ``args`` 走；
    ③ 不给纯 RL 任何特殊照顾（同观测、同预算、同奖励、同随机种子）。
    """
    env = rig.env
    bundle = rig.bundle(kind, delay_steps=int(args.action_delay), args=args, allow_missing_agent=True)
    bundle.policy.deterministic = False                  # 训练时按分布采样
    obs_dim, J = env.obs_builder.obs_dim, env.J
    bound = float(args.res_limit)
    if kind == "pure_rl":
        b = float(args.pure_rl_action_bound)
        bound = b if b > 0 else float(rig.tau_limits.max())
    agent, optimizers, cfg = build_sac_agent_for(args, obs_dim, J, device, bound)
    buffer = make_replay_buffer(args.buffer, args.buffer_size, obs_dim, J, device,
                                args.bootstrap_on_truncation, args.seed)
    normalizer = RunningMeanStd(obs_dim) if args.normalize_obs else None
    # 把**本次训练**的 agent / 归一化统计量挂到 bundle 上：评估回调里的 run_unified_episode
    # 读的就是 bundle.agent / bundle.normalizer，留着上一段赛程的（例如 105 维的）会直接维度不匹配，
    # 就算维度碰巧相同，用错归一化统计量也会让评估数不对。与 W13 的 evaluate() 用同一份统计量同理。
    bundle.agent, bundle.normalizer = agent, normalizer
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, "policy_final.pt")
    best_path = os.path.join(out_dir, "policy_best_ee.pt")
    history: list[dict] = []
    best = {"ee_cm": float("inf"), "joint_rmse_rad": float("nan"), "env_steps": -1, "path": None}

    print(f"\n[TRAIN] {tag}（{CONTROLLER_DESC.get(kind, kind)}）：{steps} 环境步 | "
          f"观测 {obs_dim} 维 | 动作 ±{bound:g} N·m（逐关节 scale "
          f"{[round(float(v), 3) for v in bundle.policy.scale.tolist()]}）")

    def eval_cb(env_step: int, updates: int, logs_last: dict) -> dict:
        keep = bundle.policy.deterministic
        bundle.policy.deterministic = True
        try:
            res = run_unified_episode(env, bundle, f"{tag}@{env_step}", args.fk_tolerance, rig.joint_names)
        finally:
            bundle.policy.deterministic = keep
        row = {"tag": tag, "env_steps": int(env_step), "updates": int(updates), **res["paper_row"]}
        history.append(row)
        if row["ee_rmse_cm"] < best["ee_cm"] - 1e-12:
            best.update(ee_cm=row["ee_rmse_cm"], joint_rmse_rad=row["joint_rmse_rad"], env_steps=int(env_step))
            best["path"] = save_policy(best_path, agent, normalizer, {
                "kind": kind, "obs_dim": obs_dim, "action_dim": J, "action_bound": bound,
                "res_limit": float(args.res_limit), "history": int(env.obs_builder.H),
                "seed": int(args.seed), "env_steps": int(env_step), "w_ee": float(args.w_ee),
                "note": "best-by-EE（统一评测循环的末端 RMSE 最好的一次）",
            })
            print(f"[BEST] {tag} 末端 RMSE 刷新到 {row['ee_rmse_cm']:.4f} cm"
                  f"（关节 {row['joint_rmse_rad']:.6f} rad）@ {env_step} 环境步")
        return res

    ta = argparse.Namespace(**vars(args))
    ta.total_steps = int(steps)
    if kind == "pure_rl":
        # 热身期随机动作的**幅值口径**：残差策略是 1.0 N·m / 1.5 N·m 动作空间 ≈ 67% 满量程，
        # 纯 RL 的动作空间是 ±τ_lim（j1–j4 87、j5–j7 12），沿用 1.0 N·m 的话连手臂自重都撑不住
        # （热身期就塌到关节限位，后面学的全是"从限位爬回来"）。所以按**满量程比例**给：
        # 0.2·max(τ_lim) ≈ 17.4 N·m（逐关节再被自己的 scale 夹住）——这是纯 RL 唯一一处不可避免的
        # 设计选择，会原样写进 JSON（warmup_res_scale），不藏。
        frac = float(getattr(args, "pure_rl_warmup_frac", 0.2))
        ta.warmup_res_scale = frac * float(rig.tau_limits.max())
        print(f"[TRAIN] 纯 RL 热身幅值 = {frac:g} × max τ_lim = {ta.warmup_res_scale:.2f} N·m"
              f"（残差策略是 {float(args.warmup_res_scale):g} N·m / 1.5 N·m 动作空间）")
    summary = train_sac(env, bundle.policy, agent, optimizers, buffer, normalizer, ta,
                        rig.tau_limits, eval_cb, history)
    save_policy(final_path, agent, normalizer, {
        "kind": kind, "obs_dim": obs_dim, "action_dim": J, "action_bound": bound,
        "res_limit": float(args.res_limit), "history": int(env.obs_builder.H),
        "wj": float(args.wj), "seed": int(args.seed), "env_steps": int(summary["env_steps"]),
        "w_ee": float(args.w_ee), "tag": tag,
    })
    write_csv(os.path.join(out_dir, "history.csv"), history)
    return {"summary": summary, "history": history, "best_ee": best, "policy_final": final_path,
            "policy_best_ee": best_path, "cfg": cfg, "history_len": int(env.obs_builder.H),
            "action_bound": bound}
def verify_dr_idempotent(rig: Rig, args) -> dict:
    """注入幂等性自检：同一组区间**连注入两次**，PhysX 回读值必须逐位相同。

    这是对 ``DynamicsRandomizer.apply()`` 累积效应的**回归测试**：W12/W13 一个进程只注入一次，
    所以这个坑从没暴露（W15 实测连注入三次手部质量 2.173 → 3.681 → 5.224 kg）；有了
    :class:`NominalModel` 的复位，两次注入的回读必须完全一致，否则压测曲线全部作废。
    """
    a = rig.inject(base_dr_ranges(args), args.seed, "idempotency#1")
    b = rig.inject(base_dr_ranges(args), args.seed, "idempotency#2")
    worst, detail = 0.0, {}
    for key in ("hand_mass_kg", "joint_damping_nms_per_rad", "joint_friction_static_nm"):
        sa, sb = (a.get("readback") or {}).get(key) or {}, (b.get("readback") or {}).get(key) or {}
        for stat in ("min", "mean", "max"):
            if stat in sa and stat in sb:
                diff = abs(float(sa[stat]) - float(sb[stat]))
                worst = max(worst, diff)
                detail[f"{key}.{stat}"] = {"first": sa[stat], "second": sb[stat], "abs_diff": diff}
    out = {"available": bool(a.get("applied")), "max_abs_diff": worst, "match": bool(worst <= 1e-12),
           "detail": detail, "note": "两次注入的回读必须逐位相同（否则 apply() 在累积，压测曲线不可信）"}
    print(f"[CHECK] DR 注入幂等性：两次注入回读最大偏差 {worst:.3e} -> "
          f"{'通过（同一组参数 ⇒ 同一个物理模型）' if out['match'] else '不通过：注入在累积'}")
    return out


def ensure_policies(rig: Rig, args, device: str, need: list[str], allow_train: bool) -> dict:
    """确保本次赛程要用的控制器权重都装好了。

    * ``residual_sac``：默认加载仓库里的主结果权重（``logs/w13_residual_sac_ee50k/policy_best_ee.pt``），
      **文件不存在就直接报错** —— 残差控制器是这个工作的主体，不允许"静默退化成零残差"；
    * ``pure_rl``：默认 ``logs/w15_pure_rl/policy_final.pt``，缺失且 ``--train_pure_rl`` 时**先训练一轮**
      （训练前会把 DR 恢复到训练分布，避免上一段赛程留下的压测失配污染训练）。
    """
    out: dict = {}
    if "residual_sac" in need:
        path = args.policy or os.path.join(_REPO, "logs", "w13_residual_sac_ee50k",
                                           f"policy_{args.residual_policy_which}.pt")
        info = load_into_rig(rig, "residual_sac", path, device)
        info.pop("agent", None)
        info.pop("normalizer", None)
        out["residual_policy"] = info
    if "pure_rl" in need:
        path = args.pure_rl_policy or os.path.join(_REPO, "logs", "w15_pure_rl", "policy_final.pt")
        if not os.path.isfile(path):
            if not allow_train:
                raise SystemExit(f"[IN] 纯 RL 权重不存在：{path}；去掉 --no_train_pure_rl 让它先训练一轮")
            rig.inject(base_dr_ranges(args), args.seed, "train_distribution(纯 RL 训练前)")
            trained = train_policy(rig, "pure_rl", int(args.pure_rl_steps), device, args,
                                   os.path.dirname(path), "pure_rl")
            out["pure_rl_training"] = {"steps": int(args.pure_rl_steps), "summary": trained["summary"],
                                       "best_ee": trained["best_ee"], "history": trained["history"],
                                       "policy_final": trained["policy_final"]}
            path = trained["policy_final"]
        info = load_into_rig(rig, "pure_rl", path, device)
        info.pop("agent", None)
        info.pop("normalizer", None)
        out["pure_rl_policy"] = info
    return out


def eval_set(rig: Rig, names: list[str], args, scenario: str, delay_steps: int = 0,
             traces: set[str] | None = None, args_override=None) -> dict[str, dict]:
    """在**当前动力学 + 当前延迟**下跑一组控制器（顺序固定为 :data:`CONTROLLER_ORDER`）。"""
    out: dict[str, dict] = {}
    for name in names:
        bundle = rig.bundle(name, delay_steps=int(delay_steps), args=args_override)
        tr = None
        if traces and name in traces:
            want = int(round(float(args.trace_windows) / rig.env.dt))
            if want > 0:
                tr = EpisodeTrace(rig.env.J, min(want, rig.env.steps), int(args.trace_env),
                                  rig.env.traj.device)
        out[name] = run_unified_episode(rig.env, bundle, scenario, args.fk_tolerance, rig.joint_names, trace=tr)
    return out


def stage_baselines(rig: Rig, args, device: str) -> dict:
    """Table 1：四个控制器在**训练分布**上的全周期指标（＋与 W12 ``run_episode`` 的逐字段对账）。

    ``pure_rl`` 没有权重时先训练一轮（同预算、同奖励、同观测、同种子），否则直接用现成权重。
    """
    env, rng = rig.env, rig.rng
    names = parse_names(args.controllers)
    dr = rig.inject(base_dr_ranges(args), args.seed, "train_distribution")
    dr_state = {"on": bool(rng.enabled)}
    out: dict = {"stage": "baselines", "scenario": "train_distribution", "controllers": names, "dr": dr}

    # ---- 权重：残差策略 + 纯 RL 策略（缺则训练；训练前必须先把失配恢复到训练分布）----
    out.update(ensure_policies(rig, args, device, names, allow_train=bool(args.train_pure_rl)))
    if out.get("pure_rl_training"):
        dr = rig.inject(base_dr_ranges(args), args.seed, "train_distribution(纯 RL 训练后复位)")
        out["dr_after_training"] = dr

    # ---- W12 老路径（run_episode）作为对账参考 ----
    refs: dict[str, dict] = {}
    if args.verify_w12_path:
        for name in names:
            if name in ("pd_dr", "pd_ff_dr"):
                refs[name] = run_baseline_scenario(env, rng, name, args, rig.kp, rig.kd,
                                                   rig.tau_limits, rig.joint_names, dr_state)

    # ---- 统一循环：四个控制器 ----
    traces = set(parse_names(args.trace_controllers)) if float(args.trace_windows) > 0 else set()
    results = eval_set(rig, names, args, "train_distribution", delay_steps=int(args.action_delay),
                       traces=traces)
    out["results"] = results
    out["w12_path_reference"] = refs
    out["self_checks"] = {
        "unified_vs_w12_path": verify_unified_vs_w12(refs, results),
        "fk": env.fk_status(args.fk_tolerance),
        "residual_policy_meta": rig.agent_res_meta,
        "pure_rl_policy_meta": rig.agent_pure_meta,
    }
    out["traces"] = {k: v["trace"] for k, v in results.items() if v.get("trace")}
    return out


def stage_sweep(rig: Rig, args, axis: str, values: list[float]) -> dict:
    """Figure 3 的一条压测轴：逐点注入失配 → 跑一组控制器 → 记录"注入证据 + 指标"。"""
    names = parse_names(args.stress_controllers)
    # 单独跑某一条压测轴时也要先把权重装好（否则 Rig.bundle 会因为 agent 为空而报错）
    pre = ensure_policies(rig, args, device=rig.device, need=names, allow_train=bool(args.train_pure_rl))
    out: dict = {"stage": axis, "axis": axis, "controllers": names, "points": [],
                 "policies": {k: v for k, v in pre.items() if not k.endswith("_training")}}
    if pre.get("pure_rl_training"):
        out["pure_rl_training"] = pre["pure_rl_training"]
    for v in values:
        spec = stress_ranges(args, axis, v)
        scenario = scenario_label(axis, v)
        delay = int(round(float(v))) if axis == "latency" else 0
        dr = rig.inject(spec["ranges"], args.seed, scenario)
        results = eval_set(rig, names, args, scenario, delay_steps=delay)
        point = {"scenario": scenario, "sweep": axis, "sweep_value": float(v), "delay_steps": delay,
                 "dr_ranges": {k: [float(x) for x in val] for k, val in spec["ranges"].items()},
                 "dr_note": spec["note"], "dr": dr,
                 "results": results,
                 "summary": {n: r["paper_row"] for n, r in results.items()}}
        base = results.get("residual_sac") or results.get("pd_ff_dr")
        if base is not None:
            for n, r in results.items():
                ref = results.get("pd_ff_dr")
                if ref is not None:
                    point.setdefault("vs_pd_ff_dr", {})[n] = {
                        "ee_rmse_cm_delta": r["paper_row"]["ee_rmse_cm"] - ref["paper_row"]["ee_rmse_cm"],
                        "ee_rmse_cm_ratio": r["paper_row"]["ee_rmse_cm"] / max(ref["paper_row"]["ee_rmse_cm"], 1e-12),
                    }
        out["points"].append(point)
        print(f"\n[SWEEP] {axis} = {v:g} | " + " | ".join(
            f"{n}: 末端 {r['paper_row']['ee_rmse_cm']:.4f} cm" for n, r in results.items()))
    return out
ABLATION_PLAN: dict[str, dict] = {
    "history": {
        "title": "消融 1：时序历史长度 H（单帧 vs 3 帧 vs 5 帧）",
        "variants": [("H=1", {"history": 1}), ("H=3(Ours)", {}), ("H=5", {"history": 5})],
    },
    "res_limit": {
        "title": "消融 2：残差动作限幅 |Δτ|（±0.5 / ±1.5(Ours) / ±5.0 N·m）",
        "variants": [("res=0.5Nm", {"res_limit": 0.5}), ("res=1.5Nm(Ours)", {}), ("res=5.0Nm", {"res_limit": 5.0})],
    },
    "jerk": {
        "title": "消融 3：抖动惩罚项 w_j（0 vs 0.01(Ours)）",
        "variants": [("w_j=0", {"wj": 0.0}), ("w_j=0.01(Ours)", {})],
    },
}


def stage_ablation(rig: Rig, args, device: str) -> dict:
    """Table 2：三组消融（历史长度 / 残差限幅 / jerk 惩罚）。

    每个变体都在**同一失配分布**下训练到同一预算（``--ablation_steps``，默认 = ``--total_steps``），
    再用同一个统一评测循环评估。标 "Ours" 的那一格默认**复用主结果权重**（``--no_ablation_reuse_main``
    可强制重训），这样表格里那一行与本仓库 W13 的主结果是同一个模型，不存在"消融实验里的 Ours
    和主结果不是同一个东西"这种低级问题。
    """
    env = rig.env
    rig.inject(base_dr_ranges(args), args.seed, "train_distribution")
    steps = int(args.ablation_steps or args.total_steps)
    main_H = int(args.history)
    main_path = args.policy or os.path.join(_REPO, "logs", "w13_residual_sac_ee50k", "policy_best_ee.pt")
    out: dict = {"stage": "ablation", "steps": steps, "delay_steps": int(args.action_delay),
                 "main_policy": os.path.abspath(main_path), "dims": {}}

    for dim in [s.strip() for s in str(args.ablation_dims).split(",") if s.strip()]:
        if dim not in ABLATION_PLAN:
            raise SystemExit(f"未知消融维度 {dim!r}；可用：{list(ABLATION_PLAN)}")
        rows: list[dict] = []
        print(f"\n{'=' * 94}\n[ABLATION] {ABLATION_PLAN[dim]['title']}\n{'=' * 94}")
        for label, overrides in ABLATION_PLAN[dim]["variants"]:
            ta = argparse.Namespace(**vars(args))
            ta.history, ta.res_limit, ta.wj = main_H, float(args.res_limit), float(args.wj)
            for key, val in overrides.items():
                setattr(ta, key, val)
            if "history" in overrides or env.obs_builder.H != int(ta.history):
                env.obs_builder = ObservationBuilder(env.E, env.J, int(ta.history), ta.history_warmup,
                                                     env.traj.device)
                print(f"[ABLATION] 观测历史改成 H={ta.history}（观测维 {env.obs_builder.obs_dim}）")
            reuse = (not overrides) and bool(args.ablation_reuse_main)
            row: dict = {"label": label, "dim": dim, "overrides": overrides, "reused_main": bool(reuse),
                         "history": int(ta.history), "res_limit": float(ta.res_limit), "wj": float(ta.wj)}
            if reuse:
                load_into_rig(rig, "residual_sac", main_path, device, args=ta)
            else:
                tag = f"abl_{dim}_{label.replace('=', '').replace('.', '').replace('(', '_').replace(')', '')}"
                trained = train_policy(rig, "residual_sac", steps, device, ta,
                                       os.path.join(_REPO, "logs", tag), tag)
                row["training"] = trained["summary"]
                row["train_curve"] = trained["history"]
                row["best_ee_training"] = trained["best_ee"]
                load_into_rig(rig, "residual_sac", trained["policy_final"], device, args=ta)
                rig.inject(base_dr_ranges(args), args.seed, f"train_distribution({tag} 训练后)")
            bundle = rig.bundle("residual_sac", delay_steps=int(args.action_delay), args=ta)
            ev = run_unified_episode(env, bundle, f"ablation_{dim}_{label}", args.fk_tolerance, rig.joint_names)
            row["eval"] = ev
            row["paper_row"] = ev["paper_row"]
            row["residual_stats"] = ev["residual_stats"]
            if not reuse:
                # 训练变体额外评估"训练期最好"的那份权重（表格里报 final，附录里给 best）
                load_into_rig(rig, "residual_sac", trained["policy_best_ee"], device, args=ta)
                bundle_best = rig.bundle("residual_sac", delay_steps=int(args.action_delay), args=ta)
                ev_best = run_unified_episode(env, bundle_best, f"ablation_{dim}_{label}(best_ee)",
                                              args.fk_tolerance, rig.joint_names)
                row["eval_best_ee"] = ev_best["paper_row"]
            rows.append(row)
            print(f"[ABLATION] {label:16s} 关节 RMSE {row['paper_row']['joint_rmse_rad']:.6f} rad | "
                  f"末端 {row['paper_row']['ee_rmse_cm']:.4f} cm | jerk {row['paper_row']['jerk_nm2']:.3f} | "
                  f"饱和 {row['paper_row']['sat_step_frac']:.3f}"
                  f"{'（复用主结果权重）' if reuse else '（本次训练）'}")
        out["dims"][dim] = {"title": ABLATION_PLAN[dim]["title"], "rows": rows}

    env.obs_builder = ObservationBuilder(env.E, env.J, main_H, args.history_warmup, env.traj.device)
    # 收尾把 rig 恢复到"主结果权重 + 主历史长度"的干净状态：否则消融留下的 35 维归一化统计量
    # 会被后续任何一次评估复用（本赛程通常最后跑，但不想留这种定时炸弹）。
    load_into_rig(rig, "residual_sac", main_path, device, args=args)
    print(f"[ABLATION] 收尾：rig 已恢复到主结果权重（{os.path.basename(main_path)}，H={main_H}）")
    return out

##
# ============================================================================
# [W15 新增 · 模块 C] 论文大表落盘（CSV/JSON）+ 终端汇总表 + main()
# ============================================================================
##
PAPER_COLUMNS = (
    "stage", "scenario", "sweep", "sweep_value", "delay_steps", "controller", "note",
    "joint_rmse_rad", "ee_rmse_cm", "ee_max_cm", "mean_tau_norm_nm", "jerk_nm2", "sat_step_frac",
    "joint_rmse_worst_env_rad", "ee_rmse_max_over_envs_cm", "mean_abs_tau_overall_nm",
    "task_reward_per_step", "mean_abs_delta_tau_nm", "tanh_sat_frac", "action_scale_nm",
    "per_env_ee_rmse_cm",
)


def flat_row(stage: str, scenario: str, sweep, sweep_value, delay_steps, controller: str,
             res: dict, note: str = "") -> dict:
    """把一次评估摊平成论文大表的一行（列见 :data:`PAPER_COLUMNS`）。

    ``per_env_ee_rmse_cm`` 用 ``;`` 拼成字符串留在同一行里：Figure 2 的箱线图要的就是这 64 个数，
    单独开一个文件反而多一个"表对不上"的机会。
    """
    pr = res["paper_row"]
    rs = res.get("residual_stats") or {}
    scale = res["controller_meta"].get("action_scale_nm") or []
    return {
        "stage": stage, "scenario": scenario, "sweep": sweep, "sweep_value": sweep_value,
        "delay_steps": int(delay_steps), "controller": controller, "note": note,
        "joint_rmse_rad": pr["joint_rmse_rad"], "ee_rmse_cm": pr["ee_rmse_cm"],
        "ee_max_cm": pr["ee_max_cm"], "mean_tau_norm_nm": pr["mean_tau_norm_nm"],
        "jerk_nm2": pr["jerk_nm2"], "sat_step_frac": pr["sat_step_frac"],
        "joint_rmse_worst_env_rad": pr["joint_rmse_worst_env_rad"],
        "ee_rmse_max_over_envs_cm": pr["ee_rmse_max_over_envs_cm"],
        "mean_abs_tau_overall_nm": pr["mean_abs_tau_overall_nm"],
        "task_reward_per_step": pr["task_reward_per_step"],
        "mean_abs_delta_tau_nm": rs.get("mean_abs_delta_tau", ""),
        "tanh_sat_frac": rs.get("tanh_saturation_frac", ""),
        "action_scale_nm": ";".join(f"{float(v):g}" for v in scale),
        "per_env_ee_rmse_cm": ";".join(f"{float(v):g}" for v in
                                       res["metrics"]["cartesian_error_cm"]["per_env_rmse_cm"]),
    }


def collect_table_rows(result: dict) -> tuple[list[dict], list[dict]]:
    """任意赛程的结果 → ``(大表行, 逐 env 行)``（逐 env 行供箱线图 / 附录用）。"""
    rows: list[dict] = []
    per_env: list[dict] = []
    stage = str(result.get("stage", "?"))

    def _add(scenario: str, sweep: str, value, delay: int, controller: str, res: dict, note: str = "") -> None:
        rows.append(flat_row(stage, scenario, sweep, value, delay, controller, res, note))
        ee = res["metrics"]["cartesian_error_cm"]["per_env_rmse_cm"]
        jr = res["metrics"]["joint_rmse_rad"]["per_env_rmse_rad"]
        for i, (a, b) in enumerate(zip(ee, jr)):
            per_env.append({"stage": stage, "scenario": scenario, "sweep": sweep, "sweep_value": value,
                            "controller": controller, "env_idx": i, "ee_rmse_cm": a, "joint_rmse_rad": b})

    if stage == "baselines":
        for name, res in result["results"].items():
            _add(result["scenario"], "train_distribution", "", res["controller_meta"]["delay"]["steps"],
                 name, res)
    elif stage == "ablation":
        for dim, blk in result["dims"].items():
            for row in blk["rows"]:
                _add(f"{dim}:{row['label']}", dim, "ours" if not row["overrides"] else str(row["overrides"]),
                     int(result.get("delay_steps", 0)), f"residual_sac[{row['label']}]", row["eval"],
                     note=("复用主结果权重" if row["reused_main"] else f"本次训练 {result['steps']} 环境步"))
    else:
        for pt in result.get("points", []):
            for name, res in pt["results"].items():
                _add(pt["scenario"], str(result.get("axis", stage)), pt["sweep_value"], pt["delay_steps"],
                     name, res, note=";".join(f"{k}={v}" for k, v in pt["dr_note"].items()))
    return rows, per_env


def write_table_csv(path: str, rows: list[dict], append: bool = True) -> str:
    """把大表行并入 CSV：按 ``(stage, scenario, controller)`` 去重，**本次结果覆盖同名旧行**。

    为什么是"并入"而不是"覆盖文件"：三条压测轴 + Table 1 是分四次命令跑的（每次都要重启 Kit），
    并进来才能让后面 :mod:`export_paper_figures` 一次读到完整矩阵；同时旧行不会被抹掉。
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    existing: list[dict] = []
    if append and os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    key = lambda r: (str(r.get("stage", "")), str(r.get("scenario", "")), str(r.get("controller", "")))  # noqa: E731
    fresh = {key(r) for r in rows}
    merged = [r for r in existing if key(r) not in fresh] + rows
    cols: list[str] = [c for c in PAPER_COLUMNS]
    for r in merged:
        cols.extend(c for c in r if c not in cols)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in merged:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"[OUT] 论文大表 CSV：{path}（旧 {len(existing)} 行 + 本次 {len(rows)} 行 = {len(merged)} 行）")
    return path
def write_per_env_csv(path: str, rows: list[dict], append: bool = True) -> str:
    """逐 env 指标 CSV（Figure 2 的箱线图 / 附录）：按 ``(stage, scenario, controller, env_idx)`` 去重。"""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return path
    existing: list[dict] = []
    if append and os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    key = lambda r: (str(r.get("stage", "")), str(r.get("scenario", "")), str(r.get("controller", "")),  # noqa: E731
                     str(r.get("env_idx", "")))
    fresh = {key(r) for r in rows}
    merged = [r for r in existing if key(r) not in fresh] + rows
    cols: list[str] = []
    for r in merged:
        cols.extend(c for c in r if c not in cols)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in merged:
            writer.writerow({c: r.get(c, "") for c in cols})
    print(f"[OUT] 逐 env CSV：{path}（{len(merged)} 行，箱线图用）")
    return path


def print_summary_table(rows: list[dict]) -> None:
    """终端打印可直接复制的对比大表（Paper 版 LaTeX 见 ``experiments/export_paper_figures.py``）。"""
    print("\n" + "=" * 132)
    print("[TABLE] W15 实验矩阵（同一失配分布 / 同一轨迹 / 同一指标代码；joint RMSE [rad]、末端 [cm]、"
          "‖τ‖ [N·m]、jerk [N·m²]）")
    print(f"{'stage':11s} {'scenario':20s} {'controller':22s} {'joint':>9s} {'ee RMSE':>9s}"
          f" {'ee max':>8s} {'|tau|':>8s} {'jerk':>9s} {'sat':>6s}")
    print("-" * 132)
    for r in rows:
        print(f"{str(r['stage'])[:11]:11s} {str(r['scenario'])[:20]:20s} {str(r['controller'])[:22]:22s} "
              f"{float(r['joint_rmse_rad']):9.6f} {float(r['ee_rmse_cm']):9.4f} {float(r['ee_max_cm']):8.4f} "
              f"{float(r['mean_tau_norm_nm']):8.3f} {float(r['jerk_nm2']):9.3f} {float(r['sat_step_frac']):6.3f}")
    print("=" * 132)
    print("[TABLE] 下一步：python experiments/export_paper_figures.py  →  Figure 1/2/3 + LaTeX 表格")


def resolve_stages(stage: str, do_ablation: bool) -> list[str]:
    """``--stage`` → 实际要跑的赛程序列（``all`` 默认**不含** ablation：9 个变体 × 训练预算 ≈ 1.5 h）。"""
    if str(stage) == "all":
        seq = ["baselines", "payload", "friction", "latency"]
        if do_ablation:
            seq.append("ablation")
        return seq
    return [str(stage)]


def parse_values(text: str) -> list[float]:
    """``--payload_list 0,1.5,3`` → ``[0.0, 1.5, 3.0]``。"""
    return [float(s) for s in str(text).split(",") if s.strip()]


def default_out_paths(args) -> dict:
    """默认产物路径（全部落在 ``docs/results/`` 下，文件名以 ``w15_`` 开头）。"""
    res_dir = os.path.join(_REPO, "docs", "results")
    return {
        "json": args.out_json or os.path.join(res_dir, f"w15_{args.stage}.json"),
        "table": args.table_csv or os.path.join(res_dir, "w15_benchmarks.csv"),
        "per_env": args.per_env_csv or os.path.join(res_dir, "w15_per_env.csv"),
        "figures": args.figures_dir or os.path.join(_REPO, "docs", "figures"),
    }
def main() -> dict:
    """W15 主流程：搭 W12/W13 的环境 → 注入失配 → 按 ``--stage`` 跑论文实验矩阵 → 落盘 JSON/CSV。

    环境搭建顺序与 W13 ``main()`` 逐行一致（**在注入失配之前**定 PD 增益、存标称重力模型），
    所以这里跑出来的 pd_dr / pd_ff_dr 与 W12/W13 报告里的数可以直接横比。
    """
    device = args_cli.device or "cuda:0"
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    torch.cuda.manual_seed_all(int(args_cli.seed))
    torch.cuda.reset_peak_memory_stats()
    stages = resolve_stages(args_cli.stage, args_cli.do_ablation)
    paths = default_out_paths(args_cli)

    print("=" * 94)
    print(f"[ENV] torch {torch.__version__} | warp {wp.__version__} | isaacsim 6.1.0")
    print(f"[ENV] device = {device} | {torch.cuda.get_device_name(0)}")
    free_start, total_mem = torch.cuda.mem_get_info()
    print(f"[ENV] GPU memory total = {total_mem / 1024**3:.2f} GiB, free = {free_start / 1024**3:.2f} GiB")
    print(f"[W15] 赛程 = {stages} | 控制器(baselines) = {args_cli.controllers} | "
          f"压测控制器 = {args_cli.stress_controllers}")
    print(f"[W15] 产物：JSON {paths['json']} | 大表 CSV {paths['table']} | 逐 env CSV {paths['per_env']}")

    # ---- 场景（与 W12/W13 同一套：纯力矩 Franka + 可选地面）----
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=device))
    scene = InteractiveScene(FrankaTrackingSceneCfg(num_envs=args_cli.num_envs,
                                                   env_spacing=args_cli.env_spacing))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()
    print(f"[SCENE] num_envs={scene.num_envs} | num_joints={robot.num_joints} | num_bodies={robot.num_bodies} | "
          f"gravity={not args_cli.no_gravity} | dt={dt:.5f} s（{1.0 / dt:.0f} Hz）")

    joint_ids, joint_names = _resolve_joint_ids(robot, args_cli.joint_expr)
    J = len(joint_ids)
    tau_limits = robot.data.joint_effort_limits.torch[:, joint_ids].to(dtype=torch.float32)
    pos_limits = robot.data.joint_pos_limits.torch[0, joint_ids].to(dtype=torch.float32)
    print(f"[MDP] 被控关节（{J} 个）：{joint_names}")
    print(f"[MDP] 力矩上限 [N·m] = {[round(float(v), 2) for v in tau_limits[0].tolist()]}")

    # ---- 参考轨迹 / 观测 / 奖励（与 W12/W13 同名同默认 ⇒ MDP 定义一致）----
    traj_seed = args_cli.traj_seed if args_cli.traj_seed >= 0 else args_cli.seed
    traj = TrajectoryGenerator(
        scene.num_envs, joint_names, pos_limits, list(args_cli.traj_center), tuple(args_cli.traj_amp),
        tuple(args_cli.traj_freq), args_cli.harmonics, args_cli.traj_margin, traj_seed, device,
    )
    traj_info, limit_info = traj.summary(), traj.check_limits(args_cli.steps, dt)
    print(f"[TRAJ] K={traj_info['harmonics']} | seed={traj_seed} | q0={traj_info['center_q0_rad']}")
    print(f"[TRAJ] A={traj_info['amp_rad_clamped']} rad | f={traj_info['freq_hz']} Hz | φ={traj_info['phase_rad']}")
    print(f"[TRAJ] 限位检查：max|q_d|={limit_info['max_abs_qd_rad']:.4f} rad，"
          f"离限位最近 {limit_info['min_margin_to_joint_limit_rad']:.4f} rad -> "
          f"{'不越限' if limit_info['inside_limits'] else '越限（必须调小 --traj_amp）'}")

    obs_builder = ObservationBuilder(scene.num_envs, J, args_cli.history, args_cli.history_warmup, device)
    reward_fn = MultiObjectiveReward({"pos": args_cli.wp, "vel": args_cli.wv, "tau": args_cli.wtau,
                                      "jerk": args_cli.wj}, device)
    reward_res = ResidualReward({"pos": args_cli.wp, "vel": args_cli.wv, "tau": args_cli.wtau,
                                 "jerk": args_cli.wj, "mag": args_cli.w_mag, "ee": args_cli.w_ee},
                                args_cli.jerk_on)
    print(f"[OBS] 单帧 {obs_builder.frame_dim} 维 × H={obs_builder.H} 帧 = {obs_builder.obs_dim} 维（策略输入）")
    print(f"[REW] 策略奖励 = {reward_res.describe()['formula']}")
    # ---- FK（末端笛卡尔误差的度量基础，与 W12/W13 同一份实现）----
    fk = PandaFK(robot, device, args_cli.ee_body, args_cli.urdf or None)
    pre = fk.check(robot.data.joint_pos.torch, robot.data.body_link_pos_w.torch[:, 0, :],
                   robot.data.body_link_quat_w.torch[:, 0, :], robot.data.body_link_pos_w.torch)
    fk_ok = pre["max_err_m"] <= args_cli.fk_tolerance
    print(f"[FK] 运动链来源 = {fk.source} | 链 {fk.num_steps} 段 | 预检 max 残差 = {pre['max_err_m']:.3e} m -> "
          f"{'通过' if fk_ok else '不通过（末端误差不可信）'}")

    # ---- DR 注入器（W11 的失配：未知负载 + 摩擦/阻尼 + 连杆质量）+ MDP 环境 ----
    rng = DynamicsRandomizer(
        robot, scene.num_envs, device,
        enabled=not args_cli.no_randomization, seed=args_cli.seed, joint_expr=args_cli.joint_expr,
        payload_body=args_cli.payload_body, payload_com=args_cli.payload_com,
        payload_mode=args_cli.payload_mode, ranges=base_dr_ranges(args_cli),
    )
    env = TrajectoryTrackingMDP(scene, sim, robot, traj, obs_builder, reward_fn, fk, joint_ids, dt,
                                args_cli.steps, fk_check=not args_cli.no_fk_check,
                                init_at_reference=args_cli.init_at_reference)

    # ---- PD 增益：在参考中心 q0 上读 M(q0) 自动定（与 W13 完全相同的顺序，且在注入之前）----
    q_center = robot.data.default_joint_pos.torch.clone()
    q_center[:, joint_ids] = traj.q0[0]
    robot.write_joint_state_to_sim_index(position=q_center, velocity=torch.zeros_like(q_center))
    scene.write_data_to_sim()
    scene.update(dt)
    if args_cli.gains == "manual":
        kp = _gain_tensor(args_cli.kp, J, device)
        kd = _gain_tensor(args_cli.kd, J, device)
        i_diag = auto_gains(robot, joint_ids, args_cli.wn, args_cli.zeta, device)[2]
    else:
        kp, kd, i_diag = auto_gains(robot, joint_ids, args_cli.wn, args_cli.zeta, device,
                                    mweighted=(args_cli.gains == "mweighted"))
    print(f"[GAIN] {args_cli.gains}：ωn = {args_cli.wn:g} rad/s，ζ = {args_cli.zeta:g} | "
          f"kp = {_gain_brief(kp)} | kd = {_gain_brief(kd)}")
    lam_p, lam_d = stability_margins(robot, joint_ids, kp, kd, dt)
    print(f"[GAIN] 离散裕度：λmax(M⁻¹Kp)·dt² = {float(lam_p.max()):.4f}（<4）；"
          f"λmax(M⁻¹Kd)·dt = {float(lam_d.max()):.4f}（<2）")

    # ---- 标称重力模型（前馈项）：读**注入前**的连杆质量/质心 ----
    env.set_nominal_model(
        robot.data.body_mass.torch.to(dtype=torch.float32).clone(),
        robot.data.body_com_pose_b.torch.to(dtype=torch.float32)[:, :, :3].clone(),
    )
    ff_nom, ff_oracle = env.gravity_ff("nominal"), env.gravity_ff("oracle")
    ff_err = float((ff_nom - ff_oracle).abs().max())
    print(f"[FF] 标称重力模型自校验（注入前）：max|ΔG| = {ff_err:.4f} N·m -> "
          f"{'通过' if ff_err <= 0.5 else '不通过'}")

    # ---- 标称模型快照 + Rig（起点 = 训练分布；每个赛程自己会重新注入，互不影响）----
    # 为什么要有这个快照：DynamicsRandomizer.apply() 每次都把"当前"PhysX 状态当成 nominal 再乘采样系数
    # ⇒ 同一进程里连续注入会累积（W12/W13 只注入一次所以没暴露，W15 的压测要注入十几次）。
    nominal = NominalModel(robot, args_cli.payload_body)
    print(f"[DR] 标称模型快照：手部质量 {nominal.hand_mass_kg:.6f} kg、质心 z {nominal.hand_com_z:.5f} m"
          f"（每次注入前复位，避免 apply() 的累积效应）")
    rig = Rig(env, rng, kp, kd, tau_limits, args_cli, device, joint_names, nominal=nominal)
    print(f"[RIG] 残差动作空间 = {args_cli.res_scale_mode} / 硬上限 {args_cli.res_limit:g} N·m | "
          f"纯 RL 动作空间 = 逐关节 τ_lim {[round(float(v), 2) for v in tau_limits[0].tolist()]}")
    dr_train = rig.inject(base_dr_ranges(args_cli), args_cli.seed, "train_distribution")
    dr_idem = verify_dr_idempotent(rig, args_cli)
    dr_train = rig.inject(base_dr_ranges(args_cli), args_cli.seed, "train_distribution(起点)")
    # ================= 赛程 =================
    results: dict[str, dict] = {}
    for st in stages:
        t0 = time.perf_counter()
        print(f"\n{'#' * 94}\n# [STAGE {st}] 开始\n{'#' * 94}")
        if st == "baselines":
            results[st] = stage_baselines(rig, args_cli, device)
        elif st in ("payload", "friction", "latency"):
            attr = {"payload": "payload_list", "friction": "friction_mult",
                    "latency": "latency_steps"}[st]
            results[st] = stage_sweep(rig, args_cli, st, parse_values(getattr(args_cli, attr)))
        elif st == "ablation":
            results[st] = stage_ablation(rig, args_cli, device)
        else:
            raise SystemExit(f"未知赛程 {st!r}")
        results[st]["wall_s"] = round(time.perf_counter() - t0, 2)
        print(f"[STAGE] {st} 用时 {results[st]['wall_s']:.1f} s")

    # ================= 摊平 + 落盘 =================
    rows: list[dict] = []
    per_env_rows: list[dict] = []
    for _st, res in results.items():
        r, p = collect_table_rows(res)
        rows += r
        per_env_rows += p
    table_csv = write_table_csv(paths["table"], rows, bool(args_cli.append_csv))
    per_env_csv = write_per_env_csv(paths["per_env"], per_env_rows, bool(args_cli.append_csv))
    print_summary_table(rows)

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    free_now, _ = torch.cuda.mem_get_info()
    result = {
        "config": _jsonable(vars(args_cli)),
        "resolved": {
            "device": device, "stages": stages, "json": os.path.abspath(paths["json"]),
            "table_csv": table_csv, "per_env_csv": per_env_csv, "figures_dir": paths["figures"],
            "obs_dim": obs_builder.obs_dim, "num_joints": J, "joint_names": joint_names,
            "tau_limits_nm": [round(float(v), 3) for v in tau_limits[0].tolist()],
            "free_gib_start": round(free_start / 1024**3, 3),
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "trajectory": traj_info, "trajectory_limits": limit_info,
        "observation": obs_builder.describe(),
        "reward": {"residual_policy_reward": reward_res.describe(),
                   "env_native_reward_w12_convention": reward_fn.describe()},
        "fk": {"source": fk.source, "num_steps": fk.num_steps, "precheck": pre,
               "tolerance_m": float(args_cli.fk_tolerance), "valid": bool(fk_ok)},
        "gains": {"mode": args_cli.gains, "wn": args_cli.wn, "zeta": args_cli.zeta,
                  "kp_diag": _gain_brief(kp), "kd_diag": _gain_brief(kd),
                  "mass_matrix_diag_nominal_kgm2": [round(float(v), 5) for v in i_diag.tolist()],
                  "stability_lambda_kp_dt2": float(lam_p.max()), "stability_lambda_kd_dt": float(lam_d.max())},
        "dr_training_ranges": {k: [float(x) for x in v] for k, v in base_dr_ranges(args_cli).items()},
        "dr_training_injection": dr_train,
        "gravity_model_check_max_abs_diff_nm": ff_err,
        "self_checks": {"dr_injection_idempotent": dr_idem, "nominal_model": nominal.describe(),
                        "dr_training_injection": dr_train},
        "stages": results,
        "table_rows": rows,
        "gpu": {"peak_allocated_gib": round(allocated, 4), "free_gib_after": round(free_now / 1024**3, 3)},
    }
    out_json = os.path.abspath(paths["json"])
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[OUT] 结果已写入 {out_json}")
    print(f"[OUT] 显存：peak allocated {allocated:.3f} GiB | 跑完剩余 {free_now / 1024**3:.2f} GiB")
    print("=" * 94)
    return result

if __name__ == "__main__":
    main()
    simulation_app.close()

