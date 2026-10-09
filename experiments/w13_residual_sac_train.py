#!/usr/bin/env python
"""W13/W14 Day 4–5：双层残差控制 —— 标称前馈+PID 兜底 + SAC 残差策略（复用 W12 环境与 W9 的 SAC）。

.. code-block:: text

    q_d, q̇_d ─┬─> 底层经典控制器（标称重力前馈 + M 加权 PD；控制器**不知道**负载/摩擦）
               │        τ_base = Ĝ(q) + Kp·(−e_q) + Kd·(−ė_q)        e_q = q − q_d
               │
               └─> 上层 SAC 残差策略（105 维 = 3 帧 × 35 维时序观测）
                            └─> Δτ ∈ [−15, +15] N·m（tanh 动作头缩放）
                                          │
        τ_exec = clip(τ_base + Δτ, −τ_lim, τ_lim) ──> Isaac Lab（W11 注入的物理失配就在这一步上）

W12 把"环境 + 开环基线"钉死了：注入失配后 **前馈+PID 1.3259 cm / 纯 PD 2.3016 cm**（标称动力学
0.6269 cm / 1.4367 cm，零力矩 73.18 cm）。W13/W14 把中间这 1–2 cm 交给 RL 去补：
**基座始终在**，策略只学"标称模型解释不了的那部分" —— 未知负载的静态重力矩、关节摩擦/阻尼、
连杆质量失配。这就是任务书的双层（residual）架构。

一、控制律（两层）
------------------
======  ==============================================================================
底层      ``τ_base = Ĝ(q) + Kp·(q_d−q) + Kd·(q̇_d−q̇)``：标称重力前馈（**注入前**读到的
          质量/质心，所以负载越大前馈越不准）+ M 加权 PD（``Kp=ωn²M̂(q0)``、``Kd=2ζωnM̂(q0)``，
          ωn=30 rad/s、ζ=1；矩阵增益，见 W12 §3.2）。**基座全程固定、不参与训练** ——
          它是安全兜底，也是 A/B 对照的另一半。
上层      SAC 连续高斯策略（W9 ``src/sac.py`` 的 ``SquashedGaussianActor``）：输入 W12 的
          105 维观测 ``s_t``，输出 ``Δτ ∈ [−15, 15] N·m``。
执行      ``τ_exec = clip(τ_base + Δτ, −τ_lim, τ_lim)``（**必须显式 clip**：否则"算出来的力矩"
          与"真机/PhysX 能给的力矩"不是一回事，饱和率会被隐藏 —— W12 §3.3 已经踩过）。
======  ==============================================================================

.. warning::
   任务书写的是 ``τ_base = Ĝ(q) + Kp·e_q + Kd·ė_q`` 而 ``e_q = q − q_d``：把定义代进去就是
   **正反馈**（关节掉下去就往下推）。W12 §3.1/§4.4 已实测复现（``--pd_sign literal`` →
   1.802 rad / 57.47 cm / 99% 撞力矩上限）。本脚本默认走标准 PD 口径（``stabilizing``，
   即 ``Kp(q_d−q) + Kd(q̇_d−q̇)``），``--pd_sign literal`` 可复现那个坑。
   观测与奖励里的 ``e_q`` 定义不变（``e_q = q − q_d``），所以这个修正只影响"控制器怎么用误差"。

二、强化学习奖励（与任务书逐字对齐）
------------------------------------
.. code-block:: text

    r_t = −10·‖e_q‖² − 0.1·‖ė_q‖² − 0.001·‖τ_exec‖² − 0.01·‖Δτ_t − Δτ_{t−1}‖²
          └─ w_p ──┘   └─ w_v ──┘   └── w_τ ─────┘   └──── w_j（残差抖动）────┘

第 4 项（jerk）**默认惩罚残差动作 Δτ** 而不是 τ_exec —— 任务书明确写的是 Δτ，物理上也对：
``τ_base`` 的抖动来自参考轨迹（``q̈_d``），那不是策略的责任。W12 的 ``MultiObjectiveReward``
用 τ_exec 的差分（因为那一周 τ_exec 就是策略的输出）；两种口径的差 = ``Δτ_base`` 的抖动 + 交叉项，
本脚本保留 ``--jerk_on {residual,exec}`` 两个开关，并把 W12 口径的分项也一起统计落盘（可对账）。

四个分项 + 回报都单独统计，不把奖励当黑箱。

三、复用关系（为什么是"内联拷贝 + import"混合）
-----------------------------------------------
* **环境**：``TrajectoryGenerator`` / ``ObservationBuilder`` / ``MultiObjectiveReward`` /
  ``PandaFK`` / ``DynamicsRandomizer`` / ``TrajectoryTrackingMDP`` / ``run_episode`` 从
  ``experiments/w12_trajectory_tracking_env.py`` **逐字内联拷贝**（文件里用
  ``[拷贝自 W12]`` 区块标出）。理由和 W11→W12 一样：W12 脚本在**模块级**就 ``AppLauncher``
  启动 Kit，不能被 import。逐字拷贝换来两件事：
  ① 指标口径（关节 RMSE / 末端 cm / 饱和率）与 W12 同源；
  ② 脚本可以拿 W12 落盘的 JSON 做**数值对账**（见下）。
* **SAC**：网络 / 双 Q / 目标网络 / 温度 α / 回放池语义全部来自 W9 手写的 ``src/sac.py``
  （``build_sac_agent`` / ``build_sac_optimizers`` / ``SACAgent.update_batch`` /
  ``squashed_gaussian_log_prob`` / ``RunningMeanStd``）——**一行都没有重写**；
  本脚本只补了两个 W9 没有的东西：
  - :class:`GpuReplayBuffer`：W9 的 ``ReplayBuffer`` 是 host numpy + 单条 ``add()``，
    64 路并行每步要 64 次 host 往返；这里换成"整块环形写入 + ``torch.randint`` 采样"的显存版，
    字段与 ``bootstrap_on_truncation`` 语义逐条对齐（``--buffer host`` 可切回 W9 原版做对照）。
  - :class:`ResidualReward` / :class:`ResidualController`：残差口径的奖励与"基座+残差"包装。

四、训练协议（Day 5）与"为什么是这几个默认值"
-----------------------------------------------
* **64 环境并行**（W11 的域随机化失配已注入、每 env 一套固定参数），每步一次 SAC 更新
  （``--gradient_steps 2`` ⇒ UTD=2:1）；回放池在显存里，容量 > 总交互步数 ⇒ 全程不覆盖。
* **热身期**（``--warmup_steps 20000`` 环境步）不做梯度更新，动作是"**零残差 + 小幅随机**"
  （``--warmup_res_scale 1.5`` N·m）：先让 Critic 看到"基座自己怎么工作"的分布，而不是 W9 那种
  整个动作空间均匀随机 —— 残差结构里 ±15 N·m 的均匀随机等于把机械臂甩出参考轨迹邻域。
* **残差动作按关节刚度归一化（``--res_scale_mode kp``，默认）**：``Δτ_j = a_norm_j · min(res_limit, deflection·Kp_j)``
  （``--res_deflection_rad 0.3``、硬上限 ``--res_limit 5``，任务书写的是 ±15）。理由：同一个 1 N·m 在
  j1（Kp=568）只偏 1.8 mrad、在 j7（Kp=2.60）偏 0.38 rad —— 差 200 倍；而 tanh 动作空间里的探索是
  "各关节等幅"的，不做归一化时探索力矩在 j7 上直接是破坏性的（实测 σ≈0.22、±5 N·m ⇒ j7 探索力矩达
  1.1 N·m ≈ 0.42 rad，比该关节的基线误差 0.17 rad 还大）。按 Kp 归一化后"满量程 = 各关节相同静态偏转"，
  与奖励里"各向同性的关节误差惩罚"对齐；``--res_scale_mode uniform --res_limit 15`` 可回到任务书字面配置
  （冒烟实测：±15/不归一化时 ``|Δτ|`` 从 0.004 → 6.2 → 8.3 N·m 单调上涨，末端误差 0.69 → 3.55 cm、
  饱和步占比 0.39 —— 行为数据只覆盖动作空间一小片，Actor 一采样就落在 Critic 外推区）。
* **奖励缩放 ``--reward_scale`` 默认 1.0**：任务书的**残差 jerk 口径**下单步奖励 ≈ −2.5
  （W12 量过：换成 τ_exec 口径才是 −22.2、jerk 项占 89%）⇒ 价值尺度 ≈ −250，本来就在合理区间；
  想复现"Critic 花几千次更新才爬到价值尺度"的坑，用 ``--jerk_on exec`` 再配 ``--reward_scale 0.05``。
* **Critic 先热身 ``--actor_delay_updates 1000``**：前 1000 次更新把 Actor 的 Adam 学习率压到 0
  （只学 Q），之后放开 —— 避免 Actor 在 Q 还没学到价值尺度时就朝"乐观的 Q"乱跑；实现只要改
  param_group 的 lr，**不需要**动 ``src/sac.py`` 一行。
* **观测归一化**（``--normalize_obs``，默认开）：105 维里 ``e_q ~ 1e-2 rad`` 而 ``q̇_d ~ 1 rad/s``、
  ``τ ~ 1e1 N·m``，量纲差 3 个数量级；用 W9 的 ``RunningMeanStd`` 在线统计（buffer 里存归一化后的
  观测，与 W9 的约定一致），评估时冻结统计量。
* **温度 α 固定成小常数**（``--target_entropy fixed`` + ``--alpha_init 0.02``）：残差 RL 要的是
  **近乎确定性的策略**（残差越粗越伤跟踪），自适应温度在这里帮倒忙 —— 实测打开自适应
  （``--target_entropy -1``）时 α 从 0.05 一路涨到 0.10，探索噪声把确定性残差从 0.002 顶到 3.1 N·m，
  末端误差同步从 1.33 cm 恶化到 2.6 cm。副产物是把 ``log_std_min`` 夹到 −5（σ ≥ 6.7e-3 ⇒
  Δτ 始终有 ~0.03 N·m 的探索噪声，不至于熵崩塌后彻底锁死），
  ``log_std_init = −1.5``（σ ≈ 0.22 ⇒ 初始 |Δτ| ≈ 0.5–1 N·m，正好是要补的量级）。
* **评估**：每 ``--eval_every`` 环境步做一次**确定性**（``tanh(μ)``）闭环评估，
  与 baseline 走**同一套指标累加代码**；``env`` 本身没有额外随机性（DR 参数固定 + 轨迹固定 +
  初值固定），所以同一策略的评估是完全可复现的。评估跑在同一个 env 上（不进 buffer），
  会把训练轨迹打断一次，脚本会自动 reset 并重开观测环。
* 训练前先跑 W12 的两条基线（``pd_dr`` / ``pd_ff_dr``）并**与 W12 的 JSON 逐项对账**，
  证明"这里的失配环境就是 W12 报告里的那一套"（实测相对偏差 **0.00e+00**：逐位相同）。

五、RL 时序契约（比 W12 的 step() 多走半步）
--------------------------------------------
W12 的 ``step()`` 在"写力矩之前"压入并返回观测，所以它返回的是**本步所作用的状态**的观测。
对读 ``tracking_error(t)`` 的控制器没问题，但 RL 的 ``next_obs`` 必须是**动作执行之后**那个状态
的观测，否则 buffer 里的 ``(O_t, a_t, r_t, O_{t+1})`` 不是标准 MDP 四元组（next_obs 滞后一步、
且与 r 不对应）。W13 因此自带一份 :class:`PolicyObsRing`（复用 W12 的 ``ObservationBuilder``，
只把 push 的时机挪到 step 之前与 step 之后各一次）。**自检**：环给出的 obs 与 ``env.step()``
返回的 obs 逐位相同，训练每步核对（实测 ``max|Δ| = 0.0e+00``）。

六、验收标准（任务书）
----------------------
=================  ================================  ==========================
指标                阈值                               参照
=================  ================================  ==========================
关节 RMSE          ``< 0.0519 rad``                   pd_ff_dr 基线 0.0519 rad
末端笛卡尔 RMSE     ``< 1.0 cm``（冲刺 ``0.5 cm``）     pd_ff_dr 基线 1.3259 cm
=================  ================================  ==========================

七、实测（64 env × 500 步 × seed 42；主结果见 ``docs/results/w13_residual_sac_ee50k.json``）
--------------------------------------------------------------------------------------------------
===========  =================  ================  ==========================================
控制器         关节 RMSE [rad]     末端 RMSE [cm]     备注
===========  =================  ================  ==========================================
``pd_dr``      0.060948           2.3016            W12 同一组数（相对偏差 0.00e+00）
``pd_ff_dr``   0.051866           1.3259            ★ 基座（前馈+PID）
残差 SAC ★     0.043342（×1.20）  0.9107（×1.46）    45 万环境步 + ``--w_ee 50000``；两条验收线都过
残差 SAC       0.047759（×1.09）  1.0642（×1.25）    30 万步 + ``--w_ee 20000``（臂 A，末端擦线）
===========  =================  ================  ==========================================

* **关节 0.043342 < 0.0519 rad ✅；末端 0.9107 < 1.0 cm ✅**（后 8 次评估均值 0.9734，最好 0.8913；
  ``first_solved @ 150144`` 环境步）；冲刺线 0.5 cm 未达。
* 奖励（任务书四项）2.4765 → 2.0883/步（−15.6%）；W12 口径总代价 22.243 → 12.630（**−43%**，
  其中 jerk 19.77→10.54、位置 0.412→0.282、速度 0.116→0.078、力矩 1.945→1.728，**每一项都降**）。
* 逐关节（相对同一组基座）：j1 −17%、j3 −19%、**j6 −43%**、j7 −14%（j7 自转不移动手部原点 ⇒ 不产生末端收益）；
  j5 −4%；j2 +21%、j4 +63% 但**绝对量极小**（0.0061→0.0073、0.0097→0.0159 rad，都远在验收线内）。
* 自检：FK 2.07e-06 m、obs 环 vs ``env.step`` ``max|Δ|=0.0e+00``、与 W12 对账 0.00e+00、饱和步占比 0.000、
  ``--eval_only`` 复现与训练末尾逐位相同。
* 每个评估点都会额外存一份**验收指标最好点**的权重（``policy_best_ee.pt``，``--no_best_ckpt`` 关掉）：
  主结果最终 0.9107 而最好 0.8913，说明"末尾权重 ≠ 最好权重"，这个开关就是给下一次训练兜底的。

**两个必须知道的设计问题**（报告 §6 有完整证据链与复现命令）：

1. 任务书四项奖励在**残差结构**里有一个激励漏洞：``‖τ_exec‖²`` 项奖励"抵消基座力矩"
   （w_mag=0 实测 ``|Δτ|`` 漂到 3.5 N·m、末端误差 1.33→2.6 cm）⇒ 本脚本默认补一项 ``w_mag‖Δτ‖²``；
2. 任务书奖励是**关节空间各向同性**的，而验收指标是**末端位置**：j7 不移动手部原点却占误差平方和 ~57%
   （w_ee=0 实测：奖励 ↓13% 而末端 1.3279 vs 1.3259 cm 持平）⇒ 主结果用 ``--w_ee`` 把末端项写进奖励。

用法::

    # ★ 主结果：64 env 并行 + 45 万环境步 + 末端项权重 5e4 + 落 JSON/CSV/图
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --num_envs 64 --steps 500 --total_steps 450000 --w_ee 50000 \\
        --out_json docs/results/w13_residual_sac_ee50k.json

    # 冒烟（几十秒跑完，验证管线与落盘）
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --total_steps 20000 --warmup_steps 2000 --eval_every 10000 --no_plot

    # 只跑基线 + 与 W12 对账（不训练）
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --baselines_only --out_json docs/results/w13_baseline_check.json

    # 消融：严格任务书四项（w_mag=0、w_ee=0、±15 N·m 均匀动作空间）/ 回放池换回 W9 host 版
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --w_mag 0 --w_ee 0 --res_limit 15 --res_scale_mode uniform
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --buffer host --total_steps 20000

    # 评估一个已保存的策略（不训练；不会覆盖训练产物：权重与 history CSV 都不写，
    # JSON 默认写成 docs/results/<run_name>_eval_only.json）
    bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \\
        --eval_only --load_policy logs/w13_residual_sac/policy_final.pt
"""
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


def _make_parser() -> argparse.ArgumentParser:
    """W13/W14 的全部命令行参数（环境部分与 W12 同名同默认，方便横向对账）。"""
    parser = argparse.ArgumentParser(
        description="W13/W14 Day 4–5：双层残差控制（标称前馈+PID 兜底 + SAC 残差策略）+ 端到端训练。",
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

    # ---- 评估 / 日志 / 落盘 ----
    parser.add_argument("--eval_every", type=int, default=25000, help="每多少个环境步做一次闭环评估（0=只在最后）。")
    parser.add_argument("--log_every", type=int, default=5000, help="每多少个环境步打印一次训练日志。")
    parser.add_argument("--baselines", type=str, default="pd_dr,pd_ff_dr",
                        help="训练前后各跑一次的开环基线（逗号分隔，取值为 W12 的 scenario 名）。")
    parser.add_argument("--baselines_only", action="store_true", help="只跑基线 + 对账，不训练。")
    parser.add_argument("--eval_only", action="store_true", help="不训练，只评估 --load_policy（或零残差）。")
    parser.add_argument("--load_policy", type=str, default="", help="加载策略权重（.pt）后再评估 / 续训。")
    parser.add_argument("--save_policy", type=str, default="", help="策略权重落盘路径（默认 logs/<run>/policy_final.pt）。")
    parser.add_argument("--no_best_ckpt", action="store_true",
                        help="不再额外保存“按验收指标（末端 RMSE）最好的一次”的权重（默认存 policy_best_ee.pt）。")
    parser.add_argument("--run_name", type=str, default="w13_residual_sac", help="日志 / 权重子目录名。")
    parser.add_argument("--w12_json", type=str, default="",
                        help="W12 结果 JSON（默认 docs/results/w12_tracking_64env.json），用于数值对账。")
    parser.add_argument("--no_cross_check", action="store_true", help="跳过与 W12 JSON 的对账。")
    parser.add_argument("--max_wall_s", type=float, default=0.0, help="墙钟上限 [s]（>0 时到点安全退出并照常落盘）。")
    parser.add_argument("--plot", action="store_true", default=True, help="训练曲线出图（默认开，缺 matplotlib 则跳过）。")
    parser.add_argument("--no_plot", action="store_false", dest="plot", help="不出图。")
    parser.add_argument("--figures_dir", type=str, default="", help="出图目录（默认 docs/figures）。")
    parser.add_argument("--out_json", type=str, default="", help="结果 JSON 落盘路径。")
    parser.add_argument("--out_csv", type=str, default="", help="训练 history CSV 落盘路径。")
    parser.add_argument("--trace_every", type=int, default=0,
                        help="baseline 评估时每多少步打印一次在线指标（0=不打印）。")

    # AppLauncher 的标准参数（--device / --headless / --livestream ...）必须挂在解析器上，
    # 否则 `AppLauncher(args_cli)` 拿不到 device/headless（W12 里是同一行调用）。
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _make_parser()
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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


def default_w12_json_path() -> str:
    """W12 主实验的 JSON 默认路径（``docs/results/w12_tracking_64env.json``）。"""
    return os.path.join(_REPO, "docs", "results", "w12_tracking_64env.json")


def cross_check_w12(path: str, results: dict, tol: float = 1e-3, args=None) -> dict:
    """把本次跑出来的 baseline 与 W12 落盘的 JSON 逐项对比（证明"用的是同一个环境"）。

    比较两个数：``joint_rmse_rad.mean_over_envs`` 与 ``cartesian_error_cm.rmse_mean_over_envs``。
    相同的种子 + 相同的 DR 采样 + 相同的轨迹 ⇒ 这两个数应当**几乎逐位相同**
    （差别只来自浮点累加顺序 / PhysX 的调度，阈值取相对 1e-3）。

    若 ``args`` 与 W12 的配置（env 数 / 步数 / dt / 种子 / 增益口径 / 符号 / 前馈来源）不一致，
    则**只打印数字、判 ``comparable=False``**：那些参数一改，指标本来就不该相同。
    """
    out = {"path": os.path.abspath(path), "available": False, "comparable": None,
           "compared": {}, "max_rel_diff": None, "match": None}
    if not os.path.isfile(path):
        print(f"[CHECK] 找不到 W12 参考 JSON（{path}），跳过对账")
        return out
    with open(path, encoding="utf-8") as fh:
        ref = json.load(fh)
    ref_cfg = ref.get("config", {})
    out["available"] = True
    out["w12_config"] = {k: ref_cfg.get(k) for k in ("num_envs", "steps", "dt", "seed", "gains_mode", "pd_sign")}
    out["w12_config"]["ff_source"] = ref.get("gravity_model", {}).get("ff_source")
    if args is not None:
        cur = {"num_envs": args.num_envs, "steps": args.steps, "dt": args.dt, "seed": args.seed,
               "gains_mode": args.gains, "pd_sign": args.pd_sign, "ff_source": args.ff}
        diff = {k: {"now": cur[k], "w12": out["w12_config"].get(k)}
                for k in cur if out["w12_config"].get(k) is not None and cur[k] != out["w12_config"][k]}
        out["config_diff"] = diff
        out["comparable"] = not diff
        if diff:
            print(f"[CHECK] 与 W12 的配置不同 {list(diff)}：只报数、不判通过（改过 --num_envs/--steps/--seed/"
                  f"--gains/--pd_sign/--ff 之后指标本来就会变）")
    worst = 0.0
    for name, res in results.items():
        old = ref.get("scenarios", {}).get(name)
        if old is None:
            continue
        pairs = {
            "joint_rmse_rad": (res["joint_rmse_rad"]["mean_over_envs"], old["joint_rmse_rad"]["mean_over_envs"]),
            "cartesian_rmse_cm": (res["cartesian_error_cm"]["rmse_mean_over_envs"],
                                  old["cartesian_error_cm"]["rmse_mean_over_envs"]),
        }
        entry = {}
        for key, (now, before) in pairs.items():
            rel = abs(now - before) / max(abs(before), 1e-12)
            worst = max(worst, rel)
            entry[key] = {"now": now, "w12": before, "rel_diff": rel}
        out["compared"][name] = entry
    out["max_rel_diff"] = worst
    out["match"] = bool(worst <= float(tol))
    for name, entry in out["compared"].items():
        print(f"[CHECK] 与 W12 对账 {name:9s} 关节 RMSE {entry['joint_rmse_rad']['now']:.6f} vs "
              f"{entry['joint_rmse_rad']['w12']:.6f}（rel {entry['joint_rmse_rad']['rel_diff']:.2e}）"
              f" | 末端 {entry['cartesian_rmse_cm']['now']:.4f} vs {entry['cartesian_rmse_cm']['w12']:.4f} cm"
              f"（rel {entry['cartesian_rmse_cm']['rel_diff']:.2e}）")
    if out["compared"]:
        if out["comparable"] is False:
            print(f"[CHECK] 最大相对偏差 {worst:.2e}（配置不同 ⇒ 仅供参考，不作通过判据）")
        else:
            verdict = ("通过：W13 的失配环境就是 W12 报告里的那一套" if out["match"]
                       else "不通过：环境/种子/增益有差异，两边的数不可直接横比")
            print(f"[CHECK] 最大相对偏差 {worst:.2e}（阈值 {tol:g}）-> {verdict}")
    return out


def zero_residual_equivalence(env, policy: ResidualPolicy, normalizer, tag: str, fk_tol: float,
                              joint_names, reference: dict | None) -> dict:
    """把残差强制为 0 跑一条 episode，与 W12 ``run_episode`` 的同一控制器结果对账。

    这是"两套指标代码其实是同一套"的硬证据：``run_policy_episode``（本脚本）与 ``run_episode``
    （W12）分别累加，但输入力矩逐位相同（Δτ ≡ 0 ⇒ τ_exec ≡ τ_base），所以
    关节 RMSE / 末端 RMSE / 每步回报都应当一致到浮点累加误差量级。
    """
    policy.zero_residual = True
    try:
        res = run_policy_episode(env, policy, None, normalizer, tag, fk_tol, joint_names)
    finally:
        policy.zero_residual = False
    out = {"tag": tag, "reference_available": reference is not None, "done": True}
    if reference is not None:
        for key, path_txt in (("joint_rmse_rad", "mean_over_envs"), ("cartesian_error_cm", "rmse_mean_over_envs")):
            now = res[key][path_txt]
            ref = reference[key][path_txt]
            out[key] = {"now": now, "w12": ref, "abs_diff": abs(now - ref), "rel_diff": abs(now - ref) / max(abs(ref), 1e-12)}
        now_r = res["reward"]["w12_reward_per_step_env_native"]
        ref_r = reference["reward"]["mean_per_step_per_env"]
        out["reward_per_step"] = {"now": now_r, "w12": ref_r,
                                  "rel_diff": abs(now_r - ref_r) / max(abs(ref_r), 1e-12),
                                  "note": "两边都用 env 原生奖励（W12 口径）比较；残差奖励的 jerk 项在 Δτ≡0 时恒为 0"}
        print(f"[CHECK] 零残差等价性（Δτ≡0 vs W12 run_episode）：关节 RMSE 差 "
              f"{out['joint_rmse_rad']['abs_diff']:.3e} rad | 末端 RMSE 差 {out['cartesian_error_cm']['abs_diff']:.3e} cm"
              f" | 回报/步 rel {out['reward_per_step']['rel_diff']:.2e} -> "
              f"{'两套指标代码一致' if out['joint_rmse_rad']['abs_diff'] < 1e-5 else '有偏差（需检查评估口径）'}")
    else:
        print("[CHECK] 零残差等价性：没有可用的 W12 baseline 参考（--baselines 里缺少对应 scenario），只做了自跑")
    return out


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
# 权重存取 / 出图 / 收尾工具
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
# 出图（沿用 experiments/w10_plot_scale.py 的 CJK 字体挑选与 140 dpi 约定）
# ============================================================================
##
_CJK_CANDIDATES = (
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "Droid Sans Fallback", "Microsoft YaHei", "SimHei",
)


def plot_history(history: list[dict], out_png: str, baselines: dict | None = None,
                 targets: dict | None = None) -> str | None:
    """把训练曲线画成 2×2 图（误差 / α 与熵差 / 残差幅值与饱和 / 损失）。

    没有 matplotlib 时**只打印一句警告**并返回 ``None``（训练结果照样落盘）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib import font_manager  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"[PLOT] 跳过出图（matplotlib 不可用：{type(exc).__name__}: {exc}）")
        return None
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            break

    rows = [r for r in history if r.get("joint_rmse_rad") is not None]
    if not rows:
        print("[PLOT] history 里没有评估点，跳过出图")
        return None
    x = [r["env_steps"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))

    ax = axes[0][0]
    ax.plot(x, [r["joint_rmse_rad"] for r in rows], "o-", color="tab:blue", label="residual SAC（闭环）")
    if baselines:
        for name, res in baselines.items():
            ax.axhline(res["joint_rmse_rad"]["mean_over_envs"], ls="--", lw=1.2,
                       label=f"{name} 基线 {res['joint_rmse_rad']['mean_over_envs']:.4f} rad")
    if targets and targets.get("joint_rmse_rad"):
        ax.axhline(targets["joint_rmse_rad"], color="tab:red", ls=":", lw=1.4,
                   label=f"验收线 {targets['joint_rmse_rad']:.4f} rad")
    ax.set_xlabel("环境步 (env-steps)")
    ax.set_ylabel("关节 RMSE [rad]")
    ax.set_title("关节跟踪误差（确定性评估，64 env × 500 步）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    ax.plot(x, [r["ee_rmse_cm"] for r in rows], "o-", color="tab:green", label="residual SAC（闭环）")
    if baselines:
        for name, res in baselines.items():
            ax.axhline(res["cartesian_error_cm"]["rmse_mean_over_envs"], ls="--", lw=1.2,
                       label=f"{name} 基线 {res['cartesian_error_cm']['rmse_mean_over_envs']:.3f} cm")
    if targets:
        ax.axhline(targets.get("ee_rmse_cm", 1.0), color="tab:red", ls=":", lw=1.4, label="验收线 1.0 cm")
        ax.axhline(targets.get("ee_rmse_cm_stretch", 0.5), color="tab:orange", ls=":", lw=1.4, label="冲刺线 0.5 cm")
    ax.set_xlabel("环境步 (env-steps)")
    ax.set_ylabel("末端笛卡尔 RMSE [cm]")
    ax.set_title("末端跟踪误差（FK 自校验通过）")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot(x, [r["alpha"] for r in rows], "o-", color="tab:purple", label="α（温度）")
    ax.set_ylabel("α", color="tab:purple")
    ax.set_xlabel("环境步 (env-steps)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, [r.get("entropy_target_gap", float("nan")) for r in rows], "s--", color="tab:brown",
             label="熵 − 目标熵")
    ax2.set_ylabel("E[−log π] − H̄", color="tab:brown")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.set_title("温度 α 与策略熵（目标熵恒为负值 ⇒ 近乎确定性）")

    ax = axes[1][1]
    ax.plot(x, [r["mean_abs_delta_tau"] for r in rows], "o-", color="tab:red", label="平均 |Δτ| [N·m]")
    ax.set_xlabel("环境步 (env-steps)")
    ax.set_ylabel("|Δτ| [N·m]", color="tab:red")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(x, [r["sat_step_frac"] for r in rows], "s--", color="tab:gray", label="饱和步占比")
    ax2.set_ylabel("饱和步占比", color="tab:gray")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax.set_title("残差幅值与力矩饱和（饱和率必须 ≈ 0，否则指标不可信）")

    fig.suptitle("W13/W14 双层残差 SAC：64 路随机化环境上的闭环跟踪", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.abspath(out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"[OUT] 训练曲线图已写入 {out_png}")
    return out_png


##
# ============================================================================
# 主流程（Day 4 架构装配 + Day 5 训练/评估/落盘）
# ============================================================================
##
def main() -> dict:
    """W13/W14 主流程：搭 W12 环境 → 注入失配 → 基线与 W12 对账 → 训练残差 SAC → 评估/落盘。"""
    device = args_cli.device or "cuda:0"
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))
    torch.cuda.manual_seed_all(int(args_cli.seed))
    torch.cuda.reset_peak_memory_stats()

    print("=" * 94)
    print(f"[ENV] torch {torch.__version__} | warp {wp.__version__} | isaacsim 6.1.0")
    print(f"[ENV] device = {device} | {torch.cuda.get_device_name(0)}")
    free_start, total_mem = torch.cuda.mem_get_info()
    print(f"[ENV] GPU memory total = {total_mem / 1024**3:.2f} GiB, free = {free_start / 1024**3:.2f} GiB")
    print(f"[W13/W14] 双层控制：τ_exec = clip(τ_base + Δτ, ±τ_lim) | τ_base = {args_cli.base_mode}"
          f"（ff={args_cli.ff}，符号={args_cli.pd_sign}）| |Δτ| ≤ {args_cli.res_limit:g} N·m")

    # ---- 场景（与 W12 同一套：纯力矩 Franka + 可选地面）----
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=device))
    scene = InteractiveScene(FrankaTrackingSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
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
    print(f"[MDP] 力矩上限 [N·m] = {[round(float(v), 2) for v in tau_limits[0].tolist()]} | "
          f"关节限位 [rad] = {[[round(float(a), 4), round(float(b), 4)] for a, b in pos_limits.tolist()]}")

    # ---- 参考轨迹 / 观测 / 奖励（与 W12 同名同默认 ⇒ 两组实验的 MDP 定义一致）----
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
    reward_fn = MultiObjectiveReward(
        {"pos": args_cli.wp, "vel": args_cli.wv, "tau": args_cli.wtau, "jerk": args_cli.wj}, device
    )
    reward_res = ResidualReward(
        {"pos": args_cli.wp, "vel": args_cli.wv, "tau": args_cli.wtau, "jerk": args_cli.wj,
         "mag": args_cli.w_mag, "ee": args_cli.w_ee}, args_cli.jerk_on
    )
    print(f"[OBS] 单帧 {obs_builder.frame_dim} 维 {obs_builder.KEYS} × H={obs_builder.H} 帧 = "
          f"{obs_builder.obs_dim} 维（策略输入；历史顺序旧→新）")
    print(f"[REW] 策略奖励 = {reward_res.describe()['formula']}，jerk 项用 {reward_res.describe()['jerk_variable']}")
    print("[REW] env 原生（W12 口径，用于对账）= 同样四项，但 jerk 项用 τ_exec 的差分")

    # ---- FK（末端笛卡尔误差的度量基础，与 W12 同一份实现）----
    fk = PandaFK(robot, device, args_cli.ee_body, args_cli.urdf or None)
    pre = fk.check(
        robot.data.joint_pos.torch,
        robot.data.body_link_pos_w.torch[:, 0, :],
        robot.data.body_link_quat_w.torch[:, 0, :],
        robot.data.body_link_pos_w.torch,
    )
    fk_ok = pre["max_err_m"] <= args_cli.fk_tolerance
    print(f"[FK] 运动链来源 = {fk.source} | 链 = {' -> '.join(fk.link_of_step)}（{fk.num_steps} 段）")
    print(f"[FK] 预检 max 残差 = {pre['max_err_m']:.3e} m（最差连杆 {pre['worst_link']}），"
          f"阈值 {args_cli.fk_tolerance:g} m -> {'通过' if fk_ok else '不通过（末端误差不可信）'}")

    # ---- 域随机化注入器（W11 的失配：未知负载 + 摩擦/阻尼 + 连杆质量）----
    rng = DynamicsRandomizer(
        robot, scene.num_envs, device,
        enabled=not args_cli.no_randomization, seed=args_cli.seed, joint_expr=args_cli.joint_expr,
        payload_body=args_cli.payload_body, payload_com=args_cli.payload_com, payload_mode=args_cli.payload_mode,
        ranges={
            "payload_mass": args_cli.payload_mass,
            "link_mass_scale": args_cli.link_mass_scale,
            "joint_damping": args_cli.joint_damping,
            "joint_friction": args_cli.joint_friction,
        },
    )
    print(f"[DR] 随机化 = {'开启' if rng.enabled else '关闭（--no_randomization）'} | seed = {args_cli.seed} | "
          f"负载 {tuple(args_cli.payload_mass)} kg | 连杆 ×{tuple(args_cli.link_mass_scale)} | "
          f"damping {tuple(args_cli.joint_damping)} | 静摩擦 {tuple(args_cli.joint_friction)}")

    # ---- MDP 环境 ----
    env = TrajectoryTrackingMDP(
        scene, sim, robot, traj, obs_builder, reward_fn, fk, joint_ids, dt, args_cli.steps,
        fk_check=not args_cli.no_fk_check, init_at_reference=args_cli.init_at_reference,
    )

    # ---- PD 增益：在参考中心 q0 上读 M(q0) 自动定（与 W12 完全相同的顺序）----
    q_center = robot.data.default_joint_pos.torch.clone()
    q_center[:, joint_ids] = traj.q0[0]
    robot.write_joint_state_to_sim_index(position=q_center, velocity=torch.zeros_like(q_center))
    scene.write_data_to_sim()
    scene.update(dt)
    if args_cli.gains == "manual":
        kp = _gain_tensor(args_cli.kp, J, device)
        kd = _gain_tensor(args_cli.kd, J, device)
        i_diag = auto_gains(robot, joint_ids, args_cli.wn, args_cli.zeta, device)[2]
        print(f"[GAIN] 手动增益（--gains manual）：kp = {args_cli.kp}，kd = {args_cli.kd}")
    else:
        mw = args_cli.gains == "mweighted"
        kp, kd, i_diag = auto_gains(robot, joint_ids, args_cli.wn, args_cli.zeta, device, mweighted=mw)
        print(f"[GAIN] {'矩阵增益 Kp=ωn²·M̂(q0)（计算力矩 PD）' if mw else '对角增益 kp_j=I_jj·ωn²'}"
              f"：ωn = {args_cli.wn:g} rad/s，ζ = {args_cli.zeta:g}")
    print(f"[GAIN] M(q0) 对角元 I_jj [kg·m²] = {[round(float(v), 5) for v in i_diag.tolist()]}")
    print(f"[GAIN] kp{'对角元' if kp.dim() == 3 else ''} = {_gain_brief(kp)} | "
          f"kd{'对角元' if kd.dim() == 3 else ''} = {_gain_brief(kd)}")
    lam_p, lam_d = stability_margins(robot, joint_ids, kp, kd, dt)
    margin_ok = float(lam_p.max()) < 4.0 and float(lam_d.max()) < 2.0
    print(f"[GAIN] 多变量离散裕度：λmax(M⁻¹Kp)·dt² = {float(lam_p.max()):.4f}（<4）；"
          f"λmax(M⁻¹Kd)·dt = {float(lam_d.max()):.4f}（<2）-> {'稳定' if margin_ok else '裕度不足（会颤振）'}")

    # ---- 标称重力模型（τ_base 的前馈项）：读**注入前**的连杆质量/质心 ----
    env.set_nominal_model(
        robot.data.body_mass.torch.to(dtype=torch.float32).clone(),
        robot.data.body_com_pose_b.torch.to(dtype=torch.float32)[:, :, :3].clone(),
    )
    ff_nominal = env.gravity_ff("nominal")
    ff_oracle = env.gravity_ff("oracle")
    ff_err = (ff_nominal - ff_oracle).abs()
    ff_rel = float(ff_err.max()) / max(float(ff_oracle.abs().max()), 1e-9)
    print(f"[FF] 标称重力模型自校验（当前位形，注入前）：max|ΔG| = {float(ff_err.max()):.4f} N·m"
          f"（相对 {ff_rel * 100:.2f}%，阈值 0.5 N·m）-> {'通过' if float(ff_err.max()) <= 0.5 else '不通过'}")

    # ---- 注入失配（训练环境必须是注入后的那一套；基线 pd_dr/pd_ff_dr 也要求注入）----
    dr_state: dict = {"on": None}
    if rng.enabled:
        rng.apply()
        dr_state["on"] = True
        env.zero_torque_warmup_step()
        dr_info = {"applied": True, "sampled": rng.sampled_summary(), "readback": rng.readback(),
                   "com_write_path": rng.com_path}
        _print_dr(dr_info)
        i_now = auto_gains(robot, joint_ids, args_cli.wn, args_cli.zeta, device)[2]
        ratio = i_now / i_diag.clamp(min=1e-9)
        dr_info["mass_matrix_diag_kgm2_dr"] = [round(float(v), 5) for v in i_now.tolist()]
        dr_info["mass_matrix_diag_ratio_dr_over_nominal"] = [round(float(v), 3) for v in ratio.tolist()]
        print(f"[DR] 注入后 M(q) 对角元 [kg·m²] = {dr_info['mass_matrix_diag_kgm2_dr']} | "
              f"相对标称 ×{dr_info['mass_matrix_diag_ratio_dr_over_nominal']}（这就是基座面对的模型失配）")
    else:
        dr_info = {"applied": False, "sampled": {}, "readback": {}, "com_write_path": "off"}
        print("[DR] 未注入失配：本轮的 baseline/训练都在标称动力学上跑（只能当消融看）")

    # ---- SAC 智能体（W9 的 src/sac.py；这里只填配置，网络/损失/更新一行没改）----
    obs_dim = obs_builder.obs_dim
    te_raw = str(args_cli.target_entropy).strip().lower()
    if te_raw in ("fixed", "none", "null", "off", "false", ""):
        target_entropy = None                     # → resolve_target_entropy(None) ⇒ 固定 α
    elif te_raw in ("auto", "-dim_a", "heuristic"):
        target_entropy = "auto"
    else:
        target_entropy = float(te_raw)
    sac_cfg = {
        "model": {
            "state_dim": obs_dim, "action_dim": J,
            "hidden_dim": args_cli.hidden, "critic_hidden_dim": args_cli.critic_hidden,
            "num_layers": args_cli.layers, "activation": args_cli.activation,
            "log_std_min": args_cli.log_std_min, "log_std_max": args_cli.log_std_max,
            "log_std_init": args_cli.log_std_init,
            "action_low": -float(args_cli.res_limit), "action_high": float(args_cli.res_limit),
            "n_critics": 2,
        },
        "training": {
            "gamma": args_cli.gamma, "tau": args_cli.tau, "init_alpha": args_cli.alpha_init,
            "target_entropy": target_entropy, "reward_scale": args_cli.reward_scale,
        },
    }
    agent = build_sac_agent(sac_cfg, device=device)
    optimizers = build_sac_optimizers(agent, lr_actor=args_cli.lr_actor, lr_critic=args_cli.lr_critic,
                                      lr_alpha=args_cli.lr_alpha)
    n_actor = int(sum(p.numel() for p in agent.actor.parameters()))
    n_critic = int(sum(p.numel() for p in agent.critic.parameters()))
    print(f"[SAC] 观测 {obs_dim} 维 → 动作 {J} 维 × ±{args_cli.res_limit:g} N·m | Actor {n_actor} 参数 "
          f"（{args_cli.layers}×{args_cli.hidden}）| Double-Q {n_critic} 参数 | γ={args_cli.gamma:g} "
          f"τ={args_cli.tau:g} | α0={args_cli.alpha_init:g} 目标熵={target_entropy} | "
          f"lr=(actor {args_cli.lr_actor:g}, critic {args_cli.lr_critic:g}, alpha {args_cli.lr_alpha:g})")
    del n_actor, n_critic

    buffer = make_replay_buffer(args_cli.buffer, args_cli.buffer_size, obs_dim, J, device,
                                args_cli.bootstrap_on_truncation, args_cli.seed)
    normalizer = RunningMeanStd(obs_dim) if args_cli.normalize_obs else None
    if args_cli.load_policy and os.path.isfile(args_cli.load_policy):
        load_policy(args_cli.load_policy, agent, normalizer, device)

    base_ctrl = BaselineController(kp, kd, env.E, env.J, device, mode=args_cli.base_mode,
                                   effort_limits=tau_limits, sign=args_cli.pd_sign, ff_source=args_cli.ff)
    res_scale = residual_action_scale(args_cli.res_scale_mode, args_cli.res_limit, args_cli.res_deflection_rad,
                                      kp, J, device)
    policy = ResidualPolicy(env, base_ctrl, reward_res, res_scale, tau_limits, device, deterministic=False)
    print(f"[RES] 残差动作空间：mode={args_cli.res_scale_mode} | 硬上限 {args_cli.res_limit:g} N·m | "
          f"逐关节 scale [N·m] = {[round(float(v), 4) for v in res_scale.tolist()]}"
          + (f"（= {args_cli.res_deflection_rad:g} rad × Kp 对角元，再夹到硬上限）"
             if args_cli.res_scale_mode == "kp" else ""))

    # ---- 训练前的开环基线（W12 的代码路径）+ 与 W12 的 JSON 数值对账 ----
    baselines: dict[str, dict] = {}
    if not args_cli.eval_only:
        for name in [s.strip() for s in str(args_cli.baselines).split(",") if s.strip()]:
            baselines[name] = run_baseline_scenario(env, rng, name, args_cli, kp, kd, tau_limits,
                                                    joint_names, dr_state)
        if rng.enabled and dr_state.get("on") is not True:      # 训练环境必须是注入后的那一套
            rng.apply()
            dr_state["on"] = True
            env.zero_torque_warmup_step()
            print("[DR] 重新注入失配（用同一次采样值），保证训练在注入后的动力学上进行")
    cross = {"skipped": True} if (args_cli.no_cross_check or args_cli.eval_only) else cross_check_w12(
        args_cli.w12_json or default_w12_json_path(), baselines, args=args_cli
    )
    ref_name = "pd_ff_dr" if args_cli.base_mode == "pd_ff" else "pd_dr"
    zero_check = {"skipped": True}
    if not args_cli.eval_only:
        zero_check = zero_residual_equivalence(env, policy, None, f"zero_res[{ref_name}]",
                                               args_cli.fk_tolerance, joint_names, baselines.get(ref_name))
    if rng.enabled and dr_state.get("on") is not True:
        rng.apply()
        dr_state["on"] = True
        env.zero_torque_warmup_step()

    # ---- 训练 + 周期性评估 ----
    history: list[dict] = []
    _T0 = [time.perf_counter()]

    # 按**验收指标**（末端笛卡尔 RMSE）挑"最好的一次"单独存一份权重：
    # 训练末尾的权重不一定就是最好的一次（主结果里最终 0.9107 cm vs 最好 0.8913 cm），
    # 没有 best-checkpoint 就只能拿到末尾那份。存权重不改任何 env/RNG 状态。
    _best = {"ee_cm": float("inf"), "joint_rad": float("nan"), "env_steps": -1, "path": None}
    _final_ckpt = args_cli.save_policy or os.path.join(_REPO, "logs", args_cli.run_name, "policy_final.pt")
    _best_ckpt = os.path.join(os.path.dirname(_final_ckpt), "policy_best_ee.pt")

    def _record_eval(env_step: int, updates: int, logs_last: dict) -> dict:
        """跑一次确定性闭环评估并记进 history（评估在同一个 env 上，**不进 buffer**）。"""
        res = evaluate(env, policy, agent, normalizer, f"residual@{env_step}", args_cli.fk_tolerance, joint_names)
        row = {
            "env_steps": int(env_step),
            "updates": int(updates),
            "wall_s": round(time.perf_counter() - _T0[0], 2),
            "joint_rmse_rad": res["joint_rmse_rad"]["mean_over_envs"],
            "joint_rmse_worst_env_rad": res["joint_rmse_rad"]["worst_env_joint"],
            "joint_rmse_velocity_rad_s": float(
                np.mean(res["joint_rmse_velocity_rad_s"]["per_joint_mean_over_envs"])),
            "ee_rmse_cm": res["cartesian_error_cm"]["rmse_mean_over_envs"],
            "ee_mae_cm": res["cartesian_error_cm"]["mae_mean_over_envs"],
            "ee_max_cm": res["cartesian_error_cm"]["max_over_envs_and_steps"],
            "return_per_step": res["reward"]["mean_per_step_per_env"],
            "return_per_step_w12_reward": res["reward"]["w12_reward_per_step_env_native"],
            "jerk_term_residual": res["reward"]["terms_per_step_mean"]["jerk"],
            "mag_term_residual": res["reward"]["terms_per_step_mean"].get("mag", float("nan")),
            "ee_term_residual": res["reward"]["terms_per_step_mean"].get("ee", float("nan")),
            "mean_abs_delta_tau": res["residual"]["mean_abs_delta_tau"],
            "max_abs_delta_tau": res["residual"]["max_abs_delta_tau"],
            "sat_step_frac": res["residual"]["sat_step_frac_any_joint"],
            "tanh_saturation_frac": res["residual"]["tanh_saturation_frac"],
            "alpha": float(agent.alpha.detach()),
            "entropy_mean": logs_last.get("entropy_mean", float("nan")),
            "entropy_target_gap": logs_last.get("entropy_target_gap", float("nan")),
            "mean_std": logs_last.get("mean_std", float("nan")),
            "critic_loss": logs_last.get("critic_loss", float("nan")),
            "actor_loss": logs_last.get("actor_loss", float("nan")),
            "q1_mean": logs_last.get("q1_mean", float("nan")),
            "buffer_fill": (buffer.size / buffer.capacity) if buffer.capacity else 0.0,
            "obs_ring_vs_env_max_abs": res["obs_ring_vs_env_max_abs"],
        }
        history.append(row)
        # 验收指标（末端 RMSE）刷新就单独存一份：训练末尾不一定是最好的那一次
        if not (args_cli.baselines_only or args_cli.eval_only or args_cli.no_best_ckpt):
            if row["ee_rmse_cm"] < _best["ee_cm"] - 1e-12:
                _best.update(ee_cm=row["ee_rmse_cm"], joint_rad=row["joint_rmse_rad"],
                             env_steps=int(env_step))
                _best["path"] = save_policy(_best_ckpt, agent, normalizer, {
                    "obs_dim": obs_dim, "action_dim": J, "res_limit": float(args_cli.res_limit),
                    "seed": int(args_cli.seed), "env_steps": int(env_step),
                    "note": "best-by-EE checkpoint：验收指标（末端 RMSE）最好的一次；训练末尾不一定就是它",
                })
                print(f"[BEST] 末端 RMSE 刷新到 {row['ee_rmse_cm']:.4f} cm（关节 {row['joint_rmse_rad']:.6f} rad）"
                      f"@ {env_step} 环境步 → {_best['path']}")
        row["best_ee_cm_so_far"] = _best["ee_cm"]
        print(f"[EVAL {env_step:>7d}] 关节 RMSE {row['joint_rmse_rad']:.6f} rad | 末端 {row['ee_rmse_cm']:.4f} cm"
              f"（最大 {row['ee_max_cm']:.4f}）| 回报/步 {row['return_per_step']:+.4f} | "
              f"|Δτ| {row['mean_abs_delta_tau']:.3f} N·m | α {row['alpha']:.4g} | "
              f"obsΔ {row['obs_ring_vs_env_max_abs']:.1e}")
        return res

    train_summary: dict = {"skipped": True}
    final_eval: dict | None = None
    baselines_after: dict[str, dict] = {}
    if args_cli.baselines_only:
        print("\n[MAIN] --baselines_only：只跑基线 + 与 W12 对账，不训练")
    elif args_cli.eval_only:
        print("\n[MAIN] --eval_only：不训练，直接评估（零残差或 --load_policy 指定的权重）")
        final_eval = _record_eval(0, 0, {})
    else:
        print("\n" + "=" * 94)
        train_summary = train_sac(env, policy, agent, optimizers, buffer, normalizer, args_cli,
                                  tau_limits, _record_eval, history)
        print("\n" + "=" * 94)
        print("[MAIN] 训练结束后的确定性评估（最终指标）")
        final_eval = _record_eval(train_summary["env_steps"], train_summary["updates"],
                                  train_summary.get("last_update_logs", {}))
        print("\n[MAIN] 训练后复测开环基线（同一次 DR 采样 ⇒ 应与训练前逐位相同：环境没被训练搞坏）")
        for name in baselines:
            baselines_after[name] = run_baseline_scenario(env, rng, name, args_cli, kp, kd, tau_limits,
                                                          joint_names, dr_state)
        if rng.enabled and dr_state.get("on") is not True:
            rng.apply()
            dr_state["on"] = True
            env.zero_torque_warmup_step()

    # ---- 结论：残差策略把 1.3 cm 压到了多少 ----
    targets = {"joint_rmse_rad": 0.0519, "ee_rmse_cm": 1.0, "ee_rmse_cm_stretch": 0.5}
    verdict: dict = {"targets": targets, "baselines": list(baselines), "fk_valid": bool(fk_ok)}
    if final_eval is not None:
        fr = final_eval
        verdict.update({
            "final_joint_rmse_rad": fr["joint_rmse_rad"]["mean_over_envs"],
            "final_ee_rmse_cm": fr["cartesian_error_cm"]["rmse_mean_over_envs"],
            "final_ee_max_cm": fr["cartesian_error_cm"]["max_over_envs_and_steps"],
            "final_mean_abs_delta_tau_nm": fr["residual"]["mean_abs_delta_tau"],
            "final_sat_step_frac": fr["residual"]["sat_step_frac_any_joint"],
            "joint_target_met": bool(fr["joint_rmse_rad"]["mean_over_envs"] < targets["joint_rmse_rad"]),
            "ee_target_met": bool(fr["cartesian_error_cm"]["rmse_mean_over_envs"] < targets["ee_rmse_cm"]),
            "ee_stretch_met": bool(fr["cartesian_error_cm"]["rmse_mean_over_envs"] < targets["ee_rmse_cm_stretch"]),
        })
        ref = baselines.get("pd_ff_dr") or baselines.get("pd_dr")
        if ref is not None:
            verdict["reference_scenario"] = "pd_ff_dr" if "pd_ff_dr" in baselines else "pd_dr"
            verdict["reference_joint_rmse_rad"] = ref["joint_rmse_rad"]["mean_over_envs"]
            verdict["reference_ee_rmse_cm"] = ref["cartesian_error_cm"]["rmse_mean_over_envs"]
            verdict["joint_rmse_improve_ratio"] = ref["joint_rmse_rad"]["mean_over_envs"] / max(
                fr["joint_rmse_rad"]["mean_over_envs"], 1e-12)
            verdict["ee_rmse_improve_ratio"] = ref["cartesian_error_cm"]["rmse_mean_over_envs"] / max(
                fr["cartesian_error_cm"]["rmse_mean_over_envs"], 1e-12)
            verdict["ee_rmse_absolute_drop_cm"] = (ref["cartesian_error_cm"]["rmse_mean_over_envs"]
                                                   - fr["cartesian_error_cm"]["rmse_mean_over_envs"])
        solved = [r["env_steps"] for r in history
                  if r["joint_rmse_rad"] < targets["joint_rmse_rad"] and r["ee_rmse_cm"] < targets["ee_rmse_cm"]]
        verdict["first_solved_env_steps"] = (int(solved[0]) if solved else None)
    verdict["w12_cross_check"] = cross
    verdict["zero_residual_equivalence"] = zero_check
    verdict["env_self_checks"] = {
        "fk_max_err_m": float(env.fk_err_max),
        "fk_tolerance_m": float(args_cli.fk_tolerance),
        "fk_valid": bool(fk_ok and env.fk_status(args_cli.fk_tolerance)["valid"]),
        "obs_ring_vs_env_max_abs": (None if train_summary.get("skipped")
                                    else train_summary.get("obs_ring_vs_env_max_abs")),
    }
    verdict["baselines_after_training_match_before"] = _baseline_repeat_check(baselines, baselines_after)

    # ---- 落盘：JSON + CSV + 图 ----
    # --eval_only 是诊断模式：不写权重、不写 history CSV，JSON 也默认换个名字，
    # 避免把训练产物（<run_name>.json）覆盖成一行评估记录。
    _tag = "_eval_only" if args_cli.eval_only else ""
    out_json = args_cli.out_json or os.path.join(_REPO, "docs", "results", f"{args_cli.run_name}{_tag}.json")
    out_csv = args_cli.out_csv or os.path.join(_REPO, "docs", "results", f"{args_cli.run_name}_history.csv")
    figures_dir = args_cli.figures_dir or os.path.join(_REPO, "docs", "figures")
    ckpt_path = args_cli.save_policy or os.path.join(_REPO, "logs", args_cli.run_name, "policy_final.pt")
    if not args_cli.baselines_only and not args_cli.eval_only:
        save_policy(ckpt_path, agent, normalizer, {
            "obs_dim": obs_dim, "action_dim": J, "res_limit": float(args_cli.res_limit),
            "seed": int(args_cli.seed), "env_steps": int(train_summary.get("env_steps", 0)),
        })
    # --eval_only 是"复现评估"用的诊断模式：不覆盖权重、也不覆盖 history CSV（否则会把训练产物冲掉）
    csv_path = write_csv(out_csv, history) if (history and not args_cli.eval_only) else None
    if _best["path"]:
        print(f"[OUT] 验收最好点权重（末端 {_best['ee_cm']:.4f} cm / 关节 {_best['joint_rad']:.6f} rad "
              f"@ {_best['env_steps']} 环境步）→ {_best['path']}")
    png_path = None
    if args_cli.plot and history:
        png_path = plot_history(history, os.path.join(figures_dir, f"{args_cli.run_name}_curves.png"),
                                baselines, targets)

    result = {
        "config": _jsonable(vars(args_cli)),
        "resolved": {
            "out_json": out_json, "out_csv": csv_path, "policy_ckpt": ckpt_path, "figure": png_path,
            "device": device, "obs_dim": obs_dim, "num_joints": J, "joint_names": joint_names,
            "tau_limits_nm": tau_limits[0].tolist(),
            "residual": {"limit_nm": float(args_cli.res_limit), "jerk_on": args_cli.jerk_on,
                         "policy": "W9 src/sac.py 的 SquashedGaussianActor（tanh 动作头缩放）"},
            "base_controller": {"mode": args_cli.base_mode, "sign": args_cli.pd_sign, "ff": args_cli.ff,
                                "gains_mode": args_cli.gains, "wn": args_cli.wn, "zeta": args_cli.zeta,
                                "kp": _gain_brief(kp), "kd": _gain_brief(kd),
                                "gain_is_matrix": bool(kp.dim() == 3)},
            "reward": reward_res.describe(),
            "env_reward_w12_convention": {"weights": {"pos": args_cli.wp, "vel": args_cli.wv,
                                                      "tau": args_cli.wtau, "jerk": args_cli.wj},
                                          "jerk_variable": "τ_exec"},
            "sac": sac_cfg,
            "buffer": {"kind": args_cli.buffer, "capacity": int(args_cli.buffer_size),
                       "bootstrap_on_truncation": bool(args_cli.bootstrap_on_truncation)},
            "normalize_obs": bool(args_cli.normalize_obs),
            "gpu_memory": {"free_gib_start": round(free_start / 1024**3, 3),
                           "total_gib": round(total_mem / 1024**3, 3)},
        },
        "trajectory": traj_info,
        "trajectory_limits": limit_info,
        "observation": obs_builder.describe(),
        "fk": {"chain": fk.describe(), "preflight": pre, "preflight_ok": bool(fk_ok),
               "status_final": env.fk_status(args_cli.fk_tolerance)},
        "gains": {"margins": {"lambda_max_MinvKp_dt2": float(lam_p.max()),
                              "lambda_max_MinvKd_dt": float(lam_d.max()),
                              "stable": bool(margin_ok)},
                  "mass_matrix_diag_kgm2_nominal": [round(float(v), 6) for v in i_diag.tolist()]},
        "gravity_model": {"ff_source": args_cli.ff, "max_abs_diff_nm": float(ff_err.max()),
                          "max_rel_diff": ff_rel,
                          "G_nominal_nm": [round(float(v), 4) for v in ff_nominal[0].tolist()],
                          "G_physx_nm": [round(float(v), 4) for v in ff_oracle[0].tolist()]},
        "dr": dr_info,
        "baselines_before_training": baselines,
        "baselines_after_training": baselines_after,
        "training": train_summary,
        "best_ee_checkpoint": {"enabled": bool(not (args_cli.baselines_only or args_cli.eval_only
                                                    or args_cli.no_best_ckpt)),
                               "ee_rmse_cm": _best["ee_cm"], "joint_rmse_rad": _best["joint_rad"],
                               "env_steps": _best["env_steps"], "policy_ckpt": _best["path"],
                               "note": "按验收指标（末端 RMSE）最好的一次；final 权重不一定就是它"},
        "history": history,
        "final_eval": final_eval,
        "verdict": verdict,
        "gpu": {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "free_gib_after": round(torch.cuda.mem_get_info()[0] / 1024**3, 3)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[OUT] 结果已写入 {out_json}")
    print(f"[OUT] 显存：torch peak allocated {result['gpu']['peak_allocated_gib']:.3f} GiB | "
          f"跑完剩余 {result['gpu']['free_gib_after']:.2f} GiB"
          f"（起始 {result['resolved']['gpu_memory']['free_gib_start']:.2f}）")

    print("\n" + "=" * 94)
    print("[SUMMARY] 双层残差控制：开环基线 vs 残差 SAC（同一次 DR 采样、同一套指标代码）")
    for name, res in baselines.items():
        print(f"  {name:9s} joint RMSE = {res['joint_rmse_rad']['mean_over_envs']:.6f} rad"
              f"  |  末端 RMSE = {res['cartesian_error_cm']['rmse_mean_over_envs']:.4f} cm"
              f"  |  饱和步占比 = {res['controller']['sat_step_frac_any_joint']:.3f}")
    if final_eval is not None:
        print(f"  residual  joint RMSE = {verdict['final_joint_rmse_rad']:.6f} rad"
              f"  |  末端 RMSE = {verdict['final_ee_rmse_cm']:.4f} cm"
              f"  |  |Δτ| = {verdict['final_mean_abs_delta_tau_nm']:.3f} N·m"
              f"  |  饱和步占比 = {verdict['final_sat_step_frac']:.3f}")
        if "ee_rmse_improve_ratio" in verdict:
            print(f"[VERDICT] 相对 {verdict['reference_scenario']} 基线：末端 RMSE "
                  f"{verdict['reference_ee_rmse_cm']:.4f} → {verdict['final_ee_rmse_cm']:.4f} cm"
                  f"（降 {verdict['ee_rmse_absolute_drop_cm']:.4f} cm，×{verdict['ee_rmse_improve_ratio']:.2f}）"
                  f"；关节 RMSE ×{verdict['joint_rmse_improve_ratio']:.2f}")
        print(f"[VERDICT] 验收：关节 < {targets['joint_rmse_rad']:.4f} rad -> "
              f"{'达标' if verdict['joint_target_met'] else '未达标'}；"
              f"末端 < {targets['ee_rmse_cm']:.1f} cm -> {'达标' if verdict['ee_target_met'] else '未达标'}；"
              f"冲刺 < {targets['ee_rmse_cm_stretch']:.1f} cm -> "
              f"{'达标' if verdict['ee_stretch_met'] else '未达标'}"
              f"（首次同时达标 @ {verdict['first_solved_env_steps']} 环境步）")
    print(f"[VERDICT] 环境自检：FK max 残差 {verdict['env_self_checks']['fk_max_err_m']:.3e} m"
          f"（{'通过' if verdict['env_self_checks']['fk_valid'] else '不通过'}）| "
          f"obs 环 vs env.step 最大偏差 {verdict['env_self_checks']['obs_ring_vs_env_max_abs']} | "
          f"与 W12 对账 {cross.get('match')} | 训练后基线复测 "
          f"{verdict['baselines_after_training_match_before'].get('match')}")
    print("=" * 94)
    return result


def _baseline_repeat_check(before: dict, after: dict) -> dict:
    """训练前后各跑一次基线（同一次 DR 采样）：两次的数字应当一致到浮点噪声。"""
    out: dict = {"available": bool(before and after), "max_rel_diff": None, "match": None}
    worst = 0.0
    for name, res in before.items():
        if name not in after:
            continue
        for key, sub in (("joint_rmse_rad", "mean_over_envs"),
                         ("cartesian_error_cm", "rmse_mean_over_envs")):
            now, bfr = after[name][key][sub], res[key][sub]
            worst = max(worst, abs(now - bfr) / max(abs(bfr), 1e-12))
    if out["available"]:
        out["max_rel_diff"] = worst
        out["match"] = bool(worst <= 1e-3)
    return out


def _jsonable(obj):
    """把 argparse 的 Namespace 转成可 JSON 化的纯 Python 结构（不认识的对象转字符串，不静默丢字段）。"""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":
    main()
    simulation_app.close()
