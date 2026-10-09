#!/usr/bin/env python
"""W12 Day 3：机械臂高精度连续轨迹跟踪的 MDP 环境（轨迹发生器 + 时序观测缓冲 + 多目标奖励）+ 开环基线探针。

W10 给了“纯力矩、64 路并行”的物理基座；W11 证明了“同一条力矩、不同 env 的动力学必然分化”。
W12 把两者拼成一个**可训练的跟踪 MDP**，并先用一个**开环 PD 基线**把“还没学习时误差有多大”钉死：

.. code-block::

    s_t = [ e_q, ė_q, q_d, q̇_d, τ_{t−1} ]              e_q = q − q_d,  ė_q = q̇ − q̇_d
    r_t = − w_p‖e_q‖² − w_v‖ė_q‖² − w_τ‖τ_t‖² − w_j‖τ_t − τ_{t−1}‖²
    q_{d,i}(t) = q_{0,i} + A_i · sin(2π f_i t + φ_i)     （解析给出 q̇_d、q̈_d）

三块内容对应任务书的三项要求：

1. **Trajectory Generator**：逐关节不同的 ``(A_i, f_i, φ_i)`` → 7 轴复合正弦（多轴连续插补运动，
   每个关节都在动但周期互不相同，不会退化成整臂同相位摆动）；解析导数供奖励里的速度误差项使用；
   幅值按**关节限位自动 clamp**，保证参考轨迹 q_d / q̇_d 全程合法（不会撞限位污染指标）。
2. **Observation Design + History Buffer**：单帧 D = 5J = 35 维（J=7 个臂关节），保留最近 **H = 3** 帧
   拼接成 105 维（隐式推断未建模摩擦 / 外载的基石）。帧的推入顺序、warm-up 策略、维度切分表都在
   :class:`ObservationBuilder` 里显式定义并落盘。
3. **Reward Engineering**：跟踪精度 / 速度误差 / 能耗 / 平滑性四项加权和
   （默认 ``w_p=10, w_v=0.1, w_τ=0.001, w_j=0.01``）。四个分项**单独统计并落盘**，
   避免“奖励是一个黑箱数”这种常见坑。

开环基线追踪探针（``--scenarios``）
-----------------------------------
在 W11 的域随机化（未知负载 + 非线性摩擦/阻尼 + 连杆质量失配）**已注入**的 64 个环境里，
用 ``τ_base = K_p e_q + K_d ė_q`` 跑 ``--steps`` 步，输出

* **关节跟踪 RMSE [rad]**：总体 + 逐关节 + 跨 env 分布（mean/max）；
* **末端笛卡尔跟踪误差 [cm]**：mean / RMSE / max。

默认再跑四条对照，用来回答“误差里有多少来自失配、多少来自控制器本身”：

=================  =========================================================================
``pd_nodr``        同控制器 + 标称动力学（不注入失配）
``pd_ff_nodr``     PD + **标称模型**重力前馈 + 标称动力学（“前馈+PID”的理想上限）
``pd_dr``          ★ 主结果：同控制器 + 注入失配（任务书要求的那个）
``pd_ff_dr``       PD + 标称模型重力前馈 + 注入失配（``--ff oracle`` 可换成"知道真实负载"的上限参照）
``zero_dr``        零力矩（完全不控制）+ 注入失配 —— 误差上界
=================  =========================================================================

.. note::
   ``pd_ff_*`` 的重力前馈默认取 :meth:`PandaFK.gravity_torque`（用**注入前**读到的标称质量/质心自己算），
   所以注入失配后这个前馈本身就带着模型误差 —— 这正是要量化的东西。
   ``--ff oracle`` 则直接读 PhysX 的 ``gravity_compensation_forces``（PhysX 按注入后的真实模型算），
   相当于"知道负载有多重"，只能当上限参照，不是控制器的能力。

实测（64 env × 500 步 × seed 42，`docs/results/w12_tracking_64env.json`）
------------------------------------------------------------------------
=================  =============  ==============  ==========
scenario            关节 RMSE [rad]  末端 RMSE [cm]   饱和率
=================  =============  ==============  ==========
``pd_nodr``         0.033389       1.4367          0.0000
``pd_ff_nodr``      0.015779       0.6269          0.0000
``pd_dr`` ★         0.060948       2.3016          0.0000
``pd_ff_dr``        0.051866       1.3259          0.0000
``zero_dr``         0.972252       73.1780         0.0000
=================  =============  ==============  ==========

结论：标称动力学下"前馈+PID"能把 7 轴连续轨迹跟到 **0.63 cm**；注入 W11 的未知负载+摩擦后掉到
**1.33 cm**（纯 PD 2.30 cm），完全不控制是 **73 cm** —— 中间这 1–2 cm 就是 Day 4 残差 RL 要吃掉的部分。

PD 增益不是手填的（``--gains mweighted``，默认）
------------------------------------------------
Franka 各关节的等效惯量跨两个数量级（实测 ``M(q0)`` 对角元
``[0.631, 2.280, 1.281, 0.866, 0.0269, 0.0348, 0.00289] kg·m²``），同一组标量 kp/kd 用在两者上必然出事。
脚本提供两种自动口径，并把**多变量离散稳定裕度**算出来落盘：

* ``--gains mweighted``（默认）：``Kp = ωn²·M̂(q0)``、``Kd = 2ζωn·M̂(q0)``（矩阵，计算力矩 PD 的 PD 部分）→
  实测 ``λmax(M⁻¹Kp)·dt² = 0.090``、``λmax(M⁻¹Kd)·dt = 0.600``（限值 4 / 2）；
* ``--gains auto``：``kp_j = I_jj·ωn²``、``kd_j = 2ζωn·I_jj``（对角）→ 实测 ωn=6 时
  ``λmax(M⁻¹Kd)·dt = 1.876``（勉强稳，但太软）；ωn=10 时 ``3.126 > 2`` → **闭环颤振**
  （末端 RMSE 差 15 倍）。

关键点：稳定性由 ``M⁻¹Kp`` / ``M⁻¹Kd`` 的**特征值**决定，不是由单关节的 ``ωn·dt`` 决定；
单关节条件只是必要条件。日志会把 ``ωn``、``Kp/Kd`` 对角元、两个裕度全部打印出来。

任务书公式里的一个符号坑（已修正，可用 ``--pd_sign literal`` 复现）
--------------------------------------------------------------------
观测定义 ``e_q = q − q_d``，而任务书写 ``τ_base = K_p e_q + K_d ė_q``；把定义代进去，
这个字面写法是 ``τ = K_p(q − q_d) + …``，即"往下掉就给往下推的力矩"——**正反馈**。
本脚本默认按标准 PD 口径取 ``τ = −(K_p e_q + K_d ė_q)``（等价于 ``K_p(q_d − q) + K_d(q̇_d − q̇)``），
并把两种口径都留成开关：``--pd_sign {stabilizing,literal}``。实测（64 env × 300 步）：
``literal`` → 关节 RMSE **1.802 rad**、末端 **57.47 cm**、**99.0%** 的步撞力矩上限；
``stabilizing`` → **0.061 rad / 2.30 cm / 0%**。

末端笛卡尔误差怎么算（本脚本的关键实现）
----------------------------------------
Isaac Lab 3.0 的 PhysX 后端**没有**“给定关节角求连杆位姿”的接口（``get_link_transforms`` 只能拿当前状态），
所以本脚本自带一个 **URDF → torch 批量 FK**：

* 从 Isaac Lab 自带的 ``isaaclab/controllers/config/data/lula_franka_gen.urdf`` 解析
  ``panda_link0 → … → panda_hand`` 的固定 origin ``(xyz, rpy)`` 与关节轴（文件找不到时退回脚本内嵌的
  同一组常量，两者数值相同，来源会打印/落盘）；
* 用 4×4 齐次矩阵在**显存里对全部 env 批量**做 FK：``T_child = T_parent · T_origin · R(axis, q)``；
* **自校验**：每一帧都拿 ``FK(q_measured)`` 与 PhysX 自己算出来的 ``body_link_pos_w`` 对比，
  残差（``--fk_tolerance``，默认 1e-4 m）与 FK 来源一并落盘 —— 不靠“我觉得对”，靠仿真真值对账。

参考末端位置 = ``FK(q_d)``（**同一个基座位姿**），笛卡尔误差 = ``‖FK(q) − FK(q_d)‖``，
所以它与 env 网格间距、世界系原点都无关。FK 校验失败的帧会在日志里大声报警，并在 JSON 里标 ``fk_valid=false``。

与 W11 的关系
-------------
* 复用 W11 的**纯力矩 Franka CFG**（actuator ``k=d=0``，否则注入的阻尼/摩擦会被 PD 项掩盖）；
* 复用 W11 的**域随机化注入器**（内联拷贝：W11 脚本模块级启动 Kit，不能被 import）；
* 唯一新增的“物理设定”是把重力默认打开（``--no_gravity`` 可关）：只有开着重力，“未知负载”才会产生
  静态重力矩，基线的跟踪误差里才看得见负载。理由与 W11 相反（W11 要的是纯粹的分化，怕限位饱和）。

用法::

    # 主实验（64 env / 500 步 / 5 条 scenario / 落 JSON）
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --num_envs 64 --steps 500 --out_json docs/results/w12_tracking_64env.json

    # 只看主结果（不做标称/零力矩对照），并关掉重力
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --scenarios pd_dr --no_gravity

    # 增益口径对照：对角增益在 100 Hz 下 "稳但软 / 直接颤振" 的边界
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --gains auto --wn 6  --scenarios pd_nodr,pd_ff_nodr,pd_dr,pd_ff_dr
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --gains auto --wn 10 --scenarios pd_ff_nodr,pd_ff_dr      # 会打印 [WARN] 裕度不足

    # 复现任务书字面符号（正反馈）
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --pd_sign literal --scenarios pd_dr --steps 300

    # 换一组更“凶”的参考轨迹（幅值/频率范围）与更宽的频带
    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py \\
        --traj_amp 0.3 0.5 --traj_freq 0.4 1.0 --harmonics 3 --wn 50
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import xml.etree.ElementTree as ET

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:  # pragma: no cover - 只在环境没配好时触发
    raise SystemExit(
        "找不到 isaaclab：请用仓库里的环境固化脚本启动（它会激活 py312 并挂好 PYTHONPATH）：\n"
        "    bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py --num_envs 64\n"
        "或者自己拼环境（等价做法）：\n"
        "    export PYTHONPATH=$(ls -d /data/zzh/IsaacLab/source/*/ | tr '\\n' ':')\n"
        "    HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python -u "
        f"{os.path.abspath(__file__)} --num_envs 64\n"
    ) from exc
##
# 命令行参数（AppLauncher 必须在所有重量级 import 之前启动 Kit）
##
parser = argparse.ArgumentParser(
    description="W12 Day 3：轨迹跟踪 MDP 环境（轨迹发生器 / 观测历史缓冲 / 多目标奖励）+ 开环 PD 基线探针。"
)

# ---- 规模与积分 ----
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数。")
parser.add_argument("--steps", type=int, default=500, help="每条 scenario 跑多少步（任务书要求 500）。")
parser.add_argument("--dt", type=float, default=1.0 / 100.0, help="物理步长 [s]，默认 100 Hz。")
parser.add_argument("--env_spacing", type=float, default=2.5, help="环境间距 [m]。")
parser.add_argument(
    "--no_gravity", action="store_true",
    help="关闭重力（默认开：只有开重力，‘未知负载’才会产生静态重力矩，基线的跟踪误差里才看得见负载）。",
)
parser.add_argument(
    "--no_ground", action="store_true",
    help="不生成地面（默认生成）。用于排除“基座/连杆与地面碰撞”对力矩与指标的影响。",
)
parser.add_argument("--seed", type=int, default=42, help="随机种子（DR 采样 / 轨迹采样的默认种子）。")

# ---- 参考轨迹（Trajectory Generator）----
parser.add_argument(
    "--traj_amp", type=float, nargs=2, default=(0.15, 0.35), metavar=("MIN", "MAX"),
    help="逐关节参考幅值 A_i ~ U(MIN, MAX) [rad]（再按关节限位 clamp）。",
)
parser.add_argument(
    "--traj_freq", type=float, nargs=2, default=(0.2, 0.6), metavar=("MIN", "MAX"),
    help="逐关节参考频率 f_i ~ U(MIN, MAX) [Hz]（互不相同 → 多轴复合）。",
)
parser.add_argument(
    "--traj_center", type=float, nargs=7, default=(0.0, -0.4, 0.0, -1.8, 0.0, 1.8, 0.0),
    metavar="Q0",
    help="参考轨迹中心 q0（7 个臂关节）[rad]。默认这一组是实测定出来的：关节限位余量 ≥1.27 rad（参考轨迹撞不到限位），"
         "同时手腕重力矩只剩 ~6 N·m（离 12 N·m 力矩上限有余量）——限位/力矩饱和都不会污染跟踪指标。",
)
parser.add_argument("--traj_seed", type=int, default=-1, help="轨迹采样种子，-1 表示跟随 --seed。")
parser.add_argument(
    "--harmonics", type=int, default=1,
    help="每个关节叠加的正弦项数 K：K=1 严格是任务书给的 q_d=q0+A·sin(...)；K>1 叠加 2f,3f… 且幅值按 1/h 衰减。",
)
parser.add_argument(
    "--traj_margin", type=float, default=0.8,
    help="幅值相对‘关节限位余量’的安全比例（0.8 表示 q0±ΣA 至少离限位还有 20%% 余量）。",
)
parser.add_argument("--init_at_reference", action="store_true", default=True,
                    help="每条 scenario 开始时把关节状态初始化到 q_d(0)/q̇_d(0)（默认开，去掉起步瞬态）。")
parser.add_argument("--no_init_at_reference", action="store_false", dest="init_at_reference",
                    help="改为从默认位形起步。")

# ---- 观测（Observation + History Buffer）----
parser.add_argument("--history", type=int, default=3, help="历史观测帧数 H（默认 3，任务书要求）。")
parser.add_argument(
    "--history_warmup", choices=["repeat", "zeros"], default="repeat",
    help="warm-up（历史还没填满时）的填充策略：repeat=重复首帧（默认），zeros=补零。",
)

# ---- 奖励（Reward Engineering）----
parser.add_argument("--wp", type=float, default=10.0, help="位置误差权重 w_p（任务书推荐 10.0）。")
parser.add_argument("--wv", type=float, default=0.1, help="速度误差权重 w_v（任务书推荐 0.1）。")
parser.add_argument("--wtau", type=float, default=0.001, help="力矩幅值权重 w_τ（任务书推荐 0.001）。")
parser.add_argument("--wj", type=float, default=0.01, help="力矩变化率（抖动）权重 w_j（任务书推荐 0.01）。")

# ---- 基线控制器 ----
parser.add_argument(
    "--gains", choices=["mweighted", "auto", "manual"], default="mweighted",
    help="PD 增益来源：mweighted=Kp=ωn²·M̂(q0) 矩阵增益（计算力矩 PD，默认）；auto=对角增益 kp=I_jj·ωn²；manual=用 --kp/--kd。",
)
parser.add_argument("--wn", type=float, default=30.0, help="目标闭环带宽 ωn [rad/s]（离散稳定要 λmax(M⁻¹Kd)·dt < 2）。")
parser.add_argument("--zeta", type=float, default=1.0, help="auto 增益的阻尼比 ζ（1.0 = 临界阻尼）。")
parser.add_argument("--kp", type=float, nargs="+", default=(100.0,),
                    help="[仅 --gains manual] PD 比例增益（1 个值=全部关节，或 7 个值=逐关节）。")
parser.add_argument("--kd", type=float, nargs="+", default=(10.0,),
                    help="[仅 --gains manual] PD 微分增益（1 个值=全部关节，或 7 个值=逐关节）。")
parser.add_argument(
    "--pd_sign", choices=["stabilizing", "literal"], default="stabilizing",
    help="PD 符号口径：stabilizing=τ=−(Kp·e_q+Kd·ė_q)（标准、收敛，默认）；literal=τ=+Kp·e_q+Kd·ė_q（任务书字面写法，正反馈，仅供对照）。",
)
parser.add_argument(
    "--ff", choices=["nominal", "oracle"], default="nominal",
    help="pd_ff_* 场景的重力前馈来源：nominal=用注入前的标称模型自己算（诚实，默认）；oracle=读 PhysX 的 τ_g（知道负载，上限参照）。",
)
parser.add_argument(
    "--scenarios", type=str, default="pd_nodr,pd_ff_nodr,pd_dr,pd_ff_dr,zero_dr",
    help="要跑的 scenario（逗号分隔）：pd_nodr / pd_ff_nodr / pd_dr / pd_ff_dr / zero_dr。",
)
# ---- W11 域随机化范围（每 env 独立采样）----
parser.add_argument("--no_randomization", action="store_true",
                    help="完全不注入失配（所有 scenario 都在标称动力学上跑）。")
parser.add_argument("--payload_mass", type=float, nargs=2, default=(0.0, 3.0), metavar=("MIN", "MAX"),
                    help="末端未知负载质量范围 [kg]。")
parser.add_argument("--payload_body", type=str, default="panda_hand", help="挂载负载的连杆名（默认法兰盘）。")
parser.add_argument("--payload_com", type=float, nargs=3, default=(0.0, 0.0, 0.05), metavar=("X", "Y", "Z"),
                    help="负载质心在载荷连杆系下的位置 [m]（默认法兰外侧 5 cm）。")
parser.add_argument("--payload_mode", choices=["parallel_axis", "mass"], default="parallel_axis",
                    help="负载惯量处理：parallel_axis=平行轴定理合成质心/惯量；mass=只加质量、惯量按比例缩放。")
parser.add_argument("--link_mass_scale", type=float, nargs=2, default=(0.8, 1.2), metavar=("LO", "HI"),
                    help="连杆质量比例扰动范围（m ← m·U(LO,HI)，惯量同比缩放）。")
parser.add_argument("--joint_damping", type=float, nargs=2, default=(0.5, 2.0), metavar=("LO", "HI"),
                    help="关节阻尼范围 [N·m·s/rad]（物理层 DOF damping）。")
parser.add_argument("--joint_friction", type=float, nargs=2, default=(0.1, 1.0), metavar=("LO", "HI"),
                    help="关节静摩擦力矩范围 [N·m]（动摩擦同时写 0.5×静摩擦）。")
parser.add_argument("--joint_expr", type=str, default=r"panda_joint[1-7]",
                    help="被控/被注入的关节正则（默认 7 个臂关节，手指不参与）。")

# ---- FK / 末端误差 ----
parser.add_argument("--ee_body", type=str, default="panda_hand", help="末端连杆名（笛卡尔误差的参考点）。")
parser.add_argument("--urdf", type=str, default="", help="Franka URDF 路径（默认自动在 isaaclab 包里找）。")
parser.add_argument("--no_fk_check", action="store_true", help="跳过“FK vs PhysX 真值”的自校验（默认做）。")
parser.add_argument("--fk_tolerance", type=float, default=1e-4, help="FK 自校验通过阈值 [m]。")

# ---- 记录/落盘 ----
parser.add_argument("--trace_every", type=int, default=50, help="每多少步打印一次在线指标（RMSE / 奖励分项）。")
parser.add_argument("--eval_skip", type=int, default=0, help="指标统计跳过的起始步数（默认 0 = 全 episode 计入）。")
parser.add_argument("--out_json", type=str, default="", help="把结果另存为 JSON 的路径。")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##
# Kit 起来之后再导入仿真相关的库
##
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
##
# 资产配置：纯力矩 Franka（沿用 W10/W11），并把“关节阻尼 / 摩擦”腾出来给 DR 注入
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
# 名字 -> 索引（用官方 API，失败则退回正则，避免版本差异把脚本卡死）
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


def _resolve_body_id(robot: Articulation, name: str) -> int:
    try:
        ids, _ = robot.find_bodies(name)
        return int(ids[0])
    except Exception:  # noqa: BLE001
        return robot.body_names.index(name)


def _stat(x: torch.Tensor) -> dict:
    """张量 -> ``{min, mean, max, std}``（搬回 host 标量，仅供打印/落盘）。"""
    x = x.reshape(-1).to(dtype=torch.float32)
    return {
        "min": float(x.min().item()),
        "mean": float(x.mean().item()),
        "max": float(x.max().item()),
        "std": float(x.std().item()),
    }


def _fmt_stat(d: dict | None, unit: str = "") -> str:
    """把 :func:`_stat` 的 dict 打成一行字。"""
    if d is None:
        return "-"
    return f"min {d['min']:.4g} / mean {d['mean']:.4g} / max {d['max']:.4g} / std {d['std']:.4g}{unit}"


def _gain_tensor(vals, num_joints: int, device: str) -> torch.Tensor:
    """把 ``--kp/--kd``（1 个值或 J 个值）变成 ``(1, J)`` 的增益张量。"""
    v = torch.tensor(list(vals), dtype=torch.float32, device=device)
    if v.numel() == 1:
        v = v.expand(num_joints)
    if v.numel() != num_joints:
        raise SystemExit(f"增益必须是 1 个值或 {num_joints} 个值，收到 {v.numel()} 个")
    return v.reshape(1, num_joints)


def _gain_brief(g: torch.Tensor) -> list[float]:
    """增益的简报（矩阵增益取对角元的跨 env 均值），供日志/落盘用。"""
    if g.dim() == 3:
        J = g.shape[1]
        return [round(float(v), 3) for v in g[:, range(J), range(J)].mean(0).tolist()]
    return [round(float(v), 3) for v in g.reshape(-1).tolist()]


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
# 模块 1：时变参考轨迹发生器（Trajectory Generator）
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
# 模块 2：URDF → torch 批量 FK（末端笛卡尔误差的度量基础）
##
#: Franka Panda 从 ``panda_link0`` 到 ``panda_hand`` 的固定运动链，数值与 Isaac Lab 自带的
#: ``isaaclab/controllers/config/data/lula_franka_gen.urdf`` **完全一致**，用作 URDF 读不到时的兜底。
#: 每项：(关节名, 父连杆, 子连杆, origin xyz [m], origin rpy [rad], 关节轴, 类型)
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

_PANDA_CHAIN_KEYS = ("name", "parent", "child", "xyz", "rpy", "axis", "type")


def _vec3(text, default) -> list[float]:
    """URDF 属性的 ``"x y z"`` → ``[x, y, z]``（缺失时用 default）。"""
    if text is None:
        return [float(v) for v in default]
    return [float(v) for v in text.split()]


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
# 模块 3：域随机化注入器（与 W11 同源：把“失配”写进 PhysX 的模型属性）
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
# 模块 4：观测（Observation Design）+ 时序历史缓冲（History Buffer）
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
# 模块 5：多目标奖励（Reward Engineering）
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
# 模块 6：开环基线控制器（PD / PD+重力前馈 / 零力矩）
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
# 模块 7：轨迹跟踪 MDP 环境（把轨迹 / 观测 / 奖励 / 物理粘成一个 reset/step 接口）
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
# 模块 8：rollout + 指标统计（关节 RMSE / 末端 cm / 奖励分项 / 吞吐）
##
def _reduce(t: torch.Tensor) -> dict:
    """跨 env 归约（``(E,)`` 或 ``(E,J)`` → 逐关节 mean/max 或整体统计）。"""
    if t.dim() == 1:
        return {"mean_over_envs": float(t.mean()), "max_over_envs": float(t.max()), "min_over_envs": float(t.min())}
    return {
        "per_joint_mean_over_envs": [float(v) for v in t.mean(dim=0)],
        "per_joint_max_over_envs": [float(v) for v in t.max(dim=0).values],
    }


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
# 模块 9：scenario 编排（标称 / 注入 / 前馈 / 零力矩 四组对照）
##
#: scenario 名 → (控制器模式, 是否注入失配, 人话描述)
_SCENARIOS: dict[str, tuple[str, bool, str]] = {
    "pd_nodr": ("pd", False, "PD 基线 + 标称动力学（不注入失配）"),
    "pd_ff_nodr": ("pd_ff", False, "PD + 标称模型重力前馈 + 标称动力学（“前馈+PID”的理想上限）"),
    "pd_dr": ("pd", True, "PD 基线 + 注入失配 —— ★ 任务书要求的基线追踪测试"),
    "pd_ff_dr": ("pd_ff", True, "PD + 重力前馈 + 注入失配（前馈来源见 --ff，默认标称模型）"),
    "zero_dr": ("zero", True, "零力矩（完全不控制）+ 注入失配 —— 误差上界"),
}


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


def _run_scenarios(env: TrajectoryTrackingMDP, rng: DynamicsRandomizer, args_cli, joint_names, effort_limits,
                   kp: torch.Tensor, kd: torch.Tensor, i_diag_nominal: torch.Tensor, device: str):
    """依次跑 ``--scenarios`` 里的每条 scenario：按需切换动力学 + 复位状态 + 指定控制器。

    “切换动力学”是在**同一个进程**里做的：``rng.apply()`` 注入失配、``rng.restore_nominal()`` 撤回。
    这样一组对照的物理差异只来自注入本身，排除了“重启仿真带来的任何差异”（W11 的对照组就是这么做的）。
    注入后还会再量一次 ``M(q)`` 对角元，和控制器的标称模型对比 —— 这就是控制器面对的**模型失配**。
    """
    names = [s.strip() for s in str(args_cli.scenarios).split(",") if s.strip()]
    unknown = [n for n in names if n not in _SCENARIOS]
    if unknown:
        raise SystemExit(f"未知 scenario {unknown}；可选：{list(_SCENARIOS)}")
    if args_cli.no_randomization:
        print("[WARN] --no_randomization：所有 *_dr scenario 都会在标称动力学上跑，结论只能当标称对照看")

    results: dict[str, dict] = {}
    dr_state: bool | None = None
    dr_info: dict = {"applied": bool(rng.enabled), "sampled": {}, "readback": {}, "com_write_path": "off"}
    for name in names:
        mode, want_dr, desc = _SCENARIOS[name]
        if rng.enabled and want_dr != dr_state:
            t_sw = time.perf_counter()
            if want_dr:
                rng.apply()
            else:
                rng.restore_nominal()
            dr_state = want_dr
            print(f"[DR] 切换动力学 -> {'注入失配' if want_dr else '标称动力学'}（{time.perf_counter() - t_sw:.3f} s）")
            if want_dr and not dr_info["sampled"]:
                dr_info["sampled"] = rng.sampled_summary()
                dr_info["readback"] = rng.readback()
                dr_info["com_write_path"] = rng.com_path
                _print_dr(dr_info)
                # 注入后空推一步物理，再量 M(q)（PhysX 的 M 在 simulate() 里算，否则可能读到缓存）
                env.zero_torque_warmup_step()
                i_now = auto_gains(env.robot, env.joint_ids, args_cli.wn, args_cli.zeta, device)[2]
                ratio = i_now / i_diag_nominal.clamp(min=1e-9)
                dr_info["mass_matrix_diag_kgm2_dr"] = [round(float(v), 5) for v in i_now.tolist()]
                dr_info["mass_matrix_diag_ratio_dr_over_nominal"] = [round(float(v), 3) for v in ratio.tolist()]
                print(f"[DR] 注入后 M(q) 对角元 [kg·m²] = {dr_info['mass_matrix_diag_kgm2_dr']}")
                print(f"[DR] 相对标称的倍率          = {dr_info['mass_matrix_diag_ratio_dr_over_nominal']}"
                      f"（这就是 PD 控制器面对的模型失配）")
        controller = BaselineController(kp, kd, env.E, env.J, device, mode=mode, effort_limits=effort_limits,
                                        sign=args_cli.pd_sign, ff_source=args_cli.ff)
        print(f"\n--- [{name}] {desc}")
        print(f"--- 控制器 = {mode}（增益 = {_gain_brief(kp)}，逐关节 kd = {_gain_brief(kd)}），"
              f"动力学 = {'注入失配' if want_dr else '标称'}，跑 {env.steps} 步 × {env.E} env")
        results[name] = run_episode(
            env, controller, name, args_cli.fk_tolerance, args_cli.eval_skip, args_cli.trace_every, joint_names
        )
    return results, dr_info
##
# 模块 9：主流程
##
def main() -> dict:
    """W12 主流程：搭场景 → 建轨迹/观测/奖励 →（可选）注入失配 → 跑 baseline scenario → 汇总落盘。"""
    device = args_cli.device or "cuda:0"
    torch.cuda.reset_peak_memory_stats()

    print("=" * 78)
    print(f"[ENV] torch {torch.__version__} | warp {wp.__version__} | isaacsim 6.1.0")
    print(f"[ENV] device = {device} | {torch.cuda.get_device_name(0)}")
    free_start, total = torch.cuda.mem_get_info()
    print(f"[ENV] GPU memory total = {total / 1024**3:.2f} GiB, free = {free_start / 1024**3:.2f} GiB")

    # ---- 仿真上下文 + 场景 ----
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=device))
    scene = InteractiveScene(FrankaTrackingSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()
    print(f"[SCENE] num_envs={scene.num_envs} | num_joints={robot.num_joints} | num_bodies={robot.num_bodies}")
    print(f"[SCENE] joint names : {robot.joint_names}")
    print(f"[SCENE] body names  : {robot.body_names}")
    print(f"[SCENE] effort limits [N·m] = {[round(v, 3) for v in robot.data.joint_effort_limits.torch[0].tolist()]}")
    print(f"[SCENE] gravity = {not args_cli.no_gravity} | dt = {dt:.5f} s（{1.0 / dt:.0f} Hz）")

    # ---- 被控关节（默认 7 个臂关节；手指不参与本任务的观测/奖励/控制）----
    joint_ids, joint_names = _resolve_joint_ids(robot, args_cli.joint_expr)
    J = len(joint_ids)
    effort_limits = robot.data.joint_effort_limits.torch[:, joint_ids].to(dtype=torch.float32)
    pos_limits = robot.data.joint_pos_limits.torch[0, joint_ids].to(dtype=torch.float32)
    print(f"[MDP] 被控关节（{J} 个）：{joint_names}")
    print(f"[MDP] 关节限位 [rad] = {[[round(float(a), 4), round(float(b), 4)] for a, b in pos_limits.tolist()]}")

    # ---- 参考轨迹 ----
    traj_seed = args_cli.traj_seed if args_cli.traj_seed >= 0 else args_cli.seed
    traj = TrajectoryGenerator(
        scene.num_envs, joint_names, pos_limits, list(args_cli.traj_center), tuple(args_cli.traj_amp),
        tuple(args_cli.traj_freq), args_cli.harmonics, args_cli.traj_margin, traj_seed, device,
    )
    traj_info = traj.summary()
    limit_info = traj.check_limits(args_cli.steps, dt)
    print(f"[TRAJ] q_d(t) = q0 + Σ_h (A/h)·sin(2π·h·f·t + φ) | K={traj_info['harmonics']} | seed={traj_seed}")
    print(f"[TRAJ] q0 [rad]        = {traj_info['center_q0_rad']}")
    print(f"[TRAJ] A 采样 [rad]    = {traj_info['amp_rad_sampled']}")
    print(f"[TRAJ] A clamp 后 [rad] = {traj_info['amp_rad_clamped']}（clamp 比例 {traj_info['amp_clamp_ratio']}）")
    print(f"[TRAJ] f [Hz]          = {traj_info['freq_hz']}")
    print(f"[TRAJ] φ [rad]         = {traj_info['phase_rad']}")
    print(
        f"[TRAJ] 全 episode 限位检查：max|q_d|={limit_info['max_abs_qd_rad']:.4f} rad，"
        f"max|q̇_d|={limit_info['max_abs_qd_dot_rad_s']:.4f} rad/s，"
        f"max|q̈_d|={limit_info['max_abs_qd_ddot_rad_s2']:.4f} rad/s²，"
        f"离限位最近 {limit_info['min_margin_to_joint_limit_rad']:.4f} rad -> "
        f"{'不越限' if limit_info['inside_limits'] else '越限（必须调小 --traj_amp）'}"
    )

    # ---- 观测 / 奖励 ----
    obs_builder = ObservationBuilder(scene.num_envs, J, args_cli.history, args_cli.history_warmup, device)
    reward_fn = MultiObjectiveReward(
        {"pos": args_cli.wp, "vel": args_cli.wv, "tau": args_cli.wtau, "jerk": args_cli.wj}, device
    )
    print(f"[OBS] 单帧 {obs_builder.frame_dim} 维 {obs_builder.KEYS} × 历史 {obs_builder.H} 帧 = {obs_builder.obs_dim} 维")
    print(f"[REW] r = -{args_cli.wp:g}‖e_q‖² - {args_cli.wv:g}‖ė_q‖² - {args_cli.wtau:g}‖τ‖² - {args_cli.wj:g}‖Δτ‖²")

    # ---- FK（末端笛卡尔误差的度量基础）----
    fk = PandaFK(robot, device, args_cli.ee_body, args_cli.urdf or None)
    print(f"[FK] 运动链来源 = {fk.source}")
    print(f"[FK] 链 = {' -> '.join(fk.link_of_step)}（{fk.num_steps} 段）")
    pre = fk.check(
        robot.data.joint_pos.torch,
        robot.data.body_link_pos_w.torch[:, 0, :],
        robot.data.body_link_quat_w.torch[:, 0, :],
        robot.data.body_link_pos_w.torch,
    )
    fk_ok = pre["max_err_m"] <= args_cli.fk_tolerance
    print(f"[FK] 预检（默认位形）：max 残差 = {pre['max_err_m']:.3e} m（最差连杆 {pre['worst_link']}）"
          f"，阈值 {args_cli.fk_tolerance:g} m -> {'通过' if fk_ok else '不通过（末端误差不可信，报告里必须写明）'}")
    print(f"[FK] 逐连杆残差 [m] = {pre['per_link_max_err_m']}")

    # ---- 域随机化注入器（此时先不注入；场景按需切换）----
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
    print(f"[DR] 随机化 = {'开启' if rng.enabled else '关闭（--no_randomization）'} | seed = {args_cli.seed}")
    print(
        f"[DR] 范围：负载 {tuple(args_cli.payload_mass)} kg | 连杆 ×{tuple(args_cli.link_mass_scale)} | "
        f"damping {tuple(args_cli.joint_damping)} N·m·s/rad | 静摩擦 {tuple(args_cli.joint_friction)} N·m"
    )

    # ---- MDP 环境 ----
    env = TrajectoryTrackingMDP(
        scene, sim, robot, traj, obs_builder, reward_fn, fk, joint_ids, dt, args_cli.steps,
        fk_check=not args_cli.no_fk_check, init_at_reference=args_cli.init_at_reference,
    )

    # ---- PD 增益：默认由 M(q0) 对角元自动定（见 auto_gains 的说明）----
    # 先把手臂摆到参考中心 q0（轨迹就住在这附近），在这个位形上量惯量与定增益
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
    wn_eff = torch.sqrt(kp.reshape(-1)[:J] / i_diag.clamp(min=1e-9)) if kp.dim() != 3 else torch.full(
        (J,), float(args_cli.wn), device=device
    )
    print(f"[GAIN] M(q0) 对角元 I_jj [kg·m²] = {[round(float(v), 5) for v in i_diag.tolist()]}")
    if kp.dim() == 3:
        print(f"[GAIN] Kp 对角元 [N·m/rad] = {_gain_brief(kp)}")
        print(f"[GAIN] Kd 对角元 [N·m·s/rad] = {_gain_brief(kd)}")
    else:
        print(f"[GAIN] kp [N·m/rad] = {_gain_brief(kp)}")
        print(f"[GAIN] kd [N·m·s/rad] = {_gain_brief(kd)}")
    print(f"[GAIN] 等效带宽 ωn [rad/s] = {[round(float(v), 2) for v in wn_eff.tolist()]}")

    # ---- 多变量离散稳定裕度：对角/矩阵增益 × 耦合质量矩阵 ----
    # 单关节稳定条件（ωn·dt<2、kd·dt/I<2）只是必要条件：M(q) 不是对角阵时，闭环是 M q̈ = −Kp e − Kd ė，
    # 稳定性由 M⁻¹Kp / M⁻¹Kd 的**特征值**决定。这里直接把两个裕度算出来（并落盘），
    # 免得"手填一组 kp/kd 看着合理、实际在力矩上限之间颤振"。
    lam_p, lam_d = stability_margins(robot, joint_ids, kp, kd, dt)
    print(f"[GAIN] 多变量离散裕度：λmax(M⁻¹Kp)·dt² = {float(lam_p.max()):.4f}（稳定要 < 4）；"
          f"λmax(M⁻¹Kd)·dt = {float(lam_d.max()):.4f}（稳定要 < 2）")
    if float(lam_p.max()) >= 4.0 or float(lam_d.max()) >= 2.0:
        print("[WARN] 多变量离散裕度不足：闭环会颤振（表现为 sat_frac 高 + jerk 项大 + velRMSE≈速度上限）")

    # ---- 标称重力模型（“前馈+PID”里的前馈）：读**注入前**的连杆质量/质心 ----
    env.set_nominal_model(
        robot.data.body_mass.torch.to(dtype=torch.float32).clone(),
        robot.data.body_com_pose_b.torch.to(dtype=torch.float32)[:, :, :3].clone(),
    )
    ff_nominal = env.gravity_ff("nominal")
    ff_oracle = env.gravity_ff("oracle")
    ff_err = (ff_nominal - ff_oracle).abs()
    ff_rel = float(ff_err.max()) / max(float(ff_oracle.abs().max()), 1e-9)
    ff_ok = float(ff_err.max()) <= 0.5
    print(f"[FF] 标称重力模型自校验（当前位形，注入前）：G(q)|nominal = "
          f"{[round(float(v), 3) for v in ff_nominal[0].tolist()]} N·m")
    print(f"[FF] PhysX gravity_compensation_forces        = "
          f"{[round(float(v), 3) for v in ff_oracle[0].tolist()]} N·m")
    print(f"[FF] max|ΔG| = {float(ff_err.max()):.4f} N·m（相对 {ff_rel * 100:.2f}%，阈值 0.5 N·m）-> "
          f"{'通过（标称模型与 PhysX 一致）' if ff_ok else '不通过（标称模型有误，前馈结论不可信）'}")

    # ---- 依次跑 scenario ----
    results, dr_info = _run_scenarios(env, rng, args_cli, joint_names, effort_limits, kp, kd, i_diag, device)

    # ---- 结论：失配把基线跟踪精度拉低了多少 ----
    verdict = {"scenarios": list(results.keys())}
    if "pd_dr" in results and "pd_nodr" in results:
        dr, no = results["pd_dr"], results["pd_nodr"]
        rmse_dr = dr["joint_rmse_rad"]["mean_over_envs"]
        rmse_no = no["joint_rmse_rad"]["mean_over_envs"]
        ee_dr = dr["cartesian_error_cm"]["rmse_mean_over_envs"]
        ee_no = no["cartesian_error_cm"]["rmse_mean_over_envs"]
        verdict.update(
            {
                "joint_rmse_rad_pd_nodr": rmse_no,
                "joint_rmse_rad_pd_dr": rmse_dr,
                "joint_rmse_degrada_ratio": rmse_dr / max(rmse_no, 1e-12),
                "cartesian_rmse_cm_pd_nodr": ee_no,
                "cartesian_rmse_cm_pd_dr": ee_dr,
                "cartesian_rmse_degrada_ratio": ee_dr / max(ee_no, 1e-12),
                "dr_makes_tracking_harder": bool(rmse_dr > rmse_no),
            }
        )
    if "zero_dr" in results:
        verdict["cartesian_rmse_cm_zero_torque_dr"] = results["zero_dr"]["cartesian_error_cm"]["rmse_mean_over_envs"]
    verdict["fk_valid"] = bool(results[list(results)[0]]["fk"]["valid"])

    print("\n" + "=" * 78)
    print("[SUMMARY] scenario 对比（关节 RMSE [rad] / 末端 RMSE [cm] / 回报每步）")
    for name, res in results.items():
        print(
            f"  {name:9s} joint RMSE = {res['joint_rmse_rad']['mean_over_envs']:.6f} rad"
            f"  |  末端 RMSE = {res['cartesian_error_cm']['rmse_mean_over_envs']:.4f} cm"
            f"  |  回报/步 = {res['reward']['mean_per_step_per_env']:.4f}"
            f"  |  饱和步占比 = {res['controller']['sat_step_frac_any_joint']:.3f}"
        )
    if "joint_rmse_degrada_ratio" in verdict:
        print(f"[VERDICT] 注入失配后 PD 基线关节 RMSE ×{verdict['joint_rmse_degrada_ratio']:.2f}、"
              f"末端误差 ×{verdict['cartesian_rmse_degrada_ratio']:.2f} -> "
              f"{'失配确实让开环基线变差（RL 残差策略有可学的空间）' if verdict['dr_makes_tracking_harder'] else '失配没有让基线变差（需要检查注入/增益）'}")
    print(f"[VERDICT] FK 自校验：{'通过' if verdict['fk_valid'] else '不通过（末端误差不可信）'}")

    # ---- 落盘 ----
    result = {
        "config": {
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "dt": dt,
            "gravity": not args_cli.no_gravity,
            "seed": args_cli.seed,
            "scenarios": list(results.keys()),
            "num_controlled_joints": J,
            "controlled_joints": joint_names,
            "ee_body": args_cli.ee_body,
            "eval_skip": args_cli.eval_skip,
            "gains_mode": args_cli.gains,
            "pd_sign": args_cli.pd_sign,
            "kpd_gains": {
                "mode": args_cli.gains,
                "wn_target_rad_s": args_cli.wn,
                "zeta": args_cli.zeta,
                "mass_matrix_diag_kgm2_nominal": [round(float(v), 6) for v in i_diag.tolist()],
                "wn_effective_rad_s": [round(float(v), 3) for v in wn_eff.tolist()],
                "wn_dt": [round(float(v) * dt, 4) for v in wn_eff.tolist()],
                "kp": _gain_brief(kp),
                "kd": _gain_brief(kd),
                "gain_is_matrix": bool(kp.dim() == 3),
                "multivariable_margin": {
                    "lambda_max_MinvKp_dt2": float(lam_p.max()),
                    "lambda_max_MinvKd_dt": float(lam_d.max()),
                    "stable": bool(float(lam_p.max()) < 4.0 and float(lam_d.max()) < 2.0),
                },
            },
            "kp": _gain_brief(kp),
            "kd": _gain_brief(kd),
            "randomization_enabled": bool(rng.enabled),
            "payload_mass_range_kg": list(args_cli.payload_mass),
            "payload_body": args_cli.payload_body,
            "payload_com_m": list(args_cli.payload_com),
            "payload_mode": args_cli.payload_mode,
            "link_mass_scale_range": list(args_cli.link_mass_scale),
            "joint_damping_range_nms_per_rad": list(args_cli.joint_damping),
            "joint_friction_range_nm": list(args_cli.joint_friction),
        },
        "trajectory": traj_info,
        "trajectory_limits": limit_info,
        "observation": obs_builder.describe(),
        "reward": reward_fn.describe(),
        "fk": {"chain": fk.describe(), "preflight": pre, "preflight_ok": bool(fk_ok)},
        "gravity_model": {
            "ff_source": args_cli.ff,
            "G_nominal_nm": [round(float(v), 4) for v in ff_nominal[0].tolist()],
            "G_physx_nm": [round(float(v), 4) for v in ff_oracle[0].tolist()],
            "max_abs_diff_nm": float(ff_err.max()),
            "max_rel_diff": ff_rel,
            "validated": bool(ff_ok),
        },
        "dr": dr_info,
        "scenarios": results,
        "verdict": verdict,
        "gpu": {
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            "free_gib_after": round(torch.cuda.mem_get_info()[0] / 1024**3, 3),
            "free_gib_start": round(free_start / 1024**3, 3),
        },
    }
    if args_cli.out_json:
        path = os.path.abspath(args_cli.out_json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=float)
        print(f"[OUT] 结果已写入 {path}")
    print(f"[OUT] 显存：torch peak allocated {result['gpu']['peak_allocated_gib']:.3f} GiB | "
          f"跑完剩余 {result['gpu']['free_gib_after']:.2f} GiB（起始 {result['gpu']['free_gib_start']:.2f}）")
    print("=" * 78)
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()

