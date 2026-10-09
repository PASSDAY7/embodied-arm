#!/usr/bin/env python
"""W11 Day 2：把“非线性摩擦 + 未知负载失配”显式注入 Isaac Lab 的物理层（域随机化 / DR）。

为什么要做这件事（顶刊故事的第一块地基）
----------------------------------------
W10 已经证明：在 Isaac Lab 里我们能以 64 路并行、2 万+ env-steps/s 的吞吐跑“纯力矩”控制回路。
但那条回路的物理世界是**理想无损刚体**：关节阻尼 0、静摩擦 0、每个 env 的连杆质量完全相同。
而真实机器人上，经典控制器（PD、计算力矩、阻抗控制……）失准的头号原因恰恰是

1. **末端负载未知**：抓起来的东西有多重、重心在哪，控制器不知道；
2. **关节摩擦/阻尼非线性**：减速器的静摩擦、黏性阻尼随温度/装配漂移，标称模型里通常写成 0；
3. **连杆质量/惯量标称值不准**：CAD 标称值与实物相差 ±20% 是常态。

本脚本把这三类“失配（mismatch）”**显式注入到物理层**（不是加观测量、不是在控制器里加噪声），
于是同一个控制器在不同 env 上面对的是**不同的真实动力学**。W11 Day 3 的 RL 残差策略要学的，
就是“在不知道自己被注入了什么参数的前提下，把这个失配补回来”。

注入项（默认范围见 ``--help``，全部按 env 独立采样）
----------------------------------------------------
=================================  =========================================================
质量与惯量扰动（Payload & Mass）    末端法兰 ``panda_hand`` 挂 0~3 kg 未知负载（均匀分布，质心
                                    可配置在法兰外侧 ``--payload_com``）；每个连杆质量
                                    ×U(0.8, 1.2)（惯量按同比例缩放，保持回转半径不变）
关节阻尼失配（Damping）             臂关节 DOF damping ← U(0.5, 2.0) N·m·s/rad
关节静摩擦失配（Static friction）    臂关节静摩擦 τ_s ← U(0.1, 1.0) N·m（Isaac Sim 6.1 同时支持
                                    dynamic / viscous 系数，见写入接口说明）
=================================  =========================================================

.. note::
   任务书里写的是 ``U(0.5, 2.0) N·s/m`` 与 ``U(0.1, 1.0) N·m``。旋转关节（revolute）的阻尼单位
   应为 **N·m·s/rad**（把平动位移换成转角，数值口径一致），静摩擦是**力矩** N·m。因此脚本直接按
   这两个量纲写 PhysX：damping 走 ``set_dof_dampings``（封装成
   ``write_joint_damping_to_sim_index``），静摩擦走 ``set_dof_friction_properties``（封装成
   ``write_joint_friction_coefficient_to_sim_index``）。

验收判据：同力不同态（Same Torque, Different Dynamics）
--------------------------------------------------------
给 **全部 64 个 env 喂完全相同的正弦力矩** τ(t) = A⊙sin(φ + 2πft)（同一行张量广播出去，
脚本会打印“τ 跨 env 标准差 == 0”作为“输入完全相同”的证据），跑 200 步后统计
**跨 env 的关节位置标准差**（standard deviation of joint positions across envs）。
物理扰动若真的生效，环境之间必然发生运动分化：

* **基线（不注入）**：参数、初值、输入三者完全一致 → 跨 env 标准差应 ≈ 0
  （实测是 ~1e-6 rad 量级的 float32 噪声地板，见报告 §3.2）；
* **注入后**：跨 env 标准差应**显著大于 0**（默认阈值 1e-3 rad / 1e-3 m，实测超阈值一个数量级以上）。

脚本默认**在一次运行内跑完 A/B 两组对照**：先跑基线，复位关节状态后注入失配，再跑同一条力矩
轨迹。所以单条命令就能给出证据链（64 env / 200 步 / seed 42 的实测值，见
``docs/reports/w11_task1_dynamics_randomization.md``）::

    [BASE] τ 跨 env std = 0.000e+00 N·m      <- 输入完全相同（严格 0）
    [BASE] 全关节 std max = 2.161e-06 rad    <- 没有失配 → std 低于 1e-6 rad 量级
    [DR]   全关节 std max = 6.977e-02 rad    <- 注入失配 → 运动分化（×3.2e4）
    [DR]   末端位置 std   = 1.605e-02 m      <- 末端位置也分化（×1.4e4）
    [VERDICT] 同力不同态：PASS

对照组（``--no_randomization``）里 A/B 两组统计量**逐位相同**，用来证明"分化只来自注入"。

关于“摩擦是不是真的进了求解器”
------------------------------
参数写得进去 ≠ 物理上生效。脚本额外做一个**摩擦有效性探针**（``--no_friction_probe`` 可关）：
在**只开静摩擦、关掉阻尼**的配置下，给某个关节施加恒定 1 N·m（小于注入的静摩擦 5 N·m）：

* 无摩擦：关节立刻加速（|Δq| 大）；
* 有静摩擦：关节被“粘住”（|Δq| ≈ 0）。

若两者没有差异，就说明 PhysX 的 joint friction 没生效 —— 这种情况报告里必须写明，
不能拿“参数回读正确”当证据。

与 Isaac Lab 3.0 的 API 差异（和 W10 脚本踩过的坑一致）
-------------------------------------------------------
* 物理视图属性是 ``robot.root_view``（**不是**旧教程里的 ``robot.root_physx_view``）；
* 质量/惯量/质心写入用 ``set_masses_index`` / ``set_inertias_index`` / ``set_coms_index``
  （旧教程里的 ``robot.root_physx_view.set_masses`` 是绕过缓存的直写，随后读回可能看不到，
  见 ``isaaclab_physx/.../articulation_data.py::body_mass`` 的 docstring）；
* 关节阻尼/摩擦写入用 ``write_joint_damping_to_sim_index`` /
  ``write_joint_friction_coefficient_to_sim_index``（后者在 Isaac Sim ≥5.0 才同时支持
  static / dynamic / viscous 三项，本机 6.1 ✓）；
* ``--headless`` 命令行参数在 3.0 被删，改环境变量 ``HEADLESS=1``。

用法::

    # 主实验（推荐：仓库里的环境固化脚本负责激活 py312 / 挂 PYTHONPATH / 设 HEADLESS）
    bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py \
        --num_envs 64 --steps 200 --out_json docs/results/w11_dr_64env.json

    # 对照组（不注入任何失配：A/B 两组统计量应逐位相同）
    bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py \
        --num_envs 64 --steps 200 --no_randomization \
        --out_json docs/results/w11_dr_64env_control.json

    # 打开重力（默认关，理由见 make_franka_torque_cfg 的 docstring）+ 调大负载范围
    bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py --gravity --payload_mass 0 5

    # 跳过摩擦探针（只做"同力不同态"验收，省 2 秒钟）
    bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py --no_friction_probe

除 Franka USD 资产（``panda_instanceable.usd``，首次运行需联网下载）外不依赖新资产。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError as exc:
    # 最常见的坑：只 `conda activate py312` 就 `python 本脚本`。
    # Isaac Lab 在本机是**源码挂载**（/data/zzh/IsaacLab/source/* 靠 PYTHONPATH 生效），
    # 不是 pip 包，所以缺 PYTHONPATH 时会 ModuleNotFoundError: No module named 'isaaclab'。
    raise SystemExit(
        f"[ERR] 无法 import isaaclab（{exc}）。\n"
        "      本机的 Isaac Lab 是源码挂载（PYTHONPATH 指向 /data/zzh/IsaacLab/source/*），\n"
        "      只 conda activate py312 是不够的。请用仓库里的启动器运行：\n\n"
        "          bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py --num_envs 64\n\n"
        "      或者自己拼环境（等价做法）：\n"
        "          export PYTHONPATH=$(ls -d /data/zzh/IsaacLab/source/*/ | tr '\\n' ':')\n"
        "          HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python -u "
        f"{os.path.abspath(__file__)} --num_envs 64\n"
    ) from exc

##
# 命令行参数（AppLauncher 必须在所有重量级 import 之前启动 Kit）
##
parser = argparse.ArgumentParser(description="W11 Day 2：物理层失配注入（负载 / 质量 / 摩擦 / 阻尼）+ 同力不同态探针。")

# ---- 规模与积分 ----
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数。")
parser.add_argument("--steps", type=int, default=200, help="A/B 两组各自跑多少步（同一条力矩轨迹）。")
parser.add_argument("--dt", type=float, default=1.0 / 100.0, help="物理步长 [s]，默认 100 Hz。")
parser.add_argument("--env_spacing", type=float, default=2.5, help="环境间距 [m]。")
parser.add_argument("--gravity", action="store_true", help="打开重力（默认关，理由见 main() 顶部说明）。")

# ---- 力矩源（A/B 两组用的是同一条轨迹）----
parser.add_argument("--torque_amp", type=float, default=10.0, help="正弦力矩幅值 A [N·m]（会被关节力矩上限 clamp）。")
parser.add_argument("--freq", type=float, default=1.0, help="正弦频率 f [Hz]。")

# ---- 域随机化范围（全部是“每 env 独立采样”）----
parser.add_argument("--seed", type=int, default=42, help="随机种子（保证同一 seed 下可复现）。")
parser.add_argument("--no_randomization", action="store_true", help="基线对照：完全不注入失配（A/B 两组都不注入）。")
parser.add_argument(
    "--payload_mass", type=float, nargs=2, default=(0.0, 3.0), metavar=("MIN", "MAX"),
    help="末端未知负载质量范围 [kg]，均匀分布。",
)
parser.add_argument(
    "--payload_body", type=str, default="panda_hand", help="挂载负载的连杆名（默认法兰盘 panda_hand）。"
)
parser.add_argument(
    "--payload_com", type=float, nargs=3, default=(0.0, 0.0, 0.05), metavar=("X", "Y", "Z"),
    help="负载质心在载荷连杆自身坐标系下的位置 [m]（默认法兰外侧 5 cm）。",
)
parser.add_argument(
    "--payload_mode", choices=["parallel_axis", "mass"], default="parallel_axis",
    help="负载惯量处理：parallel_axis=平行轴定理合成质心/惯量（更严谨）；mass=只加质量、惯量按比例缩放。",
)
parser.add_argument(
    "--link_mass_scale", type=float, nargs=2, default=(0.8, 1.2), metavar=("LO", "HI"),
    help="连杆质量比例扰动范围（m ← m·U(LO,HI)）。",
)
parser.add_argument(
    "--link_mass_scope", choices=["all", "arm"], default="all",
    help="质量扰动作用范围：all=全部连杆（含手指）；arm=只扰动 panda_link* 连杆。",
)
parser.add_argument(
    "--joint_damping", type=float, nargs=2, default=(0.5, 2.0), metavar=("LO", "HI"),
    help="关节阻尼范围 [N·m·s/rad]（物理层 DOF damping，不是 actuator 的 k/d）。",
)
parser.add_argument(
    "--joint_friction", type=float, nargs=2, default=(0.1, 1.0), metavar=("LO", "HI"),
    help="关节静摩擦力矩范围 [N·m]。",
)
parser.add_argument(
    "--joint_expr", type=str, default=r"panda_joint[1-7]",
    help="注入摩擦/阻尼的关节正则（默认只注入 7 个臂关节，手指关节不注入）。",
)
parser.add_argument(
    "--no_dynamic_friction", action="store_true",
    help="只写静摩擦、不写 dynamic friction（默认同时写 dynamic=0.5×static，更接近真实库仑摩擦）。",
)

# ---- 探针 ----
parser.add_argument("--trace_every", type=int, default=20, help="每多少步打印一次跨 env 标准差时间序列。")
parser.add_argument("--divergence_threshold", type=float, default=1e-3, help="判定“显著大于 0”的阈值 [rad] / [m]。")
parser.add_argument("--no_friction_probe", action="store_true", help="跳过“摩擦是否真的生效”探针。")
parser.add_argument("--friction_probe_steps", type=int, default=100, help="摩擦探针步数。")
parser.add_argument("--friction_probe_tau", type=float, default=1.0, help="摩擦探针施加的恒定关节力矩 [N·m]。")
parser.add_argument("--friction_probe_mu", type=float, default=5.0, help="摩擦探针注入的静摩擦力矩 [N·m]（应大于施加力矩）。")
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
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402


##
# 资产配置：沿用 W10 的“纯力矩” Franka，并把“关节阻尼”腾出来给 DR 注入
##
def make_franka_torque_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot", gravity: bool = False) -> ArticulationCfg:
    """把 Franka Panda 改成“纯力矩”配置（与 W10 完全一致）。

    两个必须做的设置：

    1. ``stiffness=damping=0``：``FRANKA_PANDA_CFG`` 默认是隐式驱动器（stiffness=80、damping=4），
       而 3.0 的隐式驱动器内核算的是 ``τ = k·(q*−q) + d·(q̇*−q̇) + τ_ff``。若不清零 k/d，我们
       写进去的 10 N·m 会额外叠加一个未知 PD 项。清零后 ``τ_applied == τ_cmd``（仅受 effort_limit 限幅）。
       **注意**：清零的只是 actuator 层的 k/d，PhysX 的 DOF damping 由 W11 的 DR 代码在
       ``sim.reset()`` 之后显式注入覆盖 —— 这正是“关节阻尼失配”的注入点。
    2. 保留 ``armature=1e-3``：转子惯量，纯力矩模式下让关节更数值稳定。

    重力默认关闭：本实验只关心“同一输入力矩下，物理参数不同的环境如何分化”。若开着重力又只喂
    正弦力矩，手臂会先被重力拽到关节限位、再在限位上做“接触求解器压力测试”，位置离散度就被
    限位饱和污染了（无法归因到参数失配）。需要时可加 ``--gravity``，脚本会把“撞限位 env 数”一并打印。
    """
    cfg = clone(FRANKA_PANDA_CFG)
    cfg.prim_path = prim_path
    cfg.actuators = {
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            joint_effort_limit=87.0,
            stiffness=0.0,
            damping=0.0,
            armature=1e-3,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            joint_effort_limit=12.0,
            stiffness=0.0,
            damping=0.0,
            armature=1e-3,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            joint_effort_limit=200.0,
            stiffness=0.0,
            damping=0.0,
        ),
    }
    cfg.spawn.rigid_props.disable_gravity = not gravity
    return cfg


@configclass
class FrankaTorqueSceneCfg(InteractiveSceneCfg):
    """地面 + 灯光 + N 台 Franka（每台挂在 ``/World/envs/env_i/Robot``）。"""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = make_franka_torque_cfg(gravity=args_cli.gravity)


##
# 名字 -> 索引（用官方 API，失败则退回正则，避免版本差异把脚本卡死）
##
def _resolve_joint_ids(robot: Articulation, expr: str) -> tuple[list[int], list[str]]:
    try:
        ids, names = robot.find_joints(expr)
        return [int(i) for i in ids], list(names)
    except Exception:  # noqa: BLE001 - 版本差异时退回手工匹配
        pat = re.compile(expr)
        names = [n for n in robot.joint_names if pat.fullmatch(n)]
        return [robot.joint_names.index(n) for n in names], names


def _resolve_body_id(robot: Articulation, name: str) -> int:
    try:
        ids, _ = robot.find_bodies(name)
        return int(ids[0])
    except Exception:  # noqa: BLE001
        return robot.body_names.index(name)


##
# 零拷贝正弦力矩源（与 W10 同一套实现，是“输入完全相同”的保证）
##
class ZeroCopySineTorqueDriver:
    """零拷贝正弦力矩源：所有运算都在预分配的显存里原地完成。

    * 初始化时一次性算好 ``amp``（按关节力矩上限 clamp）与 ``phase``（逐关节相位差，避免所有
      关节同相位摆动）；
    * :meth:`compute` 只做 4 个原地 GPU op：``copy_`` / ``add_`` / ``sin(out=)`` / ``mul_``，
      **不分配显存、不与 CPU 交换任何数据**；
    * :meth:`write_to` 走 ``set_effort_index``：Isaac Lab 用 warp kernel 把 ``value`` 里的数值写进
      命令缓冲，全程在显存里，没有 host 往返。

    关键点（对 W11 验收至关重要）：``self._tau`` 的形状是 ``(1, num_joints)`` 广播/复制成
    ``(num_envs, num_joints)``，也就是说 **64 个 env 拿到的是逐元素完全相同的力矩**。
    脚本会实测 τ 跨 env 标准差 == 0 来证明这一点。
    """

    def __init__(self, robot: Articulation, amp: float, freq: float, device: str):
        limits = robot.data.joint_effort_limits.torch.to(torch.float32)
        self.num_envs, self.num_joints = limits.shape
        # 幅值既受命令行 A 限制，也不许超过物理关节力矩上限（否则会被内核 clamp，波形失真）
        self.amp = torch.minimum(torch.full((1, self.num_joints), float(amp), device=device), limits[:1])
        # 逐关节相位，让 7 个关节错开摆动（对物理无影响，只是让曲线更好看）
        self.phase = torch.arange(self.num_joints, device=device, dtype=torch.float32)
        self.phase.mul_(math.pi / self.num_joints)
        self.phase = self.phase.unsqueeze(0)
        self.omega = 2.0 * math.pi * float(freq)
        self._arg = torch.empty((self.num_envs, self.num_joints), device=device, dtype=torch.float32)
        self._tau = torch.empty_like(self._arg)

    def compute(self, t: float) -> None:
        """原地算出 ``τ(t)``，写入 :attr:`tau`。"""
        self._arg.copy_(self.phase)
        self._arg.add_(self.omega * t)
        torch.sin(self._arg, out=self._tau)
        self._tau.mul_(self.amp)

    @property
    def tau(self) -> torch.Tensor:
        return self._tau

    def write_to(self, robot: Articulation) -> None:
        """把 ``τ`` 写进 actuator 命令缓冲（warp kernel，无 host 往返）。"""
        robot.actuators.target_command.set_effort_index(value=self._tau, full_data=True)


def step_once(
    sim: SimulationContext, scene: InteractiveScene, robot: Articulation, driver: ZeroCopySineTorqueDriver, t: float, dt: float
) -> None:
    """一步完整回路：算力矩 -> 写命令 -> 推给 PhysX -> 步进 -> 刷新 torch 视图。"""
    driver.compute(t)
    driver.write_to(robot)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)


##
# 域随机化注入器：把“失配”写进 PhysX 的**模型属性**（mass / inertia / COM / DOF damping / DOF friction）
##
class DynamicsRandomizer:
    """按 env 采样一组“失配参数”，并通过张量接口写进 PhysX。

    用到的底层接口（Isaac Lab 3.0 / ``isaaclab_physx`` 后端）：

    ==================  =====================================================  ==========================
    物理量               本脚本调用                                              最终落到 PhysX
    ==================  =====================================================  ==========================
    连杆质量             ``set_masses_index(masses=(E,B))``                      ``set_masses``
    连杆惯量             ``set_inertias_index(inertias=(E,B,9))``               ``set_inertias``
    连杆质心             ``set_coms_index(coms=(E,B,7))``                        ``set_coms``
    关节阻尼             ``write_joint_damping_to_sim_index(damping=(E,J))``    ``set_dof_dampings``
    关节静/动摩擦         ``write_joint_friction_coefficient_to_sim_index(...)``  ``set_dof_friction_properties``
    ==================  =====================================================  ==========================

    三个物理约定（写进论文方法节前必须交代清楚）：

    1. **负载合成用平行轴定理**（``--payload_mode parallel_axis``，默认）。设连杆质量 m_h、其 COM 在
       连杆坐标系的位置 c_h、绕自身 COM 的惯量 I_h；负载质量 m_p、质心位置 p_p（按点质量处理，
       自身惯量≈0）。合成后::

           m = m_h + m_p
           c = (m_h·c_h + m_p·p_p) / m
           I = I_h + m_h·PA(c_h − c) + m_p·PA(p_p − c),   PA(d) = (‖d‖²·E₃ − d·dᵀ)

       平行轴项 m·PA(d) 是半正定的，所以合成惯量一定正定 —— 不会把仿真写崩。
       若 ``set_coms_index`` 在本机不可用，脚本自动退回 ``--payload_mode mass``（只加质量、惯量按质量
       比例缩放，等价于“负载质心与手 COM 重合”）并打印 ``[WARN]``，绝不静默糊弄。
    2. **连杆质量扰动同时缩放惯量**（I ← I·s，s = m_new/m_old）。这是保持“质量分布形状（回转半径）”
       不变的常用 DR 近似；若只缩放质量、不动惯量，会得到“质量变大但转不动”的非物理刚体。
    3. **摩擦/阻尼注入的是物理层 DOF 属性，不是 actuator 的 k/d**。本脚本 actuator 层 k=d=0
       （见 :func:`make_franka_torque_cfg`），所以这里的阻尼值不会被任何 PD 项掩盖。
    """

    def __init__(self, robot: Articulation, num_envs: int, device: str) -> None:
        self.robot = robot
        self.device = device
        self.enabled = not args_cli.no_randomization
        self.num_envs = num_envs
        self.num_bodies = robot.num_bodies
        self.joint_ids, self.joint_names = _resolve_joint_ids(robot, args_cli.joint_expr)
        self.hand_id = _resolve_body_id(robot, args_cli.payload_body)
        self.payload_com = torch.tensor(args_cli.payload_com, dtype=torch.float32, device=device)
        # CPU 采样 + 固定种子：同一 --seed 在任何设备上得到同一组失配参数（可复现）
        self.gen = torch.Generator(device="cpu").manual_seed(int(args_cli.seed))
        self.com_path: str = "off"
        self.com_fallback = False
        # 采样结果（供打印/落盘/探针复用）
        self.payload_mass: torch.Tensor | None = None
        self.mass_scale: torch.Tensor | None = None
        self.damping: torch.Tensor | None = None
        self.friction: torch.Tensor | None = None

    # ---------------- 采样 ----------------
    def _uniform(self, shape: tuple[int, ...], rng_range) -> torch.Tensor:
        lo, hi = float(rng_range[0]), float(rng_range[1])
        x = torch.rand(shape, generator=self.gen, dtype=torch.float32)
        return (x * (hi - lo) + lo).to(self.device)

    def sample(self) -> None:
        E, B = self.num_envs, self.num_bodies
        if args_cli.link_mass_scope == "all":
            self.mass_scale = self._uniform((E, B), args_cli.link_mass_scale)
        else:
            # 只扰动 panda_link0..7（不含手指），其它连杆保持 1.0
            arm_ids = [i for i, n in enumerate(self.robot.body_names) if re.fullmatch(r"panda_link\d", n)]
            self.mass_scale = torch.ones((E, B), dtype=torch.float32, device=self.device)
            self.mass_scale[:, arm_ids] = self._uniform((E, len(arm_ids)), args_cli.link_mass_scale)
        self.payload_mass = self._uniform((E,), args_cli.payload_mass)
        self.damping = self._uniform((E, len(self.joint_ids)), args_cli.joint_damping)
        self.friction = self._uniform((E, len(self.joint_ids)), args_cli.joint_friction)

    # ---------------- 注入 ----------------
    def apply(self) -> None:
        """把所有失配一次性写进 PhysX（**必须在 ``sim.reset()`` 之后调用**，否则会被初始化覆盖）。"""
        if not self.enabled:
            return
        if self.mass_scale is None:
            self.sample()

        robot, dev = self.robot, self.device
        E, B, h = self.num_envs, self.num_bodies, self.hand_id

        # ---- 注入前的标称值（USD 里的值，直接读 PhysX 张量视图）----
        m0 = robot.data.body_mass.torch.to(dtype=torch.float32).clone()          # (E,B)   [kg]
        I0 = robot.data.body_inertia.torch.to(dtype=torch.float32).clone()       # (E,B,9) [kg·m²]
        com0 = robot.data.body_com_pose_b.torch.to(dtype=torch.float32).clone()  # (E,B,7) [pos, quat xyzw]

        # 记录标称值（回读时对比，用来证明“写的确实不是标称值”）
        self.nominal_hand_mass = float(m0[0, h])
        self.nominal_hand_com = com0[:, h, :3].clone()
        self.nominal_hand_inertia = I0[:, h].reshape(E, 3, 3).clone()

        # ---- (1) 连杆质量 ×U(0.8,1.2)：惯量同比例缩放（回转半径不变）----
        m = m0 * self.mass_scale
        I = I0 * self.mass_scale.unsqueeze(-1)
        com = com0.clone()

        # ---- (2) 末端未知负载：质量 + 惯量（+ 质心）----
        mp = self.payload_mass                                  # (E,)
        m_h = m[:, h].clone()                                   # (E,) 缩放后、加负载前的质量
        m[:, h] = m_h + mp                                      # 加负载后的质量
        I_h = I[:, h].reshape(E, 3, 3)                          # (E,3,3) 缩放后、加负载前的惯量
        if args_cli.payload_mode == "mass":
            I[:, h] = (I_h * (m[:, h] / m_h)[:, None, None]).reshape(E, 9)
        else:
            I_new, c_new = self._combine_payload(
                I_h, com[:, h, :3].clone(), m_h, mp, self.payload_com.unsqueeze(0).expand(E, 3)
            )
            I[:, h] = I_new.reshape(E, 9)
            com[:, h, :3] = c_new

        # ---- 写回 PhysX（模型属性；Isaac Lab 内部会转成 CPU 缓冲再 set）----
        robot.set_masses_index(masses=m.contiguous())
        robot.set_inertias_index(inertias=I.contiguous())
        if args_cli.payload_mode == "parallel_axis":
            path = self._write_coms(com.contiguous())
            self.com_path = path or "failed -> fallback_mass"
            if path is None:
                # 质心写不进去 -> 退化为 mass 模式（只加质量 + 惯量按比例缩放），并明确告警
                self.com_fallback = True
                I[:, h] = (I_h * (m[:, h] / m_h)[:, None, None]).reshape(E, 9)
                robot.set_inertias_index(inertias=I.contiguous())
                print("[WARN] set_coms 两条路径都不可用：负载质心未参与动力学，已退化为 payload_mode=mass")
        elif args_cli.payload_mode == "mass":
            self.com_path = "off (mass mode)"

        # ---- (3) 关节阻尼 + 静/动摩擦（物理层 DOF 属性）----
        jid = torch.tensor(self.joint_ids, dtype=torch.int32, device=dev)
        robot.write_joint_damping_to_sim_index(damping=self.damping, joint_ids=jid)
        mu_d = None if args_cli.no_dynamic_friction else 0.5 * self.friction
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.friction,
            joint_dynamic_friction_coeff=mu_d,
            joint_ids=jid,
        )
        print(f"[DR] 注入完成：关节 {self.joint_names}（{len(self.joint_ids)} 个），负载连杆 = {args_cli.payload_body}"
              f"（body_id={h}），质心写入路径 = {self.com_path}")

    # ---------------- 惯量/质心合成（平行轴定理） ----------------
    @staticmethod
    def _parallel_axis(mass: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """平行轴项 ``m·PA(d) = m·(‖d‖²E₃ − d dᵀ)``；mass (E,)、d (E,3) -> (E,3,3)。"""
        eye = torch.eye(3, dtype=d.dtype, device=d.device).expand(d.shape[0], 3, 3)
        dd = d.unsqueeze(-1) * d.unsqueeze(-2)
        return mass[:, None, None] * (d.pow(2).sum(dim=-1)[:, None, None] * eye - dd)

    def _combine_payload(
        self, I_h: torch.Tensor, com_h: torch.Tensor, m_h: torch.Tensor, m_p: torch.Tensor, p_p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """点质量负载 + 连杆 -> 合成刚体。返回 ``(I_new (E,3,3), com_new (E,3))``。"""
        m = m_h + m_p
        com = (com_h * m_h[:, None] + p_p * m_p[:, None]) / m[:, None]
        I = I_h + self._parallel_axis(m_h, com_h - com) + self._parallel_axis(m_p, p_p - com)
        return I, com

    def _write_coms(self, com_flat: torch.Tensor) -> str | None:
        """把 (E,B,7) 质心位姿写进 PhysX；返回走通的路径名，两条都失败返回 None。"""
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

    # ---------------- 回读（证明“写进去了”） ----------------
    def readback(self) -> dict:
        """从 PhysX 回读模型属性，并与注入前的标称值对比。"""
        robot, h = self.robot, self.hand_id
        m = robot.data.body_mass.torch.to(dtype=torch.float32)
        I = robot.data.body_inertia.torch.to(dtype=torch.float32).reshape(self.num_envs, self.num_bodies, 3, 3)
        com = robot.data.body_com_pose_b.torch.to(dtype=torch.float32)
        damp = robot.data.joint_damping.torch.to(dtype=torch.float32)[:, self.joint_ids]
        mu_s = robot.data.joint_friction_coeff.torch.to(dtype=torch.float32)[:, self.joint_ids]
        mu_d = robot.data.joint_dynamic_friction_coeff.torch.to(dtype=torch.float32)[:, self.joint_ids]
        nominal_com = getattr(self, "nominal_hand_com", None)
        nominal_mass = getattr(self, "nominal_hand_mass", None)
        out = {
            "hand_mass_kg": _stat(m[:, h]),
            "hand_inertia_trace_kgm2": _stat(I[:, h].diagonal(dim1=-2, dim2=-1).sum(dim=-1)),
            # 惯量矩阵最小特征值：> 0 说明写进去的惯量是正定的（物理上合法）
            "hand_inertia_min_eig": _stat(torch.linalg.eigvalsh(I[:, h]).min(dim=-1).values),
            "hand_com_shift_m": _stat((com[:, h, :3] - nominal_com).norm(dim=-1)) if nominal_com is not None else None,
            "joint_damping_nms_per_rad": _stat(damp),
            "joint_friction_static_nm": _stat(mu_s),
            "joint_friction_dynamic_nm": _stat(mu_d),
            "hand_mass_nominal_kg": nominal_mass,
        }
        return out

    def sampled_summary(self) -> dict:
        """采样值统计（“每 env 不同”的直接证据）。"""
        if not self.enabled:
            return {}
        return {
            "payload_mass_kg": _stat(self.payload_mass),
            "link_mass_scale": _stat(self.mass_scale),
            "joint_damping_nms_per_rad": _stat(self.damping),
            "joint_friction_static_nm": _stat(self.friction),
            "link_mass_scale_per_body_mean": [round(float(v), 4) for v in self.mass_scale.mean(dim=0)],
        }


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
# 探针 1：同力不同态（A/B 对照）
##
def reset_robot_state(scene: InteractiveScene, robot: Articulation, sim: SimulationContext, dt: float) -> None:
    """把关节状态复位到默认位形，保证 A/B 两组从**同一初值**出发。

    只用 index 接口写关节状态（直接改 PhysX 缓冲、立即生效），再调一次 ``scene.reset()`` 清掉
    actuator 命令缓冲（W10 已实测：``scene.reset()`` 不会动关节位置），最后推一次
    ``write_data_to_sim + update`` 让 torch 侧视图与 PhysX 对齐。**不推进物理**。
    """
    pos = robot.data.default_joint_pos.torch.clone()
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim_index(position=pos, velocity=vel)
    scene.reset()
    scene.write_data_to_sim()
    scene.update(dt)


def _end_effector_joint_id(robot: Articulation) -> int:
    """末端关节 = ``panda_joint7``（找不到就退回最后一个关节）。"""
    try:
        ids, _ = _resolve_joint_ids(robot, "panda_joint7")
        if ids:
            return int(ids[0])
    except Exception:  # noqa: BLE001
        pass
    return robot.num_joints - 1


def run_rollout(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    driver: ZeroCopySineTorqueDriver,
    dt: float,
    steps: int,
    trace_every: int,
    tag: str,
) -> dict:
    """喂同一条正弦力矩轨迹（所有 env 逐元素相同），记录跨 env 的关节位置离散度。

    关键测量（详见 :func:`_dispersion`）：

    * ``tau_std_max``：τ 在 env 维度上的最大标准差 —— 必须恒为 0，否则“同力”本身就不成立；
    * ``joint_std_max`` / ``joint_pos_std_final``：各关节位置在 num_envs 个 env 之间的标准差 [rad]；
    * ``joint7_std``：末端关节（``panda_joint7``）的上述标准差；
    * ``hand_pos_std`` / ``hand_pos_rel_base_std``：末端连杆（``panda_hand``）位置在 env 之间的标准差 [m]
      （**相对各自 env 原点 / 相对机器人基座**，否则量到的是 env 网格间距）；
    * ``envs_at_arm_joint_limit`` / ``arm_joint_limit_counts``：末步有多少 env 顶到**臂关节**限位
      （诚实指标：限位饱和会污染“离散度”的解释，必须一并报告）；
    * ``wall_s`` / ``sps`` / ``env_steps_per_s``：这 200 步的墙钟吞吐（含每步一次 τ 检查）。
    """
    hand_id = _resolve_body_id(robot, args_cli.payload_body)
    ee_id = _end_effector_joint_id(robot)
    # τ 跨 env 标准差在显存里累积，整条 rollout 只做一次 host 同步（否则每步 .item() 会拖慢计时）
    tau_std_max = torch.zeros((), dtype=torch.float32, device=robot.device)
    hist: list[dict] = []
    t0 = time.perf_counter()
    for k in range(steps):
        driver.compute(k * dt)
        driver.write_to(robot)
        # “输入完全相同”的证据：同一行 τ 复制到所有 env，std 必须严格为 0
        tau_std_max = torch.maximum(tau_std_max, driver.tau.std(dim=0).max())
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if (k + 1) % trace_every == 0 or k + 1 == steps:
            rec = {"step": k + 1, **_dispersion(scene, robot, hand_id, ee_id)}
            hist.append(rec)
            print(
                f"[{tag}] step {k + 1:4d}/{steps} | 关节 std max {rec['joint_std_max']:.4e} rad"
                f" | joint7 std {rec['joint7_std']:.4e} rad | 手位置 std {rec['hand_pos_std']:.4e} m"
            )
    wall = time.perf_counter() - t0
    print(
        f"[{tag}] {steps} 步墙钟 {wall:.3f} s -> {steps / wall:.1f} SPS"
        f"（{steps * scene.num_envs / wall:.0f} env-steps/s，含每步 1 次 τ 检查，无 host 往返）"
    )

    out = {
        "tag": tag,
        "steps": steps,
        "num_envs": scene.num_envs,
        "tau_std_max": float(tau_std_max.item()),
        "same_command": bool(float(tau_std_max.item()) == 0.0),
        "wall_s": wall,
        "sps": steps / wall,
        "env_steps_per_s": steps * scene.num_envs / wall,
    }
    out.update(_dispersion(scene, robot, hand_id, ee_id))
    out |= {
        "joint_pos_std_final": [float(v) for v in robot.data.joint_pos.torch.std(dim=0)],
        "end_effector_joint_name": robot.joint_names[ee_id],
        "end_effector_joint_id": ee_id,
        "trace": hist,
    }
    return out


def _dispersion(scene: InteractiveScene, robot: Articulation, hand_id: int, ee_id: int) -> dict:
    """当前时刻的跨 env 离散度（所有“位置”都换算成**相对量**，否则量到的是 env 网格间距）。

    三个关键细节：

    * ``robot.data.body_link_pos_w`` 是**世界系**位置，直接求跨 env 标准差会把 env 之间的
      2.5 m 网格间距算进去（读出恒定的 2.04 m，与物理完全无关）。必须减去 ``scene.env_origins``
      或机器人基座位置；
    * 关节 position 是相对量，直接求 std 即可；
    * “撞限位”要**分开统计臂关节和手指**：被抓取物任务的夹爪指关节几乎瞬间顶到行程端点，
      把手指算进来会让“撞限位 env 数”永远是 100%，掩盖真正需要关注的臂关节饱和。
    """
    pos = robot.data.joint_pos.torch                                     # (E,J)
    pos_std = pos.std(dim=0)                                             # (J,) 跨 env
    hand_w = robot.data.body_link_pos_w.torch[:, hand_id]                # (E,3)
    hand_rel = hand_w - scene.env_origins                                # 相对各自 env 原点
    hand_rel_base = hand_w - robot.data.body_link_pos_w.torch[:, 0]      # 相对机器人基座连杆
    lim = robot.data.joint_pos_limits.torch                              # (E,J,2)
    sat_all = (pos <= lim[..., 0] + 1e-3) | (pos >= lim[..., 1] - 1e-3)
    arm_ids, arm_names = _resolve_joint_ids(robot, args_cli.joint_expr)
    sat_arm = sat_all[:, arm_ids]                                        # (E, len(arm_ids))
    return {
        "joint_std_max": float(pos_std.max().item()),
        "joint_std_mean": float(pos_std.mean().item()),
        "joint7_std": float(pos_std[ee_id].item()),
        "hand_pos_std": float(hand_rel.std(dim=0).norm().item()),
        "hand_pos_std_xyz_m": [float(v) for v in hand_rel.std(dim=0)],
        "hand_pos_rel_base_std": float(hand_rel_base.std(dim=0).norm().item()),
        "envs_at_arm_joint_limit": int(sat_arm.any(dim=-1).sum().item()),
        "arm_joint_limit_counts": {n: int(c) for n, c in zip(arm_names, sat_arm.sum(dim=0))},
        "envs_at_any_joint_limit": int(sat_all.any(dim=-1).sum().item()),
    }


##
# 探针 2：摩擦是否真的进了求解器（参数写进去 ≠ 物理上生效）
##
def friction_probe(sim, scene, robot, dt, steps, tau_val, mu_val) -> dict:
    """证明 static friction **真的进了求解器**，而不只是“参数写得进去”。

    控制变量：

    * 被测关节 ``panda_joint4``（effort limit 87 N·m，够大，不会被限幅干扰）；
    * 所有臂关节 damping 置 0（排除黏性阻尼的干扰，只留下静摩擦这一个变量）；
    * 被测关节施加恒定 1 N·m，**小于**注入的静摩擦 5 N·m；
    * 两组对照：静摩擦 0（无摩擦） vs 静摩擦 5 N·m。

    期望：无摩擦组立刻加速（|Δq| 显著）；有摩擦组被静摩擦“粘住”（|Δq| ≈ 0）。
    若两者没有数量级差异，说明 PhysX 的 joint friction 没生效 —— 那是必须在报告里写明的事实。
    """
    E, J = scene.num_envs, robot.num_joints
    dev = robot.device
    j_idx = _resolve_joint_ids(robot, "panda_joint4")[0][0]
    arm_ids = _resolve_joint_ids(robot, args_cli.joint_expr)[0]
    jid_all = torch.tensor(arm_ids, dtype=torch.int32, device=dev)
    jid_one = torch.tensor([j_idx], dtype=torch.int32, device=dev)
    tau_const = torch.zeros((E, J), dtype=torch.float32, device=dev)
    tau_const[:, j_idx] = float(tau_val)
    out: dict = {"joint": robot.joint_names[j_idx], "applied_torque_nm": float(tau_val), "trials": {}}
    for tag, mu in (("no_friction", 0.0), ("static_friction", float(mu_val))):
        reset_robot_state(scene, robot, sim, dt)
        robot.write_joint_damping_to_sim_index(damping=torch.zeros((E, len(arm_ids)), device=dev), joint_ids=jid_all)
        # 静/动摩擦必须一起写：PhysX 有硬约束 μ_s ≥ μ_d，只把静摩擦清零会报
        # “Static friction effort must be greater than or equal to dynamic friction effort.”
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=torch.full((E, 1), mu, device=dev),
            joint_dynamic_friction_coeff=torch.full((E, 1), mu, device=dev),
            joint_ids=jid_one,
        )
        q0 = robot.data.joint_pos.torch[:, j_idx].clone()
        for _ in range(steps):
            robot.actuators.target_command.set_effort_index(value=tau_const, full_data=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
        dq = (robot.data.joint_pos.torch[:, j_idx] - q0).abs()
        out["trials"][tag] = {
            "static_friction_nm": mu,
            "dq_mean_rad": float(dq.mean().item()),
            "dq_max_rad": float(dq.max().item()),
        }
    a = out["trials"]["no_friction"]["dq_mean_rad"]
    b = out["trials"]["static_friction"]["dq_mean_rad"]
    out["ratio_no_friction_over_friction"] = float(a / max(b, 1e-12))
    out["friction_effective"] = bool(a > 1e-3 and a > 10.0 * max(b, 1e-9))
    return out


def _fmt_stat(d: dict | None, unit: str = "") -> str:
    """把 :func:`_stat` 的 dict 打成一行字。"""
    if d is None:
        return "-"
    return f"min {d['min']:.4g} / mean {d['mean']:.4g} / max {d['max']:.4g} / std {d['std']:.4g}{unit}"


##
# 主流程
##
def main() -> dict:
    """A 基线 -> 复位 -> 注入失配 -> B 对照 -> 摩擦探针 -> 汇总/落盘。

    为什么默认关重力：本实验的判据是“同一输入力矩下，物理参数不同的环境会不会分化”。开着重力 +
    只喂正弦力矩时，手臂会先被重力拽到关节限位（panda 的静止位形需要几十 N·m 的重力补偿力矩），
    末步的“离散度”就被限位饱和定义，无法归因到参数失配。需要重力场景时加 ``--gravity``，脚本会把
    “撞限位 env 数”一并打印，便于判断结果是否被饱和污染。
    """
    device = args_cli.device or "cuda:0"
    torch.cuda.reset_peak_memory_stats()

    print("=" * 78)
    print(f"[ENV] torch {torch.__version__} | warp {wp.__version__} | isaacsim 6.1.0")
    print(f"[ENV] device = {device} | {torch.cuda.get_device_name(0)}")
    free_start, total = torch.cuda.mem_get_info()
    print(f"[ENV] GPU memory total = {total / 1024**3:.2f} GiB, free = {free_start / 1024**3:.2f} GiB")

    # ---- 仿真上下文 + 场景 ----
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=device))
    scene = InteractiveScene(FrankaTorqueSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()
    print(f"[SCENE] num_envs={scene.num_envs} | num_joints={robot.num_joints} | num_bodies={robot.num_bodies}")
    print(f"[SCENE] joint names : {robot.joint_names}")
    print(f"[SCENE] body names  : {robot.body_names}")
    print(f"[SCENE] effort limits [N·m] = {robot.data.joint_effort_limits.torch[0].tolist()}")
    print(f"[SCENE] gravity = {bool(args_cli.gravity)}（disable_gravity = {not args_cli.gravity}）")

    driver = ZeroCopySineTorqueDriver(robot, args_cli.torque_amp, args_cli.freq, device)
    rng = DynamicsRandomizer(robot, scene.num_envs, device)
    print(f"[DRIVER] τ = A ⊙ sin(φ + 2πft) | A = {driver.amp[0].tolist()} | f = {args_cli.freq} Hz | dt = {dt:.5f} s")
    print(f"[DR] 随机化 = {'关闭（--no_randomization）' if not rng.enabled else '开启'} | seed = {args_cli.seed}")
    print(
        f"[DR] 目标范围：负载 {args_cli.payload_mass} kg | 连杆 ×{args_cli.link_mass_scale} | "
        f"damping {args_cli.joint_damping} N·m·s/rad | 静摩擦 {args_cli.joint_friction} N·m"
    )

    # ---- A 组：不注入（基线）----
    reset_robot_state(scene, robot, sim, dt)
    base = run_rollout(sim, scene, robot, driver, dt, args_cli.steps, args_cli.trace_every, "BASE")

    # ---- 复位 + 注入失配（必须在最后一次 reset 之后）----
    reset_robot_state(scene, robot, sim, dt)
    t_inject = time.perf_counter()
    rng.apply()
    inject_wall = time.perf_counter() - t_inject
    if rng.enabled:
        print(f"[DR] 注入墙钟 {inject_wall:.4f} s（一次性写 {scene.num_envs} 个 env 的 mass/inertia/COM/damping/friction）")
    rb = rng.readback()
    sm = rng.sampled_summary()
    if rng.enabled:
        print("[DR] 采样（每 env 独立）：")
        print(f"[DR]   负载质量 [kg]              : {_fmt_stat(sm['payload_mass_kg'])}")
        print(f"[DR]   连杆质量比例               : {_fmt_stat(sm['link_mass_scale'])}")
        print(f"[DR]   关节 damping [N·m·s/rad]   : {_fmt_stat(sm['joint_damping_nms_per_rad'])}")
        print(f"[DR]   关节静摩擦 [N·m]           : {_fmt_stat(sm['joint_friction_static_nm'])}")
    print("[DR] PhysX 回读（证明写进去了；标称值来自注入前的读回）：")
    nom_mass = rb["hand_mass_nominal_kg"]
    print(
        f"[DR]   {args_cli.payload_body} 质量 [kg]      : {_fmt_stat(rb['hand_mass_kg'])}"
        + (f"（标称 {nom_mass:.4g}）" if nom_mass is not None else "（未注入：这就是标称值）")
    )
    print(f"[DR]   {args_cli.payload_body} 惯量迹 [kg·m²] : {_fmt_stat(rb['hand_inertia_trace_kgm2'])}")
    print(f"[DR]   {args_cli.payload_body} 惯量最小特征值 : {_fmt_stat(rb['hand_inertia_min_eig'])}（>0 表示正定）")
    if rb["hand_com_shift_m"] is not None:
        print(f"[DR]   {args_cli.payload_body} 质心位移 [m]   : {_fmt_stat(rb['hand_com_shift_m'])}")
    print(f"[DR]   关节 damping [N·m·s/rad]   : {_fmt_stat(rb['joint_damping_nms_per_rad'])}")
    print(f"[DR]   关节静摩擦 [N·m]           : {_fmt_stat(rb['joint_friction_static_nm'])}")
    print(f"[DR]   关节动摩擦 [N·m]           : {_fmt_stat(rb['joint_friction_dynamic_nm'])}")

    # ---- B 组：注入后跑同一条力矩轨迹 ----
    drift = run_rollout(
        sim, scene, robot, driver, dt, args_cli.steps, args_cli.trace_every, "DR" if rng.enabled else "BASE2"
    )

    # ---- 验收：同力不同态 ----
    thr = float(args_cli.divergence_threshold)
    base_ee = base["joint7_std"]
    dr_ee = drift["joint7_std"]
    base_hp = base["hand_pos_std"]
    dr_hp = drift["hand_pos_std"]
    same_cmd = bool(base["same_command"] and drift["same_command"])
    # 判据：“输入完全相同”（τ 跨 env std == 0）且“注入后显著分化”（> 阈值）且“基线不分化”
    if rng.enabled:
        passed = bool(same_cmd and dr_ee > thr and dr_hp > thr and base_ee <= thr and base_hp <= thr)
        verdict_txt = "PASS" if passed else "NOT PASS"
    else:
        # 对照组（--no_randomization）：A/B 之间只多了一次“复位 + 空注入”，若仍不分化，
        # 说明“分化”确实来自注入的失配，而不是来自复位/第二次 rollout 本身。
        passed = bool(same_cmd and dr_ee <= thr and dr_hp <= thr)
        verdict_txt = "CONTROL-OK（对照组，未注入 → 不分化，符合预期）" if passed else "CONTROL-异常分化"

    print("-" * 78)
    print(
        f"[RESULT] 输入相同性：A 组 τ 跨 env std max = {base['tau_std_max']:.3e} N·m | "
        f"B 组 = {drift['tau_std_max']:.3e} N·m -> {'所有 env 的力矩逐元素相同' if same_cmd else '不一致！'}"
    )
    print(
        f"[RESULT] A 基线  ：{base['end_effector_joint_name']} std = {base_ee:.3e} rad | "
        f"全关节 std max = {base['joint_std_max']:.3e} rad | 末端位置 std = {base_hp:.3e} m | "
        f"撞臂关节限位 env = {base['envs_at_arm_joint_limit']}/{base['num_envs']} | "
        f"末端相对基座 {base['hand_pos_rel_base_std']:.3e} m"
    )
    print(
        f"[RESULT] B 注入后：{drift['end_effector_joint_name']} std = {dr_ee:.3e} rad | "
        f"全关节 std max = {drift['joint_std_max']:.3e} rad | 末端位置 std = {dr_hp:.3e} m | "
        f"撞臂关节限位 env = {drift['envs_at_arm_joint_limit']}/{drift['num_envs']} | "
        f"末端相对基座 {drift['hand_pos_rel_base_std']:.3e} m"
    )
    print(
        f"[RESULT] 分化倍率：末端关节 ×{dr_ee / max(base_ee, 1e-12):.3g} | "
        f"末端位置 ×{dr_hp / max(base_hp, 1e-12):.3g}（B 组相对 A 组）"
    )
    print(f"[RESULT] 逐关节 std（B 组末步, rad）：{['%.3e' % v for v in drift['joint_pos_std_final']]}")
    print(f"[RESULT] 臂关节撞限位 env 数（A 基线 / B 注入，共 {drift['num_envs']} env）：")
    for name, cnt_a in base["arm_joint_limit_counts"].items():
        print(f"[RESULT]   {name:14s}: {cnt_a:3d} / {drift['arm_joint_limit_counts'][name]:3d}")
    print(
        f"[RESULT] 任意关节撞限位 env 数（含手指）：A {base['envs_at_any_joint_limit']} / "
        f"B {drift['envs_at_any_joint_limit']}（手指行程只有 4 cm，恒为 100% 属正常）"
    )
    print(f"[VERDICT] 同力不同态：{verdict_txt}（阈值 {thr:g} rad / {thr:g} m）")
    if drift["envs_at_arm_joint_limit"] > 0:
        print(
            f"[NOTE] B 组有 {drift['envs_at_arm_joint_limit']}/{drift['num_envs']} 个 env 的臂关节顶到限位，"
            f"这部分运动的离散度会被限位截断；想纯化结论可调小 --torque_amp 或 --steps 后再跑。"
        )

    # ---- 摩擦是否真的进了求解器 ----
    fp = None
    if not args_cli.no_friction_probe:
        fp = friction_probe(
            sim, scene, robot, dt, args_cli.friction_probe_steps, args_cli.friction_probe_tau, args_cli.friction_probe_mu
        )
        t0, t1 = fp["trials"]["no_friction"], fp["trials"]["static_friction"]
        print(
            f"[FRICTION] {fp['joint']} 上施加恒定 {fp['applied_torque_nm']:g} N·m，跑 {args_cli.friction_probe_steps} 步"
            f"（所有臂关节 damping 置 0）："
        )
        print(f"[FRICTION]   静摩擦 0 N·m     ：|Δq| mean = {t0['dq_mean_rad']:.4e} rad（max {t0['dq_max_rad']:.4e}）")
        print(
            f"[FRICTION]   静摩擦 {args_cli.friction_probe_mu:g} N·m   ：|Δq| mean = {t1['dq_mean_rad']:.4e} rad"
            f"（max {t1['dq_max_rad']:.4e}）"
        )
        print(
            f"[FRICTION] 位移比（无摩擦/有摩擦）= {fp['ratio_no_friction_over_friction']:.4g} -> "
            f"{'静摩擦确实进了求解器' if fp['friction_effective'] else '静摩擦未生效（报告里必须写明！）'}"
        )

    # ---- 汇总 + 落盘 ----
    result = {
        "config": {
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "dt": dt,
            "gravity": bool(args_cli.gravity),
            "torque_amp": args_cli.torque_amp,
            "freq": args_cli.freq,
            "seed": args_cli.seed,
            "randomization_enabled": bool(rng.enabled),
            "payload_mass_range_kg": list(args_cli.payload_mass),
            "payload_body": args_cli.payload_body,
            "payload_com_m": list(args_cli.payload_com),
            "payload_mode": args_cli.payload_mode,
            "link_mass_scale_range": list(args_cli.link_mass_scale),
            "link_mass_scope": args_cli.link_mass_scope,
            "joint_damping_range_nms_per_rad": list(args_cli.joint_damping),
            "joint_friction_range_nm": list(args_cli.joint_friction),
            "joint_expr": args_cli.joint_expr,
            "com_write_path": rng.com_path,
            "dr_apply_wall_s": inject_wall,
        },
        "baseline": base,
        "randomized": drift,
        "dr_sampled": sm,
        "dr_readback": rb,
        "friction_probe": fp,
        "verdict": {
            "mode": "randomized" if rng.enabled else "control_no_randomization",
            "verdict_text": verdict_txt,
            "same_command": same_cmd,
            "passed": passed,
            "threshold": thr,
            "baseline_end_effector_std_rad": base_ee,
            "randomized_end_effector_std_rad": dr_ee,
            "baseline_hand_pos_std_m": base_hp,
            "randomized_hand_pos_std_m": dr_hp,
            "divergence_ratio_end_effector": dr_ee / max(base_ee, 1e-12),
            "divergence_ratio_hand_pos": dr_hp / max(base_hp, 1e-12),
            "arm_joint_limit_envs_baseline": base["envs_at_arm_joint_limit"],
            "arm_joint_limit_envs_randomized": drift["envs_at_arm_joint_limit"],
        },
    }
    if args_cli.out_json:
        path = os.path.abspath(args_cli.out_json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"[OUT] 结果已写入 {path}")

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    free_after, _ = torch.cuda.mem_get_info()
    print(
        f"[OUT] 显存：torch peak allocated {peak_alloc:.3f} GiB | "
        f"跑完剩余 {free_after / 1024**3:.2f} GiB（起始 {free_start / 1024**3:.2f}）"
    )
    print("=" * 78)
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()
