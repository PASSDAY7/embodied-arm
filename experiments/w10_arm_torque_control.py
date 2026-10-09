#!/usr/bin/env python
"""W10 任务三：64 路 Franka 并联 + 零拷贝张量力矩控制回路（Isaac Lab 3.0 / Isaac Sim 6.1）。

这个脚本要回答三个问题
----------------------
1. **能不能跑起来**：在一张 11GB 的 RTX 2080 Ti 上同时仿真 64 台 Franka Panda；
2. **快不快**：500 步纯力矩控制回路的 SPS（steps per second），以及其中“写力矩”这一步
   有没有把数据搬回 CPU（如果有 ``.cpu()`` / ``.item()`` / numpy 之类的往返，SPS 会被
   PCIe 往返和同步拖垮）；
3. **通路对不对**：命令力矩是否真的原样落进了仿真（``applied_torque`` 遥测 == ``τ_cmd``）。

因此本脚本刻意把“张量通路”做成 **GPU 原地运算 + warp kernel 写命令缓冲**：

    τ = clamp(A, ±effort_limit) ⊙ sin(φ_j + 2π f t)          # 全程在显存里算
    robot.actuators.target_command.set_effort_index(value=τ)   # warp kernel 直接写命令缓冲
    scene.write_data_to_sim()                                  # PhysX 取走命令
    sim.step(render=False)                                     # 物理步进
    scene.update(dt)                                           # 刷新 torch 侧状态视图

与 Isaac Lab 3.0 的 API 差异（旧教程里的写法在本版本已废弃）
------------------------------------------------------------
* ``InteractiveScene.step()`` 不存在 —— 步进要靠 ``scene.write_data_to_sim()`` + ``sim.step()``
  + ``scene.update(dt)`` 三连；
* ``robot.set_joint_effort_target*()`` 已废弃（3.0 会发 DeprecationWarning），新接口是
  ``robot.actuators.target_command.set_effort_index(value=...)``；
* ``--headless`` 命令行参数在 3.0 的 ``AppLauncher`` 里被删了，改用环境变量 ``HEADLESS=1``
  （见 ``isaaclab/app/app_launcher.py:_resolve_headless_settings``）；
* 状态量从“裸 tensor 属性”变成了 ``ProxyArray``：``robot.data.joint_pos.torch`` 是
  ``warp.to_torch`` 的**零拷贝视图**，``.warp`` 才是原始 warp 数组。

用法::
    # 推荐用仓库里的环境固化脚本（负责激活 py312、挂 PYTHONPATH、设 HEADLESS/EULA）
    bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py --num_envs 64

    # 想自己拼环境的话：
    cd /data/zzh/IsaacLab && export PYTHONPATH=$(ls -d source/*/ | tr '\n' ':')
    HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python -u \\
        /data/zzh/embodied-arm/experiments/w10_arm_torque_control.py

    # 额外验证“只 reset 一部分 env 时其它 env 是否被牵连”
    bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py --probe_partial_reset

注意：Franka 的 USD 资产（``Isaac/IsaacLab/Robots/FrankaEmika/Legacy/panda_instanceable.usd``）
首次运行会从 Omniverse S3 下载，需要联网。
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
        "          bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py --num_envs 64\n\n"
        "      或者自己拼环境（等价做法）：\n"
        "          export PYTHONPATH=$(ls -d /data/zzh/IsaacLab/source/*/ | tr '\\n' ':')\n"
        "          HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES python -u "
        f"{os.path.abspath(__file__)} --num_envs 64\n"
    ) from exc

##
# 命令行参数（AppLauncher 必须在所有重量级 import 之前启动 Kit）
##
parser = argparse.ArgumentParser(description="64 路 Franka 零拷贝张量力矩控制回路（Isaac Lab 3.0）。")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数。")
parser.add_argument("--steps", type=int, default=500, help="计时步数（不含 warmup）。")
parser.add_argument("--warmup", type=int, default=20, help="预热步数（warp kernel JIT + PhysX 稳定）。")
parser.add_argument("--blocks", type=int, default=5, help="把计时切成几段分别统计 SPS，用来看抖动。")
parser.add_argument("--host_bench", type=int, default=500, help="额外测“只算力矩+写命令缓冲”的主机侧开销的迭代次数。")
parser.add_argument("--torque_amp", type=float, default=10.0, help="力矩幅值 A [N·m]（会被关节力矩上限 clamp）。")
parser.add_argument("--freq", type=float, default=1.0, help="正弦频率 f [Hz]。")
parser.add_argument("--dt", type=float, default=1.0 / 100.0, help="物理步长 [s]，默认 100 Hz。")
parser.add_argument("--env_spacing", type=float, default=2.5, help="环境间距 [m]。")
parser.add_argument("--gravity", action="store_true", help="打开重力（默认关闭，见报告里的说明）。")
parser.add_argument("--probe_partial_reset", action="store_true", help="额外做一次“部分 env reset”验证。")
parser.add_argument("--ablate", action="store_true", help="额外做一次分阶段开销归因（会改变物理状态，只做量级参考）。")
parser.add_argument("--ablate_steps", type=int, default=200, help="分阶段归因每档的迭代次数。")
parser.add_argument("--out_json", type=str, default="", help="把结果另存为 JSON 的路径。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##
# Kit 起来之后再导入仿真相关的库
##
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import clone, configclass  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402

def make_franka_torque_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot", gravity: bool = False) -> ArticulationCfg:
    """把 Franka Panda 改成“纯力矩”配置。

    为什么必须改：``FRANKA_PANDA_CFG`` 用的是隐式驱动器（``ImplicitActuatorCfg``，stiffness=80、
    damping=4）。3.0 的隐式驱动器内核算的是

        τ = k·(q* − q) + d·(q̇* − q̇) + τ_ff      （见 actuator_kernels.compute_implicit_actuator_batch）

    也就是说 ``τ_ff`` 只是**前馈项**。若不清零 k/d，我们“写 10 N·m”实际会得到 10 N·m 加上一个
    未知的 PD 项。这里把 k=d=0（保留 armature 转子惯量，数值更稳），于是

        τ_applied == τ_cmd   （仅受 joint_effort_limit 限幅）

    重力默认关闭：本实验只关心“张量通路 + 吞吐”，不做重力补偿；开着重力再只给正弦力矩，
    手臂会砸到关节限位、变成接触求解器压力测试，SPS 反而不可比。需要时可加 ``--gravity``。
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


class ZeroCopySineTorqueDriver:
    """零拷贝正弦力矩源：所有运算都在预分配的显存里原地完成。

    * 初始化时一次性算好 ``amp``（按关节力矩上限 clamp）与 ``phase``（逐关节相位差，避免所有
      关节同相位摆动）；
    * :meth:`compute` 只做 4 个原地 GPU op：``copy_`` / ``add_`` / ``sin(out=)`` / ``mul_``，
      **不分配显存、不与 CPU 交换任何数据**；
    * :meth:`write_to` 走 ``set_effort_index``：Isaac Lab 用 warp kernel 把 ``value`` 里的数值写进
      命令缓冲（``ActuatorTargetCommand.effort``），全程在显存里，没有 host 往返。

    之所以能确认没有隐性拷贝：``robot.actuators.target_command.effort.torch`` 是
    ``warp.to_torch`` 出来的**视图**，其 ``data_ptr()`` 与 ``.warp.ptr`` 相同（脚本会打印校验行）。
    """

    def __init__(self, robot: Articulation, amp: float, freq: float, device: str):
        # joint_effort_limits 是 ProxyArray -> .torch 是零拷贝视图，形状 (num_envs, num_joints)
        limits = robot.data.joint_effort_limits.torch.to(torch.float32)
        self.num_envs, self.num_joints = limits.shape
        # 幅值既受命令行 A 限制，也不许超过物理关节力矩上限（否则会被内核 clamp，波形失真）
        self.amp = torch.minimum(torch.full((1, self.num_joints), float(amp), device=device), limits[:1])
        # 逐关节相位，让 7 个关节错开摆动（对 SPS 无影响，只是让曲线更好看）
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


def _time_loop(n_steps: int, fn) -> float:
    """跑 ``n_steps`` 次 ``fn(i)``，返回平均墙钟 [ms/次]（前后各同步一次）。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_steps):
        fn(i)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_steps * 1e3


def ablate_phases(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    driver: ZeroCopySineTorqueDriver,
    dt: float,
    n_steps: int,
) -> dict:
    """逐段剥掉回路里的环节，用差值给每段开销定责。

    四档（每档都先热身一次，再同步计时 ``n_steps`` 次）::

        full           compute + write + write_data_to_sim + sim.step + scene.update
        no_scene_update 去掉 scene.update(dt)
        no_sim_step     再去掉 sim.step(render=False)
        host_only       只剩 compute + write（纯张量通路，完全不碰 PhysX/PhysX 之外）

    于是 ``cost_update = full - no_scene_update`` 是“把状态刷回 torch 视图”的代价，
    ``cost_step = no_scene_update - no_sim_step`` 是物理步进的代价，
    ``cost_write = no_sim_step - host_only`` 是 ``scene.write_data_to_sim()``（含 actuator 内核写
    actuation force）的代价，``cost_host`` 是张量通路本身的代价。

    .. caution::
        剥掉 ``sim.step`` 之后物理世界不再推进，四档测的**不是同一个物理状态**。这里只用来做
        量级归因（谁占大头），不要拿它当严格性能模型。
    """
    t_base = 200.0

    def full(i: int) -> None:
        step_once(sim, scene, robot, driver, t_base + i * dt, dt)

    def no_scene_update(i: int) -> None:
        driver.compute(t_base + i * dt)
        driver.write_to(robot)
        scene.write_data_to_sim()
        sim.step(render=False)

    def no_sim_step(i: int) -> None:
        driver.compute(t_base + i * dt)
        driver.write_to(robot)
        scene.write_data_to_sim()

    def host_only(i: int) -> None:
        driver.compute(t_base + i * dt)
        driver.write_to(robot)

    out: dict[str, float] = {}
    for name, fn in (
        ("full", full),
        ("no_scene_update", no_scene_update),
        ("no_sim_step", no_sim_step),
        ("host_only", host_only),
    ):
        fn(0)  # 热身：把该路径上的分支/缓存都过一遍
        out[name] = _time_loop(n_steps, fn)
    out["cost_scene_update"] = out["full"] - out["no_scene_update"]
    out["cost_sim_step"] = out["no_scene_update"] - out["no_sim_step"]
    out["cost_write_data_to_sim"] = out["no_sim_step"] - out["host_only"]
    out["cost_tensor_path"] = out["host_only"]
    out["steps_per_variant"] = n_steps
    return out


def self_check(
    sim: SimulationContext, scene: InteractiveScene, robot: Articulation, driver: ZeroCopySineTorqueDriver, dt: float
) -> dict:
    """跑一步并检查“命令力矩是否原样落到仿真里”。"""
    driver.compute(dt)
    driver.write_to(robot)
    cmd = robot.actuators.target_command.effort
    written = (cmd.torch - driver.tau).abs().max().item()  # 命令缓冲里的值 vs 我们算的 τ

    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    applied = robot.data.applied_torque.torch
    telemetry = robot.data.computed_torque.torch
    return {
        "cmd_buffer_err": written,
        "applied_torque_err": (applied - driver.tau).abs().max().item(),
        "telemetry_err": (telemetry - driver.tau).abs().max().item(),
        "cmd_view_ptr": cmd.torch.data_ptr(),
        "cmd_warp_ptr": cmd.warp.ptr,
        "view_is_alias": cmd.torch.data_ptr() == cmd.warp.ptr,
        "tau_is_cuda": bool(driver.tau.is_cuda),
        "tau_dtype": str(driver.tau.dtype),
    }


def probe_partial_reset(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    driver: ZeroCopySineTorqueDriver,
    dt: float,
    reset_env_ids=(0, 1, 2),
) -> dict:
    """验证：只写 / 只 reset 一部分 env 时，**其它 env 一个 bit 都没被改**。

    关键设计：两次快照之间 **不推进物理**。因为 ``write_joint_position_to_sim_index`` 和
    ``scene.reset`` 都是“直接改缓冲/直接调 PhysX tensor API”，不依赖下一次 ``sim.step``。这样
    “没被点名的 env 变了多少”就是纯粹的越界污染指标（只要不是 0 就是有问题），不会被
    “所有 env 都在同一步里被力矩推动”这种物理运动混淆。

    三个快照点：

    1. ``pos0``/``cmd0``：先跑几步力矩控制，让所有 env 的状态都离开默认位形；
    2. 用 index 接口把 ``reset_env_ids`` 的关节位置写成 ``default + 0.5 rad`` → ``pos1``：
       期望被点名的 env 跳 ~0.5 rad，其它 env **严格等于 0**；
    3. ``scene.reset(env_ids)``（即 ``DirectRLEnv._reset_idx`` 内部走的那条路，
       ``InteractiveScene.reset -> Articulation.reset -> ActuatorCollection.reset``）→
       读命令缓冲 ``cmd1`` 与关节位置 ``pos2``：期望被点名 env 的命令被清、其它 env
       **严格等于 0**；同时关节位置应完全不变（``Articulation.reset`` 只重置 actuator 与
       wrench composer，不碰关节位姿）。
    """
    n = scene.num_envs
    ids = torch.tensor(sorted(reset_env_ids), device=robot.device, dtype=torch.int32)
    mask = torch.zeros(n, dtype=torch.bool, device=robot.device)
    mask[ids.long()] = True

    # 让所有 env 都处于“被力矩推着动”的状态，且命令缓冲里都是非零值
    for i in range(3):
        step_once(sim, scene, robot, driver, 100.0 + (i + 1) * dt, dt)

    pos0 = robot.data.joint_pos.torch.clone()
    cmd0 = robot.actuators.target_command.effort.torch.clone()

    # ---- (2) 只写这几行关节位置（index 接口 -> warp kernel -> PhysX set_dof_positions(indices=...)）----
    target_pos = robot.data.default_joint_pos.torch[: len(ids)].clone() + 0.5
    robot.write_joint_position_to_sim_index(position=target_pos, env_ids=ids)
    pos1 = robot.data.joint_pos.torch.clone()
    d_write = (pos1 - pos0).abs().amax(dim=-1)  # (num_envs,)

    # ---- (3) 只重置这几行的 actuator / wrench 状态 ----
    scene.reset(ids)
    cmd1 = robot.actuators.target_command.effort.torch.clone()
    pos2 = robot.data.joint_pos.torch.clone()
    d_cmd = (cmd1 - cmd0).abs().amax(dim=-1)
    d_pos = (pos2 - pos1).abs().amax(dim=-1)

    def _stat(x: torch.Tensor, sel: torch.Tensor) -> tuple[float, float]:
        return x[sel].max().item(), x[sel].min().item()

    write_max_ids, _ = _stat(d_write, mask)
    write_max_others, _ = _stat(d_write, ~mask)
    cmd_max_ids, cmd_min_ids = _stat(d_cmd, mask)
    cmd_max_others, _ = _stat(d_cmd, ~mask)
    pos_max_ids, _ = _stat(d_pos, mask)
    pos_max_others, _ = _stat(d_pos, ~mask)
    return {
        "reset_env_ids": [int(i) for i in ids],
        "num_envs": n,
        # (2) 关节位置 index 写入：被点名 env 跳变，其余 env 必须严格 0
        "write_index_delta_reset_envs_max": write_max_ids,
        "write_index_delta_other_envs_max": write_max_others,
        "write_index_isolated": bool(write_max_others == 0.0),
        # (3) scene.reset(env_ids)：命令缓冲只清被点名的 env，其余严格 0
        "scene_reset_cmd_delta_reset_envs_max": cmd_max_ids,
        "scene_reset_cmd_delta_reset_envs_min": cmd_min_ids,
        "scene_reset_cmd_delta_other_envs_max": cmd_max_others,
        "scene_reset_cmd_isolated": bool(cmd_max_others == 0.0),
        # scene.reset(env_ids) 不碰关节状态
        "scene_reset_joint_pos_delta_reset_envs_max": pos_max_ids,
        "scene_reset_joint_pos_delta_other_envs_max": pos_max_others,
        "scene_reset_touches_joint_state": bool(max(pos_max_ids, pos_max_others) > 0.0),
    }


def main() -> dict:
    device = args_cli.device or "cuda:0"
    torch.cuda.reset_peak_memory_stats()
    import warp  # 只为了打印版本

    print("=" * 78)
    print(f"[ENV] torch {torch.__version__} | warp {warp.__version__} | isaacsim 6.1.0")
    print(f"[ENV] device = {device} | {torch.cuda.get_device_name(0)}")
    free_start, total = torch.cuda.mem_get_info()
    print(f"[ENV] GPU memory total = {total / 1024**3:.2f} GiB, free = {free_start / 1024**3:.2f} GiB")

    # ---- 仿真上下文 + 场景 ----
    sim = SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=device))
    scene = InteractiveScene(FrankaTorqueSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing))
    sim.reset()
    robot: Articulation = scene["robot"]
    print(f"[SCENE] num_envs={scene.num_envs} num_joints={robot.num_joints}")
    print(f"[SCENE] joint names : {robot.joint_names}")
    print(f"[SCENE] effort limits: {robot.data.joint_effort_limits.torch[0].tolist()}")

    driver = ZeroCopySineTorqueDriver(robot, args_cli.torque_amp, args_cli.freq, device)
    dt = sim.get_physics_dt()
    print(f"[DRIVER] τ = A ⊙ sin(φ + 2πft) | A = {driver.amp[0].tolist()} | f = {args_cli.freq} Hz | dt = {dt:.5f} s")

    # ---- 正确性自检 ----
    chk = self_check(sim, scene, robot, driver, dt)
    print(f"[CHECK] 命令缓冲写入误差 max|cmd − τ| = {chk['cmd_buffer_err']:.3e}")
    print(f"[CHECK] applied_torque 误差 max|τ_applied − τ_cmd| = {chk['applied_torque_err']:.3e} N·m")
    print(f"[CHECK] computed_torque 遥测误差 max|τ_telemetry − τ_cmd| = {chk['telemetry_err']:.3e} N·m")
    print(
        f"[CHECK] 零拷贝：effort.torch.data_ptr={chk['cmd_view_ptr']} vs effort.warp.ptr={chk['cmd_warp_ptr']}"
        f" -> {'同一块显存（视图别名）' if chk['view_is_alias'] else '不是同一块显存！'}"
    )

    reset_probe = None
    if args_cli.probe_partial_reset:
        reset_probe = probe_partial_reset(sim, scene, robot, driver, dt)
        for k, v in reset_probe.items():
            print(f"[RESET-PROBE] {k} = {v}")
        print(
            f"[RESET-PROBE] 结论：index 写入隔离={reset_probe['write_index_isolated']}，"
            f"scene.reset 命令隔离={reset_probe['scene_reset_cmd_isolated']}，"
            f"scene.reset 是否动关节状态={reset_probe['scene_reset_touches_joint_state']}"
        )

    # ---- 预热（warp kernel JIT + PhysX 稳定，不计入统计）----
    for i in range(args_cli.warmup):
        step_once(sim, scene, robot, driver, (i + 1) * dt, dt)
    torch.cuda.synchronize()

    # ---- 计时：切成 blocks 段，每段之间同步一次，看 SPS 抖动 ----
    steps_per_block = max(1, args_cli.steps // args_cli.blocks)
    block_sps: list[float] = []
    done = 0
    wall_total = 0.0
    for _ in range(args_cli.blocks):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps_per_block):
            done += 1
            step_once(sim, scene, robot, driver, done * dt, dt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        wall_total += elapsed
        block_sps.append(steps_per_block / elapsed)

    sps = done / wall_total
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    free_after, _ = torch.cuda.mem_get_info()
    vram_used = (free_start - free_after) / 1024**3  # 进程实际吃掉的显存（含 PhysX/warp 的非 torch 分配）

    # ---- 纯主机侧开销：只跑“算力矩 + 写命令缓冲”，完全不碰 PhysX ----
    # 目的是给“每步有没有 host 往返”一个量级参考：如果这里只有几十微秒，而整步是几毫秒，
    # 说明瓶颈在 PhysX 步进/数据刷新，不在张量通路。
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for k in range(args_cli.host_bench):
        driver.compute((done + k) * dt)
        driver.write_to(robot)
    torch.cuda.synchronize()
    host_bench_us = (time.perf_counter() - t0) / args_cli.host_bench * 1e6

    ablation = ablate_phases(sim, scene, robot, driver, dt, args_cli.ablate_steps) if args_cli.ablate else None

    print("-" * 78)
    print(f"[RESULT] num_envs = {args_cli.num_envs}, steps = {done}, 累计物理步 = {done * args_cli.num_envs}")
    print(f"[RESULT] 平均 SPS = {sps:.1f} steps/s  ( = {sps * args_cli.num_envs:,.0f} env-steps/s )")
    print(f"[RESULT] 分段 SPS = {[round(v, 1) for v in block_sps]} (min {min(block_sps):.1f} / max {max(block_sps):.1f})")
    print(f"[RESULT] 单步墙钟 = {1000.0 / sps:.2f} ms  （其中主机侧算力矩+写命令缓冲 = {host_bench_us:.1f} µs/步）")
    print(
        f"[RESULT] 显存：进程占用 {vram_used:.2f} GiB | torch peak allocated {peak_alloc:.3f} GiB / "
        f"reserved {peak_reserved:.3f} GiB | 跑完剩余 {free_after / 1024**3:.2f} GiB（起始 {free_start / 1024**3:.2f}）"
    )
    print(f"[RESULT] τ 样例 (env0, joint1..3) = {driver.tau[0, :3].tolist()}")
    if ablation is not None:
        print(
            f"[ABLATE] ms/次 @ {args_cli.num_envs} envs（{ablation['steps_per_variant']} 次/档）："
            f"full {ablation['full']:.3f} | 去掉 scene.update {ablation['no_scene_update']:.3f} | "
            f"再去掉 sim.step {ablation['no_sim_step']:.3f} | 只剩张量通路 {ablation['host_only']:.3f}"
        )
        print(
            f"[ABLATE] 归因 [ms/步]：scene.update {ablation['cost_scene_update']:.3f} + "
            f"sim.step {ablation['cost_sim_step']:.3f} + write_data_to_sim {ablation['cost_write_data_to_sim']:.3f} + "
            f"张量通路 {ablation['cost_tensor_path']:.3f}"
        )
    print("=" * 78)

    result = {
        "num_envs": args_cli.num_envs,
        "steps": done,
        "blocks": args_cli.blocks,
        "dt": dt,
        "torque_amp": args_cli.torque_amp,
        "freq": args_cli.freq,
        "gravity_on": bool(args_cli.gravity),
        "sps_mean": sps,
        "sps_blocks": block_sps,
        "sps_min": min(block_sps),
        "sps_max": max(block_sps),
        "ms_per_step": 1000.0 / sps,
        "env_steps_per_s": sps * args_cli.num_envs,
        "host_bench_us_per_step": host_bench_us,
        "vram_process_gib": vram_used,
        "torch_peak_allocated_gib": peak_alloc,
        "torch_peak_reserved_gib": peak_reserved,
        "free_vram_after_gib": free_after / 1024**3,
        "gpu": torch.cuda.get_device_name(0),
        "self_check": chk,
        "partial_reset_probe": reset_probe,
        "ablation_ms_per_step": ablation,
    }
    if args_cli.out_json:
        with open(args_cli.out_json, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 结果已写入 {args_cli.out_json}")
    return result


if __name__ == "__main__":
    main()
    simulation_app.close()
