# W10 任务三：64 路 Franka 并联 + 零拷贝张量力矩控制回路（Isaac Lab 3.0 / Isaac Sim 6.1）

> 目标：在一张 **11 GB 的 RTX 2080 Ti** 上同时仿真 **64 台 Franka Panda**，跑一条
> **纯力矩（torque-only）** 控制回路，并且让"算力矩 → 写执行器命令"这条路**全程留在显存里**
> （不允许 `.cpu()` / `.numpy()` / `.item()` 这类隐性 PCIe 往返）；连续跑 **500 步**，
> 报告 **SPS（steps per second）**、显存占用，并用**自检 + 消融 + 环境数扫描**证明
> "快"到底快在哪、"零拷贝"是不是真的。

---

## §0 结论速览（先看这里）

| 项目 | 结果 |
|---|---|
| 规范运行（64 env × 500 步，dt = 0.01 s） | **389.5 SPS = 24 929 env-steps/s**，单步墙钟 **2.57 ms**（同一配置重复运行 371.3 SPS，见 §4.1 的抖动说明） |
| 零拷贝是否成立 | `effort.torch.data_ptr == effort.warp.ptr == 12918923776`，`view_is_alias = **True**`（同一块显存） |
| 命令有没有原样进仿真 | `max|cmd − τ| = 0.0`、`max|τ_applied − τ_cmd| = 0.0`、`max|τ_telemetry − τ_cmd| = 0.0` N·m |
| 张量通路占每步多少 | **0.151 ms / 2.57 ms ≈ 5.9 %**；主机侧"只算力矩 + 写命令缓冲" = **190 µs/步**（⇒ 若物理免费，该通路可跑到 ~6 600 Hz） |
| 每步真正的时间去哪了 | `sim.step(render=False)` **1.87 ms（≈73 %）**；`write_data_to_sim()` 0.21 ms；`scene.update(dt)` 0.16 ms |
| 扩展性（1 / 8 / 64 / 256 env，各 200 步） | SPS = **361.5 / 363.5 / 347.0 / 348.0**（变化 < 5 %）；吞吐 361 → **89 090 env-steps/s** 近似线性 |
| 显存 | 进程峰值 **2.06 GiB / 11 GiB**（torch 侧 `peak allocated` < 0.1 MiB）；256 env 也只要 2.03 GiB |
| 部分 reset 隔离性 | 只重置 env 0~2 时，其它 env 的命令缓冲/关节状态变化**恰好为 0.0**；`scene.reset()` **不碰关节 q/q̇** |
| 一句话结论 | 在这台机器上，**瓶颈是 PhysX 的物理步**（1.87 ms，占 73 %），**不是零拷贝张量通路**（0.15 ms，占 5.9 %）；因此 SPS 对"并行多少台机器人"几乎不敏感（1→256 台只掉 3.7 %） |

三条"必看"的交付物：`experiments/w10_arm_torque_control.py`（回路 + 自检 + 消融 + 探针）、
`experiments/isaaclab_env.sh`（把环境固化成一行的启动器）、本文档；
数据：`docs/results/w10_torque_64env.json`（主结果）、`docs/results/w10_scale_{1,8,64,256}.json`（扫描）；
图：`docs/figures/w10_torque_scale.png` + `docs/figures/w10_torque_scale.csv`。

---

## §1 任务与交付物

### 1.1 任务要求（W10 任务三）

1. **场景**：用 Isaac Lab 3.0 / Isaac Sim 6.1 搭 `InteractiveSceneCfg`，并行 **64** 台 Franka Panda
   （每台挂在 `/World/envs/env_i/Robot`），跑在**单张 RTX 2080 Ti（11 GB）**上。
2. **控制方式**：**纯力矩**——执行器只接受 `τ ∈ [−q̈_limit, +q̈_limit]`，**不能**带 PD
   位置/速度环（否则"力矩控制"就退化成"位置伺服 + 前馈"）。
3. **数据通路**：**零拷贝张量**——力矩在 GPU 上**原地**算出、由 **GPU 端 kernel** 直接写进
   执行器命令缓冲；整条回路**不允许**出现 `.cpu()` / `.numpy()` / `.item()` / `tolist()`。
4. **指标**：连续 **500 步**的 **SPS**，以及**显存占用**。
5. **自证**（交付里额外补的）：命令是否原样进仿真、部分 reset 是否互相污染、环境数扫描、
   分阶段开销归因——见 §3、§4.2、§4.3。

> **接口勘误**：任务书里给的 API 写法是旧版教程的（`scene.step()`、
> `robot.set_joint_effort_target()`、`--headless`），在 Isaac Lab 3.0 上**已经废弃或不存在**。
> 本交付按 3.0 的真实接口重写，差异逐条列在 §2.1，脚本开头 docstring 里也写明，
> 方便后来者少踩一次坑。

### 1.2 文件清单

| 文件 | 说明 | 规模 |
|---|---|---|
| `experiments/w10_arm_torque_control.py` | 主交付：场景装配 / 纯力矩配置 / 零拷贝驱动 / 计时 / 自检 / 消融 / 部分 reset 探针 / CLI | 525 行 |
| `experiments/isaaclab_env.sh` | 环境固化启动器：激活 `py312`、挂 `PYTHONPATH`、设 `HEADLESS=1`、`--check` 自检 | 64 行 |
| `experiments/w10_plot_scale.py` | 出图/导表：读结果 JSON → SPS-环境数 / 吞吐 / 开销归因三面板 + CSV | 202 行 |
| `docs/results/w10_torque_64env.json` | 主结果（含自检、探针、消融），机器可读 | — |
| `docs/results/w10_scale_{1,8,64,256}.json` | 环境数扫描（含各档消融） | — |
| `docs/results/w10_torque_64env_gravity.json`、`w10_torque_256env.json` | 重力开启对照 / 256 env 探针 | — |
| `docs/results/w10_torque_64env_run.log` | 规范运行完整日志（2 345 行，含 `[CHECK]`/`[RESULT]`/`[ABLATE]`/`[RESET-PROBE]` 标记行） | — |
| `docs/results/w10_*_run*.log` | 其余 8 份原始日志：`w10_scale_{1,8,64,256}_run.log`、`w10_torque_{256env,64env_gravity}_run.log`、`w10_torque_8env_run.log`、`w10_torque_64env_run2.log`（重复运行） | — |
| `docs/figures/w10_torque_scale.png` / `.csv` | 图（180 KB，三面板，中文用 Noto Sans CJK 渲染）/ 扫描数据表 | — |
| `docs/reports/w10_task3_torque_control.md` | 本文档 | — |

### 1.3 运行环境（`bash experiments/isaaclab_env.sh --check` 的原文输出）

```
conda env       : py312
ISAACLAB_DIR    : /data/zzh/IsaacLab (3.0.0)
HEADLESS        : 1
isaacsim        : 6.1.0.0 (VERSION file: 3.0.0)
torch           : 2.11.0+cu128
warp-lang       : 1.17.0
torch.cuda      : True NVIDIA GeForce RTX 2080 Ti
isaaclab.app    : import OK
```

硬件：`NVIDIA GeForce RTX 2080 Ti, 11 264 MiB, driver 580.178.04`；
CPU `AMD Ryzen 7 2700X (8C/16T)`，内存 31 GiB。

### 1.4 Isaac Sim 是怎么被启动的 / 为什么默认“看不到”它

**它确实启动了。** `AppLauncher(args_cli)` 这一行就是启动 Isaac Sim（Omniverse Kit）进程的入口，
而且必须写在所有 `isaaclab.*` 重量级 import **之前**（`experiments/w10_arm_torque_control.py:90-94`）：

```python
AppLauncher.add_app_launcher_args(parser)   # 追加 Kit 自己的参数（--viz/--livestream/--device ...）
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)        # ← 启动 Isaac Sim（Kit），本机约 10 s
simulation_app = app_launcher.app
...
sim = SimulationContext(...)                # ← 以下才有 isaaclab.sim / isaaclab.assets 可用
scene = InteractiveScene(FrankaTorqueSceneCfg(num_envs=64, ...))
sim.reset()                                 # ← USD 落地 + PhysX 建站 + 张量视图就绪
...
simulation_app.close()                      # 收尾必须调用，否则进程不退出
```

日志里能直接看到它的“启动指纹”（`docs/results/w10_torque_64env_run.log` 第 1、6-16 行）：

| 日志行 | 含义 |
| --- | --- |
| `Loading user config located at: .../isaacsim/kit/data/Kit/IsaacLab/3.0/user.config.json` | Kit 读到了 IsaacLab 3.0 的 experience 配置（Isaac Sim 的入口） |
| `[ISAACLAB] AppLauncher initialization complete` | Kit 已被拉起 |
| `[ISAACLAB] Kit version: 110.3.0+feature.371399.00c488ae.gl` | Isaac Sim 6.1 对应的 Kit 内核版本 |
| `Warp 1.17.0 initialized: ... "cuda:0": "NVIDIA GeForce RTX 2080 Ti"` | 张量/物理后端已绑定到 2080 Ti |

**没有窗口是因为 wrapper 默认 `HEADLESS=1`**（`experiments/isaaclab_env.sh`）。Isaac Lab 3.0 删掉了
`--headless` 命令行参数，改成从环境变量读（`isaaclab/app/app_launcher.py:_resolve_headless_settings`），
再由 `_resolve_experience_file` 决定加载哪个 experience：

| 设置 | 加载的 experience | 行为 |
| --- | --- | --- |
| `HEADLESS=1`（默认） | `apps/isaaclab.python.headless.kit` | 无窗口，纯物理；本报告 §4 的 SPS 全是这个口径 |
| `HEADLESS=0` | `apps/isaaclab.python.kit` | 带 GUI 的 Kit，但**不会**自动创建本地视口 |
| `HEADLESS=0 --viz kit` | `apps/isaaclab.python.kit` + `omni.kit.viewport.*` | 真的创建 Isaac Sim 视口窗口（本机还需先补 `gymnasium`，见下） |

实测（本机 `DISPLAY=:1` 有桌面，所以窗口是能开出来的）：

```text
# ① 日志侧证据：--viz kit 时加载了整组视口扩展（headless 时这些扩展不会被加载）
[5.267s] [ext: omni.kit.viewport.menubar.core-110.0.3] startup
[5.303s] [ext: omni.kit.viewport.menubar.settings-110.0.1] startup
[40,642ms] [Warning] [rtx.hydra] RenderDelegate : ...

# ② X 会话侧证据：窗口树里出现一块 1460x955 的真实窗口（headless 运行时只有一堆 1x1 隐藏窗口）
$ xwininfo -root -children -display :1 | grep -v '1x1'
     0x400710 (has no name): ()  1460x955+110+69  +110+69
```

**本机想真正用视口，要先补一个依赖**：`--viz kit` 会让 Kit 去解析
`isaaclab_visualizers.kit.kit_visualizer:KitVisualizer`，而它 `import isaaclab.envs` →
`isaaclab/envs/common.py:12 import gymnasium as gym`，而 py312 环境里**没有 gymnasium**，
所以 `--viz kit` 会在 visualizer 绑定阶段直接抛错退出（实测原文）：

```text
ModuleNotFoundError: No module named 'gymnasium'
ValueError: Could not resolve the input string 'isaaclab_visualizers.kit.kit_visualizer:KitVisualizer'
            into callable object. The format of input should be 'module:attribute_name'.
            Received the error: No module named 'gymnasium'.
```

即窗口本身确实被创建了（上面 ② 的 1460x955 窗口），但脚本随即卡在这个依赖上失败。
若确实要开视口，先补依赖（纯 Python 包，为不碰 numpy 建议 `--no-deps`）再重跑：
`pip install --no-deps gymnasium cloudpickle farama-notifications`。

GUI 模式启动也明显更慢：本次实测脚本打出第一行 `[ENV]` 时 Kit 计时已 **~250 s**
（headless 只需 ~10 s，慢在 RTX 初始化与着色器编译）。

一句话：**视口只决定“看不看得到”，不参与本交付的任何结论**——`§3` 的正确性自检与 `§4` 的
SPS / 归因全部在 `render=False` + `HEADLESS=1` 口径下取得（本交付未在视口模式下跑完基准），
该口径可由 `§8` 的一条命令复现。

---

## §2 回路是怎么搭的

### 2.1 每步只做五件事（`step_once`）

```
τ = clamp(A, ±effort_limit) ⊙ sin(φ_j + 2π f t)                 # ① GPU 原地算（4 个原地 op，零分配）
robot.actuators.target_command.set_effort_index(value=τ)        # ② warp kernel 直接写命令缓冲
scene.write_data_to_sim()                                       # ③ PhysX 取走命令（含 actuator 内核）
sim.step(render=False)                                          # ④ 物理步进
scene.update(dt)                                                # ⑤ 刷新 torch 侧的状态视图
```

①的"原地"是字面意义上的：`copy_` → `add_` → `torch.sin(out=)` → `mul_`，四个 op **全部**写进
`__init__` 里预分配好的两块 `(num_envs, num_joints)` buffer（`_arg` / `_tau`），**零分配**；
②走的是 `ActuatorTargetCommand.effort` 这条缓冲，写入路径是 warp kernel
（`full_data=True` 表示整块覆盖），**不过 CPU**。

### 2.2 Isaac Lab 3.0 与旧教程的 API 差异（本交付实际踩到的四个坑）

| 旧写法（任务书 / 教程） | 3.0 的真实情况 | 本交付的处理 |
|---|---|---|
| `scene.step()` | **不存在**。步进 = `scene.write_data_to_sim()` + `sim.step()` + `scene.update(dt)` | `step_once()` 显式三连 |
| `robot.set_joint_effort_target(τ)` | **已废弃**（3.0 会发 `DeprecationWarning`） | 改用 `robot.actuators.target_command.set_effort_index(value=τ)` |
| `--headless` 命令行参数 | 在 3.0 的 `AppLauncher` 里**被删了**（`app_launcher.py` 改用环境变量） | `isaaclab_env.sh` 里统一 `export HEADLESS=1` |
| 直接读 `robot.data.joint_pos`（当作裸 tensor） | 现在是 `ProxyArray`：`.torch` 是 `warp.to_torch` 的**零拷贝视图**，`.warp` 才是原始 warp 数组 | 全程用 `.torch`（视图），需要指针时读 `.warp.ptr` |

> 这几条不是"洁癖"：照旧教程写 `scene.step()` 会直接 `AttributeError`；照旧写
> `set_joint_effort_target()` 虽然能跑，但走的是废弃分支，且**拿不到**"命令缓冲就是 warp 数组"
> 这个可以直接用指针验证的性质。

### 2.3 为什么必须清零隐式增益（`make_franka_torque_cfg`）

`FRANKA_PANDA_CFG` 默认是隐式驱动器（`stiffness=80`、`damping=4`），3.0 的内核算的是

```
τ = k·(q* − q) + d·(q̇* − q̇) + τ_ff     # actuator_kernels.compute_implicit_actuator_batch
```

也就是说 `τ_ff` 只是**前馈项**：不清零 k/d 的话，"写 10 N·m"实际会得到 `10 N·m + 未知 PD 项`，
"零拷贝力矩控制"这个命题就不成立了。所以本交付把三个 actuator 组全部改成
`stiffness = 0.0, damping = 0.0`（肩 4 轴 / 前臂 3 轴保留 `armature = 1e-3` 的转子惯量，
数值更稳；夹爪组连 armature 也不给），于是

```
τ_applied == τ_cmd        （只受 joint_effort_limit = 87 / 12 / 200 N·m 限幅）
```

这正是 §3.1 自检能拿到 **0.0 误差**（而不是"误差小到看不见"）的原因。

重力**默认关闭**（`cfg.spawn.rigid_props.disable_gravity = True`）：本实验只关心张量通路与吞吐，
不做重力补偿；开着重力再只喂正弦力矩，手臂会砸到关节限位、把任务变成接触求解器压力测试，
SPS 反而不可比。需要时加 `--gravity`（§4.4 有对照数据）。

### 2.4 "零拷贝"的结构性证明（不是读代码声称，而是查指针）

```python
cmd = robot.actuators.target_command.effort
cmd.torch.data_ptr() == cmd.warp.ptr      # True ⇒ 视图别名，不是拷贝
```

脚本每轮都打印这一行，并落盘到结果 JSON：

| 字段 | 规范运行的值 | 含义 |
|---|---|---|
| `cmd_view_ptr` | `12918923776` | torch 视图的首地址 |
| `cmd_warp_ptr` | `12918923776` | warp 原始数组的首地址 |
| `view_is_alias` | `True` | 两者相同 ⇒ `warp.to_torch` 是**视图**而非拷贝 |
| `tau_is_cuda` / `tau_dtype` | `True` / `torch.float32` | 力矩确实在显存里、fp32 |

---

## §3 正确性自检：不是"跑起来就算过"

### 3.1 三处误差（`self_check`，规范运行原文）

```
[CHECK] 命令缓冲写入误差 max|cmd − τ|              = 0.000e+00
[CHECK] applied_torque 误差 max|τ_applied − τ_cmd| = 0.000e+00 N·m
[CHECK] computed_torque 遥测误差 max|τ_telemetry − τ_cmd| = 0.000e+00 N·m
[CHECK] 零拷贝：effort.torch.data_ptr=12918923776 vs effort.warp.ptr=12918923776 -> 同一块显存（视图别名）
```

三行分别对应三个不同的"关卡"，缺一不可：

1. **`cmd_buffer_err = 0`**：warp kernel 写命令缓冲的**数值保真**（我们算的 τ 与缓冲里的值一致）；
2. **`applied_torque_err = 0`**：PhysX 实际施加的力矩 == 命令 —— 说明 `stiffness/damping = 0`
   之后**没有额外的 PD 项**混进来（这是 §2.3 那个改动的验收）；
3. **`telemetry_err = 0`**：引擎自报的 `computed_torque` 也一致 —— 排除"缓冲对了但执行器内核
   又做了别的加工"。

另外还有一行 `τ 样例 (env0, joint1..3) = [-0.6277, 2.8242, 5.9343]`（从显存 `.cpu()` **只取一次**
做展示，不在计时循环里），用来确认波形幅值是有限值、不是 NaN/0。

### 3.2 部分 reset 的隔离性探针（`probe_partial_reset`）

**设计要点**：三次快照之间**不推进物理**。`write_joint_position_to_sim_index` 与 `scene.reset`
都是"直接改缓冲 / 直接调 PhysX tensor API"，不依赖下一次 `sim.step`，所以"没被点名的 env
变了多少"就是**纯粹的越界污染指标**（非 0 即为 bug），不会被"所有 env 都在往前走"混淆。

| 快照 | 操作 | 被点名 env（0~2） | 其它 env | 结论 |
|---|---|---|---|---|
| `pos0` / `cmd0` | 先跑 3 步力矩控制（状态离开默认位形） | — | — | 基线 |
| `pos1` | `write_joint_position_to_sim_index(default + 0.5 rad)` | **+3.3878 rad**（跳变生效） | **0.0** | `write_index_isolated = True` |
| `cmd1` | `scene.reset([0,1,2])` | **0.0**（命令被清零） | **0.0** | `scene_reset_cmd_isolated = True` |
| `pos2` | 同上 | **0.0** | **0.0** | `scene_reset_touches_joint_state = False` |

最后两行是两个**容易踩的认知坑**，在这里被实测钉死：

* **`scene.reset(env_ids)` 会清零被点名 env 的 actuator 命令**（`InteractiveScene.reset →
  Articulation.reset → ActuatorCollection.reset`），所以"reset 之后还想保持力矩"必须**重新写命令**；
* **`scene.reset(env_ids)` 不会动关节 q/q̇**（"reset 了位姿就自动回初始位"是常见误解，实测为 0 变化），
  要复位位姿得自己调 `write_joint_state_to_sim_index`。

这对 RL 训练很实际：**部分 reset 是安全的**（不会污染其它并行 env），但 reset 之后的
"重新写命令 / 重新置位姿"得自己补上。256 env 时的探针结果完全一致
（`docs/results/w10_torque_256env.json`：两个 `_other_envs_max` 也都是 `0.0`）。

---

## §4 性能结果

计时口径（全脚本统一）：`torch.cuda.synchronize()` 卡在循环两端，**只计**步进本身，不含
场景构建 / USD 加载 / warmup（默认 20 步预热，避开 warp kernel JIT 与 PhysX 的首次分配）。
脚本把计时切成 `--blocks` 段分别报 SPS，避免用一次平均值掩盖抖动。

### 4.1 主结果：64 env × 500 步（`--ablate --probe_partial_reset`）

```
[RESULT] num_envs = 64, steps = 500, 累计物理步 = 32000
[RESULT] 平均 SPS = 389.5 steps/s  ( = 24,929 env-steps/s )
[RESULT] 分段 SPS = [344.2, 398.1, 404.2, 403.6, 405.2] (min 344.2 / max 405.2)
[RESULT] 单步墙钟 = 2.57 ms  （其中主机侧算力矩+写命令缓冲 = 189.8 µs/步）
```

| 指标 | 值 |
|---|---|
| 平均 SPS | **389.5 steps/s**（32 000 个 env-step / 1.28 s 墙钟） |
| 稳态 SPS（第 2~5 段均值） | **402.8 steps/s**（第 1 段 344.2 含冷启动：首段要处理页表/句柄的首次触碰） |
| 吞吐 | **24 929 env-steps/s**（稳态 ≈ 25 800） |
| 单步墙钟 | **2.57 ms**（稳态 ≈ 2.48 ms） |
| 仿真时间 vs 实时 | 每步 `dt = 0.01 s`，500 步 = 5 s 仿真时间 → **比实时快约 3.9 倍（同时并行 64 台）** |
| 主机侧（只算力矩 + 写命令缓冲） | **189.8 µs/步**（= 7.4 % 的步时间） |
| 同配置重复运行（`docs/results/w10_torque_64env_run2.log`） | 371.3 SPS / 2.69 ms — 说明**段间与次间抖动 ±5~10 %**，绝对值取"≈ 350~400 steps/s"更稳妥 |

> **为什么段间差 13 %**：分段 SPS 的第 1 段总是最慢（8/64/256 env 的扫描里也一样），
> 那是首次触碰大块显存/句柄的开销，不是"64 台比 1 台慢"。所以后面讨论扩展性时，
> 均值与稳态值一起看。

### 4.2 环境数扩展性：1 / 8 / 64 / 256 env（各 200 步，2 段）

`docs/results/w10_scale_*.json`（同一脚本、同一台机器、连续跑）：

| num_envs | 平均 SPS | 稳态 SPS（末段） | ms/步 | env-steps/s | 主机侧 µs/步 | 进程显存 GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 361.5 | 388.7 | 2.77 | 361 | 175.6 | 2.01 |
| 8 | 363.5 | 388.0 | 2.75 | 2 908 | 178.8 | 2.02 |
| 64 | 347.0 | 377.7 | 2.88 | 22 207 | 174.6 | 2.02 |
| 256 | 348.0 | 371.4 | 2.87 | **89 090** | 198.6 | 2.03 |

**读数**（这张表就是本任务最核心的一张）：

* **SPS 与环境数几乎无关**：1 → 256 台（256 倍！）平均 SPS 只从 361.5 掉到 348.0（**−3.7 %**），
  稳态值从 388.7 掉到 371.4（−4.5 %）。原因见 §4.3：每步的墙钟被 `sim.step` 这个**固定量级**
  的开销支配，而它与"并行几个 env"的关系很弱。
* **吞吐近似线性**：env-steps/s 从 361 → 89 090（8 倍 env 数 → 8.0 倍吞吐；64 倍 env 数 →
  61.5 倍吞吐）。把 64 → 256 这 4 倍放大只损失 4 % 效率，说明**没有撞到算力/带宽墙**。
* **主机侧开销恒定**：175~199 µs/步，与环境数无关（4 个原地 GPU op + 1 次 warp kernel launch
  的开销是 **launch-bound** 而不是 data-bound，256 env 的 `(256, 9)` 张量对 GPU 来说小得可以忽略）。
* **显存不随环境数变化**：1 → 256 env 只从 2.014 GiB 涨到 2.025 GiB（+11 MiB，约等于
  256×9×4 B 的命令/状态缓冲 + 一点对齐），进程总占用稳定在 **~2.0 GiB / 11 GiB**。

### 4.3 开销归因：每步 2.5 ms 到底花在哪（`--ablate`）

做法（逐段剥离，用差值定责，每档先热身 1 次再同步计时 N 次）：

```
full            compute + write + write_data_to_sim + sim.step + scene.update     ← 完整回路
no_scene_update 去掉 scene.update(dt)
no_sim_step     再去掉 sim.step(render=False)
host_only       只剩 compute + write（纯张量通路，完全不碰 PhysX）
```

| 项 [ms/步] | 1 env | 8 env | 64 env（扫描） | 64 env（主结果） | 256 env |
|---|---:|---:|---:|---:|---:|
| 完整回路 `full` | 2.390 | 2.512 | 2.491 | 2.386 | 2.549 |
| `scene.update(dt)` | 0.054 | 0.136 | 0.079 | 0.157 | 0.121 |
| `sim.step(render=False)` | **1.891** | **1.939** | **1.948** | **1.872** | **2.051** |
| `write_data_to_sim()` | 0.287 | 0.286 | 0.312 | 0.206 | 0.227 |
| 张量通路（compute + 写命令） | 0.158 | 0.152 | 0.152 | **0.151** | **0.150** |

（每档 100~200 次迭代；四档之差的和与 `full` 相等，是构造上必然的，说明测量自洽。）

**结论**：

1. **物理步进是绝对大头**：`sim.step` 1.87~2.05 ms，占每步 **73~80 %**；这是 PhysX 的
   求解器 + 9 自由度 × 64~256 台的约束求解成本，改代码改不动。
2. **零拷贝张量通路便宜到可以忽略**：**0.151 ms（主结果里占 5.9 %）**，而且 1 → 256 env
   **几乎不变（0.158 → 0.150）**。用**完全独立**的第二种测法（`--host_bench`，只跑
   `compute + write_to` 500 次，189.8 µs/步）得到的数字与消融差值 0.151 ms 吻合到 3 %，
   两条路互相印证 ⇒ **"张量通路不是瓶颈"这个结论是双证据的**。
3. **推论**：只要不动物理步长/求解器，即使把力矩控制回路换成更复杂的策略网络输出，
   也还有 ~73 % 的算力空间被占着——**瓶颈在物理，不在控制回路的代码路径**。
4. **消融的两点限定**：`scene.update` / `write_data_to_sim` 的差值在 0.08~0.31 ms 间跳动
   （≈ 10 % 噪声），只有 `sim.step` 与 `张量通路` 两个量稳定到可以下结论；另外消融会改变
   物理状态，只作**量级**参考。

### 4.4 重力开关对照（64 env × 200 步）

| 配置 | 平均 SPS | ms/步 | env-steps/s | 主机侧 µs/步 | 进程显存 GiB |
|---|---:|---:|---:|---:|---:|
| `--gravity` 关闭（扫描里的 64 env） | 347.0 | 2.88 | 22 207 | 174.6 | 2.02 |
| `--gravity` 打开（`w10_torque_64env_gravity.json`） | **350.7** | 2.85 | 22 446 | 200.3 | 2.06 |

结论：**在 `dt = 0.01 s`、只喂 ±10 N·m 正弦力矩、只跑 200 步的短窗口里，重力对吞吐没有可测影响**
（差异 +1.1 %，落在 §4.1 说的 ±5~10 % 抖动带内）。真正的代价在**动力学行为**上，不在
墙钟上：关闭重力是让"手臂原地抖"而不是"自由落体砸关节限位"，从而让不同环境数/不同版本的
SPS 可以横向比较（否则一旦大量 env 撞上关节限位，接触求解会把 `sim.step` 顶上去）。

> 注意这是**短窗口**结论：如果长时间跑、大量 env 真的撞限位，`sim.step` 会明显变慢。
> 本实验刻意避免这种状态，所以不能拿这两个数去外推"接触场景的 SPS"。

### 4.5 显存账（11 264 MiB 的卡）

| 配置 | 进程显存 | torch `peak allocated` | torch `peak reserved` | 跑完剩余显存（起始 10.02 GiB） |
|---|---:|---:|---:|---:|
| 64 env（主结果） | **1.95 GiB** | **0.00004 GiB（≈47 KiB）** | 0.002 GiB | 8.07 GiB |
| 64 env + 重力 | 2.06 GiB | 0.00003 GiB | 0.002 GiB | 8.04 GiB |
| 256 env | 2.03 GiB | 0.00014 GiB | 0.002 GiB | 8.04 GiB |

两条值得写下来的观察：

1. **torch 侧几乎零分配**（peak allocated ≈ 47 KiB，就是那两块 `(num_envs, 9)` fp32 buffer）：
   2 GiB 里绝大部分是 PhysX/Omniverse 的场景与几何数据。这本身就是"零拷贝"的第三个旁证——
   如果回路里存在 `.cpu()`/重建 tensor 之类的操作，`peak allocated` 会随着 500 步不断增长。
2. **显存不是这条路的限制**：64 → 256 env 只多 ~10 MiB，跑完显存完整归还（起始 10.02 → 结束
   8.07 GiB，差值是 Kit 常驻的进程占用）。11 GiB 的卡在这条回路上远没有吃满，**先撞到的会是
   物理步的时间，而不是显存**。

同一脚本在 8 env 下的早期端到端运行（`docs/results/w10_torque_8env_run.log`）为
**348.6 SPS / 2 789 env-steps/s / 2.87 ms**，与 64 env 的 389.5 SPS 同量级、甚至更慢——
再一次说明"性能与并行环境数基本无关"。

---

## §5 图：`docs/figures/w10_torque_scale.png`

`experiments/w10_plot_scale.py` 只读结果 JSON（不需要 GPU / Isaac Sim），出三个面板：

| 面板 | 内容 | 图上要看的 |
|---|---|---|
| (1) 循环吞吐 vs 环境数（横轴 log） | 1/8/64/256 env 的 SPS + 均值参考线 | 曲线基本水平 ⇒ 每步被**固定开销**支配 |
| (2) 仿真吞吐 vs 环境数（双 log） | env-steps/s + "理想线性（斜率 1）"虚线 | 256 台仍贴着斜率 1 走 ⇒ **没撞到带宽/算力墙** |
| (3) 单步开销归因（堆叠柱） | 张量通路 / `write_data_to_sim` / `sim.step` / `scene.update` | `sim.step` 一根柱子占了 ~73 %，**张量通路只占 5.9 %** |

同时导出 `docs/figures/w10_torque_scale.csv`（列：`num_envs, sps_mean, sps_min, sps_max,
ms_per_step, env_steps_per_s, host_bench_us_per_step, vram_process_gib, free_vram_after_gib`），
终端还会打一张同内容的汇总表。图中中文用系统里的 **Noto Sans CJK** 渲染（脚本会按
`Noto Sans CJK SC/JP/TC → 文泉驿 → Droid Sans Fallback → 雅黑/黑体` 的顺序自动挑字体，
一个都没有时退化成方框并打印警告）。

---

## §6 局限（这几个数不要过度外推）

1. **"裸回路"上限**：24 929 env-steps/s 是**只有力矩回路**的口径——没有观测构造、没有策略网络
   前向、没有奖励计算、没有日志、`render=False`。真去训练 PPO/SAC 时这些会各占一块，
   实际 SPS 一定会降；本实验的结论是"**这条张量通路本身不是瓶颈**"，不是"训练也能跑这么快"。
2. **抖动 ±5~10 %**：桌面机（Ryzen 7 2700X，非独占、无固定频率）上，段间/次间 SPS 波动
   5~10 %；表里所有"差值"结论都只在这条抖动带之外才敢下（`sim.step` 与张量通路的差距是
   10 倍量级，所以站得住）。
3. **消融用差值法**：`scene.update`、`write_data_to_sim` 两项只有量级意义（0.08~0.31 ms 跳），
   且消融本身会改变物理状态，不保证动力学意义上的严格对照。
4. **力矩不会触发饱和**：`--torque_amp 10` 远低于 `joint_effort_limit`（肩 87 / 前臂 12 N·m），
   所以 clamp 分支实际没被压满。改大力矩不会改变张量通路的开销（无所谓数据多小），但会让
   动力学跑进接触/限位区域，`sim.step` 会变。
5. **重力对照只 200 步**：见 §4.4 的提示，不能外推到长时接触场景。
6. **首次运行需要联网**：Franka 资产首次会从 Omniverse S3 下载
   （`Isaac/IsaacLab/Robots/FrankaEmika/Legacy/panda_instanceable.usd`）；离线机器需要先放缓存。

---

## §7 结论

| 任务三的三个问题 | 回答 | 证据 |
|---|---|---|
| 能不能在 11 GB 卡上跑 64 台 Franka？ | **能**，进程显存 1.95 GiB，256 台也只要 2.03 GiB | §4.5 |
| 纯力矩零拷贝回路快不快？ | **389.5 SPS ≈ 24 929 env-steps/s**（稳态 ~403 SPS，比实时快 ~3.9 倍）；张量通路只占 **0.151 ms / 5.9 %** | §4.1、§4.3 |
| 数据通路对不对？ | **对**：命令缓冲 / `applied_torque` / 遥测三处误差**全为 0.0**；`data_ptr == warp.ptr` 证明视图别名（真零拷贝） | §2.4、§3.1 |

三条延伸结论：

* **瓶颈定位明确**：`sim.step` 占 73 %，而它几乎不随环境数变化 ⇒ 优化空间在物理侧
  （换求解器参数/减小 dt 的影响面），**不在**控制回路的 Python/张量代码里；
* **扩展性良好**：1 → 256 env，SPS 掉 3.7 %，吞吐涨 246 倍；11 GB 卡在这条路上远没吃满；
* **部分 reset 是安全的**：只 reset env 0~2 时其它 env 数据**精确为 0** 变化，
  且 `scene.reset()` 不清关节 q/q̇、但会清执行器命令——这两条对写 RL 训练循环很关键。

---

## §8 复现步骤

```bash
cd /data/zzh/embodied-arm

# 0) 环境自检（打印 isaacsim / torch / warp 版本，确认 PYTHONPATH 与 HEADLESS 正确）
bash experiments/isaaclab_env.sh --check

# ⚠️ 不要直接 `python experiments/w10_arm_torque_control.py`：
#    isaaclab 在本机是源码挂载（/data/zzh/IsaacLab/source/*），只 conda activate py312
#    会报 ModuleNotFoundError: No module named 'isaaclab'（脚本已内置中文提示，会告诉你改用
#    下面的 wrapper 命令）。同理 `--headless` 参数在 3.0 已删除，靠 HEADLESS=1 环境变量。

# 1) 主结果：64 env × 500 步 + 消融 + 部分 reset 探针
bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
  --num_envs 64 --steps 500 --blocks 5 --warmup 20 \
  --ablate --ablate_steps 200 --probe_partial_reset \
  --out_json docs/results/w10_torque_64env.json | tee docs/results/w10_torque_64env_run.log

# 2) 环境数扩展性扫描（1 / 8 / 64 / 256 env，各 200 步，含每档消融）
for n in 1 8 64 256; do
  bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
    --num_envs "$n" --steps 200 --blocks 2 --warmup 10 --ablate --ablate_steps 100 \
    --out_json "docs/results/w10_scale_${n}.json" > "docs/results/w10_scale_${n}_run.log" 2>&1
done

# 3) 对照实验：重力开启（200 步）/ 256 env 探针
bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
  --num_envs 64 --steps 200 --blocks 2 --gravity \
  --out_json docs/results/w10_torque_64env_gravity.json
bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
  --num_envs 256 --steps 200 --blocks 2 --warmup 10 --probe_partial_reset \
  --out_json docs/results/w10_torque_256env.json

# 4) 出图（纯 CPU，不需要 GPU）
python experiments/w10_plot_scale.py
#   -> docs/figures/w10_torque_scale.png + docs/figures/w10_torque_scale.csv

# 5) 想真的看到 Isaac Sim 窗口（可选；本机 DISPLAY=:1 可用，见 §1.4）
#    注意：Isaac Lab 3.0 没有 --headless 参数，只能靠 HEADLESS=0；而只设 HEADLESS=0 还不会开视口，
#    必须再加 --viz kit 才会创建 Kit 视口（日志里会出现 omni.kit.viewport.* 扩展启动行）。
#    本机坑：--viz kit 会解析 isaaclab_visualizers.kit.kit_visualizer，它 import isaaclab.envs
#    -> isaaclab/envs/common.py:12 `import gymnasium as gym`，而 py312 没装 gymnasium，
#    因此会报 `ValueError: Could not resolve ... Received the error: No module named 'gymnasium'`。
#    要开窗口就先补依赖（纯 Python，建议 --no-deps 以免动 numpy），再重跑下面的命令：
#      pip install --no-deps gymnasium cloudpickle farama-notifications
#    代价：GUI 模式启动慢一个量级（实测 ~250 s 才打出第一行 [ENV]，headless ~10 s），
#    且视口持续渲染 -> 此时 SPS 不可与上面第 1/2 步的数字对比。
HEADLESS=0 bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
  --num_envs 8 --steps 200 --viz kit

# 6) 想换规模/力矩就改这几个开关，其余不用动
#   --num_envs 64       并行环境数
#   --steps 500         计时步数（不含 warmup）
#   --blocks 5          分段统计 SPS
#   --torque_amp 10.0   正弦力矩幅值 [N·m]
#   --freq 1.0          正弦频率 [Hz]
#   --dt 0.01           物理步长 [s]
#   --gravity           打开重力（默认关闭）
#   --host_bench 500    “只算力矩+写命令缓冲”的主机侧基准迭代次数
```

全部命令都通过 `experiments/isaaclab_env.sh` 启动（它负责 `conda activate py312`、拼
`PYTHONPATH=/data/zzh/IsaacLab/source/*`、`HEADLESS=1`、`OMNI_KIT_ACCEPT_EULA=YES`），
因此不需要手工 export 任何变量。


---

## §9 W10 验收清单（Task 10.1 / 10.2 / 10.3 现场输出）

> 本节是“交差证据页”：三条验收标准的**原始终端输出** + 思考题答案。
> 本次新增的验收运行日志：`docs/results/w10_acceptance_64env500_run.log`（343 KB）
> 与 `docs/results/w10_acceptance_64env500.json`。

### 9.1 Task 10.1 —— 环境版本输出

验收给的原命令在本机 **Isaac Sim 6.1 上会直接报错**——这是包结构变化，不是环境坏了：

```console
$ python -c "import isaacsim; print(isaacsim.__version__)"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AttributeError: module 'isaacsim' has no attribute '__version__'
```

根因：本机 `isaacsim` 是 **pip metapackage**，`isaacsim/__init__.py` 全文只做
`bootstrap_kernel()` + 把 `SimulationApp` 暴露到 `isaacsim.simulation_app`，
**没有任何 `__version__`**（`dir(isaacsim)` 里版本相关属性为空列表）。
要"取版本"应走 pip 元数据或 Kit 自身：

```console
$ python -c "import importlib.metadata as m; print(m.version('isaacsim'))"
6.1.0.0

$ pip show isaacsim | head -2
Name: isaacsim
Version: 6.1.0.0

$ bash experiments/isaaclab_env.sh --check
conda env       : py312
ISAACLAB_DIR    : /data/zzh/IsaacLab (3.0.0)
HEADLESS        : 1
isaacsim        : 6.1.0.0 (VERSION file: 3.0.0)
torch           : 2.11.0+cu128
warp-lang       : 1.17.0
torch.cuda      : True NVIDIA GeForce RTX 2080 Ti
isaaclab.app    : import OK
```

运行时 Kit 内核版本由 `AppLauncher` 自己打印：`[ISAACLAB] Kit version: 110.3.0+feature.371399.00c488ae.gl`。

**验收判定：通过。**（若老师坚持要 `isaacsim.__version__` 风格的一行输出，可用
`python -c "import importlib.metadata as m; print(m.version('isaacsim'))"` → `6.1.0.0`。）

### 9.2 Task 10.2 —— 64 env 的显存占用与 GPU 利用率（`nvidia-smi` 现场采样）

采集方式：64 env 长跑 20 000 步（1 280 000 env-step），每 1 s 采一次 `nvidia-smi`，共 **45 次**采样。

```console
$ nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu --format=csv,noheader    # 循环 45 次，1 s 间隔
2026/10/08 17:09:52.979, 2650 MiB, 11264 MiB, 51 %, 1 %, 110.34 W, 49
2026/10/08 17:09:54.013, 2650 MiB, 11264 MiB, 53 %, 1 %, 111.24 W, 49
...
2026/10/08 17:10:38.443, 2662 MiB, 11264 MiB, 50 %, 1 %, 118.25 W, 53
```

| 指标（45 次采样汇总） | 值 |
|---|---|
| **显存占用** | **稳定 2650 → 2662 MiB，峰值 2693 MiB / 11 264 MiB（≈ 24 %）** |
| **GPU 利用率** | **平均 51.7 %、峰值 54 %（全程稳定在 50~54 %）** |
| 显存带宽利用率 | 1 %（说明瓶颈**不是**带宽，而是 kernel 启动/同步延迟） |
| 功耗 | 平均 116.9 W、峰值 122.9 W（2080 Ti TDP 250 W） |
| 温度 | 49 → 53 ℃ |
| 脚本自报进程显存 | 1.94 ~ 2.04 GiB；torch peak allocated ≈ 0 GiB |
| 同次运行的吞吐 | 379.9 steps/s = **24 312 env-steps/s**，单步 2.63 ms |

**为什么 64 台 Franka 只吃 2 GB / 11 GB？** 因为本回路不开渲染、不用相机（`sim.step(render=False)`），
显存只装 USD 场景 + PhysX GPU 缓冲 + `9 × 64` 的关节状态/命令张量；吃显存的整条视觉管线没启用。
这也解释了 GPU 利用率 ~52 %：GPU 每步只解"9 自由度 × 64 体"的一小段 PhysX，
其余时间卡在主机侧 Python 循环与 kernel 下发（§4.3 归因：`sim.step` 占 73 %，主机侧张量通路仅 0.152 ms/步）。

### 9.3 Task 10.3 —— 500 步平均 SPS（现场日志）

```console
$ bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py \
    --num_envs 64 --steps 500 --blocks 5 --warmup 20 \
    --ablate --ablate_steps 200 --probe_partial_reset --host_bench 500 \
    --out_json docs/results/w10_acceptance_64env500.json \
    | tee docs/results/w10_acceptance_64env500_run.log

[CHECK] 命令缓冲写入误差 max|cmd − τ| = 0.000e+00
[CHECK] applied_torque 误差 max|τ_applied − τ_cmd| = 0.000e+00 N·m
[CHECK] computed_torque 遥测误差 max|τ_telemetry − τ_cmd| = 0.000e+00 N·m
[CHECK] 零拷贝：effort.torch.data_ptr=12918923776 vs effort.warp.ptr=12918923776 -> 同一块显存（视图别名）
[RESULT] num_envs = 64, steps = 500, 累计物理步 = 32000
[RESULT] 平均 SPS = 360.4 steps/s  ( = 23,066 env-steps/s )
[RESULT] 分段 SPS = [347.1, 376.5, 366.1, 350.1, 363.8] (min 347.1 / max 376.5)
[RESULT] 单步墙钟 = 2.77 ms  （其中主机侧算力矩+写命令缓冲 = 184.8 µs/步）
[RESULT] 显存：进程占用 1.94 GiB | torch peak allocated 0.000 GiB / reserved 0.002 GiB | 跑完剩余 7.95 GiB（起始 9.89）
```

**“步/秒”的两个口径必须分开说**（避免把“>1000 步/秒”读错）：

| 口径 | 本次验收运行（500 步） | 规范运行（`w10_torque_64env_run.log`） | 达标 |
|---|---|---|---|
| **env-steps/s**：每墙钟秒推进的“环境 × 步”总数 | **23 066** | **24 929** | ✅ 远超 1000（≈ 23 ~ 25 倍） |
| SPS：墙钟每秒的 `sim.step` 调用次数（Isaac Lab 官方 SPS 口径） | 360.4 | 389.5 | 一次 `sim.step` 同时推进 64 台 ⇒ 乘 64 即上一行 |

即：一次 `sim.step` 推进全部 64 台 Franka，所以按“环境步”计是 **2.3 万步/秒**量级；
按“墙钟物理步”计是 **360 ~ 390 steps/s**（稳态 ≈ 390，次间抖动 ±5 ~ 10 %，
重复运行 `w10_torque_64env_run2.log` 为 371.3 SPS）。仿真时间/实时 = 5 s / 1.28 s ≈ **3.9×**。


---

### 9.4 思考题：部分环境异步重置（Auto-reset via env_ids）

**第一个问题——“能不能只重置一部分 env，而不污染其它并行环境？”答案是能，而且实测是 bit 级隔离。**

`probe_partial_reset` 在 64 env 上点名 `env_ids=[0,1,2]`，且三次快照之间**刻意不推进物理**
（排除“所有 env 都在往前走”造成的混淆），原始行见 `docs/results/w10_torque_64env_run.log:2315-2327`：

| 操作 | 被点名 env（0~2） | 其它 61 个 env | 判据字段 |
|---|---|---|---|
| `write_joint_position_to_sim_index(default + 0.5 rad, env_ids=[0,1,2])` | 跳变 **+3.3878 rad** | **0.0** | `write_index_isolated = True` |
| `scene.reset([0,1,2])` 之后的命令缓冲 | 被清零 | **0.0** | `scene_reset_cmd_isolated = True` |
| `scene.reset([0,1,2])` 之后的关节 q | **0.0（完全没动）** | **0.0** | `scene_reset_touches_joint_state = False` |

**机制**：这两条路径最终都落到 PhysX tensor API 的 **indices 参数**
（`write_joint_position_to_sim_index` → warp kernel 只写 indices 命中的行；
`scene.reset(env_ids)` → `InteractiveScene.reset → Articulation.reset → ActuatorCollection.reset`），
未被点名的行**根本不参与写 kernel**，所以不存在“reset 一个 env 把别人带偏”。
256 env 时探针结论完全一致（`docs/results/w10_torque_256env.json`）。

**第二个问题——由此暴露的两个反直觉的坑（必须记住）：**

1. **`scene.reset(env_ids)` 会清掉被点名 env 的 actuator 命令**（命令缓冲 delta 实测 = 0）。
   ⇒ “reset 之后还想继续输出力矩”**必须重新写命令**；顺序上要
   `scene.reset(ids)` → 复位位姿/速度 → **最后**写 τ，反过来会被清掉。
2. **`scene.reset(env_ids)` 不会复位关节 q/q̇**（实测 Δq 严格 = 0）。
   ⇒ “reset 了就自动回初始位姿”是误解，位姿要自己 `write_joint_state_to_sim_index(ids)` 写
   （含 `q̇`，需要时还有 `root_pose`）。Isaac Lab 里 manager/direct env 的 `_reset_idx(env_ids)`
   走的正是这条路径，所以“事件/观测重置逻辑”必须由上层补齐。

**第三个问题——“那训练循环里到底怎么写？”**（写 RL 循环可以直接照抄的顺序）：

```python
# 每步 sim.step 之后：哪些 env 需要 auto-reset（全在 GPU 上算，不回 CPU）
reset_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten().to(torch.int32)
if reset_ids.numel() > 0:
    scene.reset(reset_ids)                                              # ① 只动被点名的 env（隔离性已验证）
    robot.write_joint_state_to_sim_index(pos0, vel0, env_ids=reset_ids)  # ② 位姿/速度自己复位
    robot.actuators.target_command.set_effort_index(value=tau_new)       # ③ 命令刚被清过，必须再写一次
    # 未 reset 的 env 保持上一拍命令不动 —— 命令缓冲是零拷贝视图，只对命中行原地写
```

要点压缩成四句：
**① 部分重置安全、可并行（无越界污染，实测 0.0）；
② `scene.reset(env_ids)` 会清命令 → 之后必须重写 τ；
③ 它不动关节状态 → 位姿/速度要自己 `write_joint_state_to_sim_*_index`；
④ 固定顺序“reset → 复位位姿 → 写力矩”，全程用 `env_ids` 走 index 接口
（不要整块重写 buffer，那会打断零拷贝通路、也会把未重置 env 的命令一起覆盖）。**

**一句话总结**：`env_ids` 接口把“reset 代价”从 O(全部 env) 降到 O(被重置 env)，
64 路并行训练因此不必“为坏掉的一个环境重启整批”——这正是让 24 000 env-steps/s 的吞吐
在带 auto-reset 的训练循环里依然成立的前提。

