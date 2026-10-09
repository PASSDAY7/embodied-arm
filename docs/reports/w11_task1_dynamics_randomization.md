# W11 Day 2：把"未知负载 + 非线性摩擦 + 质量失配"注入 Isaac Lab 物理层（域随机化）

> 目标：在 Isaac Lab 3.0 / Isaac Sim 6.1 里，把三类**真实世界失配**——
> (1) 末端未知负载（质量 / 惯量 / 质心）、(2) 非线性的关节摩擦与阻尼、(3) 连杆质量标称值不准——
> **显式写进 PhysX 的模型属性**（不是加观测噪声、不是在控制器里加扰动），
> 让 64 个并行环境各自面对**不同的真实动力学**；
> 并用一次运行内的 **A/B 对照**证明"**同力不同态**"：
> 给全部 64 个 env 喂**逐元素完全相同**的正弦力矩，注入前跨 env 关节位置标准差 ≈ 0，
> 注入后显著大于 0。

---

## §0 结论速览（先看这里）

| 项目 | 结果 |
|---|---|
| 规范运行 | 64 env × 200 步 × A/B 两组，`dt = 10 ms`（100 Hz），正弦力矩 `τ = A⊙sin(φ+2πft)`，`A = 10 N·m`、`f = 1 Hz`，`seed = 42` |
| 输入是否真的相同 | **`τ` 跨 env 标准差 = `0.000e+00` N·m**（A 组、B 组都是严格 0；实现上 `τ` 只有一行 `(1, 9)`，`set_effort_index` 广播到 64 个 env） |
| A 组（未注入）| 关节位置跨 env std：**max `2.161e-06` rad**、`panda_joint7` `5.087e-07` rad；末端位置 std `1.167e-06` m；臂关节撞限位 **0/64** |
| B 组（注入后）| 关节位置跨 env std：**max `6.977e-02` rad**、`panda_joint7` `5.056e-03` rad；末端位置 std **`1.605e-02` m**；臂关节撞限位 **0/64** |
| 分化倍率（B ÷ A）| 最大关节 std **×3.23e+04**、末端关节 ×9.94e+03、末端位置 ×1.38e+04（阈值 1e-3 rad / 1e-3 m → 超阈值 70× / 16×） |
| 验收判据 | **PASS**（`docs/results/w11_dr_64env_run.log` 的 `[VERDICT] 同力不同态：PASS`） |
| 对照组（`--no_randomization`）| A/B 两组统计量**逐位相同**（`5.086528744868701e-07` rad 两次完全一致），分化倍率 **×1** → 分化确实来自注入，而不是"复位 + 第二次 rollout"本身 |
| 参数有没有真的进 PhysX | `panda_hand` 质量 `0.6143 / 2.173 / 3.756` kg（标称 `0.6804`）、惯量最小特征值 `4.07e-4 > 0`（正定）、质心位移均值 `0.0156 m`；关节 damping `0.5063 / 1.253 / 1.998` N·m·s/rad；静摩擦 `0.1035 / 0.5539 / 0.9929` N·m、动摩擦 = 0.5×静摩擦（全部为 PhysX **回读值**，不是采样值） |
| 摩擦是否真进求解器 | 摩擦有效性探针：`panda_joint4` 加恒定 1 N·m，无摩擦 → `|Δq| = 0.5135` rad；注入 5 N·m 静摩擦 → `|Δq| = 2.01e-07` rad（**位移比 2.55e+06**）→ 关节被静摩擦"粘住"，PhysX 确实在用这个系数 |
| 开销 | 注入一次 **0.116 s**（写 64 env 的 mass/inertia/COM/damping/friction）；rollout **304.6 / 357.6 SPS**（19 493 / 22 888 env-steps/s，含每步一次 τ 检查且无 host 往返） |
| 显存 | torch 侧 `peak allocated` **0.016 GiB**；跑完进程剩余 **7.95 / 10.56 GiB** |
| 一句话结论 | 在这台 RTX 2080 Ti 上，**"失配注入"是物理层的一次性写操作（0.12 s），注入后动力学成本几乎不变**；同时 64 个 env 在同一力矩下必然分化（跨 env std 从 `1e-6` 涨到 `1e-2` 量级），为 Day 3 的残差 RL 提供了"同力不同态"的训练分布 |

交付物：`experiments/w11_dynamics_randomization.py`（脚本）、`docs/results/w11_dr_64env{,_control}.json` +
`docs/results/w11_dr_64env{,_control}_run.log`（原始证据）、本文档。

---

## §1 任务与交付物

### 1.1 任务要求（W11 Day 2）

1. 在 Isaac Lab 的 **物理层**注入三类失配（按 env 独立采样）：
   末端未知负载 `U(0, 3) kg`（含惯量/质心）、关节阻尼 `U(0.5, 2.0)`、关节静摩擦 `U(0.1, 1.0)`，
   外加连杆质量 `×U(0.8, 1.2)`。
2. 注入必须是**物理层模型属性**（mass / inertia / COM / DOF damping / DOF friction），
   而不是观测噪声或控制器里的扰动项。
3. **验收**：64 env 在**同一**正弦力矩下，关节位置跨 env 标准差显著 > 0。
4. （本交付额外补的自证）输入是否真的完全相同、参数是否真的写进 PhysX、
   摩擦是否真的进了求解器、分化是否被关节限位污染、以及"不注入就不分化"的对照组。

### 1.2 文件清单

| 文件 | 说明 | 规模 |
|---|---|---|
| `experiments/w11_dynamics_randomization.py` | 主交付：纯力矩 Franka CFG / 零拷贝正弦力矩源 / `DynamicsRandomizer` / A-B 探针 / 摩擦探针 / 计时 / CLI / JSON 落盘 | 991 行 |
| `docs/results/w11_dr_64env.json` | 主结果（配置、A/B 统计、采样与回读、摩擦探针、判定），机器可读 | 21 KB |
| `docs/results/w11_dr_64env_run.log` | 主运行完整日志（2 381 行，含 `[BASE]`/`[DR]`/`[RESULT]`/`[FRICTION]`/`[VERDICT]` 标记行） | 348 KB |
| `docs/results/w11_dr_64env_control.json` + `_control_run.log` | 对照组（`--no_randomization`）：证明"不注入 → 不分化" | 20 KB / 347 KB |
| `docs/reports/w11_task1_dynamics_randomization.md` | 本文档 | — |
| `experiments/isaaclab_env.sh`（W10 已有，未改动） | 环境固化启动器：激活 `py312`、挂 `PYTHONPATH`、`HEADLESS=1` | 64 行 |

### 1.3 运行环境

```
[ENV] torch 2.11.0+cu128 | warp 1.17.0 | isaacsim 6.1.0
[ENV] device = cuda:0 | NVIDIA GeForce RTX 2080 Ti
[ENV] GPU memory total = 10.56 GiB, free = 10.00 GiB
[SCENE] num_envs=64 | num_joints=9 | num_bodies=11
[SCENE] joint names : ['panda_joint1', ..., 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
[SCENE] body names  : ['panda_link0', ..., 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
[SCENE] effort limits [N·m] = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 200.0, 200.0]
[SCENE] gravity = False（disable_gravity = True）
[DRIVER] τ = A ⊙ sin(φ + 2πft) | A = [10.0 × 9] | f = 1.0 Hz | dt = 0.01000 s
```

### 1.4 一条命令复现

```bash
cd /data/zzh/embodied-arm

# 主实验（注入失配 + A/B + 摩擦探针，落 JSON）
bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py \
     --num_envs 64 --steps 200 --trace_every 20 \
     --out_json docs/results/w11_dr_64env.json

# 对照组（不注入：A/B 都不分化，验证"分化只来自注入"）
bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py \
     --num_envs 64 --steps 200 --no_randomization \
     --out_json docs/results/w11_dr_64env_control.json

# 全部参数见 --help；常用：--payload_mass 0 5 --joint_friction 0.5 2 --gravity --torque_amp 5
```

---

## §2 注入方法学：写进去的是 PhysX 的模型属性

### 2.1 三类失配 → 张量接口 → 落在哪里

| 失配 | 采样（每 env 独立） | 本脚本调用（Isaac Lab 3.0） | 最终落到 PhysX |
|---|---|---|---|
| 末端未知负载（质量） | `m_p ~ U(0, 3)` kg | `robot.set_masses_index(masses=(E,B))` | `set_masses` |
| 末端未知负载（惯量 + 质心） | 平行轴定理合成 | `set_inertias_index(inertias=(E,B,9))` + `set_coms_index(coms=(E,B,7))` | `set_inertias` / `set_coms` |
| 连杆质量标称值不准 | `s ~ U(0.8, 1.2)`，`m ← m·s` | `set_masses_index` / `set_inertias_index` | 同上 |
| 关节黏性阻尼 | `b ~ U(0.5, 2.0)` N·m·s/rad | `write_joint_damping_to_sim_index(damping=(E,J))` | `set_dof_dampings` |
| 关节静摩擦 | `τ_s ~ U(0.1, 1.0)` N·m | `write_joint_friction_coefficient_to_sim_index(...)` | `set_dof_friction_properties` |
| 关节动摩擦（库仑） | `τ_d = 0.5·τ_s` | 同上（`joint_dynamic_friction_coeff`） | 同上 |

`(E, B)`：64 env × 11 连杆；`(E, J)`：64 env × 7 臂关节（`panda_joint[1-7]`，手指不注入）。
**每一步写入都带 `_index`**：Isaac Lab 3.0 的 `_index` 版本会同步刷新用户侧缓冲并打时间戳，
所以**紧接着的 `robot.data.*` 回读就能看到新值**（旧教程里的 `root_physx_view.set_masses` 直写
则可能被缓存挡住，见 `articulation_data.py::body_mass` 的 docstring）。

### 2.2 负载为什么必须用平行轴定理合成（不能只加质量）

设手部连杆质量 `m_h`、其 COM 在连杆系的位置 `c_h`、绕自身 COM 的惯量 `I_h`；
负载质量 `m_p`、质心位置 `p_p`（按点质量处理，自身惯量 ≈ 0，`--payload_com 0 0 0.05` 表示法兰外侧 5 cm）：

```
m = m_h + m_p
c = (m_h·c_h + m_p·p_p) / m
I = I_h + m_h·PA(c_h − c) + m_p·PA(p_p − c),      PA(d) = ‖d‖²·E₃ − d·dᵀ
```

两个要点：

* **平行轴项 `m·PA(d)` 恒半正定**，所以合成惯量一定正定 → 不会写出"负惯量"把 PhysX 写崩。
  实测 64 env 的合成惯量最小特征值 `min = 4.07e-4`、`mean = 7.19e-4`、`max = 9.53e-4` kg·m²（全 > 0）；
* 只加质量而不动惯量/质心，等价于"负载是**穿过**法兰的一个点"——末端转动惯量被严重低估，
  抓取类任务上等于没注入。**质心偏移**才是"未知负载"里最难的部分：实测质心位移
  `mean 0.0156 m`、`max 0.0214 m`（相对标称 COM），它会同时改变重力矩（开着重力时）和转动惯量。

### 2.3 单位口径（任务书写的是平动量纲，旋转关节要换算）

任务书给的是 `U(0.5, 2.0) N·s/m` 与 `U(0.1, 1.0) N·m`。但 Franka 全是 **revolute** 关节：

* 黏性阻尼：`N·s/m`（平动）↔ **`N·m·s/rad`**（转动）——把"位移"换成"转角"，数值口径一致；
* 静摩擦：`N·m`（力矩），不是 `N`。

脚本直接按这两个量纲写 PhysX（`--joint_damping` 默认 `0.5 2.0`、`--joint_friction` 默认 `0.1 1.0`），
避免"参数看起来对、物理上是另一个量"的隐性错误。

### 2.4 为什么 actuator 层必须 `k = d = 0`（否则注入被掩盖）

`FRANKA_PANDA_CFG` 默认是隐式驱动器（`stiffness=80, damping=4`），3.0 的隐式驱动器内核算的是
`τ = k·(q*−q) + d·(q̇*−q̇) + τ_ff`。若不清零 `k/d`：

* 我们写 10 N·m 指令，实际进物理的是"10 N·m + 一个未知 PD 项"；
* 注入的 **DOF damping 会被 actuator 的速度环掩盖**（甚至互相抵消），"摩擦/阻尼失配"就白注入了。

所以脚本沿用 W10 的纯力矩配置（`stiffness=0, damping=0`，保留 `armature=1e-3`）。
日志里 Isaac Lab 自己打印的关节表可以直接核对：注入前 9 个关节 `Stiffness / Damping / Static / Dynamic = 0.000`。

### 2.5 采样与可复现

* 采样用 **CPU `torch.Generator(device="cpu").manual_seed(--seed)`**：同一 seed 在任何设备上
  得到同一组失配参数（GPU RNG 在不同卡/驱动上不保证逐位一致）；
* 采样顺序固定为：连杆质量 → 负载质量 → damping → 静摩擦 → 动摩擦；
* 结果 JSON 里同时保存**采样值**与 **PhysX 回读值**，两者必须都打印（见 §3.4）。

---

## §3 验收证据链（"同力不同态"）

单条命令跑完 A/B 两组对照，证据链是**自洽**的：
`输入相同 → 不注入不分化 → 注入后分化 → 参数确实进 PhysX → 摩擦确实进求解器 → 不是限位造成的`。

### 3.1 证据 1：输入完全相同（τ 跨 env 标准差 == 0）

```python
# ZeroCopySineTorqueDriver：τ 先是 (1, 9) 的一行，再广播成 (64, 9)
self._arg.copy_(self.phase); self._arg.add_(self.omega * t)
torch.sin(self._arg, out=self._tau); self._tau.mul_(self.amp)      # 全部原地、无 host 往返
robot.actuators.target_command.set_effort_index(value=self._tau, full_data=True)   # warp kernel
```

脚本**每一步**都在显存里累积 `τ.std(dim=0).max()`（整条 rollout 只做一次 host 同步）：

```
[RESULT] 输入相同性：A 组 τ 跨 env std max = 0.000e+00 N·m | B 组 = 0.000e+00 N·m -> 所有 env 的力矩逐元素相同
```

严格 0（不是"很小"）：64 个 env 的 9 维力矩向量**逐元素相同**。
意义：B 组出现的任何分化，都**不可能**归因于输入差异。

### 3.2 证据 2：A/B 对照（同一条力矩轨迹）

流程（`main()`）：

```
reset(关节回默认位形)  →  A 组：跑 200 步，记录跨 env std
reset(关节回默认位形)  →  rng.apply() 注入失配（0.116 s）
                       →  B 组：跑同一条力矩轨迹，记录跨 env std
```

A/B 的**初值**由 `robot.write_joint_state_to_sim_index(position=default, velocity=0)` 强制一致
（再 `scene.reset()` 清 actuator 命令缓冲；W10 已实测 `scene.reset()` 不碰关节 q/q̇）。

| 指标（第 200 步，跨 64 env 的标准差） | A 组（未注入） | B 组（注入后） | 倍率 |
|---|---|---|---|
| 全关节 std 最大 | `2.161e-06` rad | **`6.977e-02` rad** | **×3.23e+04** |
| `panda_joint7`（末端关节） | `5.087e-07` rad | **`5.056e-03` rad** | ×9.94e+03 |
| 末端位置（相对各自 env 原点） | `1.167e-06` m | **`1.605e-02` m** | ×1.38e+04 |
| 臂关节撞限位 env 数 | 0 / 64 | 0 / 64 | — |

时间序列（`--trace_every 20`，单位见列头）：

| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 |
|---|---|---|---|---|---|---|---|---|---|---|
| A 关节 std max [µrad] | 0.33 | 0.71 | 0.82 | 1.51 | 1.54 | 1.71 | 1.67 | 1.75 | 2.04 | 2.16 |
| A 末端位置 std [µm] | 0.45 | 0.84 | 0.75 | 0.89 | 1.10 | 1.26 | 1.54 | 1.55 | 1.08 | 1.17 |
| B 关节 std max [mrad] | 31.0 | 51.6 | 72.8 | 74.1 | 50.4 | 45.8 | 63.9 | 90.6 | 92.1 | 69.8 |
| B 末端位置 std [mm] | 17.1 | 36.4 | 35.0 | 23.2 | 9.8 | 16.9 | 32.8 | 32.8 | 25.2 | 16.1 |
| B `joint7` std [mrad] | 28.5 | 6.7 | 5.6 | 2.8 | 2.8 | 29.8 | 8.4 | 7.2 | 5.0 | 5.1 |

两点解读（诚实版）：

* A 组不是"严格 0"，而是 `1e-6 ~ 2e-6` rad 量级并**单调缓慢增长** —— 这是 **float32 + PhysX 求解器
  在不同 env 上的浮点累积差异**（同样的数值、同样的初值，但求解顺序/归约顺序不保证逐位一致）。
  它比阈值 `1e-3` 低 **3 个数量级**，且**与"是否注入"无关**（对照组里 A、B 完全一致，见 §3.3）；
* B 组不是单调增长，而是随正弦相位起伏（`2.8 ~ 92 mrad`）—— 因为失配参数是"**速度/力矩相关**"的
  （库仑摩擦、黏性阻尼），相位不同时分化程度不同。这也正是"非线性摩擦"该有的样子：
  它不会像一次线性扰动那样一直线性放大。

### 3.3 证据 3：对照组（`--no_randomization`）—— 分化只来自注入

| 指标 | A 组 | B 组（复位 + 空注入 + 同轨迹） | 倍率 |
|---|---|---|---|
| 末端关节 std [rad] | `5.086528744868701e-07` | `5.086528744868701e-07` | ×1 |
| 全关节 std max [rad] | `2.1615e-06` | `2.1615e-06` | ×1 |
| 末端位置 std [m] | `1.1668574870782322e-06` | `1.1668574870782322e-06` | ×1 |
| 判定 | `[VERDICT] 同力不同态：CONTROL-OK（对照组，未注入 → 不分化，符合预期）` | | |

**逐位相同**（不是"接近"）：说明 `reset → rollout → reset → rollout` 这条流程本身**不引入任何分化**，
也说明 §3.2 里 A 与 B 的差异**只能**来自 `rng.apply()` 写进 PhysX 的那些参数。这是整个验收里最关键的
一次对照 —— 没有它，"A/B 不同"也可能被解释成"第二次 rollout 的数值路径不同"。

### 3.4 证据 4：参数确实写进了 PhysX（不是只写进了采样张量）

| 量 | 采样值（`dr_sampled`） | PhysX 回读（`dr_readback`） | 标称值 |
|---|---|---|---|
| 负载质量 [kg] | min 0.0088 / mean 1.499 / max 2.957 | `panda_hand` 质量：min **0.6143** / mean **2.173** / max **3.756** | 0.6804 |
| 连杆质量比例 | min 0.8009 / mean 0.992 / max 1.1999 | （分布于 11 个连杆，逐连杆 mean 见 JSON `link_mass_scale_per_body_mean`） | 1.0 |
| `panda_hand` 惯量迹 [kg·m²] | — | min 0.004201 / mean 0.005267 / max 0.006446 | — |
| `panda_hand` 惯量最小特征值 | — | min **4.07e-4** / mean 7.19e-4 / max 9.53e-4（> 0 ⇒ 正定） | — |
| `panda_hand` 质心位移 [m] | — | min 3.7e-4 / mean **0.01564** / max 0.02143 | 0 |
| 关节 damping [N·m·s/rad] | min 0.5063 / mean 1.253 / max 1.998 | **与采样值一致**（0.5063 / 1.253 / 1.998） | 0.0（注入前） |
| 关节静摩擦 [N·m] | min 0.1035 / mean 0.5539 / max 0.9929 | **与采样值一致** | 0.0（注入前） |
| 关节动摩擦 [N·m] | — | min 0.0517 / mean 0.2769 / max 0.4964（= 0.5×静摩擦） | 0.0（注入前） |

日志原文片段：

```
[DR] 注入完成：关节 ['panda_joint1', ..., 'panda_joint7']（7 个），负载连杆 = panda_hand（body_id=8），质心写入路径 = set_coms_index
[DR]   panda_hand 质量 [kg]      : min 0.6143 / mean 2.173 / max 3.756 / std 0.9644（标称 0.6804）
[DR]   panda_hand 惯量最小特征值 : min 0.0004067 / mean 0.0007193 / max 0.0009535（>0 表示正定）
[DR]   关节 damping [N·m·s/rad]   : min 0.5063 / mean 1.253 / max 1.998 / std 0.4294
[DR]   关节静摩擦 [N·m]           : min 0.1035 / mean 0.5539 / max 0.9929 / std 0.2479
```

`质心写入路径 = set_coms_index`：Isaac Lab 3.0 的一级接口在本机**可用**（脚本仍保留
`root_view.set_coms` 的二级回退 + `payload_mode=mass` 的三级退化，并会打印 `[WARN]`）。

### 3.5 证据 5：摩擦真的进了求解器（参数写得进去 ≠ 物理上生效）

探针设计（控制变量）：

| 变量 | 处理 |
|---|---|
| 被测关节 | `panda_joint4`（effort limit 87 N·m，不会被限幅干扰） |
| 施加力矩 | 恒定 **1 N·m**（小于注入的 5 N·m 静摩擦） |
| 所有臂关节 damping | 置 0（排除黏性耗散的干扰，只留静摩擦一个变量） |
| 对照 | 静摩擦 0 N·m × 100 步 ↔ 静摩擦 5 N·m × 100 步 |

| 组 | `|Δq|` mean [rad] | `|Δq|` max [rad] |
|---|---|---|
| 静摩擦 0 N·m | **0.5135** | 0.8677 |
| 静摩擦 5 N·m | **2.01e-07** | 2.38e-07 |
| 位移比（无摩擦 / 有摩擦） | **2.55e+06** | — |

结论行：`[FRICTION] 位移比（无摩擦/有摩擦）= 2.553e+06 -> 静摩擦确实进了求解器`。
换句话说：**PhysX 真的在用 `set_dof_friction_properties` 写的那个值**，而不是把参数吃掉返 0；
这也排除了 §3.2 的分化只是"质量失配"单一来源的可能（damping/friction 至少各自可独立验证）。

### 3.6 证据 6：分化不是"关节限位饱和"造成的

`[RESULT] 臂关节撞限位 env 数` 逐关节打印，A / B 两组都是 **0 / 0 / 0 / 0 / 0 / 0 / 0**：

```
[RESULT]   panda_joint1  :   0 /   0      ...     panda_joint7  :   0 /   0
[RESULT] 任意关节撞限位 env 数（含手指）：A 64 / B 64（手指行程只有 4 cm，恒为 100% 属正常）
```

* 7 个臂关节在整个 200 步里**从未顶到限位** → 跨 env 离散度是"自由运动分化"，不是"卡在限位上被夹平/拉开"；
* 夹爪指关节行程只有 `[0, 0.04] m`，10 N·m 的指令会瞬间把它顶到端点（64/64）。所以脚本**把"撞限位"
  的统计口径拆成"臂关节"和"任意关节"两个**，避免用"含手指"的数字误判实验有效性。

---

## §4 开销与吞吐

| 阶段 | 64 env × 200 步 | 折算 |
|---|---|---|
| A 组 rollout（未注入） | 0.657 s | **304.6 SPS** = 19 493 env-steps/s |
| `rng.apply()`（一次写入 64 env 的 mass/inertia/COM/damping/friction） | **0.116 s** | 只做一次，与步数无关 |
| B 组 rollout（注入后） | 0.559 s | **357.6 SPS** = 22 888 env-steps/s |
| 显存 | torch `peak allocated` **0.016 GiB**；进程剩余 7.95 / 10.56 GiB | 与 W10（64 env 2.06 GiB 峰值）同一量级 |

* 每次 rollout 的计时**包含每步一次 τ 跨 env 检查**（在显存里累积，整条只同步一次），
  且仍然比 W10 的 64 env 基线（≈ 371–389 SPS）同量级 → **注入失配不改变动力学成本的量级**；
* B 组比 A 组快 ~17%，是 GPU 时钟/调度抖动（同量级重复运行 W10 也见过类似抖动），
  不应解读为"摩擦让仿真变快"。

---

## §5 诚实披露：本验收**不能**证明什么

| 项 | 事实 | 影响 / 处理 |
|---|---|---|
| A 组不是严格 0 | 未注入时跨 env std ≈ `1e-6 ~ 2e-6` rad（随步数缓慢增长） | PhysX/float32 的求解顺序差异；比阈值低 3 个数量级；对照组证明它与注入无关。若论文里要写"完全同步"，应写"低于 `1e-6` rad 量级" |
| "显著大于 0" 是**开环**分化 | 本实验没有控制器，只是"同一力矩下自由运动分化" | 这是 W11 Day 2 的**正确**目标（证明物理层失配生效）；"控制器性能下降"要等 Day 3 的残差 RL 才能量化 |
| 重力默认关 | 默认 `disable_gravity = True` | 原因见脚本 docstring：只喂正弦力矩 + 开重力，手臂会先被拽到限位、再在限位做"接触求解器压力测试"，离散度被限位污染。需要时 `--gravity`，脚本会打印撞限位 env 数 |
| 手指关节恒撞限位 | 64/64（行程仅 4 cm） | 指标统计口径已拆开（`envs_at_arm_joint_limit` / `envs_at_any_joint_limit`）；手指关节不注入摩擦/阻尼 |
| 负载质量含 0 | `U(0, 3) kg`，实测 min `0.0088 kg` | 少数 env 近乎"空手"，属于预期分布（任务书给的范围就是 0~3 kg） |
| PhysX 硬约束 `μs ≥ μd` | 首次跑探针时报 `Static friction effort must be greater than or equal to dynamic friction effort.` | 只把静摩擦清零而 `dynamic` 保留旧值时会被 PhysX 拒绝；脚本现在**静/动摩擦成对写入**（探针里两者同值），报错已消失（最终日志中 `Error` 计数 = 0） |
| 只注入模型属性 | 没有注入随机外扰 / 传感器噪声 / 观测掩码 | 这是刻意选择：要证明的是"物理层失配"，观测层随机化是 Day 3 的事 |
| `set_coms_index` 可用性 | 本机走通一级接口（`com_write_path = set_coms_index`） | 二级/三级回退路径（`root_view.set_coms` → `payload_mode=mass`）已实现但**本机未被触发**，未实测（若换机器要留意 `[WARN]` 行） |

---

## §6 与 Day 3（残差 RL）的衔接

| 本日产出 | Day 3 用法 |
|---|---|
| `DynamicsRandomizer`（按 env 独立采样 + 张量写 PhysX） | 直接作为 RL 环境的 **DR 层**：`reset` 时重采样一次（脚本把"采样"与"注入"拆成了两个方法，就是为这一步留的） |
| `sampled_summary()` / `readback()` | RL 训练时落盘"本回合注入了什么"，用于事后分析策略是否"适应"了特定参数 |
| 摩擦有效性探针 | 说明这些 DR 参数**真的在影响物理**，因此"策略在 DR 下更难"不是采样噪声 |
| "同力不同态"证据链 | 报告里"我们确实构造了 sim-to-real gap"的第一块地基；Day 3 要展示的是"残差策略把 gap 补回来"（同力下末端轨迹 std 变小 / 跟踪误差下降） |
| `--seed` + CPU generator | DR 参数可复现，便于做"同参数、有无残差策略"的配对比较 |

---

## §7 产物与可核对清单

```
experiments/w11_dynamics_randomization.py          主脚本（991 行）
docs/results/w11_dr_64env.json                     主结果（config / baseline / randomized / dr_sampled / dr_readback / friction_probe / verdict）
docs/results/w11_dr_64env_run.log                  主运行日志（2 381 行 / 348 KB）
docs/results/w11_dr_64env_control.json             对照组结果（mode = control_no_randomization）
docs/results/w11_dr_64env_control_run.log          对照组日志
docs/reports/w11_task1_dynamics_randomization.md   本文档
```

核对用的关键 `grep`：

```bash
grep -n 'RESULT\|VERDICT\|FRICTION' docs/results/w11_dr_64env_run.log        # 主结论
grep -n 'RESULT\|VERDICT'          docs/results/w11_dr_64env_control_run.log # 对照结论
grep -c 'Error'                    docs/results/w11_dr_64env_run.log         # 应为 0（无 PhysX 报错）
```
