# W4 Task 1：动作块行为克隆（Action Chunking BC）

## 1. 目标

在已跑通的单步 BC 流水线（W3）上加三点改造，并给出**闭环 20 回合**的定量对比：

1. **动作块标签**：对专家轨迹做滑窗切分，标签由 `a_t` 变为 `[a_t, ..., a_{t+K-1}]`（形状 `(K, action_dim)`），
   窗口**不跨回合边界**、末端不足 `K` 步处**零填充**；
2. **网络输出头**：维度由 `action_dim` 变为 `K * action_dim`；
3. **闭环执行**：每隔 `K` 步推理一次，一次前向打出 `K` 步动作块并开环执行。

## 2. 实现与文件

| 文件 | 说明 |
| --- | --- |
| `src/engine.py` | **新增**：共享训练引擎（参数解析 / 配置继承 / 运行目录 / 日志 / TensorBoard / 训练循环 / 优雅停机 / 断点续训 / 闭环评估上报）。单步 BC 与动作块 BC 共用，保证两者训练数学完全一致 |
| `experiments/train_bc.py` | 重构到 `src/engine.py`（对外行为不变，W3 日志复现逐位一致） |
| `experiments/train_chunking_bc.py` | **新增**：动作块 BC 训练入口（K 解析 + 多执行模式评估） |
| `configs/chunking_bc_config.yaml` | **新增**：`defaults: bc_config.yaml` 继承，唯一变量是 `model.chunk_size` 与 `inference.*` |
| `src/datasets.py` | 新增 `collect_expert_trajectories`（保留回合边界）/ `build_action_chunks`（滑窗 + 填充 + `valid` 掩码）/ `ChunkingBCDataset` / `build_chunking_dataloaders`；`_rollout_closed_loop` 统一单步与动作块执行调度 |
| `src/networks.py` | `BasePolicy` 统一 `chunk_size` / `output_dim = K * action_dim` / `forward_actions()`；`K=1` 时与改造前逐位一致 |
| `tests/test_action_chunking.py` | **新增**：18 个单元测试（切分 / 填充 / 不跨界 / 输出头维度 / K=1 兼容 / 数据对齐 / 闭环调度 / EMA 时序对齐） |

关键代码位置：

* 滑窗切分：`src/datasets.py: build_action_chunks`
* 动作块数据集：`src/datasets.py: ChunkingBCDataset`、`build_chunking_dataloaders`
* 输出头：`src/networks.py: BasePolicy.__init__`（`output_dim = chunk_size * action_dim`）
* 推理调度：`src/datasets.py: _rollout_closed_loop`（`每 execute_steps 步推理一次`）

## 3. 与单步 BC 的可比性（对比公平性）

* 两者使用**同一批专家数据**（`expert_pendulum.npz` 与 `expert_pendulum_traj.npz` 内容逐位相同，已由单测断言）；
* 切分方式逐样本对齐：同为 `numpy.random.default_rng(seed).permutation(N)` + `train_ratio`，
  训练/验证的**状态集合完全一致**，唯一差别是标签由 `a_t` 变成 `K` 步动作块；
* 网络结构 / 优化器 / 学习率 / epoch 数 / 数据增强 / 种子全部继承自 `bc_config.yaml`；
* 因此闭环回报差异只来自"动作块建模 + 每隔 K 步推理"这一变量。

## 4. 单元测试

```
python tests/test_action_chunking.py      # 无需 pytest
```

覆盖点：

1. 切分与填充：`chunk[t, j] == a_{t+j}`、末端零填充位置与 `valid` 掩码精确对应；
2. **不跨轨迹边界**：上一条轨迹末尾窗口不会取到下一条轨迹开头的动作；
3. 输出头维度：`K ∈ {1,4,8}` 时 `output_dim == K * action_dim`，`forward` / `forward_actions` 形状正确（MLP/CNN/GRU）；
4. **K=1 向后兼容**：未配置 `chunk_size` 时输出形状、参数与输出数值与改造前的单步 BC 逐位一致；
5. 标签对齐：`chunk[:, 0]` 恒等于单步 BC 的标签 `a_t`（零填充不污染即时响应）；
6. DataLoader 对齐：两种构造方式的训练/验证状态逐样本一致；
7. 闭环调度：`K=1` 与单步评估逐回合一致、前向次数相同；`K=8` 时前向次数 ≈ 200/8；`replan_every` 精确控制开销；
8. **EMA 时序对齐**：用替身环境逐动作解析验证 EMA 按绝对时刻平滑（`0,1,6,7,16,17,...`），
   `ema_alpha=1` 退化为开环执行，`replan_every = K`（无重叠）时 EMA 与开环逐位一致。

结果：`18/18 通过`。

## 5. 实验结果

实验设置：Pendulum-v1，专家数据 10000 步（50 条完整轨迹 × 200 步），
训练 8000 / 验证 2000，批大小 64，Adam lr 1e-3，100 epoch，种子 42，闭环评估 20 回合（同一种子序列）。

### 5.1 闭环回报对比（20 回合，同一环境种子序列 42..61）

| 方法 | K | 执行方式 | 重规划间隔 | 每回合前向次数 | 平均回报 | 标准差 | 最优 / 最差回合 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **单步 BC（W3 基线，同一引擎）** | 1 | 逐步推理 | 1 | 200 | **-156.95** | 68.99 | -115.01 / -361.58 |
| 动作块 BC | 4 | 开环整块 `chunk` | 4 | **50** | **-155.58** | 66.22 | -114.95 / -354.02 |
| 动作块 BC | 4 | 块间 EMA（α=0.5） | 2 | 100 | -155.94 | 66.64 | -115.22 / -354.59 |
| 动作块 BC | 4 | 逐步重规划 `receding` | 1 | 200 | -158.06 | 71.42 | -114.86 / -358.84 |
| 动作块 BC | 8 | 开环整块 `chunk` | 8 | **25** | **-219.68** | 113.51 | -114.82 / -483.78 |
| 动作块 BC | 8 | 块间 EMA（α=0.5） | 4 | 50 | -363.12 | 87.90 | -245.99 / -523.53 |
| 动作块 BC | 8 | 逐步重规划 `receding` | 1 | 200 | -158.32 | 71.77 | -114.67 / -354.45 |
| 专家控制器（oracle 参考上界） | — | 每步能量塑形 + PD | 1 | — | -175.00 | 83.72 | -116.37 / -394.20 |

> 参考：Pendulum-v1 单回合 200 步，奖励 `-(θ² + 0.1θ̇² + 0.001τ²)`（永远是负数，越接近 0 越好）：
> 完全下垂静止 ≈ -π²/步 → 200 步 ≈ **-1974**（理论最差、乱转可到 ≈ -3254），倒立静止 ≈ 0；
> 表中所有方法的"最好回合"都落在 -114 ~ -116，即**已经能在整个回合里保持倒立**。
>
> 低层次解读（面向零基础）见 [`docs/零基础读本.md`](../零基础读本.md)。
>
> 运行目录（TensorBoard / checkpoint 存档）：单步 BC `logs/bc_pendulum_20260915_164250`、
> K=4 `logs/chunking_bc_k4_20260915_164844`、K=8 `logs/chunking_bc_pendulum_20260915_165015`；
> 对应的文本训练日志已归档为 `docs/results/w3_bc_pendulum_train.log`、
> `docs/results/w4_chunking_bc_k4_train.log`、`docs/results/w4_chunking_bc_k8_train.log`。

### 5.2 验证集 MSE（标签为 K 步动作，数值跨 K 不可直接比较）

| 方法 | 最佳验证 MSE | 取得 epoch |
| --- | --- | --- |
| 单步 BC（K=1） | 0.0928 | 86 |
| 动作块 BC（K=4） | 0.1963 | 99 |
| 动作块 BC（K=8） | 0.2162 | 98 |

（K 越大，标签里"越靠后的未来动作"越难预测，逐元素 MSE 自然上升；
因此**主指标取闭环回报**，MSE 只用于确认训练收敛与选最优权重。）

### 5.3 复现命令

```bash
# 专家 oracle 基线（回报的参照系，约 5 秒）
python experiments/eval_expert.py

# 单步 BC 基线（同一引擎 / 同一数据 / 同一训练设置）
python experiments/train_bc.py

# 动作块 BC：K=8（默认配置）
python experiments/train_chunking_bc.py

# 动作块 BC：K=4 与 EMA 主模式
python experiments/train_chunking_bc.py --set model.chunk_size=4
python experiments/train_chunking_bc.py --set inference.execution=ema

# 单元测试（无需 pytest）
python tests/test_action_chunking.py
```

## 6. 结论

## 6. 结论

1. **核心结论**：`K=4` 的动作块 BC 用 **1/4 的推理次数**（50 次/回合 vs 200 次/回合）拿到与单步 BC
   **同等**的闭环回报（-155.58 vs -156.95，差异远小于回合间标准差 67~69），即"一次前向打出 K 步动作、
   开环执行"在 Pendulum 上不损失控制质量，却显著降低推理开销 —— 这正是动作块（ACT 风格）方法用于
   高频真实机械臂时的价值所在。
2. **K 变大后开环误差累积**：`K=8` 开环回报降到 -219.68（标准差也从 66 涨到 114），
   但把 `K=8` 的策略改成 **receding 执行**（每步重规划、只执行块首动作）立刻回到 -158.32 ——
   说明退化来自**开环执行时间过长**，而非"块条件策略"本身学得差：块条件网络在逐步闭环下
   与单步 BC 完全同级。
3. **EMA 平滑需按绝对时刻对齐**：初版实现按"块内下标"混合相邻规划，等价于把 4 步前的动作混进当前动作，
   回报崩到 -981（K=4）/ -1000（K=8）。修正为**按绝对时刻对齐**（ACT temporal ensembling）后，
   K=4 的 EMA 恢复到 -155.94（与开环同级），K=8 的 EMA 为 -363.12（仍弱于 receding，因为 α=0.5 的
   平滑会延迟闭环纠偏）。该陷阱已写成单元测试 `test_closed_loop_ema_blends_by_absolute_time` 固化。
4. **与专家上界对比**：专家控制器自身 20 回合为 -175.00，而 BC 策略（单步或 K=4 动作块）都略优
   （-156），说明网络学到了比原控制器更平滑的力矩（原专家为 bang-bang 能量塑形 + PD，抖动更大）。
   表中"最好回合"≈ -115 已接近倒立静止的理论上限。
5. **工程可复用性**：单步 BC 与动作块 BC 共用 `src/engine.py`（训练/日志/续训/评估）与 `src/datasets.py`
   的数据切分口径，`K=1` 时新代码与 W3 版本逐位一致（回归对比：3 epoch 的 train/val MSE 逐位相同，
   见 `/tmp/regress/`），因此"动作块"是一次可开关、可回退的增量改造。

## 7. 后续可做（W4 余下 / W5 接入点）

* 把 `chunk_size` 做成调度（训练时随机 K / 推理时 K 可调），观察回报随 K 的曲线拐点；
* 用 DP（Diffusion Policy）或 ACT 的 CVAE 头替换直接回归头，处理多模态专家数据；
* 在真实机械臂上验证"每 K 步一次推理"的延迟收益（当前 Pendulum 环境动力学成本远低于网络前向）。
