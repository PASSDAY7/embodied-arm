# embodied-arm —— 具身智能 / 机械臂强化学习工程

从零手写深度学习与强化学习算法（**行为克隆 → 动作块 → 扩散策略 → REINFORCE → PPO → SAC**），
再把它们接到**物理级仿真**上（Isaac Sim 6.1 + Isaac Lab 3.0：纯力矩零拷贝控制、域随机化、轨迹跟踪 MDP），
最后落到课题主线：**双层残差控制（标称前馈 + PID 兜底 + SAC 学残差）端到端训练** ——
45 万环境步后关节跟踪 RMSE `0.051866 → 0.045955 rad`（×1.13）、末端笛卡尔 RMSE `1.3259 → 0.8651 cm`（×1.53），
两条验收线都压过去。最终目标：2027 年 IROS / RA-L 投稿的「残差 RL 机械臂鲁棒轨迹跟踪」（见 §8 路线图）。

| 想干什么 | 去哪看 |
| --- | --- |
| 完全没有基础，先搞懂"这些实验到底在干什么" | [`docs/零基础读本.md`](docs/零基础读本.md) |
| 看每周实验报告（数据表 / 消融 / 排错记录 / 结论） | [`docs/reports/`](docs/reports/) |
| 只看主线结果（W13/W14 残差 SAC 端到端） | [`docs/reports/w13_task1_residual_sac.md`](docs/reports/w13_task1_residual_sac.md) |
| 看训练曲线 | [`docs/figures/`](docs/figures/) |
| 逐项对账（JSON / CSV / 训练日志） | [`docs/results/`](docs/results/) |
| 英文工程说明 | [`README.en.md`](README.en.md) |

---

## 1. 目录结构

```
embodied-arm/
├── README.md / README.en.md   # 中文 / 英文入口
├── requirements.txt           # 纯 PyTorch 侧依赖（W2-W9）
│
├── configs/                   # 超参数（改实验不改代码）
│   ├── bc_config.yaml            # W3   单步行为克隆
│   ├── chunking_bc_config.yaml   # W4-1 动作块 BC（defaults 继承 bc_config）
│   ├── diffusion_config.yaml     # W4-2 迷你条件扩散策略
│   ├── reinforce_config.yaml     # W6   REINFORCE
│   ├── ppo_config.yaml           # W8   PPO-Clip + GAE
│   └── sac_config.yaml           # W9   SAC
│
├── src/                       # 手写算法实现（模块名 = 算法名，tests/ 直接导入）
│   ├── networks.py  datasets.py  engine.py  utils.py   # W3/W4 BC 底座（网络/数据/引擎/日志）
│   ├── diffusion.py                                    # W4-2 DDPM 调度 + 噪声预测 + 采样
│   ├── reinforce.py  ppo.py  sac.py                    # W6 / W8 / W9 三个 RL 算法（纯手写）
│
├── experiments/               # 实验入口（每个都能直接跑）
│   ├── eval_expert.py  train_bc.py  train_chunking_bc.py  train_diffusion_policy.py
│   ├── train_reinforce.py  train_ppo.py  train_sac.py
│   ├── export_tb_curves.py                    # TensorBoard 事件 → PNG + 标量 CSV
│   ├── isaaclab_env.sh                        # ★ Isaac Sim/Lab 环境固化启动器（一条命令跑实验）
│   ├── w10_arm_torque_control.py  w10_plot_scale.py     # W10  64 路 Franka 纯力矩零拷贝回路
│   ├── w11_dynamics_randomization.py                    # W11  物理层域随机化（负载/摩擦/质量）
│   ├── w12_trajectory_tracking_env.py                   # W12  轨迹跟踪 MDP + 开环基线
│   ├── w13_residual_sac_train.py                        # ★ W13/W14 双层残差 SAC（自包含，108 个 CLI 开关）
│   ├── w15_paper_benchmarks.py                          # ★ W15  论文实验矩阵：四控制器 Baseline + 极端压测 + 消融
│   └── export_paper_figures.py                          # W16  论文级矢量图（Fig 1–3）+ LaTeX 大表（Table 1/2）
│
├── tests/                     # 纯标准库单元测试（不需要 pytest，合计 148 项）
│   ├── test_action_chunking.py(18)  test_diffusion.py(15)  test_reinforce.py(29)
│   └── test_ppo.py(38)              test_sac.py(48)
│
├── docs/
│   ├── 零基础读本.md          # 白话读本：动作块 / 指标怎么看 / FAQ
│   ├── reports/               # ★ 每周实验报告（W4 → W13，含数据表与排错记录）
│   ├── figures/               # 曲线 PNG + 原始标量 CSV
│   ├── notes/                 # W2 深度学习笔记（CNN / AlexNet）
│   └── results/               # 归档证据：*.json（结构化结果）/ *_history.csv / *.log（训练日志）
│
├── w2_pytorch_basics/         # W2 PyTorch 基础练习（单文件小脚本）
├── data/                      # 专家数据 npz、CIFAR-10 缓存（已 gitignore，可重新生成）
└── logs/                      # 运行输出：checkpoint / TB 事件 / 原始控制台日志（已 gitignore）
```

> **不进版本库**：`logs/`、`data/`、`*.pt` —— 跑一次实验就会重建（见 §7）。
> **进版本库的证据**：`docs/results/*.json`（结构化结果）、`*_history.csv`（逐次评估）、
> `*.log`（训练日志，已剔除 Isaac Sim / PhysX / Warp 启动噪声，每个文件头部有说明；
> 未删减原件保留在 `logs/raw_console_logs/`，不入库）。

---

## 2. 环境（两条线）

| | W2–W9（纯 PyTorch） | W10–W13（Isaac Lab） |
| --- | --- | --- |
| conda 环境 | `rl`（Python 3.10） | `py312`（Isaac Sim 6.1 已装在 site-packages 里） |
| 依赖 | `pip install -r requirements.txt` | Isaac Sim 6.1 + Isaac Lab 3.0（源码 `/data/zzh/IsaacLab`）+ torch 2.6.0+cu124 / warp 1.17 |
| 入口 | `python experiments/train_*.py` | `bash experiments/isaaclab_env.sh experiments/w1*_*.py …` |

```bash
# A. 纯 PyTorch 侧（W2-W9）
conda create -n rl python=3.10 -y && conda activate rl
pip install -r requirements.txt          # torch / numpy / gymnasium / pyyaml / tensorboard / matplotlib
python experiments/train_sac.py          # 例：W9 的 SAC

# B. Isaac Lab 侧（W10-W13）——环境由脚本固化，不需要 isaaclab.sh，也不需要 uv
bash experiments/isaaclab_env.sh --check                     # 只打印版本自检，不启动仿真
bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py --num_envs 64 --steps 500
```

> `isaaclab_env.sh` 可用环境变量覆盖：`CONDA_ENV` / `ISAACLAB_DIR` / `PYTHON` / `HEADLESS`；
> 它用 `python -u` 启动目标脚本，否则把日志重定向到文件时会看不到输出。
> 没有显卡也能跑 W2–W9：所有训练脚本都支持 `--device cpu`（只是慢）。
> 硬件：RTX 2080 Ti（11 GB）/ Ubuntu 22.04。

---

## 3. 快速开始

### 3.1 先跑测试（约 1 分钟，确认代码没坏）

```bash
python tests/test_sac.py               # 48 项，CPU 约 18 s
python tests/test_ppo.py               # 38 项，约 14 s
python tests/test_reinforce.py         # 29 项
python tests/test_action_chunking.py   # 18 项
python tests/test_diffusion.py         # 15 项
```

全部输出 `通过 N/N` 即为正常（纯标准库，无需安装 pytest）。

### 3.2 按周复现实验

| 周次 | 任务 | 一条命令 | 主要产物 |
| --- | --- | --- | --- |
| W3 | 单步 BC 基线 | `python experiments/eval_expert.py`<br>`python experiments/train_bc.py` | 专家 oracle −175.00；BC 闭环 −156.95 |
| W4-1 | 动作块 BC（K=4 / K=8） | `python experiments/train_chunking_bc.py --set model.chunk_size=4` | K=4 闭环 −155.58、前向 200 → 50 |
| W4-2 | 迷你扩散策略 + 消融 | `python -u experiments/train_diffusion_policy.py` | 歧义任务命中 98.5%；闭环 −331.8（慢 132.8×） |
| W6 | REINFORCE（CartPole） | `python experiments/train_reinforce.py` | 解到 500.00 @ep120 |
| W8 | PPO-Clip（Pendulum） | `python experiments/train_ppo.py` | 194 560 步达 −199.91 |
| W9 | SAC（Pendulum） | `python experiments/train_sac.py` | 4 000 步达 −173.70（比 PPO 省 ≈48×） |
| W10 | 64 路纯力矩零拷贝回路 | `bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py --num_envs 64 --steps 500` | 389.5 SPS = 24 929 env-steps/s；零拷贝 `data_ptr` 相同 |
| W10 | 环境数扫描（1/8/64/256） | 同上逐个改 `--num_envs … --steps 200`，再 `python experiments/w10_plot_scale.py` | `docs/figures/w10_torque_scale.png` |
| W11 | 域随机化注入（A/B） | `bash experiments/isaaclab_env.sh experiments/w11_dynamics_randomization.py --num_envs 64 --steps 200` | 跨 env std 1e-6 → 7e-2 rad（×3.2e4）；`--no_randomization` 为对照组 |
| W12 | 轨迹跟踪 MDP + 5 基线 | `bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py --num_envs 64 --steps 500` | `pd_dr` 2.3016 cm / `pd_ff_dr` 1.3259 cm |
| W13/W14 | 双层残差 SAC 端到端 | 见 §3.3 | 关节 0.045955 rad、末端 0.8651 cm（两条验收线都过） |
| W15 | 论文实验矩阵（3 条赛程） | 见 §3.4 | Table 1 四控制器对比 + Figure 3 压测 + Table 2 消融 → `docs/results/w15_*.json` |
| W16 | 论文级矢量图 + LaTeX 表 | `python experiments/export_paper_figures.py` | `docs/figures/fig{1,2,3}_*.pdf/png`、`docs/results/w15_tables.tex` |

> 每个 Isaac 脚本都是 `--help` 自解释（W10 27 个开关、W11 32 个、W12 40+ 个、W13 108 个、W15 113 个）；
> 各周报告里都写了"这条数字是哪条命令跑出来的"。

### 3.3 ★ 复现主线结果（W13/W14，450k 步 ≈ 10 分钟）

```bash
# --run_name 决定三件东西的落点：权重目录 logs/<run_name>/、history CSV、曲线图文件名。
# 必须与落盘 JSON 里的 config.run_name 一致，否则会写进默认目录 logs/w13_residual_sac/（= 臂 A 的目录），
# 把臂 A 的 policy_final.pt / *_history.csv / *_curves.png 覆盖掉。
bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \
    --num_envs 64 --steps 500 --total_steps 450000 --w_ee 50000 --run_name w13_residual_sac_ee50k \
    --out_json docs/results/w13_residual_sac_ee50k.json > docs/results/w13_residual_sac_ee50k.log 2>&1

# 长跑建议后台执行，另开一个终端看日志：
#   nohup bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py … > log 2>&1 &
#   tail -f log

# 权重读回 + 评估（--eval_only 不写权重、不覆盖 history CSV；JSON 默认落 <run>_eval_only.json）
bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \
    --num_envs 64 --steps 500 --eval_only \
    --load_policy logs/w13_residual_sac_ee50k/policy_best_ee.pt --run_name w13_best_ee_check
```

### 3.4 论文实验矩阵（W15/W16）

```bash
# ① Table 1：四控制器横向对比（PD / 标称前馈+PD / 纯 RL / 双层残差 Ours）
#    纯 RL 权重不存在时会先按同一预算（45 万环境步、同奖励、同观测）训练一轮再评估
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage baselines

# ② Figure 3：三条极端压测轴（载荷 0→5 kg / 摩擦阻尼 ×1→×3 / 动作延迟 0→3 步）
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage payload
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage friction
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage latency
# 一次全跑（ablation 默认不排：9 个变体各要一轮训练 ≈ 1.5 h）
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage all --do_ablation

# ③ Table 2：消融（历史长度 H / 残差限幅 |Δτ| / jerk 惩罚项 w_j），每个变体单独训练到同一预算
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage ablation --ablation_dims history,res_limit,jerk

# ④ 出图 + LaTeX 大表（纯 numpy/matplotlib，不需要 Isaac Sim）
python experiments/export_paper_figures.py
```

产物：`docs/results/w15_{baselines,payload,friction,latency,ablation}.json`、四段赛程**累积**写入的
`docs/results/w15_benchmarks.csv`（论文大表的机器可读版，按 `stage+scenario+controller` 去重）、
`docs/results/w15_per_env.csv`（逐 env 指标，箱线图用）、`docs/figures/fig{1,2,3}_*.pdf|png`、
`docs/results/w15_tables.tex`（Table 1/2 的 LaTeX 源码，同时打印到终端）。

判据：`--stage baselines` 会自动用 **W12 的 `run_episode`** 再跑一遍 `pd_dr`/`pd_ff_dr`，与新的统一评测
循环逐字段对账（阈值 1e-9），结果落在 JSON 的 `stages.baselines.self_checks.unified_vs_w12_path`——
这份对账是"Table 1 里那两行仍然等于 W12/W13 报告里那两个数"的硬证据。

---

## 4. 每周进展一览（W2 → W16）

| 周次 | 任务 | 核心代码 | 报告 | 一句话结论 |
| --- | --- | --- | --- | --- |
| W2 | PyTorch 基础：线性拟合、MLP 拟合 sin、CNN、序列预测 | `w2_pytorch_basics/` | [`docs/notes/`](docs/notes/) | 手写训练循环 / 优化器 / 卷积网络 |
| W3 | 行为克隆工程化：配置、日志、TensorBoard、断点续训 | `experiments/train_bc.py`、`src/{networks,datasets,engine,utils}.py` | [`docs/零基础读本.md`](docs/零基础读本.md) | 单步 BC 闭环 **−156.95 ± 68.99**（专家 oracle −175.00） |
| W4-1 | 动作块 BC（Action Chunking） | `experiments/train_chunking_bc.py` | [`w4_task1_action_chunking.md`](docs/reports/w4_task1_action_chunking.md) | K=4 开环 **−155.58**，每回合前向 200 → **50**；K=8 反而退化到 −219.68 |
| W4-2 | 迷你条件扩散策略 vs 确定性回归 | `src/diffusion.py`、`experiments/train_diffusion_policy.py` | [`w4_task2_diffusion_policy.md`](docs/reports/w4_task2_diffusion_policy.md)（[零基础版](docs/reports/w4_task2_diffusion_policy_零基础版.md)） | 歧义任务命中两个真动作 **98.5%**（回归只输出 0）；倒立摆 **−331.8 vs −181.7**、慢 **132.8×** → 买到了分布，没买到分数 |
| W6 | REINFORCE + baseline + Return 归一化 | `src/reinforce.py`、`experiments/train_reinforce.py` | [`w6_task1_reinforce.md`](docs/reports/w6_task1_reinforce.md) | CartPole 解到 **500.00 @ep120**；`baseline=none` + 不归一化会崩到 **33.40** |
| W8 | PPO-Clip + GAE-λ（连续控制） | `src/ppo.py`、`experiments/train_ppo.py` | [`w8_task1_ppo.md`](docs/reports/w8_task1_ppo.md) | **194 560 步**达到 −199.91（达标线 −200），含 4 组消融 |
| W9 | SAC（off-policy 最大熵 RL） | `src/sac.py`、`experiments/train_sac.py` | [`w9_task1_sac.md`](docs/reports/w9_task1_sac.md) | **4 000 步**即 −173.70，比 PPO 省 ≈ **48×** 环境交互步数 |
| W10 | Isaac Lab 物理基座：64 路 Franka 纯力矩、零拷贝张量回路、SPS/显存/扩展性 | `experiments/w10_arm_torque_control.py`、`experiments/isaaclab_env.sh` | [`w10_task3_torque_control.md`](docs/reports/w10_task3_torque_control.md) | **389.5 SPS = 24 929 env-steps/s**；`data_ptr` 相同 ⇒ 零拷贝成立；瓶颈是 PhysX `sim.step`（73%），1→256 台只掉 3.7% |
| W11 | 把未知负载 / 非线性摩擦 / 质量失配**注入物理层**（域随机化） | `experiments/w11_dynamics_randomization.py` | [`w11_task1_dynamics_randomization.md`](docs/reports/w11_task1_dynamics_randomization.md) | "同力不同态" PASS：跨 env 关节 std **1e-6 → 7e-2 rad**（×3.2e4），注入一次性耗时 0.116 s |
| W12 | 轨迹跟踪 MDP（轨迹发生器 / 105 维时序观测 / 多目标奖励）+ 开环基线 | `experiments/w12_trajectory_tracking_env.py` | [`w12_task1_trajectory_tracking_env.md`](docs/reports/w12_task1_trajectory_tracking_env.md) | 前馈+PID 在失配下 **1.3259 cm**、纯 PD **2.3016 cm**、零力矩 **73.18 cm**；任务书那条力矩公式是**正反馈**（实测 1.80 rad） |
| W13/W14 | **双层残差控制**：标称前馈+PID 兜底 + SAC 学残差，端到端 45 万步 | `experiments/w13_residual_sac_train.py` | [`w13_task1_residual_sac.md`](docs/reports/w13_task1_residual_sac.md) | 关节 **0.051866 → 0.045955 rad**（×1.13）、末端 **1.3259 → 0.8651 cm**（×1.53），两条验收线都过；发现两个奖励结构问题 |
| W15 | **论文实验矩阵**：四控制器统一评测（Table 1）+ 载荷/摩擦/延迟三维极端压测（Figure 3）+ 消融（Table 2） | `experiments/w15_paper_benchmarks.py` | [`w15_task1_paper_benchmarks.md`](docs/reports/w15_task1_paper_benchmarks.md) | 同预算下：末端 **PD 2.3016 / 前馈+PD 1.3259 / 纯 RL 86.43 / 残差 SAC 0.8612 cm**（Ours −35.0%，饱和 0.00%）；纯 RL 同一预算直接失败（饱和 95.8%）；压测轴 Ours 全程最优 |
| W16 | 论文级矢量图（Figure 1–3）+ LaTeX 大表（Table 1/2）自动生成 | `experiments/export_paper_figures.py` | 同上 | `docs/figures/fig{1,2,3}_*.pdf|png`、`docs/results/w15_tables.tex` |

---

## 5. 主线结果速览（W13/W14：双层残差 SAC 端到端）

**规范运行**：64 env × 500 步（100 Hz）、`seed = 42`、W11 域随机化开、45 万环境步（≈ 10 min，单卡 2080 Ti）。
完整报告：[`docs/reports/w13_task1_residual_sac.md`](docs/reports/w13_task1_residual_sac.md)；
曲线：[`docs/figures/w13_residual_sac_ee50k_curves.png`](docs/figures/w13_residual_sac_ee50k_curves.png)。

| 控制器 | 关节 RMSE [rad] | 末端笛卡尔 RMSE [cm] |
| --- | --- | --- |
| `pd_dr`（纯 PD + 失配） | 0.060948 | 2.3016 |
| `pd_ff_dr`（★ 基座：标称前馈 + PID + 失配） | 0.051866 | 1.3259 |
| **残差 SAC（45 万步，最终评估）** | **0.045955**（×1.13，↓11.4%） | **0.8651**（×1.53，↓34.7%） |
| 残差 SAC（最好评估点 @400k，权重 `policy_best_ee.pt`） | 0.045967 | **0.8612** |
| 残差 SAC（后 8 次评估均值） | 0.047909 | 0.9240（波动 0.86–1.02） |
| `zero_dr`（零力矩上界） | 0.972252 | 73.1780 |

* **验收**：关节 < 0.0519 rad ✅、末端 < 1.0 cm ✅（冲刺线 0.5 cm ❌）；首次同时达标 @ **150 144** 环境步。
* **成本**：450 048 环境步 / 13 438 次更新（UTD≈2:1）/ 527 s（**854 env-steps/s**）；显存 peak **0.277 GiB**；
  残差幅值 `|Δτ| = 1.03 N·m/关节`（硬上限 1.5）、力矩饱和步占比 **0.000**（指标没被限幅污染）。

训练里最值得读的是**两个奖励结构问题**（报告 §6 给了可复现开关与实测数据）：

1. `‖τ_exec‖²` 项在残差结构里有**激励漏洞**——它奖励"用 Δτ 抵消基座力矩"
   （`w_mag=0` 时 `|Δτ|` 漂到 3.5 N·m、末端 1.33 → 2.6 cm）⇒ 默认补 `w_mag‖Δτ‖²`；
2. 关节空间的各向同性奖励与"末端位置"验收指标**不同构**（j7 独占误差平方和 ~57% 却几乎不移动手部原点）
   ⇒ 主结果把 `w_ee‖p(q)−p(q_d)‖²` 写进奖励。

> **数据版本说明（重要）**：同一条命令重复运行**不是逐位一致**的——25k/50k 两次评估逐位相同，75k 起分叉。
> 本仓库落盘的 JSON / CSV / 图是**后一次**运行（关节 0.045955 rad、末端 0.8651 cm；第一次是 0.043342 / 0.9107），
> 两次都过验收，差异来自 GPU kernel 的非确定性（脚本自带 `--deterministic` 开关，默认关）。
> 逐项对账链（W12 对账、零残差等价性、obs 环自检、FK 对账、权重读回…）在两次运行里都是 `0.00e+00`。

---

## 6. 结果与证据放在哪

| 想找什么 | 位置 | 说明 |
| --- | --- | --- |
| 实验报告（数据表 / 消融 / 排错 / 结论） | `docs/reports/*.md` | 每周一份；报告里的数字都能在下面几项里对账 |
| 结构化结果 | `docs/results/*.json` | 每次规范运行的完整落盘（配置 / 指标 / 奖励分项 / 对账 / 判决） |
| 逐次评估表 | `docs/results/*_history.csv` | 每个评估点的关节/末端 RMSE、奖励分项、`|Δτ|`、α、`best_ee_cm_so_far` |
| 训练日志（纯文本，可 grep） | `docs/results/*.log` | 已剔除 Isaac Sim / PhysX / Warp 启动噪声；未删减原件在 `logs/raw_console_logs/`（不入库） |
| 曲线 + 原始标量 | `docs/figures/*.png`、`*.csv` | PNG 多面板图；CSV 长表 `tag,step,value`（`experiments/export_tb_curves.py` 导出） |
| 重跑产生的原始输出 | `logs/`（已 gitignore） | checkpoint（含 `policy_best_ee.pt`）、TB 事件、逐次运行日志 |

---

## 7. 什么不进版本库（`.gitignore`）

| 忽略项 | 原因 |
| --- | --- |
| `logs/` | checkpoint / TensorBoard 事件 / 原始控制台日志；跑一次实验即重建（要入库的那份日志已瘦身放在 `docs/results/`） |
| `*.pt` / `*.pth` | 权重（`policy_final.pt`、`policy_best_ee.pt` 各 ≈1.9 MB）；用 §3.3 的命令重跑即可得到 |
| `data/expert_pendulum/`、`data/cifar-10*/`、`w2_pytorch_basics/data/` | 专家数据 npz 与 CIFAR 缓存，由脚本采集 / 下载 |
| `__pycache__/`、`*.py[cod]`、`.pytest_cache/` | Python 缓存 |
| `*~`、`*.bak`、`.DS_Store` | 编辑器 / 系统残留 |

---

## 8. 路线图与硬件

- [x] W2–W5：深度学习基础（PyTorch 从零：拟合 / CNN / 序列预测 / 行为克隆 / 动作块 / 扩散策略）
- [x] W6–W9：强化学习从零（REINFORCE / PPO / SAC，全部手写 + 单元测试）
- [x] W10–W12：Isaac Lab 物理基座（纯力矩零拷贝回路 / 物理层域随机化 / 轨迹跟踪 MDP）
- [x] W13–W14：主课题第一步 —— 双层残差 SAC 端到端（两条验收线都过）
- [ ] W15+：更大预算（600k–1M 步）、多 seed、`w_ee` 课程、与验收指标同构的动作缩放
- [ ] 2027.1–2：真机部署
- [ ] 2027.3：论文投稿（IROS / RA-L）

**硬件 / 环境**：RTX 2080 Ti（11 GB）/ Ubuntu 22.04；纯 PyTorch 侧 Python 3.10，仿真侧 Python 3.12 + Isaac Sim 6.1 + Isaac Lab 3.0。
**约定**：所有算法**手写**（不使用 SB3、rlkit 等 RL 库）；单元测试只用标准库（不需要 pytest）。
