# embodied-arm —— 具身智能 / 机械臂强化学习学习工程

从零手写深度学习与强化学习算法（**行为克隆 → 动作块 → 扩散策略 → REINFORCE → PPO → SAC**），
全部在 `Pendulum-v1` / `CartPole-v1` 上跑通并留下可复现的数据、曲线与实验报告。
最终目标：2027 年 IROS / RA-L 投稿的「**残差 RL 机械臂轨迹跟踪**」课题（见文末路线图）。

| 想干什么 | 去哪看 |
| --- | --- |
| 完全没基础，先搞懂"这些实验到底在干什么" | [`docs/零基础读本.md`](docs/零基础读本.md)（白话讲清动作块、日志指标、FAQ） |
| 看每周的实验数据、消融与结论 | [`docs/reports/`](docs/reports/) |
| 看训练曲线 | [`docs/figures/`](docs/figures/) |
| 英文工程说明 | [`README.en.md`](README.en.md) |

---

## 1. 目录结构

```
embodied-arm/
├── README.md                 # 本文件：唯一中文入口
├── README.en.md              # 英文工程说明（配置项 / 续训机制 / 命令行）
├── requirements.txt          # 依赖清单
│
├── configs/                  # 所有超参数（改实验不用改代码）
│   ├── bc_config.yaml            # W3   单步行为克隆（BC）基线
│   ├── chunking_bc_config.yaml   # W4-1 动作块 BC（defaults 继承 bc_config）
│   ├── diffusion_config.yaml     # W4-2 迷你条件扩散策略
│   ├── reinforce_config.yaml     # W6   REINFORCE（CartPole）
│   ├── ppo_config.yaml           # W8   PPO-Clip + GAE-λ（Pendulum）
│   └── sac_config.yaml           # W9   SAC（最大熵 off-policy，Pendulum）
│
├── src/                      # 核心实现（模块名 = 算法名，被 tests/ 直接导入）
│   ├── networks.py               # 策略网络 MLP / CNN / GRU + 动作块输出头
│   ├── datasets.py               # 专家控制器、数据采集与缓存、动作块切分、闭环评估
│   ├── diffusion.py              # DDPM 调度表、噪声预测网络、DiffusionPolicy（DDPM/DDIM）
│   ├── engine.py                 # 通用训练引擎（参数/配置/日志/TB/断点续训，BC 系列共用）
│   ├── reinforce.py              # W6  REINFORCE + 三种 baseline + Return 归一化
│   ├── ppo.py                    # W8  PPO-Clip、GAE-λ、RolloutBuffer、观测归一化
│   ├── sac.py                    # W9  SAC：双重 Q、压缩高斯 Actor、自适应温度 α
│   └── utils.py                  # 配置合并、日志器、TensorBoard、Checkpointer
│
├── experiments/              # 训练 / 评估入口，每个都能直接 python 运行
│   ├── train_bc.py               # W3   BC
│   ├── train_chunking_bc.py      # W4-1 动作块 BC（--set model.chunk_size=K）
│   ├── train_diffusion_policy.py # W4-2 扩散策略 + 确定性对照 + 探针 / 消融
│   ├── train_reinforce.py        # W6   REINFORCE
│   ├── train_ppo.py              # W8   PPO
│   ├── train_sac.py              # W9   SAC
│   ├── eval_expert.py            # 专家 oracle 基线（给"回报"提供参照系）
│   └── export_tb_curves.py       # 把 TensorBoard 事件导出成 PNG + CSV
│
├── tests/                    # 单元测试（纯标准库，不需要 pytest，合计 148 项）
│   ├── test_action_chunking.py   # 18 项
│   ├── test_diffusion.py         # 15 项
│   ├── test_reinforce.py         # 29 项
│   ├── test_ppo.py               # 38 项
│   └── test_sac.py               # 48 项
│
├── docs/                     # 文档：报告 / 读本 / 笔记 / 曲线 / 原始日志
│   ├── 零基础读本.md             # W3/W4 白话读本（动作块、指标解读、FAQ、文件地图）
│   ├── reports/                  # ★ 每周实验报告（数据表、消融、结论）
│   ├── notes/                    # W2 深度学习学习笔记（CNN / AlexNet）
│   ├── figures/                  # 归档曲线 PNG + 原始标量 CSV（w8_ppo_* / w9_sac_*）
│   └── results/                  # 归档训练文本日志 + W4 扩散实验的 json 报告
│
├── w2_pytorch_basics/        # W2 PyTorch 基础练习（单文件小脚本）
├── data/                     # 数据：专家数据 npz、CIFAR-10 缓存（已 gitignore）
└── logs/                     # 运行输出：TensorBoard 事件 + checkpoint（已 gitignore，可随时删）
```

> `logs/` 与 `data/` 下的数据**不提交到 git**：跑一次实验就会自动重建。
> 报告里引用的关键数字，已把对应的**文本训练日志**归档在 `docs/results/`。

---

## 2. 环境

实测环境：Ubuntu 22.04 + Python 3.10 + PyTorch 2.6.0+cu124 + gymnasium 1.3.0，RTX 2080 Ti（11 GB）。
本仓库使用 conda 环境 `rl`：

```bash
conda create -n rl python=3.10 -y && conda activate rl
pip install -r requirements.txt        # torch / numpy / gymnasium / pyyaml / tensorboard / matplotlib
```

没有显卡也能跑：所有训练脚本都支持 `--device cpu`（只是慢一些）。

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

全部输出 `通过 N/N` 即为正常（无需安装 pytest）。

### 3.2 再按周复现实验

| 周次 | 任务 | 一条命令 | 大致耗时 | 主要产物 |
| --- | --- | --- | --- | --- |
| W3 | 单步 BC 基线 | `python experiments/eval_expert.py`<br>`python experiments/train_bc.py` | 4 s<br>≈ 90 s | 专家回报标尺 + BC 策略 |
| W4-1 | 动作块 BC（K=4 / K=8） | `python experiments/train_chunking_bc.py`<br>`python experiments/train_chunking_bc.py --set model.chunk_size=4` | ≈ 90 s / 次 | 闭环回报 + 前向次数对比表 |
| W4-2 | 迷你扩散策略 + 消融 | `python -u experiments/train_diffusion_policy.py` | ≈ 12 min | `diffusion_report.json` |
| W6 | REINFORCE（CartPole） | `python experiments/train_reinforce.py` | ≈ 2 min | 解 500 分 + 消融表 |
| W8 | PPO（Pendulum） | `python experiments/train_ppo.py` | ≈ 8.8 min | 194 560 步达 −199.91 |
| W9 | SAC（Pendulum） | `python experiments/train_sac.py` | ≈ 45 s（4 000 步提前达标） | 4 000 步达 −173.70 |

```bash
# 常用开关（所有训练脚本通用）
python experiments/train_sac.py --device cpu          # 强制 CPU
python experiments/train_sac.py --no-tb --no-eval     # 冒烟测试：不写 TensorBoard、不评估
python experiments/train_sac.py --resume auto         # 断点续训（自动找最新 checkpoint）
python experiments/train_sac.py --set training.total_timesteps=40000   # 命令行覆盖任意超参

# 重新导出曲线（先跑过实验才有 logs/）
python experiments/export_tb_curves.py --logdir logs --out-dir docs/figures --prefix w9_sac
# → docs/figures/w9_sac_curves.png（多面板曲线）+ w9_sac_scalars.csv（长表 tag,step,value）
```

---

## 4. 每周进展一览（W2 → W9）

| 周次 | 任务 | 核心代码 | 报告 | 一句话结论 |
| --- | --- | --- | --- | --- |
| W2 | PyTorch 基础：线性拟合、MLP 拟合 sin、CNN、序列预测 | `w2_pytorch_basics/` | `docs/notes/` | 手写训练循环 / 优化器 / 卷积网络 |
| W3 | 行为克隆工程化：配置、日志、TensorBoard、断点续训 | `experiments/train_bc.py`、`src/{networks,datasets,engine,utils}.py` | [`docs/零基础读本.md`](docs/零基础读本.md) | 单步 BC 闭环 **−156.95 ± 68.99**（专家 oracle −175.00） |
| W4-1 | 动作块 BC（Action Chunking） | `experiments/train_chunking_bc.py`、`configs/chunking_bc_config.yaml` | [`w4_task1_action_chunking.md`](docs/reports/w4_task1_action_chunking.md) | K=4 开环 **−155.58**，每回合前向 200 → **50**；K=8 开环反而退化到 −219.68 |
| W4-2 | 迷你条件扩散策略 vs 确定性回归 | `src/diffusion.py`、`experiments/train_diffusion_policy.py` | [`w4_task2_diffusion_policy.md`](docs/reports/w4_task2_diffusion_policy.md)（[零基础版](docs/reports/w4_task2_diffusion_policy_零基础版.md)） | 歧义任务上扩散命中两个真动作 **98.5%**（回归只输出 0）；倒立摆上 **−331.8 vs −181.7**、慢 **132.8×** → 买到了分布，没买到分数 |
| W6 | REINFORCE + baseline + Return 归一化 | `src/reinforce.py`、`experiments/train_reinforce.py` | [`w6_task1_reinforce.md`](docs/reports/w6_task1_reinforce.md) | CartPole 解到 **500.00 @ep120**；`baseline=none` + 不归一化会崩到 **33.40** |
| W8 | PPO-Clip + GAE-λ（连续控制） | `src/ppo.py`、`experiments/train_ppo.py` | [`w8_task1_ppo.md`](docs/reports/w8_task1_ppo.md) | **194 560 步**达到 −199.91（达标线 −200），含 4 组消融 |
| W9 | SAC（off-policy 最大熵 RL） | `src/sac.py`、`experiments/train_sac.py` | [`w9_task1_sac.md`](docs/reports/w9_task1_sac.md) | **4 000 步**即 −173.70，比 PPO 省 ≈ **48×** 环境交互步数 |

---

## 5. W9 SAC 结果速览（最新一周）

实现要点：经验回放池 + 双重 Q（Double-Q 取 min）+ 重参数化压缩高斯 Actor
+ 自适应温度 α（目标熵 `H̄ = −dim(A) = −1`）+ 目标网络 Polyak 软更新。

| 指标 | 结果 |
| --- | --- |
| 首次突破 −200（`eval/first_solved_step`） | **4 000 步**，评估回报 **−173.70**（耗时 44.7 s） |
| 与 W8 PPO 对比 | PPO 需 194 560 步 ⇒ SAC 少用约 **48×** 环境交互 |
| 40 k 步完整训练的最优评估 | **−129.71 ± 120.29**（第 22 k 步；20 回合、固定种子、均值动作评估） |
| 温度 α 的自动轨迹 | 1.0 → **0.0090**，平衡点即 `E[log π] → dim(A) = +1` |
| 正确性在线断言 | `train/critic_frozen_ok` 全程 **1.0**（Actor 更新时 Critic 确实没拿到梯度） |
| 消融（4 组） | 固定 α / 不做 truncation bootstrap / 单 Q / Actor 只用 Q₁ |
| 单元测试 | `tests/test_sac.py` **48/48 通过** |

```bash
python experiments/train_sac.py                                                                    # 默认配置（4000 步提前达标）
python experiments/train_sac.py --set experiment.name=sac_pendulum_full training.target_reward=1e9 # 跑满 40k
python experiments/train_sac.py --set experiment.name=sac_abl_single_q model.n_critics=1           # 消融示例
```

公式推导、两道思考题、逐项指标解读与消融表见 [`docs/reports/w9_task1_sac.md`](docs/reports/w9_task1_sac.md)。

---

## 6. 结果与证据放在哪

| 想找什么 | 位置 | 说明 |
| --- | --- | --- |
| 实验报告（数据表 / 消融 / 结论） | `docs/reports/*.md` | 每周一份，报告里的数字都可在下面几项中对账 |
| 曲线图 + 原始标量 | `docs/figures/*.png`、`*.csv` | 由 `experiments/export_tb_curves.py` 从 TB 事件导出；CSV 为长表 `tag,step,value` |
| 训练文本日志 | `docs/results/*.log` | 归档的各次正式运行 `train.log`（含逐次评估回报），纯文本、可 grep |
| 结构化结果 | `docs/results/w4_diffusion_policy_report.json` | W4-2 扩散实验完整 json（探针 / 消融 / 时延） |
| 重新训练产生的原始输出 | `logs/`（已 gitignore） | TensorBoard 事件 + checkpoint；占空间、可随时 `rm -rf logs` 重跑 |

---

## 7. 路线图

- [x] W2–W5：深度学习基础（PyTorch 从零）
- [x] W6–W9：强化学习从零（REINFORCE / PPO / SAC）
- [ ] W10–W12：Isaac Lab 环境与自定义任务
- [ ] W13–W16：主课题 —— 残差 RL 实现鲁棒的机械臂轨迹跟踪
- [ ] 2027.1–2：真机部署
- [ ] 2027.3：论文投稿

## 8. 硬件与环境

- RTX 2080 Ti（11 GB）/ Ubuntu 22.04 / Python 3.10 / PyTorch 2.6（cu124）
- 约定：所有算法**手写**，不使用 SB3、rlkit 等 RL 库；单元测试只用标准库（不需要 pytest）
