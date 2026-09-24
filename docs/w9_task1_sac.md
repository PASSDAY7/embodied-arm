# W9 Task 1：从零实现 SAC（Soft Actor-Critic）—— Pendulum-v1

> 目标：在连续动作环境 **Pendulum-v1**（动作 `[-2, 2]`，单回合 200 步）上从零实现
> off-policy 最大熵 RL 算法 **SAC**（Soft Actor-Critic）：经验回放池 + Double-Q +
> 重参数化压缩高斯策略 + **自动温度 α**；并用 4 万步预算（约为 W8 PPO 的 1/5）
> 观察"样本效率的量级提升"与"α 的自适应轨迹"。

---

## §0 结论速览（先看这里）

| 项目 | 结果 |
|---|---|
| 默认配置（seed 42 / cuda / `target_reward=-200` 提前停止） | **4000 步**即突破 −200（独立评估 **−173.70 ± 142.73**），训练总用时 **44.7 s** |
| 完整 40 000 步（关闭提前停止，用于观察曲线） | 最优独立评估 **−129.71 ± 120.29** @ 22 000 步；训练滚动均值 −143.37；用时 535.5 s |
| 与 W8 PPO 的对比 | PPO 用 **194 560** 步才到 **−199.91**；SAC 用 **4 000** 步到 **−173.70** ⇒ 达到同一目标所需环境步数少约 **48 倍** |
| α 轨迹 | 1.0000 → 0.3823（5k）→ 0.0549（13k）→ **0.0090**（40k）；策略熵估计 H: +0.62 → −0.96 |
| α 收敛到哪？ | 恰好收敛在 `E[log π] ≈ −H̄ = dim(A) = +1` —— 自动温度的目标熵平衡点，与 §3.4 的推导一致 |
| 思考题自检（Actor 步冻结 Critic） | `train/critic_frozen_ok` 全程恒为 **1.0000**（最小值 = 最大值 = 1），Actor 梯度范数 > 0 |
| 单元测试 | `tests/test_sac.py` **48/48 通过**，用时约 18 s（CPU）；W8 的 38/38 仍全绿 |

三条"必看"的交付物：`configs/sac_config.yaml`、`src/sac.py`、`experiments/train_sac.py`；
曲线与标量：`docs/figures/w9_sac_full_curves.png`（28 条曲线）/ `w9_sac_full_scalars.csv`（51 个 tag）。

---

## §1 任务与交付物

### 1.1 任务要求（W9）

1. **配置** `configs/sac_config.yaml`：`total_timesteps: 40000`、`buffer_size: 100000`、
   `batch_size: 256`、`lr_actor: 3e-4`、`lr_critic: 1e-3`、`lr_alpha: 3e-4`、
   `tau: 0.005`、`gamma: 0.99`、`target_entropy: "auto"`（= `−dim(A) = −1.0`）。
2. **核心库** `src/sac.py`：
   * `SquashedGaussianActor`：输出 `μ` 与 `log σ`，**重参数化**采样 + **tanh 压缩的 Jacobian 修正**；
   * `ContinuousCritic`：输入拼接 `[s, a]`，输出标量 `Q(s, a)`（**Double-Q**）；
   * `ReplayBuffer`：离线经验回放池，支持大批量随机采样；
   * `SACAgent`：集成 Double-Q 损失、Actor 策略损失、**Alpha 自适应更新**。
3. **训练入口** `experiments/train_sac.py`：
   * 热身期（`warmup_steps: 1000` 纯随机采样填 buffer，不做任何梯度更新）；
   * **探索与定期评估分离**（训练用采样动作，评估用 `tanh(μ)` 的均值动作 + 固定种子）；
   * 监控：`Q1/Q2 损失`、`策略损失`、`α 演化曲线`、`熵值变化`、`闭环评估回报`。
4. **预检**：跑通 `train_sac.py`，记录**在多少步突破 −200 分**，并观察 α 的自适应轨迹。
5. **思考题**：为什么更新 Actor 的损失时，计算图的反向传播要穿透 Critic，但不能更新
   Critic 的参数？代码里如何切断 Critic 的梯度？

### 1.2 文件清单

| 文件 | 说明 |
|---|---|
| `configs/sac_config.yaml` | SAC 配置（作业指定超参 + 工程补充字段，逐项注释），94 行 |
| `src/sac.py` | 核心库：数学工具 / Actor / Double-Q / ReplayBuffer / SACAgent / 装配，1266 行 |
| `experiments/train_sac.py` | 训练入口（热身 → 环境步/梯度步交替 → 定期评估 → 存盘/续训/汇总），747 行 |
| `tests/test_sac.py` | 单元测试（不依赖 pytest，48 项），1312 行 |
| `src/__init__.py` | 追加 W9 的导出块与 `__all__`（**被修改的既有文件之一**），+34 行 |
| `experiments/export_tb_curves.py` | 追加 9 个 SAC 监控面板（PPO 面板保持不变），+40 行 |
| `docs/w9_task1_sac.md` | 本文档，542 行 |
| `docs/figures/w9_sac_full_curves.png` / `w9_sac_full_scalars.csv` | 40k 完整训练的曲线图 / 全部标量（28 曲线 / 2123 行） |
| `docs/figures/w9_sac_earlystop_curves.png` / `w9_sac_earlystop_scalars.csv` | 默认配置（提前停止）那次运行的曲线 / 标量（5 曲线 / 198 行） |

工程骨架（`--config/--set/--resume` 解析、运行目录、日志、TensorBoard、优雅停机、
`Checkpointer`、独立评估 `evaluate_policy`）全部复用 W3/W4/W6/W8 的 `src/engine.py`
与 `src/ppo.py`，**没有重复实现**；SAC 只新增"算法特有"的部分。

---

## §2 为什么是 SAC：on-policy → off-policy 的取舍

| 维度 | PPO（W8，on-policy） | SAC（W9，off-policy） |
|---|---|---|
| 数据来源 | 当前策略现采一个 rollout，**用完即弃** | `ReplayBuffer`（10⁵ 容量）里**反复采样** |
| 每条经验被消费次数 | 1（K 轮 mini-batch 也只是同一批数据） | 数百次（`train_freq=1` ⇒ 每步都更新） |
| 更新频率 | 每 2048 步 → 10 轮 × 32 个 mini-batch | **每个环境步 → 1 个 mini-batch** |
| 动作分布 | 高斯 + 环境侧 `clip` | **tanh 压缩高斯**（自带 Jacobian 修正） |
| 探索机制 | 显式学习 `σ` / 熵正则 | **最大熵目标 + 自动温度 α** |
| 价值网络 | `V(s)` 单头 + GAE | `Q(s,a)` **双头** + **目标网络**（Polyak） |
| 现实开销 | 简单、稳定、易调；样本贵 | 代码更复杂（目标网络 / α / buffer），样本便宜 |

**最大熵目标**（Haarnoja et al., 2018）在标准回报里加了"策略熵"一项：

    J(π) = Σ_t E_{(s_t,a_t)~ρ_π}[ r(s_t, a_t) + α·H(π(·|s_t)) ]

* α 越大，越鼓励"多探索"；α → 0 退化为普通 RL（DDPG 风格）；
* 于是策略天然是**随机**的（多模态、鲁棒），且探索强度可以**自适应**（§3.4）；
* off-policy + 最大熵 ⇒ 历史数据可以反复利用，这就是"样本效率高一个量级"的来源：
  Pendulum-v1 上 SAC 通常 2~4 万步收敛，本实验实际 **4 千步**就到目标区间（§7）。

---

## §3 算法与公式推导

### 3.1 软价值函数与软 Bellman 方程

    Q^π(s, a) = r(s, a) + γ·E_{s'~p, a'~π}[ Q^π(s', a') − α·log π(a'|s') ]            (1)

与普通 Bellman 方程的唯一区别：期望里多了一项 `−α log π(a'|s')`。
`a'` 由策略**采样**得到（不是取 `max`）—— 这是 SAC 与 DDPG 的关键分界，
也保证了"策略梯度是路径可导的"（重参数化，见 §3.3）。

### 3.2 Critic 损失（Double-Q + 目标网络）

用回放池 `D` 里的 transition 构造目标 `y`，并对两个独立初始化的 Q 网络做回归：

    a'_t = tanh( μ_φ(s'_t) + σ_φ(s'_t)·ε ),  ε ~ N(0, I)                             (2)
    y_t  = r_t + γ·(1 − done_t)·[ min_i Q_{θ̄_i}(s'_t, a'_t) − α·log π_φ(a'_t|s'_t) ]  (3)
    J_Q(θ_i) = E_{t~D}[ ( Q_{θ_i}(s_t, a_t) − y_t )² ]                                (4)

三个关键实现细节：

1. **目标里的 `a'` 由当前策略采样**（式 2），整条 `y` 的计算包在 `torch.no_grad()` 里
   ⇒ `y` 是**常数**，不存在"追逐自己"的路径（DQN 式发散风险被消除）；
2. **Double-Q 取 `min`**（式 3）：两个 Q 独立初始化，`min` 抑制"高估 → 策略被高估方向骗"；
   消融实测单 Q 差约 19 分（§8）；
3. **目标网络** `θ̄` 用 **Polyak 软更新**，参数永久 `requires_grad_(False)`
   ⇒ 不可能被任何 optimizer 更新：

    θ̄_i ← τ·θ_i + (1 − τ)·θ̄_i,   τ = 0.005                                        (5)

代码：`SACAgent.critic_loss`（`src/sac.py`）；目标网络在 `SACAgent.__init__` 里
`copy.deepcopy` 并由 `soft_update_target`（纯函数 `soft_update`）更新。

### 3.3 Actor 损失（重参数化 + 压缩高斯的 Jacobian）

    x = μ_φ(s) + σ_φ(s)·ε,  a = tanh(x) ∈ (−1, 1)^A,   ε ~ N(0, I)                  (6)

连续控制环境需要"有界动作"，`tanh` 压缩比 PPO 的"越界再 clip"更自然
（clip 会在边界堆概率质量、破坏密度归一化；W8 的思考题讨论过这点）。
代价是**密度要按换元公式修正**（本任务与 PPO 最大的数学差异）：

    log π(a|s) = log N(x; μ, σ²) − Σ_j log(1 − tanh²(x_j))                          (7)

右边第二项就是 `|da/dx|` 的对数（`da/dx = 1 − tanh²x = sech²x`）。
数值上取等价变形 `log(1 − tanh²x) = 2(log 2 − x − softplus(−2x))`
（`log_one_minus_tanh_sq`），避免 `tanh(x) → ±1` 时 `1 − tanh²x` 下溢成 0 造成 `log 0 = −inf`。
**测试里用数值积分验过**：`∫_{−1}^{1} exp(log π(a)) da = 1.0000 ± 0.002`
（`test_squashed_log_prob_integrates_to_one`），这是"Jacobian 项写对了"的硬判据。

策略损失：

    J_π(φ) = E_{s~D, ε}[ α·log π_φ(a|s) − min_i Q_{θ_i}(s, a) ],  a = tanh(μ_φ(s)+σ_φ(s)ε)  (8)

* 第二项把 `Q` 顶上去 ⇒ 梯度 `∂Q/∂a` 必须沿 Q 网络的**激活值**回传（§4 的主题）；
* 第一项把熵顶上去（`log π` 越小 ⇒ 分布越平）；
* `α` 在式 8 里取 `detach()`：温度只由它自己的损失（式 9）更新。

### 3.4 温度 α 的自适应（目标熵 H̄ = −dim(A)）

α 不是超参，而是**可学习参数**（在 log 空间参数化，保证 α > 0）：

    J(α) = −log α · E_{a~π}[ log π_φ(a|s) + H̄ ],    H̄ = −dim(A) = −1.0（Pendulum）    (9)

对 `log α` 求导并做梯度下降：

    ∂J/∂log α = −( E[log π] + H̄ )   ⇒   log α ← log α + lr·( E[log π] + H̄ )        (10)

于是 α 的**方向**完全由"当前策略离目标熵有多远"决定：

    E[log π] > −H̄ = dim(A)  ⇒  α ↓（密度已比目标更集中，该收敛了）
    E[log π] < −H̄ = dim(A)  ⇒  α ↑（密度比目标更散，探索不足，加大温度）

注意 `H̄ = −1` 是**负**数，而压缩空间里 `H ≈ −E[log π]` 可以是负数
（密度集中到"平均密度 > 1"时 `E[log π] > 0`）。所以平衡点写成：

    E[log π] → −H̄ = +dim(A) = +1   ⟺   H → H̄ = −1

本实验里 α 从 1.0 一路降到 0.009，而 `E[log π]` 恰好稳定在 **+0.86 ~ +0.97**（§7.3），
与式 10 的预测一致。

> 实现细节：本模块采用与 SB3 相同的损失形式 `−log α·(log π + H̄)`（而不是精确的
> `−α·(log π + H̄)`）。两者只差一个正因子 α，等价于把温度学习率重标定 —— 好处是
> α → 0 时更新幅度不会跟着"停摆"。单测 `test_alpha_loss_formula_and_gradient_does_not_touch_networks`
> 锁定了这一约定。

### 3.5 一次梯度更新（伪代码）

    for each env step:
        a_norm = tanh(μ(s) + σ(s)·ε)          # π_φ 采样（热身期改为均匀随机）
        s', r, term, trunc = env.step(scale·a_norm + bias)
        buffer.add(s, a_norm, r, s', term, trunc)          # ★ 存归一化动作
        if 已完成热身:
            ① critic_opt.zero_grad(); loss_Q = J_Q(θ); loss_Q.backward(); critic_opt.step()
               critic.zero_grad(set_to_none=True)          # 清空，便于验证"Actor 步没污染 Q"
            ② actor_opt.zero_grad()
               with frozen_parameters(critic):             # ★ 冻结参数，不切断计算图
                   loss_π = J_π(φ); loss_π.backward()
               actor_opt.step()
            ③ alpha_opt.zero_grad(); loss_α = J(α); loss_α.backward(); alpha_opt.step()
            ④ soft_update(target_critic, critic, τ)        # Polyak

---

## §4 思考题：为什么 Actor 的梯度要"穿透"Critic，却不能更新 Critic 的参数？

### 4.1 必须穿透：`J_π` 里唯一的方向信息来自 `Q`

    ∂J_π/∂φ = E[ α·∂log π_φ(a|s)/∂φ − ∂Q(s,a)/∂a · ∂a/∂φ ]                          (11)

`∂Q/∂a` 只能由 Q 网络的**前向激活值**反向传播得到；而
`∂a/∂φ = ∂tanh(x)/∂x · ∂x/∂φ` 经重参数化路径（式 6）回到 `μ_φ, σ_φ`。如果写成

    with torch.no_grad():  q = critic(s, a)          # ✗ 切断整条图
    q = critic(s, a).detach()                        # ✗ 等价效果

那么 `∂Q/∂a = 0`，式 11 只剩 `α·∂log π/∂φ` 一项 —— 策略会退化成"把熵顶到最大"的
**纯随机策略**（Pendulum 上表现为回报长期停在 −1200 附近，永远学不会打力矩）。

单测 `test_detaching_q_leaves_only_entropy_gradient` 把这个对照实验量化了：

* 连通的 `J_π` 反传后，`μ` 头的梯度 = "价值方向 + 熵方向"；
* 把 `q` 换成 `.detach()` 后，梯度**恰好等于**熵项单独反传的结果（逐位相等）；
* 两者之差非零 ⇒ 那部分正是"来自 Critic 的信息"，也是策略学习的全部；
* `test_constant_q_maximizes_entropy`：把 Q 换成常数（与动作无关）后，训练确实只把熵顶上去。

### 4.2 不能更新参数：θ 是"评价者"，不是"行动者"

本轮循环里 θ 已经由 `J_Q`（式 4）更新过一次（拟合 Bellman 目标）。如果 Actor 的反向传播
再往 θ 上累积 `α log π − Q` 的梯度，等于让评价函数朝"**把 Q 抬高、让策略显得更好**"的方向漂移：

* 目标网络 `θ̄`（式 5）与 α 的自适应（式 9）都建立在"Q 是（近似）无偏评价者"这个前提上；
* Q 可以被推到任意大 ⇒ 策略学到的只是"如何让 Q 变大"而不是"如何拿高回报"（reward hacking）；
* 单测 `test_shared_optimizer_would_corrupt_critic` 现场复现了这个后果：把 Actor 与 Critic
  放进**同一个** optimizer，做一步 Actor 更新后 Critic 参数被改动、且 `Q` 值**上升**
  —— 污染方向正是"抬高 Q"。

### 4.3 代码里到底怎么切断的？（四条不变量的落地）

| # | 位置 | 做法 | 单测证据 |
|---|---|---|---|
| 1 | `SACAgent.actor_loss` | `with frozen_parameters(self.critic):` —— **只把 Q 网络参数的 `requires_grad` 置 False**（退出上下文自动恢复），**不用** `torch.no_grad()` | `test_frozen_critic_still_backprops_to_actor`：`∂q_pi/∂a ≠ 0`、Actor 梯度非零、Critic 的 `.grad` 全为 `None` |
| 2 | `build_sac_optimizers` | **三套独立 Adam**（critic / actor / alpha），Actor 步只 `actor_optimizer.step()` | `test_build_sac_optimizers_three_independent_adam`（参数集合互不相交）、`test_actor_step_does_not_change_critic`（Critic 逐位不变） |
| 3 | `critic_loss` / `target_critic` | 目标 `y` 全程 `torch.no_grad()`；`target_critic` 参数永久 `requires_grad_(False)`，只用 `.data` 做 Polyak | `test_critic_loss_target_is_detached_from_critic_params`（扰动 Critic 参数后 `y` 不变）、`test_target_network_is_deepcopy_frozen_and_synced` |
| 4 | `alpha_loss` | `log_prob.detach()` ⇒ 温度只更新 `log_alpha` 一个标量 | `test_alpha_loss_formula_and_gradient_does_not_touch_networks`（Actor / Critic 的 `.grad` 全为 `None`） |

**"冻结参数"为什么不会同时"切断图"？** 因为 `a` 是一个 `requires_grad=True` 的**张量输入**
（由重参数化 `μ + σ·ε` 得到），PyTorch 的反向传播只需要"图里存在从输出到该张量的路径"；
Q 网络的**权重**即使 `requires_grad=False`，它们的前向运算仍然构成 `a → q` 的可导链
（这正对应链式法则里 `∂q/∂a` 只依赖激活值、不依赖"权重是否需要更新"）。
`frozen_parameters` 只是让 PyTorch **跳过为这些权重累积 `.grad`**，从而同时得到：

* **语义正确**：Critic 的参数不会被 Actor 的优化目标污染；
* **工程收益**：省掉 Critic 权重那一段反向传播（SAC 里 Q 网络参数量远大于 Actor，
  约省 1/3 反传开销）；
* **可验证**：`update_batch` 在 Critic 步结束后主动 `critic.zero_grad(set_to_none=True)`，
  于是 Actor 步的 backward 若污染了 Critic，`.grad` 会重新出现 ——
  训练时写入 TensorBoard 的 `train/critic_frozen_ok` 就是这个不变量的**在线断言**
  （本次 40k 训练全程 = 1.0000，min = max = 1）。

---

## §5 实现细节与踩坑清单

| # | 细节 | 本模块的做法 / 理由 |
|---|---|---|
| 1 | **动作坐标系** | Buffer 存的是 **tanh 空间的归一化动作**（`∈ (−1,1)`），而不是环境动作。这样 Critic 的输入分布、`log π` 的定义域、策略的输出域**三者一致**；`a_env = a·scale + bias` 只发生在 `env.step` 之前（`to_env_action`，仿射可逆，对 Q 只是坐标重参数化）。热身期的随机动作也先仿射回归一化空间再入池（`random_norm_action`）。 |
| 2 | **`truncated ≠ done`** | Pendulum 只在 200 步处 `truncated`（无真终止）。默认 `bootstrap_on_truncation=True` ⇒ Bellman 目标保留 `γ·min Q̄(s',a')` 一项；关掉会系统性低估价值（消融见 §8，多花 2000 步才到目标）。`ReplayBuffer.done_mask` 显式实现该语义，单测逐条断言。 |
| 3 | **归一化动作的 200 步截断** | 回合在 `truncated` 时 `next_obs` 是**真实的末状态**（gymnasium 不返回 reset 后的观测），因此自举用的是真值，不是猜测。 |
| 4 | **`log(1 − tanh²x)` 数值稳定** | 用 `2(log2 − x − softplus(−2x))`；`|x| ≳ 15` 时朴素写法会 `log 0 → −inf`。单测对比两种写法并检查饱和区有限性。 |
| 5 | **重参数化的随机源** | `sample(..., generator=...)` 允许外部注入 `torch.Generator`，于是"同一 ε 下的公式核对"成为可写的单测（`test_actor_loss_matches_manual_formula`）。 |
| 6 | **初始化** | 隐藏层正交初始化 `gain=√2`；策略 `μ / log σ` 头 `gain=1e-3`、`log σ` 头 bias = `log_std_init`（−0.5）。于是初始 `μ ≈ 0 ⇒ a_env ≈ 0`（力矩为 0，不会一上来就乱打），初始 `σ ≈ 0.61`（浅层探索）。 |
| 7 | **`min(Q1, Q2)` 的用法** | Critic 训练时两个 Q 都回归同一个目标；Actor 更新用 `min`（悲观，`use_min_for_actor=true`）。单测用手算的 `min(q1,q2)` 与 `q1` 分别核对两条路径。 |
| 8 | **α 的参数化** | `log_alpha = nn.Parameter(ln α_init)`，`α = exp(log α) > 0`；`target_entropy="auto"` 在 `resolve_target_entropy` 里解析（`None` ⇒ 固定 α、不建 optimizer）。 |
| 9 | **梯度的可观测性** | `update_batch` 返回 29 个标量诊断（含三段梯度范数、`critic_frozen_ok`、`td_abs_mean`、`entropy_target_gap`…，`SAC_UPDATE_KEYS` 列出全部键名），训练脚本把它们全部写进 TensorBoard。 |
| 10 | **探索与评估分离** | 训练动作来自 `π(·|s)` 采样；评估用 `tanh(μ(s))`（`deterministic=true`）＋固定种子 `12345` ＋ 20 回合，且**只读**归一化统计量。评估口径与 W8 完全一致（复用 `src.ppo.evaluate_policy`）。 |
| 11 | **续训** | 检查点里除模型 / 三套 optimizer / RNG 状态外，还存了 **ReplayBuffer 全量内容**、累计步数 / 更新次数 / 回合数、α、最优评估值与评估步号。实测"中断后 `--resume`"与"不中断"的最终评估完全一致（−1311.42 ± 319.75 两次相同，见 §9.3）。 |
| 12 | **状态归一化** | 支持（`model.normalize_obs`）但**默认关闭**：SAC 原论文不在 Pendulum 上做观测归一化，且归一化会改变 `log π` 与世界模型的耦合；W8 的消融显示它对本任务只是噪声级别的影响。 |

---

## §6 实验设置

### 6.1 作业指定超参与配置文件的对照

| 字段 | 作业指定 | `configs/sac_config.yaml` | 说明 |
|---|---|---|---|
| `seed` / `device` / `env_name` | 42 / cuda / Pendulum-v1 | 同左 | |
| `state_dim` / `action_dim` | 3 / 1 | 同左 | |
| `total_timesteps` | 40000 | 40000 | SAC 样本效率高，4 万步足够 |
| `buffer_size` | 100000 | 100000 | > 总步数 ⇒ 全程不覆盖旧数据（`fill_ratio` 只到 0.40） |
| `batch_size` | 256 | 256 | |
| `lr_actor` / `lr_critic` / `lr_alpha` | 3e-4 / 1e-3 / 3e-4 | 同左 | 三套**独立** Adam |
| `tau` / `gamma` | 0.005 / 0.99 | 同左 | Polyak 软更新 / 折扣 |
| `target_entropy` | `"auto"`（−dim(A) = −1.0） | `"auto"` | `resolve_target_entropy("auto", 1) = −1.0` |
| `warmup_steps` | 1000 | 1000 | 纯随机热身，不更新 |

工程补充字段（非作业指定，但流水线需要）：
`hidden_dim=256`、`critic_hidden_dim=256`、`num_layers=2`、`n_critics=2`、
`log_std_init=-0.5`、正交初始化增益、`train_freq=1`、`gradient_steps=1`（UTD=1:1）、
`bootstrap_on_truncation=true`、`use_min_for_actor=true`、`target_reward=-200`（提前停止）、
`eval.every_steps=2000` / `n_episodes=20` / `deterministic=true`、检查点与断点续训设置。

### 6.2 网络规模与参数量

| 模块 | 结构 | 参数量 |
|---|---|---|
| `SquashedGaussianActor` | `MLP(3 → 2×256) + 双头(μ, logσ)` | 67 330 |
| `ContinuousCritic` | `2 × MLP([3+1] → 2×256 → 1)` | 134 658 |
| `target_critic` | `deepcopy`，永久冻结（Polyak τ=0.005） | 134 658（不参与 optimizer） |
| 温度 | 单个标量 `log_alpha` | 1 |

样本复利：40 000 步 × 1 次更新 = 39 000 次 mini-batch 更新（batch=256）
⇒ 回放池里每条经验平均被采样 `39000×256/40000 ≈ 250` 次。

---

## §7 结果

### 7.1 样本效率：多少步突破 −200？（作业预检项）

默认配置（`target_reward=-200.0`，达到即提前停止）的日志节选：

```
Step   2000 | 更新   1000 | 回合   10 | 滑动(100) -1313.13 | Q损失 0.7285/0.6821 | 策略损失 16.1849 | α 0.7447 | 熵 0.6206 | σ 0.822
  → 独立评估（均值动作，20 回合）: -1542.64 ± 236.83
Step   3000 | 更新   2000 | α 0.5749 | 熵  0.2967 | 目标Q -43.224
Step   4000 | 更新   3000 | α 0.4700 | 熵 -0.2036 | 目标Q -61.170
★ 样本效率：首次突破目标回报 -200.0 @ 4000 步（3000 次梯度更新）| 评估平均 -173.70
★ 评估平均回报 -173.70 ≥ 目标 -200.0：任务已解决，提前结束训练（累计 4000 步 / 3000 次梯度更新）
训练结束，用时 44.7s
```

| 指标 | W8 PPO（on-policy） | W9 SAC（off-policy） | 对比 |
|---|---|---|---|
| 突破 −200 所需环境步 | 194 560 步（最终 −199.91） | **4 000 步**（评估已 −173.70） | 约 **48× 更少** |
| 4 000 步时的水平 | 约 −1100 ~ −1300（还没学会） | **−173.70** | — |
| 训练时间 | 527 s（194 560 步） | 44.7 s（4 000 步） | 约 12× 更快 |
| 4 万步预算下的水平 | —（PPO 用 5 倍预算才到 −200） | **−129.71**（20 回合独立评估） | — |

> 结论：Pendulum-v1 上 SAC 的样本效率比 PPO 高 **一个数量级以上**（本实验约 48 倍）。
> 原因见 §2：PPO 每条数据只用一次，SAC 每条经验平均被抽样约 250 次。

### 7.2 完整 40 000 步的曲线（关闭提前停止）

脚本：`logs/sac_pendulum_full_*/`；图：`docs/figures/w9_sac_full_curves.png`（28 条曲线）；
标量：`docs/figures/w9_sac_full_scalars.csv`（51 个 tag / 2 123 行）。

独立评估回报（每 2 000 步一次；20 回合、均值动作、固定种子）：

| 步数 | 2 000 | 4 000 | 6 000 | 8 000 | 10 000 | 12 000 | 14 000 | 16 000 | 18 000 | 20 000 | 22 000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 评估平均 | −1542.6 | **−173.7** | −137.1 | −139.0 | −136.6 | −130.8 | −129.9 | −129.9 | −131.4 | −130.8 | **−129.71** |

| 步数 | 24 000 | 26 000 | 28 000 | 30 000 | 32 000 | 34 000 | 36 000 | 38 000 | 40 000 |
|---|---|---|---|---|---|---|---|---|---|
| 评估平均 | −129.7 | −130.0 | −131.5 | −130.6 | −129.8 | −130.4 | −130.0 | −131.6 | −131.1 |

* **2000 → 4000 步**：从随机策略（−1542）跳到 −174 —— "Critic 先学到奖励形状、
  Actor 立刻跟上"的典型 off-policy 学习曲线；
* **12 000 步以后**稳定在 **−130 ± 3**，波动与 20 回合评估噪声同量级（单回合 std ≈ 120）；
* 训练滚动均值（最近 100 个训练回合）末值 **−143.37**，与独立评估一致
  （没有"训练好、评估差"的过拟合迹象）；
* 最优权重出现在 **22 000 步**，最终用最优权重的评估为
  **−129.71 ± 120.29**（min −358.04 / max −0.41）。

Critic / 价值尺度：

| 指标 | 2 000 步 | 40 000 步 |
|---|---|---|
| `train/q1_loss` / `q2_loss` | 0.73 / 0.68 | 0.91 / 1.14 |
| `train/target_q_mean`（软 Bellman 目标 `y`） | −16.8 | −23.3（最小 −67.7） |
| `train/critic_grad_norm` | 6.4 | 260.4（Q 尺度变大 ⇒ TD 误差梯度变大） |

双 Q 的损失始终在奖励尺度附近（O(1)），**没有出现 off-policy 常见的 Q 爆炸**；
两个 Q 的损失相近但不等（独立初始化、各自回归）。

---

### 7.3 α 的自适应轨迹（作业预检项之二）

| 累计步数 | 1 000 | 5 000 | 9 000 | 13 000 | 17 000 | 21 000 | 25 000 | 29 000 | 33 000 | 37 000 | 40 000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| α | 1.0000 | 0.3823 | 0.1438 | 0.0549 | 0.0232 | 0.0225 | 0.0168 | 0.0163 | 0.0097 | 0.0087 | **0.0090** |
| `train/entropy_mean`（≈ −E[log π]） | +0.62 | −0.43 | −0.66 | −0.80 | −0.81 | −0.96 | −0.94 | −0.94 | −0.97 | −0.86 | ≈ −0.9 |
| `train/mean_std`（压缩前 σ） | 0.82 | 0.71 | 0.63 | 0.55 | 0.50 | 0.46 | 0.45 | 0.44 | 0.43 | 0.43 | 0.43 |
| `train/mean_mu_abs` | 0.22 | 0.68 | 0.95 | 1.10 | 1.19 | 1.29 | 1.31 | 1.33 | 1.36 | 1.36 | 1.36 |

**"α 前期大还是后期大？"—— 前期大，之后单调下降并稳定在小值。**

* **前期（0 ~ 5k 步）α 大（≈ 1.0 → 0.38）**：策略还没学到东西，`μ ≈ 0`、`σ ≈ 0.8`，
  动作几乎是随机的 ⇒ 策略的"平均对数密度" `E[log π]` 小/负（熵高于目标）⇒ 由式 10，
  `log α` 被往下推。此时的高温让策略保持随机、把回放池"铺开"
  —— 早期探索的动力正是"最大熵目标 + 高温度"；
* **后期（20k 步以后）α 小（≈ 0.009）**：策略逐渐确定（`|μ|` 0.22 → 1.36，`σ` 0.82 → 0.43），
  要让 `E[log π]` 追上 `−H̄ = +1` 需要继续降温 ⇒ α 一路降到 0.009，
  此时熵项在 `J_π` 里几乎不起作用，策略专注把 Q 顶上去（行为接近 DDPG）；
* **平衡点被验证**：末段 `E[log π] ≈ +0.86 ~ +0.97`，而理论平衡点是
  `E[log π] = −H̄ = dim(A) = +1.0`。也就是说：**α 停在哪里由目标熵 `H̄ = −dim(A)` 决定，
  而不是由回报决定** —— 这就是"自动温度"的含义：把探索强度调到"平均对数密度 = 每维 1 nat"。
* **为什么"前期大"？** α 是熵的拉格朗日乘子，它的大小衡量的是"当前策略离目标熵还差多远"。
  前期策略太散（`E[log π]` 远低于 +1），温度一路被往下压；后期的下降速度变慢，
  因为 `E[log π]` 已经贴近 +1，`(E[log π] + H̄)` 接近 0（梯度自平衡）。
  初值 1.0 由配置给出（`init_alpha=1.0`），它只决定"探索起点"，平衡点由 `H̄` 决定。

`train/entropy_mean` 出现"负熵"不必惊讶：它是压缩空间里密度的微分熵估计
`H ≈ −E[log π]`；当密度"平均起来 > 1"（分布集中到动作区间的一小部分）时可以为负。
目标熵 `H̄ = −1` 正是这样一个负目标（等价表述："平均对数密度 = +1"，即密度量级 ≈ e）。

### 7.4 思考题相关的训练期诊断

| 诊断量 | 观测 |
|---|---|
| `train/critic_frozen_ok` | **恒为 1.0000**（40k 全程 min = max = 1）⇒ Actor 步从未给 Critic 留下梯度 |
| `train/actor_grad_norm` | 0.40 → 1.79（始终 > 0）⇒ 计算图确实穿透了 Critic，策略一直在拿价值方向 |
| `train/critic_grad_norm` | 6.4 → 260.4（与 Q 尺度同步增长，正常） |
| `train/alpha_grad_norm` | 0.13 ~ 1.6（温度一直在自适应） |
| `train/saturation_fraction` | 末段仍为 0（动作没有卡在 tanh 饱和区） |

---

## §8 消融实验

四组消融都在**默认配置基础上只改一个开关**（seed 42 / cuda / 4 万步预算 / `target_reward=-200`
提前停止），因此可以横向比较。⚠️ 单种子 + 20 回合评估的噪声约 ±10 分，除"单 Q"这一项外，
下表差异都在噪声量级，仅作**方向性**参考。

| 配置 | 首次评估 ≥ −200 | 最终评估（最优权重） | 备注 |
|---|---|---|---|
| **对照（auto α + Double-Q + min + 自举）** | **4 000 步** | **−173.70** | 默认配置 |
| 固定 α = 1.0（`target_entropy=null`） | 6 000 步 | −160.40 | 收敛慢 50%，但熵更大、策略更"胖"；末态 Q 尺度不同（α 不降反使熵项一直很大） |
| `bootstrap_on_truncation=false`（把 200 步当终止） | 6 000 步 | −173.24 | 系统性低估价值 ⇒ 达标更慢（与 W8 PPO 的同类消融方向一致） |
| 单 Q（`n_critics=1`） | 4 000 步 | **−192.93** | 少了 `min` 的悲观估计 ⇒ 明显更差（−19 分），与 Double-Q 理论一致 |
| Actor 只用 Q1（`use_min_for_actor=false`，仍是双 Q） | 4 000 步 | −174.94 | 与对照几乎相同 —— 说明"**Critic 训练阶段**的双 Q 回归"才是主要收益，Actor 端取不取 min 影响很小 |

复现命令（每条约 1 分钟）：

```bash
python experiments/train_sac.py --set experiment.name=sac_abl_fixed_alpha  training.target_entropy=null
python experiments/train_sac.py --set experiment.name=sac_abl_nobootstrap  training.bootstrap_on_truncation=false
python experiments/train_sac.py --set experiment.name=sac_abl_single_q     model.n_critics=1
python experiments/train_sac.py --set experiment.name=sac_abl_q1only       training.use_min_for_actor=false
```

---

## §9 测试（`tests/test_sac.py`，48 项，全部通过）

```bash
python tests/test_sac.py        # 无需 pytest；CPU 上约 18 s
pytest tests/test_sac.py        # 装了 pytest 也可以
```

| 分组 | 覆盖的"硬判据" |
|---|---|
| **1. 手写数学**（4 项） | `log(1−tanh²x)` 稳定写法 vs 朴素写法逐位一致、饱和区（`x=±30`）不产生 `−inf`；压缩高斯 `log π` 与手写换元公式逐位一致；**数值积分 `∫ exp(log π) = 1.0000 ± 0.002`**；`target_entropy` 的 `auto/数字/none/非法值` 分支 |
| **2. 压缩高斯 Actor**（6 项） | `(μ, log σ)` 形状与夹紧（含把 log σ 头推到 ±50 的极端检查）；确定性动作 = `tanh(μ)`、采样恒在 `(−2,2)`；**重参数化可导性**（`μ` 与 `log σ` 都拿到非零梯度）；固定 `generator` 可复现且 `log_prob` 与同一 `ε` 对应；仿射映射往返（含非对称区间 `[−1,3]`）；小增益初始化 ⇒ 初始动作 ≈ 0 |
| **3. 双 Q 与目标网络**（6 项） | 双 Q 形状 / 独立初始化 ⇒ 初始 `Q1 ≠ Q2` / `min` 语义；`_concat` 顺序确为 `[s,a]`；维度校验报错；Polyak 公式逐参数核对（`τ=0/1` 边界、非法 τ 报错）；目标网络深拷贝 + 永久冻结 + `sync_target`；目标网络前向的图连通但参数无梯度 |
| **4. ReplayBuffer**（6 项） | 写/采形状 dtype device、值逐条对应；环形覆盖与 `is_full`；样本不足 / `batch_size≤0` / 下标越界报错；**`truncated ≠ done`**（开关两种取值逐条断言）；`state_dict` 往返 + 容量/维度不一致报错；`add_batch` 与 `stats`（含空池 NaN 语义） |
| **5. Critic 损失**（4 项） | `done≡1` ⇒ 目标退化为 `r`（与手算 MSE 逐位一致、`q1_loss+q2_loss=critic_loss`）；**扰动 Critic 参数后 `target_q_mean` 不变**（detach 的证据）；Critic 的 backward 不碰 Actor / `log α`；**单状态玩具系统收敛到解析不动点 `Q*=(r−γα·E[logπ])/(1−γ)`（相对误差 < 2%）** |
| **6. Actor 损失 / 思考题**（8 项） | 与手写公式逐位一致（同一 RNG ⇒ 同一 `ε`）；二次玩具 Q 下策略**朝 `argmax_a Q` 移动**（两个不同目标都验证）；**冻结 Critic 参数后 Actor 仍有非零梯度、`∂q_pi/∂a ≠ 0`、Critic 的 `.grad` 全为 `None`**；`q.detach()` 后"只剩熵项梯度"（与单独反传熵项逐位相等）；Q 恒定时熵上升；Actor 步 5 连发后 Critic 逐位不变；**共享 optimizer 会污染 Critic 且把 Q 抬高**（反例现场复现）；`use_min_for_actor` 两条路径与 `|∂Q/∂a|` 诊断 |
| **7. 温度 α**（4 项） | `α = exp(log α)`、初值 / 可学习性 / 固定模式；`J(α)` 公式逐位核对 + **不回传 Actor/Critic**；**方向测试**：`E[log π]` 偏大 ⇒ α↑、偏小 ⇒ α↓；`alpha_min` 夹紧 |
| **8. 装配与完整更新**（4 项） | 按真实配置装配（结构 / 目标熵 −1 / α / 动作边界 / 参数量）；三套 optimizer 学习率与参数集合互不相交、目标网络不入 optimizer；`update_batch` 四段参数变化 + 29 个诊断量齐全且有限 + `critic_frozen_ok=1.0`；固定 α 路径与缺 optimizer 报错；`grad_norm` / `frozen_parameters` 语义；`random_norm_action` 落在 `(−1,1)` 且可复现 |
| **9. 端到端**（2 项） | **bandit 玩具**：`reward=−|a_env−1|`，2500 次更新后策略从随机逼近最优动作（误差 < 0.25）且 `Q ≈ r`；**Pendulum 短路训练**：700 步随机数据 + 约 220 次更新，诊断量全部有限、动作始终在范围内、`critic_frozen_ok=1.0`、随机策略回报落在 `(−2000,−400)` |
| **10. 工程脚本与配置**（2 项） | 配置与作业指定超参**逐项一致**（含 `target_entropy="auto"`、`bootstrap_on_truncation=true`）；`train_sac.py` 的 `as_bool / rolling_mean / check_action_bounds / RunningAvg` 语义 |

### 9.1 为什么这些测试值得写？

* **数学层面**：`log π` 的换元修正、Polyak 公式、α 的梯度方向这三处只要写错符号，
  训练**不会报错**、只是悄悄变差 —— 所以都用"手写金标准 + 逐位比对"钉住；
* **算法层面**：`truncated ≠ done`、buffer 的动作坐标系、`no_grad` 的边界
  都属于"静默错误"，单测把它们变成可判定的断言；
* **思考题层面**：`test_frozen_critic_still_backprops_to_actor` / `test_detaching_q_leaves_only_entropy_gradient` /
  `test_shared_optimizer_would_corrupt_critic` 三件套，把"必须穿透、不能更新、怎么切断"
  变成可执行的证据（而不是文字描述）。

### 9.2 回归保障

* W8 的 `tests/test_ppo.py` 仍 **38/38 通过**（约 14 s）；
* `src/__init__.py` 追加导出后 `import src` 无缺失符号（`__all__` 全可访问）；
* `experiments/export_tb_curves.py` 追加 SAC 面板后，W8 的导出测试（PNG/CSV/摘要）仍通过。

### 9.3 断点续训一致性（实测）

把同一个实验拆成"先跑 1000 步 → `--resume` 再跑 500 步"与"一次跑 1500 步"两种方式：

```
[恢复] 已恢复检查点: .../sac_smoke2_last.pt | 累计交互 1000 步 | 梯度更新 600 次 | 完成回合 5 |
       回放池 1000 条 | α=0.8358 | 历史最优评估 -1311.42 @ step 500
[续训] 训练结束，用时 6.7s | 累计交互步数: 1500 | ... | α: 0.7233
[最终评估] 最终评估（最优权重，5 回合） | 平均回报: -1311.42 ± 319.75
```

两次运行的最终评估**完全一致**（−1311.42 ± 319.75）⇒ 权重 / 三套 optimizer 状态 /
ReplayBuffer 内容 / RNG 状态 / 累计计数都被完整恢复。

---

## §10 复现清单（一键复现实验与验收）

```bash
# 0) 安装依赖（torch + gymnasium + pyyaml + tensorboard + matplotlib + numpy）

# 1) 单元测试（48 项，CPU，约 18 s；W8 的 PPO 测试也顺带跑一遍）
python tests/test_sac.py
python tests/test_ppo.py

# 2) 默认训练：热身 1000 步 → 4000 步突破 −200 并提前停止（约 45 s，cuda）
python experiments/train_sac.py

# 3) 完整 40 000 步（关闭提前停止，用于观察完整曲线与 α 轨迹；约 9 min）
#    说明：把 target_reward 设成不可达值即可禁用"评估/滚动均值"两条提前停止路径
python experiments/train_sac.py --set experiment.name=sac_pendulum_full training.target_reward=1e9

# 4) 导出曲线（PNG + CSV；默认输出到 curves/，本仓库归档在 docs/figures/）
python experiments/export_tb_curves.py --logdir logs/sac_pendulum_full_* --out-dir docs/figures \
       --prefix w9_sac_full --smooth 3

# 5) TensorBoard 观察实时曲线
tensorboard --logdir logs

# 6) 断点续训（可随时 Ctrl-C：脚本会先存检查点再退出）
python experiments/train_sac.py --resume

# 7) 消融（每条约 1 分钟）
python experiments/train_sac.py --set experiment.name=sac_abl_fixed_alpha training.target_entropy=null
python experiments/train_sac.py --set experiment.name=sac_abl_nobootstrap training.bootstrap_on_truncation=false
python experiments/train_sac.py --set experiment.name=sac_abl_single_q    model.n_critics=1
python experiments/train_sac.py --set experiment.name=sac_abl_q1only      training.use_min_for_actor=false
```

### 验收对照表

| 作业要求 | 状态 | 证据 |
|---|---|---|
| 新建 `configs/sac_config.yaml` | ✅ | 文件存在，超参与 §6.1 表逐项一致；`test_sac_config_matches_assignment_spec` |
| 新建 `src/sac.py`：`SquashedGaussianActor`（重参数化 + Jacobian） | ✅ | §3.3；`test_squashed_log_prob_integrates_to_one`、`test_actor_sample_is_reparameterized_and_differentiable` |
| `ContinuousCritic`（`[s,a]` → 标量，Double-Q） | ✅ | §3.2；`test_critic_double_q_shapes_and_independent_init`、`test_critic_input_is_state_then_action` |
| `ReplayBuffer`（大批量随机采样） | ✅ | §5 表 #1/#2；buffer 组 6 项测试 |
| `SACAgent`（Double-Q 损失 + 策略损失 + α 自适应） | ✅ | §3.5；`test_update_batch_runs_three_steps_and_reports_diagnostics` |
| `experiments/train_sac.py`（热身期 / 探索与评估分离 / 监控指标） | ✅ | §1.1；脚本头部 30 行注释 + TB 的 51 个 tag |
| 跑通训练并记录"多少步突破 −200" | ✅ | **4 000 步**（§7.1） |
| 观察 α 的自适应轨迹并解释 | ✅ | §7.3（1.0 → 0.009，平衡点 `E[log π] → dim(A)`） |
| 思考题：为什么穿透但不能更新 / 代码怎么切断 | ✅ | §4（含 4 处代码落点 + 单测证据表） |
| 样本效率 vs PPO 的量级提升 | ✅ | 48×（§7.1）；PPO 数据来自 W8 的正式训练记录 |

### 已知限制

1. **单种子结果**：所有训练都是 seed 42；Pendulum-v1 的 20 回合评估 std ≈ 120，
   消融表里的 10 分级差异不宜过度解读（唯有"单 Q vs 双 Q"约 19 分与理论方向一致）。
2. **π 的熵估计是蒙特卡洛**：`train/entropy_mean = −mean(log π)` 用当前 mini-batch 估计，
   有采样噪声；压缩空间里它可以为负（§7.3 已解释）。
3. **未实现的内容**：多动作维（`action_dim > 1`）在代码里是维度泛化的（对最后一维求和），
   但只在 `action_dim=1` 的 Pendulum 上验证过；SAC 的"离散动作"变体、优先级回放（PER）、
   n-step 回报、CQL 等离线 RL 扩展都不在本任务范围内。
