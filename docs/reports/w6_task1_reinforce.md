# W6 Task 1：从零实现 REINFORCE（策略梯度 + Baseline + Return 归一化）

## 1. 目标

前四周的流水线（W3/W4）都建立在**专家数据**之上（行为克隆 / 动作块 BC / 扩散策略）。
W6 把数据来源换成**策略自己与环境交互产生的轨迹**，从零手写最基础的策略梯度算法
**REINFORCE**（Williams, 1992），不依赖 SB3 / rlkit 等任何 RL 库，并回答两个问题：

> **思考题 1**：损失函数里的负号（`L = −Σ_t log π_θ(a_t|s_t) · A_t`）为什么是必须的？
>
> **思考题 2**：如果把 Return 归一化（`A = (G − μ) / (σ + ε)`）拿掉，训练会发生什么？

交付物：

| 文件 | 说明 |
| --- | --- |
| `src/reinforce.py` | **新增**：核心库。`PolicyNetwork`（策略 `π_θ(a\|s)`：MLP → logits → `Categorical`）、`ValueNetwork`（进阶基线 `V_φ(s)`）、`build_policy` / `build_value_network` / `make_env` / `env_max_steps`、`compute_returns`（折扣回报 + Return 归一化）、`compute_advantages`（三种基线）、`reinforce_loss`（含负号 + 可选熵奖励）、`value_loss`、`run_episode`（`no_grad` 采样并缓存 `(s, a, logπ, r)`）、`stack_episodes`（多回合拼批）、`update_policy`（一次策略改进 + 全部诊断量）、`evaluate_policy`（贪心评估） |
| `configs/reinforce_config.yaml` | **新增**：`experiment`（seed 42 / device）/ `env`（CartPole-v1）/ `model`（4 → 2×128 → 2）/ `training`（lr 2e-3、γ 0.99、max_episodes 600、target_reward 475、log_interval 20、`update_batch_episodes`、`max_grad_norm`）/ `algorithm`（`baseline: batch_mean`、`normalize_returns: true`、`normalize_advantages: false`、`entropy_coef: 0.0`、`critic_coef`、`reduction`）/ `eval`（每 20 回合用 20 回合贪心评估，`seed: 12345`）/ `paths`（含 `resume`） |
| `experiments/train_reinforce.py` | **新增**：训练入口（复用 `src/engine.py` 的 `RunContext` / 日志 / TensorBoard / 检查点 / `--set` / `--resume`），含 `rolling_mean`、`as_bool`、`build_optimizer`、`train_reinforce`、`main`，并在文件头部回答两道思考题 |
| `src/__init__.py` | 统一导出 REINFORCE 相关符号（`src.diffusion` 的同名 `count_parameters` 不参与 `__all__` 展开，避免命名冲突） |
| `tests/test_reinforce.py` | **新增**：29 个单元测试（`python tests/test_reinforce.py` 直接可跑，也兼容 pytest） |

**核心机理（一句话）**：REINFORCE 不做任何"动作值"建模，它只做一件事 ——
把**整条回合的蒙特卡洛回报**当成"这次采样到的动作序列好不好"的无偏评分，
然后用 `∇_θ log π_θ(a_t|s_t) · A_t` 沿着"提高好动作概率、压低坏动作概率"的方向更新参数。

## 2. 算法数学（本文用到的全部公式）

**优化目标**（最大化期望折扣回报）：

```
J(θ) = E_{τ ~ π_θ} [ G_0 ],      G_t = Σ_{k ≥ t} γ^{k−t} r_k
```

**策略梯度定理**（`A_t` 可为任意只依赖状态的基线 `G_t − b(s_t)`）：

```
∇_θ J(θ) = E_{τ ~ π_θ} [ Σ_t  ∇_θ log π_θ(a_t | s_t) · A_t ]
```

因为 `E_{a ~ π_θ(·|s)}[∇_θ log π_θ(a|s) · b(s)] = b(s) · ∇_θ Σ_a π_θ(a|s) = 0`，
减基线 / Return 归一化都是**无偏的方差缩减**。

**本实现的四个可测部件**：

| 部件 | 公式 | 代码 |
| --- | --- | --- |
| 折扣回报 | `G_t = r_t + γ · G_{t+1}`（从末尾反向递推） | `src/reinforce.py: compute_returns` |
| Return 归一化 | `Â_t = (A_t − mean(A)) / (std(A) + ε)`；默认 `b = mean(G)` 时即 `(G − μ) / (σ + ε)` | `update_policy` 第 3 步（批级统计量） |
| 策略损失（**负号**） | `L(θ) = −Σ_t log π_θ(a_t\|s_t) · Â_t`（`reduction: mean` 时为均值） | `src/reinforce.py: reinforce_loss` |
| 梯度更新 | `∇_θ L = −Σ_t Â_t · ∇_θ log π_θ(a_t\|s_t)`，`θ ← θ − lr · ∇_θ L` | `update_policy` 第 5 步 |

**基线三档**（`algorithm.baseline`）：

```
none        b = 0            →  A_t = G_t                 （原始回报，方差最大）
batch_mean  b = mean(G)      →  A_t = G_t − mean(G)       （最简 Baseline，默认）
value       b = V_φ(s_t)     →  A_t = G_t − V_φ(s_t)      （进阶 Baseline，训一个小 Critic）
```

**总损失**（`baseline: value` 与 `entropy_coef > 0` 时才多出后两项）：

```
L_total = −Σ_t log π_θ(a_t|s_t) · Â_t  +  c · MSE(V_φ(s_t), G_t)  −  β · Σ_t H[π_θ(·|s_t)]
```

**数据流（一次更新）**：

```
环境交互 ──no_grad──> (s_t, a_t, r_t, logπ_cache) ──反向递推──> G_t
                                                  │
      (s_t, a_t) ──重新前向（带梯度）──> logπ_θ, H ─┤
                                                  ↓
        G_t ──− b(s_t)──> A_t ──归一化──> Â_t ──> L = −Σ logπ·Â (+c·MSE −β·H)
                                                  ↓
                          backward ──> clip_grad_norm_(可选) ──> optimizer.step()
```

## 3. 思考题 1：负号为什么是必须的？

### 3.1 结论

因为**优化器只会"最小化"**，而我们要的是"**最大化**期望回报"。
把目标写成 `L(θ) = −Σ_t log π_θ(a_t|s_t) · A_t` 之后，`minimize L` 就等价于 `maximize J`：

```
θ ← θ − lr · ∇_θ L = θ − lr · ( −Σ_t Â_t ∇_θ log π_θ(a_t|s_t) ) = θ + lr · Σ_t Â_t ∇_θ log π_θ(a_t|s_t)
                    └────────────────── 梯度上升（ascent）──────────────────┘
```

也就是说，负号把"PyTorch 的梯度**下降**"翻译成了"策略梯度的**上升**"。
如果写成 `L = +Σ_t log π_θ · A_t` 而仍然用 `optimizer.step()` 最小化它，
更新方向就整体反号，变成了"**降低**期望回报"。

### 3.2 两个容易混淆的点

1. **符号由 `Â_t` 的符号承载，不是由 `log π` 承载**。
   `∇_θ log π_θ(a|s)` 永远指向"让刚采样到的那个动作概率变大"的方向（对 `a` 是无条件的、恒正的上升方向），
   "该动作是好是坏"这件事**只能**通过 `Â_t` 的正负来表达；而"要最大化而不是最小化"这层意思，
   只能靠损失前面的负号来表达。两者缺一不可。
2. **`Â_t` 只是常数权重，必须 `detach`**。
   优势里的基线（尤其是 `baseline: value` 的 `V_φ(s_t)`）如果在计算图里没被切断，
   策略损失就会顺着"把基线也一起抬高/压低"的方向去训练 Critic，
   让 Critic 的回归目标被策略梯度污染。`src/reinforce.py` 在
   `compute_advantages` 与 `reinforce_loss` 里各做一次 `.detach()`，
   并用 `test_advantage_enters_loss_as_a_constant_weight` 断言"优势的梯度不回传"。

### 3.3 量化证据（`tests/test_reinforce.py`）

* `test_minus_sign_is_required_for_ascent`：固定一批 `(s, a, G)` 与一个初始策略，
  分别用 `L = −Σ logπ·Â` 与 `L = +Σ logπ·Â` 用同一个 Adam 走同样 20 步。
  前者让"正优势动作"的概率升到 **0.9 以上**、目标 `J = logπ·Â` 上升；
  后者把它压到 **0.1 以下**、`J` 下降（即"训练去最大化坏动作的概率"）——
  两者相差约一个数量级，方向完全相反。
* `test_update_policy_moves_log_probs_along_the_advantage_direction`：
  用真实轨迹验证 `Â>0` 的步被提高、`Â<0` 的步被压低，且方向与 `Â` 的符号严格一致。
* `test_reinforce_loss_equals_negative_sum_of_log_prob_times_advantage`：
  直接把 `reinforce_loss` 的返回值与手写的 `−Σ logπ·Â` 逐位对齐（含 `mean` / `sum` 两种 reduction），
  把"公式里必须有一个负号"钉在数值上。

## 4. 思考题 2：拿掉 Return 归一化会发生什么？

### 4.1 归一化到底做了什么

`Â = (A − mean(A)) / (std(A) + ε)` 是**仿射变换**：一个平移 + 一个缩放。

* **平移**部分会被基线吸收：默认 `baseline: batch_mean` 时 `A = G − mean(G)` 本身已经零均值，
  再减一次 `mean(Â) ≈ 0` 无实质影响；
* **缩放**部分是关键：它把梯度的模长归一化，于是**有效步长与回报的量纲/幅度解耦**。

定量地：带基线时"归一化 / 不归一化"两套梯度严格成比例，比例系数就是 `1 / (σ + ε)`
（`test_return_normalization_only_rescales_the_gradient` 用同一条轨迹验证了这一点：
方向完全相同，模长之比 = `1/σ`）。

### 4.2 不归一化的三个后果

1. **梯度尺度随回报幅度漂移 ⇒ 学习率隐性漂移。**
   CartPole 的采样回报从训练初期的 ~10 涨到后期的 ~500（跨近两个数量级），
   不归一化时梯度范数也同步放大。本机实测（第 4.3 节同一批消融）：
   `baseline=none, normalize=false` 的 `grad_norm` 从 ep1 的 3.06 一路涨到 ep100 的 11.65，
   而同配置下归一化版本始终在 0.1~0.5 —— 相差 **1~2 个数量级**；
   即便是有基线的一对（第 1 行 vs 第 2 行）也差 **3~10 倍**。
   同一个 `lr = 2e-3`，在回报还小的时候合适，回报变大之后就变成了"步长过大"。
2. **正回报偏移（offset）没有被消除 ⇒ 只靠动作之间的相对权重起作用，方差更大。**
   若没有基线（`baseline: none`），CartPole 里每一步奖励都是 `+1`，因此所有 `G_t > 0`：
   `Σ Â_t ∇log π` 里的每一步都在"提高刚采样动作的概率"，好动作与坏动作的差异
   只体现在系数大小的比较上。REINFORCE 的原始形式依然**无偏**
   （因为 `E[∇log π · b(s)] = 0`），但方差显著变大，实际表现为训练曲线更抖、
   更容易被个别长回合"带跑偏"。
3. **奖励量纲一变，超参全部失效。** 如果把 CartPole 的每步奖励改成 `+100`（满分 50000），
   不归一化时等效学习率就 ×100，训练立刻发散；归一化后训练行为（几乎）不变 ——
   这就是"reward-scale invariant"的实际价值，也是把它默认打开的理由。

### 4.3 实测（同一份配置，仅改两个开关；seed 42，CPU，150 回合上限）

| # | `baseline` | `normalize_returns` | 训练期最好评估 | ep150 评估 | 最近 100 回合训练均值 | `grad_norm` 量级 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `batch_mean` | **true**（默认） | **500.00 @ep120**（提前结束） | 500.00 ± 0.00 | 188.1 | 0.1 ~ 0.5 |
| 2 | `batch_mean` | false | **500.00 @ep120**（提前结束） | 500.00 ± 0.00 | 168.5 | 1.0 ~ 1.8 |
| 3 | `none` | **true** | **500.00 @ep120**（提前结束） | 500.00 ± 0.00 | 116.5 | 0.1 ~ 0.3 |
| 4 | `none` | false | 348.40 @ep60 → **崩到 33.40 @ep150** | 33.40 ± 2.80 | 63.1 | 3.1 ~ 11.7 |

**怎么读这张表**（结论比"能不能解"更重要）：

* **归一化 = 控制尺度**：第 1 行 vs 第 2 行、第 3 行 vs 第 4 行，
  归一化把 `grad_norm` 压小了 **1~2 个数量级**（有基线时 3~10 倍，无基线时 10~100 倍），
  并且让第 4 行的"训练崩溃"变成第 3 行的稳定收敛 —— 这是不归一化最真实的代价。
* **基线 = 消除偏移**：第 2 行（有基线、不归一化）仍能解，而第 4 行（无基线、不归一化）崩溃；
  说明两者**互补但不等价**：基线抵消"常数偏移"，归一化抵消"幅度漂移"。
* 第 1 行与第 2 行都解了任务，说明 CartPole 上"不归一化也能收敛"（任务本身回报量纲变化不够剧烈、
  lr 又偏小）；但第 2 行的 `grad_norm` 已经大了 3~10 倍，处在"换个大一点的学习率就炸"的边缘 ——
  这类隐性风险正是工程上默认开启归一化的原因。
* 崩溃的具体形态（第 4 行）：策略熵从 0.69 掉到 0.54，`policy_loss` 从 6.6 膨胀到 19.0，
  随后策略塌缩到"存活 30 步左右"的局部解；这是典型的大步长 + 高方差导致的**过早收敛 / 策略崩塌**。

## 5. 关键实现细节与踩过的坑

1. **采样用 `no_grad`，更新时重新前向算 `logπ`。**
   采样阶段整条回合包在 `@torch.no_grad()` 里，只缓存 `(s, a, logπ, r, H)` 这些**数值**
   （避免整条回合的计算图常驻显存）；`update_policy` 再用 `(states, actions)` 重新前向一次。
   因为"采样 → 更新"之间参数没有变，重算值与缓存值**逐位一致**
   （`test_recomputed_log_probs_match_cached` 的 `log_prob_drift` 断言等于 0），
   但重算让"多回合拼批更新"成为可能。日志里的 `熵` 也来自这一次重算。
2. **优势必须 `detach`**（见 3.2 第 2 点）。`compute_advantages` 里对基线 `detach`，
   `reinforce_loss` 里对优势再 `detach` 一次 —— 双保险，因为优势只是"常数权重"。
3. **"单优化器"与"双优化器"等价（刻意选择单优化器）。**
   `baseline: value` 时策略损失与 Critic 的 MSE 写成一个联合损失交给**同一个 Adam**：
   由于 `advantage.detach()` 切断了流向 Critic 的梯度，两项损失各自只依赖
   **互不相交**的参数集合，而 Adam 又是**逐参数**更新的，所以数学上完全等价
   （`test_joint_optimizer_matches_two_separate_optimizers` 用逐参数对齐验证）。
   好处：检查点里只需维护一份优化器状态，断点续训逻辑与 W3/W4 完全复用。
4. **`truncated` 不自举。** CartPole-v1 的 500 步是 `TimeLimit` 截断（`truncated=True`）
   而不是失败，本任务把它当回合结束（不做 `V(s_T)` 自举），因为 500 步就是任务满分；
   这也让"回合回报 = 存活步数"，评估指标与 Gym 的 `CartPole-v1` 解决标准（475/500）直接对齐。
5. **数值稳定与效率**：网络只输出 logits，不手动 softmax；`log π` / `H` 都走
   `Categorical.log_prob` / `entropy`（内部 `logsumexp`）。`update_policy` 里
   `policy.distribution(states)` **只调用一次**，同时拿到 `logπ` 与熵，避免重复前向。
6. **归一化统计量在"整个更新批"上算**，而不是单回合内。
   `stack_episodes` 把 `k` 条回合（`update_batch_episodes`）拼成一个 `(N,)` 批，
   `mean/std` 在整个批上估计 —— 批越大，基线与尺度估计越准、方差越小（代价是更新频率降低）。
7. **`episode_index` 保留回合边界**：拼批后仍能按回合切片做诊断（每回合回报、长度），
   这也是 `test_duplicated_episode_batch_matches_single_episode_gradient` 能验证
   "拼批只是把 `Σ_t` 的范围放大"的原因。
8. **随机源可复现**：每个训练回合用 `seed + episode − 1` 重置环境（`env.reset(seed=...)`），
   评估固定 `eval.seed`（且训练期间用 `torch.no_grad` + 孪生 RNG 状态，不污染训练随机流）。
9. **断点续训的额外字段**：策略 / 优化器 / `episode` / 最优指标由 `Checkpointer` 统一处理，
   本任务还在检查点的 `extra` 里额外保存 `critic_state_dict`（价值网络）与 `steps_total`
   （累计交互步数），恢复后累计步数**继续累加**（实测：ep10 存档 277 步 → 续训到 ep20 显示 645 步）。
10. **`--set` 的布尔陷阱**：`src.engine.cast_value` 用 `ast.literal_eval` 解析命令行，
    因此 PyYAML 风格的小写 `false` 会解析失败并**退化成字符串 `"false"`**（字符串是 truthy！），
    结果 `--set algorithm.normalize_returns=false` 反而打开了归一化。
    脚本里用 `as_bool()` 显式识别 `true/false/1/0/yes/no/on/off`，两种写法行为一致。
11. **熵项默认关闭但"始终统计"**：`entropy_coef = 0.0` 时熵只是诊断量、不改变梯度
    （保留 `−βH` 的代码路径，方便做探索性消融）。

## 6. 正式实验（默认配置，seed 42）

**设置**：`CartPole-v1`；策略 MLP `4 → 128 → 128 → 2`（17,410 个可训练参数）；
Adam `lr = 2e-3`；`γ = 0.99`；`baseline = batch_mean`；`normalize_returns = true`；
每回合更新一次（`update_batch_episodes = 1`）；`max_episodes = 600`；
每 20 回合用 **20 回合贪心评估**（`eval.seed = 12345`）；`target_reward = 475`；
`entropy_coef = 0.0`。

**结果**：

| 指标 | 数值 |
| --- | --- |
| 提前停止 | **episode 120**（评估 500.00 ≥ 475） |
| 最优评估回报 | **500.00 ± 0.00**（20 回合全部满 500 步） |
| 训练滑动均值（最近 100 回合，ep120） | 97.9 |
| 累计与环境交互步数 | 10,339 |
| 用时 | 约 36 s（GPU，含 CUDA 初始化 + 6 次 × 20 回合贪心评估；同等交互量在 CPU 上是 10 s 量级） |
| 产物 | `logs/reinforce_cartpole_<时间戳>/{train.log, events.out.tfevents.*}`（TensorBoard + 文本日志）、`logs/checkpoints/reinforce_cartpole_<时间戳>/{reinforce_cartpole_best.pt, *_last.pt}` |

**训练曲线（评估回报）**：

```
ep20   43.8  ± 16.7
ep40  160.5  ± 144.4   （已有单条评估回合打到 500！）
ep60  169.7  ± 104.0
ep80  101.5  ±  27.6   （回退：策略仍在"探索噪声"里摆动）
ep100 139.1  ±  85.2
ep120 500.0  ±   0.0   → 触发 target_reward，提前结束
```

三个值得记录的观察：

1. **训练回合均值（97.9）远低于贪心评估（500）**：训练时策略是按 `π_θ(·|s)` **采样**动作的
   （必须保留探索），评估时取 `argmax`。这条差距是策略梯度类算法的常态，
   也说明"用训练回报直接当评估指标"会严重低估策略能力。
2. **ep40 就已经有单条评估回合打满 500**，但直到 ep120 才做到"20 回合平均全满"——
   梯度的作用是把**整个分布**推向确定性正确，而不是让某一次采样变好。
3. **训练不需要专家数据**，全程只有 `r = +1 / 步`，10,339 步交互即在 36 秒内解出 CartPole；
   与 W3/W4 的 BC 路线相比，BC 的瓶颈是"专家数据的覆盖度"，RL 的瓶颈是"**方差**"
   （这里靠 baseline + Return 归一化 + 多回合拼批把它压住）。

## 7. 单元测试覆盖（29 个，全部通过）

```bash
python tests/test_reinforce.py      # 直接运行（无需 pytest），输出 "通过 29/29"
```

| 分组 | 测试 |
| --- | --- |
| 1. 回报与优势 | `test_returns_match_naive_backward_recursion`、`test_returns_with_gamma_zero_equals_immediate_rewards`、`test_returns_normalization_is_zero_mean_unit_std`、`test_advantage_baseline_forms`、`test_value_baseline_reduces_advantage_variance`、`test_batch_mean_baseline_makes_update_immune_to_reward_offset`、`test_advantage_length_mismatch_raises` |
| 2. 策略 / 价值网络 | `test_policy_distribution_is_a_valid_categorical`、`test_zero_logits_give_uniform_policy_with_max_entropy`、`test_policy_act_deterministic_and_sampling`、`test_policy_log_prob_matches_direct_softmax_formula`、`test_value_network_shape_and_mse_loss` |
| 3. 损失函数与负号（思考题 1） | `test_reinforce_loss_equals_negative_sum_of_log_prob_times_advantage`、`test_advantage_enters_loss_as_a_constant_weight`、`test_minus_sign_is_required_for_ascent`、`test_update_policy_moves_log_probs_along_the_advantage_direction` |
| 4. 回合采样与拼批 | `test_run_episode_caches_policy_outputs`、`test_episode_terminated_truncated_and_default_limit_paths`、`test_stack_episodes_concatenates_and_keeps_episode_index`、`test_duplicated_episode_batch_matches_single_episode_gradient` |
| 5. 一次策略改进 | `test_update_policy_changes_parameters_and_reports_consistent_stats`、`test_recomputed_log_probs_match_cached`、`test_joint_optimizer_matches_two_separate_optimizers`、`test_grad_clip_limits_global_gradient_norm`、`test_return_normalization_only_rescales_the_gradient`、`test_value_baseline_trains_critic_towards_returns`、`test_update_policy_validates_baseline_and_critic` |
| 6. 端到端 | `test_evaluate_policy_is_reproducible_and_equals_survived_steps`、`test_short_training_on_cartpole_improves_return` |

其中与两道思考题**直接对应**的三个测试：

* `test_minus_sign_is_required_for_ascent`：同一个 Adam、同一批数据，
  `−Σ logπ·Â` 让正优势动作概率单调上升到 ~1，而 `+Σ logπ·Â` 把它压到 ~0
  （两条曲线的终点近似互换）—— 负号不是"风格"，是方向。
* `test_return_normalization_only_rescales_the_gradient`：同一条轨迹下
  两套梯度**方向完全一致**、模长比严格等于 `1/(σ+ε)`，因此归一化**不改变上升方向**，
  只改变有效步长。
* `test_batch_mean_baseline_makes_update_immune_to_reward_offset`：
  把整条回合的奖励统一加一个大常数（`r → r + C`），`baseline: batch_mean` 下
  参数更新**逐位不变** —— 这解释了"为什么 baseline 是必需的"，
  也解释了第 4 节里"缺基线 + 不归一化"最容易崩。

## 8. 复现命令

```bash
# 单元测试
conda run -n rl python tests/test_reinforce.py           # 通过 29/29

# 正式训练（默认 seed 42 / cuda，约 36 s 解出 CartPole-V1 并提前停止）
python experiments/train_reinforce.py
python experiments/train_reinforce.py --device cpu        # 无 GPU
python experiments/train_reinforce.py --no-tb             # 关掉 TensorBoard

# 思考题 2 的消融（2×2：baseline × normalize_returns）
python experiments/train_reinforce.py --set algorithm.normalize_returns=false
python experiments/train_reinforce.py --set algorithm.baseline=none
python experiments/train_reinforce.py --set algorithm.baseline=none algorithm.normalize_returns=false

# 进阶 Baseline（小 Critic V_φ(s)）+ 熵奖励（鼓励探索）
python experiments/train_reinforce.py --set algorithm.baseline=value
python experiments/train_reinforce.py --set algorithm.baseline=value algorithm.entropy_coef=0.01

# 多回合拼批更新 + 梯度裁剪（降方差 / 提稳）
python experiments/train_reinforce.py --set training.update_batch_episodes=4 training.max_grad_norm=1.0

# 断点续训（自动找 checkpoint_dir 下最新的 last.pt）
python experiments/train_reinforce.py --resume
python experiments/train_reinforce.py --resume logs/checkpoints/<run>/reinforce_cartpole_last.pt

# 看曲线
tensorboard --logdir logs
```

**TensorBoard 关键曲线**：`train/episode_return`、`train/rolling_return`、`train/episode_length`、
`train/policy_loss`、`train/entropy`、`train/advantage_abs_mean`、`train/advantage_raw_std`、
`train/return_raw_mean`、`train/grad_norm`、`train/log_prob_drift`、`train/critic_loss`（仅 `baseline=value`）、
`eval/mean_return`（+`std/min/max`）、`eval/rolling_train_return`、`eval/final_mean_return`。

## 9. 局限与后续方向

* **REINFORCE 的本质局限是方差**：它用整条回合的蒙特卡洛回报 `G_t` 当优势估计，
  方差随回合长度 `T` 增长（本任务 CartPole 最多 500 步），因此需要大批量 / 多回合拼批 /
  归一化 / 基线来压。第 6 节的曲线抖动（`ep80` 回退）就是这个方差的直接体现。
* **下一步（同一套代码即可扩展）**：
  1. **n-step / GAE(λ) 回报**：把 `compute_returns` 换成 `G_t^{(n)} = Σ_{k<t+n} γ^{k−t} r_k + γ^n V(s_{t+n})`，
     用 Critic 做自举 —— 这就是从 REINFORCE 走到 A2C 的那一步；
  2. **A2C / PPO**：并行多个环境同步采样（降相关方差）+ PPO 的重要性比裁剪（允许同一批数据多轮更新，
     样本效率显著提升），本任务的 `update_policy` 已经是"策略损失 + Critic 损失 + 熵项 + 梯度裁剪"
     的完整骨架，改损失即可；
  3. **共享主干 + 连续动作**：把 `PolicyNetwork` 换成 Gaussian / 扩散策略（复用 W4 的 `src/diffusion.py`），
     在 `Pendulum` / 机械臂这类连续控制任务上验证 —— 那时 `log π` 与熵要换成连续分布形式；
  4. **探索调度**：本任务 `entropy_coef = 0.0` 也能解 CartPole；更困难的任务需要熵系数退火 /
     动作噪声 / count-based 探索。
* **本任务的定位**：它不是"最强算法"，而是把**策略梯度定理的每一个符号**
  都变成可运行、可断言、可消融的代码 —— 之后的 A2C/PPO/GAE 都只是在这条主线上加部件。

