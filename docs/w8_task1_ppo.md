# W8 Task 1：从零实现连续空间 PPO（PPO-Clip + GAE-λ）

## 1. 目标

W6 的 REINFORCE 是**回合级蒙特卡洛**策略梯度：一个回合只能更新一次、方差极大、只能处理
离散动作（`Categorical`）。W8 把它升级成现代 on-policy 算法的标准形态：

* **Critic 学 `V_φ(s)`**，用 **GAE-λ** 把"整条回合的回报"换成**多步自举的优势**（偏差/方差可调）；
* **一次采样、多次复用**：攒 `rollout_steps = 2048` 步，再做 `update_epochs = 10` 轮
  mini-batch 更新（32 × 64），样本效率大幅提升；
* **PPO-Clip** 用重要性采样比值的双侧裁剪代替 TRPO 的二阶约束，让"重用同一批数据"仍然安全；
* **连续动作空间**：策略输出对角高斯 `π_θ(a|s) = N(μ_θ(s), σ_θ²)`，`σ = exp(log_std)` 可学习。

环境是 **Pendulum-v1**：状态 `[cos θ, sin θ, θ̇]`（3 维）、动作是**单维力矩** `[-2, 2]`、
单回合 200 步、每步奖励 `−(θ² + 0.1θ̇² + 0.001u²)`（恒 ≤ 0，越接近 0 越好）。

> **思考题**：在连续动作空间里，如果动作被截断成 `clip(action, -2, 2)` 再交给环境，
> 计算 `log_prob` 时要注意什么？—— 完整回答见第 3 节。

交付物（全部为新增文件，未改动 W3/W4/W6 任何代码）：

| 文件 | 行数 | 说明 |
| --- | --- | --- |
| `src/ppo.py` | 1366 | 核心库：`Actor` / `Critic` / `ActorCritic`、`RolloutBuffer`、`compute_gae`、`policy_loss_clip` / `value_loss` / `ppo_losses`、`ppo_update`、`RolloutCollector`、`evaluate_policy` / `explained_variance`、`RunningMeanStd`、`build_actor_critic` / `build_ppo_optimizer` |
| `configs/ppo_config.yaml` | 85 | 作业指定超参（seed 42 / cuda / Pendulum-v1 / 3→64→1×2 / 200000 步 / 2048 / γ 0.99 / λ 0.95 / ε 0.2 / K 10 / batch 64 / lr (3e-4, 1e-3) / clip 0.5 / target −200）+ 补充字段（归一化、消融开关、评估、续训） |
| `experiments/train_ppo.py` | 624 | 训练入口：复用 `src/engine.py` 的 `RunContext` / 日志 / TensorBoard / 检查点 / `--set` / `--resume`，含 `as_bool` / `optional_float` / `rolling_mean` / `check_action_bounds` |
| `experiments/export_tb_curves.py` | 233 | 把 TensorBoard 标量导出为多面板 PNG + 长表 CSV（9 个面板 / 37 个 tag），横轴统一为累计交互步数 |
| `tests/test_ppo.py` | 1036 | **38 个单元测试**（`python tests/test_ppo.py` 直接可跑，也兼容 pytest），全部通过 |
| `docs/figures/w8_ppo_curves.png`、`docs/figures/w8_ppo_scalars.csv` | — | 本次正式运行的曲线与原始标量 |
| `src/__init__.py` | 160（+42） | 统一导出 PPO 符号（与 `reinforce` 同名但语义不同的 `value_loss / make_env / env_max_steps / evaluate_policy / count_parameters` 沿用 W6 约定不重复导出） |


## 2. 算法数学（本文用到的全部公式）

**优化目标**（最大化期望折扣回报）：

```
J(θ) = E_{τ ~ π_θ} [ Σ_t γ^t r_t ]
```

**同策略重要性采样比值**（`θ_old` = 采样时的参数）：

```
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) = exp( logπ_θ(a_t|s_t) − logπ_{θ_old}(a_t|s_t) )
```

**PPO-Clip 代理目标**（`ε = 0.2`）：

```
L^{CLIP}(θ) = − E_t [ min( r_t·Â_t , clip(r_t, 1−ε, 1+ε)·Â_t ) ]
```

`Â_t > 0`（好动作）时概率最多涨到 `1+ε` 倍；`Â_t < 0`（坏动作）时最多压到 `1−ε` 倍。
超出裁剪区间的样本**梯度为 0**（`ratio` 里额外的 `∂logπ/∂θ` 被 `min` 的常数分支挡住），
这正是"信任域"的效果 —— 也是它能安全地反复用同一批数据 10 轮的原因。

**GAE-λ**（广义优势估计，`t` 从 `T−1` 倒序递推）：

```
δ_t      = r_t + γ·V_φ(s_{t+1})·(1 − done_t) − V_φ(s_t)      # TD 残差
Â_t      = δ_t + (γλ)·(1 − done_t)·Â_{t+1}                   # λ=0 → TD(0)；λ=1 → 蒙特卡洛
Target_t = Â_t + V_φ(s_t)                                     # 价值网络的回归目标
```

**价值损失与总损失**：

```
L^{VF}(φ) = ½ · E_t [ (V_φ(s_t) − Target_t)² ]
L^{total} = L^{CLIP}(θ) + c₁·L^{VF}(φ) − c₂·H[π_θ]            # 本作业 c₁ = 0.5，c₂ = 0（σ 可学习）
H[N(μ, σ²)] = Σ_j ( ½·log(2πe) + log σ_j )                    # 对角高斯熵（与 μ 无关）
```

**一次更新的数据流**：

```
env ──π_θold 采样 2048 步──> buffer{s, a_raw, logπ_old, r, terminated, truncated, V}
                                    │
                      倒序递推（CPU float64）──> δ_t, Â_t, Target_t
                                    ↓
        ┌── 重复 K=10 轮 ──────────────────────────────────────────┐
        │  randperm(2048) → 32 个 mini-batch（每个 64 条）          │
        │  前向 → r_t, L^CLIP, L^VF, H → L^total                    │
        │  优势标准化（mini-batch 内零均值单位标准差）              │
        │  backward → clip_grad_norm_(0.5) → Adam.step()            │
        └───────────────────────────────────────────────────────────┘
```

**为什么用一个 Adam + 两个 param group**：Adam 是**逐参数**更新的（每个参数各自维护
`exp_avg / exp_avg_sq / step`，更新量只用自己的梯度与该 group 的学习率），因此
"一个 Adam + 两组参数（lr 3e-4 / 1e-3）"与"两个独立 Adam"在数学上完全等价 ——
`test_two_param_groups_equiv_two_separate_adam` 逐步逐参数比对了两者的参数轨迹
（误差 < 1e-6），而检查点里只需要保存一份优化器状态，断点续训逻辑与 W3/W4/W6 完全复用。

## 3. 思考题：`clip(action, -2, 2)` 时 `log_prob` 要注意什么？

### 3.1 结论（一句话）

**`a_env = clip(a_raw, −2, 2)` 只能交给 `env.step()`；`log_prob` / 比值 / 更新必须用截断前的
`a_raw`。** 本实现的 buffer 字段因此直接命名为 `actions_raw`，`a_env` 只在
`RolloutCollector.collect` 里进 `env.step`。

### 3.2 三个理由（机理）

**理由 1：把 `a_env` 代进高斯密度，算出来的已经不是概率密度了。**
`a_env = clip(a_raw)` 的分布是"连续部分 + 两个点质量"的混合：
`P(a_env = −2) = P(a_raw < −2)`、`P(a_env = +2) = P(a_raw > 2)`。点质量**没有密度**，
而 `logπ(a_env)` 却会把 `±2` 当成普通高斯点算出有限值。策略梯度定理的推导
`∇J = E_{a~π}[∇logπ(a)·Â]` 要求"积分变量与密度描述同一个随机变量"，替换成
`a_env` 后这个前提就没了 —— 得到的既不是 `∇J`，也不是任何目标函数的梯度。

**理由 2：用 `clip(a)` 会产生与奖励无关的系统性伪梯度。**
高斯得分函数对 `μ` 的期望恒为零（这也是 REINFORCE 能"无基线"就无偏的原因）：

```
正确： E_{a~N(μ,σ²)}[ ∂logπ(a)/∂μ ] = E[ (a − μ)/σ² ] = 0
错误： E[ ∂logπ(clip(a))/∂μ ]        = ( E[clip(a)] − μ ) / σ²
```

只要 `μ` 越过边界，`E[clip(a)]` 就锁死在边界值上，伪梯度 ≈ `(边界 − μ)/σ²`
把 `μ` **硬拽向动作边界**：它与奖励完全无关，量级却与 `1/σ²` 成正比。
单测实测（`μ = ±5, σ = 0.1`，20 万样本）分别是 **∓300**，而正确实现的梯度均值
≈ 0（只有 `1/(σ√n)` 量级的采样噪声）—— 相差**两个数量级以上**。一旦策略均值接近
边界，这个伪力就会主导 `×Â_t` 之后的更新方向。

**理由 3：PPO "第 0 个 epoch 的比值必须等于 1"会直接坏掉。**
更新时要在新参数下重算 `logπ_θ(a_t)`，用来算 `r_t = exp(logπ_new − logπ_old)`。
如果采样时把 `a_env` 存进 buffer：

* `logπ_old` 是按 `a_raw` 算的（采样时就该按 `a_raw` 算），而重算时却代入 `a_env`，
  两者算在**两个不同的动作点**上 ⇒ 参数还没动，`r_t ≠ 1`；
* `clip(r_t, 0.8, 1.2)` 的裁剪区间是以 `r_t = 1` 为基准的，基准错了之后
  `min(r·Â, clip(r)·Â)` 既可能把"没变化"的样本误判为越界，也可能掩盖真正的越界；
* 实测（`log_std_init = 1.5`，σ ≈ 4.5，56% 的动作越界）把 `a_env` 当动作时
  `ratio_mean ≈ 2.95`、`clip_fraction ≈ 0.28`，**代理目标反而"显得更优"**
  （`L^CLIP = −1.08 < −1.00`）—— 优化器会在追逐一个虚构的收益。

### 3.3 量化证据（`tests/test_ppo.py`，全部通过）

| 测试 | 断言的内容 | 实测 |
| --- | --- | --- |
| `test_buffer_keeps_raw_action_while_env_gets_clipped_action` | buffer 存 `a_raw`；`a_env == clip(a_raw, −2, 2)`；`μ` 允许越界 | `μ ≈ 5` 时 `a_env ≡ 2`，buffer 里仍是 5 附近的采样值 |
| `test_clipped_action_log_prob_is_biased` | 用 `a_env` 算 `∂logπ/∂μ` 的期望是非零伪梯度；用 `a_raw` 恒为零均值 | `μ=+5 → −300`、`μ=−5 → +300`；`a_raw` 侧 `< 0.5`（纯采样噪声） |
| `test_ratio_is_one_only_when_raw_action_is_stored` | 存 `a_raw` ⇒ `r_t ≡ 1`；存 `a_env` ⇒ `\|r−1\|` 明显偏离 | 存 `a_raw`：`max\|r−1\| < 1e-5`；存 `a_env`：`> 0.1` |
| `test_ppo_losses_uses_stored_raw_action` | 传 `a_raw` 时 `L^CLIP = −mean(Â) = −1`（精确）；传 `a_env` 时目标错位 | `−1.0000` vs `−1.078`（重跑因采样略有波动）、`ratio_mean 2.95` |
| `test_mean_may_exceed_bounds_but_env_action_is_clipped` | `μ(s)` 允许越界（高斯定义在 ℝ 上），只有执行侧被截断 | `μ > 2` 而 `action_env ≤ 2` |
| `test_ratio_is_one_only_when_raw_action_is_stored`（同测第二段：σ = 4.5 档） | σ 越大 ⇒ 越界样本越多 ⇒ 错误做法的偏离越大 | 用 `a_env` 算 `max\|r−1\| > 0.1`（该档 `clip_fraction ≈ 0.28`），用 `a_raw` 算 `< 1e-5` |
| 采样器/评估（§8） | 喂给 `env.step` 的动作永远在 `[-2, 2]` 内 | 端到端 512 步全部满足 |

> 一个实现上的坑（如实记录）：`ppo_losses` 返回的是**总损失** `L^CLIP + 0.5·L^VF`，
> 用默认 `value_coef=0.5` 去断言"精确等于 −1"会差 1e-3 量级；显式传 `value_coef=0.0`
> 后逐位吻合 —— 这反过来验证了三项损失是**可加**的、且 `Â` 是常数权重
> （`test_policy_loss_advantage_is_detached`）。

**训练日志里的旁证**：正式运行最后阶段 `a_raw` 的越界比例是 **10.5%**（TensorBoard 面板
"动作截断比例（思考题诊断）"）—— 每 10 步就有 1 步 `a_env ≠ a_raw`。如果 `log_prob` 用错，
这 10% 的样本会持续注入伪梯度；而实际训练收敛到 −199.9、`σ` 从 0.61 平滑压到 0.37，
没有出现"μ 贴边 / 熵崩塌"的病态。

### 3.4 另一种"策略自己受限"的正确做法（对照）

如果希望策略**自身**只输出合法动作，标准做法是 **tanh 压缩高斯**：

```
a_env = low + (high − low)/2 · tanh(a_raw)
logπ(a_env|s) = log N(a_raw; μ, σ²) − Σ_j log|∂a_env_j/∂a_raw_j|
```

必须补上 Jacobian 项 `log|da_env/da_raw|`，否则密度没有归一化。本作业要求的是"截断"（clip），
且 tanh 得到的是开区间（还需额外缩放）、实现更复杂，因此本模块只用 clip；同时
`build_actor_critic` 里**显式拒绝**未知的 `action_transform`（配错就报错，而不是静默变差），
以后真要换成 tanh 压缩也必须先把 Jacobian 项补上。

## 4. 关键实现细节与踩过的坑

### 4.1 惰性 reset 与"边界自举"（Pendulum 的灵魂）

Pendulum-v1 **永远不会 `terminated`**，回合结束只有 TimeLimit 的 `truncated = True`（第 200 步）。
两种处理方式在数学上不同：

```
done_t = terminated_t                        # 本实现（bootstrap_on_truncation=True）
done_t = max(terminated_t, truncated_t)      # 关掉开关时的做法
```

* 到点只是"被时间截断"，不是"掉进吸收态"，所以**继续自举** `γ·V(s_200)` 才是无偏的
  （等价于把任务看成"永不终止的连续控制"；CleanRL 的 `use_proper_time_limits`）。
* 采样器用**惰性 reset**：只有在"上一步真的结束了回合"时，才在**下一个动作之前** `reset`。
  于是 rollout 边界处的自举值 `V(s_T)` 来自"真正的 `s_T` 继续往下走"；若像常见实现那样在
  回合结束处立刻 auto-reset，边界处会拿到**新回合初始状态**的价值（错误信号）。
  单测 `test_collector_bootstraps_from_true_next_state` 与
  `test_collector_zero_bootstrap_when_episode_ends_at_boundary` 分别钉住这两种情况。

### 4.2 定长 RingBuffer + 纯函数 GAE

* `RolloutBuffer(capacity=2048)` 的 9 个字段一次性分配；`reset()` 只把写指针归零
  （不清数据，便于异常时看现场）⇒ 98 次 rollout 的分配次数从 ~20 万次降到 0。
* `compute_gae` 是**纯函数**，倒序递推在 **CPU float64** 上做：公式是逐标量递推（T ≈ 2048 次），
  numpy 标量运算约 0.1 µs，而 GPU 上每个标量算子的启动开销是 5–10 µs —— 在 CPU 上算比在
  GPU 上**快两个数量级**，结果再搬回 `cuda` 即可（优势是常数，不需要梯度）。
* 递推正确性不靠"看着对"：`tests/test_ppo.py` 里另写了一份**按定义展开的朴素 O(T²) 实现**
  （每个 `k` 都重新累乘掩码），两者逐位一致才算过。

### 4.3 归一化、缩放与"防止静默变差"

| 机制 | 做法 | 单测 |
| --- | --- | --- |
| 状态归一化 | 手写 `RunningMeanStd`（Welford），**先 `update` 再 `normalize`**（SB3 `VecNormalize` 风格）；buffer 里存的就是网络实际吃到的数值 | `test_running_mean_std_matches_numpy_and_roundtrips`（与 numpy 逐位一致 + 存档往返 + 逐样本更新等价） |
| 评估只读 | `evaluate_policy` 不 `update` 归一化统计量，保证"评估分布"与训练末端一致 | `test_evaluate_policy_pendulum_validity_and_reproducibility` |
| `reward_scale` | 只乘进 buffer 的奖励（改善价值回归的数值条件），日志里的回合回报**永远是原始回报** | `test_collector_reward_scale_only_affects_buffer` |
| 动作边界 | `check_action_bounds` 用**环境真实** `action_space` 覆盖配置里的 `action_low/high` 并打印警告（把 Pendulum 写成 `[-1,1]` 这类错不会报错、只会默默变差） | `test_train_ppo_script_helpers` |
| 熵项符号 | `L^total = L^CLIP + c₁L^VF − c₂H`：最小化总损失 ⇒ 最大化熵；`σ` 可学习时默认 `c₂ = 0` | `test_entropy_term_rewards_exploration` |
| 裁剪实现 | `clip(r, 1−ε, 1+ε)` 用 `torch.clamp`，并对"裁剪分支是否挡住梯度"单独断言 | `test_policy_loss_gradient_blocked_when_clipped` |
| KL 早停 | `target_kl` 非空时，某 epoch 平均 `approx_kl` 超标即停止后续 epoch（默认关闭） | `test_ppo_update_kl_early_stop` |

`approx_kl` 用方差更小的 **k3 估计量** `(r − 1) − log r ≥ 0`，而不是 `mean(logπ_old − logπ_new)`。

### 4.4 调试过程中被测试推翻的四个"想当然"（如实记录）

1. **`ppo_losses` 返回的是总损失**，不是 `L^CLIP`。想断言"`r≡1` 时等于 `−mean(Â)`"必须
   `value_coef=0.0`（或直接看 `stats["policy_loss"]`）。
2. **用 `clip(a)` 算 `∂logπ/∂μ`，伪梯度的方向是"把 μ 拽向边界"而不是"推到界外"**：
   `E[clip(a)]` 在界外恒等于边界值 ⇒ 期望 = `(边界 − μ)/σ²`。所以 `μ = +5` 给 **−300**
   （下降方向 ⇒ 拽回 +2），`μ = −5` 给 **+300**。测试里同时写上正负两档才不会被符号误导。
3. **clipped value loss 不是"更小的损失"，而是悲观下界**：
   `½·max(unclipped, clipped)` ≥ 未裁剪值；当裁剪分支占优（新 `V` 一次移动超过 ε）时
   **该方向的梯度恰好为 0**。验证只能这样断言，不能指望"损失变小"。
4. **短 rollout 的价值损失不能当"Critic 是否在学"的判据**：512 步 × 2 epoch 只有 8 次更新，
   且相邻 rollout 的价值目标本身在剧烈变化（回报从 −1200 涨到 −300），损失完全可能先降后升。
   端到端测试因此改成"**固定同一批目标**做监督式回归"：先记下 `V(s)` 与 `Target` 的
   `½MSE`，跑 160 次更新（20 epoch × 8 mini-batch）后必须降到 60% 以下
   （实测 3582.8 → 1410.2，即降到 39.4%；`explained_variance` 从 0.000 升到 0.261）。

## 5. 正式实验（默认配置，seed 42）

```
python experiments/train_ppo.py                     # 设备 cuda，约 8.8 min
```

| 项目 | 值 |
| --- | --- |
| 环境 / 状态 / 动作 | `Pendulum-v1` / 3 维 `[cos θ, sin θ, θ̇]` / 1 维力矩 `[-2, 2]` |
| 网络 | Actor 与 Critic **各自独立**的 `MLP(3 → 2×64 → 1)`（ReLU）；Actor 4482 参数（含 `log_std`），Critic 4481 参数 |
| 动作分布 | `N(μ(s), σ²)`，`log_std` 可学习，初值 −0.5（σ ≈ 0.61），夹紧 `[-20, 2]` |
| 归一化 | `RunningMeanStd(3)`，训练中在线更新（最终 `count = 194652`，`std = [0.73, 0.52, 3.09]`） |
| 超参 | γ 0.99 / λ 0.95 / ε 0.2 / K 10 / batch 64 / rollout 2048 / lr (3e-4, 1e-3) / grad-clip 0.5 / 优势标准化 on / 熵系数 0 |
| 预算与判据 | 200000 步 ≈ 98 个 rollout；每 5 个 rollout 用 20 个确定性回合评估；评估均值 ≥ −200 即提前停止 |
| **实际结果** | **95 个 rollout / 194560 步 / 527 s**，最后 100 回合训练均值 **−207.36**，独立评估 **−199.91 ± 200.16**（最优 −199.905） ⇒ **达标提前结束** |

### 5.1 学习曲线（每 5 个 rollout 一次独立评估，20 回合均值）

| 步数(k) | 10.2 | 20.5 | 30.7 | 41.0 | 51.2 | 61.4 | 71.7 | 81.9 | 92.2 | 102.4 | 112.6 | 122.9 | 133.1 | 143.4 | 153.6 | 163.8 | 174.1 | 184.3 | **194.6** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 评估回报 | −1267 | −1203 | −1103 | −1002 | −922 | −819 | −671 | −544 | −243 | −230 | −206 | −213 | −213 | −205 | −205 | −212 | −209 | −204 | **−199.9** |

（共 19 个评估点，步数 10.24k→194.56k，每 5 个 rollout 一次；完整数值见
`docs/figures/w8_ppo_scalars.csv` 的 `eval/mean_return` 行。）

训练侧的关键转折点（TensorBoard `train/*`）：

| rollout | 步数 | 训练回合均值 | 滑动(100) | L^CLIP | L^VF | 熵 | approx_kl | clip 占比 | grad_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2048 | −1247 | −1247 | −0.0019 | 2565.5 | 0.916 | 0.0019 | 0.015 | 490.6 |
| 20 | 40960 | −976 | −1046 | −0.0053 | 227.2 | 0.804 | 0.0069 | 0.085 | 296.1 |
| 40 | 81920 | −603 | −680 | −0.0068 | 314.3 | 0.714 | 0.0135 | 0.163 | 277.4 |
| 60 | 122880 | −153 | −272 | −0.0014 | 318.1 | 0.518 | 0.0145 | 0.135 | 154.4 |
| 80 | 163840 | −293 | −252 | −0.0018 | 720.4 | 0.479 | 0.0082 | 0.068 | 273.5 |
| 95 | 194560 | −334 | **−207** | −0.0031 | 535.6 | 0.416 | 0.0092 | 0.092 | 222.9 |

读法：
* **`L^CLIP` 一直在 ±0.01 内** —— 这是 PPO 的正常形态，不是"没学"：它被 mini-batch 优势
  标准化固定住了量级（`|L| ≈ O(1)` 之外大量正负抵消），真正的信息在 `approx_kl`
  （0.002 → 0.009，全程峰值 0.0155，始终远离会引发崩溃的 0.02+）与 `clip 占比`（0.015 → 0.09）上；
* **Critic 先快后慢**：`L^VF` 从 2565 掉到 200–300 量级（前 20 个 rollout），之后随回报尺度
  变化上下波动，价值解释度从 0.09 升到 **0.772**；
* **σ 单调收紧** 0.61 → 0.37、熵 0.92 → 0.42：策略从"乱试"变成"有把握"，但没有崩塌迹象
  （`log_std` 远未碰到下界 −20）；
* 训练滑动均值（−207）与独立评估（−199.9）**互相印证**，没有"评估调参"的空间；
* `σ` 只降了约 40% 就停止 —— 因为 200 步折扣任务里"确定性最优"本来就需要一定探索量，
  PPO 把 `σ` 学成了 0.37 这个"刚好够用"的值（`log_std` 是**可学习参数**，不是退火出来的）。

### 5.2 曲线导出（验收项：导出 TensorBoard 曲线）

```
python experiments/export_tb_curves.py --logdir logs --out-dir docs/figures --prefix w8_ppo
# → docs/figures/w8_ppo_curves.png（9 个面板 / 14 条曲线，344 KB）
# → docs/figures/w8_ppo_scalars.csv（37 个 tag / 2683 行，长表 tag,step,value，102 KB）
# 终端同时打印每个关键 tag 的"首个 / 最后 / 最优"摘要
```

9 个面板：学习曲线（训练回报 + 滑动均值）、独立评估回报（含 −200 参考线）、`L^CLIP`、
`L^VF`、熵与 σ、KL 与 clip 占比、梯度范数、价值解释度与单步奖励、
**动作截断比例**（思考题诊断）；横轴统一为**累计交互步数**（与 RL 论文曲线一致），
`--smooth N` 可开滑动平均，`--all` 则变成"每个 tag 一个面板"（37 个，含
`train/last_value`、`buffer/*` 等全部标量），CSV 始终导出全部 37 个 tag。

## 6. 消融实验（各 40000 步 = 20 个 rollout，其余与默认配置完全一致）

```
# 默认配置（= 对照组）
python experiments/train_ppo.py --set experiment.name=ppo_abl_control
# 关掉 truncated 自举
python experiments/train_ppo.py --set experiment.name=ppo_abl_nobootstrap training.bootstrap_on_truncation=false
# 关掉状态归一化
python experiments/train_ppo.py --set experiment.name=ppo_abl_nonorm model.normalize_obs=false
# 共享主干（单层）
python experiments/train_ppo.py --set experiment.name=ppo_abl_shared model.shared_backbone=true
```

（四条都额外加了 `training.total_timesteps=40000`，即 20 个 rollout。CLI 参数只有
`--config` / `--set KEY=VALUE` / `--device` / `--seed` / `--resume` / `--snapshot-every` /
`--no-tb` / `--no-eval` 八个，任何配置项都可以用 `--set` 点号路径覆盖，
`experiment.name` 决定 TensorBoard / 检查点的目录名。）

| 变体 | 独立评估回报（**最优权重**，20 回合） | 训练滑动(100) | σ | L^VF | 价值解释度 | grad_norm | 采样越界比例 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **对照（默认）** | **−971.90 ± 228.94** | −1051.63 | 0.547 | 1270.7 | 0.498 | 562.4 | 0.009 |
| `bootstrap_on_truncation=false` | −1074.29 ± 252.56 | −1116.16 | 0.560 | 1503.6 | 0.772 | 694.6 | 0.001 |
| `normalize_obs=false` | **−867.49 ± 204.85** | −956.38 | 0.588 | 801.6 | 0.464 | 1017.5 | 0.011 |
| `shared_backbone=true` | −1252.08 ± 407.75 | −1231.17 | 0.565 | 2001.1 | 0.668 | 1091.7 | 0.001 |

四条曲线的评估轨迹（rollout 5 / 10 / 15 / 20，即 10k/20k/30k/40k 步）：

| 变体 | 10k | 20k | 30k | 40k | 趋势 |
| --- | --- | --- | --- | --- | --- |
| 对照 | −1267 | −1203 | −1103 | −972 | 稳定下降 |
| 无 truncated 自举 | −1264 | −1208 | −1157 | −1074 | 下降更慢，40k 落后 ~100 |
| 不做状态归一化 | −1200 | −1010 | −867 | −890 | **起步最快**，但 30k 后开始回落 |
| 共享主干 | −1283 | −1252 | −1275 | −1260 | **几乎不学** |

（上表主列取**最优权重**的评估值（对照与无自举的最优点恰好是第 20 个 rollout，无归一化
最优点在第 15 个 rollout，共享主干在第 10 个），本表的四个点则是**每次评估的原始值** ——
所以"无归一化"一行会出现 −867.5（最优，30k）与 −890.3（末次，40k）两个数。）

### 6.1 结论 1：`bootstrap_on_truncation=false` 更差（与理论一致）

40k 步时落后对照组约 **102 分**，且训练侧的诊断同步变差：价值损失 1503 vs 1271、
梯度范数 695 vs 562。原因是**优势估计有系统性偏差**：Pendulum 永远只在第 200 步
`truncated`，关掉开关等价于把 `V(s_200) ← 0` 当成真值，于是每个回合最后一步的 TD 残差
`δ_199 = r_199 + 0 − V(s_199)` 人为地变得更负（回合末的 `V` 本应是 −5 左右，现在被当成 0），
这份偏差会沿着 GAE 递推**向前污染整条回合**（越靠回合开始衰减越多，系数 `(γλ)^k`）。
有趣的是"无自举"那组的 `价值解释度` 反而更高（0.772 vs 0.498）—— 因为它要拟合的目标
恰好是"到回合末截断"的短视回报，方差更小、更好拟合；这说明**解释度高≠策略学得好**，
评估回报才是唯一可信的判据。

### 6.2 结论 2：`normalize_obs=false` 在这个小任务上并没吃亏（与直觉相反，如实记录）

它在 40k 步时甚至**反超对照组约 104 分**（−867.5 vs −971.9），而且 10k 步时就已经
领先 67 分。三点解释/提醒：

1. **Pendulum 的状态只有 3 维、量纲差异只有约 6 倍**（实测 `θ̇` 的 std 3.09 vs `cos θ` 的 0.73、
   `sin θ` 的 0.52，见正式运行的归一化统计），
   而 Adam 本身就是**逐参数自适应尺度**的优化器，剩下的这点"输入尺度失配"基本被它吸收 ——
   证据是 `grad_norm` 大了近 1 倍（1017 vs 562）却没崩，训练曲线反而更陡。
2. **归一化有一个"热身代价"**：前几个回合 `RunningMeanStd` 的统计量还很粗糙
   （count 从 0 涨到 4 万），归一化本身就是个**漂移的输入分布**，Critic 要追着它跑；
   对照组的 `L^VF`（1270）比无归一化组（802）高，正是这个"移动靶"造成的。
3. **但这是"4 万步 + 单一随机种子"的比较，噪声很大**：20 回合评估的标准差就有 200–410，
   同一变体相邻两次评估常差 50–100 分（如对照组 20k → 30k 刚好差 100 分）。要看真正的
   差距需要 3–5 个种子 × 20 万步，
   而正式 200k 运行里归一化组的收敛（−199.9，σ 0.61→0.37）完全健康。
   ⇒ 结论只能写成：**归一化在小状态空间 + Adam 的组合下收益不明显（本次甚至略负），
   但它是"无害的保险"**，在状态维度高、量纲差异更极端（如图像/力矩混搭）时更关键，
   所以默认保留 `normalize_obs=true`，并把这个反直觉现象如实记录。

### 6.3 结论 3：`shared_backbone=true` 明显更差（20 个 rollout 几乎没学）

评估回报一直卡在 −1250 附近（最好的一次才 −1252），价值损失 2001（四组里最差）。
两个可解释的原因：

1. **容量变了**：`num_layers=2` 时共享主干只有 **1 层 64**（`MLP(3 → 64×1) + 双头`，
   总参数 4547），而独立网络是 `3 → 64 → 64 → 1` **两套**（4482 + 4481 = 8963）。
   要共享主干又保持容量，应该把深度加上去（如 `3 → 2×64` 主干 + 双头）。
2. **学习率被"共用的那部分"拉低**：主干参数归策略组，按 `lr_actor = 3e-4` 更新
   （日志里 "策略参数张量 7 个 / 价值参数张量 2 个" —— 价值头只剩 2 个张量），
   而独立网络里 Critic 用的是 `lr_critic = 1e-3`。共享主干下 **Critic 的表示学习慢了 3.3 倍**，
   而 PPO 的 `Â` 质量直接依赖 Critic ⇒ 策略也跟着变慢。这不是"共享一定错"，
   而是"共享时必须重新调 lr 与深度"。

**消融小结**：`bootstrap_on_truncation` 与 `shared_backbone` 两项的差异**方向与理论预测一致**
且幅度可观（100 / 280 分）；`normalize_obs` 在本任务上差异不显著甚至反向，需更严谨的多
种子实验才能定论 —— 我们把结论与不确定性都写出来，而不硬凑一个"每项消融都变差"的故事。

## 7. 单元测试（38 个，`python tests/test_ppo.py` 全部通过，约 14 s）

测试脚本自带 `main()`：**不依赖 pytest** 也能直接跑（也可以用 `pytest tests/test_ppo.py -q`）。
每个小节测一类契约，命名即断言内容：

| 小节 | 测试 | 钉住的契约 |
| --- | --- | --- |
| §1 数学基元 | `test_handwritten_gaussian_log_prob_matches_torch`、`test_handwritten_entropy_matches_torch`、`test_std_is_exp_of_clipped_log_std` | 手写 `log N(a;μ,σ²)` / 对角高斯熵与 `torch.distributions` **逐位一致**；σ 的夹紧区间生效 |
| §2 动作截断（思考题） | `test_mean_may_exceed_bounds_but_env_action_is_clipped`、`test_buffer_keeps_raw_action_while_env_gets_clipped_action`、`test_clipped_action_log_prob_is_biased`、`test_ratio_is_one_only_when_raw_action_is_stored` | μ 允许越界；buffer 存 `a_raw`；用 `a_env` 算得分函数会产生 ≈∓300 的伪梯度；只有存 `a_raw` 才能保证 `r_t ≡ 1` |
| §3 缓冲区 | `test_rollout_buffer_write_read_and_capacity`、`test_minibatches_cover_each_sample_exactly_once` | RingBuffer 的 9 个字段形状/容量；`randperm` 分块后**每个样本恰好被访问一次**（不漏不重） |
| §4 GAE | `test_gae_matches_naive_ot2_expansion`、`test_gae_lambda_zero_is_td0`、`test_gae_lambda_one_gamma_one_is_monte_carlo`、`test_gae_done_mask_stops_advantage_leakage`、`test_gae_truncation_bootstrap_flag_matters` | 与朴素 O(T²) 展开逐位一致；λ=0 ⇒ 一步 TD；λ=γ=1 ⇒ 蒙特卡洛；`done` 处优势不跨回合泄漏；`truncated` 自举开关可观测 |
| §5 损失 | `test_policy_loss_matches_handwritten_formula`、`test_policy_loss_ratio_one_is_negative_advantage_mean`、`test_policy_loss_gradient_blocked_when_clipped`、`test_policy_loss_advantage_is_detached`、`test_advantage_normalization_is_zero_mean_unit_std`、`test_value_loss_is_half_mse_and_clip_option`、`test_entropy_term_rewards_exploration`、`test_ppo_losses_uses_stored_raw_action`、`test_explained_variance_extremes` | 与手写公式一致；裁剪分支**梯度为 0**；`Â` 是 stop-gradient 的常数权重；优势标准化零均值单位方差；价值裁剪 = 悲观下界；熵项方向正确；`−mean(Â)` 的精确值 |
| §6 优化器等价性 | `test_two_param_groups_equiv_two_separate_adam` | 一个 Adam + 两组 lr ≡ 两个独立 Adam，逐参数轨迹误差 < 1e-6 |
| §7 `ppo_update` | `test_ppo_update_with_zero_lr_keeps_ratio_one`、`test_ppo_update_changes_params_and_reports_finite_stats`、`test_gradient_clipping_shrinks_parameter_step`、`test_ppo_update_kl_early_stop` | lr=0 ⇒ 参数不动、`r_t ≡ 1`；K 轮 × mini-batch 全覆盖；`clip_grad_norm` 真的缩小了步长；`target_kl` 早停只影响后续 epoch |
| §8 采样/评估/归一化 | `test_running_mean_std_matches_numpy_and_roundtrips`、`test_collector_bootstraps_from_true_next_state`、`test_collector_zero_bootstrap_when_episode_ends_at_boundary`、`test_collector_reward_scale_only_affects_buffer`、`test_evaluate_policy_pendulum_validity_and_reproducibility` | 手写 Welford 与 numpy 一致 + 存档往返；rollout 末尾自举取**真实下一个状态**；边界结束（如 2048 步刚好收尾）时自举为 0；`reward_scale` 不污染日志回报；评估可复现且只读 |
| §9 端到端 | `test_ppo_solves_quadratic_bandit`、`test_pendulum_short_training_is_stable_and_critic_learns` | 自造 `QuadraticBanditEnv`（最优动作 1.5）上 **μ 收敛到 1.5 ± 0.25**；Pendulum 短训不崩 + Critic 在同一批目标上**损失降到 60% 以下**（3582.8 → 1410.2） |
| §10 配置/脚本 | `test_config_file_matches_assignment_spec`、`test_train_ppo_script_helpers`、`test_export_tb_curves_writes_png_and_csv` | `configs/ppo_config.yaml` 与作业给定的每一条超参**逐个断言**；`train_ppo.py` 的 `as_bool` / `optional_float` / `rolling_mean` / `check_action_bounds` 行为；导出器真的产出 PNG + CSV |

## 8. 运行方式与产物

```bash
# 1) 训练（默认 200000 步，命中 target_reward=-200 会提前结束）
python experiments/train_ppo.py
#    常用变体
python experiments/train_ppo.py --device cpu                     # 强制 CPU（同一份代码可跑）
python experiments/train_ppo.py --seed 0 --set experiment.name=ppo_seed0
python experiments/train_ppo.py --resume auto                    # 断点续训（自动找最新检查点）
python experiments/train_ppo.py --no-tb --no-eval                # 冒烟测试用

# 2) 看曲线
tensorboard --logdir logs
# 3) 导出曲线（PNG + CSV）
python experiments/export_tb_curves.py --logdir logs --out-dir docs/figures --prefix w8_ppo
# 4) 单元测试（38 个）
python tests/test_ppo.py
```

产物（每次运行一个带时间戳的目录，命名取 `experiment.name`）：

```
logs/ppo_pendulum_20260918_142705/
├── events.out.tfevents.<...>      # TensorBoard（37 个标量 tag，含 hparams 文本）
└── train.log                      # 完整训练日志（含每次评估与最终汇总）
logs/checkpoints/ppo_pendulum_20260918_142705/
├── ppo_pendulum_best.pt           # 监控指标（评估平均回报，mode=max）最优
└── ppo_pendulum_last.pt           # 最近一次（含优化器状态 / 归一化统计 / 步数 / RNG，可续训）
```

检查点里保存：模型权重、优化器状态、`steps_total`、`rollout_index`、`RunningMeanStd` 的
`count/mean/M2`、Python/NumPy/PyTorch 的 RNG 状态、以及完整配置字典 —— 因此
`--resume auto` 之后的曲线与日志是**连续**的（复用原运行目录）。

## 9. 验收对照与结论

| 作业要求 | 实现 | 证据 |
| --- | --- | --- |
| 从零实现 PPO-Clip（不调用 SB3 等现成 RL 库） | `src/ppo.py` 全手写（网络/高斯策略/GAE/损失/更新/采样器），只依赖 torch + gymnasium + numpy | `src/ppo.py` 无 `stable_baselines3` / `sb3` 依赖；`grep -r "sb3"` 为空 |
| 连续动作空间 | 对角高斯策略 `N(μ(s), σ²)`，`log_std` 可学习且夹紧 | §1 三个数学单测与 `torch.distributions` 逐位一致 |
| 能跑通并**达标** | Pendulum-v1 上 **独立评估 −199.91**（20 回合均值动作，std 200.16），194560 步提前命中 −200 | 第 5 节表格 + `logs/.../train.log`；曲线 `docs/figures/w8_ppo_curves.png` |
| GAE-λ 正确 | 倒序递推 + `done` 掩码 + `truncated` 自举开关 | §4 五个单测（含与朴素 O(T²) 展开逐位一致） |
| 动作截断时 `log_prob` 的正确处理（思考题） | buffer 存 `actions_raw`，`a_env = clip(a_raw)` 只进 `env.step` | 第 3 节 + §2 四个单测（伪梯度 ≈∓300、`r_t ≡ 1`） |
| 导出 TensorBoard 曲线 | `experiments/export_tb_curves.py` | `docs/figures/w8_ppo_curves.png`（9 面板 / 14 曲线）、`w8_ppo_scalars.csv`（37 tag / 2683 行） |
| 单元测试 | 38 个测试 / 10 个小节 | `python tests/test_ppo.py` → `通过 38/38`，`全部通过 ✓` |
| 消融 / 敏感性 | 3 项消融（自举、归一化、共享主干）× 40k 步 | 第 6 节四张表的对照数据 |

**与 W6 REINFORCE 的对照（算法机制层面；W6 跑的是 `CartPole-v1` 离散任务，W8 是
`Pendulum-v1` 连续任务，两者不构成同环境性能对比）**

| | REINFORCE（W6） | PPO（W8） |
| --- | --- | --- |
| 优势估计 | 回合级蒙特卡洛 `Σγ^k r_{t+k}`（方差极大） | GAE-λ（`γλ` 加权自举，可调偏差/方差） |
| 数据复用 | 1 个回合更新 1 次 | 2048 步数据 × **10 轮 × 32 个 mini-batch** |
| 更新约束 | 无（靠小 lr） | PPO-Clip 的双侧裁剪（信任域）+ KL 诊断 |
| 动作空间 | `Categorical`（仅离散） | 对角高斯（连续，力矩 ∈ [-2, 2]） |
| 方差控制 | 无基线（或简单基线） | Critic + 优势标准化 + 梯度裁剪 |
| 结论 | 能解任务但样本效率低、曲线抖动大 | 194.5k 步内稳定达到 −199.9，`approx_kl` 峰值 0.0155 |

**局限与后续**

1. **单种子**：主结果只跑了 seed 42（作业要求），消融是 40k 步的单次运行，评估噪声
   （std ≈ 200）意味着 100 分级别的差异（如状态归一化那组）不能下强结论；严谨做法是
   3–5 个种子并报告均值±标准差。代码已支持 `--seed`，脚本化重复实验即可。
2. **PPO 未做过的增强**：可加 `c₂>0` 的熵奖励（防止 σ 过早塌缩）、学习率退火、
   `value_clip`（已实现但默认关）、obs 的 `clip_obs` 上限（防归一化后出现极端值）、
   以及"PPO-Clip vs PPO-KL penalty"的对照。
3. **共享主干那次失败是超参问题而非结构问题**：想用共享主干应同时加深主干并把主干
   放进 lr 更大的那组（或对主干用中间 lr），这部分留给"网络结构消融"任务。
4. `tests/test_ppo.py` 里的 Pendulum 端到端测试是"短训 + 监督式 Critic 回归"，不是为了
   复现 −200（在 CI 里跑几分钟不现实）；真正的性能验证靠 `experiments/train_ppo.py`。

## 10. 变更范围与复现清单

**改动范围**：本次**只新增文件** —— `src/ppo.py`、`tests/test_ppo.py`、`configs/ppo_config.yaml`、
`experiments/train_ppo.py`、`experiments/export_tb_curves.py`、`docs/w8_task1_ppo.md`、
`docs/figures/w8_ppo_curves.png`、`docs/figures/w8_ppo_scalars.csv`；
唯一被修改的既有文件是 `src/__init__.py`（+42 行：PPO 导入块 + `__all__` 条目，**纯新增**）。
W3/W4/W5/W6 的库代码、配置与测试**一行未动**，并已复跑确认无回归：
`tests/test_reinforce.py` 29/29、`tests/test_diffusion.py` 15/15、`tests/test_action_chunking.py` 18/18。

**从零复现**（GPU ≈ 10 min，纯 CPU 也可跑）：

```bash
python tests/test_ppo.py                                                    # 38/38，≈ 14 s
python experiments/train_ppo.py                                             # 194.5k 步 ≈ 8.8 min → −199.91 ✓
python experiments/export_tb_curves.py --logdir logs --out-dir docs/figures --prefix w8_ppo
```

**一句话结论**：`clip(a, −2, 2)` 的坑只有一条 —— **喂给 `env.step` 的动作是截断后的，
喂给 `log_prob`（以及比值、更新）的动作必须是截断前的**。本实现把这条约束物化成
buffer 字段名 `actions_raw`，并用单测量化了违反它的后果（伪梯度 ≈ ∓300、`ratio_mean 2.95`、
`L^CLIP` 反而"显得更优"）；在 Pendulum-v1 上 194.5k 步稳定收敛到 **−199.91**（目标 −200），
`approx_kl` 峰值 0.0155，全程无崩溃。

