# embodied-arm

Residual RL for robust robot arm trajectory tracking.

Target: IROS 2027 / RA-L submission (March 2027).

> 🇨🇳 **零基础读本（推荐先从这份看起）**：[`docs/零基础读本.md`](docs/零基础读本.md) ——
> 用倒立摆讲清楚"什么是动作块""日志里每个指标是什么意思"，不需要模仿学习基础。

## Roadmap
- [ ] W2-W5: Deep learning basics (PyTorch from scratch)
- [ ] W6-W9: RL from scratch (PPO / SAC)
- [ ] W10-W12: Isaac Lab environment & custom task
- [ ] W13-W16: Main project — residual RL for robust tracking
- [ ] 2027.1-2: Real robot deployment
- [ ] 2027.3: Paper submission

## Hardware
- RTX 2080 Ti (11GB), Ubuntu 22.04, Python 3.10, PyTorch 2.6 (cu124)

## Structure
- `w2_pytorch_basics/` — deep learning fundamentals (single-file toy scripts)
- `docs/` — reports (`docs/reports/`), beginner guide (`docs/零基础读本.md`), notes, figures and archived text logs
- `logs/` — runtime output (TensorBoard events + checkpoints), gitignored and safe to delete

## Behavior Cloning (engineered project)

The single-file script `w2_pytorch_basics/day2_behavior_cloning.py` has been refactored
into a standard, reusable and monitorable algorithm project layout:

```
embodied-arm/
├── configs/                  # Hyper-parameter configs (YAML)
│   ├── bc_config.yaml        # BC experiment config
│   ├── chunking_bc_config.yaml  # Action-chunking BC config (defaults: bc_config.yaml)
│   └── diffusion_config.yaml    # Mini-diffusion-policy config (W4 Task 2)
├── src/                      # Core source code
│   ├── networks.py           # Policy networks (MLP, CNN, GRU) + action-chunk head + factory
│   ├── diffusion.py          # DDPM schedule / noise-prediction net / diffusion policy (W4 T2)
│   ├── datasets.py           # Expert data, chunking dataset & augmentation, closed-loop eval
│   ├── engine.py             # Shared training engine (args/config/run dir/log/TB/train loop/resume)
│   └── utils.py              # Config, logging, TensorBoard, Checkpointer
├── experiments/              # Experiment run scripts
│   ├── train_bc.py           # Behavior cloning training entry point
│   ├── train_chunking_bc.py  # Action-chunking BC training entry point (W4 Task 1)
│   ├── train_diffusion_policy.py  # Mini diffusion policy training entry point (W4 Task 2)
│   └── eval_expert.py        # Expert-controller (oracle) return baseline
├── tests/                    # Plain-stdlib test scripts (no pytest needed)
└── logs/                     # TensorBoard logs & model checkpoints (gitignored)
```

### Configuration (`configs/bc_config.yaml`)

```yaml
experiment:                 # experiment-level settings
  name: "bc_pendulum"       # used to name TensorBoard run dir & checkpoint file
  seed: 42
  device: "cuda"            # auto | cpu | cuda
model:
  state_dim: 3
  action_dim: 1
  hidden_dim: 64
training:
  batch_size: 64
  epochs: 100
  lr: 0.001
  max_grad_norm: 1.0        # gradient clipping threshold
  checkpoint:               # resume / crash-recovery settings
    monitor: val_mse        # metric used to decide the "best" checkpoint
    mode: min               # min | max
    snapshot_every: 0       # archive a snapshot every N epochs (0 = off)
    keep_last_n: 2          # how many archived snapshots to keep
data:
  num_expert_steps: 10000   # total expert steps to collect
  train_ratio: 0.8
paths:
  log_dir: logs
  checkpoint_dir: logs/checkpoints
  resume: null              # null | auto | <checkpoint file or directory>
```

`experiments/train_bc.py` reads this file with **PyYAML** and uses
`experiment.name` for the TensorBoard log directory (`logs/<name>_<timestamp>/`),
the Checkpointer weight filenames (`<name>_best.pt`, `<name>_last.pt`).

### Resume training / crash recovery (W3)

`Checkpointer` stores the **complete** training state — model, optimizer,
scheduler, finished epoch, best metric **and the RNG state of every random
source** (Python / NumPy / Torch CPU / Torch CUDA) — so an interrupted run
resumes bit-exactly where it stopped:

| artefact | when written | purpose |
| --- | --- | --- |
| `<name>_last.pt` | every epoch | resume / crash recovery |
| `<name>_best.pt` | when the monitored metric improves | final weights, closed-loop eval |
| `<name>_epochXXXX.pt` | every `snapshot_every` epochs (rotated, `keep_last_n`) | history |

* Writes are **atomic**: temp file → `fsync` → `os.replace` → directory `fsync`,
  so a `kill -9` never leaves a half-written checkpoint behind.
* Every file carries a **sha256 digest** that is verified on load; damaged or
  truncated files raise `CheckpointIntegrityError` instead of silently loading
  garbage (legacy plain `torch.save` files are still accepted).
* Because the RNG state is restored too, `4 epochs + resume 2 epochs` produces
  **bit-identical** weights to an uninterrupted 6-epoch run.
* The interrupted run's TensorBoard directory is reused, so curves stay continuous.

```bash
# auto-discover the newest checkpoint and continue where it stopped
python experiments/train_bc.py --resume

# resume from a specific file / directory / extend the schedule
python experiments/train_bc.py --resume logs/checkpoints/<run>/bc_pendulum_last.pt
python experiments/train_bc.py --resume logs/checkpoints --set training.epochs=200
```

Interruptions are handled gracefully: the first `Ctrl-C` / `SIGTERM` finishes the
current epoch, saves a checkpoint and exits (printing the exact `--resume`
command to continue); a second signal force-quits. Any unexpected exception
triggers an emergency `save_last()` before the traceback is re-raised. Set
`paths.resume` in the config to make a run resume-on-restart by default.

### Quick start

```bash
pip install -r requirements.txt

# Train with the default MLP policy
python experiments/train_bc.py --config configs/bc_config.yaml

# Switch network / override hyper-params from the command line
python experiments/train_bc.py --config configs/bc_config.yaml --set model.name=gru
python experiments/train_bc.py --config configs/bc_config.yaml --set training.epochs=50
python experiments/train_bc.py --config configs/bc_config.yaml --device cpu --no-eval

# Resume an interrupted run (auto-find the newest checkpoint)
python experiments/train_bc.py --resume

# Monitor training
tensorboard --logdir logs
```

### Action chunking BC (W4)

`experiments/train_chunking_bc.py` extends the same pipeline from "one action per
step" to **action chunks**: the label becomes `[a_t, ..., a_{t+K-1}]` and the network
head grows from `action_dim` to `K * action_dim`, so a single forward pass predicts
`K` future actions and the closed loop **re-plans every `K` steps** instead of every step.

* **Exact windowing** (`build_action_chunks`): sliding window per trajectory,
  never crossing episode boundaries, zero-padded at the tail with a `valid` mask.
* **Identical data split**: `build_chunking_dataloaders` uses the very same
  `default_rng(seed).permutation(N)` + `train_ratio` split, so the train/val
  *states* are sample-by-sample identical to the single-step baseline — only the
  labels differ. The comparison is therefore clean (same data, net, optimizer,
  epochs, augmentation, seed).
* **Backward compatible**: with `model.chunk_size: 1` (default) the network output
  shape and numbers are bit-identical to the pre-W4 single-step BC.
* **Execution modes** (all reported to TensorBoard): `chunk` (open loop, one
  forward per `K` steps), `ema` (time-aligned EMA between overlapping plans,
  ACT-style temporal ensembling; `replan_every < K` required), `receding`
  (re-plan every step, execute the first action of the chunk).

| run | K | execution | forward passes / episode | mean return (20 episodes) |
| --- | --- | --- | --- | --- |
| single-step BC (baseline) | 1 | step-by-step | 200 | -156.95 |
| chunking BC | 4 | open loop | **50** | **-155.58** |
| chunking BC | 4 | EMA | 100 | -155.94 |
| chunking BC | 8 | open loop | **25** | -219.68 |
| chunking BC | 8 | receding | 200 | -158.32 |
| expert controller (oracle) | — | energy shaping + PD | — | -175.00 |

`K=4` matches the single-step baseline with **4x fewer forward passes**; `K=8`
open loop degrades (compounding open-loop error) but recovers to baseline level
once executed receding. Full write-up: `docs/reports/w4_task1_action_chunking.md`.

```bash
# Action-chunking BC (K=8 default, or override K from the CLI)
python experiments/train_chunking_bc.py
python experiments/train_chunking_bc.py --set model.chunk_size=4
python experiments/train_chunking_bc.py --set inference.execution=ema

# Expert-controller (oracle) return baseline, the reference scale for "return"
python experiments/eval_expert.py

# Unit tests (plain stdlib, prints a PASS/FAIL summary)
python tests/test_action_chunking.py
```

### Mini diffusion policy (W4 Task 2)

`experiments/train_diffusion_policy.py` replaces "one forward pass that regresses a chunk" with
**DDPM conditioning**: an `ε_θ(s, a^t, t)` noise-prediction network (FiLM MLP, `src/diffusion.py`)
is trained with the noise-prediction MSE, and actions are produced by denoising
`a^N ~ N(0, I)` in `N` DDIM steps. The dataset comes from a **multimodal** expert
(`expert_controller_multimodal`) that deliberately kicks the swing-up to either side whenever the
pole hangs almost still, so `p(a|s)` really is bimodal — the regime where plain MSE provably
converges to `E[a|s]` (an action that is *neither* mode).

* Cosine schedule with `T = 100`, closed-form `q_sample`, `ddim_step` with `N = 16`, `eta = 0`.
* **Fair comparison**: same episodes/split, same chunk labels (`K = 8`), same MLP head shape,
  same Adam/lr/epochs (200) — the only variable is *sampling a distribution* vs *regressing its mean*.
* The noise loss is injected via `engine.train_loop(step_fn=...)`, so `src/engine.py` stays untouched.
* Extras produced in the same run: `N`/`replan_every` ablation, plan-latency benchmark,
  unseen-ambiguous-state probe, plan-consistency diagnostic, sparse-data ablation, bimodal toy task.

| run | closed-loop return (20 ep) | forward passes / episode | plan latency | params |
| --- | --- | --- | --- | --- |
| deterministic MLP chunk (mean) | **-181.72 ± 85.99** | **25** | **0.186 ms** | 18,056 |
| mini diffusion policy (sampling) | -331.83 ± 173.47 | 400 | 24.72 ms | 358,152 |

Mechanism evidence (why the task is interesting even though diffusion does **not** win here):

| probe | deterministic MSE | mini diffusion |
| --- | --- | --- |
| bimodal toy task — hit either mode | 0.0% (mean \|a\| = 0.010) | **98.5%** (both modes covered) |
| unseen ambiguous Pendulum states — hit either mode | 62.5% | **71.7%** |
| … distance to nearest mode | 0.518 | **0.311** |

Sampling-cost ablation (diffusion only, closed-loop return): `N = 4 / 10 / 16 / 32` →
`-494.75 / -325.53 / -315.16 / -428.72`; halving the replan interval changes nothing
(`-318.00` at every 4 steps vs `-315.16` at every 8).

**Conclusion**: diffusion really does learn the conditional distribution (on the toy task MSE
collapses to the average of the two modes while diffusion recovers both, `|a|` per state matching
`|μ(s)|` pointwise), and it improves the mode-hit rate on unseen ambiguous states (71.7% vs 62.5%).
But on Pendulum that 9.2 pp gain does not pay for the **132.8x slower plan** and the 49.2%
plan-to-plan mode flips, so the deterministic regressor wins the closed-loop comparison by 150
points; a sparse-data ablation (16% of the trajectories) favours it even more (-203.95 vs -882.73),
i.e. diffusion is *not* more sample-efficient here. Full write-up:
`docs/reports/w4_task2_diffusion_policy.md`; a beginner-level Chinese walkthrough (diffusion from scratch,
all formulas explained, FAQ) lives in `docs/reports/w4_task2_diffusion_policy_零基础版.md`.

```bash
# Mini diffusion policy (W4 Task 2): ~12 min; trains + runs deterministic control, probes & ablations
python -u experiments/train_diffusion_policy.py

# Cheaper variants
python -u experiments/train_diffusion_policy.py --no-eval --set training.epochs=20
python -u experiments/train_diffusion_policy.py --set diffusion.sample_steps=32

# Unit tests (15 diffusion + 18 action-chunking, plain stdlib)
python tests/test_diffusion.py
```

### Key features
- All hyper-parameters live in `configs/bc_config.yaml` — no code changes needed.
- `src/networks.py` exposes a registry + `build_network(config)` to switch MLP / CNN / GRU.
- `src/datasets.py` caches expert data on disk and supports Gaussian-noise augmentation.
- `src/utils.py` provides logging, TensorBoard monitoring and best-model checkpointing.
- Command-line overrides (`--set key=value`) enable batch hyper-parameter sweeps.
- Industrial `Checkpointer`: atomic + checksummed writes, full training state
  (model / optimizer / scheduler / epoch / best metric / RNG), `--resume` with
  graceful `SIGINT` / `SIGTERM` handling and emergency save on exceptions.

