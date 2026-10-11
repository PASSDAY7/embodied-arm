# embodied-arm — embodied AI / robot-arm RL engineering workspace

This repo grows in three layers:

1. **W2–W9 — algorithms from scratch** (PyTorch only, no RL libraries): behaviour cloning → action
   chunking → mini diffusion policy → REINFORCE → PPO-Clip → SAC, each with unit tests, reports,
   curves and archived run logs on `Pendulum-v1` / `CartPole-v1`;
2. **W10–W12 — physics-grade base** (Isaac Sim 6.1 + Isaac Lab 3.0): 64 parallel Franka arms under
   torque-only control with a zero-copy tensor loop, domain randomization injected at the PhysX
   level, and a trainable continuous trajectory-tracking MDP with open-loop baselines;
3. **W13/W14 — main project**: **two-layer residual control** — a nominal feed-forward + PID base
   with a learned SAC residual `Δτ` — trained end to end for 450k env steps.

Main result (64 env, 500 steps, `seed=42`, domain randomization on):

| controller | joint RMSE [rad] | end-effector Cartesian RMSE [cm] |
| --- | --- | --- |
| `pd_dr` (PD only, mismatch) | 0.060948 | 2.3016 |
| `pd_ff_dr` (feed-forward + PID base) | 0.051866 | 1.3259 |
| **residual SAC (450k steps, final)** | **0.045955** (×1.13) | **0.8651** (×1.53) |
| residual SAC (best eval @400k, `policy_best_ee.pt`) | 0.045967 | **0.8612** |
| `zero_dr` (no torque) | 0.972252 | 73.1780 |

Acceptance: joint < 0.0519 rad ✅ and end-effector < 1.0 cm ✅ (stretch goal 0.5 cm not met).
Cost: 450 048 env steps / 13 438 updates / 527 s on one RTX 2080 Ti, 0.277 GiB peak VRAM.

> 🇨🇳 Chinese entry point (canonical, more detail): [`README.md`](README.md);
> beginner guide: [`docs/零基础读本.md`](docs/零基础读本.md).

---

## Structure

```
embodied-arm/
├── configs/        # YAML hyper-parameters (bc / chunking / diffusion / reinforce / ppo / sac)
├── src/            # from-scratch implementations: networks datasets engine utils
│                   #   diffusion.py  reinforce.py  ppo.py  sac.py
├── experiments/    # runnable entry points
│   ├── train_{bc,chunking_bc,diffusion_policy,reinforce,ppo,sac}.py   eval_expert.py
│   ├── export_tb_curves.py        # TensorBoard events -> PNG + CSV
│   ├── isaaclab_env.sh            # pins the Isaac Sim/Lab environment (one-liner launcher)
│   ├── w10_arm_torque_control.py  w10_plot_scale.py
│   ├── w11_dynamics_randomization.py
│   ├── w12_trajectory_tracking_env.py
│   ├── w13_residual_sac_train.py  # ★ self-contained residual-SAC trainer (108 CLI flags)
│   ├── w15_paper_benchmarks.py    # ★ W15 paper experiment matrix: 4-controller baseline + stress + ablation
│   └── export_paper_figures.py    # W16 vector figures (Fig 1-3) + LaTeX tables (Table 1/2)
├── tests/          # plain-stdlib unit tests (148 checks, no pytest needed)
├── docs/           # reports/ figures/ notes/ results/ + 零基础读本.md
├── w2_pytorch_basics/   data/ (gitignored)   logs/ (gitignored)
```

## Environment

| | W2–W9 (pure PyTorch) | W10–W13 (Isaac Lab) |
| --- | --- | --- |
| conda | `rl` (Python 3.10) | `py312` (Isaac Sim 6.1 pre-installed in site-packages) |
| deps | `pip install -r requirements.txt` | Isaac Sim 6.1 + Isaac Lab 3.0 (`/data/zzh/IsaacLab`) + torch 2.6.0+cu124 / warp 1.17 |
| entry | `python experiments/train_*.py` | `bash experiments/isaaclab_env.sh experiments/w1*_*.py …` |

```bash
# A. pure PyTorch (W2-W9)
conda create -n rl python=3.10 -y && conda activate rl && pip install -r requirements.txt
python experiments/train_sac.py

# B. Isaac Lab (W10-W13) — no isaaclab.sh / uv needed, the launcher pins everything
bash experiments/isaaclab_env.sh --check                       # version self-check only
bash experiments/isaaclab_env.sh experiments/w12_trajectory_tracking_env.py --num_envs 64 --steps 500
```

## Quick start

```bash
# unit tests (~1 min, plain stdlib)
python tests/test_sac.py   # 48 checks
python tests/test_ppo.py   # 38
python tests/test_reinforce.py tests/test_action_chunking.py tests/test_diffusion.py

# ★ reproduce the main result (W13/W14, 450k env steps ≈ 10 min)
# --run_name decides where checkpoints / history CSV / curve PNG land: it must match
# config.run_name inside the JSON, otherwise everything is written into the default
# directory logs/w13_residual_sac/ (arm A) and overwrites arm A's artefacts.
bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \
    --num_envs 64 --steps 500 --total_steps 450000 --w_ee 50000 --run_name w13_residual_sac_ee50k \
    --out_json docs/results/w13_residual_sac_ee50k.json > docs/results/w13_residual_sac_ee50k.log 2>&1

# read a checkpoint back (--eval_only writes no weights and does not touch the history CSV)
bash experiments/isaaclab_env.sh experiments/w13_residual_sac_train.py \
    --num_envs 64 --steps 500 --eval_only \
    --load_policy logs/w13_residual_sac_ee50k/policy_best_ee.pt --run_name w13_best_ee_check
```

### 3.4 Paper experiment matrix (W15/W16)

```bash
# (1) Table 1: four controllers (PD / PD+nominal-FF / pure RL / two-layer residual SAC)
#     pure RL is trained first with the identical budget (450k env steps, same reward/observation)
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage baselines

# (2) Figure 3: three stress axes (payload 0-5 kg / friction x1-x3 / action delay 0-3 steps)
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage payload
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage friction
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage latency

# (3) Table 2: ablations (observation history H / residual torque limit / jerk penalty)
bash experiments/isaaclab_env.sh experiments/w15_paper_benchmarks.py --stage ablation --ablation_dims history,res_limit,jerk

# (4) figures + LaTeX tables (plain numpy/matplotlib, no Isaac Sim needed)
python experiments/export_paper_figures.py
```

Artefacts: `docs/results/w15_*.json`, the cumulatively written `docs/results/w15_benchmarks.csv`
(the machine-readable version of the paper tables) and `w15_per_env.csv` (per-environment values for
the boxplot), `docs/figures/fig{1,2,3}_*.pdf|png`, `docs/results/w15_tables.tex`.

`--stage baselines` re-runs `pd_dr`/`pd_ff_dr` through the original W12 `run_episode` path and compares
it field by field with the new unified evaluation loop (tolerance 1e-9) — the evidence that the two
baseline rows in Table 1 are still the same numbers as in the W12/W13 reports.

## Weekly progress (W2 → W16)

| week | task | report | headline |
| --- | --- | --- | --- |
| W2 | PyTorch basics (fit sin, MLP, CNN, sequence) | `docs/notes/` | hand-written training loops |
| W3 | behaviour cloning, engineered (config/log/TB/resume) | [`docs/零基础读本.md`](docs/零基础读本.md) | closed-loop **−156.95** (expert oracle −175.00) |
| W4-1 | action chunking | [`w4_task1_action_chunking.md`](docs/reports/w4_task1_action_chunking.md) | K=4 **−155.58**, forward passes 200 → **50** |
| W4-2 | mini diffusion policy vs regression | [`w4_task2_diffusion_policy.md`](docs/reports/w4_task2_diffusion_policy.md) | bimodal hit **98.5%**, but 132.8× slower (−331.8 vs −181.7) |
| W6 | REINFORCE + baselines + return norm | [`w6_task1_reinforce.md`](docs/reports/w6_task1_reinforce.md) | CartPole **500.00 @ep120** |
| W8 | PPO-Clip + GAE-λ | [`w8_task1_ppo.md`](docs/reports/w8_task1_ppo.md) | **194 560 steps** to −199.91 |
| W9 | SAC (off-policy max-entropy) | [`w9_task1_sac.md`](docs/reports/w9_task1_sac.md) | **4 000 steps** to −173.70 (≈48× fewer than PPO) |
| W10 | 64-arm torque-only zero-copy loop | [`w10_task3_torque_control.md`](docs/reports/w10_task3_torque_control.md) | **389.5 SPS = 24 929 env-steps/s**; PhysX `sim.step` is 73% of the time |
| W11 | domain randomization injected into PhysX | [`w11_task1_dynamics_randomization.md`](docs/reports/w11_task1_dynamics_randomization.md) | same torque, different states: cross-env std **1e-6 → 7e-2 rad** |
| W12 | trajectory-tracking MDP + open-loop baselines | [`w12_task1_trajectory_tracking_env.md`](docs/reports/w12_task1_trajectory_tracking_env.md) | feed-forward PID **1.3259 cm** vs PD **2.3016 cm** vs zero torque **73.18 cm** |
| W13/W14 | **two-layer residual SAC, end to end** | [`w13_task1_residual_sac.md`](docs/reports/w13_task1_residual_sac.md) | joint **0.045955 rad** (×1.13), EE **0.8651 cm** (×1.53) — both acceptance lines cleared |
| W15 | paper experiment matrix (baseline / stress / ablation) | [`w15_task1_paper_benchmarks.md`](docs/reports/w15_task1_paper_benchmarks.md) | same budget: EE RMSE **PD 2.3016 / FF+PD 1.3259 / pure RL 86.43 / residual SAC 0.8612 cm** (Ours −35.0%, sat 0.00%); pure RL fails outright (sat 95.8%); Ours best on every stress point |
| W16 | vector figures + LaTeX tables | `python experiments/export_paper_figures.py` | `docs/figures/fig{1,2,3}_*.pdf/png`, `docs/results/w15_tables.tex` |

## Evidence layout

| what | where |
| --- | --- |
| reports (tables, ablations, debugging log) | `docs/reports/*.md` |
| structured results of every formal run | `docs/results/*.json` |
| per-evaluation history | `docs/results/*_history.csv` |
| training logs (grep-able; Isaac Sim noise stripped) | `docs/results/*.log` (raw originals in gitignored `logs/raw_console_logs/`) |
| curves | `docs/figures/*.png` + `*.csv` |
| runtime output (checkpoints, TB events) | `logs/` — gitignored, regenerated by re-running |

**Run-to-run note**: repeating the *same* command is **not** bit-identical (the first two evals at
25k/50k match bitwise, they diverge afterwards; 0.043342 rad / 0.9107 cm vs 0.045955 rad / 0.8651 cm,
both pass acceptance). The JSON/CSV/PNG committed here are from the **second** run; the cross-check
chain (W12 reconciliation, zero-residual equivalence, obs-ring check, FK check, checkpoint read-back)
reports `0.00e+00` in both. Pass `--deterministic` to trade speed for reproducibility.

## Roadmap / hardware

- [x] W2–W5 deep learning from scratch · [x] W6–W9 RL from scratch
- [x] W10–W12 Isaac Lab base · [x] W13–W14 residual SAC end to end
- [ ] W15+: larger budget (600k–1M steps), multi-seed, `w_ee` curriculum, acceptance-aligned action scaling
- [ ] 2027.1–2 real robot deployment · [ ] 2027.3 paper submission (IROS / RA-L)

RTX 2080 Ti (11 GB) / Ubuntu 22.04; all algorithms hand-written (no SB3/rlkit); unit tests are plain stdlib.
