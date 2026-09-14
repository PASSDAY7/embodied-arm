# embodied-arm

Residual RL for robust robot arm trajectory tracking.

Target: IROS 2027 / RA-L submission (March 2027).

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
- `w2_dl_basics/` — deep learning fundamentals (single-file toy scripts)
- `docs/` — weekly logs and experiment records

## Behavior Cloning (engineered project)

The single-file script `w2_dl_basics/day2_behavior_cloning.py` has been refactored
into a standard, reusable and monitorable algorithm project layout:

```
embodied-arm/
├── configs/                  # Hyper-parameter configs (YAML)
│   └── bc_config.yaml        # BC experiment config
├── src/                      # Core source code
│   ├── networks.py           # Policy networks (MLP, CNN, GRU) + factory
│   ├── datasets.py           # Expert data collection, dataset & augmentation
│   └── utils.py              # Config, logging, TensorBoard, Checkpointer
├── experiments/              # Experiment run scripts
│   └── train_bc.py           # Behavior cloning training entry point
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

### Key features
- All hyper-parameters live in `configs/bc_config.yaml` — no code changes needed.
- `src/networks.py` exposes a registry + `build_network(config)` to switch MLP / CNN / GRU.
- `src/datasets.py` caches expert data on disk and supports Gaussian-noise augmentation.
- `src/utils.py` provides logging, TensorBoard monitoring and best-model checkpointing.
- Command-line overrides (`--set key=value`) enable batch hyper-parameter sweeps.
- Industrial `Checkpointer`: atomic + checksummed writes, full training state
  (model / optimizer / scheduler / epoch / best metric / RNG), `--resume` with
  graceful `SIGINT` / `SIGTERM` handling and emergency save on exceptions.

