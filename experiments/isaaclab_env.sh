#!/usr/bin/env bash
# W10 环境固化脚本：把“Isaac Sim 6.1 + Isaac Lab 3.0 + py312 conda 环境”这套组合固定下来，
# 让仓库里的实验脚本可以用一条命令复现运行（不需要 `isaaclab.sh`，也不需要 `uv`）。
#
# 为什么不用官方推荐流程？
# ------------------------
# IsaacLab `develop`（VERSION 3.0.0）已经弃用 `./isaaclab.sh --install`，改用 `uv run isaaclab`；
# 但本机的 Isaac Sim 6.1 是**已经装进 py312 conda 环境**的（site-packages 内自带 kit/），
# 再让 uv 建新环境既装不下（/ 只剩 ~13GB）也没必要。因此这里直接复用既有环境，
# 只做三件事：激活 conda 环境、把 IsaacLab 的 source/* 挂到 PYTHONPATH、设置 Kit 启动所需变量。
#
# 用法::
#
#     bash experiments/isaaclab_env.sh experiments/w10_arm_torque_control.py --num_envs 64
#     bash experiments/isaaclab_env.sh --check          # 只打印环境自检信息
#
# 可用环境变量覆盖：CONDA_ENV / ISAACLAB_DIR / HEADLESS / PYTHON。
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-py312}"
ISAACLAB_DIR="${ISAACLAB_DIR:-/data/zzh/IsaacLab}"
CONDA_BIN="${CONDA_BIN:-/home/zhihao/anaconda3/etc/profile.d/conda.sh}"
PYTHON="${PYTHON:-python}"
# HEADLESS=1：Isaac Lab 3.0 已经没有 `--headless` 命令行参数了，
# 它从环境变量 HEADLESS 读取（见 isaaclab/app/app_launcher.py:_resolve_headless_settings）。
export HEADLESS="${HEADLESS:-1}"
# Kit 首次启动会交互式询问 Omniverse EULA；无 TTY 时会直接报 “Bad file descriptor” 退出。
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

# 1) 激活 conda 环境（py312 里已有 isaacsim 6.1.0.0 / torch / warp）
# shellcheck disable=SC1090
source "$CONDA_BIN"
conda activate "$CONDA_ENV"

# 2) 挂 IsaacLab 源码：isaaclab / isaaclab_physx / isaaclab_assets / isaaclab_tasks ...
ISAACLAB_PYTHONPATH="$(ls -d "$ISAACLAB_DIR"/source/*/ | tr '\n' ':')"
export PYTHONPATH="${ISAACLAB_PYTHONPATH}${PYTHONPATH:-}"

# 3) 自检模式：只汇报版本信息，不跑仿真
if [[ "${1:-}" == "--check" ]]; then
  echo "conda env       : $CONDA_ENV ($(command -v python))"
  echo "ISAACLAB_DIR    : $ISAACLAB_DIR ($(cat "$ISAACLAB_DIR/VERSION"))"
  echo "HEADLESS        : $HEADLESS"
  echo "PYTHONPATH      : ${PYTHONPATH%:}"
  python - "$ISAACLAB_DIR" <<'PY'
import importlib.metadata as md
import pathlib, sys

print("isaacsim         :", md.version("isaacsim"), "(VERSION file:", pathlib.Path(sys.argv[1], "VERSION").read_text().strip() + ")")
for name in ("torch", "warp-lang", "isaaclab"):
    try:
        print(f"{name:17s}:", md.version(name))
    except md.PackageNotFoundError:
        print(f"{name:17s}: (未以 pip 包形式安装，走 PYTHONPATH 源码)")
import torch
print("torch.cuda       :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
from isaaclab.app import AppLauncher  # noqa: F401  （只验证可导入，不启动 Kit）
print("isaaclab.app     : import OK")
PY
  exit 0
fi

# 4) 运行目标脚本（-u 关掉 Python 输出缓冲，否则重定向到日志时看不到 [INFO] 行）
exec "$PYTHON" -u "$@"
