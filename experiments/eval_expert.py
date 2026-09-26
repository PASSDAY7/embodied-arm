#!/usr/bin/env python
"""专家控制器（oracle）闭环回报基线 —— 读懂"闭环回报"数值的参照系。

为什么需要它
------------
行为克隆（BC）的目标是"模仿专家"，因此判断一个策略好不好，最直观的参照系就是
**专家自己在同一个环境里能拿多少回报**：

* 专家（``src/datasets.py: expert_controller``）—— 手写控制器（能量塑形摆起 +
  PD 平衡），代表"数据来源"的水平，也叫 **oracle**（因为它直接读到真实状态、
  不需要学习，属于"有答案的人"）；
* 训练出来的策略回报若与 oracle 相当甚至更好，说明模仿成功。

本脚本不加载任何网络，只把专家控制器放到环境里跑 N 个回合，输出
平均 / 标准差 / 最好 / 最差回报，供 ``docs/reports/w4_task1_action_chunking.md`` 与
``docs/零基础读本.md`` 中的对比表引用。评估所用回合种子（``seed + ep``）与
``src/datasets.py: evaluate_closed_loop`` 完全一致，保证可比。

用法::

    python experiments/eval_expert.py                       # 20 回合，seed=42
    python experiments/eval_expert.py --episodes 50 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import expert_controller  # noqa: E402


def evaluate_expert(
    env_name: str = "Pendulum-v1",
    n_episodes: int = 20,
    seed: int = 42,
    render: bool = False,
) -> dict:
    """把专家控制器放到环境里跑闭环，返回回报统计。

    Args:
        env_name: Gymnasium 环境名。
        n_episodes: 评估回合数。
        seed: 起始种子；第 ``ep`` 个回合使用 ``seed + ep``（与策略评估一致）。
        render: 是否开启可视化窗口。

    Returns:
        含 ``mean_return`` / ``std_return`` / ``min_return`` / ``max_return`` /
        ``episode_returns`` 的字典。
    """
    import gymnasium as gym

    env = gym.make(env_name, render_mode="human" if render else None)
    returns = []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_return, done = 0.0, False
        while not done:
            action = expert_controller(state)                 # 专家给出力矩 τ ∈ [-2, 2]
            state, reward, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            ep_return += float(reward)
            done = terminated or truncated
        returns.append(ep_return)
    env.close()

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "min_return": float(arr.min()),
        "max_return": float(arr.max()),
        "episode_returns": [float(r) for r in arr],
        "n_episodes": int(n_episodes),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="专家控制器（oracle）闭环回报基线")
    parser.add_argument("--env", default="Pendulum-v1", help="Gymnasium 环境名")
    parser.add_argument("--episodes", type=int, default=20, help="评估回合数")
    parser.add_argument("--seed", type=int, default=42, help="起始回合种子")
    parser.add_argument("--render", action="store_true", help="开启可视化窗口")
    args = parser.parse_args()

    result = evaluate_expert(
        env_name=args.env, n_episodes=args.episodes, seed=args.seed, render=args.render
    )
    print(f"专家(oracle) {result['n_episodes']} 回合 (seed {result['seed']}): "
          f"mean {result['mean_return']:.2f} "
          f"std {result['std_return']:.2f} "
          f"min {result['min_return']:.2f} "
          f"max {result['max_return']:.2f}")


if __name__ == "__main__":
    main()
