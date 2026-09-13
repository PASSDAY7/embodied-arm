import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 固定随机种子，保证可复现
np.random.seed(42)
torch.manual_seed(42)


# ===================== 1. 专家控制器（能量摆起 + PD平衡） =====================
def expert_controller(state):
    """
    倒立摆专家控制器：
    - 偏离倒立较远时：能量塑形控制，将摆从下垂位置摆起
    - 接近倒立位置时：PD反馈控制，保持平衡
    输入: state = [cosθ, sinθ, θ_dot]
    输出: 力矩τ，范围[-2.0, 2.0]
    """
    cos_theta, sin_theta, theta_dot = state
    theta = np.arctan2(sin_theta, cos_theta)  # 还原角度θ，范围[-π, π]

    # 倒立摆系统参数（与Pendulum-v1一致）
    g = 10.0
    mass = 1.0
    length = 1.0
    inertia = mass * length ** 2
    E_target = mass * g * length  # 倒立位置的总机械能（动能为0时的势能）

    # 当前总机械能
    kinetic = 0.5 * inertia * theta_dot ** 2
    potential = mass * g * length * cos_theta
    E_total = kinetic + potential

    # 双模态控制
    if abs(theta) < 0.5:  # 接近倒立，切换PD平衡控制
        kp = 12.0
        kd = 1.5
        tau = -kp * theta - kd * theta_dot
    else:  # 能量塑形摆起，向目标能量收敛
        k_energy = 2.5
        tau = k_energy * theta_dot * np.sign(E_target - E_total)

    # 裁剪到动作空间边界
    return np.clip(tau, -2.0, 2.0)


# ===================== 2. 专家数据收集与数据集划分 =====================
def collect_expert_data(env_name="Pendulum-v1", n_episodes=50):
    """收集50回合专家交互数据，按8:2切分训练/验证集"""
    env = gym.make(env_name)
    states_list = []
    actions_list = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = expert_controller(state)
            states_list.append(state)
            actions_list.append(action)
            # 环境步进：动作需传入1维数组
            state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated

    env.close()

    # 转为numpy数组
    states = np.array(states_list, dtype=np.float32)       # (N, 3)
    actions = np.array(actions_list, dtype=np.float32).reshape(-1, 1)  # (N, 1)

    print(f"专家数据收集完成：共 {len(states)} 步，{n_episodes} 回合")

    # 打乱并切分80%训练、20%验证
    indices = np.random.permutation(len(states))
    split_idx = int(0.8 * len(states))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, y_train = states[train_idx], actions[train_idx]
    X_val, y_val = states[val_idx], actions[val_idx]

    return X_train, y_train, X_val, y_val


# ===================== 3. 策略网络（行为克隆） =====================
class PolicyNet(nn.Module):
    """MLP策略网络：输入状态3维，输出动作1维，MSE损失拟合专家动作"""
    def __init__(self, state_dim=3, hidden_dim=64, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出归一化到[-1,1]，再缩放至动作空间
        )
        self.action_scale = 2.0  # 对应动作空间[-2.0, 2.0]

    def forward(self, x):
        return self.net(x) * self.action_scale


# ===================== 4. 训练与验证 =====================
def train_bc(X_train, y_train, X_val, y_val, epochs=100, lr=1e-3, batch_size=256):
    """训练行为克隆策略，返回最优模型和验证集MSE"""
    # 构建数据加载器
    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    model = PolicyNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mse = float("inf")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_sum = 0.0
        for batch_s, batch_a in train_loader:
            pred_a = model(batch_s)
            loss = criterion(pred_a, batch_a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_s.size(0)

        train_mse = train_loss_sum / len(train_set)

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_mse = criterion(val_pred, y_val_t).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f}")

    print(f"\n训练完成 | 最佳验证集 MSE: {best_val_mse:.6f}")
    return model, best_val_mse


# ===================== 5. 闭环评估 =====================
def evaluate_closed_loop(model, env_name="Pendulum-v1", n_episodes=20):
    """将策略放入环境闭环运行20回合，计算平均回合总回报"""
    env = gym.make(env_name)
    episode_returns = []

    model.eval()
    with torch.no_grad():
        for ep in range(n_episodes):
            state, _ = env.reset()
            ep_return = 0.0
            done = False
            while not done:
                state_t = torch.from_numpy(state).float().unsqueeze(0)  # 增加batch维度
                action = model(state_t).squeeze().numpy()
                state, reward, terminated, truncated, _ = env.step([action])
                ep_return += reward
                done = terminated or truncated
            episode_returns.append(ep_return)

    env.close()
    mean_return = np.mean(episode_returns)
    print(f"闭环评估完成 | 20回合平均总回报: {mean_return:.2f}")
    return mean_return


# ===================== 主函数 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("行为克隆（Behavior Cloning）实验：Pendulum-v1")
    print("=" * 50)

    # 1. 收集专家数据
    print("\n[1/3] 收集专家数据...")
    X_train, y_train, X_val, y_val = collect_expert_data()

    # 2. 训练策略
    print("\n[2/3] 训练行为克隆策略...")
    policy_model, val_mse = train_bc(X_train, y_train, X_val, y_val)

    # 3. 闭环评估
    print("\n[3/3] 闭环环境评估...")
    mean_return = evaluate_closed_loop(policy_model)

    # 最终结果
    print("\n" + "=" * 50)
    print("最终实验结果")
    print("=" * 50)
    print(f"验证集 MSE: {val_mse:.6f}")
    print(f"20回合闭环平均回报: {mean_return:.2f}")


