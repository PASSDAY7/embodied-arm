import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===================== 1. 数据生成与构造 =====================
# 生成带高频噪声的正弦序列
t = np.linspace(0, 100, 5000)
data = np.sin(t) + 0.1 * np.sin(10 * t)  # 主频 + 高频噪声

# 滑动窗口构造样本：过去10步预测第11步
window_size = 10
X, y = [], []
for i in range(len(data) - window_size):
    X.append(data[i:i+window_size])
    y.append(data[i+window_size])
X = np.array(X)  # 形状 (4990, 10)
y = np.array(y)  # 形状 (4990,)

# 80%训练集、20%测试集划分
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch浮点张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 对齐输出维度 (N, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ===================== 2. 模型定义 =====================
class MLP(nn.Module):
    """基线MLP模型：将窗口序列当作静态向量输入"""
    def __init__(self, window_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)


class GRU(nn.Module):
    """GRU时序模型：按时间步处理序列，捕捉时序依赖"""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)
    
    def forward(self, x):
        # 输入形状 (batch, 10) → 增加特征维度 → (batch, 10, 1)
        x = x.unsqueeze(-1)
        # GRU前向传播，输出所有时间步的隐藏状态
        out, _ = self.gru(x)  # out形状: (batch, 10, 32)
        # 取最后一个时间步的隐藏状态做预测
        return self.fc(out[:, -1, :])


# ===================== 3. 训练函数 =====================
def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loss_history = []
    test_loss_history = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        y_pred_train = model(X_train)
        train_loss = criterion(y_pred_train, y_train)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        # 测试阶段
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = criterion(y_pred_test, y_test)
        
        train_loss_history.append(train_loss.item())
        test_loss_history.append(test_loss.item())
        
        # 每20轮打印一次
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_loss.item():.6f} | Test MSE: {test_loss.item():.6f}")
    
    return model, train_loss_history, test_loss_history


# ===================== 4. 模型训练 =====================
print("===== 训练 MLP 模型 =====")
mlp_model = MLP(window_size)
mlp_model, mlp_train_loss, mlp_test_loss = train_model(
    mlp_model, X_train, y_train, X_test, y_test, epochs=100
)

print("\n===== 训练 GRU 模型 =====")
gru_model = GRU()
gru_model, gru_train_loss, gru_test_loss = train_model(
    gru_model, X_train, y_train, X_test, y_test, epochs=100
)

# 输出最终测试MSE
print("\n===== 最终测试结果 =====")
print(f"MLP Test MSE: {mlp_test_loss[-1]:.6f}")
print(f"GRU Test MSE: {gru_test_loss[-1]:.6f}")
print(f"是否满足验收标准(<0.01): MLP={mlp_test_loss[-1]<0.01}, GRU={gru_test_loss[-1]<0.01}")


# ===================== 5. 预测对比绘图 =====================
# 生成预测结果
mlp_model.eval()
gru_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(X_test).numpy().flatten()
    gru_pred = gru_model(X_test).numpy().flatten()
    true_vals = y_test.numpy().flatten()

# 取测试集前200个点可视化
plot_points = 200
plt.figure(figsize=(12, 6))
plt.plot(range(plot_points), true_vals[:plot_points], 
         label='Ground Truth', color='black', linewidth=1.8)
plt.plot(range(plot_points), mlp_pred[:plot_points], 
         label='MLP Prediction', color='#1f77b4', linestyle='--', alpha=0.85)
plt.plot(range(plot_points), gru_pred[:plot_points], 
         label='GRU Prediction', color='#d62728', linestyle='-.', alpha=0.85)

plt.xlabel('Time Step (Test Set)', fontsize=11)
plt.ylabel('Value', fontsize=11)
plt.title('Time Series One-step Prediction: MLP vs GRU (First 200 Test Points)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ===================== 6. 损失曲线可视化 =====================
plt.figure(figsize=(10, 4))
plt.plot(mlp_test_loss, label='MLP Test MSE', color='#1f77b4')
plt.plot(gru_test_loss, label='GRU Test MSE', color='#d62728')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Test MSE during Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()

