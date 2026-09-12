import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 中文设置
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_all = torch.linspace(0, 2*np.pi, 1000).unsqueeze(1)  # 创建一个从0到2*pi的200个点的张量
y_all = torch.sin(x_all)  # 计算对应的正弦值

#随机打乱数据，并划分训练集和测试集
indices = torch.randperm(x_all.size(0))
x_train = x_all[indices[:800]].to(device)  # 将训练数据搬到GPU上
y_train = y_all[indices[:800]].to(device)  # 将训练标签搬到GPU上
x_test = x_all[indices[800:]].to(device)  # 将测试数据搬到GPU上
y_test = y_all[indices[800:]].to(device)  # 将测试标签搬到GPU上

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    def forward(self, x):
        return self.layers(x)

model = MLP().to(device)  # 将模型搬到GPU上
loss_fn = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

# 训练模型
epochs = 5000
for epoch in range(epochs):
    y_pred = model(x_train)  # 前向传播
    loss = loss_fn(y_pred, y_train)  # 计算损失
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    if(epoch % 500 == 0):
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 在评估时不需要计算梯度
            y_pred_test = model(x_test)
            test_loss = loss_fn(y_pred_test, y_test)
        print(f'Epoch {epoch}, Loss: {loss.item()}, Test Loss: {test_loss.item()}')
        model.train()  # 设置回训练模式
        

# 绘制结果
with torch.no_grad():  # 在绘图时不需要计算梯度
    x_plot = torch.linspace(0, 2*np.pi, 1000).unsqueeze(1).to(device)
    y_plot = model(x_plot)
    plt.figure(figsize=(10,5))
    plt.plot(x_plot.cpu().numpy(), y_plot.cpu().numpy(), label='MLP预测', color='r')
    plt.plot(x_all.cpu().numpy(), y_all.cpu().numpy(), label='真实正弦函数', color='b')
    plt.legend()
    plt.title('MLP拟合正弦函数')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.show()




