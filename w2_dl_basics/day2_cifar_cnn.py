import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torchvision  # 导入计算机视觉库
import torchvision.transforms as transforms  # 导入数据变换与增强模块

# ----------------- 1. 数据增强与加载 -----------------
# 训练集数据增强：随机裁剪与翻转能有效防止过拟合，提高泛化能力
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 对32x32图片四周填充4像素，再随机裁剪为32x32
    transforms.RandomHorizontalFlip(),  # 依50%概率水平翻转图像
    transforms.ToTensor(),  # 将PIL图像转为Tensor，数值范围缩放到[0.0, 1.0]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10各通道均值方差归一化
])

# 测试集只做标准化，不做随机变换
transform_test = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转为Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 同样标准化
])

# 下载并加载CIFAR-10训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# 训练数据加载器：batch_size设为128，打乱数据，使用2个子进程加载
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# 下载并加载CIFAR-10测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# 测试数据加载器：不打乱数据
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# ----------------- 2. 基础残差块定义 -----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()  # 继承父类构造函数
        # 第一个卷积层：当stride=2时进行空间尺寸减半下采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化加速收敛
        self.relu = nn.ReLU(inplace=True)  # 原地ReLU激活函数
        # 第二个卷积层：保持空间尺寸和通道数不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批归一化
        
        # 快捷连接（Shortcut）：若输入输出维度不匹配，用1x1卷积调整通道和分辨率
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # 1x1卷积升维/下采样
                nn.BatchNorm2d(out_channels)  # 批归一化
            )

    def forward(self, x):
        identity = self.shortcut(x)  # 计算快捷路径分支
        out = self.relu(self.bn1(self.conv1(x)))  # 主路径第一层：Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))  # 主路径第二层：Conv -> BN
        out += identity  # 残差相加：F(x) + x
        out = self.relu(out)  # 相加后再过一次ReLU激活函数
        return out  # 返回残差块输出


# ----------------- 3. 构建整体CNN网络 -----------------
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()  # 继承父类构造函数
        self.in_channels = 32  # 初始通道数
        
        # 初始输入特征提取层：3通道输入，32通道输出，32x32分辨率保持不变
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活
        
        # 逐层堆叠残差块
        self.layer1 = self._make_layer(32, num_blocks=2, stride=1)  # 特征图保持 32x32
        self.layer2 = self._make_layer(64, num_blocks=2, stride=2)  # 特征图下采样至 16x16
        self.layer3 = self._make_layer(128, num_blocks=2, stride=2)  # 特征图下采样至 8x8
        
        # 全局池化与分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化，输出变为 128x1x1
        self.fc = nn.Linear(128, num_classes)  # 全连接层映射至10分类

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块控制下采样，后续块stride均为1
        layers = []  # 存储块的列表
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))  # 添加残差块
            self.in_channels = out_channels  # 更新输入通道数
        return nn.Sequential(*layers)  # 组装为Sequential网络容器

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 初始卷积与归一化
        out = self.layer1(out)  # 第一层残差块组 (32通道, 32x32)
        out = self.layer2(out)  # 第二层残差块组 (64通道, 16x16)
        out = self.layer3(out)  # 第三层残差块组 (128通道, 8x8)
        out = self.avg_pool(out)  # 全局自适应池化为 (batch_size, 128, 1, 1)
        out = torch.flatten(out, 1)  # 展平成二维向量 (batch_size, 128)
        out = self.fc(out)  # 全连接层输出类别得分 (batch_size, 10)
        return out  # 返回预测logits


# ----------------- 4. 训练与测试流程 -----------------
# 自动检测并配置GPU/CPU环境
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SimpleResNet().to(device)  # 实例化网络并移至指定设备

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
# 选用带动量的SGD优化器，配合Weight Decay进行L2正则化
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# 余弦退火学习率调度器，训练30个epoch逐步降低学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)



# 开始训练主循环（约需 25-30 个 epoch 即可稳定突破 88% - 91% 准确率）
for epoch in range(30):
    net.train()  # 切换为训练模式（开启BN与Dropout）
    train_loss = 0.0  # 累计训练损失
    correct = 0  # 预测正确样本数
    total = 0  # 累计处理总样本数

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据送入指定设备
        optimizer.zero_grad()  # 清空历史梯度
        outputs = net(inputs)  # 前向传播计算输出
        loss = criterion(outputs, targets)  # 计算损失值
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 优化器更新参数

        train_loss += loss.item()  # 累加loss
        _, predicted = outputs.max(1)  # 获取预测概率最大的类别索引
        total += targets.size(0)  # 累加样本数
        correct += predicted.eq(targets).sum().item()  # 统计预测正确的样本数

    scheduler.step()  # 每个epoch更新一次学习率

    # 评估测试集性能
    net.eval()  # 切换为评估模式（固定BN统计量）
    test_loss = 0.0  # 累计测试损失
    test_correct = 0  # 测试正确数
    test_total = 0  # 测试总样本数

    with torch.no_grad():  # 评估阶段关闭梯度计算，节省显存与算力
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)  # 数据移入设备
            outputs = net(inputs)  # 前向预测
            loss = criterion(outputs, targets)  # 计算测试loss
            test_loss += loss.item()  # 累加loss
            _, predicted = outputs.max(1)  # 获取预测类别
            test_total += targets.size(0)  # 累计测试样本
            test_correct += predicted.eq(targets).sum().item()  # 累计正确数

    test_acc = 100.0 * test_correct / test_total  # 计算当前epoch测试准确率百分比
    print(f"Using device: {device}")
    print(f"Model on: {next(net.parameters()).device}")
    print(f'Epoch [{epoch + 1}/30] Test Acc: {test_acc:.2f}% | Train Loss: {train_loss / len(trainloader):.4f}')


