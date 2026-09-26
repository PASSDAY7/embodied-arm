import torch

print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 设备名称:", torch.cuda.get_device_name(0))
    print("当前使用的设备索引:", torch.cuda.current_device())
else:
    print("警告：正在使用 CPU 训练！")