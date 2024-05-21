import torch
from torchvision import datasets, transforms

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载完整的数据集
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 计算训练集和验证集的划分比例
train_ratio = 0.8  # 训练集比例
valid_ratio = 1 - train_ratio  # 验证集比例

# 计算划分数量
train_size = int(train_ratio * len(full_dataset))
valid_size = len(full_dataset) - train_size

# 划分数据集
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# 使用train_loader和valid_loader进行训练和验证
...