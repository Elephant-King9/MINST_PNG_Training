from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import *

# 训练集地址
train_root = "../datasets/mnist_png/training"
# 测试集地址
test_root = '../datasets/mnist_png/testing'

# 进行数据的处理，定义数据转换
data_transform = transforms.Compose([transforms.Resize((28, 28)),
                                     transforms.Grayscale(),
                                     transforms.ToTensor()])


# 加载数据集
train_dataset = ImageFolder(train_root, transform=data_transform)
test_dataset = ImageFolder(test_root, transform=data_transform)

# Dataset ImageFolder
#     Number of datapoints: 60000
#     Root location: ../datasets/mnist_png/training
#     StandardTransform
# Transform: Compose(
#                Resize(size=(28, 28), interpolation=bilinear, max_size=None, antialias=True)
#                ToTensor()
#            )
# print(train_dataset)

# print(train_dataset[0])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = Net().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epoch = 10

writer = SummaryWriter('../logs')
total_step = 0

for i in range(epoch):
    model.train()
    pre_step = 0
    pre_loss = 0
    for data in train_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        pre_loss = pre_loss + loss.item()
        pre_step += 1
        total_step += 1
        if pre_step % 100 == 0:
            print(f"Epoch: {i+1} ,pre_loss = {pre_loss/pre_step}")
            writer.add_scalar('train_loss', pre_loss / pre_step, total_step)

    model.eval()
    pre_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            pre_accuracy += outputs.argmax(1).eq(labels).sum().item()
    print(f"Test_accuracy: {pre_accuracy/len(test_dataset)}")
    writer.add_scalar('test_accuracy', pre_accuracy / len(test_dataset), i)
    torch.save(model, f'../models/model{i}.pth')

writer.close()


