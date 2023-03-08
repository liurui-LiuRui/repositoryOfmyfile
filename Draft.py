import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch.nn as nn
import torchvision.transforms as transforms

# always check your version
print(torch.__version__)

# check if gpu/cpu
def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
device = torch.device(get_default_device())
print('Using device:', device)

class TinyImage30Data(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)
        self.root_dir = root_dir

    def __getitem__(self, index):
        # return a single item from the dataset
        # you can use self.imgs[index] to get the image path and label
        # you can use self.loader(path) to load an image from a path
        # you can use self.transform(image) to apply the transform if not None
        path, label = self.imgs[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        # return the size of the dataset
        return len(self.imgs)

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

my_dataset = TinyImage30Data('./comp5625M_data_assessment_1/train_set', transform=transform)

total_count = len(my_dataset)
train_count = int(0.8 * total_count)
valid_count = total_count - train_count
train_dataset, valid_dataset = torch.utils.data.random_split(my_dataset, [train_count, valid_count])

# Creat DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


# 定义超参数
input_size = 64 * 64 * 3  # 输入单元数，等于图像的像素数乘以通道数
hidden_size = 500  # 隐藏单元数，可以自己选择
num_classes = 30  # 输出单元数，等于图像的类别数
num_epochs = 20  # 训练轮数，可以自己选择
batch_size = 128  # 批次大小，可以自己选择
learning_rate = 1e-2  # 学习率，可以自己选择

# # 定义数据集和数据加载器
# transform = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 定义图像预处理方法，将图像转换为张量并标准化
#
# train_dataset = torchvision.datasets.ImageFolder(root='./data/train',
#                                                  transform=transform)  # 定义训练数据集，使用ImageFolder类并指定根目录和预处理方法
#
# test_dataset = torchvision.datasets.ImageFolder(root='./data/test',
#                                                 transform=transform)  # 定义测试数据集，使用ImageFolder类并指定根目录和预处理方法
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
#                                            shuffle=True)  # 定义训练数据加载器，使用DataLoader类并指定数据集、批次大小和是否打乱顺序
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
#                                           shuffle=False)  # 定义测试数据加载器，使用DataLoader类并指定数据集、批次大小和是否打乱顺序


# 定义MLP模型类，继承nn.Module类，并重写__init__和forward方法
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()  # 调用父类的构造函数
        self.fc1 = nn.Linear(input_size, hidden_size)  # 定义第一个全连接层（线性变换），输入单元数为input_size，输出单元数为hidden_size
        self.relu = nn.ReLU()  # 定义激活函数（ReLU），将所有负值变为零
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 定义第二个全连接层（线性变换），输入单元数为hidden_size，输出单元数为num_classes

    def forward(self, x):
        out = self.fc1(x)  # 将输入x通过第一个全连接层得到输出out
        out = self.relu(out)  # 将输出out通过激活函数得到新的输出out
        out = self.fc2(out)  # 将输出out通过第二个全连接层得到新的输出out
        return out


# 实例化MLP模型对象，并将其移动到设备（CPU或GPU）上
model = MLP(input_size, hidden_size, num_classes)
model.to(device)

# 定义损失函数（交叉熵损失）和优化器（随机梯度下降）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)  # 将图像张量展平成一维向量，并传输到设备上（CPU或GPU）
        labels = labels.to(device)  # 将标签传输到设备上

        outputs = model(images)  # 将图像输入模型得到输出
        loss = criterion(outputs, labels)  # 计算输出和标签之间的损失

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型
model.eval() # 设置模型为评估模式
with torch.no_grad(): # 不计算梯度，节省内存
    correct = 0 # 预测正确的图像数
    total = 0 # 总共的图像数
    for images, labels in valid_loader: # 遍历测试集中的每一个批次
        images = images.reshape(-1, input_size).to(device) # 将图像张量展平成一维向量，并传输到设备上（CPU或GPU）
        labels = labels.to(device) # 将标签传输到设备上
        outputs = model(images) # 将图像输入模型得到输出
        _, predicted = torch.max(outputs.data, 1) # 得到每个输出的最大值和对应的索引，索引就是预测的类别
        total += labels.size(0) # 更新总图像数，等于标签的个数
        correct += (predicted == labels).sum().item() # 更新预测正确的图像数，等于预测和标签相等的个数

    print('Accuracy of the network on the test images: {} %'.format(100 * correct / total)) # 打印测试集上的准确率