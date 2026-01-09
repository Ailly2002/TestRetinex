# 导入PyTorch库
import torch.nn as nn
import torch.nn.functional as F


# 定义LeNet-5模型结构
class LeNet5(nn.Module):
    # 初始化网络结构
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 卷积层C1: 输入图像为32x32大小，使用6个卷积核大小为5x5的filter进行滤波，
        # 步长stride为1，从而得到28x28x6的输出。
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 池化层S2: 输入特征图为28x28x6大小，
        # 使用2x2大小的filter进行平均池化，从而得到14x14x6大小的输出。
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层C3: 输入特征图为14x14x6大小，使用16个卷积核大小为5x5的filter进行滤波，
        # 步长stride为1，从而得到10x10x16的输出。
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 池化层S4: 输入特征图为10x10x16大小，
        # 使用2x2大小的filter进行平均池化，从而得到5x5x16大小的输出。
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层C5(也可当作全连接层): 输入特征图为5x5x16大小，
        # 使用120个卷积核大小为5x5的filter进行滤波，步长stride为1，从而得到120维的输出。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层F6: 输入特征图为120维大小，通过全连接层，得到84维的输出。
        self.fc2 = nn.Linear(120, 84)
        # 全连接层F7: 输入特征图为84维大小，通过全连接层，得到num_classes维的输出，
        # 代表num_classes类目标的预测得分值。
        self.fc3 = nn.Linear(84, num_classes)


    def forward(self, x):
        # 卷积层C1 + 激活函数relu
        x = F.relu(self.conv1(x))
        # 池化层S2
        x = self.avgpool1(x)
        # 卷积层C3 + 激活函数relu
        x = F.relu(self.conv2(x))
        # 池化层S4
        x = self.avgpool2(x)
        # 改变维度以匹配之后的全连接层（展平操作）
        x = x.view(x.size(0), -1)
        # 全连接层F6 + 激活函数relu
        x = F.relu(self.fc1(x))
        # 全连接层F7 + 激活函数relu
        x = F.relu(self.fc2(x))
        # 全连接层F8得到输出结果
        x = self.fc3(x)
        return x