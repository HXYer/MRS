import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import sys

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)

# 残差模块的定义
class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, down_sample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, 
                               out_channels=o_channel, 
                               kernel_size=3, 
                               stride=stride, 
                               padding=1,
                               bias=False)
        # BatchNorm2d(）对小批量3d数据组成的4d输入进行批标准化操作
        # 主要为了防止神经网络退化
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=o_channel, 
                               out_channels=o_channel, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.down_sample = down_sample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 将单元的输入直接与单元输出加在一起
        if self.down_sample:
            residual = self.down_sample(x) # 下采样
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_classes=11):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, 2)
        self.layer2 = self.make_layer(block, 32, 2, 2)
        self.layer3 = self.make_layer(block, 64, 2, 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):  
        # blocks=layers,残差模块的数量
        down_sample = None
        # 判断是否in_channels(输入)与(输出)是否在同一维度
        # 即输入的3d数据的长宽高与输出的数据的长宽高是否一样
        if (stride != 1) or (self.in_channels != out_channels):
            # 如果不一样就转换一下维度
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)  # 添加所有残差块

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.fc(out)

        return out

if __name__ == '__main__':
    net=ResNet(Residual_Block, [2,2,2,2])
    y = net(Variable(torch.randn(7,3,32,52)))
    print(y.size())
