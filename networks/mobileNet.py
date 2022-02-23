import time
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
from torch.autograd import Variable


# conv_bnΪ����ĵ�һ������飬����Ϊ2
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# conv_dwΪ��ȿɷ�����
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 3x3�����ȡ����������Ϊ2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1���������Ϊ1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_channels):
        super(MobileNet, self).__init__()
        self.layer1 = nn.Sequential(
            # ��һ������飬����Ϊ2��ѹ��һ��
            conv_bn(n_channels, 32, 1),  # 416,416,3 -> 208,208,32

            # ��һ����ȿɷ�����������Ϊ1
            conv_dw(32, 64, 1),  # 208,208,32 -> 208,208,64

            # ������ȿɷ�������
            conv_dw(64, 128, 2),  # 208,208,64 -> 104,104,128
            conv_dw(128, 128, 1),

            # 104,104,128 -> 52,52,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        # 52,52,256 -> 26,26,512
        self.layer2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        # 26,26,512 -> 13,13,1024
        self.layer3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)

        x = x.view(-1, 1024)
        x = self.fc(x)
        return x