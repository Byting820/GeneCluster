import torch
import torch.nn as nn
import math
from random import random as rd
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# __all__ = [ 'VGG', 'vgg16']
# out_put_size = [1,2,3]
out_put_size = [1,3,6,8,10,16]

class VGG(nn.Module):

    def __init__(self, features, num_classes):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * sum(out_put_size), 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True)
        )
        self.top_layer = nn.Linear(2048, num_classes)   #num_classes 分类的类别个数
        self._initialize_weights()
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        """ 权重初始化"""       
        for y,m in enumerate(self.modules()): # 遍历模块的每一层
            if isinstance(m, nn.Conv1d):      # 如果遍历的当前层是卷积层
                #print(y)
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:   #如果卷积核采用了偏置
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(input_dim, batch_norm):
    """ 根据不同的配置文件,定义不同的网络卷积层 """
    layers = []   #用来存放定义的每一层结构
    in_channels = input_dim
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'SPP']   #数字表示卷积层的个数，M表示最大池化层
    for v in cfg:
        if v == 'M':    # 生成池化层
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        elif v == 'SPP':
            layers += [SPP_1d(out_put_size)]
        else:           #生成卷积层
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)   # v是卷积核的个数64
            if batch_norm:   #如果是batch_norm，加入批次正交化，加速网络训练
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]   
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]   ##将卷积层和激活函数拼在一起，添加到layers中
            in_channels = v
    return nn.Sequential(*layers)   

class SPP_1d(nn.Module):
    '''
    1D Spatial pyramid pool layer
    '''
    def __init__(self, out_pool_size): 
        '''
        out_pool_size:池化层金字塔每层输出的大小
        '''
        super(SPP_1d, self).__init__()
        self.out_pool_size = out_pool_size
        
    def forward(self, x):
        N, C, L = x.size() #N:样本量， C:通道数， L:输入长度
        for i in range(len(self.out_pool_size)):
            wid = int(math.ceil(L / self.out_pool_size[i]))   #向上取整
            pad = wid * self.out_pool_size[i] - L             # 填充
            maxpool = nn.MaxPool1d(wid, stride=wid)
            x_pad = F.pad(x, (0, pad), "constant", 0)
            output = maxpool(x_pad)
            if (i == 0):
                spp = output.view(N, -1)
            else:
                spp = torch.cat((spp, output.view(N, -1)), 1)
        return spp

def vgg16(bn=True, out=100):
    """实例化给定的配置模型"""
    dim = 1
    model = VGG(make_layers(dim, bn), out)
    return model


# 实例化
# model = vgg16()
# input = torch.randn(20,1,499)
# a = model.features(input)
# print(a.size())
# print(a.type())
# print(model(input).shape)