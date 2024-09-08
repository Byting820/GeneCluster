import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

#(通道数(卷积核数量)，卷积核尺寸，步长,padding)
# config = [(96, 11, 2, 1),(256, 5, 1, 1),(256, 5, 1, 1),(384, 3, 1, 1),(384, 3, 1, 1),(512, 3, 1, 1),(256, 3, 1, 1),"SPP"]
# config = [(256, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1 ,1), (1024, 3, 1, 1), (1024, 3, 1, 1), (2048, 3, 1, 1), (2048, 3, 1, 1), (1024, 3, 1, 1), (1024, 3, 1, 1), (512, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1), "SPP"]
config = [(256, 3, 1, 1), (512, 3, 1, 1), (1024, 3, 1, 1),(1024, 3, 1, 1), (512, 3, 1, 1), (256, 3, 1, 1), "SPP"]
out_put_size = [1,3,6,8,10,16]

class SPPNet(nn.Module):
    def __init__(self, features, num_classes):
        super(SPPNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                            nn.Linear(256 * sum(out_put_size), 2048), #加了一层空间金字塔池化
                            nn.ReLU(inplace=False),
                            nn.Dropout(0.2),
                            nn.Linear(2048, 2048),
                            nn.ReLU(inplace=False))
                            
        self.top_layer = nn.Linear(2048, num_classes)
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv1d):            #判断m是否是nn.Conv1d
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

          

def make_layers_features(config, input_dim, bn):
    layers = []
    in_channels = input_dim
    for i in config:
        if i == 'MP':
            layers += [nn.MaxPool1d(kernel_size=3, stride=2)]
        elif i == 'SPP':
            layers += [SPP_1d(out_put_size)]
        else:
            conv1d = nn.Conv1d(in_channels, i[0], kernel_size=i[1],
                               stride=i[2], padding= i[3])
            if bn:       #如果是batch_norm，加入批次正交化，加速网络训练
                layers += [conv1d, nn.BatchNorm1d(i[0]), nn.ReLU(inplace=False)]  
            else:
                layers += [conv1d, nn.ReLU(inplace=False)]    #将卷积层和激活函数拼在一起，添加到layers中
            in_channels = i[0]
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
                spp = output.view(N, -1)     # -1表示自动补齐向量长度
            else:
                spp = torch.cat((spp, output.view(N, -1)), 1)    # 将两个张量按列拼接在一起（维数1）
        return spp
            

             
def sppCNN(bn=True, out=100):
    dim = 1
    model = SPPNet(make_layers_features(config, dim, bn=bn), out)
    return model


# model = sppCNN()
# input = torch.randn(20, 1, 758)
# a = model.features(input)
# print(a.size())    #[20,1536]
# print(a.type())
# print(a.type())   #[20,100]
