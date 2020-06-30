# -*- coding: utf-8 -*-

#仅对conv层及FC层的乘权重进行权重衰减
import torch
import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary
from torchvision.models import resnet18,resnet50
import torch.nn.functional as F


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
class Resnet18(nn.Module):
    def __init__(self, n_classes = 5):
        super(Resnet18,self).__init__()

        src_net = resnet18(pretrained=True)
        modules = list(src_net.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(512,n_classes)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
'''

class Resnet50(nn.Module):
    def __init__(self, n_classes = 5):          #n_classees类别数量
        super(Resnet50,self).__init__()

        src_net = resnet50(pretrained=True)     #返回在ImageNet上训练好的模型
        modules = list(src_net.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.classifier = nn.Linear(2048,n_classes) #维度  全连接层   b和c要相等，否则不能点乘
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):               #前向传播
        features = self.features(x)
        out = F.leaky_relu(features)
        out = F.adaptive_avg_pool2d(out,(1,1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

#网络可视化
def netview(model):

    #调用make_dot()函数构造图对象
    x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
    out = model(x)
    g=make_dot(out)
    #保存模型，以PDF格式保存
    #g.view('model_structure.pdf','./netview/')
    g.render('./netview/espnet_model', view=False) # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开

if __name__ == '__main__':
    # net = Resnet18()
    net = Resnet50()
    #标准正态分布
    aa = torch.randn((5,3,100,100))
    print(net(aa).size())
    #查看模型结构
    summary(net, input_size=(3, 224, 224), batch_size=1, device='cpu')

    print('是否需要生成保存网络模型可视化图？(y/n):')
    i = input('')
    if i == 'y'or i == 'Y':
        #生成可视化网络模型
        netview(net)
    else:exit()
