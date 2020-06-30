from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from res_network import Resnet50
import torchvision.transforms as transforms
from torch.autograd import Variable as V
import torch as t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# totensor 转换
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def prediect(img):
    print('wait..')
    #classes = ('飞机','机场','棒球场','篮球场','海滩','桥梁','丛林','教堂','圆形农田','云','商业区','密集居民区','沙漠','森林','高速公路','高尔夫球场','田径场','港口','工业区','交叉路口','岛','湖','草地','中型住宅','移动家园','山','天桥','宫殿','停车场','铁路','铁路车站','矩形农场','河','环岛','跑道','海冰','船','雪山','稀少的住宅','体育场','存储罐','网球场','露台','热电站','湿地')
    classes = ('海滩', '灌木', '沙漠', '森林', '草地')

    #读入图片
    #img = Image.open('图片路径')
    img=trans(img)        #这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    img = img.unsqueeze(0)      #增加一维，输出的img格式为[1,C,H,W]

    model = Resnet50().to(device)   #导入网络模型
    model.eval()
    model.load_state_dict(t.load('D:/Graduation_Design/Gra_Dedign/model_out/5-18/final_model.pth'))        #加载训练好的模型文件

    input = V(img.to(device))
    score = model(input)            #将图片输入网络得到输出
    probability = t.nn.functional.softmax(score,dim=1)      #计算softmax，即该图片属于各类的概率
    max_value,index = t.max(probability,1)          #找到最大概率对应的索引号，该图片即为该索引号对应的类别
    #print(index)
    msg = ' {} 可能是：{}'.format('这张图',classes[index])
    print(msg)
    return msg

# ------------------------------------------------------------------------

if __name__ == '__main__':
    #args = parse_args()

    img = Image.open('D:/RSdata_dir/gra_data_dir/test/image (33).jpg')
    plt.imshow(img)
    plt.show()
    prediect(img)

