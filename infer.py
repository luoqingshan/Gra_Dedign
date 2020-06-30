# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from rs_dataset import RSDataset,InferDataset
from res_network import Resnet50
import time
from functools import wraps


# gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timefn(fn):
    """计算性能的修饰器"""
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} used {t2 - t1: .5f} s")
        return result
    return measure_time

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--test_batch_size',type=int,default=1)
    parse.add_argument('--num_workers', type=int, default=2)

    #local
    parse.add_argument('--data_dir',type=str,default='D:/RSdata_dir/gra_data_dir')      #/mnt/rssrai_cls

    #运行之前修改这里的模型路径
    parse.add_argument('--model_out_name',type=str,default='./model_out/5-6-100/final_model.pth')

    return parse.parse_args()

@timefn
def main_worker(args):
    print('wait...')
    #classes = ('飞机','机场','棒球场','篮球场','海滩','桥梁','丛林','教堂','圆形农田','云','商业区','密集居民区','沙漠','森林','高速公路','高尔夫球场','田径场','港口','工业区','交叉路口','岛','湖','草地','中型住宅','移动家园','山','天桥','宫殿','停车场','铁路','铁路车站','矩形农场','河','环岛','跑道','海冰','船','雪山','稀少的住宅','体育场','存储罐','网球场','露台','热电站','湿地')
    classes = ('海滩', '灌木丛', '沙漠', '森林', '草地')

    data_set = InferDataset(rootpth=args.data_dir)
    data_loader = DataLoader(data_set,
                             batch_size=args.test_batch_size,
                             drop_last=True,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True)


    net = Resnet50()

    net.load_state_dict(torch.load(args.model_out_name))

    net.to(device)

    net.eval()

    with open('results.txt','w') as f:
        with torch.no_grad():
            for img,names in data_loader:
                img = img.to(device)
                size = img.size(0)
                outputs = net(img)
                outputs = F.softmax(outputs, dim=1)
                predicted = torch.max(outputs, dim=1)[1].cpu().numpy()

                for i in range(size):
                    msg = '{} {} 可能是：{}'.format(names[i], predicted[i]+1,classes[predicted[i]])
                    f.write(msg)
                    f.write('\n')

    print('--------Classification--Done!----------')


# 用于测试验证集
@timefn
def evaluate_val(args):
    print('wait...')
    val_set = RSDataset(rootpth=args.data_dir, mode='val')
    val_loader = DataLoader(val_set,
                            batch_size=args.test_batch_size,
                            drop_last=True,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    net = Resnet50()

    net.load_state_dict(torch.load(args.model_out_name))

    net.to(device)

    net.eval()

    total = 0
    correct = 0
    net.eval()
    with torch.no_grad():
        for img, lb in val_loader:
            img, lb = img.to(device), lb.to(device)
            outputs = net(img)
            outputs = F.softmax(outputs, dim=1)
            predicted = torch.max(outputs, dim=1)[1]
            total += lb.size()[0]

            correct += (predicted == lb).sum().cpu().item()
    print('correct:{}/{}={:.4f}'.format(correct, total, correct * 1. / total))

    print('--------valuate_val--Done!----------')


if __name__ == '__main__':
    args = parse_args()

    print('a:推理测试集')
    print("b:测试验证集")
    print('选择需要进行的操作(a/b):')
    i = input('')
    if i=='a':
        # 推理测试集
        main_worker(args)
    elif i == 'b':
        # 用于测试验证集
        evaluate_val(args)
    else:exit()
