
## 简介
##数据集：NWPU-RESISC45部分图像
http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

取5个场景

['海滩', '灌木丛', '沙漠', '森林', '草地']

划分数据集 train：val：test = 7：2：1

## 环境依赖
pytorch==1.1 or 1.0 

tensorboard==1.8

tensorboardX 

pillow

注意调低batch_size参数特别是像我这样的渣渣显卡

## 使用方法
只需要指明数据集路径参数即可，就可以得到最终模型以及log、tensorboard_log了


train:开始训练
```
python train_resnet.py 
```

数据集文件夹应为 之下的目录结构应如下：
```
your data_dir/
       |->train
       |->val
       |->test
       |->ClassnameID.txt
```

#运行生成result.txt
```
python infer.py
```

渣渣cpu跑的验证集：

correct:697/704=0.9901

--------valuate_val--Done!----------

