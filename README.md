# yolov3老鼠检测

yolov3实现的老鼠位置检测

# 效果展示

![pic](mouse_show.gif)

# 训练

## 数据集

自己做的数据集,其中大概有800张老鼠的图片,因此此模型应该能检测到大多数的老鼠

## 缺陷

由于分类只有一类,因此似乎各种外部的因素,都可能会被网络区分为老鼠,因此需要增加数据集容量,增加更多负样本,增加网络泛化能力

<img src="assets/image-20220304215559714.png" alt="image-20220304215559714" style="zoom: 50%;" />



# 修改记录

## 2022.03.02

输入图片大小w\*h:720\*1280

使用最开始的yolov3(darknet53主干),其他原封不动

使用cpu,fps只有可怜的2

使用rtx3060 cuda,fps基本达到21

## 2022.03.04

增加负样本数据集,增加很多类似这种未标注的图像,设置目标个数为0.相当于告诉网络其他乱七八糟的不是mouse目标

<img src="assets/image-20220304234433054.png" alt="image-20220304234433054" style="zoom:50%;" />

效果

<img src="assets/image-20220304234229019.png" alt="image-20220304234229019" style="zoom:50%;" />

# 致谢

主体yolo部分代码参考

* https://github.com/bubbliiiing/yolo3-pytorch

