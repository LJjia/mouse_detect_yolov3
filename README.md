# yolov3老鼠检测

yolov3实现的老鼠位置检测

# 效果展示

![pic](mouse_show.gif)

# 训练

## 数据集

自己做的数据集,其中大概有800张老鼠的图片,因此此模型应该能检测到大多数的老鼠

## 缺陷

由于分类只有一类,因此似乎各种外部的因素,都可能会被网络区分为老鼠,因此需要增加数据集容量,增加更多负样本

# 性能

## 2022.03.02

使用最开始的yolov3(darknet53主干),其他原封不动

使用cpu,fps只有可怜的2

使用rtx3060 cuda,fps基本达到21

# 致谢

主体yolo部分代码参考

* https://github.com/bubbliiiing/yolo3-pytorch

