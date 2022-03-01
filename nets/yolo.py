from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    '''
    很好理解,自定义带padding的卷积+bn+relu
    和darknet骨干中下采样+残差的模组定义不同
    padding的值为半个卷积核的大小,目的也就是带卷积之后,特征图大小不变
    :param filter_in: 输出维度
    :param filter_out: 输出维度
    :param kernel_size: 卷积核大小
    :return:
    '''
    # // 表示整除,除法去小数点
    # 为了让kernel_size-1 为正数,加了个判断
    # 如果定义kernel大小,则pad=(kernel_size - 1) // 2
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        # 根据公式,(n-f+2p)/s+1
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=(kernel_size,kernel_size), stride=(1,1), padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    '''
    对应三种大小特征图的预测后处理
    :param filters_list: 降维升维特征图通道数,为2个变量
    :param in_filters: 整个后处理输入特征图通道数
    :param out_filter: 整个后处理输出特征图通道数
    :return: 组合好的nn.Sequential
    '''
    m = nn.Sequential(
        # 1x1降维 卷积图通道数变为一半
        conv2d(in_filters, filters_list[0], 1),
        # 3x3升维 卷积图通道数变为两倍
        conv2d(filters_list[0], filters_list[1], 3),
        # 1x1降维
        conv2d(filters_list[1], filters_list[0], 1),
        # 3x3升维
        conv2d(filters_list[0], filters_list[1], 3),
        # 1x1降维
        conv2d(filters_list[1], filters_list[0], 1),

        # 下面两个用于预测
        # 1x1降维
        conv2d(filters_list[0], filters_list[1], 3),
        # 1x1 预测类别
        nn.Conv2d(filters_list[1], out_filter, kernel_size=(1,1), stride=(1,1), padding=0, bias=True)
    )
    return m

class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False):
        '''

        :param anchors_mask: 先验框尺寸,这里仅用到每个尺寸的特征图一共有几个先验框
        :param num_classes:
        :param pretrained:
        '''
        super(YoloBody, self).__init__()
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #   52x52x256->26x26x512是下采样,跨距为2,深度变深了
        #---------------------------------------------------#

        # 返回dark53主骨干网络,不带特征提取层
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        #---------------------------------------------------#
        #   out_filters : [64, 128, 256, 512, 1024]
        #---------------------------------------------------#

        # python默认都是public对象 这里是[64, 128, 256, 512, 1024]的列表
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言,20个类别 3x(1+4+20)=75
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))
        # 取第五层输出,将13x13x512的特征图上采样为26x26x256,然后跟26x26的特征图融合
        self.last_layer1_conv       = conv2d(512, 256, 1)
        # scale_factor为上采样放大倍数,上采样不会改变通道数目,但是会把宽高x2
        # mode='nearest'表示不用等差数列,上采样直接将原始值复制为两倍
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        # 取第五层 26x26x256特征图压缩到128,再上采样成52x52x128
        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   x2:52,52,256；x1:26,26,512；x0:13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 注意如果是voc数据集,那么对应75,coco对应85
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,75

        # out0_branch这个是要contact到26x26的特征图输出上的
        out0_branch = self.last_layer0[:5](x0)
        # out0为13x13的特征图输出
        out0        = self.last_layer0[5:](out0_branch)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        # 13x13x512的特征图变换为26x26x256 准备和26x26x512的融合
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)

        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)
        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,75
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out3 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,75
        out2 = self.last_layer2(x2_in)
        # 对应分别为13x13x75 26x26x75 52x52x75的特征图
        return out0, out1, out2

