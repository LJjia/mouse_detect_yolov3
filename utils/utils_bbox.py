import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(DecodeBox, self).__init__()
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape # [416,416]
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors_mask   = anchors_mask

    def decode_box(self, inputs):
        '''
        :param inputs:
        :return:一个list,三个元素,分别13,26,52的特征图输出归一化为小数的形式
        例如13的特征图输出为1x3x13x13x25=1x507x25 表示batchx先验框个数x图片宽x图片高x(框在特征图上xywh+置信度+类别)
        '''
        outputs = []
        for i, input in enumerate(inputs):
            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 255, 13, 13
            #   batch_size, 255, 26, 26
            #   batch_size, 255, 52, 52
            #-----------------------------------------------#
            batch_size      = input.size(0)
            input_height    = input.size(2)
            input_width     = input.size(3)

            #-----------------------------------------------#
            #   输入为416x416时
            #   stride_h = stride_w = 32、16、8
            #-----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            #-------------------------------------------------#
            #   此时获得的scaled_anchors大小是相对于特征层的
            #   对于13x13的特征图,宽大小为<class 'list'>: [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]
            #-------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors[self.anchors_mask[i]]]

            #-----------------------------------------------#
            #   输入的input一共有三个，他们的shape分别是
            #   batch_size, 3, 13, 13, 85
            #   batch_size, 3, 26, 26, 85
            #   batch_size, 3, 52, 52, 85
            #-----------------------------------------------#
            # input为13x13特征图的输出,1x75x13x13 展开成 -> 1x3x13x13x25
            prediction = input.view(batch_size, len(self.anchors_mask[i]),
                                    self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

            #-----------------------------------------------#
            #   先验框的中心位置的调整参数
            #   获取最后一个维度的第0 1列 pre维度 1x3x13x13x25 x维度1x3x13x13
            #   sigma(x)表示预测出来的框中心,偏离该矩形框左上角x的距离
            #   sigma(y)表示预测出来的框中心,偏离该矩形框左上角y的距离
            #-----------------------------------------------#
            x = torch.sigmoid(prediction[..., 0])  
            y = torch.sigmoid(prediction[..., 1])
            #-----------------------------------------------#
            #   先验框的宽高调整参数
            #   exp(w) * anchor_w表示,预测出来的框是先验框宽的几倍
            #   exp(h) * anchor_h表示,预测出来的框是先验框高的几倍
            #-----------------------------------------------#
            w = prediction[..., 2]
            h = prediction[..., 3]
            #-----------------------------------------------#
            #   获得置信度，是否有物体
            #-----------------------------------------------#
            conf        = torch.sigmoid(prediction[..., 4])
            #-----------------------------------------------#
            #   种类置信度 pred_cls 1x3x13x13x20
            #-----------------------------------------------#
            pred_cls    = torch.sigmoid(prediction[..., 5:])

            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #----------------------------------------------------------#
            #   生成网格，先验框中心，网格左上角 
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            # grid_x,grid_y维度为1x3x13x13
            # 相当于13x13特征图上每个格子左上角xy座标,x三个先验框
            grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #----------------------------------------------------------#
            #   按照网格格式生成先验框的宽高
            #   batch_size,3,13,13
            #----------------------------------------------------------#
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
            # 1x3x13x13的矩阵中,每个13x13的矩阵中分别存了先验框h 2.18, 6.18, 10.18 三个值 宽类似

            #----------------------------------------------------------#
            #   利用预测结果对先验框进行调整
            #   首先调整先验框的中心，从先验框中心向右下角偏移
            #   再调整先验框的宽高。
            #----------------------------------------------------------#
            pred_boxes          = FloatTensor(prediction[..., :4].shape)
            # 构造了一个1x3x13x13x4的矩阵叫pred_boxes
            pred_boxes[..., 0]  = x.data + grid_x
            pred_boxes[..., 1]  = y.data + grid_y
            pred_boxes[..., 2]  = torch.exp(w.data) * anchor_w
            pred_boxes[..., 3]  = torch.exp(h.data) * anchor_h
            # 计算了先验框在特征图上的位置,中心点x,y绝对座标+预测框宽高
            # pred_boxes是1x3x13x13x4的矩阵,记录预测的坐标

            #----------------------------------------------------------#
            #   将输出结果归一化成小数的形式
            #----------------------------------------------------------#
            _scale = torch.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = torch.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            # output是一个1x507x25的矩阵,表示batchx先验框个数x图片宽x图片高x(框在特征图上xywh+置信度+类别)
            # 实际是1x13x13x3x25
            outputs.append(output.data)
        return outputs

    def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
        '''

        :param box_xy:目标个数x2 矩阵
        :param box_wh: 目标个数x2 矩阵
        :param input_shape: [416,416]
        :param image_shape: 图像原始大小
        :param letterbox_image: 是否填黑边
        :return:
        '''
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #-----------------------------------------------------------------#
            #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
            #   new_shape指的是宽高缩放情况
            #-----------------------------------------------------------------#
            new_shape = np.round(image_shape * np.min(input_shape/image_shape))
            offset  = (input_shape - new_shape)/2./input_shape
            scale   = input_shape/new_shape

            box_yx  = (box_yx - offset) * scale
            box_hw *= scale

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        # concatenate,数组拼接,按照axis传入的第几个维度拼接,
        # 返回的结果的维度:axis对应的维度相加,其他维度不变
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        # 再归一化到原始大图上的像素位置
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        # boxes 维度 目标个数* xy xy 注意这里的xy变成左上角和右下角了
        return boxes

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        '''
        非极大值抑制
        :param prediction: 1x10647x25 10647是因为 3x13x13+3x26x26+3x52x52=10647
        :param num_classes: 20类,int
        :param input_shape: 图像输入大小[416,416]
        :param image_shape: 图像实际大小
        :param letterbox_image: 是否填黑边,False效果更好?
        :param conf_thres: 置信度,int
        :param nms_thres: 抑制区间,int
        :return:
        '''
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape) # 创建一个box_corner和pre的shape一样
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        # 分别表示左上角和右下角坐标
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))] # [None] 和batch维度相关
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
            # class_conf为10647x1 class_pred为10647x1的维度 表示种类预测的最大值和最大的类别
            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            # 求(有目标的置信度*这个目标最有可能的类别对应的种类的置信度)>设定阈值
            # 对应的先验框为true
            conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
            # conf_mask 10647 一个维度 对应值为True False
            #----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            # 第二个维度不变,第一个维度,只有true的被保存下来,例如本次运行后是17x25的矩阵
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            # 这两个对应17x1的矩阵

            if not image_pred.size(0):
                # 没有目标,则开始下一个batch
                continue
            #-------------------------------------------------------------------------#
            #   detections  [置信度有效的目标个数, 7]
            #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            #------------------------------------------#
            #   获得预测结果中包含的所有种类 ,
            #   第二个维度的最后一列是label标签,detections[:, -1]是一个[17]的向量
            #   就算输入的是多维数组,unique也会展开成行向量
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]

                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #------------------------------------------#
                keep = nms(
                    # 第一个维度是多少个目标,第二个维度是左上xy和右下xy 5维
                    detections_class[:, :4],
                    # 置信度认为是物体置信度*类别置信度 2维
                    detections_class[:, 4] * detections_class[:, 5],
                    # 标量
                    nms_thres
                )
                # 返回的keep,1维,长度为满足要求目标的索引,结果类似第几个目标是符合要求的
                max_detections = detections_class[keep]
                # max_detections 目标个数x7(x1, y1, x2, y2, obj_conf, class_conf, class_pred) 的二维矩阵
                
                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data
                
                # Add max detections to outputs
                # 会把每张图的所有目标都堆叠在一起i表示batch的第几张图
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
            
            if output[i] is not None:
                # output[i]表示第i张图的目标 为 目标个数x7(xy信息等) 的矩阵
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                # output[i][:, :4]为修正xy坐标,可能要填黑边
                output[i][:, :4]    = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output
