import torch
import torch.nn as nn
import math
import numpy as np

class YOLOLoss(nn.Module):

    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        '''

        :param anchors:形如 [[10,13], [16,30],...] 9x2矩阵
        :param num_classes: voc 数据集为 20
        :param input_shape:[416,416]
        :param cuda:
        :param anchors_mask: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        '''
        super(YOLOLoss, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors        = anchors
        self.num_classes    = num_classes
        self.bbox_attrs     = 5 + num_classes
        self.input_shape    = input_shape
        self.anchors_mask   = anchors_mask

        self.ignore_threshold = 0.6
        self.cuda = cuda

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        # 小于等于t_min的等于t_min,大于等于t_max的等于t_max,其他保持不变
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def MSELoss(self, pred, target):
        '''
        分类预测中均方差损失,未求平均
        :param pred:
        :param target:
        :return:
        '''
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        '''
        计算交叉熵 binary cross entropy 二值交叉熵损失 未求平均
        :param pred: 预测值
        :param target: 数据集真实值
        :return:
        '''
        epsilon = 1e-7
        pred    = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output  = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output

    def forward(self, l, input, targets=None):
        #----------------------------------------------------#
        #   l代表的是，当前输入进来的有效特征层，是第几个有效特征层
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        #   targets代表的是真实框。
        #----------------------------------------------------#
        #--------------------------------#
        #   获得图片数量，特征层的高和宽
        #   13和13
        #--------------------------------#
        bs      = input.size(0)
        in_h    = input.size(2)
        in_w    = input.size(3)
        #-----------------------------------------------------------------------#
        #   计算步长
        #   每一个特征点对应原来的图片上多少个像素点
        #   原图416x416大小
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        #   stride_h和stride_w都是32。
        #-----------------------------------------------------------------------#
        # input shape是416x416
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#

        #   相当于将框变换到特征图上后,框的对应大小
        scaled_anchors  = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   bs, 3*(5+num_classes), 13, 13 => batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #-----------------------------------------------#
        # batch_size 先验框个数 (框位置,置信度,类别) (特征图wh)
        '''
        permute表示维度调换 将permute(2,0,1)参数中的维度索引按照输入参数的顺序摆放
        a = torch.ones([1, 2, 3])
        a.permute(2,0,1).shape
        输出:torch.Size([3, 1, 2])
        这里permute(0, 1, 3, 4, 2)就是把第三个维度(75的维度)移到最后面
        
        contiguous()操作保证tensor布局方式为默认的,内存连续,方便view调用
        '''
        prediction = input.view(bs, len(self.anchors_mask[l]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()
        # 结束后prediction 尺寸batch_size, 3, 13, 13, 5 + num_classes

        #-----------------------------------------------#
        #   先验框的中心位置的调整参数
        #-----------------------------------------------#
        # 例如尺寸为2,3,4 [...,0]表示所有最后一个维度索引为0构成的 2x3的矩阵
        # 也就是sigmod(x) sigmod(y)
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        #-----------------------------------------------#
        #   先验框的宽高调整参数
        #-----------------------------------------------#
        w = prediction[..., 2]
        h = prediction[..., 3]
        #-----------------------------------------------#
        #   获得置信度，是否有物体
        #-----------------------------------------------#
        conf = torch.sigmoid(prediction[..., 4])
        #-----------------------------------------------#
        #   种类置信度
        #-----------------------------------------------#
        # batchx3x13x13x20维度的矩阵
        pred_cls = torch.sigmoid(prediction[..., 5:])

        #-----------------------------------------------#
        #   获得网络应该有的预测结果
        #-----------------------------------------------#
        y_true, noobj_mask, box_loss_scale = self.get_target(l, targets, scaled_anchors, in_h, in_w)
        # y_true尺度 8x3x13x13x25 表示真实框的预测结果,xy外面有sigmoid
        # noobjmask 8x3x13x13 表示特征图上哪个框没有对应预测目标,为1为没有,0为发现目标
        # box_loss_scale 8x3x13x13 表示大目标loss权重小，小目标loss权重大,暂时不清楚

        #---------------------------------------------------------------#
        #   将预测结果进行解码，判断预测结果和真实值的重合程度
        #   如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #   作为负样本不合适
        #----------------------------------------------------------------#
        # 交并比阈值大于0.5的都认为是有目标的
        noobj_mask = self.get_ignore(l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true          = y_true.cuda()
            noobj_mask      = noobj_mask.cuda()
            box_loss_scale  = box_loss_scale.cuda()
        #-----------------------------------------------------------#
        #   reshape_y_true[...,2:3]和reshape_y_true[...,3:4]
        #   表示真实框的宽高，二者均在0-1之间
        #   真实框越大，比重越小，小框的比重更大。
        #-----------------------------------------------------------#
        box_loss_scale = 2 - box_loss_scale
        #-----------------------------------------------------------#
        #   计算中心偏移情况的loss，使用BCELoss效果好一些,注意,位置损失只有GT框所对应的anchor才会计算损失
        #  每个GT对应一个anchor,大多数anchor都没有真实框对应
        #-----------------------------------------------------------#
        # 注意,xy对应的是sigmoid值的BCE,而不是直接tx,ty的BCE
        loss_x = torch.sum(self.BCELoss(x, y_true[..., 0]) * box_loss_scale * y_true[..., 4])
        loss_y = torch.sum(self.BCELoss(y, y_true[..., 1]) * box_loss_scale * y_true[..., 4])
        #-----------------------------------------------------------#
        #   计算宽高调整值的loss MSE
        #-----------------------------------------------------------#
        # 注意,wh对应的是tw,th的MSE,而不是e^{tw}
        loss_w = torch.sum(self.MSELoss(w, y_true[..., 2]) * 0.5 * box_loss_scale * y_true[..., 4])
        loss_h = torch.sum(self.MSELoss(h, y_true[..., 3]) * 0.5 * box_loss_scale * y_true[..., 4])
        #-----------------------------------------------------------#
        #   计算置信度的loss,这个loss为:GT框个数*BCE+iou大于0.5的anchor数量*BCE
        #-----------------------------------------------------------#
        loss_conf   = torch.sum(self.BCELoss(conf, y_true[..., 4]) * y_true[..., 4]) + \
                      torch.sum(self.BCELoss(conf, y_true[..., 4]) * noobj_mask)
        # 同样,也是只有GT框所对应的anchor才会计算分类损失
        loss_cls    = torch.sum(self.BCELoss(pred_cls[y_true[..., 4] == 1], y_true[..., 5:][y_true[..., 4] == 1]))

        loss        = loss_x  + loss_y + loss_w + loss_h + loss_conf + loss_cls
        # num_pos每个batch中GT框的个数,求和后获得这个batch中gt框的个数
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        return loss, num_pos

    def calculate_iou(self, _box_a, _box_b):
        '''

        :param _box_a:[num_true_box, 4] 每行前两个为0 后两个为gt wh大小
        :param _box_b:[9, 4] 每行前两个为0 后两个为先验框wh大小
        :return: [num_true_box,9] 表示每个真实框和对应先验框的交并比
        '''
        #
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)

        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy  = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy  = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        # 输入张量限制到min-max区间,这步会把不相交的两个框框inter表示的wh计算成0
        inter   = torch.clamp((max_xy - min_xy), min=0)
        inter   = inter[:, :, 0] * inter[:, :, 1] # 计算w*h面积 [A,B]维度,每个值表示相交矩阵部分的面积
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]
    
    def get_target(self, l, targets, anchors, in_h, in_w):
        '''

        :param l: 第几个特征层
        :param targets:GT框,维度为[目标 batch , 左上角xy和右下角xy(归一化) 置信度]
        :param anchors: 缩放到特征图上后先验框的大小
        :param in_h:特征图高
        :param in_w:特征图宽
        :return:
        y_true:32x3x13x13x25矩阵,最后的25维表示sigmoid(tx),sigmoid(ty),tw,th,有无目标,类别
        noobj_mask:32x3x13x13 无目标为1,有目标为0
        box_loss_scale:loss比例,大目标loss权重小，小目标loss权重大
        '''
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs              = len(targets)
        #-----------------------------------------------------#
        #   用于选取哪些先验框不包含物体
        #-----------------------------------------------------#
        # len(self.anchors_mask[l]) 值为3 ,表示每个特征图中先验框个数
        noobj_mask      = torch.ones(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   让网络更加去关注小目标
        #-----------------------------------------------------#
        box_loss_scale  = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, requires_grad = False)
        #-----------------------------------------------------#
        #   batch_size, 3, 13, 13, 5 + num_classes
        #-----------------------------------------------------#
        y_true          = torch.zeros(bs, len(self.anchors_mask[l]), in_h, in_w, self.bbox_attrs, requires_grad = False)
        for b in range(bs):
            # 针对每张图片进行分析,此图片不含目标,跳过
            if len(targets[b])==0:
                continue
            # batch_target 每张图上的目标
            batch_target = torch.zeros_like(targets[b])
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            # batch_target[:,[0,2]]等同于batch_target[...,[0,2]]
            # 把最后一个维度索引为0和2的索引抽出来,造成一个同维度的矩阵
            # x w 和 y h 还有置信度,原先都是归一化的值,映射到特征图尺寸上
            batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
            batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            #-------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            #-------------------------------------------------------#
            # ground_truth
            # cat:拼接,后面的1表示按维度1拼接,维度计数从0开始
            # 相当于列数增加矩阵变宽
            gt_box          = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #-------------------------------------------------------#
            #   将先验框转换一个形式
            #   9, 4
            #-------------------------------------------------------#
            # anchors为9x2矩阵 输出anchor_shapes为[9,4]
            anchor_shapes   = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            #-------------------------------------------------------#
            #   计算交并比
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #   因为先验框是没有位置的,只有宽高,并且这个位置不是计算最匹配的交并比的要素,我们计算最合适的先验框肯定是计算交并比最大的
            #   所以传入的时候,gt框和anchor框的xy中心位置坐标都为0,因为不需要考虑,只找个交并比最大的框
            #-------------------------------------------------------#
            # torch.argmax(, dim=-1) 返回tensor最大值索引,dim=-1表示倒数第一个维度的索引,如果是二维,则是列索引
            # 这里写错了,argmax会降维,返回的best_ns是 [num_GT] 表示每一个真实框和最重合的先验框的序号
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[l]:
                    # 如果交并比最大的框不是属于这张特征图的框框,,则退出,让下一张合适的特征图框框来预测
                    continue
                #----------------------------------------#
                #   判断这个先验框是当前特征点的哪一个先验框
                #----------------------------------------#
                k = self.anchors_mask[l].index(best_n)
                #----------------------------------------#
                #   获得真实框属于哪个网格点,也就是找到GT框对应的xy位置坐标
                #----------------------------------------#
                i = torch.floor(batch_target[t, 0]).long()
                j = torch.floor(batch_target[t, 1]).long()
                #----------------------------------------#
                #   取出真实框的种类
                #----------------------------------------#
                c = batch_target[t, 4].long()

                #----------------------------------------#
                #   noobj_mask代表无目标的特征点
                # 发现了目标设置为有目标,b表示batch,k表示l特征图的第几个先验框,j,i表示特征点y,x坐标
                #----------------------------------------#
                noobj_mask[b, k, j, i] = 0
                #----------------------------------------#
                #   tx、ty代表中心调整参数的真实值
                #----------------------------------------#
                y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float() #求出对应sigmodi(x)值 ,距离x还有一层sigmoid函数还不是直接的x
                y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0]) # 求出tw值
                y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1]) # 求出th值
                y_true[b, k, j, i, 4] = 1
                y_true[b, k, j, i, c + 5] = 1
                #----------------------------------------#
                #   用于获得xywh的比例
                #   大目标loss权重小，小目标loss权重大
                #----------------------------------------#
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale

    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        #-----------------------------------------------------#
        #   计算一共有多少张图片
        #-----------------------------------------------------#
        bs = len(targets)

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor  = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        #-----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        #-----------------------------------------------------#
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高,只针对这张特征图的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))
        # 变成32x3x13x13维度的矩阵,3代表先验框个数
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        #-------------------------------------------------------#
        #   计算调整后的先验框中心与宽高,输入的x,y,w,h都是预测给出的值,x,y是经过了sigmoid的wh未经过exp
        #-------------------------------------------------------#
        pred_boxes_x    = torch.unsqueeze(x.data + grid_x, -1) # 变换后pred_boxes_x为[batch,3,13,13,1]
        pred_boxes_y    = torch.unsqueeze(y.data + grid_y, -1)
        pred_boxes_w    = torch.unsqueeze(torch.exp(w.data) * anchor_w, -1)
        pred_boxes_h    = torch.unsqueeze(torch.exp(h.data) * anchor_h, -1)
        # 最后维度拼接,变成[batch,3,13,13,4]
        pred_boxes      = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h], dim = -1)
        
        for b in range(bs):           
            #-------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      [num_anchors, 4]
            #-------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            #-------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            #-------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                #-------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                #-------------------------------------------------------#
                batch_target[:, [0,2]] = targets[b][:, [0,2]] * in_w
                batch_target[:, [1,3]] = targets[b][:, [1,3]] * in_h
                batch_target = batch_target[:, :4]
                #-------------------------------------------------------#
                #   计算交并比 这里是拿实际在特征图上的坐标计算交并比,已经转换过batch_target为GT,pred_boxes_for_ignore为预测
                #   anch_ious       num_true_box, num_anchors
                #-------------------------------------------------------#
                # 获得真实框和所有预测输出的框框(基于先验框调整)的iou
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)
                #-------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                #-------------------------------------------------------#
                # 获得每个先验框和哪个真实框iou最大, 获取最大的iou
                anch_ious_max, _    = torch.max(anch_ious, dim = 0)
                # 再转换成特征图的尺寸,如[3,13,13]
                anch_ious_max       = anch_ious_max.view(pred_boxes[b].size()[:3])
                # noobj_mask:[batch,3,13,13],iou大于阈值的设置为0,有目标
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask

def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('weights_init initialize network with %s type' % init_type)
    net.apply(init_func)
