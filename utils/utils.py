import numpy as np
from PIL import Image

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    # 有三维度,并且颜色np.shape(image)[2]维度为3
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    
#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    '''
    :param image:
    :param size:
    :param letterbox_image: 是否等比缩放,剩下的部分会填黑边
    :return:
    '''
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        # Image.BICUBIC ：三次样条插值
        image   = image.resize((nw,nh), Image.BICUBIC)
        # 源代码填的是灰边,这里填纯黑边
        # new_image = Image.new('RGB', size, (128,128,128))
        new_image = Image.new('RGB', size, (0,0,0))
        # 居中粘贴
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    '''
    voc数据集为20个类

    :param classes_path:
    :return: 类名list+类个数
    '''
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''
    获得先验框

    :param anchors_path:
    :return:返回的是一个二维数组 形如[[10,13], [16,30],...]

    第二个参数为先验框个数

    '''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    # float转化函数中有空格也没事的
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    #
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    '''
    0-255归一化0-1
    :param image:
    :return:
    '''
    image /= 255.0
    return image
