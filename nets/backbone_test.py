#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  backbone_test.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/07 16:30
#        Email @  LJjiahf@163.com
#  Description @  
# ********************************************************************


import torch
import numpy as np
import os

def load_pic(path,out_size=(224,224)):
    from PIL import Image
    '''

    :param path:
    :return: torch.Tensor 维度[1,3,224,224]
    '''
    # 加载一张图片预测看一下
    img=Image.open(path)
    img=img.resize(out_size)
    print('img resize size ',img.size)

    data=np.array(img,dtype=np.float32)/255.
    data=torch.from_numpy(data)
    data=data.permute(2,0,1).contiguous().unsqueeze(dim=0)
    return data


def net_test(net,myself_pth_path,img_path,net_type='classify'):
    '''

    :param net:
    :param myself_pth_path:
    :param img_path:
    :param net_type: 网络类型
    classify 分类
    feature 获取特征图
    :return:
    '''
    assert os.path.exists(myself_pth_path)
    assert os.path.exists(img_path)
    net.eval()
    model_dict = net.state_dict()
    # print("model struct")
    # print("=" * 50)
    # for k in model_dict.keys():
    #     print(k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型到device上
    pretrained_dict = torch.load(myself_pth_path, map_location=device)
    # 预训练模型和本地模型shape相等则加载权重
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict and np.shape(model_dict[k]) == np.shape(v))}
    print("=" * 50)

    # 查看未加载的权重
    for k in model_dict.keys():
        if k not in pretrained_dict:
            print("local model layer [%s] [%s]not load!" % (k, np.shape(model_dict[k])),model_dict[k])
    print("load all model weight")
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    data = load_pic(img_path, out_size=(416, 416))
    out = net(data)
    if net_type=='classify':
        print("forward ok ", out.shape)
        print(torch.max(out, dim=1)[1])
    elif net_type=='feature':
        for m in out:
            print(m.shape)
    else :
        print('input correct net type')



def change_offical_name_to_myself(net):
    myself_pth = 'change_official_ghost.pth'
    offical_pth = 'state_dict_73.98.pth'
    net.eval()

    model_dict = net.state_dict()
    # 加载模型到device上
    pretrained_dict = torch.load(offical_pth)
    for k,v in pretrained_dict.items():
        # 修改一下变量名,然后重新保存字典
        loc_name=k.replace('blocks.','blocks')
        if loc_name in model_dict:
            model_dict[loc_name]=v
        else:
            print(k,"not load")
    torch.save(model_dict,myself_pth)


def show_dict_keys(d):
    print("print OrderedDict")
    for k, v in d.items():
        print(k, np.shape(v))
    print('\n')

if __name__ == '__main__':
    # save_offical2myself()
    # classify_test()
    pass

