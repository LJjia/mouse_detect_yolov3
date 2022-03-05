#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  deocode_mov.py
#       Author @  Jia Liangjun
#  Create date @  2022/02/23 18:52
#        Email @  LJjiahf@163.com
#  Description @  尝试解码mov视频
# ********************************************************************


import cv2

# 可以读取mp4或者mov视频
raw = cv2.VideoCapture('./film.mp4')

while 1:
    ret, frame = raw.read()
    print("ret",ret)
    cv2.imshow('hsv', frame)
    cv2.waitKey(10)

# import torchvision.models as models
# net=models.()
# print(net)

