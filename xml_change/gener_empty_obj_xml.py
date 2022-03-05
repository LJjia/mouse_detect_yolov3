#!/usr/bin/env python
#-*- coding:utf-8 _*-
__author__ = 'LJjia'
# *******************************************************************
#     Filename @  gener_empty_obj_xml.py
#       Author @  Jia Liangjun
#  Create date @  2022/03/05 10:59
#        Email @  LJjiahf@163.com
#  Description @  生成空的xml文件
# ********************************************************************

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree,Element
import os
import cv2

xml_head_str='''
<annotation>
	<folder>Desktop</folder>
	<filename>憨憨.jpg</filename>
	<path>/Users/ljjia/Desktop/憨憨.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>690</width>
		<height>690</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
</annotation>
'''

def read_xml_fromfile(in_path):
    '''''读取并解析xml文件
       in_path: xml路径
       return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def read_xml_fromstr(string):
    '''''读取并解析xml字符串
       in_path: xml路径
       return: ElementTree'''
    tree = ET.fromstring(string)
    return tree


def write_xml(tree, out_path):
    '''''将xml文件写出
       tree: xml树,注意一定是树对象,element对象没有write方法
       out_path: 写出路径'''
    # 省略前面的版本标注 xml_declaration=False
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


JPEGImages="JPEGImages"
Annotations="Annotations"


def proc(JPEGImages,Annotations,xml_str):
    pwd = os.getcwd()
    img_list=[x for x in os.listdir(JPEGImages)]
    for file_path in img_list:
        img=cv2.imread(os.path.join(JPEGImages,file_path))
        h,w,c=img.shape
        xml=read_xml_fromstr(xml_str)
        xml.find('size').find('width').text=str(w)
        xml.find('size').find('height').text = str(h)
        xml.find('size').find('depth').text = str(c)
        xml.find('folder').text = str(JPEGImages)
        xml.find('filename').text = str(file_path)
        xml.find('path').text = str(os.path.join(pwd,file_path))
        # element obj 要转换为elementtree才能保存
        tree = ET.ElementTree(xml)
        # 保存为对应路径下的xml文件
        dst_file=os.path.splitext(file_path)[0]+'.xml'
        write_xml(tree,os.path.join(Annotations,dst_file))
        print("save xml file ",dst_file)



if __name__ == '__main__':
    proc(JPEGImages,Annotations,xml_head_str)
    # img=cv2.imread(os.path.join(JPEGImages,'呕吼,完蛋.jpg'))
    # print(img.shape,img.size)

