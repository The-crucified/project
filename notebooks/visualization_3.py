# Copyright 2017 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

save_path_1 = r'E:\project\detection\SSD-Tensorflow_gw\Flask\static\result'
save_path = r'E:/project/detection/SSD-Tensorflow_gw/results'   #台式机路径
# save_path = r'D:/project/detection/SSD-Tensorflow/results'
# =========================================================================== #
# Some colormaps.
# =========================================================================== #
def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors_plasma = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


# =========================================================================== #
# OpenCV drawing.
# =========================================================================== #
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """Draw a collection of lines on an image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def draw_bbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0]+15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    # cv2.imshow('name', img)
    # cv2.waitKey()
    # cv2.imwrite(save_path + '/' + 'name' + '.jpg', img)    ##保存结果图，但是有问题，保存方法对

def bboxes_draw_on_img(img, classes, scores, bboxes, colors, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)
        # cv2.imwrite(save_path + '/' + 'name' + '.jpg', img)


# =========================================================================== #
# Matplotlib show...
# modifed by wangjc,2017.10.18
# =========================================================================== #


def plt_bboxes(img, result_name, classes, scores, bboxes, figsize=(10,10), linewidth=1.5):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """

    #################added
    List = {}
    # import SSD_Tensorflow_master.datasets.pascalvoc_2007 as pas
    from datasets import pascalvoc_2007 as pas
    x = pas.pascalvoc_common.VOC_LABELS.items()
    for name, item in x:
        List[name] = 0
    ##### 将数字转化为类别进行显示
    def num2class(n):
        # import tensorflow.models.SSD_Tensorflow_master.datasets.pascalvoc_2007 as pas
        # from datasets import pascalvoc_2007 as pas
        # x=pas.pascalvoc_common.VOC_LABELS.items()
        for name,item in x:
            if n in item:
                List[name] = List[name] + 1
                return name, List
        #print(num)
    ###########################added

    fig = plt.figure(figsize=figsize)
    # cv2.imshow('name1', img)
    # cv2.waitKey()
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    # person_num = 0       ##初始化图中没有人
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            #score = 0.01
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            #class_name = str(cls_id)
            ###################added
            #class_name = ['haha','a','ss']
            class_name, value = num2class(cls_id)
            # person_num = person_num +NUM    ####### 统计图中人数
            ##################added
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')

    values = list(value.values())
    l = len(values)
    Name = list(value.keys())
    test = open(save_path_1 + '/test.txt', 'w')  # 将检测的文字说明进行保存
    test.truncate()     # 清空文件
    for i in range(l):
        if values[i] > 0:
            lable = ('%s %s; ' % (values[i], Name[i]))
            print(lable)
            test.write(lable)   # 将文本写入
            test.write("\n")       # 换行
    test.close()
    # print('There are %s people'% person_num)
    # print(values)
    # print(Name)
    # cv2.imshow('name', img)
    # cv2.waitKey()
    plt.savefig(save_path_1 + '/' + result_name)
    plt.savefig(save_path + '/' + result_name)
    # plt.show()
    # plt.savefig(save_path_1 + '/' + 'test.jpg')
    # plt.savefig(save_path + '/' + 'test.jpg')
