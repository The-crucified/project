# coding: utf-8
import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
slim = tf.contrib.slim

# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
# sys.path.append('../')
# sys.path.append(r'D:\project\detection\SSD-Tensorflow')   #为其他文件夹下的文件添加路径
sys.path.append(r'E:\project\detection\SSD-Tensorflow_gw')  ## 台式机路径
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization_camera  # visualization
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)      # 所有的去掉compat.v1就是原来的，但是会发出警告
config = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.compat.v1.InteractiveSession(config=config)

# ## SSD 300 Model
#
# The SSD 300 network takes 300x300 image inputs. In order to feed any image, the latter is resize to this input shape (i.e.`Resize.WARP_RESIZE`). Note that even though it may change the ratio width / height, the SSD model performs well on resized images (and it is the default behaviour in the original Caffe implementation).
#
# SSD anchors correspond to the default bounding boxes encoded in the network. The SSD net output provides offset on the coordinates and dimensions of these anchors.

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.compat.v1.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'  # 可更改为自己的模型路径
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.compat.v1.global_variables_initializer())
# saver = tf.train.Saver()
saver = tf.compat.v1.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# ## Post-processing pipeline
#
# The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:
#
# * Select boxes above a classification threshold;
# * Clip boxes to the image shape;
# * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
# * If necessary, resize bounding boxes to original image shape.


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# # Test on some demo image and visualize output.
# path = '../demo/'
# image_names = sorted(os.listdir(path))

# img = mpimg.imread(path + image_names[-5])
# rclasses, rscores, rbboxes =  process_image(img)

# # visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma)
# visualization.plt_bboxes(img, rclasses, rscores, rbboxes)


##### following are added for camera demo####
# input = r'D:/project/detection/SSD-Tensorflow/demo/cars.mp4'   # 本地视频路径
# input = r'E:/project/detection/SSD-Tensorflow/demo/cars.mp4'    ## 台式机路径
cap = cv2.VideoCapture(0)#调出本地摄像头
#cap = cv2.VideoCapture(1)#调出外接摄像头
# 安卓实时
# url = 'rtsp://admin:admin@192.168.1.188:8554/live'
# cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(input)    ####输入视频路径

# fourcc = cv2.VideoWriter_fourcc(*'DIXV')  # *'DIXV'旧版avi的编码格式，*'XVID'新版avi的编码格式，可以尝试调换使用
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')     # mp4格式文件
# out = cv2.VideoWriter('D:/project/detection/SSD-Tensorflow/results/test.avi', fourcc, fps, size, True)

# fps = cap.get(cv2.CAP_PROP_FPS)          ## 本地视频
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# fourcc = cap.get(cv2.CAP_PROP_FOURCC)
# fourcc = cv2.CAP_PROP_FOURCC(*'CVID')
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')     #视频编码格式
out = cv2.VideoWriter('E:/project/detection/SSD-Tensorflow/results/test.avi', fourcc, 30.0, size, True)   ##本机摄像头

# print('fps=%d,size=%r,fourcc=%r' % (fps, size, fourcc))     # 本地视频
# delay = 30 / int(fps)                   # 本地视频 每一帧持续的时间
delay = 60 / int(30)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #          image = Image.open(image_path)
        #          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = frame
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = image
        #          image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        rclasses, rscores, rbboxes = process_image(image_np)
        # Visualization of the results of a detection.
        visualization_camera.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
        #          plt.figure(figsize=IMAGE_SIZE)
        #          plt.imshow(image_np)

        # out.write(frame)
        # plt.imshow(image_np)
        cv2.imshow('frame', image_np)
        cv2.waitKey(1)    # 本机摄像头
        # cv2.waitKey(np.uint(delay))    # 本地视频

        #print('there are %s cars'% num)
        print('Ongoing...')
    else:
        break
    out.write(frame)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
