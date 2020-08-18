# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, Response, json
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
import numpy as np
from notebooks import video_3, visualization_camera_on
url=''
def gen(cap):
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image_np = frame
            image_np_expanded = np.expand_dims(image_np, axis=0)
            rclasses, rscores, rbboxes = video_3.process_image(image_np)
            ## www为json格式的文字说明
            visualization_camera_on.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
            return frame

def get_frame(cap):
    frame = gen(cap)
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def stream(cap):
    while True:
        # ff = jpeg.tobytes()
        ff = get_frame(cap)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + ff + b'\r\n\r\n')

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/online', methods=['POST'])  # 添加路由
def offline():
    if request.method == 'POST':
        # f = request.files['text']
        data = json.loads(request.form.get('data'))
        print(data)
        username = data['username']   ##输入的内容
        username = username['0']
        username = username['value']
        print(username)
        global url
        url = username  ##使用本机摄像头
        # url = 'http://admin:admin@'+ username   ## 使用局域网
        # url = 'rtsp://admin:admin@' + username + '/live'   ## 使用IP摄像头rtsp
        # print(url)
        # basepath = os.path.dirname(__file__)  # 当前文件所在路径
        # result_name = secure_filename(f.filename)
        # upload_path = os.path.join(basepath, 'static/images', result_name)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images', 'test.mp4')
        # global path  ## 设为全局变量,报错
        # path = upload_path
        # print(path)
        l = 'video_feed'
        V = {'lj': l}
        VVV = jsonify(V)
        return VVV

@app.route('/video_feed')
def video_feed():
    # 安卓实时
    # url = "http://admin:admin@192.168.1.101:8081"
    # cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(url)      ## 使用IP摄像头rtsp
    cap = cv2.VideoCapture(int(url))   ##使用本机摄像头
    video_stream = Response(stream(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    return video_stream

@app.route('/word', methods=['GET'])
def explain():
    f = open('E:\project\detection\SSD-Tensorflow_gw\Flask\static/result/video_on.txt')
    a = f.read()
    f.close()
    W = {'w': a}
    WWW = jsonify(W)
    return WWW

if __name__ == '__main__':
    app.run(debug=True)
