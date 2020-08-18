# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, Response
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
import numpy as np
from notebooks import video_3, visualization_camera_off
# path=''
pathlist = []

def list(path):
    cap = cv2.VideoCapture(path)
    if len(pathlist) > 1:
        for i in range(len(pathlist)-1, len(pathlist)):
            if pathlist[i-1] == pathlist[i]:
                break
            else:
                cap = cv2.VideoCapture(path)
                pathlist[i-1] = pathlist[i]
    else:
        pass
    return cap

def gen(cap):
    if len(pathlist) > 1:
        for i in range(len(pathlist)-1, len(pathlist)):
            if pathlist[i-1] == pathlist[i]:
                break
            else:
                # cap.release()
                cap = cv2.VideoCapture(path)
                pathlist[i-1] = pathlist[i]
    else:
        pass
    # cap = cv2.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image_np = frame
            image_np_expanded = np.expand_dims(image_np, axis=0)
            rclasses, rscores, rbboxes = video_3.process_image(image_np)
            ## www为json格式的文字说明
            visualization_camera_off.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
            return frame

# def gen(cap):
#     while (cap.isOpened()):
#         ret, frame = cap.read()
#         if ret == True:
#             image_np = frame
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             rclasses, rscores, rbboxes = video_3.process_image(image_np)
#             ## www为json格式的文字说明
#             visualization_camera_off.bboxes_draw_on_img(image_np, rclasses, rscores, rbboxes)
#             return frame

def get_frame(cap):
    frame = gen(cap)
    # print('1')
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

@app.route('/offline', methods=['POST'])  # 添加路由
def offline():
    if request.method == 'POST':
        f = request.files['file']
        user_input = request.form.get("name")
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        result_name = secure_filename(f.filename)
        upload_path = os.path.join(basepath, 'static/images', result_name)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images', 'test.mp4')
        global path, pathlist   ## 设为全局变量,报错
        path = upload_path
        pathlist.append(path)
        print(len(pathlist))
        # cap = cv2.VideoCapture(path)
        # cap = list(path)
        # print(cap)
        # cap.release()
        # if path == '':
        #     pass
        # else:
        #     cap.release()
        # cap = cv2.VideoCapture(path)
        # cv2.destroyAllWindows()
        l = 'video_feed'
        V = {'lj': l}
        VVV = jsonify(V)
        return VVV
## 切换文件后，为何不运行
@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    print('ok')
    # cap.release()
    # cap = list(path)
    cap = cv2.VideoCapture(path)
    video_stream = Response(stream(cap), mimetype='multipart/x-mixed-replace; boundary=frame')
    return video_stream

@app.route('/word', methods=['GET'])
def explain():
    f = open('E:\project\detection\SSD-Tensorflow_gw\Flask\static/result/video_off.txt')
    a = f.read()
    f.close()
    W = {'w': a}
    WWW = jsonify(W)
    return WWW

if __name__ == '__main__':
    app.run(debug=True)
