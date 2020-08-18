from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, Response
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
import json
# import sys
# sys.path.append(r'E:/project/detection/SSD-Tensorflow/')
from notebooks import demo_1

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

def explain():
    f = open('E:\project\detection\SSD-Tensorflow_gw\Flask\static/result/test.txt')
    a = f.read()
    f.close()
    return a

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

# app.route('/upload', methods=['POST', 'GET'])
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pic', methods=['POST'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")
        print(user_input)

        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        result_name = secure_filename(f.filename)
        print(result_name)
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        path = upload_path
        # path = r'E:\project\detection\SSD-Tensorflow\Flask\static\images/test.jpg'
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

        demo_1.test(path, result_name)
        A = explain()
        B = 'static/result'+'/'+result_name
        print(B)
        sss = {'username': A, 'lj': B}
        SSS = jsonify(sss)
        return SSS


if __name__ == '__main__':
    app.run(debug = True)
