1.调通在线视频的检测，如何通过按钮结束进程（如何做到点击按钮运行程序）
2.结合流视频的方法，将检测后的视频显示在网页上
3.将图像检测和视频检测放在一个框架里
4.将原始视频显示在网页上

demo_test.py
调用visualizaton.py检测图片显示类别，还可以统计图中人的数量，通过判断标签是否为人进行统计。
调用visualizaton_2.py检测图片显示类别，还可以统计图中每一类别的数量并将结果保存在results。
from notebooks import visualization_2

ssd_off.py    ssd_online.py
from notebooks import visualization_camera

app.py
from notebooks import demo_1
from notebooks import visualization_3

app_on.py
from notebooks import video_3, visualization_camera_on

app_off.py
from notebooks import video_3, visualization_camera_off


注意路径问题！！！
