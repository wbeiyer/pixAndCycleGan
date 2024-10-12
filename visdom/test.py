from visdom import Visdom
import numpy as np
import time
import requests
import os

# 实例化一个窗口
wind = Visdom()
# 初始化窗口信息
Y=np.array([0.])
wind.line( Y, # Y的第一个点的坐标
          Y, # X的第一个点的坐标
           win = 'train_loss', # 窗口的名称
           opts = dict(title = 'train_loss') # 图像的标例
         )
 
# 更新数据
for step in range(10):
    # 随机获取loss,这里只是模拟实现
    loss = np.random.randn() * 0.5 + 2
    wind.line([loss],[step], win = 'train_loss', update = 'append')
    time.sleep(0.5)
# 初始化窗口参数
# wind.line([[0.,0.]],[0.], win = 'train',opts = dict(title = 'loss&acc',legend = ['loss','acc']))
 
 
# # 更新窗口数据
# for step in range(10):
#     loss = 0.2 * np.random.randn() + 1
#     acc =  0.1 * np.random.randn() + 0.5
#     wind.line([[loss, acc]],[step],win = 'train',update = 'append')
#     time.sleep(0.5)
# 图片
# 单张图片
# wind.image(
#      np.random.rand(3, 512, 256),
#      opts={
#         'title': 'Random',
#         'showlegend': True
#     }
# )