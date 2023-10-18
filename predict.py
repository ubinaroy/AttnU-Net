import glob
import numpy as np
import torch
import os
import cv2
from model.model import AttentionUNet

from tqdm import tqdm

from parser import args

def predict(net, tests_path, device):

    # 遍历所有图片
    for test_path in tqdm(tests_path):
        # 保存结果地址
        save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        cv2.imwrite(save_res_path, pred)

        ###############################################################################
        # pred 就是输出，或可直接 load 