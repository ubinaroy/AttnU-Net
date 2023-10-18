from model.model import AttentionUNet
from utils.dataset import ISBI_Loader
from utils.save_loss import save_loss_history

from torch import optim
import torch.nn as nn
import torch

import glob
import numpy as np
import os
import cv2

from train import train_net
from predict import predict

from parser import args
from tqdm import tqdm

def main():
    ################################# Initialization() #########################



    ################################## DataLoader() ############################
    print("\n>>>>>> Data Loading...")
    isbi_dataset = ISBI_Loader(args.train_path)
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in tqdm(train_loader):
        print(image.shape)
    print("<<<<<< Data Loaded.\n")
    ############################################################################
    ################################## if train(): ###########################
    
    if args.type == "train":
      print("\n>>>>>> Start Training...\n")
      # 选择设备，有cuda用cuda，没有就用cpu
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      # 加载网络，图片单通道1，分类为1。
      net = AttentionUNet()
      # 将网络拷贝到deivce中
      net.to(device=device)
      
      # 指定训练集地址，开始训练
      train_path = args.train_path

      # 保存 loss, 以便绘图
      loss_history = []

      train_net(net, device, train_path, loss_history)

      save_loss_history(loss_history, args.save_loss_path)
      
      print("\n <<<<<< Train Finished.\n")

    ##########################################################################
    ################################## if predict(): ##########################
    if args.type == "predict":
      if not os.path.exists(args.save_model):
        assert("\n\nYou must Train before Predict!!!\n")
      print ("\n>>>>>> Strating Predict......\n")
      # 选择设备，有cuda用cuda，没有就用cpu
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      # 加载网络，图片单通道1，分类为1。
      net = AttentionUNet()
      # 将网络拷贝到deivce中
      net.to(device=device)
      net.load_state_dict(torch.load(args.save_model, map_location=device))
      # 测试模式
      net.eval()
      # 读取所有图片路径
      tests_path = glob.glob(args.test_path + '*.png')

      predict(net, tests_path, device)
      print("\n <<<<<< Predicting Finished.\n")
    ############################################################################


if __name__ == "__main__":
    main()