from model.model import AttentionUNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

from tqdm import tqdm

from parser import args

def train_net(net, device, data_path, loss_history, epochs=40, batch_size=1, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷 
    # 为什么要初始化为 inf？ 因为 BCEWithLogitsLoss 的数据将经过 softmax：frac{1}{1+\math{exp}^(-x)}
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in tqdm(train_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)

            # 计算loss
            loss = criterion(pred, label)
            loss_history.append(loss)

            print(f'Epoch {epoch}   Loss/train: ', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), args.save_model)
            # 更新参数
            loss.backward()
            optimizer.step()