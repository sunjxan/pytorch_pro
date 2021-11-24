import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


model_pkl = 'lenet.pkl'
parameters_pkl = 'lenet-parameters.pkl'
optimizer_pkl = 'lenet-optimizer.pkl'
epochs = 100
batch_size_train = 60
batch_size_test = 1000
learning_rate = 1e-3
log_interval_steps = 200
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


# 转换器，将PIL Image转换为Tensor，提供MNIST数据集单通道数据的平均值和标准差，将其转换为标准正态分布
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
if not os.path.isdir('./data/VOCdevkit'):
    # (60000, 2, 1, 28, 28)
    train_set = tv.datasets.voc.VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)
    val_set  = tv.datasets.voc.VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=transform)
else:
    train_set = tv.datasets.voc.VOCDetection(root='./data', year='2012', image_set='train', download=False, transform=transform)
    val_set  = tv.datasets.voc.VOCDetection(root='./data', year='2012', image_set='val', download=False, transform=transform)

pass