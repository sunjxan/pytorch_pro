import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


epochs = 100
batch_size_train = 100
learning_rate = 0.01
momentum = 0.5
log_interval_steps = 100
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


class DataSet(torch.utils.data.Dataset):
    def __init__(self):
        # 转换器，将PIL Image转换为Tensor，提供MNIST数据集单通道数据的平均值和标准差，将其转换为标准正态分布
        transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
        if not os.path.isdir('./data/MNIST'):
            # 训练集，(60000, 2, 1, 28, 28)
            train_set = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            # 测试集，(10000, 2, 1, 28, 28)
            test_set = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        else:
            train_set = tv.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
            test_set = tv.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
        self._dataset = []
        for image, label in train_set:
            self._dataset.append((image, 1))
        for image, label in test_set:
            self._dataset.append((image, 1))

    def __len__(self):
        return len(self._dataset)

	def __getitem__(self, index):
        if 0 <= index < len(self._dataset):
            return self._dataset[index]
        return None

real_data_set = DataSet()        
# 真实数据生成器，先随机打乱，然后按batch_size分批，样本不重不漏，最后一个batch样本数量可能不足batch_size
real_data_loader = torch.utils.data.DataLoader(real_data_set, batch_size=batch_size_train, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh()
        )
    def forward(self, input):
        img = self.model(input)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input)

def GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
    def forward(self, input):
        return self.G(input)


def get_noises(*shape):
    return torch.rand(shape)

def fit(gan, optimizer, epochs, initial_epoch=0):
    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        for step_index, (images, labels) in enumerate(real_data_loader):
            step_input_images = images.shape[0]
            
            real_images = images.to(device)
            real_labels = labels.to(device)
            real_outputs = gan.D(real_images)
            
            noises = get_noises(step_input_images, 128).to(device)
            fake_images = gan.G(noises).detach()
            fake_labels = torch.zeros(step_input_images, 1).to(device)
            fake_outpus = gan.D(fake_images)
