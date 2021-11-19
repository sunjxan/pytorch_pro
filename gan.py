import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


epochs = 10
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval_steps = 100
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


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
# 训练数据生成器，先随机打乱，然后按batch_size分批，样本不重不漏，最后一个batch样本数量可能不足batch_size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)


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

class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = Generator().to(device)
        self.D = Discriminator().to(device)
        self.G_criterion = nn.BCELoss()
        self.G_optimizer = optim.SGD(self.G.parameters(), lr=learning_rate, momentum=momentum)
        self.D_criterion = nn.BCELoss()
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=learning_rate, momentum=momentum)
    def forward(self, input):
        return self.G(input)


# 可视化 tensorboard --logdir=runs --bind_all
writer = SummaryWriter(logdir='runs-alexnet')
# 设备
# 执行前设置环境变量 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 mnist.py
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_count = torch.cuda.device_count()
    print('use {} gpu(s)'.format(device_count))
else:
    print('use cpu')
# 程序中会对可见GPU重新从0编号
device = torch.device("cuda:0" if cuda_available else "cpu")
# 模型
gan = GAN()


def get_noises(*shape):
    return torch.rand(shape)

def fit(gan, epochs, initial_epoch=0):

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
    
        for step_index, (images, labels) in enumerate(train_loader):

            step_input_images = images.shape[0]
            
            gan.D_optimizer.zero_grad()
            
            real_images = images.to(device)
            real_labels = torch.ones(step_input_images, 1).to(device)
            real_outputs = gan.D(real_images)
            
            D_loss = gan.D_criterion(real_outputs, real_labels)
            D_loss.backward()
            
            noises = get_noises(step_input_images, 128).to(device)
            fake_images = gan.G(noises).detach()
            fake_labels = torch.zeros(step_input_images, 1).to(device)
            fake_outputs = gan.D(fake_images)
            
            D_loss = gan.D_criterion(fake_outputs, fake_labels)
            D_loss.backward()
            
            gan.D_optimizer.step()
            
            gan.G_optimizer.zero_grad()
            
            noises = get_noises(step_input_images, 128).to(device)
            fake_images = gan.G(noises)
            fake_outputs = gan.D(fake_images)
            
            G_loss = gan.G_criterion(fake_outputs, real_labels)
            G_loss.backward()
            
            gan.G_optimizer.step()

global_step = 0
fit(gan, epochs, 0)


def evaluate(gan):
    
    with torch.no_grad():
    
        for step_index, (images, labels) in enumerate(test_loader):

            step_input_images = images.shape[0]
            
            real_images = images.to(device)
            real_labels = torch.ones(step_input_images, 1).to(device)
            real_outputs = gan.D(real_images)
            
            D_loss = gan.D_criterion(real_outputs, real_labels)
            
            noises = get_noises(step_input_images, 128).to(device)
            fake_images = gan.G(noises)
            fake_labels = torch.zeros(step_input_images, 1).to(device)
            fake_outputs = gan.D(fake_images)
            
            D_loss = gan.D_criterion(fake_outputs, fake_labels)

global_step = 0
evaluate(gan)

# https://zhuanlan.zhihu.com/p/72279816
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
# https://clay-atlas.com/blog/2020/01/09/pytorch-chinese-tutorial-mnist-generator-discriminator-mnist/
# https://www.pytorchtutorial.com/50-lines-of-codes-for-gan/
# https://github.com/devnag/pytorch-generative-adversarial-networks/blob/master/gan_pytorch.py
# https://zhuanlan.zhihu.com/p/137571225
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
