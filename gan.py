import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

# import matplotlib.pyplot as plt

import os, time


model_pkl = 'gan.pkl'
parameters_pkl = 'gan-parameters.pkl'
optimizer_G_pkl = 'gan-optimizer-G.pkl'
optimizer_D_pkl = 'gan-optimizer-D.pkl'
epochs = 500
batch_size_train = 100
batch_size_test = 1000
learning_rate = 1e-3
log_interval_steps = 100
noise_size = 100
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


# 转换器，将PIL Image转换为Tensor，提供MNIST数据集单通道数据的平均值和标准差，将其转换为标准正态分布
normalize = tv.transforms.Normalize((0.1307,), (0.3081,))
transform = tv.transforms.Compose([tv.transforms.ToTensor(), normalize])
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
            nn.Linear(in_features=noise_size, out_features=200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(in_features=200, out_features=28*28),
            nn.Sigmoid()
        )
    def forward(self, input):
        img = self.model(input)
        return img.view(img.size(0), 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(in_features=200, out_features=1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.model(input)

class GAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()
    def forward(self, input):
        return self.G(input)


# 可视化 tensorboard --logdir=runs-gan --bind_all
writer = SummaryWriter(logdir='runs-gan')
# 设备
# 执行前设置环境变量 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 filename.py
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
if os.path.isfile(parameters_pkl):
    gan.load_state_dict(torch.load(parameters_pkl))
if cuda_available and device_count > 1:
    gan = nn.DataParallel(gan, device_ids=list(range(device_count)), output_device=0)
gan = gan.to(device)
# 损失函数
criterion_G = nn.BCELoss()
criterion_D = nn.BCELoss()
# 优化器
optimizer_G = optim.Adam(gan.G.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(gan.D.parameters(), lr=learning_rate)
if os.path.isfile(optimizer_G_pkl):
    optimizer_G.load_state_dict(torch.load(optimizer_G_pkl))
if os.path.isfile(optimizer_D_pkl):
    optimizer_D.load_state_dict(torch.load(optimizer_D_pkl))


def get_noises(num_images):
    return torch.randn(num_images, noise_size)

def visualize_model(images, filename, dirname='gan-images'):
    if not os.path.isdir(dirname):
        os.system('mkdir {:s}'.format(dirname))
    # grid = tv.utils.make_grid(images, nrow=10).permute(1, 2, 0)
    # plt.imshow(grid.cpu())
    # plt.savefig('./{:s}/{:s}'.format(dirname, filename))
    tv.utils.save_image(images, './{:s}/{:s}'.format(dirname, filename), nrow=10)

def fit(gan, epochs, initial_epoch=1):
    global global_step

    # 设置model.training为True，使模型中的Dropout和BatchNorm起作用
    gan.train()

    steps_per_epoch = len(train_loader)
    total_train_images = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        print('Train Epoch {}/{}'.format(epoch_index, initial_epoch + epochs - 1))
        print('-' * 20)

        G_epoch_loss_sum = 0
        D_epoch_loss_sum = 0

        # True Positive(真正，TP)：将正类预测为正类
        # True Negative(真负，TN)：将负类预测为负类
        # False Positive(假正，FP)：将负类预测为正类
        # False Negative(假负，FN)：将正类预测为负类
        epoch_TP_images = 0
        epoch_TN_images = 0

        epoch_begin = time.time()

        for step_index, (images, labels) in enumerate(train_loader, 1):
            step_input_images = images.shape[0]

            step_begin = time.time()

            optimizer_D.zero_grad()

            real_images = images.to(device)
            real_labels = torch.ones(step_input_images, 1).to(device)
            real_outputs = gan.D(real_images)
            D_real_loss = criterion_D(real_outputs, real_labels)
            D_real_loss.backward()

            noises = get_noises(step_input_images).to(device)
            fake_images = gan.G(noises).detach()
            fake_labels = torch.zeros(step_input_images, 1).to(device)
            fake_outputs = gan.D(normalize(fake_images))
            D_fake_loss = criterion_D(fake_outputs, fake_labels)
            D_fake_loss.backward()

            optimizer_D.step()

            optimizer_G.zero_grad()

            noises = get_noises(step_input_images).to(device)
            sample_images = gan.G(noises)
            sample_outputs = gan.D(normalize(sample_images))

            G_loss = criterion_G(sample_outputs, real_labels)
            G_loss.backward()

            optimizer_G.step()

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            G_step_loss = G_loss.item()
            D_step_loss = (D_real_loss.item() + D_fake_loss.item()) / 2
            G_epoch_loss_sum += G_step_loss * step_input_images
            D_epoch_loss_sum += D_step_loss * step_input_images

            step_TP_images = (real_outputs >= 0.5).sum().item()
            step_TN_images = (fake_outputs < 0.5).sum().item()
            epoch_TP_images += step_TP_images
            epoch_TN_images += step_TN_images

            if step_index % log_interval_steps == 0:
                visualize_model(sample_images, 'gan-{:d}.png'.format(global_step))

                torch.save(gan.state_dict(), parameters_pkl)
                torch.save(optimizer_G.state_dict(), optimizer_G_pkl)
                torch.save(optimizer_D.state_dict(), optimizer_D_pkl)

                writer.add_scalars('train/loss', {'Generator': G_step_loss, 'Discriminator': D_step_loss}, global_step)
                writer.add_scalars('train/accuracy', {'TP': step_TP_images / step_input_images, 'TN': step_TN_images / step_input_images, 'ACC': (step_TP_images + step_TN_images) / (2 * step_input_images)}, global_step)

                print('Step {}/{}  Time: {:.0f}s {:.0f}ms  G_Loss: {:.4f}  D_Loss: {:.4f}  TP: {}/{} ({:.1f}%)  TN: {}/{} ({:.1f}%)  ACC: {}/{} ({:.1f}%)'.format(step_index, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, G_step_loss, D_step_loss, step_TP_images, step_input_images, 1e2 * step_TP_images / step_input_images, step_TN_images, step_input_images, 1e2 * step_TN_images / step_input_images, step_TP_images + step_TN_images, 2 * step_input_images, 1e2 * (step_TP_images + step_TN_images) / (2 * step_input_images)))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)

        print('-' * 20)
        print('Train Epoch {}/{}  Time: {:.0f}s {:.0f}ms  G_Loss: {:.4f}  D_Loss: {:.4f}  TP: {}/{} ({:.1f}%)  TN: {}/{} ({:.1f}%)  ACC: {}/{} ({:.1f}%)'.format(epoch_index, initial_epoch + epochs - 1, int(epoch_period / 1e3), epoch_period % 1e3, G_epoch_loss_sum / total_train_images, D_epoch_loss_sum / total_train_images,  epoch_TP_images, total_train_images, 1e2 * epoch_TP_images / total_train_images, epoch_TN_images, total_train_images, 1e2 * epoch_TN_images / total_train_images, epoch_TP_images + epoch_TN_images, 2 * total_train_images, 1e2 * (epoch_TP_images + epoch_TN_images) / (2 * total_train_images)))
        print()

global_step = 0
fit(gan, epochs, 1)


def evaluate(gan):
    global global_step

    # 设置model.training为False，使模型中的Dropout和BatchNorm不起作用
    gan.eval()

    steps_total = len(test_loader)
    G_total_loss_sum = 0
    D_total_loss_sum = 0
    total_TP_images = 0
    total_TN_images = 0
    total_input_images = 0

    with torch.no_grad():
        print('Eval')
        print('-' * 20)

        test_begin = time.time()

        for step_index, (images, labels) in enumerate(test_loader, 1):
            step_input_images = images.shape[0]
            total_input_images += step_input_images

            step_begin = time.time()

            real_images = images.to(device)
            real_labels = torch.ones(step_input_images, 1).to(device)
            real_outputs = gan.D(real_images)
            
            D_real_loss = criterion_D(real_outputs, real_labels)
            
            noises = get_noises(step_input_images).to(device)
            fake_images = gan.G(noises)
            fake_labels = torch.zeros(step_input_images, 1).to(device)
            fake_outputs = gan.D(normalize(fake_images))
            
            D_fake_loss = criterion_D(fake_outputs, fake_labels)
            G_loss = criterion_G(fake_outputs, real_labels)

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            G_step_loss = G_loss.item()
            D_step_loss = (D_real_loss.item() + D_fake_loss.item()) / 2
            G_total_loss_sum += G_step_loss * step_input_images
            D_total_loss_sum += D_step_loss * step_input_images

            step_TP_images = (real_outputs >= 0.5).sum().item()
            step_TN_images = (fake_outputs < 0.5).sum().item()
            total_TP_images += step_TP_images
            total_TN_images += step_TN_images

            writer.add_scalars('test/loss', {'Generator': G_step_loss, 'Discriminator': D_step_loss}, global_step)
            writer.add_scalars('test/accuracy', {'TP': step_TP_images / step_input_images, 'TN': step_TN_images / step_input_images, 'ACC': (step_TP_images + step_TN_images) / (2 * step_input_images)}, global_step)

            print('Step {}/{}  Time: {:.0f}s {:.0f}ms  G_Loss: {:.4f}  D_Loss: {:.4f}  TP: {}/{} ({:.1f}%)  TN: {}/{} ({:.1f}%)  ACC: {}/{} ({:.1f}%)'.format(step_index, steps_total, int(step_period / 1e3), step_period % 1e3, G_step_loss, D_step_loss, step_TP_images, step_input_images, 1e2 * step_TP_images / step_input_images, step_TN_images, step_input_images, 1e2 * step_TN_images / step_input_images, step_TP_images + step_TN_images, 2 * step_input_images, 1e2 * (step_TP_images + step_TN_images) / (2 * step_input_images)))


        test_end = time.time()
        test_period = round((test_end - test_begin) * 1e3)

        print('-' * 20)
        print('Eval  Time: {:.0f}s {:.0f}ms  G_Loss: {:.4f}  D_Loss: {:.4f}  TP: {}/{} ({:.1f}%)  TN: {}/{} ({:.1f}%)  ACC: {}/{} ({:.1f}%)'.format(int(test_period / 1e3), test_period % 1e3, G_total_loss_sum / total_input_images, D_total_loss_sum / total_input_images, total_TP_images, total_input_images, 1e2 * total_TP_images / total_input_images, total_TN_images, total_input_images, 1e2 * total_TN_images / total_input_images, total_TP_images + total_TN_images, 2 * total_input_images, 1e2 * (total_TP_images + total_TN_images) / (2 * total_input_images)))
        print()

global_step = 0
evaluate(gan)


torch.save(gan, model_pkl)
writer.add_graph(gan, torch.zeros(1, noise_size).to(device))
writer.close()
