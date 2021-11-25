import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


model_pkl = 'gru.pkl'
parameters_pkl = 'gru-parameters.pkl'
optimizer_pkl = 'gru-optimizer.pkl'
num_layers = 2
input_size = 28
hidden_size = 64
epochs = 100
batch_size_train = 60
batch_size_test = 1000
learning_rate = 1e-3
log_interval_steps = 200
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


# 转换器，将PIL Image转换为Tensor，提供MNIST数据集单通道数据的平均值和标准差，将其转换为标准正态分布，并去除多余的维度
transform = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,)), torch.squeeze])
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


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        # 将batch维度放到最顶层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        outputs, hns = self.gru(x)
        # 取输出的向量outputs中的最后一个向量最为最终输出
        x = outputs[:, -1, :]
        x = self.fc(x)
        return x


# 可视化 tensorboard --logdir=runs-gru --bind_all
writer = SummaryWriter(logdir='runs-gru')
# 设备
# 执行前设置环境变量 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 filename.py
# 程序中会对可见GPU重新从0编号
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    device_count = torch.cuda.device_count()
    print('use {} gpu(s)'.format(device_count))
else:
    print('use cpu')
# 模型
model = Net(input_size, hidden_size, num_layers)
if os.path.isfile(parameters_pkl):
    model.load_state_dict(torch.load(parameters_pkl))
if device.type == 'cuda' and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile(optimizer_pkl):
    optimizer.load_state_dict(torch.load(optimizer_pkl))

def fit(model, optimizer, epochs, initial_epoch=1, baseline=True):
    global global_step

    # 设置model.training为True
    model.train()

    steps_per_epoch = len(train_loader)
    total_train_images = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        print('Train Epoch {}/{}'.format(epoch_index, initial_epoch + epochs - 1))
        print('-' * 20)

        epoch_loss_sum = 0
        epoch_correct_images = 0
        epoch_begin = time.time()

        for step_index, (images, labels) in enumerate(train_loader, 1):
            step_input_images = images.shape[0]

            step_begin = time.time()

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            step_loss = loss.item()
            epoch_loss_sum += step_loss * step_input_images
            step_correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
            epoch_correct_images += step_correct_images
            
            if step_index % log_interval_steps == 0:
                torch.save(model.state_dict(), 'lenet-parameters.pkl')
                torch.save(optimizer.state_dict(), 'lenet-optimizer.pkl')

                writer.add_scalar('train/loss', step_loss, global_step)
                writer.add_scalar('train/accuracy', step_correct_images / step_input_images, global_step)

                print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(step_index, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)

        print('-' * 20)
        print('Epoch {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index, initial_epoch + epochs - 1, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_images, epoch_correct_images, total_train_images, 1e2 * epoch_correct_images / total_train_images))
        print()

    if baseline:
        steps_total_test = len(test_loader)
        baseline_loss = epoch_loss_sum / total_train_images
        baseline_accuracy = epoch_correct_images / total_train_images
        for i in range(1, steps_total_test + 1):
            writer.add_scalars('test/loss', {'baseline': baseline_loss}, i)
            writer.add_scalars('test/accuracy', {'baseline': baseline_accuracy}, i)
            
global_step = 0
fit(model, optimizer, epochs)


def evaluate(model):
    global global_step

    # 设置model.training为False
    model.eval()

    steps_total = len(test_loader)
    total_loss_sum = 0
    total_correct_images = 0
    total_input_images = 0

    with torch.no_grad():
        print('Eval')
        print('-' * 20)

        test_begin = time.time()

        for step_index, (images, labels) in enumerate(test_loader, 1):
            step_input_images = images.shape[0]
            total_input_images += step_input_images

            step_begin = time.time()

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            step_loss = loss.item()
            total_loss_sum += step_loss * step_input_images
            step_correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
            total_correct_images += step_correct_images

            writer.add_scalars('test/loss', {'current': step_loss, 'average': total_loss_sum / total_input_images}, global_step)
            writer.add_scalars('test/accuracy', {'current': step_correct_images / step_input_images, 'average': total_correct_images / total_input_images}, global_step)

            print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(step_index, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        test_end = time.time()
        test_period = round((test_end - test_begin) * 1e3)

        print('-' * 20)
        print('Eval  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(int(test_period / 1e3), test_period % 1e3, total_loss_sum / total_input_images, total_correct_images, total_input_images, 1e2 * total_correct_images / total_input_images))
        print()

global_step = 0
evaluate(model)

torch.save(model, model_pkl)
writer.add_graph(model, torch.zeros(1, 28, 28).to(device))
writer.close()
