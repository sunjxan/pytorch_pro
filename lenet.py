import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


epochs = 100
batch_size_train = 60
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval_steps = 200


# 转换器，将PIL Image转换为Tensor
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
if not os.path.isdir('./data/MNIST'):
    # 训练集，(60000, 2, 1, 28, 28)
    train_set = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 测试集，(10000, 2, 1, 28, 28)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    train_set = tv.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
# 训练数据生成器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)


class LeNet(nn.Module):
    def __init__(self):
        # 为提高精度，将AvgPool2d改为MaxPool2d，Sigmoid改为ReLU
        super().__init__()
        # 卷积/池化后大小为 math.ceil((W - kernel_size + 1 + 2 * padding) / stride)
        self.features = nn.Sequential(
            # (B, 1, 28, 28)
            # 因为MNIST图片大小为28*28，不是32*32，所以修改padding为2
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            # (B, 6, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False),
            # (B, 6, 14, 14)
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            # (B, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
            # (B, 16, 5, 5)
        )
        self.classifier = nn.Sequential(
            # (B, 400)
            nn.Linear(in_features=400, out_features=120, bias=True),
            # (B, 120)
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84, bias=True),
            # (B, 84)
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10, bias=True)
            # (B, 10)
        )
        self._init_parameters()
    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x
    def _init_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.zeros_(layer.bias)


# 可视化 tensorboard --logdir=runs --bind_all
writer = SummaryWriter(logdir='runs-lenet')
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
model = LeNet()
writer.add_graph(model, (torch.zeros(1, 1, 28, 28), ))
if cuda_available and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def fit(model, optimizer, epochs, initial_epoch=0):
    global global_step

    steps_per_epoch = len(train_loader)
    total_train_images = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        epoch_loss_sum = 0
        epoch_correct_images = 0
        epoch_begin = time.time()

        for step_index, (images, labels) in enumerate(train_loader):
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
            
            if step_index and step_index % log_interval_steps == 0:
                torch.save(model.state_dict(), 'lenet-parameters.pkl')
                torch.save(optimizer.state_dict(), 'lenet-optimizer.pkl')

                writer.add_scalar('train/loss', step_loss, global_step)
                writer.add_scalar('train/accuracy', step_correct_images / step_input_images, global_step)

                print('Epoch {}/{}  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index + 1, initial_epoch + epochs, step_index + 1, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)

        print('[Epoch {}/{}]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index + 1, initial_epoch + epochs, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_images, epoch_correct_images, total_train_images, 1e2 * epoch_correct_images / total_train_images))

global_step = 0
fit(model, optimizer, epochs, 0)


def evaluate(model):
    global global_step

    steps_total = len(test_loader)
    total_loss_sum = 0
    total_correct_images = 0
    total_input_images = 0

    with torch.no_grad():
        test_begin = time.time()

        for step_index, (images, labels) in enumerate(test_loader):
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

            print('Evaluate  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(step_index + 1, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        test_end = time.time()
        test_period = round((test_end - test_begin) * 1e3)

        print('[Evaluate]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(int(test_period / 1e3), test_period % 1e3, total_loss_sum / total_input_images, total_correct_images, total_input_images, 1e2 * total_correct_images / total_input_images))

global_step = 0
evaluate(model)

writer.close()
