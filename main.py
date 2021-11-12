import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
dropout_rate = 0.5


# 转换器，将PIL Image转换为Tensor
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
# 训练集，(60000, 2, 1, 28, 28)
train_set = tv.datasets.MNIST(root='./data', train=True, download=not os.path.isfile('./data/MNIST/processed/training.pt'), transform=transform)
# 测试集，(10000, 2, 1, 28, 28)
test_set = tv.datasets.MNIST(root='./data', train=False, download=not os.path.isfile('./data/MNIST/processed/test.pt'), transform=transform)
# 训练数据生成器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), padding_mode='zeros')
        self.fc1 = nn.Linear(in_features=20*7*7, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self.bn = nn.BatchNorm2d(20)
        self._init_parameters()

    def forward(self, x):
        # (64, 1, 28, 28)
        x = self.conv1(x)
        # (64, 10, 28, 28)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # (64, 10, 14, 14)
        x = self.conv2(x)
        # (64, 20, 14, 14)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # (64, 20, 7, 7)
        x = self.bn(x)
        x = nn.Flatten()(x)
        # (64, 980)
        x = self.fc1(x)
        # (64, 50)
        x = F.relu(x)
        x = F.dropout(x, p=dropout_rate, training=self.training)
        x = self.fc2(x)
        # (64, 10)
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
writer = SummaryWriter()
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
model = Net()
writer.add_graph(model, (torch.zeros(1, 1, 28, 28), ))
if cuda_available and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


global_step = 0
steps_per_epoch = len(train_loader)
total_train_images = len(train_set)

for epoch_index in range(epochs):

    epoch_loss_sum = 0
    epoch_correct_images = 0
    epoch_begin = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):
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
        step_period = step_end - step_begin

        global_step += 1
        step_loss = loss.item()
        step_correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
        epoch_correct_images += step_correct_images
        epoch_loss_sum += step_loss
        writer.add_scalar('train/loss', step_loss, global_step)
        writer.add_scalar('train/accuracy', step_correct_images / step_input_images, global_step)

        print('Epoch {}/{}  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index + 1, 2 * epochs, batch_index + 1, steps_per_epoch, int(step_period), int(step_period * 1e3) % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

    epoch_end = time.time()
    epoch_period = epoch_end - epoch_begin
    print('[Epoch {}/{}]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index + 1, 2 * epochs, int(epoch_period), int(epoch_period * 1e3) % 1e3, epoch_loss_sum / steps_per_epoch, epoch_correct_images, total_train_images, 1e2 * epoch_correct_images / total_train_images))


torch.save(model.state_dict(), 'parameters.pkl')
torch.save(optimizer.state_dict(), 'optimizer.pkl')

model2 = Net()
if cuda_available and device_count > 1:
    model2 = nn.DataParallel(model2, device_ids=list(range(device_count)), output_device=0)
model2.load_state_dict(torch.load('parameters.pkl'))
model2 = model2.to(device)
optimizer2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer2.load_state_dict(torch.load('optimizer.pkl'))


for epoch_index in range(epochs, 2 * epochs):
    epoch_begin = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):
        input_images = images.shape[0]
    
        step_begin = time.time()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        step_end = time.time()
        step_period = step_end - step_begin

        global_step += 1
        correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
        writer.add_scalar('train/loss', loss.item(), global_step)
        writer.add_scalar('train/accuracy', correct_images / input_images, global_step)

        print('Epoch {}/{}  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index + 1, 2 * epochs, batch_index + 1, steps_per_epoch, int(step_period), int(step_period * 1e3) % 1e3, loss.item(), correct_images, input_images, 1e2 * correct_images / input_images))

    epoch_end = time.time()
    epoch_period = epoch_end - epoch_begin
    print('[Epoch {}/{}]  Time: {:.0f}s {:.0f}ms'.format(epoch_index + 1, 2 * epochs, int(epoch_period), int(epoch_period * 1e3) % 1e3))


torch.save(model2, 'model.pkl')

model3 = torch.load('model.pkl')
if cuda_available and device_count > 1:
    model3 = nn.DataParallel(model3, device_ids=list(range(device_count)), output_device=0)
model3 = model3.to(device)


global_step = 0
steps_total = len(test_loader)
step_correct_images = 0
total_correct_images = 0
total_input_images = len(test_set)

with torch.no_grad():
    test_begin = time.time()
    for batch_index, (images, labels) in enumerate(test_loader):
        step_input_images = images.shape[0]
        
        step_begin = time.time()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model3(images)
        loss = criterion(outputs, labels)

        step_end = time.time()
        step_period = step_end - step_begin

        global_step += 1
        step_correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
        total_correct_images += step_correct_images
        writer.add_scalar('test/loss', loss.item(), global_step)
        writer.add_scalar('test/accuracy', step_correct_images / step_input_images, global_step)

        print('Test Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(batch_index + 1, steps_total, int(step_period), int(step_period * 1e3) % 1e3, loss.item(), step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))
    test_end = time.time()
    test_period = test_end - test_begin
print('[Test]  Time: {:.0f}s {:.0f}ms  Accuracy: {}/{} ({:.1f}%)'.format(int(test_period), int(test_period * 1e3) % 1e3, total_correct_images, total_input_images, 1e2 * total_correct_images / total_input_images))

writer.close()
