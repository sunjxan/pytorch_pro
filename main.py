import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os, time


epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
dropout_rate = 0.5


# 转换器，将PIL Image转换为Tensor
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
if os.path.isdir('./data/MNIST'):
    # 训练集，(60000, 2, 1, 28, 28)
    train_set = tv.datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    # 测试集，(10000, 2, 1, 28, 28)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
else:
    # 训练集，(60000, 2, 1, 28, 28)
    train_set = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 测试集，(10000, 2, 1, 28, 28)
    test_set = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 训练数据生成器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), padding_mode='zeros')
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=20*7*7, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
        self._init_parameters()

    def forward(self, x):
        # (64, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))
        # (64, 10, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))
        # (64, 20, 7, 7)
        x = nn.Flatten()(x)
        # (64, 980)
        x = F.relu(self.fc1(x))
        # (64, 50)
        x = F.dropout(x, p=dropout_rate, training=self.training)
        # (64, 50)
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


# 设备
# 执行前设置环境变量 export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
cuda_available = torch.cuda.is_available()
if cuda_available:
    device_count = torch.cuda.device_count()
    print('use {} gpu(s)'.format(device_count))
else:
    print('use cpu')
device = torch.device("cuda:0" if cuda_available else "cpu")
# 模型
model = Net()
if cuda_available and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


steps_per_epoch = len(train_loader)
train_images = 0
train_counter = []
train_losses = []

for epoch_index in range(epochs):
    epoch_begin = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):
        step_begin = time.time()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step_end = time.time()

        train_images += batch_size_train
        train_counter.append(train_images)
        train_losses.append(loss.item())

        print('Epoch {}/{}  Step {}/{}  Time: {:.0f}ms  Loss: {:.6f}'.format(epoch_index + 1, 2 * epochs, batch_index + 1, steps_per_epoch, (step_end - step_begin) * 1e3, loss.item()))

    epoch_end = time.time()
    print('Epoch {}/{}  Time: {:.1f}s  Avg Loss: {:.6f}'.format(epoch_index + 1, 2 * epochs, epoch_end - epoch_begin, sum(train_losses[-steps_per_epoch:]) / steps_per_epoch))

    
torch.save(model.state_dict(), 'parameters.pt')
torch.save(optimizer.state_dict(), 'optimizer.pt')

model2 = Net()
if cuda_available and device_count > 1:
    model2 = nn.DataParallel(model2, device_ids=list(range(len(device_ids))), output_device=0)
model2.load_state_dict(torch.load('parameters.pt'))
model2 = model2.to(device)
optimizer2 = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer2.load_state_dict(torch.load('optimizer.pt'))


for epoch_index in range(epochs, 2 * epochs):
    epoch_begin = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):
        step_begin = time.time()

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()

        step_end = time.time()

        train_images += batch_size_train
        train_counter.append(train_images)
        train_losses.append(loss.item())

        print('Epoch {}/{}  Step {}/{}  Time: {:.0f}ms  Loss: {:.6f}'.format(epoch_index + 1, 2 * epochs, batch_index + 1, steps_per_epoch, (step_end - step_begin) * 1e3, loss.item()))

    epoch_end = time.time()
    print('Epoch {}/{}  Time: {:.1f}s  Avg Loss: {:.6f}'.format(epoch_index + 1, 2 * epochs, epoch_end - epoch_begin, sum(train_losses[-steps_per_epoch:]) / steps_per_epoch))


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.legend(['Train Loss'], loc='upper right')
plt.xlabel('images count')
plt.ylabel('loss')
plt.savefig('losses.png')


torch.save(model2, 'model.pt')

model3 = torch.load('model.pt')
if cuda_available and device_count > 1:
    model3 = nn.DataParallel(model3, device_ids=list(range(len(device_ids))), output_device=0)
model3 = model3.to(device)


steps_test = len(test_loader)
test_images = 0
correct_images = 0
test_losses_sum = 0
with torch.no_grad():
    for batch_index, (images, labels) in enumerate(test_loader):
        step_begin = time.time()

        images = images.to(device)
        labels = labels.to(device)

        outputs = model3(images)
        loss = criterion(outputs, labels)

        step_end = time.time()

        correct_images += (torch.argmax(outputs, -1) == labels).sum().item()
        test_images += batch_size_test
        test_losses_sum += loss.item()

        print('Step {}/{}  Time: {:.0f}ms  Avg Loss: {:.6f}'.format(batch_index + 1, steps_test, (step_end - step_begin) * 1e3, test_losses_sum / steps_test))
print('Test  Avg Loss: {:.6f}  Accuracy: {}/{} ({:.1f}%)'.format(test_losses_sum / steps_test, correct_images, test_images, 1e2 * correct_images / test_images))
