import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import random, time


data_set_size = 10000
train_ratio = .7
epochs = 100
batch_size_train = 70
batch_size_test = 1000
learning_rate = 0.01


# x = 1a + 2b + 3c + 4d + 5e
# y = 2a + 3b + 4c + 5d + 6e
# z = 3a + 4b + 5c + 6d + 7e
class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_set_size):
        self._dataset = []
        random.seed(1)
        for i in range(data_set_size):
            a = random.random()
            b = random.random()
            c = random.random()
            d = random.random()
            e = random.random()
            x = 1 * a + 2 * b + 3 * c + 4 * d + 5 * e + random.normalvariate(0, 1)
            y = 2 * a + 3 * b + 4 * c + 5 * d + 6 * e + random.normalvariate(0, 1)
            z = 3 * a + 4 * b + 5 * c + 6 * d + 7 * e + random.normalvariate(0, 1)
            sample = (torch.tensor([a, b, c, d, e]), torch.tensor([x, y, z]))
            self._dataset.append(sample)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        if 0 <= index < len(self._dataset):
            return self._dataset[index]
        return None

data_set = DataSet(data_set_size)
train_set_size = round(data_set_size * train_ratio)
test_set_size = data_set_size - train_set_size
train_set, test_set = torch.utils.data.random_split(data_set, [train_set_size, test_set_size])
# 训练数据生成器，先随机打乱，然后按batch_size分批，样本不重不漏，最后一个batch样本数量可能不足batch_size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(in_features=5, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=3)
        self._init_parameters()

    def forward(self, x):
        # (B, 5)
        x = self.bn(x)
        x = self.fc1(x)
        # (B, 10)
        x = torch.dropout(x, p=0.1, train=self.training)
        x = self.fc2(x)
        # (B, 3)
        return x

    def _init_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
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
writer.add_graph(model, torch.zeros(1, 5))
if cuda_available and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.MSELoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def fit(model, optimizer, epochs, initial_epoch=0, baseline=True):
    global global_step
    
    steps_per_epoch = len(train_loader)
    total_train_samples = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        epoch_loss_sum = 0
        epoch_begin = time.time()

        for step_index, (samples, labels) in enumerate(train_loader):
            step_input_samples = samples.shape[0]
            
            step_begin = time.time()

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            step_loss = loss.item()
            epoch_loss_sum += step_loss * step_input_samples
            writer.add_scalar('train/loss', step_loss, global_step)

            print('Epoch {}/{}  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(epoch_index + 1, initial_epoch + epochs, step_index + 1, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)
        print('[Epoch {}/{}]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(epoch_index + 1, initial_epoch + epochs, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_samples))

    if baseline:
        steps_total_test = len(test_loader)
        baseline_value = epoch_loss_sum / total_train_samples
        for i in range(1, steps_total_test + 1):
            writer.add_scalars('test/loss', {'baseline': baseline_value}, i)

global_step = 0
fit(model, optimizer, epochs, 0, False)


torch.save(model.state_dict(), 'parameters.pkl')
torch.save(optimizer.state_dict(), 'optimizer.pkl')

model2 = Net()
if cuda_available and device_count > 1:
    model2 = nn.DataParallel(model2, device_ids=list(range(device_count)), output_device=0)
model2.load_state_dict(torch.load('parameters.pkl'))
model2 = model2.to(device)
optimizer2 = optim.SGD(model.parameters(), lr=learning_rate)
optimizer2.load_state_dict(torch.load('optimizer.pkl'))

fit(model, optimizer2, epochs, epochs)


torch.save(model2, 'model.pkl')

model3 = torch.load('model.pkl')
if cuda_available and device_count > 1:
    model3 = nn.DataParallel(model3, device_ids=list(range(device_count)), output_device=0)
model3 = model3.to(device)


def evaluate(model):
    global global_step
    
    steps_total = len(test_loader)
    total_loss_sum = 0
    total_input_samples = 0

    with torch.no_grad():
        test_begin = time.time()

        for step_index, (samples, labels) in enumerate(test_loader):
            step_input_samples = samples.shape[0]
            total_input_samples += step_input_samples
            
            step_begin = time.time()

            samples = samples.to(device)
            labels = labels.to(device)

            outputs = model(samples)
            loss = criterion(outputs, labels)

            step_end = time.time()
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            step_loss = loss.item()
            total_loss_sum += step_loss * step_input_samples
            writer.add_scalars('test/loss', {'current': step_loss, 'average': total_loss_sum / total_input_samples}, global_step)

            print('Test  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(step_index + 1, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss))
        
        test_end = time.time()
        test_period = round((test_end - test_begin) * 1e3)
        print('[Test]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(int(test_period / 1e3), test_period % 1e3, total_loss_sum / total_input_samples))

global_step = 0
evaluate(model3)

writer.close()
