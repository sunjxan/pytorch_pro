import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, random, time


model_pkl = 'main.pkl'
parameters_pkl = 'main-parameters.pkl'
optimizer_pkl = 'main-optimizer.pkl'
data_set_size = 10000
train_ratio = .7
epochs = 100
batch_size_train = 70
batch_size_test = 1000
learning_rate = 1e-3
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


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
    def __init__(self, init_weights=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(5)
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 3)
        self.dropout = nn.Dropout(p=0.1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # (B, 5)
        x = self.bn(x)
        x = self.fc1(x)
        # (B, 10)
        x = self.dropout(x)
        x = self.fc2(x)
        # (B, 3)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 可视化 tensorboard --logdir=runs --bind_all
writer = SummaryWriter()
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
if os.path.isfile(parameters_pkl):
    model = Net(init_weights=False)
    model.load_state_dict(torch.load(parameters_pkl))
else:
    model = Net()
if device.type == 'cuda' and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.MSELoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile(optimizer_pkl):
    optimizer.load_state_dict(torch.load(optimizer_pkl))


def fit(model, optimizer, epochs, initial_epoch=1, baseline=True):
    global global_step

    # 设置model.training为True
    model.train()

    steps_per_epoch = len(train_loader)
    total_train_samples = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
        print('Train Epoch {}/{}'.format(epoch_index, initial_epoch + epochs - 1))
        print('-' * 20)

        epoch_loss_sum = 0
        epoch_begin = time.time()

        for step_index, (samples, labels) in enumerate(train_loader, 1):
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

            torch.save(model.state_dict(), parameters_pkl)
            torch.save(optimizer.state_dict(), optimizer_pkl)
            writer.add_scalar('train/loss', step_loss, global_step)

            print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(step_index, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)

        print('-' * 20)
        print('Train Epoch {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(epoch_index, initial_epoch + epochs - 1, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_samples))
        print()

    if baseline:
        steps_total_test = len(test_loader)
        baseline_value = epoch_loss_sum / total_train_samples
        for i in range(1, steps_total_test + 1):
            writer.add_scalars('test/loss', {'baseline': baseline_value}, i)

global_step = 0
fit(model, optimizer, epochs)


def evaluate(model):
    global global_step

    # 设置model.training为False
    model.eval()

    steps_total = len(test_loader)
    total_loss_sum = 0
    total_input_samples = 0

    with torch.no_grad():
        print('Eval')
        print('-' * 20)

        test_begin = time.time()

        for step_index, (samples, labels) in enumerate(test_loader, 1):
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

            print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(step_index, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss))
        
        test_end = time.time()
        test_period = round((test_end - test_begin) * 1e3)

        print('-' * 20)
        print('Eval  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}'.format(int(test_period / 1e3), test_period % 1e3, total_loss_sum / total_input_samples))
        print()

global_step = 0
evaluate(model)


torch.save(model, model_pkl)
writer.add_graph(model, torch.zeros(1, 5).to(device))
writer.close()
