import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time, math


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


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, tanh=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.tanh = tanh
        self.fc_ih = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self._init_parameters()

    def forward(self, input, hx=None):
        output = self.fc_ih(input)
        if hx is not None:
            output += self.fc_hh(hx)
        output = torch.tanh(output) if self.tanh else torch.relu(output)
        return output

    def _init_parameters(self):
        k = math.sqrt(1 / self.hidden_size)
        nn.init.uniform_(self.fc_ih.weight, -k, k)
        nn.init.uniform_(self.fc_hh.weight, -k, k)
        if self.bias:
            nn.init.uniform_(self.fc_ih.bias, -k, k)
            nn.init.uniform_(self.fc_hh.bias, -k, k)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, tanh=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                self.layers.append(RNNCell(input_size, hidden_size, bias, tanh))
            else:
                self.layers.append(RNNCell(hidden_size, hidden_size, bias, tanh))

    def __call__(self, inputs, hxs=None):
        outputs = []
        for input in inputs:
            out_hxs = []
            for i in range(self.num_layers):
                hx = self.layers[i](input, None if hxs is None else hxs[i])
                input = hx
                out_hxs.append(hx)
            hxs = torch.stack(out_hxs)
            outputs.append(out_hxs[-1])
        return torch.stack(outputs), hxs


# 可视化 tensorboard --logdir=runs --bind_all
writer = SummaryWriter(logdir='runs-rnn')
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
model = RNN()
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
                torch.save(model.state_dict(), 'rnn-parameters.pkl')
                torch.save(optimizer.state_dict(), 'rnn-optimizer.pkl')

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
