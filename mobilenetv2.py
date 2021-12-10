import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


model_pkl = 'mobilenet_v2.pkl'
parameters_pkl = 'mobilenet_v2-parameters.pkl'
optimizer_pkl = 'mobilenet_v2-optimizer.pkl'
epochs = 200
batch_size_train = 100
batch_size_val = 1000
learning_rate = 1e-3
log_interval_steps = 250
# 对于可重复的实验，设置随机种子
torch.manual_seed(seed=1)


# 转换器，将PIL Image转换为Tensor
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
# 训练集（有标签），(100000, 2, 3, 64, 64)
train_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
# 验证集（有标签），(10000, 2, 3, 64, 64)
val_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
# 测试集（无标签），(10000, 2, 3, 64, 64)
test_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/test', transform=transform)
# 训练数据生成器，先随机打乱，然后按batch_size分批，样本不重不漏，最后一个batch样本数量可能不足batch_size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()

        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, block=None, init_weights=True):
        super().__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# PyTorch版本不同预训练权重地址可能不同
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


# 可视化 tensorboard --logdir=runs-mobilenet_v2 --bind_all
writer = SummaryWriter(logdir='runs-mobilenet_v2')
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
    model = MobileNetV2(init_weights=False)
    model.load_state_dict(torch.load(parameters_pkl))
    print('parameters loaded')
else:
    model = MobileNetV2()
if device.type == 'cuda' and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
if os.path.isfile(optimizer_pkl):
    optimizer.load_state_dict(torch.load(optimizer_pkl))
    print('optimizer loaded')


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
                torch.save(model.state_dict(), parameters_pkl)
                torch.save(optimizer.state_dict(), optimizer_pkl)

                writer.add_scalar('train/loss', step_loss, global_step)
                writer.add_scalar('train/accuracy', step_correct_images / step_input_images, global_step)

                print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(step_index, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)

        print('-' * 20)
        print('Train Epoch {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index, initial_epoch + epochs - 1, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_images, epoch_correct_images, total_train_images, 1e2 * epoch_correct_images / total_train_images))
        print()

    if baseline:
        steps_total_val = len(val_loader)
        baseline_loss = epoch_loss_sum / total_train_images
        baseline_accuracy = epoch_correct_images / total_train_images
        for i in range(1, steps_total_val + 1):
            writer.add_scalars('test/loss', {'baseline': baseline_loss}, i)
            writer.add_scalars('test/accuracy', {'baseline': baseline_accuracy}, i)

global_step = 0
fit(model, optimizer, epochs)


def evaluate(model):
    global global_step

    # 设置model.training为False
    model.eval()

    steps_total = len(val_loader)
    total_loss_sum = 0
    total_correct_images = 0
    total_input_images = 0

    with torch.no_grad():
        print('Eval')
        print('-' * 20)

        val_begin = time.time()

        for step_index, (images, labels) in enumerate(val_loader, 1):
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

            writer.add_scalars('validate/loss', {'current': step_loss, 'average': total_loss_sum / total_input_images}, global_step)
            writer.add_scalars('validate/accuracy', {'current': step_correct_images / step_input_images, 'average': total_correct_images / total_input_images}, global_step)

            print('Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(step_index, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        val_end = time.time()
        val_period = round((val_end - val_begin) * 1e3)

        print('-' * 20)
        print('Eval  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(int(val_period / 1e3), val_period % 1e3, total_loss_sum / total_input_images, total_correct_images, total_input_images, 1e2 * total_correct_images / total_input_images))
        print()

global_step = 0
evaluate(model)


torch.save(model, model_pkl)
print('model saved')
writer.add_graph(model, torch.zeros(1, 3, 64, 64).to(device))
writer.close()
