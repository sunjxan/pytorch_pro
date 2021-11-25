import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


model_pkl = 'resnet.pkl'
parameters_pkl = 'resnet-parameters.pkl'
optimizer_pkl = 'resnet-optimizer.pkl'
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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, groups=1, base_width=64, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d, groups=1, base_width=64, dilation=1):
        super().__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, init_weights=True, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()
        if zero_init_residual:
            self._zero_initialize_residual()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _zero_initialize_residual(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self._norm_layer, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self._norm_layer, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
# resnet34 = ResNet(BasicBlock, [3, 4, 6, 3])
# resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
# resnet101 = ResNet(Bottleneck, [3, 4, 23, 3])
# resnet152 = ResNet(Bottleneck, [3, 8, 36, 3])
# resnext50_32x4d = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)
# resnext101_32x8d = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)
# wide_resnet50_2 = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64*2)
# wide_resnet101_2 = ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64*2)


# 可视化 tensorboard --logdir=runs-resnet --bind_all
writer = SummaryWriter(logdir='runs-resnet')
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
    model = ResNet(Bottleneck, [3, 4, 6, 3], init_weights=False)
    model.load_state_dict(torch.load(parameters_pkl))
else:
    model = ResNet(Bottleneck, [3, 4, 6, 3])
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
                torch.save(model.state_dict(), 'resnet-parameters.pkl')
                torch.save(optimizer.state_dict(), 'resnet-optimizer.pkl')

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
writer.add_graph(model, torch.zeros(1, 3, 64, 64).to(device))
writer.close()
