import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import os, time


epochs = 10
batch_size_train = 64
batch_size_val = 1000
learning_rate = 0.01
momentum = 0.5


# 转换器，将PIL Image转换为Tensor
transform = tv.transforms.Compose([tv.transforms.ToTensor()])
# 训练集（有标签），(100000, 2, 3, 64, 64)
train_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
# 验证集（有标签），(10000, 2, 3, 64, 64)
val_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
# 测试集（无标签），(10000, 2, 3, 64, 64)
test_set = tv.datasets.ImageFolder(root='./data/tiny-imagenet-200/test', transform=transform)
# 训练数据生成器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
# 测试数据生成器
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False)


def buildNet(string, num_classes=None, para_pth_path=None):
    ONE_SPACE = ' '
    IN_TAB_WIDTH = 2
    OUT_TAB_WIDTH = 4
    IN_TAB = ONE_SPACE * IN_TAB_WIDTH
    OUT_TAB = ONE_SPACE * OUT_TAB_WIDTH

    lines = string.strip().split('\n')
    size = len(lines)
    header = '''class {:s}(nn.Module):\n{:s}def __init__(self):\n{:s}super().__init__()\n'''.format(lines[0][:-1], OUT_TAB, 2 * OUT_TAB)
    sections = []
    sectionBegin = None
    for i in range(1, size):
        line = lines[i]
        if line.startswith(IN_TAB + '('):
            if sectionBegin is not None:
                sections.append((sectionBegin, sectionBegin))
            sectionBegin = i
        elif line.startswith(IN_TAB + ')'):
            sections.append((sectionBegin, i))
            sectionBegin = None

    import re
    pattern = re.compile('^\s*\(([^)]+)\): (.+)$')
    body = ''
    funcs = []
    for first, last in sections:
        matches = pattern.match(lines[first])
        groups = matches.groups()
        body += '{:s}self.{:s} = nn.{:s}\n'.format(2 * OUT_TAB, *groups)
        funcs.append('self.{:s}'.format(groups[0]))
        if first != last:
            for i in range(first + 1, last):
                matches = pattern.match(lines[i])
                groups = matches.groups()
                if num_classes is not None and i == size - 3:
                    begin = groups[1].find('out_features=') + len('out_features=')
                    end = groups[1].find(',', begin)
                    body += '{:s}nn.{:s}{:d}{:s}{:s}\n'.format(3 * OUT_TAB, groups[1][:begin], num_classes, groups[1][end:], '' if i == last - 1 else ',')
                else:
                    body += '{:s}nn.{:s}{:s}\n'.format(3 * OUT_TAB, groups[1], '' if i == last - 1 else ',')
            body += '{:s}{:s}\n'.format(2 * OUT_TAB, lines[last].strip())
    if para_pth_path is not None:
        body += '{:s}self.load_state_dict(torch.load("{:s}"))\n'.format(2 * OUT_TAB, para_pth_path)
    funcs.insert(-1, 'nn.Flatten()')
    footer = '''{:s}def forward(self, x):\n{:s}{:s}return x\n'''.format(OUT_TAB, ''.join(['{:s}x = {:s}(x)\n'.format(2 * OUT_TAB, func) for func in funcs]), 2 * OUT_TAB)
    return header + body + footer

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # (B, 3, W, H)
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            # (B, 64, W1=math.ceil((W - 6) / 4), H1=math.ceil((H - 6) / 4))
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            # (B, 64, W2=math.ceil(W1 / 2 - 1), H2=math.ceil(H1 / 2 - 1))
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            # (B, 192, W2, H2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            # (B, 192, W3=math.ceil(W2 / 2 - 1), H3=math.ceil(H2 / 2 - 1))
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # (B, 384, W3, H3)
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # (B, 256, W3, H3)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # (B, 256, W3, H3)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
            # (B, 256, W4=math.ceil(W3 / 2 - 1), H4=math.ceil(H3 / 2 - 1))
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(6, 6))
        # (B, 256, 6, 6)
        self.classifier = nn.Sequential(
            # (B, 9216)
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            # (B, 4096)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            # (B, 4096)
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
            # (B, 1000)
        )
        self.load_state_dict(torch.load("/home/sunjian/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x


# 可视化 tensorboard --logdir=runs --bind_all
writer = SummaryWriter(logdir='runs-alexnet')
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
model = AlexNet()
writer.add_graph(model, (torch.zeros(1, 3, 64, 64), ))
if cuda_available and device_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(device_count)), output_device=0)
model = model.to(device)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)


def fit(model, optimizer, epochs, initial_epoch=1):
    global global_step

    steps_per_epoch = len(train_loader)
    total_train_images = len(train_set)

    for epoch_index in range(initial_epoch, initial_epoch + epochs):
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
            step_period = round((step_end - step_begin) * 1e3)

            global_step += 1
            step_loss = loss.item()
            epoch_loss_sum += step_loss * step_input_images
            step_correct_images = (torch.argmax(outputs, -1) == labels).sum().item()
            epoch_correct_images += step_correct_images
            writer.add_scalar('train/loss', step_loss, global_step)
            writer.add_scalar('train/accuracy', step_correct_images / step_input_images, global_step)

            print('Epoch {}/{}  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index, initial_epoch + epochs - 1, batch_index + 1, steps_per_epoch, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        epoch_end = time.time()
        epoch_period = round((epoch_end - epoch_begin) * 1e3)
        print('[Epoch {}/{}]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(epoch_index, initial_epoch + epochs - 1, int(epoch_period / 1e3), epoch_period % 1e3, epoch_loss_sum / total_train_images, epoch_correct_images, total_train_images, 1e2 * epoch_correct_images / total_train_images))


global_step = 0
fit(model, optimizer, epochs, 1)


def evaluate(model):
    global global_step

    steps_total = len(val_loader)
    total_loss_sum = 0
    total_correct_images = 0
    total_input_images = 0

    with torch.no_grad():
        val_begin = time.time()

        for batch_index, (images, labels) in enumerate(val_loader):
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

            print('Validate  Step {}/{}  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(batch_index + 1, steps_total, int(step_period / 1e3), step_period % 1e3, step_loss, step_correct_images, step_input_images, 1e2 * step_correct_images / step_input_images))

        val_end = time.time()
        val_period = round((val_end - val_begin) * 1e3)
        print('[Validate]  Time: {:.0f}s {:.0f}ms  Loss: {:.4f}  Accuracy: {}/{} ({:.1f}%)'.format(int(val_period / 1e3), val_period % 1e3, total_loss_sum / total_input_images, total_correct_images, total_input_images, 1e2 * total_correct_images / total_input_images))


global_step = 0
evaluate(model)

writer.close()