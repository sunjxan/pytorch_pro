import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from torchvision.ops import StochasticDepth
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation

import os, time, math
from functools import partial
from collections import OrderedDict


model_pkl = 'regnet.pkl'
parameters_pkl = 'regnet-parameters.pkl'
optimizer_pkl = 'regnet-optimizer.pkl'
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

class SimpleStemIN(ConvNormActivation):
    def __init__(self, width_in, width_out, norm_layer, activation_layer):
        super().__init__(width_in, width_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=activation_layer)

class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, width_in, width_out, stride, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio):
        layers = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer)
        layers["b"] = ConvNormActivation(w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer)

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(w_b, width_out, kernel_size=1, stride=1,
                                         norm_layer=norm_layer, activation_layer=None)
        super().__init__(layers)

class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in, width_out, stride, norm_layer, activation_layer, group_width=1, bottleneck_multiplier=1.0, se_ratio=None):
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)
        self.f = BottleneckTransform(width_in, width_out, stride, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation_layer(inplace=True)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    def __init__(self, width_in, width_out, stride, depth, block_constructor, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio=None, stage_index=0):
        super().__init__()

        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(self, depths, widths, group_widths, bottleneck_multipliers, strides, se_ratio=None):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(cls, depth, w_0, w_a, w_m, group_width, bottleneck_multiplier=1.0, se_ratio=None):
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(stage_widths, bottleneck_multipliers, group_widths)

        return cls(depths=stage_depths, widths=stage_widths, group_widths=group_widths, bottleneck_multipliers=bottleneck_multipliers, strides=strides, se_ratio=se_ratio)

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths, bottleneck_ratios, group_widths):
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(self, block_params, num_classes=1000, stem_width=32, stem_type=None, block_type=None, norm_layer=None, activation=None, init_weights=True):
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = stem_type(3, stem_width, norm_layer, activation)

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

# PyTorch版本不同预训练权重地址可能不同
model_urls = {
    "regnet_y_400mf": "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth",
    "regnet_y_800mf": "https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth",
    "regnet_y_1_6gf": "https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth",
    "regnet_y_3_2gf": "https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth",
    "regnet_y_8gf": "https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth",
    "regnet_y_16gf": "https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth",
    "regnet_y_32gf": "https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth",
    "regnet_x_400mf": "https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth",
    "regnet_x_800mf": "https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth",
    "regnet_x_1_6gf": "https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth",
    "regnet_x_3_2gf": "https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth",
    "regnet_x_8gf": "https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth",
    "regnet_x_16gf": "https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth",
    "regnet_x_32gf": "https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth",
}

# regnet_y_400mf = RegNet(BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_800mf = RegNet(BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_1_6gf = RegNet(BlockParams.from_init_params(depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_3_2gf = RegNet(BlockParams.from_init_params(depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_8gf = RegNet(BlockParams.from_init_params(depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_16gf = RegNet(BlockParams.from_init_params(depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_y_32gf = RegNet(BlockParams.from_init_params(depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_400mf = RegNet(BlockParams.from_init_params(depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_800mf = RegNet(BlockParams.from_init_params(depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_1_6gf = RegNet(BlockParams.from_init_params(depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_3_2gf = RegNet(BlockParams.from_init_params(depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_8gf = RegNet(BlockParams.from_init_params(depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_16gf = RegNet(BlockParams.from_init_params(depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
# regnet_x_32gf = RegNet(BlockParams.from_init_params(depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))


# 可视化 tensorboard --logdir=runs-regnet --bind_all
writer = SummaryWriter(logdir='runs-regnet')
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
    model = RegNet(BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1), init_weights=False)
    model.load_state_dict(torch.load(parameters_pkl))
else:
    model = RegNet(BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25), norm_layer=partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
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
writer.add_graph(model, torch.zeros(1, 3, 64, 64).to(device))
writer.close()
