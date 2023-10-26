import math
import torch
import torch.nn as nn
import torch.nn.functional as func


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(func.relu(self.bn1(x)))
        y = self.conv2(func.relu(self.bn2(y)))
        x = torch.cat([y, x], 1)
        return x

    def get_sequential_output(self, x, lens):
        output_list = []
        y = func.relu(self.bn1(x))
        output_list.append(y[:, :lens])
        y = func.relu(self.bn2(self.conv1(y)))
        output_list.append(y)
        y = self.conv2(y)
        x = torch.cat([y, x], 1)
        lens = y.size()[1]
        return x, output_list, lens


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(func.relu(self.bn(x)))
        x = func.avg_pool2d(x, 2)
        return x

    def get_sequential_output(self, x, lens):
        output_list = []
        x = func.relu(self.bn(x))
        output_list.append(x[:, :lens])
        x = self.conv(x)
        x = func.avg_pool2d(x, 2)
        lens = x.size()[1]
        return x, output_list, lens


class DenseNet(nn.Module):
    def __init__(self, num_block, block=Bottleneck, growth_rate=12, reduction=0.5, num_classes=10, n_channels=3):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.num_classes = num_classes
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(n_channels, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.trans3(self.dense3(x))
        x = self.dense4(x)
        x = func.avg_pool2d(func.relu(self.bn(x)), 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_sequential_output(self, x):
        output_list = []
        x = self.conv1(x)
        lens = x.size()[1]
        for i in range(len(self.dense1)):
            x, intermediate_list, lens = self.dense1[i].get_sequential_output(x, lens)
            output_list.extend(intermediate_list)
        x, intermediate_list, lens = self.trans1.get_sequential_output(x, lens)
        output_list.extend(intermediate_list)
        for i in range(len(self.dense2)):
            x, intermediate_list, lens = self.dense2[i].get_sequential_output(x, lens)
            output_list.extend(intermediate_list)
        x, intermediate_list, lens = self.trans2.get_sequential_output(x, lens)
        output_list.extend(intermediate_list)
        for i in range(len(self.dense3)):
            x, intermediate_list, lens = self.dense3[i].get_sequential_output(x, lens)
            output_list.extend(intermediate_list)
        x, intermediate_list, lens = self.trans3.get_sequential_output(x, lens)
        output_list.extend(intermediate_list)
        for i in range(len(self.dense4)):
            x, intermediate_list, lens = self.dense4[i].get_sequential_output(x, lens)
            output_list.extend(intermediate_list)
        x = func.relu(self.bn(x))
        output_list.append(x[:, :lens])
        x = func.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        output_list.append(x)
        return output_list

    def set_optimizer(self, cfg, args=None):
        if args is not None and not args.finetune_momentum:
            self.optim = torch.optim.SGD(self.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        else:
            self.optim = torch.optim.SGD(self.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM,
                                    weight_decay=cfg.WD)
        if not cfg.CONST_LR and not hasattr(cfg, "WARM_STEP"):
            self.sched = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=cfg.STEPS_DROP,
                                                              gamma=cfg.GAMMA,
                                                              last_epoch=cfg.START_EPOCH - 1)
        elif not cfg.CONST_LR and hasattr(cfg, "WARM_STEP"):
            warm_up_with_multistep_lr = lambda \
                    epoch: (epoch + 1) / cfg.WARM_STEP if epoch < cfg.WARM_STEP else cfg.GAMMA ** len(
                [item for item in cfg.STEPS_DROP if item <= epoch])
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=warm_up_with_multistep_lr)


def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)


def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


def densenet_cifar():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=12)
