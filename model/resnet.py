import torch.nn as nn
import torch.nn.functional as F
from model.initializer import *


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(ResNet.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

        def get_sequential_output(self, x):
            intermediate_list = []
            out = F.relu(self.bn1(self.conv1(x)))
            intermediate_list.append(out)
            out = self.bn2(self.conv2(out))
            intermediate_list.append(F.relu(out))
            out += self.shortcut(x)
            if len(self.shortcut._modules.items()):
                intermediate_list.append(F.relu(self.shortcut(x)))
            return F.relu(out), intermediate_list

    def __init__(self, plan=None, initializer=model_init_fn, num_classes=10, n_channels=3):
        super(ResNet, self).__init__()

        self.num_classes = num_classes

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(n_channels, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(ResNet.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_sequential_output(self, x):
        output_list = []
        out = F.relu(self.bn(self.conv(x)))
        output_list.append(out)
        for i in range(len(self.blocks)):
            out, intermediate_list = self.blocks[i].get_sequential_output(out)
            output_list.extend(intermediate_list)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        output_list.append(out)
        return output_list

    def set_optimizer(self, cfg, args=None):
        if args is not None and not args.finetune_momentum:
            self.optim = torch.optim.SGD(self.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        elif args is not None and args.use_adam:
            self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
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

