import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)

    def get_sequential_output(self, x):
        output_list = []
        y1 = self.b1(x)
        output_list.append(y1)
        y2 = x
        for i in range(len(self.b2)):
            y2 = self.b2[i](y2)
            if isinstance(self.b2[i], nn.ReLU):
                output_list.append(y2)
        y3 = x
        for i in range(len(self.b3)):
            y3 = self.b3[i](y3)
            if isinstance(self.b3[i], nn.ReLU):
                output_list.append(y3)
        y4 = self.b4(x)
        output_list.append(y4)
        return torch.cat([y1,y2,y3,y4], 1), output_list



class GoogLeNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.pre_layers = nn.Sequential(
            nn.Conv2d(n_channels, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_sequential_output(self, x):
        output_list = []
        x = self.pre_layers(x)
        output_list.append(x)
        x, intermediate_list = self.a3.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.b3.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x = self.max_pool(x)
        x, intermediate_list = self.a4.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.b4.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.c4.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.d4.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.e4.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x = self.max_pool(x)
        x, intermediate_list = self.a5.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x, intermediate_list = self.b5.get_sequential_output(x)
        output_list.extend(intermediate_list)
        x = self.avgpool(x)
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