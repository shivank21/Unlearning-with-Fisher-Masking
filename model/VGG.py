import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, n_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.num_classes = num_classes
        self.features = self._make_layers(cfg[vgg_name], in_channels=n_channels)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_sequential_output(self, x):
        output_list = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            if isinstance(self.features[i], nn.ReLU):
                output_list.append(x)
        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        output_list.append(out)
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


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')