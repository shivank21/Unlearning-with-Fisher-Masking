import copy, os
from itertools import chain
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import grad
from model.all_cnn import *
from model.resnet import *
from model.lenet import *
from model.VGG import *
from model.GoogleNet import *
from model.DenseNet import *
from model.AlexNet import *

_MODELS = {}


def _add_model(model_fn):
    _MODELS[model_fn.__name__] = model_fn
    return model_fn


@_add_model
def allcnn(**kwargs):
    return AllCNN(**kwargs)


@_add_model
def resnet20(**kwargs):
    D = (20-2)//6
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]
    return ResNet(plan, **kwargs)

@_add_model
def lenet(**kwargs):
    return LeNet(**kwargs)


@_add_model
def resnet50(**kwargs):
    D = (50-2) // 6
    W = 16
    plan = [(W, D), (2*W, D), (4*W, D)]
    return ResNet(plan, **kwargs)

@_add_model
def vgg16(**kwargs):
    return VGG('VGG16', **kwargs)

@_add_model
def densenet(**kwargs):
    return DenseNet(num_block=[6, 12, 24, 16], **kwargs)

@_add_model
def googlenet(**kwargs):
    return GoogLeNet(**kwargs)

@_add_model
def alexnet(**kwargs):
    return AlexNet(**kwargs)


def get_model(name, **kwargs):
    return _MODELS[name](**kwargs)


def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss


def set_batchnorm_mode(model, train=True):
    if isinstance(model, torch.nn.BatchNorm1d) or isinstance(model, torch.nn.BatchNorm2d):
        print(model)
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_batchnorm_mode(l, train=train)


def set_dropout_mode(model, train=True):
    if isinstance(model, torch.nn.Dropout):
        if train:
            model.train()
        else:
            model.eval()
    for l in model.children():
        set_dropout_mode(l, train=train)


def pretrain(model_name, cfg, logger, data_augment=False):
    if data_augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR100(root=os.path.join(cfg.data_dir, cfg.PRE_DATASET), train=True,
                                             download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.N_BATCH, shuffle=True,
                                               num_workers=cfg.N_WORKER)
    model = get_model(model_name, num_classes=len(np.unique(np.array(trainset.targets))))
    model.cuda()
    model.train()
    model.set_optimizer_for_pretrain(cfg)
    model_init = copy.deepcopy(model)
    loss_fn = F.cross_entropy
    for i in range(cfg.PRE_N_EPOCH):
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            model.optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if cfg.L2_NORM:
                loss += l2_penalty(model, model_init, cfg.WD)
            loss.backward()
            model.optim.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            if batch_idx % 50 == 0:
                logger.info('Pre-Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                    i, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), acc))
    torch.save(model.state_dict(), cfg.PRE_MODEL)

from train import set_seed
def fine_tune(logger, cfg, args, model, model_init, train_loader, test_loader,
              forget_loader, loss_fn, test_acc, epochs=5):
    set_seed(cfg.SEED)
    if hasattr(model, 'sched'):
        num_epochs = 5
        if args.finetune_warmup == 0:
            mile_stone = [int(item * 1.0 / cfg.N_EPOCH * num_epochs * len(train_loader)) for item in cfg.STEPS_DROP]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optim, milestones=mile_stone, gamma=cfg.GAMMA)
        else:
            warm_up_with_multistep_lr = lambda epoch: epoch / len(train_loader) / args.finetune_warmup \
                if epoch <= len(train_loader) * args.finetune_warmup \
                else cfg.GAMMA ** len([int(item * 1.0 / cfg.N_EPOCH * num_epochs * len(train_loader))
                                       for item in cfg.STEPS_DROP if item <= epoch])
            scheduler = torch.optim.lr_scheduler.LambdaLR(model.optim, lr_lambda=warm_up_with_multistep_lr)
    elif args.finetune_warmup > 0:
        warm_up_with_multistep_lr = lambda epoch: epoch / len(train_loader) / args.finetune_warmup \
            if epoch <= len(train_loader) * args.finetune_warmup else 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(model.optim, lr_lambda=warm_up_with_multistep_lr)

    acc_history, train_loss_history, forget_acc_history = [], [], []
    for i in range(epochs):
        if args.finetune_eval_mode:
            model.eval()
        else:
            model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            model.optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if hasattr(cfg, "L2_NORM") and cfg.L2_NORM:
                loss += l2_penalty(model, model_init, cfg.WD)
            loss.backward()
            model.optim.step()
            if hasattr(model, 'sched') or args.finetune_warmup > 0:
                scheduler.step()

            train_loss += loss.item()
        epoch_test_acc = test(None, model, test_loader, loss_fn)
        epoch_forget_acc = test(None, model, forget_loader, loss_fn)
        acc_history.append(epoch_test_acc)
        forget_acc_history.append(epoch_forget_acc)
        train_loss_history.append(train_loss)
        #if epoch_test_acc >= test_acc:
        #    break
    return model, acc_history, train_loss_history, forget_acc_history

