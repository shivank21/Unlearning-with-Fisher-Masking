import argparse, configparser
import os, time
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.utils.dual_channel_logger import dual_channel_logger

from datasets.utils import get_dataset
from utils import *
from model.utils import *
from unlearning.unlearn import *
from unlearning.influence_functions import cal_if_lissa
import time

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(logger, model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets, reduction='sum')

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if logger is not None:
        print('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.3f}%) Error: {:.3f}%\n'.format(
                test_loss/len(test_loader.dataset), correct, len(test_loader.dataset),
                100.0*correct/total, 100 - 100.0*correct/total))

    return 100.0*correct/total


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cal_remain_continuous(data):
    num_continuous = 0
    for i in range(1, len(data)):
        num_continuous += data[i] - data[i-1]
    return num_continuous


def cal_forget_continuouos(data):
    num_continuous = 0
    for i in range(len(data)-1):
        num_continuous += data[i] - data[i-1]
    return num_continuous


def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default="class")
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                        "forget num if forget type is random", default=0)
    parser.add_argument('--finetune_epochs', type=int, help="finetune_epochs", default=30)
    parser.add_argument('--data_augment', action='store_true', help="Use which model (resnet, allcnn)",
                        default=False)

    args, extra_args = parser.parse_known_args()

    cfg = IniConfigurator('config/'+args.dataset+'_vgg16.cfg', extra_args)

    model_list = ['cifar10_resnet20', 'cifar10_vgg16', 'cifar10_densenet', 'cifar10_googlenet',
                  'cifar100_vgg16', 'cifar100_resnet50', 'cifar100_densenet', 'cifar100_googlenet',
                  'mnist_vgg16', 'mnist_resnet20', 'mnist_densenet', 'mnist_googlenet',
                    'tinyimagenet_vgg16', 'tinyimagenet_googlenet', 'tinyimagenet_resnet50',
                  'tinyimagenet_densenet']

    finetune_info, random_info, tfidf_info = {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}, \
                                             {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}, \
                                             {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}
    act_info, grad_info, fisher_info = {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}, \
                                       {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}, \
                                       {'remain':[], 'forget':[], 'remain_continuous':[], 'forget_continuous':[]}

    noise_info = {'remain':[], 'forget':[]}

    seed_list = ['666', '777', '888']

    for item in model_list:
        item_config = 'config/' + item + '.cfg'
        attribute = item.split("_")
        dataset_name, model_name = attribute[0], attribute[1]
        cfg = IniConfigurator(item_config, extra_args)
        testset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                              train=False, download=True, data_augment=args.data_augment)
        model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
        model.cuda()

        for seed in seed_list:
            set_seed(int(seed))
            ckpt_dir = './ckpts/' + cfg.exp_name + '/seed_' + seed
            best_model = "{}/best.pt".format(ckpt_dir)

            model0_file = "{}/forget_{}_{}_best.pt".format(ckpt_dir, args.forget_type,
                                                       args.forget_num)
            model0 = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
            model0.cuda()
            model0.load_state_dict(torch.load(model0_file)['model'])
            model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
            model.cuda()
            model.load_state_dict(torch.load(best_model)['model'])

            # random mask
            random_finetune_model_path = "{}/finetune_random_pruned_{}_{}_model.pth". \
                format(ckpt_dir, args.forget_type, args.forget_num)
            random_model = torch.load(random_finetune_model_path)
            random_acc_history = random_model['acc_history']
            random_forget_acc = random_model['forget_acc']
            random_info['remain'].append(random_acc_history[0])
            random_info['forget'].append(random_forget_acc[0])
            random_info['remain_continuous'].append(cal_forget_continuouos(random_acc_history))
            random_info['forget_continuous'].append(cal_remain_continuous(random_forget_acc))


            # tf_idf
            ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_model.pth". \
                format(ckpt_dir, args.forget_type, args.forget_num)
            ti_model = torch.load(ti_finetune_model_path)
            ti_acc_history = ti_model['acc_history']
            ti_forget_acc = ti_model['forget_acc']
            tfidf_info['remain'].append(ti_acc_history[0])
            tfidf_info['forget'].append(ti_forget_acc[0])
            tfidf_info['remain_continuous'].append(cal_remain_continuous(ti_acc_history))
            tfidf_info['forget_continuous'].append(cal_forget_continuouos(ti_forget_acc))

            # finetune
            finetune_file = "{}/finetune_baseline_{}_{}.pt".format(ckpt_dir, args.forget_type,
                                                                   args.forget_num)
            model_ft = torch.load(finetune_file)
            ft_acc_history = model_ft['acc_history']
            ft_forget_acc = model_ft['forget_acc']
            finetune_info['remain'].append(ft_acc_history[0])
            finetune_info['forget'].append(ft_forget_acc[0])
            finetune_info['remain_continuous'].append(cal_remain_continuous(ft_acc_history))
            finetune_info['forget_continuous'].append(cal_forget_continuouos(ft_forget_acc))


            # activation
            act_mask_finetune_model_path = "{}/finetuned_mask_pruned_{}_{}_model.pth". \
                format(ckpt_dir, args.forget_type, args.forget_num)
            act_model_mask = torch.load(act_mask_finetune_model_path)
            act_acc_history = act_model_mask['acc_history']
            act_forget_acc = act_model_mask['forget_acc']
            act_info['remain'].append(act_acc_history[0])
            act_info['forget'].append(act_forget_acc[0])
            act_info['remain_continuous'].append(cal_remain_continuous(act_acc_history))
            act_info['forget_continuous'].append(cal_forget_continuouos(act_forget_acc))

            # grad
            grad_mask_finetune_model_path = "{}/finetuned_grad_mask_pruned_{}_{}_model.pth". \
                format(ckpt_dir, args.forget_type, args.forget_num)
            grad_model_mask = torch.load(grad_mask_finetune_model_path)
            grad_acc_history = grad_model_mask['acc_history']
            grad_forget_acc = grad_model_mask['forget_acc']
            grad_info['remain'].append(grad_acc_history[0])
            grad_info['forget'].append(grad_forget_acc[0])
            grad_info['remain_continuous'].append(cal_remain_continuous(grad_acc_history))
            grad_info['forget_continuous'].append(cal_forget_continuouos(grad_forget_acc))

            # fisher
            fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_model.pth". \
                format(ckpt_dir, args.forget_type, args.forget_num)
            fisher_model_mask = torch.load(fisher_mask_finetune_model_path)
            fisher_acc_history = fisher_model_mask['acc_history']
            fisher_forget_acc = fisher_model_mask['forget_acc']
            fisher_info['remain'].append(fisher_acc_history[0])
            fisher_info['forget'].append(fisher_forget_acc[0])
            fisher_info['remain_continuous'].append(cal_remain_continuous(fisher_acc_history))
            fisher_info['forget_continuous'].append(cal_forget_continuouos(fisher_forget_acc))

            # noise
            modelf_file = "{}/fisher_baseline_{}_{}_model.pt".format(cfg.ckpt_dir, args.forget_type,
                                                                     args.forget_num)
            # prepare hessian_noise baseline
            modelf = torch.load(modelf_file)
            noise_info['remain'].append(modelf['acc_history'])
            noise_info['forget'].append(modelf['forget_acc'])

    info_list = [random_info, finetune_info, tfidf_info, act_info, grad_info, fisher_info, noise_info]
    for item in info_list:
        for name in item:
            item[name] = sum(item[name])/len(item[name])

    print("--------random-------")
    for k, v in random_info.items():
        print(k, v)
    print("ok\n")

    print("--------finetune-------")
    for k, v in finetune_info.items():
        print(k, v)
    print("ok\n")

    print("--------tfidf-------")
    for k, v in tfidf_info.items():
        print(k, v)
    print("ok\n")

    print("--------act-------")
    for k, v in act_info.items():
        print(k, v)
    print("ok\n")

    print("--------grad-------")
    for k, v in grad_info.items():
        print(k, v)
    print("ok\n")

    print("--------fisher-------")
    for k, v in fisher_info.items():
        print(k, v)
    print("ok\n")

    print("--------noise-------")
    for k, v in noise_info.items():
        print(k, v)
    print("ok\n")




if __name__ == '__main__':
    main()









