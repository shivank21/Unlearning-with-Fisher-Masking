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
import time
import shutil

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

def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/tinyimagenet_resnet50.cfg')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default="class")
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                        "forget num if forget type is random", default=0)
    parser.add_argument('--no_train', action='store_true', help="Whether or not to train model.",
                        default=True)
    parser.add_argument('--fine_prune', action='store_true', help="Whether or not to prune and fine-tune model.",
                        default=True)
    parser.add_argument('--experiment', default='only_forget')
    parser.add_argument('--sample_small', action='store_true', default=False,
                        help="use samll dataset to finetune or not")
    parser.add_argument('--finetune_epochs', type=int, help="finetune_epochs", default=30)
    parser.add_argument('--data_augment', action='store_true', help="Use which model (resnet, allcnn)",
                        default=False)
    parser.add_argument('--finetune_momentum', action='store_false', default=True,
                        help="whether or not to use finetune momentum")
    parser.add_argument('--finetune_warmup', type=int, default=0, help="finetune warmup epochs")
    parser.add_argument('--finetune_eval_mode', action='store_true', default=False, help="use eval mode in finetune")
    parser.add_argument('--use_adam', action='store_true', default=False, help="use Adam optimizer instead")

    args, extra_args = parser.parse_known_args()
    cfg = IniConfigurator(args.config, extra_args)
    forget_ckpt_dir = cfg.ckpt_dir + '/' + args.experiment

    if os.path.exists(forget_ckpt_dir):
        shutil.rmtree(forget_ckpt_dir)
    os.makedirs(forget_ckpt_dir)
    os.popen('cp {}/best.pt {}'.format(cfg.ckpt_dir, forget_ckpt_dir))
    os.popen('cp {}/last.pt {}'.format(cfg.ckpt_dir, forget_ckpt_dir))
    os.popen('cp {}/forget_class_0_best.pt {}'.format(cfg.ckpt_dir, forget_ckpt_dir))
    os.popen('cp {}/forget_class_0_last.pt {}'.format(cfg.ckpt_dir, forget_ckpt_dir))

    set_seed(cfg.SEED)

    log_file = cfg.LOG
    if args.forget_type is not None and args.fine_prune:
        log_file = "{}/forget_{}_{}_fineprune_log".format(forget_ckpt_dir, args.forget_type,
                                                             args.forget_num)
    elif args.forget_type is not None and not args.fine_prune:
        log_file = "{}/forget_{}_{}_log".format(forget_ckpt_dir, args.forget_type, args.forget_num)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=log_file,
        file_model='w',
        formatter="%(asctime)s - %(levelname)s - %(message)s",
        time_formatter="%m-%d %H:%M")

    attribute = args.config[7:-4].split("_")
    dataset_name, model_name = attribute[0], attribute[1]
    print('==> Preparing data..')
    loss_fn = F.cross_entropy

    testset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET), train=False,
                          download=True, data_augment=args.data_augment)

    model0_file = "{}/forget_{}_{}_best.pt".format(forget_ckpt_dir, args.forget_type,
                                                   args.forget_num)
    model0 = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
    model0.cuda()
    model0.load_state_dict(torch.load(model0_file)['model'])
    model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
    model.cuda()
    model.load_state_dict(torch.load(cfg.BEST)['model'])

    forget_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=True, download=True, data_augment=args.data_augment,
                                 forget_type=args.forget_type, forget_num=args.forget_num,
                                 only_forget=True)
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.N_BATCH, shuffle=True,
                          num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)
    remain_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                      train=False, download=True, data_augment=args.data_augment,
                                      forget_type=args.forget_type, forget_num=args.forget_num)
    remain_test_loader = DataLoader(remain_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)
    forget_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                      train=False, download=True, data_augment=args.data_augment,
                                      forget_type=args.forget_type, forget_num=args.forget_num,
                                      only_forget=True)
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    remainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                           train=True, download=True, data_augment=args.data_augment,
                           forget_type=args.forget_type, forget_num=args.forget_num)
    remain_loader = DataLoader(remainset, batch_size=cfg.N_BATCH, shuffle=True,
                              num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)
    test_acc = test(logger, model0, remain_test_loader, loss_fn)
    print(test_acc)


    logger.info("begin test ... ")
    if args.sample_small:
        idx = random.sample(range(len(remain_loader.dataset)), 50)
        sample_subset = torch.utils.data.Subset(remain_loader.dataset, idx)
        data_loader_small = torch.utils.data.DataLoader(sample_subset,
                                                        batch_size=remain_loader.batch_size,
                                                        num_workers=remain_loader.num_workers)
    else:
        data_loader_small = None

    set_seed(cfg.SEED)
    unlearn = Unlearn(data_loader_small, forget_loader)
    act_model_mask, mask_index, total_num = unlearn("activation", cfg=cfg, args=args, model=model,
                                     remove_ratio=cfg.REMOVE_RATIO, largest=True)
    test_acc_begin = test(None, act_model_mask, remain_test_loader, loss_fn)
    forget_acc_begin = test(None, act_model_mask, forget_test_loader, loss_fn)
    if data_loader_small is None:
        print(test_acc_begin, forget_acc_begin)
    else:
        act_model_mask.set_optimizer(cfg, args)
        act_model_mask, act_acc_history, act_train_loss, act_forget_acc = \
            fine_tune(logger, cfg, args, act_model_mask, None, data_loader_small,
                      remain_test_loader, forget_test_loader,
                      loss_fn, test_acc, epochs=args.finetune_epochs)
        act_acc_history.insert(0, test_acc_begin)
        act_forget_acc.insert(0, forget_acc_begin)
        print(act_acc_history)
        print(act_forget_acc)
    logger.info("act mask ok")

    set_seed(cfg.SEED)
    unlearn = Unlearn(data_loader_small, forget_loader)
    fisher_model_mask, mask_index, total_num = unlearn("fisher", cfg=cfg, args=args, model=model,
                                                        remove_ratio=cfg.REMOVE_RATIO, largest=True)
    test_acc_begin = test(None, fisher_model_mask, remain_test_loader, loss_fn)
    forget_acc_begin = test(None, fisher_model_mask, forget_test_loader, loss_fn)
    if data_loader_small is None:
        print(test_acc_begin, forget_acc_begin)
    else:
        fisher_model_mask.set_optimizer(cfg, args)
        fisher_model_mask, fisher_acc_history, fisher_train_loss, fisher_forget_acc = \
            fine_tune(logger, cfg, args, fisher_model_mask, None, data_loader_small,
                      remain_test_loader, forget_test_loader,
                      loss_fn, test_acc, epochs=args.finetune_epochs)
        fisher_acc_history.insert(0, test_acc_begin)
        fisher_forget_acc.insert(0, forget_acc_begin)
        print(fisher_acc_history)
        print(fisher_forget_acc)
    logger.info("fisher mask ok")


if __name__ == '__main__':
    main()


