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
from baseline import tf_idf_baseline
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


def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/cifar100_resnet50.cfg')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default=None)
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                        "forget num if forget type is random", default=0)
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
    attribute = args.config[7:-4].split("_")
    dataset_name, model_name = attribute[0], attribute[1]

    exper_ckpt_dir = cfg.ckpt_dir + '/remove_ratio'

    if not os.path.exists(exper_ckpt_dir):
        os.makedirs(exper_ckpt_dir)

    set_seed(cfg.SEED)

    log_file = exper_ckpt_dir + '/log'
    if args.forget_type is not None:
        log_file = "{}/forget_{}_{}_log".format(exper_ckpt_dir, args.forget_type, args.forget_num)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=log_file,
        file_model='w',
        formatter="%(asctime)s - %(levelname)s - %(message)s",
        time_formatter="%m-%d %H:%M")

    print('==> Preparing data..')
    trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                           train=True, download=True, data_augment=args.data_augment,
                           forget_type=args.forget_type, forget_num=args.forget_num)
    train_loader = DataLoader(trainset, batch_size=cfg.N_BATCH, shuffle=True,
                              num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    testset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET), train=False,
                          download=True, data_augment=args.data_augment)
    test_loader = DataLoader(testset, batch_size=cfg.N_BATCH, shuffle=False,
                             num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
    model.cuda()
    loss_fn = F.cross_entropy
    BEST_MODEL = "{}/best.pt".format(exper_ckpt_dir)

    set_seed(cfg.SEED)

    model0_file = "{}/forget_{}_{}_best.pt".format(exper_ckpt_dir, args.forget_type,
                                                   args.forget_num)
    model0 = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
    model0.cuda()
    model0.load_state_dict(torch.load(model0_file)['model'])
    model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
    model.cuda()
    model.load_state_dict(torch.load(BEST_MODEL)['model'])

    forget_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=True, download=True, data_augment=args.data_augment,
                                 forget_type=args.forget_type, forget_num=args.forget_num,
                                 only_forget=True)
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.N_BATCH, shuffle=True,
                               num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    remain_dataset, remain_loader = trainset, train_loader
    sample_train_loader = DataLoader(remain_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    sampler=torch.utils.data.RandomSampler(
                                        train_loader.dataset, True, num_samples=len(remain_dataset) // 10),
                                    num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)


    remain_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET), train=False,
                                      download=True, data_augment=args.data_augment,
                                      forget_type=args.forget_type, forget_num=args.forget_num)
    remain_test_loader = DataLoader(remain_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                     num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    forget_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET), train=False,
                                      download=True, data_augment=args.data_augment,
                                      forget_type=args.forget_type, forget_num=args.forget_num,
                                      only_forget=True)
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

    full_trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                train=True, download=True, data_augment=args.data_augment)
    full_train_loader = DataLoader(full_trainset, batch_size=cfg.N_BATCH, shuffle=False,
                                   num_workers=cfg.N_WORKER)

    test_acc = test(logger, model0, test_loader, loss_fn)

    logger.info("begin test ... ")

    fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_{}_model.pth". \
        format(exper_ckpt_dir, cfg.REMOVE_RATIO, args.forget_type, args.forget_num)
    if not os.path.exists(fisher_mask_finetune_model_path):
        set_seed(cfg.SEED)
        t1 = time.time()
        unlearn = Unlearn(remain_loader, forget_loader)
        fisher_model_mask, mask_index, total_num = unlearn("fisher", cfg=cfg, args=args, model=model,
                                                           remove_ratio=cfg.REMOVE_RATIO, largest=True)
        t2 = time.time()
        fisher_model_mask.set_optimizer(cfg, args)
        test_acc_begin = test(None, fisher_model_mask, remain_test_loader, loss_fn)
        forget_acc_begin = test(None, fisher_model_mask, forget_test_loader, loss_fn)
        t3 = time.time()
        fisher_model_mask, fisher_acc_history, fisher_train_loss, fisher_forget_acc = \
            fine_tune(logger, cfg, args, fisher_model_mask, None, sample_train_loader,
                      remain_test_loader, forget_test_loader,
                      loss_fn, test_acc, epochs=args.finetune_epochs)
        t4 = time.time()
        logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1 + t4 - t3))
        fisher_acc_history.insert(0, test_acc_begin)
        fisher_forget_acc.insert(0, forget_acc_begin)
        torch.save({
            'model': fisher_model_mask.state_dict(),
            'acc_history': fisher_acc_history,
            'forget_acc': fisher_forget_acc,
            'retrain_test_acc': test_acc,
            "train_loss": fisher_train_loss,
        }, fisher_mask_finetune_model_path)
    else:
        fisher_model_mask = torch.load(fisher_mask_finetune_model_path)
        fisher_acc_history = fisher_model_mask['acc_history']
        fisher_forget_acc = fisher_model_mask['forget_acc']
    logger.info("fisher mask ok")

    # TF_IDF baseline
    ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_{}_model.pth". \
        format(exper_ckpt_dir, cfg.REMOVE_RATIO, args.forget_type, args.forget_num)
    if not os.path.isfile(ti_finetune_model_path):
        set_seed(cfg.SEED)
        ti_pruned_net = copy.deepcopy(model)
        prefix_file_name = "{}/baseline_pruned_{}_{}_{}". \
            format(exper_ckpt_dir, cfg.REMOVE_RATIO, args.forget_type, args.forget_num)
        ti_pruned_net = tf_idf_baseline(args, cfg, logger, ti_pruned_net, model_name,
                                        full_train_loader, forget_test_loader,
                                        remain_test_loader, loss_fn, prefix_file_name)
        ti_pruned_net.set_optimizer(cfg, args)
        ti_test_acc_begin = test(None, ti_pruned_net, remain_test_loader, loss_fn)
        ti_forget_acc_begin = test(None, ti_pruned_net, forget_test_loader, loss_fn)

        t1 = time.time()
        model_ti, ti_acc_history, ti_train_loss, ti_forget_acc = \
            fine_tune(logger, cfg, args, ti_pruned_net, None, remain_loader,
                      remain_test_loader, forget_test_loader, loss_fn, test_acc,
                      epochs=args.finetune_epochs)
        t2 = time.time()
        logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1))
        ti_acc_history.insert(0, ti_test_acc_begin)
        ti_forget_acc.insert(0, ti_forget_acc_begin)

        torch.save({
            'model': ti_pruned_net.state_dict(),
            'acc_history': ti_acc_history,
            'forget_acc': ti_forget_acc,
            'retrain_test_acc': test_acc,
            "train_loss": ti_train_loss,
        }, ti_finetune_model_path)
    else:
        ti_model = torch.load(ti_finetune_model_path)
        ti_acc_history = ti_model['acc_history']
        ti_forget_acc = ti_model['forget_acc']
    logger.info("tf-idf baseline ok")




if __name__ == '__main__':
    main()
