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
                        default='config/cifar10_resnet20.cfg')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default=None)
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                        "forget num if forget type is random", default=0)
    parser.add_argument('--no_train', action='store_true', help="Whether or not to train model.",
                        default=False)
    parser.add_argument('--fine_prune', action='store_true', help="Whether or not to prune and fine-tune model.",
                        default=False)
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
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    set_seed(cfg.SEED)

    log_file = cfg.LOG
    if args.forget_type is not None and args.fine_prune:
        log_file = "{}/forget_{}_{}_fineprune_log".format(cfg.ckpt_dir, args.forget_type,
                                                             args.forget_num)
    elif args.forget_type is not None and not args.fine_prune:
        log_file = "{}/forget_{}_{}_log".format(cfg.ckpt_dir, args.forget_type, args.forget_num)

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

    if hasattr(cfg, "PRE_TRAIN") and cfg.PRE_TRAIN:
        if not os.path.isfile(cfg.PRE_MODEL):
            set_seed(cfg.SEED)
            pretrain(model_name, cfg, logger, args.data_augment)
        if model_name == 'allcnn':
            classifier_name = 'classifier.'
        elif 'resnet' in model_name:
            classifier_name = 'fc.'
        state = torch.load(cfg.PRE_MODEL)
        state = {k: v for k, v in state.items() if not k.startswith(classifier_name)}
        incompatible_keys = model.load_state_dict(state, strict=False)
        assert all([k.startswith(classifier_name) for k in incompatible_keys.missing_keys])
        model_init_file = "{}/model_init.pt".format(cfg.ckpt_dir)
        if not os.path.isfile(model_init_file):
            torch.save(model.state_dict(), model_init_file)

    if hasattr(cfg, "L2_NORM") and cfg.L2_NORM:
        model_init_file = "{}/model_init.pt".format(cfg.ckpt_dir)
        model_init = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
        model_init.cuda()
        model_init.load_state_dict(torch.load(model_init_file))

    if not args.no_train:
        last_model = cfg.LAST if args.forget_type is None else \
            "{}/forget_{}_{}_last.pt".format(cfg.ckpt_dir, args.forget_type, args.forget_num)
        best_model = cfg.BEST if args.forget_type is None else \
            "{}/forget_{}_{}_best.pt".format(cfg.ckpt_dir, args.forget_type, args.forget_num)

        model.set_optimizer(cfg)  # build optimizers

        start_epoch = cfg.START_EPOCH
        best_acc, best_epoch = 0.0, 0
        if cfg.IS_RESUME:
            ckpt = torch.load(last_model)
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['model'])
            best_acc, best_epoch = ckpt['best']
            model.optim.load_state_dict(ckpt['optim'])
            if hasattr(model, 'sched'):
                model.sched.load_state_dict(ckpt['sched'])

        set_seed(cfg.SEED)
        train_epoch = cfg.N_EPOCH
        num_step = len(train_loader)//10
        for i in range(start_epoch, train_epoch):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            #if hasattr(cfg, "DIS_BN") and cfg.DIS_BN:
            #    set_batchnorm_mode(model, train=False)
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                model.optim.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                if hasattr(cfg, "L2_NORM") and cfg.L2_NORM:
                    loss += l2_penalty(model, model_init, cfg.WD)
                loss.backward()
                model.optim.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = 100. * correct / total

                if batch_idx % num_step == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                        i, batch_idx * len(inputs), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item(), acc))

            torch.save({
                'epoch': i,
                'model': model.state_dict(),
                'best': (best_acc, best_epoch),
                'optim': model.optim.state_dict(),
                'sched': model.sched.state_dict()
                if hasattr(model, 'sched') else {},
            }, last_model)

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
            acc = 100.0 * correct / total
            logger.info('\nTest set: Average loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss / len(test_loader.dataset), correct, len(test_loader.dataset), acc))

            if hasattr(model, 'sched'):
                model.sched.step()
                logger.info("learning rate in epoch {}: lr:{}".
                      format(i+1, model.optim.state_dict()['param_groups'][0]['lr']))

            if acc > best_acc:
                best_acc, best_epoch = acc, i
                os.popen(f'cp {last_model} {best_model}')
        logger.info("train finish, best epoch: {}, best acc: {:.3f}%".format(best_epoch, best_acc))

    if args.fine_prune:
        model0_file = "{}/forget_{}_{}_best.pt".format(cfg.ckpt_dir, args.forget_type,
                                                       args.forget_num)
        model0 = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
        model0.cuda()
        model0.load_state_dict(torch.load(model0_file)['model'])
        model = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
        model.cuda()
        model.load_state_dict(torch.load(cfg.BEST)['model'])
        if hasattr(cfg, "L2_NORM") and cfg.L2_NORM:
            model_init_file = "{}/model_init.pt".format(cfg.ckpt_dir)
            model_init = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
            model_init.cuda()
            model_init.load_state_dict(torch.load(model_init_file))
        else:
            model_init = None

        forget_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                     train=True, download=True, data_augment=args.data_augment,
                                     forget_type=args.forget_type, forget_num=args.forget_num,
                                     only_forget=True)
        forget_loader = DataLoader(forget_dataset, batch_size=cfg.N_BATCH, shuffle=True,
                              num_workers=cfg.N_WORKER, worker_init_fn=worker_init_fn)

        remain_dataset, remain_loader = trainset, train_loader

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

        test_acc = test(logger, model0, remain_test_loader, loss_fn)

        logger.info("begin test ... ")

        # TF_IDF baseline
        ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_model.pth". \
            format(cfg.ckpt_dir, args.forget_type, args.forget_num)
        if not os.path.isfile(ti_finetune_model_path):
            set_seed(cfg.SEED)
            ti_pruned_net = copy.deepcopy(model)
            ti_pruned_net.set_optimizer(cfg, args)
            pruned_model_path = "{}/baseline_pruned_{}_{}_model.pth". \
                format(cfg.ckpt_dir, args.forget_type, args.forget_num)
            load_model_pytorch(ti_pruned_net, pruned_model_path, model_name)
            ti_test_acc_begin = test(None, ti_pruned_net, remain_test_loader, loss_fn)
            ti_forget_acc_begin = test(None, ti_pruned_net, forget_test_loader, loss_fn)

            t1 = time.time()
            model_ti, ti_acc_history, ti_train_loss, ti_forget_acc = \
                fine_tune(logger, cfg, args, ti_pruned_net, model_init, remain_loader,
                          remain_test_loader, forget_test_loader, loss_fn, test_acc,
                          epochs=args.finetune_epochs)
            t2 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2-t1))
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

        # random mask baseline
        random_finetune_model_path = "{}/finetune_random_pruned_{}_{}_model.pth". \
            format(cfg.ckpt_dir, args.forget_type, args.forget_num)
        if not os.path.exists(random_finetune_model_path):
            set_seed(cfg.SEED)
            t1 = time.time()
            unlearn = Unlearn(remain_loader, forget_loader)
            random_model_mask, mask_index, total_num = unlearn("random", cfg=cfg, args=args, model=model,
                                             remove_ratio=cfg.REMOVE_RATIO, largest=True)
            t2 = time.time()
            random_model_mask.set_optimizer(cfg, args)
            random_test_acc_begin = test(None, random_model_mask, remain_test_loader, loss_fn)
            random_forget_acc_begin = test(None, random_model_mask, forget_test_loader, loss_fn)
            t3 = time.time()
            random_model_mask, random_acc_history, random_train_loss, random_forget_acc = \
                fine_tune(logger, cfg, args, random_model_mask, model_init, remain_loader,
                               remain_test_loader, forget_test_loader, loss_fn,
                               test_acc, epochs=args.finetune_epochs)
            t4 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1 + t4 - t3))
            random_acc_history.insert(0, random_test_acc_begin)
            random_forget_acc.insert(0, random_forget_acc_begin)
            torch.save({
                'model': random_model_mask.state_dict(),
                'acc_history': random_acc_history,
                'forget_acc': random_forget_acc,
                'retrain_test_acc': test_acc,
                "train_loss": random_train_loss,
            }, random_finetune_model_path)
        else:
            random_model = torch.load(random_finetune_model_path)
            random_acc_history = random_model['acc_history']
            random_forget_acc = random_model['forget_acc']
        logger.info("random mask ok")

        act_mask_finetune_model_path = "{}/finetuned_mask_pruned_{}_{}_model.pth". \
            format(cfg.ckpt_dir, args.forget_type, args.forget_num)
        if not os.path.exists(act_mask_finetune_model_path):
            set_seed(cfg.SEED)
            t1 = time.time()
            unlearn = Unlearn(remain_loader, forget_loader)
            act_model_mask, mask_index, total_num = unlearn("activation", cfg=cfg, args=args, model=model,
                                             remove_ratio=cfg.REMOVE_RATIO, largest=True)
            t2 = time.time()
            act_model_mask.set_optimizer(cfg, args)
            test_acc_begin = test(None, act_model_mask, remain_test_loader, loss_fn)
            forget_acc_begin = test(None, act_model_mask, forget_test_loader, loss_fn)
            t3 = time.time()
            act_model_mask, act_acc_history, act_train_loss, act_forget_acc = \
                fine_tune(logger, cfg, args, act_model_mask, model_init, remain_loader,
                               remain_test_loader, forget_test_loader,
                               loss_fn, test_acc, epochs=args.finetune_epochs)
            t4 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1 + t4 - t3))
            act_acc_history.insert(0, test_acc_begin)
            act_forget_acc.insert(0, forget_acc_begin)
            torch.save({
                'model': act_model_mask.state_dict(),
                'acc_history': act_acc_history,
                'forget_acc': act_forget_acc,
                'retrain_test_acc': test_acc,
                "train_loss": act_train_loss,
            }, act_mask_finetune_model_path)
        else:
            act_model_mask = torch.load(act_mask_finetune_model_path)
            act_acc_history = act_model_mask['acc_history']
            act_forget_acc = act_model_mask['forget_acc']
        logger.info("act mask ok")

        fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_model.pth". \
            format(cfg.ckpt_dir, args.forget_type, args.forget_num)
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
                fine_tune(logger, cfg, args, fisher_model_mask, model_init, remain_loader,
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

        grad_mask_finetune_model_path = "{}/finetuned_grad_mask_pruned_{}_{}_model.pth". \
            format(cfg.ckpt_dir, args.forget_type, args.forget_num)
        if not os.path.exists(grad_mask_finetune_model_path):
            set_seed(cfg.SEED)
            t1 = time.time()
            unlearn = Unlearn(remain_loader, forget_loader)
            grad_model_mask, mask_index, total_num = unlearn("gradients", cfg=cfg, args=args, model=model,
                                                               remove_ratio=cfg.REMOVE_RATIO, largest=True)
            t2 = time.time()
            grad_model_mask.set_optimizer(cfg, args)
            test_acc_begin = test(None, grad_model_mask, remain_test_loader, loss_fn)
            forget_acc_begin = test(None, grad_model_mask, forget_test_loader, loss_fn)
            t3 = time.time()
            grad_model_mask, grad_acc_history, grad_train_loss, grad_forget_acc = \
                fine_tune(logger, cfg, args, grad_model_mask, model_init, remain_loader,
                          remain_test_loader, forget_test_loader,
                          loss_fn, test_acc, epochs=args.finetune_epochs)
            t4 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1 + t4 - t3))
            grad_acc_history.insert(0, test_acc_begin)
            grad_forget_acc.insert(0, forget_acc_begin)
            torch.save({
                'model': grad_model_mask.state_dict(),
                'acc_history': grad_acc_history,
                'forget_acc': grad_forget_acc,
                'retrain_test_acc': test_acc,
                "train_loss": grad_train_loss,
            }, grad_mask_finetune_model_path)
        else:
            grad_model_mask = torch.load(grad_mask_finetune_model_path)
            grad_acc_history = grad_model_mask['acc_history']
            grad_forget_acc = grad_model_mask['forget_acc']
        logger.info("grad mask ok")

        modelf_file = "{}/fisher_baseline_{}_{}_model.pt".format(cfg.ckpt_dir, args.forget_type,
                                                               args.forget_num)
        # prepare hessian_noise baseline
        if not os.path.isfile(modelf_file):
            set_seed(cfg.SEED)
            modelf = copy.deepcopy(model)

            for p in modelf.parameters():
                p.data0 = copy.deepcopy(p.data.clone())

            model_hessian_file = "{}/model_hessian_remain_{}_{}.pt".format(cfg.ckpt_dir,
                                                                           args.forget_type,
                                                                           args.forget_num)
            t1 = time.time()
            if os.path.isfile(model_hessian_file):
                model_grad = copy.deepcopy(model)
                model_grad.load_state_dict(torch.load(model_hessian_file))
            else:
                model_grad = hessian(remain_loader.dataset, modelf)
                torch.save(model_grad.state_dict(), model_hessian_file)

            fisher_dir = []
            alpha = 1e-6
            torch.manual_seed(cfg.SEED)
            for g, p in zip(model_grad.parameters(), modelf.parameters()):
                if args.forget_type != 'class':
                    print("some error might happen here")
                    exit(0)
                mu, var = get_mean_var(g, p, args.forget_type, args.forget_num,
                                       testset.num_class(), False, alpha=alpha)
                p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
                fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
            t2 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1))

            test_acc_f = test(None, modelf, remain_test_loader, loss_fn)
            forget_acc_f = test(None, modelf, forget_test_loader, loss_fn)
            torch.save({
                'model': modelf.state_dict(),
                'acc_history': test_acc_f,
                'forget_acc': forget_acc_f,
                'retrain_test_acc': test_acc,
            }, modelf_file)
        else:
            # modelf = get_model(model_name, num_classes=testset.num_class(), n_channels=cfg.IN_CHANNEL)
            # modelf.load_state_dict(torch.load(modelf_file))
            # modelf.cuda()
            modelf = torch.load(modelf_file)
            test_acc_f = modelf['acc_history']
            forget_acc_f = modelf['forget_acc']
        logger.info("fisher baseline is ok")

        # prepare finetune baseline
        finetune_file = "{}/finetune_baseline_{}_{}.pt".format(cfg.ckpt_dir, args.forget_type,
                                                               args.forget_num)

        if not os.path.isfile(finetune_file):
            model_ft = copy.deepcopy(model)
            model_ft.set_optimizer(cfg, args)
            ft_test_acc_begin = test(None, model_ft, remain_test_loader, loss_fn)
            ft_forget_acc_begin = test(None, model_ft, forget_test_loader, loss_fn)
            t1 = time.time()
            model_ft, ft_acc_history, ft_train_loss, ft_forget_acc = \
                fine_tune(logger, cfg, args, model_ft, model_init, remain_loader,
                          remain_test_loader, forget_test_loader, loss_fn, test_acc,
                          epochs=args.finetune_epochs)
            t2 = time.time()
            logger.info("finetune {} epochs, time :{}".format(args.finetune_epochs, t2 - t1))
            ft_acc_history.insert(0, ft_test_acc_begin)
            ft_forget_acc.insert(0, ft_forget_acc_begin)
            torch.save({
                'model': model_ft.state_dict(),
                'acc_history': ft_acc_history,
                'forget_acc':ft_forget_acc,
                'retrain_test_acc': test_acc,
                "train_loss": ft_train_loss,
            },  finetune_file)
        else:
            model_ft = torch.load(finetune_file)
            ft_acc_history = model_ft['acc_history']
            ft_forget_acc = model_ft['forget_acc']
        logger.info("finetune ok")


        pic_data = {"random_test_acc": random_acc_history, "act_test_acc": act_acc_history,
                    "fisher_test_acc": fisher_acc_history, "grad_test_acc": grad_acc_history,
                    "ft_test_acc": ft_acc_history, "ti_test_acc": ti_acc_history,
                    "fisher_noise_test_acc": test_acc_f,
                    "random_forget_acc": random_forget_acc, "act_forget_acc": act_forget_acc,
                    "fisher_forget_acc":fisher_forget_acc, "grad_forget_acc": grad_forget_acc,
                    "ft_forget_acc":ft_forget_acc, "ti_forget_acc": ti_forget_acc,
                    "fisher_noise_forget_acc":forget_acc_f,
                    #"random_train_loss": random_train_loss, "train_loss": train_loss,
                    #"ft_train_loss":ft_train_loss, "ti_train_loss": ft_train_loss,
                    "retrain_remain_test_acc": test_acc}

        np.save("{}/dict_{}_{}.npy".format(cfg.ckpt_dir, args.forget_type, args.forget_num), pic_data)

        picture_remain_acc(cfg, args, model_name + '_' + dataset_name, pic_data)
        picture_forget_acc(cfg, args, model_name + '_' + dataset_name, pic_data)


if __name__ == '__main__':
    main()


