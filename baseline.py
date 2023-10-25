import random
import argparse
from torch.utils.data import DataLoader
from torch.backends import cudnn
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.utils.dual_channel_logger import dual_channel_logger
from datasets.utils import get_dataset
from model.utils import *
import torch.optim as optim
from unlearning.class_pruner import acculumate_feature, calculate_cp, \
    get_threshold_by_sparsity, TFIDFPruner
from unlearning.scheduler import *
import time

def setup_seed(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(net, epochs, lr, train_loader, test_loader, save_path, save_acc=80.0, seed=0,
          start_epoch=0, device='cuda',
          label_smoothing=0, warmup_step=0, warm_lr=10e-5):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """
    # print('==> Preparing data..')

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)

    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    # best_acc = 0  # best test accuracy
    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        print('\nEpoch: %d' % epoch, '/ %d;' % epochs, 'learning_rate:',
              optimizer.state_dict()['param_groups'][0]['lr'])
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # if warmup scheduler==None or not in scope of warmup -> if_warmup=False
            if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            if if_warmup:
                warmup_scheduler.step()

            # clear masked weights to zero
            '''for n, m in net.named_modules():
                if isinstance(m, nn.Conv2d):
                    mask = m.mask
                    m.weight.data *= mask
                    m.weight.grad.data *= mask'''

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        training_loss = train_loss / (batch_idx + 1)
        training_acc = correct / total
        print("Train Loss=%.8f, Train acc=%.8f" % (training_loss, training_acc))

        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()

        """
        Start the testing code.
        """
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        num_val_steps = len(test_loader)
        val_acc = correct / total

        if val_acc * 100 > save_acc:
            _save_acc = val_acc * 100
            print('save path', save_path)
            torch.save(net.state_dict(), save_path)

        testing_loss = test_loss / (num_val_steps)
        print("Test Loss=%.8f, Test acc=%.8f" % (testing_loss, val_acc))
        # epoch; training loss, test loss, training acc, test acc
        print(str(epoch + 1) + ' ' + str(training_loss) + ' ' + str(testing_loss) + ' ' +
                     str(training_acc) + ' ' + str(val_acc))


def load_model_pytorch(model, load_model_path, model_name):
    #print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model_path)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    if model_name == "vgg":
        from collections import OrderedDict

        load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            #print(key, model.state_dict()[key].shape)
        #print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            #print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, False)


def tf_idf_baseline(args, cfg, logger, model, model_name, full_train_loader,
                    forget_test_loader, remain_test_loader, loss_fn,
                    prefix_file_name, stop_batch=1, coe=0):
    print("------begin tf-idf--------")
    feature_iit, classes = acculumate_feature(model, full_train_loader, stop_batch)
    total_class = full_train_loader.dataset.num_class()
    tf_idf_map = calculate_cp(feature_iit, classes, total_class, coe,
                              unlearn_class=args.forget_num)
    threshold = get_threshold_by_sparsity(tf_idf_map, cfg.REMOVE_RATIO)
    print('threshold', threshold)

    '''test before pruning'''
    list_allclasses = list(range(total_class))
    # unlearn_listclass = [args.forget_num]
    list_allclasses.remove(args.forget_num)  # rest classes
    print('*' * 5 + 'testing before pruning' + '*' * 15)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    test(logger, model, forget_test_loader, loss_fn)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(logger, model, remain_test_loader, loss_fn)
    print('*' * 40)

    '''pruning'''
    t1 = time.time()
    cp_config = {"threshold": threshold, "map": tf_idf_map}
    config_list = [{
        'sparsity': cfg.REMOVE_RATIO,
        'op_types': ['Conv2d']
    }]
    pruner = TFIDFPruner(model, config_list, cp_config=cp_config)
    pruner.compress()
    t2 = time.time()
    logger.info("time used: {}".format(t2 - t1))
    pruned_model_path = prefix_file_name + '_model.pth'
    pruned_mask_path = prefix_file_name + '_mask.pth'
    pruner.export_model(pruned_model_path, pruned_mask_path)
    pruned_net = copy.deepcopy(model)
    load_model_pytorch(pruned_net, pruned_model_path, model_name)

    '''test after pruning'''
    print('*' * 5 + 'testing after pruning' + '*' * 12)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    test(logger, pruned_net, forget_test_loader, loss_fn)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(logger, pruned_net, remain_test_loader, loss_fn)
    print('*' * 40)
    print("------end tf-idf--------")
    return pruned_net

def main():
    # configuration
    parser = argparse.ArgumentParser(description='Class Pruning')
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/cifar10_resnet20.cfg')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default="class")
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                                                       "forget num if forget type is random", default=0)
    parser.add_argument('--stop_batch', type=int, default=1,
                        help="Sample batch number (default: 1)")
    parser.add_argument('--search_batch_size', type=int, default=256,
                        help='input batch size for search (default: 256)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs to fine tune (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate to fine tune (default: 0.1)')
    parser.add_argument('--coe', type=int, default=0,
                        help='whether to use balance coefficient')
    parser.add_argument('--save_acc', type=float, default=0.0,
                        help='save accuracy')
    parser.add_argument('--label_smoothing', type=float, default='0.0',
                        help='label smoothing rate')
    parser.add_argument('--warmup_step', type=int, default='0',
                        help='warm up epochs')
    parser.add_argument('--warm_lr', type=float, default='10e-5',
                        help='warm up learning rate')
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
    print(args)

    log_file = cfg.LOG
    if args.forget_type is not None:
        log_file = "{}/forget_{}_{}_baseline_log".format(cfg.ckpt_dir, args.forget_type,
                                                         args.forget_num)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=log_file,
        file_model='w',
        formatter="%(asctime)s - %(levelname)s - %(message)s",
        time_formatter="%m-%d %H:%M")

    setup_seed(cfg.SEED)

    attribute = args.config[7:-4].split("_")
    dataset_name, model_name = attribute[0], attribute[1]

    print('==> Preparing data..')
    remainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                            train=True, download=True, data_augment=args.data_augment,
                            forget_type=args.forget_type, forget_num=args.forget_num)
    remain_loader = DataLoader(remainset, batch_size=cfg.N_BATCH, shuffle=False,
                               num_workers=cfg.N_WORKER)

    full_trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                train=True, download=True, data_augment=args.data_augment)
    full_train_loader = DataLoader(full_trainset, batch_size=cfg.N_BATCH, shuffle=False,
                                   num_workers=cfg.N_WORKER)

    remain_testset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=False, download=True, data_augment=args.data_augment,
                                 forget_type=args.forget_type, forget_num=args.forget_num)
    remain_test_loader = DataLoader(remain_testset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER)

    forget_testset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=False, download=True, data_augment=args.data_augment,
                                 forget_type=args.forget_type, forget_num=args.forget_num,
                                 only_forget=True)
    forget_test_loader = DataLoader(forget_testset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER)

    model = get_model(model_name, num_classes=full_trainset.num_class(), n_channels=cfg.IN_CHANNEL)
    model.cuda()
    model.load_state_dict(torch.load(cfg.BEST)['model'])
    loss_fn = F.cross_entropy

    '''pre-processing'''
    feature_iit, classes = acculumate_feature(model, full_train_loader, args.stop_batch)
    tf_idf_map = calculate_cp(feature_iit, classes, full_trainset.num_class(), args.coe,
                              unlearn_class=args.forget_num)
    threshold = get_threshold_by_sparsity(tf_idf_map, cfg.REMOVE_RATIO)
    print('threshold', threshold)

    '''test before pruning'''
    list_allclasses = list(range(full_trainset.num_class()))
    # unlearn_listclass = [args.forget_num]
    list_allclasses.remove(args.forget_num)  # rest classes
    print('*' * 5 + 'testing before pruning' + '*' * 15)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    test(logger, model, forget_test_loader, loss_fn)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(logger, model, remain_test_loader, loss_fn)
    print('*' * 40)


    '''pruning'''
    t1 = time.time()
    cp_config = {"threshold": threshold, "map": tf_idf_map}
    config_list = [{
        'sparsity': cfg.REMOVE_RATIO,
        'op_types': ['Conv2d']
    }]
    pruner = TFIDFPruner(model, config_list, cp_config=cp_config)
    pruner.compress()
    t2 = time.time()
    logger.info("time used: {}".format(t2-t1))
    pruned_model_path = "{}/baseline_pruned_{}_{}_model.pth".\
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    pruned_mask_path = "{}/baseline_pruned_{}_{}_mask.pth".\
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    pruner.export_model(pruned_model_path, pruned_mask_path)
    pruned_net = copy.deepcopy(model)
    load_model_pytorch(pruned_net, pruned_model_path, model_name)

    '''test after pruning'''
    print('*' * 5 + 'testing after pruning' + '*' * 12)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    test(logger, pruned_net, forget_test_loader, loss_fn)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(logger, pruned_net, remain_test_loader, loss_fn)
    print('*' * 40)

    # return #for test

    '''fine tuning'''
    #finetune_saved_path = "{}/baseline_finetuned_{}_{}_model.pth". \
    #    format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    #train(pruned_net, epochs=args.epochs, lr=args.lr, train_loader=remain_loader,
    #      test_loader=remain_test_loader, save_path=finetune_saved_path, save_acc=args.save_acc, seed=args.seed,
    #      label_smoothing=args.label_smoothing, warmup_step=args.warmup_step, warm_lr=args.warm_lr)

    '''pruned_net.cuda()
    pruned_net.set_optimizer(cfg, args)
    pruned_net, err, epoch = fine_tune(logger, cfg, args, pruned_net, None, remain_loader, loss_fn)

    #test after fine-tuning
    print('*' * 5 + 'testing after fine-tuning' + '*' * 12)
    print('*' * 5 + 'testing in unlearn_data' + '*' * 12)
    test(logger, pruned_net, forget_test_loader, loss_fn)
    print('*' * 40)
    print('*' * 5 + 'testing in rest_data' + '*' * 15)
    test(logger, pruned_net, remain_test_loader, loss_fn)
    print('*' * 40)

    print('finished')'''


if __name__ == '__main__':
    main()
