from collections import defaultdict
from model.utils import *


def get_error(output, target):
    if output.shape[1]>1:
        pred = output.argmax(dim=1, keepdim=True)
        return 1. - pred.eq(target.view_as(pred)).float().mean().item()
    else:
        pred = output.clone()
        pred[pred > 0] = 1
        pred[pred <= 0] = -1
        return 1 - pred.eq(target.view_as(pred)).float().mean().item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = defaultdict(int)
        self.avg = defaultdict(float)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, n=1, **val):
        for k in val:
            self.val[k] = val[k]
            self.sum[k] += val[k] * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]


def get_metrics(model, dataloader, loss_fn, samples_correctness=False):
    activations = []
    predictions = []
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=1, shuffle=False)
    model.eval()
    metrics = AverageMeter()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = loss_fn(output, target)
        if samples_correctness:
            activations.append(torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().squeeze())
            predictions.append(get_error(output, target))
        metrics.update(n=data.size(0), loss=loss.item(), error=get_error(output, target))
    if samples_correctness:
        return metrics.avg, np.stack(activations), np.array(predictions)
    else:
        return metrics.avg


def activations_predictions(logger, model, dataloader, name):
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics, activations, predictions = get_metrics(model, dataloader, loss_fn, True)
    logger.info(f"{name} -> Loss:{np.round(metrics['loss'], 3)}, Error:{metrics['error']}")
    return activations, predictions, metrics['loss']


def predictions_distance(logger, l1, l2, name):
    dist = np.sum(np.abs(l1 - l2))
    logger.info(f"Predictions Distance {name} -> {dist}")


def activations_distance(logger, a1, a2, name):
    dist = np.linalg.norm(a1 - a2, ord=1, axis=1).mean()
    logger.info(f"Activations Distance {name} -> {dist}")


def get_mean_var(g, p, forget_type, forget_num, num_classes, is_base_dist=False, alpha=3e-6):
    var = copy.deepcopy(1. / (g.data + 1e-8))
    var = var.clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()

    if not is_base_dist:
        mu = copy.deepcopy(p.data0.clone())
    else:
        mu = copy.deepcopy(p.data0.clone())

    if p.size(0) == num_classes and forget_type == 'class':
        mu[forget_num] = 0
        var[forget_num] = 0.0001
    if p.size(0) == num_classes:
        # Last layer
        var *= 10
    elif p.ndim == 1:
        # BatchNorm
        var *= 10
    #         var*=1
    return mu, var


def readout_retrain(cfg, model, data_loader, test_loader, loss_fn, epochs=100, threshold=0.01):
    torch.manual_seed(cfg.SEED)
    model = copy.deepcopy(model)
    model.set_optimizer(cfg)
    sampler = torch.utils.data.RandomSampler(data_loader.dataset, replacement=True,
                                             num_samples=500)
    data_loader_small = torch.utils.data.DataLoader(data_loader.dataset,
                                                    batch_size=data_loader.batch_size,
                                                    sampler=sampler,
                                                    num_workers=data_loader.num_workers)
    #optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LR, momentum=cfg.MOMENTUM,
    #                            weight_decay=cfg.WD)
    model_init = copy.deepcopy(model)
    for epoch in range(epochs):
        metrics = get_metrics(model, test_loader, loss_fn)
        if metrics['loss'] <= threshold:
            break
        model.train()
        for inputs, targets in data_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            model.optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            if hasattr(cfg, "L2_NORM") and cfg.L2_NORM:
                loss += l2_penalty(model, model_init, cfg.WD)
            loss.backward()
            model.optim.step()
        if hasattr(model, 'sched'):
            model.sched.step()
    return epoch


def all_readouts(logger, cfg, model, train_loader_full, forget_loader, loss_fn, thresh=0.1, name='method'):
    retrain_time = readout_retrain(cfg, model, train_loader_full, forget_loader,
                                   loss_fn, epochs=100, threshold=thresh)

    forget_error = get_metrics(model, forget_loader, loss_fn)['error']

    logger.info(f"{name} ->"
          f"\tForget error: {forget_error:.2%}"
          f"\tFine-tune time: {retrain_time + 1} steps")
    return dict(forget_error=forget_error, retrain_time=retrain_time)


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


import matplotlib
from matplotlib import pyplot as plt
from matplotlib import style
style.use('seaborn-whitegrid')
#matplotlib.rcParams['text.usetex'] = True
def picture_remain_acc(cfg, args, title, data):
    plt.figure(figsize=(10, 10), dpi=300)
    begin_idx = 1
    data['random_test_acc'] = data['random_test_acc'][begin_idx:]
    data['ft_test_acc'] = data['ft_test_acc'][begin_idx:]
    data['act_test_acc'] = data['act_test_acc'][begin_idx:]
    data['fisher_test_acc'] = data['fisher_test_acc'][begin_idx:]
    data['grad_test_acc'] = data['grad_test_acc'][begin_idx:]
    #plt.plot([data['retrain_remain_test_acc']] * len(data['act_test_acc']), color='grey', linestyle='--')
    #plt.plot([data['fisher_noise_test_acc']] * len(data['act_test_acc']), color='black', linestyle='--')
    plt.axhline(y=data['retrain_remain_test_acc'], color='grey', linestyle='--')
    plt.plot(data['random_test_acc'], label='RandomMask', color="lightcoral", linewidth=3.0)
    plt.plot(data['ft_test_acc'], label='Finetune', color="burlywood", linewidth=3.0)
    if not hasattr(args, "random_exper"):
        data['ti_test_acc'] = data['ti_test_acc'][begin_idx:]
        plt.axhline(y=data['fisher_noise_test_acc'], color='black', linestyle='--')
        plt.plot(data['ti_test_acc'], label='TF-IDF', color="mediumturquoise", linewidth=3.0)
    plt.plot(data['act_test_acc'], label='ActivationMask', color="mediumpurple", linewidth=3.0)
    plt.plot(data['fisher_test_acc'], label='FisherMask', color="yellowgreen", linewidth=3.0)
    plt.plot(data['grad_test_acc'], label='GradMask', color="cornflowerblue", linewidth=3.0)
    random_max = np.argmax(data['random_test_acc'])
    ft_max = np.argmax(data['ft_test_acc'])
    act_max = np.argmax(data['act_test_acc'])
    fisher_max = np.argmax(data['fisher_test_acc'])
    grad_max = np.argmax(data['grad_test_acc'])
    plt.plot(random_max, data['random_test_acc'][random_max], marker='*', color="lightcoral", markersize=15)
    plt.plot(ft_max, data['ft_test_acc'][ft_max], marker='*', color="burlywood", markersize=15)
    if not hasattr(args, "random_exper"):
        ti_max = np.argmax(data['ti_test_acc'])
        plt.plot(ti_max, data['ti_test_acc'][ti_max], marker='*', color="mediumturquoise", markersize=15)
    plt.plot(act_max, data['act_test_acc'][act_max], marker='*', color="mediumpurple", markersize=15)
    plt.plot(fisher_max, data['fisher_test_acc'][fisher_max], marker='*', color="yellowgreen", markersize=15)
    plt.plot(grad_max, data['grad_test_acc'][grad_max], marker='*', color="cornflowerblue", markersize=15)
    plt.title(title, fontsize=40)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Test Acc', fontsize=40)
    plt.tick_params(labelsize=30)
    plt.legend(fontsize=30, loc="best")
    plt.xlim(1, len(data['act_test_acc']) + 1)
    ckpt_dir = cfg.ckpt_dir if not hasattr(args, "random_exper") \
        else cfg.ckpt_dir + '/random' + str(args.forget_num)
    fig_path = "{}/test_remain_acc_{}_{}.png". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    plt.savefig(fig_path)
    plt.show()

def picture_forget_acc(cfg, args, title, data):
    plt.figure(figsize=(10, 10), dpi=300)
    begin_idx = 1
    data['random_forget_acc'] = data['random_forget_acc'][begin_idx:]
    data['ft_forget_acc'] = data['ft_forget_acc'][begin_idx:]
    if not hasattr(args, "random_exper"):
        data['ti_forget_acc'] = data['ti_forget_acc'][begin_idx:]
        plt.axhline(y=data['fisher_noise_forget_acc'], color='black', linestyle='--')
    data['act_forget_acc'] = data['act_forget_acc'][begin_idx:]
    data['fisher_forget_acc'] = data['fisher_forget_acc'][begin_idx:]
    data['grad_forget_acc'] = data['grad_forget_acc'][begin_idx:]
    #plt.plot([data['fisher_noise_forget_acc']] * len(data['act_forget_acc']), color='black', linestyle='--')
    plt.plot(data['random_forget_acc'], label='RandomMask', color="lightcoral", linewidth=3.0)
    plt.plot(data['ft_forget_acc'], label='Finetune', color="burlywood", linewidth=3.0)
    if not hasattr(args, "random_exper"):
        plt.plot(data['ti_forget_acc'], label='TF-IDF', color="mediumturquoise", linewidth=3.0)
    plt.plot(data['act_forget_acc'], label='ActivationMask', color="mediumpurple", linewidth=3.0)
    plt.plot(data['fisher_forget_acc'], label='FisherMask', color="yellowgreen", linewidth=3.0)
    plt.plot(data['grad_forget_acc'], label='GradMask', color="cornflowerblue", linewidth=3.0)
    plt.title(title, fontsize=40)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Test Acc', fontsize=40)
    plt.tick_params(labelsize=30)
    plt.legend(fontsize=30, loc="best")
    plt.xlim(1, len(data['act_forget_acc']) + 1)
    ckpt_dir = cfg.ckpt_dir if not hasattr(args, "random_exper") \
        else cfg.ckpt_dir + '/random' + str(args.forget_num)
    fig_path = "{}/test_forget_acc_{}_{}.png". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    plt.savefig(fig_path)
    plt.show()





