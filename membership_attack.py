import argparse, os, random, copy
import numpy as np
from antu.io.configurators.ini_configurator import IniConfigurator
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, random_split
from datasets.utils import get_dataset
from model.utils import get_model
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def set_seed(seed):
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AttackModel(nn.Module):
    def __init__(self, hidden_size):
      super(AttackModel, self).__init__()
      self.fc1 = nn.Linear(hidden_size, 256)
      self.fc2 = nn.Linear(256, 128)
      self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.dropout(x, training=self.training)
      x = F.relu(self.fc2(x))
      x = F.dropout(x, training=self.training)
      x = self.fc3(x)
      return torch.sigmoid(x)

# Testing method for attack that returns full confusion matrix
def fulltestattacker(model, loader):
    model.eval()
    with torch.no_grad():
      tp = 0
      tn = 0
      fp = 0
      fn = 0
      for data, target in loader:
        data = data.cuda()
        output = model(data)
        output = torch.flatten(output)
        pred = torch.round(output)
        for i in range(len(pred)):
          if pred[i] == target[i] == 1:
              tp += 1
          if pred[i] == target[i] == 0:
              tn += 1
          if pred[i] == 1 and target[i] == 0:
              fp += 1
          if pred[i] == 0 and target[i] == 1:
              fn += 1
    P = tp*1.0/(tp+fp) if tp+fp else 0
    R = tp*1.0/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    return tp, tn, fp, fn, P, R, F1


def test(model, data_loader, loss_fn, cuda=True):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets, reduction='sum')

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    print("acc is {}".format(acc))


def get_attack_dataset(model, data_loader, batch_size):
    attack_x = []
    attack_y = []
    sm = nn.Softmax()
    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            pred = model(data)
            attack_x.append(sm(pred))
            attack_y.extend([1]*data.size()[0])
    tensor_x = torch.cat(attack_x, 0)
    tensor_y = torch.Tensor(attack_y)
    attack_datasets = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attackloader = torch.utils.data.DataLoader(attack_datasets,
                                               batch_size=batch_size,
                                               shuffle=True)
    return attackloader


def load_model_and_attack(cfg, args, model_name, num_class, model_file, test_loader, attack_model, loss_fn, prefix):
    model = get_model(model_name, num_classes=num_class, n_channels=cfg.IN_CHANNEL)
    model.cuda()
    model.load_state_dict(torch.load(model_file)['model'])
    print("test load {} model, ".format(prefix), end='')
    test(model, test_loader, loss_fn)
    attack_loader = get_attack_dataset(model, test_loader, args.batch_size)
    print("attack result is: ", end='')
    print(fulltestattacker(attack_model, attack_loader))

def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/cifar10_resnet20.cfg')
    parser.add_argument('--ckpt', type=str, help="Path to save model for membership attack.",
                        default='ckpts/membership_attack')
    parser.add_argument('--forget_class', type=int, help="Forget class index", default=0)
    parser.add_argument('--no_train', action='store_true', help="Whether or not to train model.",
                        default=False)
    parser.add_argument('--finetune_epochs', type=int, help="finetune_epochs", default=30)
    parser.add_argument('--shadow_nums', type=int, help="number of shadow models", default=20)
    parser.add_argument('--batch_size', type=int, help="number of batch size for attack model",
                        default=128)

    args, extra_args = parser.parse_known_args()
    cfg = IniConfigurator(args.config, extra_args)
    if not os.path.exists(args.ckpt+'/'+args.config[7:-4]):
        os.makedirs(args.ckpt+'/'+args.config[7:-4])

    set_seed(cfg.SEED)

    attribute = args.config[7:-4].split("_")
    dataset_name, model_name = attribute[0], attribute[1]

    print('==> Preparing data..')
    full_trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                           train=True, download=True)

    loss_fn = F.cross_entropy

    if not args.no_train:
        # We need to train each shadow model on the in_data for that model
        # Create shadow datasets.
        shadow_datasets = []
        for i in range(args.shadow_nums):
            shadow_datasets.append(random_split(full_trainset,
                                                [int(len(full_trainset) / 20 * 9),
                                                int(len(full_trainset) / 2) - int(len(full_trainset) / 20 * 9),
                                                int(len(full_trainset) / 2)]))

        # Create shadow models.
        shadow_models = []
        for i in range(args.shadow_nums):
            shadow_models.append(get_model(model_name, num_classes=full_trainset.num_class(),
                                           n_channels=cfg.IN_CHANNEL))

        '''for i, shadow_model in enumerate(shadow_models):
            last_model = "{}/{}_{}/last_model_{}.pt".format(args.ckpt, dataset_name, model_name, i)
            best_model = "{}/{}_{}/best_model_{}.pt".format(args.ckpt, dataset_name, model_name, i)
            shadow_model.cuda()
            shadow_model.set_optimizer(cfg)
            in_loader = DataLoader(shadow_datasets[i][0], batch_size=cfg.N_BATCH, shuffle=True,
                                   num_workers=cfg.N_WORKER)
            dev_loader = DataLoader(shadow_datasets[i][1], batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER)
            print(f"Training shadow model {i}")
            best_acc, best_epoch = 0, 0
            for j in range(cfg.N_EPOCH):
                shadow_model.train()
                for batch_idx, (inputs, targets) in enumerate(in_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    shadow_model.optim.zero_grad()
                    outputs = shadow_model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    shadow_model.optim.step()

                torch.save({
                    'epoch': j,
                    'model': shadow_model.state_dict(),
                    'best': (best_acc, best_epoch),
                    'optim': shadow_model.optim.state_dict(),
                    'sched': shadow_model.sched.state_dict()
                    if hasattr(shadow_model, 'sched') else {},
                }, last_model)

                shadow_model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(dev_loader):
                        inputs, targets = inputs.cuda(), targets.cuda()
                        outputs = shadow_model(inputs)
                        loss = loss_fn(outputs, targets, reduction='sum')

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                acc = 100.0 * correct / total

                if hasattr(shadow_model, 'sched'):
                    shadow_model.sched.step()

                if acc > best_acc:
                    best_acc, best_epoch = acc, j
                    os.popen(f'cp {last_model} {best_model}')
            print("{} train finish, best epoch: {}, best acc: {:.3f}%".format(i, best_epoch, best_acc))'''

        # Create attack model training sets
        sm = nn.Softmax()
        attack_x = []
        attack_y = []
        dev_loader = DataLoader(shadow_datasets[0][1], batch_size=cfg.N_BATCH, shuffle=False,
                                num_workers=cfg.N_WORKER)
        with torch.no_grad():
            # Generate attack training set for current class
            for i, shadow_model in enumerate(shadow_models):
                shadow_model.load_state_dict(
                    torch.load("{}/{}_{}/best_model_{}.pt".
                               format(args.ckpt, dataset_name, model_name, i))['model'])
                test(shadow_model, dev_loader, loss_fn, cuda=False)
                print(f"\rGenerating class {args.forget_class} set from model {i}")
                in_loader = torch.utils.data.DataLoader(shadow_datasets[i][0], batch_size=1,
                                                        num_workers=cfg.N_WORKER)
                for data, target in in_loader:
                    if target == args.forget_class:
                        pred = shadow_model(data)
                        if torch.argmax(pred).item() == args.forget_class:
                            attack_x.append(sm(pred))
                            attack_y.append(1)
                out_loader = torch.utils.data.DataLoader(shadow_datasets[i][2], batch_size=1,
                                                         num_workers=cfg.N_WORKER)
                for data, target in out_loader:
                    if target == args.forget_class:
                        pred = shadow_model(data)
                        attack_x.append(sm(pred))
                        attack_y.append(0)
                print("class 0 : {}, class 1 : {}".format(len(attack_y)-sum(attack_y), sum(attack_y)))


        # Save datasets
        tensor_x = torch.stack(attack_x)
        tensor_y = torch.Tensor(attack_y)
        xpath = "{}/{}_{}/attack_x_{}.pt".format(args.ckpt, dataset_name,
                                                         model_name, args.forget_class)
        ypath = "{}/{}_{}/attack_y_{}.pt".format(args.ckpt, dataset_name,
                                                        model_name, args.forget_class)
        torch.save(tensor_x, xpath)
        torch.save(tensor_y, ypath)

        xpath = "{}/{}_{}/attack_x_{}.pt".format(args.ckpt, dataset_name,
                                                        model_name, args.forget_class)
        ypath = "{}/{}_{}/attack_y_{}.pt".format(args.ckpt, dataset_name,
                                                        model_name, args.forget_class)
        tensor_x = torch.load(xpath)
        tensor_x = tensor_x.squeeze(1)
        tensor_y = torch.load(ypath)
        print("data size: {}, {}".format(tensor_x.size(), tensor_y.size()))

        # Create test and train dataloaders for attack dataset
        attack_datasets = []
        attack_datasets.append(torch.utils.data.TensorDataset(tensor_x, tensor_y))
        attack_train, attack_test = torch.utils.data.random_split(
            attack_datasets[0], [int(0.9 * len(attack_datasets[0])),
                                 len(attack_datasets[0]) - int(0.9 * len(attack_datasets[0]))])
        attackloader = torch.utils.data.DataLoader(attack_train, batch_size=args.batch_size, shuffle=True)
        attacktester = torch.utils.data.DataLoader(attack_test, batch_size=args.batch_size, shuffle=True)

        hidden_size = tensor_x.size()[1]
        # Create and train an attack model
        attack_model = AttackModel(hidden_size=hidden_size)
        attack_model.cuda()
        if dataset_name == 'mnist':
            attack_optimizer = SGD(attack_model.parameters(), lr=0.0001)
        else:
            attack_optimizer = Adam(attack_model.parameters())

        for epoch in range(10):
            attack_model.train()
            for batch_idx, (data, target) in enumerate(attackloader):
                data, target = data.cuda(), target.cuda()
                attack_optimizer.zero_grad()
                output = attack_model(data)
                output = torch.flatten(output)
                loss = F.binary_cross_entropy(output, target)
                loss.backward()
                attack_optimizer.step()
                if batch_idx % 64 == 0:
                    print("\rEpoch: {} [{:6d}]\tLoss: {:.6f}".format(
                        epoch, batch_idx * len(data), loss.item()))

        print(fulltestattacker(attack_model, attacktester))
        # Save attack model
        path = "{}/{}_{}/attack_model_{}.pt".format(args.ckpt, dataset_name,
                                                model_name, args.forget_class)
        torch.save({
            'model_state_dict': attack_model.state_dict(),
        }, path)

    '''set_seed(cfg.SEED+1)
    best_model = "{}/{}_{}/best_model_target.pt".format(args.ckpt, dataset_name, model_name)
    if not os.path.exists(best_model):
        target_in, target_dev, target_out = random_split(full_trainset, [int(len(full_trainset) / 20 * 9),
                                            int(len(full_trainset) / 2) - int(len(full_trainset) / 20 * 9),
                                            int(len(full_trainset) / 2)])
        target_model = get_model(model_name, num_classes=full_trainset.num_class(),
                  n_channels=cfg.IN_CHANNEL)

        last_model = "{}/{}_{}/last_model_target.pt".format(args.ckpt, dataset_name, model_name)

        target_model.cuda()
        target_model.set_optimizer(cfg)

        in_loader = DataLoader(target_in, batch_size=cfg.N_BATCH, shuffle=True,
                               num_workers=cfg.N_WORKER)
        dev_loader = DataLoader(target_dev, batch_size=cfg.N_BATCH, shuffle=False,
                                num_workers=cfg.N_WORKER)
        print(f"Training target model")
        best_acc, best_epoch = 0, 0
        for j in range(cfg.N_EPOCH):
            target_model.train()
            for batch_idx, (inputs, targets) in enumerate(in_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                target_model.optim.zero_grad()
                outputs = target_model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                target_model.optim.step()

            torch.save({
                'epoch': j,
                'model': target_model.state_dict(),
                'best': (best_acc, best_epoch),
                'optim': target_model.optim.state_dict(),
                'sched': target_model.sched.state_dict()
                if hasattr(target_model, 'sched') else {},
            }, last_model)

            target_model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dev_loader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = target_model(inputs)
                    loss = loss_fn(outputs, targets, reduction='sum')

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total

            if hasattr(target_model, 'sched'):
                target_model.sched.step()

            if acc > best_acc:
                best_acc, best_epoch = acc, j
                os.popen(f'cp {last_model} {best_model}')
        print("target model train finish, best epoch: {}, best acc: {:.3f}%".format(best_epoch, best_acc))
    else:
        target_model = get_model(model_name, num_classes=full_trainset.num_class(),
                                 n_channels=cfg.IN_CHANNEL)
        target_model.load_state_dict(torch.load(best_model)['model'])
        target_model.cuda()'''

    set_seed(cfg.SEED)
    forget_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=True, download=True,
                                 forget_type="class", forget_num=args.forget_class,
                                 only_forget=True)
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.N_BATCH, shuffle=True,
                               num_workers=cfg.N_WORKER)

    attack_model = AttackModel(hidden_size=full_trainset.num_class())
    path = "{}/{}_{}/attack_model_{}.pt".format(args.ckpt, dataset_name,
                                                       model_name, args.forget_class)
    attack_model.load_state_dict(torch.load(path)['model_state_dict'])
    attack_model.cuda()

    # test origin model
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          cfg.BEST, forget_loader, attack_model, loss_fn, "original")

    # test finetune model
    finetune_file = "{}/finetune_baseline_class_{}.pt".format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          finetune_file, forget_loader, attack_model, loss_fn, "finetune")

    # test tf-idf model
    ti_finetune_model_path = "{}/finetune_baseline_pruned_class_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          ti_finetune_model_path, forget_loader, attack_model, loss_fn, "tf-idf")

    # test random-mask model
    random_finetune_model_path = "{}/finetune_random_pruned_class_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          random_finetune_model_path, forget_loader, attack_model, loss_fn, "random-mask")

    # test activation model
    act_mask_finetune_model_path = "{}/finetuned_mask_pruned_class_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          act_mask_finetune_model_path, forget_loader, attack_model, loss_fn, "activation-mask")

    # test fisher-mask model
    fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_class_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          fisher_mask_finetune_model_path, forget_loader, attack_model, loss_fn, "fisher-mask")

    # test grad-mask model
    grad_mask_finetune_model_path = "{}/finetuned_grad_mask_pruned_class_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          grad_mask_finetune_model_path, forget_loader, attack_model, loss_fn, "grad-mask")

    # test fisher-noise model
    modelf_file = "{}/fisher_baseline_class_{}_model.pt".format(cfg.ckpt_dir, args.forget_class)
    load_model_and_attack(cfg, args, model_name, full_trainset.num_class(),
                          modelf_file, forget_loader, attack_model, loss_fn, "fisher-noise")





























if __name__ == '__main__':
    main()
