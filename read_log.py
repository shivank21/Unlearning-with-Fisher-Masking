import argparse
from antu.io.configurators.ini_configurator import IniConfigurator
from utils import *


parser = argparse.ArgumentParser(description="Usage for image classification.")
parser.add_argument('--config', type=str, help="Path to config file.",
                    default='config/cifar10_resnet20.cfg')
parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default="class")
parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                        "forget num if forget type is random", default=0)
args, extra_args = parser.parse_known_args()
cfg = IniConfigurator(args.config, extra_args)
if not os.path.exists(cfg.ckpt_dir):
    os.makedirs(cfg.ckpt_dir)

attribute = args.config[7:-4].split("_")
dataset_name, model_name = attribute[0], attribute[1]

default_ckpt_dir = "./ckpts/{}_{}/seed_".format(dataset_name, model_name)
seed_list = [666, 777, 888]

remain_acc_history, forget_acc_history = {}, {}
for seed in seed_list:
    ckpt_dir = default_ckpt_dir + str(seed)

    ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_model.pth". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    ti_model = torch.load(ti_finetune_model_path)
    if "tf-idf" not in remain_acc_history:
        remain_acc_history['tf-idf'] = [ti_model['acc_history']]
        forget_acc_history['tf-idf'] = [ti_model['forget_acc']]
    else:
        remain_acc_history['tf-idf'].append(ti_model['acc_history'])
        forget_acc_history['tf-idf'].append(ti_model['forget_acc'])
    if "retrain" not in remain_acc_history:
        remain_acc_history['retrain'] = [ti_model['retrain_test_acc']]
    else:
        remain_acc_history['retrain'].append(ti_model['retrain_test_acc'])

    random_finetune_model_path = "{}/finetune_random_pruned_{}_{}_model.pth". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    random_model = torch.load(random_finetune_model_path)
    if 'random' not in remain_acc_history:
        remain_acc_history['random'] = [random_model['acc_history']]
        forget_acc_history['random'] = [random_model['forget_acc']]
    else:
        remain_acc_history['random'].append(random_model['acc_history'])
        forget_acc_history['random'].append(random_model['forget_acc'])

    act_mask_finetune_model_path = "{}/finetuned_mask_pruned_{}_{}_model.pth". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    act_model_mask = torch.load(act_mask_finetune_model_path)
    if 'activation' not in remain_acc_history:
        remain_acc_history['activation'] = [act_model_mask['acc_history']]
        forget_acc_history['activation'] = [act_model_mask['forget_acc']]
    else:
        remain_acc_history['activation'].append(act_model_mask['acc_history'])
        forget_acc_history['activation'].append(act_model_mask['forget_acc'])

    fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_model.pth". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    fisher_model_mask = torch.load(fisher_mask_finetune_model_path)
    if 'fisher_mask' not in remain_acc_history:
        remain_acc_history['fisher_mask'] = [fisher_model_mask['acc_history']]
        forget_acc_history['fisher_mask'] = [fisher_model_mask['forget_acc']]
    else:
        remain_acc_history['fisher_mask'].append(fisher_model_mask['acc_history'])
        forget_acc_history['fisher_mask'].append(fisher_model_mask['forget_acc'])

    grad_mask_finetune_model_path = "{}/finetuned_grad_mask_pruned_{}_{}_model.pth". \
        format(ckpt_dir, args.forget_type, args.forget_num)
    grad_model_mask = torch.load(grad_mask_finetune_model_path)
    if 'gradient' not in remain_acc_history:
        remain_acc_history['gradient'] = [grad_model_mask['acc_history']]
        forget_acc_history['gradient'] = [grad_model_mask['forget_acc']]
    else:
        remain_acc_history['gradient'].append(grad_model_mask['acc_history'])
        forget_acc_history['gradient'].append(grad_model_mask['forget_acc'])

    modelf_file = "{}/fisher_baseline_{}_{}_model.pt".format(ckpt_dir, args.forget_type,
                                                             args.forget_num)
    modelf = torch.load(modelf_file)
    if 'fisher_noise' not in remain_acc_history:
        remain_acc_history['fisher_noise'] = [modelf['acc_history']]
        forget_acc_history['fisher_noise'] = [modelf['forget_acc']]
    else:
        remain_acc_history['fisher_noise'].append(modelf['acc_history'])
        forget_acc_history['fisher_noise'].append(modelf['forget_acc'])

    finetune_file = "{}/finetune_baseline_{}_{}.pt".format(ckpt_dir, args.forget_type,
                                                           args.forget_num)
    model_ft = torch.load(finetune_file)
    if 'finetune' not in remain_acc_history:
        remain_acc_history['finetune'] = [model_ft['acc_history']]
        forget_acc_history['finetune'] = [model_ft['forget_acc']]
    else:
        remain_acc_history['finetune'].append(model_ft['acc_history'])
        forget_acc_history['finetune'].append(model_ft['forget_acc'])

np.save("./ckpts/{}_{}/dict_remain_{}_{}.npy".format(dataset_name, model_name,
                                              args.forget_type, args.forget_num),
        remain_acc_history)
np.save("./ckpts/{}_{}/dict_forget_{}_{}.npy".format(dataset_name, model_name,
                                              args.forget_type, args.forget_num),
        forget_acc_history)









