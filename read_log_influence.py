import argparse
from antu.io.configurators.ini_configurator import IniConfigurator
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/cifar100_resnet50.cfg')
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
        ckpt_dir = default_ckpt_dir + str(seed) +'/pic'

        finetune_model_path = "{}/finetune_baseline_{}_{}.pt". \
            format(ckpt_dir, args.forget_type, args.forget_num)
        finetune_model = torch.load(finetune_model_path)
        if 'finetune' not in remain_acc_history:
            remain_acc_history['finetune'] = [finetune_model['acc_history']]
            forget_acc_history['finetune'] = [finetune_model['forget_acc']]
        else:
            remain_acc_history['finetune'].append(finetune_model['acc_history'])
            forget_acc_history['finetune'].append(finetune_model['forget_acc'])

        influence_model_path = "{}/finetune_influence_{}_{}.pt". \
            format(ckpt_dir, args.forget_type, args.forget_num)
        influence_model = torch.load(influence_model_path)
        if 'influence' not in remain_acc_history:
            remain_acc_history['influence'] = [influence_model['acc_history']]
            forget_acc_history['influence'] = [influence_model['forget_acc']]
        else:
            remain_acc_history['influence'].append(influence_model['acc_history'])
            forget_acc_history['influence'].append(influence_model['forget_acc'])

        fisher_mask_path = "{}/finetuned_fisher_mask_pruned_{}_{}_model.pth". \
            format(ckpt_dir, args.forget_type, args.forget_num)
        fisher_model = torch.load(fisher_mask_path)
        if 'fisher' not in remain_acc_history:
            remain_acc_history['fisher'] = [fisher_model['acc_history']]
            forget_acc_history['fisher'] = [fisher_model['forget_acc']]
        else:
            remain_acc_history['fisher'].append(fisher_model['acc_history'])
            forget_acc_history['fisher'].append(fisher_model['forget_acc'])

    np.save("./ckpts/{}_{}/pic_influence_remain_{}_{}.npy".format(dataset_name, model_name,
                                                         args.forget_type, args.forget_num),
            remain_acc_history)
    np.save("./ckpts/{}_{}/pic_influence_forget_{}_{}.npy".format(dataset_name, model_name,
                                                         args.forget_type, args.forget_num),
            forget_acc_history)


if __name__ == '__main__':
    main()








