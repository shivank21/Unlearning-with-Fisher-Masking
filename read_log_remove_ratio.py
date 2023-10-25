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
    for ratio in ['0.0', '0.04', '0.08', '0.12']:
        remain_acc_history[ratio+'_fisher'] = []
        remain_acc_history[ratio+'_ti'] = []
        forget_acc_history[ratio+'_fisher'] = []
        forget_acc_history[ratio+'_ti'] = []
        for seed in seed_list:
            ckpt_dir = default_ckpt_dir + str(seed) +'/remove_ratio'

            fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_{}_model.pth". \
                format(ckpt_dir, ratio, args.forget_type, args.forget_num)
            fisher_model_mask = torch.load(fisher_mask_finetune_model_path)
            remain_acc_history[ratio+'_fisher'].append(fisher_model_mask['acc_history'])
            forget_acc_history[ratio+'_fisher'].append(fisher_model_mask['forget_acc'])

            ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_{}_model.pth". \
                format(ckpt_dir, ratio, args.forget_type, args.forget_num)
            ti_model_mask = torch.load(ti_finetune_model_path)
            remain_acc_history[ratio+'_ti'].append(ti_model_mask['acc_history'])
            forget_acc_history[ratio+'_ti'].append(ti_model_mask['forget_acc'])


    np.save("./ckpts/{}_{}/remove_ratio_remain_{}_{}.npy".format(dataset_name, model_name,
                                                         args.forget_type, args.forget_num),
            remain_acc_history)
    np.save("./ckpts/{}_{}/remove_ratio_forget_{}_{}.npy".format(dataset_name, model_name,
                                                         args.forget_type, args.forget_num),
            forget_acc_history)


if __name__ == '__main__':
    main()








