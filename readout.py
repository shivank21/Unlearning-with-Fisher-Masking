import argparse, os, random, copy
import numpy as np
from antu.io.configurators.ini_configurator import IniConfigurator
from antu.utils.dual_channel_logger import dual_channel_logger
from torch.utils.data import DataLoader, random_split
from datasets.utils import get_dataset
from model.utils import get_model
from utils import *
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


def load_model(cfg, model_name, num_class, model_file, prefix):
    model = get_model(model_name, num_classes=num_class, n_channels=cfg.IN_CHANNEL)
    model.cuda()
    model.load_state_dict(torch.load(model_file)['model'])
    return model


def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for image classification.")
    parser.add_argument('--config', type=str, help="Path to config file.",
                        default='config/cifar10_resnet20.cfg')
    parser.add_argument('--ckpt', type=str, help="Path to save model for membership attack.",
                        default='ckpts/membership_attack')
    parser.add_argument('--forget_type', type=str, help="Forget what kind of data, class or random",
                        default=None)
    parser.add_argument('--forget_num', type=int, help="Forget class index if forget_type is class, "
                                                       "forget num if forget type is random", default=0)
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

    log_file = "{}/readout_{}_{}_log".format(cfg.ckpt_dir, args.forget_type,
                                                      args.forget_num)
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
    full_trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                           train=True, download=True)

    loss_fn = F.cross_entropy

    set_seed(cfg.SEED)
    remain_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                      train=False, download=True,
                                      forget_type=args.forget_type, forget_num=args.forget_num)
    remain_test_loader = DataLoader(remain_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER)

    forget_test_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                      train=False, download=True,
                                      forget_type=args.forget_type, forget_num=args.forget_num,
                                      only_forget=True)
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=cfg.N_BATCH, shuffle=False,
                                    num_workers=cfg.N_WORKER)


    # test origin model
    model = load_model(cfg, model_name, full_trainset.num_class(), cfg.BEST, "original")

    # test re-trained model
    model0_file = "{}/forget_{}_{}_best.pt".format(cfg.ckpt_dir, args.forget_type,
                                     args.forget_num)
    model0 = load_model(cfg, model_name, full_trainset.num_class(), model0_file, "re-train")

    # test finetune model
    finetune_file = "{}/finetune_baseline_{}_{}.pt".format(cfg.ckpt_dir, args.forget_type,
                                                           args.forget_num)
    model_ft = load_model(cfg, model_name, full_trainset.num_class(), finetune_file, "finetune")

    # test tf-idf model
    ti_finetune_model_path = "{}/finetune_baseline_pruned_{}_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    model_ti = load_model(cfg, model_name, full_trainset.num_class(), ti_finetune_model_path,
                          "tf-idf")

    # test random-mask model
    random_finetune_model_path = "{}/finetune_random_pruned_{}_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    model_random = load_model(cfg, model_name, full_trainset.num_class(), random_finetune_model_path,
               "random-mask")

    # test activation model
    act_mask_finetune_model_path = "{}/finetuned_mask_pruned_{}_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    model_act = load_model(cfg, model_name, full_trainset.num_class(), act_mask_finetune_model_path,
               "activation-mask")

    # test fisher-mask model
    fisher_mask_finetune_model_path = "{}/finetuned_fisher_mask_pruned_{}_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    model_fisher = load_model(cfg, model_name, full_trainset.num_class(), fisher_mask_finetune_model_path,
               "fisher-mask")

    # test grad-mask model
    grad_mask_finetune_model_path = "{}/finetuned_grad_mask_pruned_{}_{}_model.pth". \
        format(cfg.ckpt_dir, args.forget_type, args.forget_num)
    model_grad = load_model(cfg, model_name, full_trainset.num_class(), grad_mask_finetune_model_path,
               "grad-mask")

    # test fisher-noise model
    modelf_file = "{}/fisher_baseline_{}_{}_model.pt".format(cfg.ckpt_dir, args.forget_type,
                                                               args.forget_num)
    modelf = load_model(cfg, model_name, full_trainset.num_class(), modelf_file, "fisher-noise")

    m0_D_r_activations, m0_D_r_predictions, _ = activations_predictions(logger, model0,
                                                                        remain_test_loader,
                                                                        'Retrain_Model_D_r')
    m0_D_f_activations, m0_D_f_predictions, _ = activations_predictions(logger, model0,
                                                                        forget_test_loader,
                                                                        'Retrain_Model_D_f')

    m_D_r_activations, m_D_r_predictions, _ = activations_predictions(logger, model,
                                                                      remain_test_loader,
                                                                      'Origin_Model_D_r')
    m_D_f_activations, m_D_f_predictions, _ = activations_predictions(logger, model,
                                                                        forget_test_loader,
                                                                        'Origin_Model_D_f')

    ft_D_r_activations, ft_D_r_predictions, _ = activations_predictions(logger, model_ft,
                                                                        remain_test_loader,
                                                                        'Finetune_Model_D_r')
    ft_D_f_activations, ft_D_f_predictions, _ = activations_predictions(logger, model_ft,
                                                                        forget_test_loader,
                                                                        'Finetune_Model_D_f')

    random_D_r_activations, random_D_r_predictions, _ = activations_predictions(logger, model_random,
                                                                        remain_test_loader,
                                                                        'Random_Model_D_r')
    random_D_f_activations, random_D_f_predictions, _ = activations_predictions(logger, model_random,
                                                                        forget_test_loader,
                                                                        'Random_Model_D_f')

    ti_D_r_activations, ti_D_r_predictions, _ = activations_predictions(logger, model_ti,
                                                                        remain_test_loader,
                                                                        'TF-IDF_Model_D_r')
    ti_D_f_activations, ti_D_f_predictions, _ = activations_predictions(logger, model_ti,
                                                                        forget_test_loader,
                                                                        'TF-IDF_Model_D_f')

    fisher_D_r_activations, fisher_D_r_predictions, _ = activations_predictions(logger, modelf,
                                                                                remain_test_loader,
                                                                                'Fisher_D_r')
    fisher_D_f_activations, fisher_D_f_predictions, _ = activations_predictions(logger, modelf,
                                                                                forget_test_loader,
                                                                                'Fisher_D_f')

    actmask_D_r_activations, actmask_D_r_predictions, _ = activations_predictions(logger, model_act,
                                                                            remain_test_loader,
                                                                            'ActMask_D_r')
    actmask_D_f_activations, actmask_D_f_predictions, _ = activations_predictions(logger, model_act,
                                                                            forget_test_loader,
                                                                            'ActMask_D_f')

    fishermask_D_r_activations, fishermask_D_r_predictions, _ = activations_predictions(logger, model_fisher,
                                                                                  remain_test_loader,
                                                                                  'FisherMask_D_r')
    fishermask_D_f_activations, fishermask_D_f_predictions, _ = activations_predictions(logger, model_fisher,
                                                                                  forget_test_loader,
                                                                                  'FisherMask_D_f')

    gradmask_D_r_activations, gradmask_D_r_predictions, _ = activations_predictions(logger, model_grad,
                                                                                        remain_test_loader,
                                                                                        'GradMask_D_r')
    gradmask_D_f_activations, gradmask_D_f_predictions, _ = activations_predictions(logger, model_grad,
                                                                                        forget_test_loader,
                                                                                        'GradMask_D_f')

    predictions_distance(logger, m0_D_f_predictions, m_D_f_predictions, 'Retrain_Origin_D_f')
    activations_distance(logger, m0_D_f_activations, m_D_f_activations, 'Retrain_Origin_D_f')
    activations_distance(logger, m0_D_r_activations, m_D_r_activations, 'Retrain_Origin_D_r')

    predictions_distance(logger, m0_D_f_predictions, ft_D_f_predictions, 'Retrain_Finetune_D_f')
    activations_distance(logger, m0_D_f_activations, ft_D_f_activations, 'Retrain_Finetune_D_f')
    activations_distance(logger, m0_D_r_activations, ft_D_r_activations, 'Retrain_Finetune_D_r')

    predictions_distance(logger, m0_D_f_predictions, random_D_f_predictions, 'Retrain_Random_D_f')
    activations_distance(logger, m0_D_f_activations, random_D_f_activations, 'Retrain_Random_D_f')
    activations_distance(logger, m0_D_r_activations, random_D_r_activations, 'Retrain_Random_D_r')

    predictions_distance(logger, m0_D_f_predictions, ti_D_f_predictions, 'Retrain_TF-IDF_D_f')
    activations_distance(logger, m0_D_f_activations, ti_D_f_activations, 'Retrain_TF-IDF_D_f')
    activations_distance(logger, m0_D_r_activations, ti_D_r_activations, 'Retrain_TF-IDF_D_r')

    predictions_distance(logger, m0_D_f_predictions, fisher_D_f_predictions, 'Retrain_FisherNoise_D_f')
    activations_distance(logger, m0_D_f_activations, fisher_D_f_activations, 'Retrain_FisherNoise_D_f')
    activations_distance(logger, m0_D_r_activations, fisher_D_r_activations, 'Retrain_FisherNoise_D_r')

    predictions_distance(logger, m0_D_f_predictions, actmask_D_f_predictions, 'Retrain_actmask_D_f')
    activations_distance(logger, m0_D_f_activations, actmask_D_f_activations, 'Retrain_actmask_D_f')
    activations_distance(logger, m0_D_r_activations, actmask_D_r_activations, 'Retrain_actmask_D_r')

    predictions_distance(logger, m0_D_f_predictions, gradmask_D_f_predictions, 'Retrain_gradmask_D_f')
    activations_distance(logger, m0_D_f_activations, gradmask_D_f_activations, 'Retrain_gradmask_D_f')
    activations_distance(logger, m0_D_r_activations, gradmask_D_r_activations, 'Retrain_gradmask_D_r')

    predictions_distance(logger, m0_D_f_predictions, fishermask_D_f_predictions, 'Retrain_fishermask_D_f')
    activations_distance(logger, m0_D_f_activations, fishermask_D_f_activations, 'Retrain_fishermask_D_f')
    activations_distance(logger, m0_D_r_activations, fishermask_D_r_activations, 'Retrain_fishermask_D_r')

    forget_dataset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                                 train=True, download=True, forget_type=args.forget_type,
                                 forget_num=args.forget_num, only_forget=True)
    forget_loader = DataLoader(forget_dataset, batch_size=cfg.N_BATCH, shuffle=True,
                               num_workers=cfg.N_WORKER)
    full_trainset = get_dataset(dataset_name, root=os.path.join(cfg.data_dir, cfg.DATASET),
                           train=True, download=True)
    full_train_loader = DataLoader(full_trainset, batch_size=cfg.N_BATCH, shuffle=True,
                              num_workers=cfg.N_WORKER)



    readouts = {}
    _, _, Origin_model_Df_loss = activations_predictions(logger, model, forget_loader,
                                                            'Origin_D_f')
    thresh = Origin_model_Df_loss + 1e-5
    readouts["a"] = all_readouts(logger, cfg, copy.deepcopy(model), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'Origin')
    readouts["b"] = all_readouts(logger, cfg, copy.deepcopy(model0), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'Retrain')
    readouts["c"] = all_readouts(logger, cfg, copy.deepcopy(model_ft), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'Finetune')
    readouts["d"] = all_readouts(logger, cfg, copy.deepcopy(model_random), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'Random')
    readouts["e"] = all_readouts(logger, cfg, copy.deepcopy(model_ti), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'TF-IDF')
    readouts["f"] = all_readouts(logger, cfg, copy.deepcopy(modelf), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'FisherNoise')
    readouts["g"] = all_readouts(logger, cfg, copy.deepcopy(model_act), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'ActMask')
    readouts["h"] = all_readouts(logger, cfg, copy.deepcopy(model_grad), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'GradMask')
    readouts["I"] = all_readouts(logger, cfg, copy.deepcopy(model_fisher), full_train_loader,
                                 forget_loader, loss_fn, thresh, 'FisherMask')



























if __name__ == '__main__':
    main()
