import torch
import numpy as np
import random
import os

from train_eval import run
from net import logger_init
import argparse

parser = argparse.ArgumentParser(description='Fairness Training')

parser.add_argument('--exp-name', default="pre_exp", type=str)
parser.add_argument('--save-path', default="./Released_results/", type=str)
parser.add_argument('--dataset', default='', type=str)
parser.add_argument('--img-root', default='/data/datasets/chexpert/', type=str)

parser.add_argument('--loss', default="hinge_vr", type=str)
parser.add_argument('--total-epochs', default=60, type=int)
parser.add_argument('--num-ths-for-eva', default=1000, type=int)
parser.add_argument('--decay-scale', default=1.0, type=float)

parser.add_argument('--val-size', default=0.2, type=float)
parser.add_argument('--sampling-rate', default=0.5, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight-decay', default=1e-3, type=float)
parser.add_argument('--seed', default=42, type=int)

### loss paras
parser.add_argument('--beta', default=10, type=float)
parser.add_argument('--kappa', default=0.005, type=float)
parser.add_argument('--scaling', default=1, type=float)
parser.add_argument('--gamma', default=0.8, type=float)
parser.add_argument('--gamma_p', default=0.1, type=float)
parser.add_argument('--lamda', default=0.1, type=float) # outer sm coef
parser.add_argument('--mmt', default=0.1, type=float) #momentum
parser.add_argument('--nu', default=0.1, type=float) #moreau envelope
parser.add_argument('--outlr', default=0.1, type=float) # lr for out-loop
parser.add_argument('--in_iter', default=10, type=int) # num of inner iters
parser.add_argument('--restart', action='store_true') # num of inner iters

parser.add_argument('--th-start', default=-3, type=float)
parser.add_argument('--th-end', default=3.1, type=float)
parser.add_argument('--th-step', default=0.5, type=float)

### loss con_ex
parser.add_argument('--tau', default=10, type=float)
parser.add_argument('--mu', default=1e-3, type=float)
parser.add_argument('--theta', default=0.1, type=float)


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
if __name__ == '__main__':
    args = parser.parse_args()
    set_all_seeds(args.seed)
    
    if args.loss in ['con_ex']:
        args.save_path = args.save_path + args.exp_name + \
                                f'/{args.loss}/mu{args.mu}_tau{args.tau}_theta{args.theta}_lr{args.lr}/seed{args.seed}'
    else:
        args.save_path = args.save_path + args.exp_name + \
                                f'/{args.loss}/beta{args.beta}_gam{args.gamma}_gamP{args.gamma_p}_lr{args.lr}/seed{args.seed}'
    if not os.path.exists(args.save_path):   
        os.makedirs(args.save_path)

    logger = logger_init(args.save_path+'/log.log')
    
    logger.info(args)
    run(args, logger)