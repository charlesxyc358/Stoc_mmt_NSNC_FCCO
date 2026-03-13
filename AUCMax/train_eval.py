import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import TensorDataset, DataLoader


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from libauc.sampler import DualSampler
import matplotlib.pyplot as plt

from net import TwoLayerNN
from load_data import load_db_by_name
from loss import  AUC_Loss, AUC_ROC_Penalty, AUC_ROC_Penalty_Hinge_VR, AUC_ROC_Penalty_SH, ConEx, AUC_ROC_Penalty_smHinge_VR
from optimizer import MEAdamW

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def run(args, logger):

    # # HyperParameters
    # batch_size = 128
    # total_epochs = 60
    # decay_epochs = [30, 45]
    # weight_decay = 0.001
    # val_size = 0.2
    # sampling_rate = 0.5
    # log_dir = './Released_results/'

    ## data
    train_data, test_data = load_db_by_name(args.dataset)
    X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
                train_data[0], train_data[1], train_data[2], test_size=args.val_size,
                random_state=42)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                torch.tensor(y_train, dtype=torch.long), 
                                torch.tensor(z_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                torch.tensor(y_val), 
                                torch.tensor(z_val))
    test_dataset = TensorDataset(torch.tensor(test_data[0], dtype=torch.float32), 
                                torch.tensor(test_data[1]), 
                                torch.tensor(test_data[2]))
    
    # dataloaders
    sampler = DualSampler(train_dataset, args.batch_size, labels=y_train, sampling_rate=args.sampling_rate)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    trainVal_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    ### model initialization
    input_size = X_train.shape[1]
    model = TwoLayerNN(input_size=input_size, hidden_size=input_size, output_size=1)
    model_b = copy.deepcopy(model)
    
    model = model.cuda()
    model_b = model_b.cuda()

    if args.loss in ['hinge_vr']:
        loss_fn = AUC_ROC_Penalty_Hinge_VR(beta=args.beta, gamma=args.gamma, gamma_p=args.gamma_p,
                                scaling=args.scaling, kappa=args.kappa, 
                                ths=torch.arange(args.th_start, args.th_end, args.th_step))
    elif args.loss in ['squared_hinge']:
        loss_fn = AUC_ROC_Penalty_SH(beta=args.beta, gamma=args.gamma,
                                scaling=args.scaling, kappa=args.kappa, 
                                ths=torch.arange(args.th_start, args.th_end, args.th_step))
    elif args.loss in ['con_ex']:
        loss_fn = ConEx(model.parameters(), tau=args.tau, mu=args.mu, theta = args.theta,
                                scaling=args.scaling, kappa=args.kappa, 
                                ths=torch.arange(args.th_start, args.th_end, args.th_step))
    elif args.loss in ['alexr2']:
        loss_fn = AUC_ROC_Penalty_smHinge_VR(beta=args.beta, gamma=args.gamma, gamma_p=args.gamma_p,
                                scaling=args.scaling, kappa=args.kappa, lamda=args.lamda,
                                ths=torch.arange(args.th_start, args.th_end, args.th_step))
    else:
        loss_fn = AUC_ROC_Penalty(beta=args.beta, gamma=args.gamma, scaling=args.scaling, 
                                kappa=args.kappa, 
                                ths=torch.arange(args.th_start,args.th_end,args.th_step))
        
    if args.loss in ['con_ex', 'hinge_vr']: #
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.loss == 'alexr2':
        optimizer = MEAdamW(model.parameters(), lr=args.outlr, lr_prox = args.lr, eps=1e-2, \
                         weight_decay=args.weight_decay, betas= (1-args.mmt, 0.999), mor_coef=args.nu, restart=args.restart) #
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    #### train
    logger.info('Start Training')
    logger.info('-'*30)

    train_log = []
    test_log = []
    best_test = 999
    best_val = 999
    stats= []
    ada_lr_lb, ada_lr_ub = float('inf'), 0.0
    
    #### initial model 
    # torch.save({'model':model.state_dict(), 'epoch':-1}, args.save_path+f'/epoch_-1.ckpt')
    for epoch in range(args.total_epochs):
        if epoch in [int(args.total_epochs*0.5), int(args.total_epochs*0.75)]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1 * param_group['lr']
        if args.loss in ['con_ex'] and epoch and  (epoch)%10==0:
            loss_fn.update_ref() 

        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        
        train_loss = []
        model.train()
        model_b.train()
        print("length of train_loader: ", len(train_loader))
        # for X_batch, y_batch, z_batch in train_loader:
        for i, (X_batch, y_batch, z_batch) in enumerate(train_loader):
            X_batch, y_batch, z_batch  = X_batch.cuda(), y_batch.cuda(), z_batch.cuda()
            
            if args.loss in ['hinge_vr', 'alexr2']:
                with torch.no_grad():
                     y_pred_b = model_b(X_batch)
                
                y_pred = model(X_batch)
                # print("diff of model and model_b: ", torch.norm(y_pred_b-y_pred))
                loss = loss_fn(y_pred, y_batch.float(), z_batch, y_pred_b)
                ##"""update of the backup encoder"""
                with torch.no_grad():
                    for param, param_b in zip(model.parameters(), model_b.parameters()  ):
                        # param_b.data = param.data 
                        param_b.data.copy_(param.data)
            elif args.loss in ['squared_hinge']:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch.float(), z_batch)
                
            if args.loss in ['con_ex']:
                loss_fn.update_s_t(model, model_b, X_batch, y_batch, z_batch)

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch.float(), z_batch)
                ##"""update of the backup encoder"""
                with torch.no_grad():
                    for param, param_b in zip(model.parameters(), model_b.parameters()  ):
                        param_b.data = param.data 
                
            # loss = loss_fn(y_pred, torch.reshape(y_batch.float(), (-1, 1)), z_batch) # z_batch
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            # optimizer.step()
            if args.loss == 'alexr2':
                optimizer.prox_step()
                if (i+1) % args.in_iter == 0:
                    optimizer.step()  
                    with torch.no_grad():
                        # """update of the backup encoder"""
                        for param, param_b in zip(model.parameters(), model_b.parameters()  ):
                            param_b.data.copy_(param.data)
            else:
                optimizer.step()
            train_loss.append(loss.item())

            
        train_loss = np.mean(train_loss)
        
        ### evaluation
        if args.loss in ['alexr2']:
            optimizer.eval()
        unfairness_train = eval_fn(model, trainVal_loader, epoch_stats, set_type = 'train', args=args)
        unfairness_val = eval_fn(model, val_loader, epoch_stats, set_type = 'val', args=args)# epoch_stats,
        unfairness_test = eval_fn(model, test_loader, epoch_stats, set_type = 'test', args=args)
        if args.loss in ['alexr2']:
            optimizer.train()


        if best_val > unfairness_val:
            best_val = unfairness_val
            best_test = unfairness_test
            torch.save({'model':model.state_dict(), 'epoch':epoch}, args.save_path+'/best_b4.ckpt')

        # torch.save({'model':model.state_dict(), 'epoch':epoch}, args.save_path+f'/epoch_{epoch}.ckpt')
        # print results
        logger.info("epoch: %s, train_unfairness: %.4f, val_unfairness: %.4f, test_unfairness: %.4f, best_test_unfairness: %.4f, lr: %.4f"%(epoch, unfairness_train, unfairness_val, unfairness_test, best_test, optimizer.param_groups[0]['lr'] ))
        
        epoch_stats['Train Loss'] = round(train_loss, 4)
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)

        os.makedirs(args.save_path, exist_ok=True)
        stats_df.to_csv(args.save_path + '/stats.tsv', sep='\t')
    #     train_log.append(train_auc)
    #     test_log.append(test_auc)


def approx_indicator(x, c=1):
    x= torch.tensor(x)
    return torch.sigmoid(c*x).numpy()


def compute_unfairness(y_true, y_pred, z_attr, set_type, args=None, plot = False): #ths=np.arange(-2.5,3,0.5)
    
    y_pred = y_pred.reshape(-1, 1)
    pos_mask = (y_true == 1).squeeze()
    z0_mask = (z_attr==0).squeeze()
    
    ths = np.r_[np.sort(y_pred.reshape(-1), kind="mergesort"), np.inf]
    #### sample ths for training set
    if set_type=='train' and args and args.num_ths_for_eva > 0:
        ths_ids = np.linspace(0,len(ths)-1, args.num_ths_for_eva).astype(int)
        ths = ths[ths_ids]
    
    # calculate tpr and fpr 
    rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
    mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
    for i, mask in enumerate(mask_arr):
        rate_arr[i] = (y_pred[mask] - ths >=0).mean(axis=0)

#     plt.plot(rate_arr[0], tpr,  lw=0.5, ls='--', label=f'All (AUC = {roc_auc:.3f})')
    
    delta_tpr = np.abs(rate_arr[0]-rate_arr[2])
    delta_fpr = np.abs(rate_arr[1]-rate_arr[3])
    unfairness = ( delta_tpr.mean()+delta_fpr.mean() )/2

    if plot:
        plt.plot(rate_arr[0], rate_arr[2],  lw=2, label=f'TPR')
        plt.plot(rate_arr[1], rate_arr[3],  lw=2, label=f'FPR')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
        plt.xlabel('Group 0', fontsize=14)
        plt.ylabel('Group 1', fontsize=14)
        plt.title(f'Unfairness:{unfairness:.4f}', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(alpha=0.4)
        plt.show()
    
    return unfairness

def eval_constraint(y_true, y_pred, z_attr, epoch_stats, set_type, args): #ths=np.arange(-2.5,3,0.5)
    
    ths = np.arange(args.th_start, args.th_end, args.th_step)
    
    y_pred = y_pred.reshape(-1, 1)
    pos_mask = (y_true == 1).squeeze()
    z0_mask = (z_attr==0).squeeze()
    
    rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
    mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]

    for i, mask in enumerate(mask_arr):
        rate_arr[i] = approx_indicator(y_pred[mask]-ths, c = args.scaling).mean(axis=0)

    tpr_cons = np.abs(rate_arr[0]-rate_arr[2]) - args.kappa
    fpr_cons = np.abs(rate_arr[1]-rate_arr[3]) - args.kappa
            
    for i, th in enumerate(ths):
        epoch_stats[f'{set_type}_tpr_th_{th}'] = np.round(tpr_cons[i],4)
        epoch_stats[f'{set_type}_fpr_th_{th}'] = np.round(fpr_cons[i],4)
        
    #### objective auc
    f_ps = y_pred[pos_mask]
    f_ns = y_pred[~pos_mask].squeeze()
    approx_auc = approx_indicator((f_ps - f_ns), c = args.scaling)
    epoch_stats[f'{set_type}_approx_auc'] = np.round(-approx_auc.mean(), 4)


def eval_fn(model, test_loader,  epoch_stats=None, set_type = 'train', plot=False, args=None ):
    model.eval()

    test_pred_list = []
    test_true_list = []
    test_attr_list = []
    with torch.no_grad():
        for X_batch, y_batch, z_batch in test_loader:
                X_batch = X_batch.cuda()
                test_pred = model(X_batch)
                test_pred_list.append(test_pred.cpu().detach().numpy())
                test_true_list.append(y_batch.numpy())
                test_attr_list.append(z_batch.numpy())
    test_true = np.concatenate(test_true_list)
    test_pred = np.concatenate(test_pred_list)
    test_attr = np.concatenate(test_attr_list)
    
    # if verbose:
    #     print(test_pred.max(), test_pred.min())
    
    unfairness = compute_unfairness(test_true, test_pred, test_attr, set_type, plot = plot ,args=args)
    fpr, tpr, thresholds = roc_curve(test_true, test_pred)  # False Positive Rate, True Positive Rate
    roc_auc = auc(fpr, tpr)  # Compute AUC
    
    if epoch_stats is not None:
        epoch_stats[f'{set_type}_auc'] = np.round(roc_auc, 4)
        epoch_stats[f'{set_type}_unfairness'] = np.round(unfairness, 4)
        eval_constraint(test_true, test_pred, test_attr, epoch_stats, set_type=set_type, args=args )
    
    if plot:
        # test_auc =  roc_auc_score(test_true, test_pred)
        ### group0
        mask = test_attr==0
        fpr_0, tpr_0, thresholds = roc_curve(test_true[mask], test_pred[mask])  # False Positive Rate, True Positive Rate
        roc_auc_0 = auc(fpr_0, tpr_0)  # Compute AUC
        ### group1
        fpr_1, tpr_1, thresholds = roc_curve(test_true[~mask], test_pred[~mask])  # False Positive Rate, True Positive Rate
        roc_auc_1 = auc(fpr_1, tpr_1)  # Compute AUC

        # Plot the ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr,  lw=0.5, ls='--', label=f'All (AUC = {roc_auc:.3f})')
        plt.plot(fpr_0, tpr_0,  lw=2, label=f'Group 0 (AUC = {roc_auc_0:.3f})')
        plt.plot(fpr_1, tpr_1,  lw=2, label=f'Group 1 (AUC = {roc_auc_1:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve, Unfairness:{unfairness:.4f}', fontsize=16)
        plt.legend(loc='lower right', fontsize=12)
        plt.grid(alpha=0.4)
        plt.show()

    return unfairness



