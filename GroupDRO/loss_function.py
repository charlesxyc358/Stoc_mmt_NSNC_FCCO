import torch
import torch.nn as nn
from utils import *


class SONEX(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.2, theta=0.1, lamda=0.1, lr_c = 0.1, beta = 0.1, n_groups=10, n_groups_per_batch=8):
        # Initialize the loss function with the hyperparameters
        # alpha: CVaR percentile; (gamma,theta): MSVR coef; lamda: smoothing coef
        # 
        super().__init__()
        print("Initializing SONEX")
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.lamda = lamda
        # self.n_groups = n_groups
        self.n_groups_per_batch = n_groups_per_batch
        self.u = torch.zeros(n_groups, requires_grad=False).cuda()
        self.c = torch.tensor(0.0, requires_grad=False).cuda()
        self.c_buf = torch.tensor(0.0, requires_grad=False).cuda()
        self.c_state = {'step':0, 'exp_avg': 0.0, 'exp_avg_sq': 0.0}
        self.beta = beta
        self.lr_c = lr_c
        self.baseloss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, epoch, logits, targets, group_ids, aux_ce_loss=None):
        # Cross-entropy loss per group
        bsz = group_ids.shape[0]
        bsz_per_group = bsz // self.n_groups_per_batch
        group_ids = group_ids.reshape(self.n_groups_per_batch, bsz_per_group)[:,0]
        ce_loss = self.baseloss(logits, targets).reshape(self.n_groups_per_batch, bsz_per_group).mean(dim=1)

        if aux_ce_loss is None:    
            return 1, ce_loss
        else:
            if self.theta != 0:
                self.u[group_ids] = self.u[group_ids] + self.gamma * (ce_loss.detach() - self.c - self.u[group_ids]) + \
                    self.theta * ((ce_loss.detach() - self.c) - (aux_ce_loss.detach() - self.c_buf))
            else:
                self.u[group_ids] = self.u[group_ids] + self.gamma * (ce_loss.detach() - self.c - self.u[group_ids])
                        
            alg_c = 's'  # 's' for sgd, 'm' for momentum
            self.c_buf = self.c
            if alg_c == 'm':            
                # for mmt: 1.0 term in grad regarded as wd instead of grad
                grad_c = - torch.mean(grad_smCVaR(self.u[group_ids], self.lamda, self.alpha)).detach()
                exp_avg = self.c_state['exp_avg']
                self.c_state['step'] += 1
                # Update the momentum moving average
                exp_avg = exp_avg * self.beta + (1 - self.beta) * grad_c
                self.c_state['exp_avg'] = exp_avg
                bias_correction1 = 1 - self.beta ** self.c_state['step']
                self.c -= self.lr_c * (1.0 + exp_avg / bias_correction1) 
            else:
                grad_c = 1.0 - 1.0 * torch.mean(grad_smCVaR(self.u[group_ids], self.lamda, self.alpha)).detach()  
                self.c -= self.lr_c * grad_c

            loss = grad_smCVaR(self.u[group_ids], self.lamda, self.alpha) * ce_loss
            return loss.mean()

class OOA(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.2, n_groups=10, n_groups_per_batch=8):
        # Initialize the loss function with the hyperparameters
        # alpha: CVaR percentile; gamma: dual lr; lamda: smoothing coef
        # 
        super().__init__()
        print("Initializing OOA")
        self.alpha = alpha
        self.gamma = gamma
        self.n_groups = n_groups
        self.n_groups_per_batch = n_groups_per_batch
        self.dual_var = torch.zeros(n_groups, requires_grad=False).cuda()
        self.baseloss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets, group_ids):
        # Cross-entropy loss per group
        bsz = group_ids.shape[0]
        bsz_per_group = bsz // self.n_groups_per_batch
        group_ids = group_ids.reshape(self.n_groups_per_batch, bsz_per_group)[:,0]
        ce_loss = self.baseloss(logits, targets).reshape(self.n_groups_per_batch, bsz_per_group).mean(dim=1)

        # dual update
        dual_grad = torch.zeros_like(self.dual_var, requires_grad=False).cuda()
        dual_grad[group_ids] = (self.n_groups / self.n_groups_per_batch) * ce_loss.detach().clone()
        dual_var_ = self.dual_var * torch.exp(self.gamma * dual_grad)  # elementwise multiplication
        self.dual_var = project_onto_capped_simplex(dual_var_, tau=1 / (self.alpha * self.n_groups),
                                                        device=torch.device("cuda"))
        loss = torch.sum(self.dual_var[group_ids] * (self.n_groups / self.n_groups_per_batch) * ce_loss)
        return loss.sum()

class SONX(nn.Module):
    def __init__(self, alpha=0.2, gamma=0.2, theta=0.1, lr_c = 0.1, n_groups=10, n_groups_per_batch=8):
        # Initialize the loss function with the hyperparameters
        # alpha: CVaR percentile; (gamma,theta): MSVR coef
        # 
        super().__init__()
        print("Initializing SONX")
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        # self.n_groups = n_groups
        self.n_groups_per_batch = n_groups_per_batch
        self.u = torch.zeros(n_groups, requires_grad=False).cuda()
        self.c = torch.tensor(0.0, requires_grad=False).cuda()
        self.c_buf = torch.tensor(0.0, requires_grad=False).cuda()
        self.lr_c = lr_c
        self.baseloss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, epoch, logits, targets, group_ids, aux_ce_loss=None):
        # Cross-entropy loss per group
        bsz = group_ids.shape[0]
        bsz_per_group = bsz // self.n_groups_per_batch
        group_ids = group_ids.reshape(self.n_groups_per_batch, bsz_per_group)[:,0]
        ce_loss = self.baseloss(logits, targets).reshape(self.n_groups_per_batch, bsz_per_group).mean(dim=1)

        if aux_ce_loss is None:    
            # aux_forward(no aux_loss provided)            
            return ce_loss
        else:
            if epoch == 0:               
                self.u[group_ids] = ce_loss.detach() - self.c
            else:
                if self.theta == 0:
                    self.u[group_ids] = self.u[group_ids] + self.gamma * (ce_loss.detach() - self.c - self.u[group_ids])
                else:
                    self.u[group_ids] = self.u[group_ids] + self.gamma * (ce_loss.detach() - self.c - self.u[group_ids]) + \
                        self.theta * ((ce_loss.detach() - self.c) - (aux_ce_loss.detach() - self.c_buf))
                        
            self.c_buf = self.c
            grad_c = 1.0 - 1.0 * torch.mean(grad_CVaR(self.u[group_ids], self.alpha)).detach()
            self.c -= self.lr_c * grad_c
            loss = grad_CVaR(self.u[group_ids], self.alpha) * ce_loss
            return loss.mean()
    
