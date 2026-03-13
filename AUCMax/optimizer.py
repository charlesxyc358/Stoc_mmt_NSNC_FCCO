import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
import math

# Moreau-Envelope SGD 
class MESGD(Optimizer): 
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, lr_prox = required, mor_coef = required, restart = False):
        self.restart = restart
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr_prox is not required and lr_prox < 0.0:
            raise ValueError("Invalid learning rate for proximal-point: {}".format(lr_prox))
        if mor_coef is not required and mor_coef < 0.0:
            raise ValueError("Invalid moreau envelope coefficient: {}".format(mor_coef))
        self.buff = []

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay, lr_prox=lr_prox, mor_coef=mor_coef)
        super(MESGD, self).__init__(params, defaults)
        print("@@@ MESGD initialized!")

    def __setstate__(self, state):
        super(MESGD, self).__setstate__(state)

    def train(self):
        if len(self.buff) == 0:
            print("@@@ Warning! Already in train mode. Nothing will be done.")
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(self.buff.pop(0))

        assert len(self.buff) == 0, 'Model size incompatible! Please double check.'

    def eval(self):
        self.buff = []
        for group in self.param_groups:
            for p in group['params']:
                self.buff.append(torch.clone(p).detach())

                param_state = self.state[p]
                p.data.copy_(param_state['x'])

    def step(self, closure=None):
        """Performs a single optimization step for model.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            lr = group['lr']
            mor_coef = group['mor_coef']

            for p in group['params']:
                if p.grad is None:
                    continue
                # d_p = p.grad.data
                # if weight_decay != 0:
                #     d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'x' not in param_state:
                    x = param_state['x'] = torch.clone(p).detach()
                else:
                    x = param_state['x']
                
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(x-p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - momentum, x-p.data)
                else:
                    buf = torch.clone(x-p).detach()

                x.add_(-lr*mor_coef, buf)
                if self.restart:
                    # every time finish an outer loop, restart prox_estimator from current iterate
                    p.data.copy_(x)

    def prox_step(self, closure=None):
        """Performs a single optimization step for prox model.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr_prox = group['lr_prox']  # gamma in draft
            mor_coef = group['mor_coef'] # = 1/lambda in bokun's draft
            corr_term = 1+mor_coef*lr_prox # close to 1            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'x' not in param_state:
                    x = param_state['x'] = torch.clone(p).detach()
                else:
                    x = param_state['x']
                d_prox_est = d_p.add(x, alpha = -mor_coef).detach()

                p.data.mul_(1 - lr_prox * (weight_decay + mor_coef/corr_term))
                p.data.add_(-lr_prox/corr_term, d_prox_est)
                

        return loss
    
# Moreau-Envelope Double-Loop AdamW 
class MEAdamW(Optimizer): 
    def __init__(self, params, lr=required, betas=(0.9, 0.999), eps=1e-13,
                 weight_decay=0, lr_prox = required, mor_coef = required, restart = False):
        self.restart = restart
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError("Invalid beta values: {}".format(betas))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr_prox is not required and lr_prox < 0.0:
            raise ValueError("Invalid learning rate for proximal-point: {}".format(lr_prox))
        if mor_coef is not required and mor_coef < 0.0:
            raise ValueError("Invalid moreau envelope coefficient: {}".format(mor_coef))
        self.buff = []

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, lr_prox=lr_prox, mor_coef=mor_coef)
        super(MEAdamW, self).__init__(params, defaults)
        print("@@@ MEAdamW initialized!")
        print(f"Hyper-param: lr{lr}, betas{betas}, eps{eps}, wd{weight_decay}, lr_prox{lr_prox}, mor_coef{mor_coef}")
        print(f"Restart: {restart}")

    def __setstate__(self, state):
        super(MEAdamW, self).__setstate__(state)

    def train(self):
        if len(self.buff) == 0:
            print("@@@ Warning! Already in train mode. Nothing will be done.")
        for group in self.param_groups:
            for p in group['params']:
                p.data.copy_(self.buff.pop(0))

        assert len(self.buff) == 0, 'Model size incompatible! Please double check.'

    def eval(self):
        self.buff = []
        for group in self.param_groups:
            for p in group['params']:
                self.buff.append(torch.clone(p).detach())

                param_state = self.state[p]
                p.data.copy_(param_state['x'])

    def step(self, closure=None):
        """Performs a single optimization step for model.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            mor_coef = group['mor_coef']

            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'step' not in param_state: #1st time call step()
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p.data)
                    param_state['x'] = torch.clone(p).detach()
                    
                exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                param_state['step'] += 1
                x = param_state['x']
                
                # if momentum != 0:
                #     if 'momentum_buffer' not in param_state:
                #         buf = param_state['momentum_buffer'] = torch.clone(x-p).detach()
                #     else:
                #         buf = param_state['momentum_buffer']
                #         buf.mul_(momentum).add_(1 - momentum, x-p.data)
                # else:
                #     buf = torch.clone(x-p).detach()
                
                # x.add_(-lr*mor_coef, buf)
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, x-p.data)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, x-p.data, x-p.data)
                
                bias_correction1 = 1 - beta1 ** param_state['step']
                bias_correction2 = 1 - beta2 ** param_state['step']
                step_size = lr / bias_correction1

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # # Perform stepweight decay 
                # x.mul_(1 - group['lr'] * weight_decay) # do wd in prox step
                x.addcdiv_(-step_size, exp_avg, denom)

                if self.restart:
                    # every time finish an outer loop, restart prox_estimator from current iterate
                    p.data.copy_(x)


    def prox_step(self, closure=None):
        """Performs a single optimization step for prox model.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            lr_prox = group['lr_prox']  
            mor_coef = group['mor_coef'] 
            corr_term = 1+mor_coef*lr_prox # close to 1            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'x' not in param_state:
                    x = param_state['x'] = torch.clone(p).detach()
                else:
                    x = param_state['x']
                d_prox_est = d_p.add(x, alpha = -mor_coef).detach()

                p.data.mul_(1 - lr_prox * (weight_decay + mor_coef/corr_term))
                p.data.add_(-lr_prox/corr_term, d_prox_est)

        return loss
