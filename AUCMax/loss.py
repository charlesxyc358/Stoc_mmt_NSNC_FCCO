import warnings
import torch 
import torch.nn.functional as F

def approx_indicator(x, c=1):
    # 1/(1 + exp(-cx))
    return torch.sigmoid(c*x)

def grad_smHinge(x:torch.Tensor, lamda:float, beta:float):
    return torch.where(x < 0, 0.0, torch.where(x < lamda, beta * x/lamda, beta))

class AUC_Loss(torch.nn.Module): 
    def __init__(self, scaling=1.0): 
        super(AUC_Loss, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling                   


    def forward(self, y_pred, y_true): 
        y_pred = y_pred.reshape(-1, 1)

        pos_mask = (y_true == 1).squeeze() 
        neg_mask = (y_true == 0).squeeze() 
        assert sum(pos_mask) > 0, "Input data has no positive sample!"
        f_ps = y_pred[pos_mask]
        f_ns = y_pred[neg_mask].squeeze()
        approx_auc = approx_indicator((f_ps - f_ns), c = self.scaling)
        return -approx_auc.mean()
    
class AUC_ROC_Penalty_smHinge_VR(torch.nn.Module): 
    def __init__(self, beta, gamma=0.8, gamma_p=0.1, lamda=0.1, kappa=0.001, scaling=1.0, ths=torch.arange(-3,3.1,0.5)): 
        super(AUC_ROC_Penalty_smHinge_VR, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling 
        self.auc_loss_fn = AUC_Loss(scaling=scaling)
        
        ### for constraints
        self.ths = ths.to(self.device)
        self.u_tpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_tpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_arr = [self.u_tpr0, self.u_fpr0, self.u_tpr1, self.u_fpr1]
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.lamda = lamda
        self.kappa = kappa
        self.beta = beta

    def forward(self, y_pred, y_true, z_attr, y_pred_b): 
        ### auc objective
        auc_loss = self.auc_loss_fn(y_pred, y_true)
        
        ### roc constraints
        y_pred = y_pred.reshape(-1, 1)
        y_pred_b = y_pred_b.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        rate_arr_b = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred[mask]-self.ths, c = self.scaling).mean(dim=0)
                rate_arr_b[i] = approx_indicator(y_pred_b[mask]-self.ths, c = self.scaling).mean(dim=0)
                ### update estimators
                if self.u_arr[i].sum()==0:
                    self.u_arr[i][:] = rate_arr[i].detach() + self.gamma_p*(rate_arr[i]-rate_arr_b[i]).detach()
                else:
                    self.u_arr[i][:] = (1-self.gamma)*self.u_arr[i][:]+ self.gamma*rate_arr[i].detach() +  \
                                        self.gamma_p*(rate_arr[i]-rate_arr_b[i]).detach()
                
        # print(torch.abs(self.u_tpr0-self.u_tpr1))
        ### constraints
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            # tpr_loss = torch.where( torch.abs(self.u_tpr0-self.u_tpr1)-self.kappa >0 ,
            #                                 self.beta, 0).detach() * torch.abs(rate_arr[0]-rate_arr[2])
            tpr_loss = grad_smHinge(torch.abs(self.u_tpr0-self.u_tpr1)-self.kappa, \
                            self.lamda, self.beta).detach() * torch.abs(rate_arr[0]-rate_arr[2])
        else:
            tpr_loss = torch.tensor(0.)
            
        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            # fpr_loss = torch.where( torch.abs(self.u_fpr0-self.u_fpr1)-self.kappa >0 ,
            #                                 self.beta, 0).detach() * torch.abs(rate_arr[1]-rate_arr[3])
            fpr_loss = grad_smHinge(torch.abs(self.u_fpr0-self.u_fpr1)-self.kappa, \
                            self.lamda, self.beta).detach() * torch.abs(rate_arr[1]-rate_arr[3])
        else:
            fpr_loss = torch.tensor(0.)
        
        return auc_loss + (tpr_loss.mean() + fpr_loss.mean())/2 
    
class AUC_ROC_Penalty_Hinge_VR(torch.nn.Module): 
    def __init__(self, beta, gamma=0.8, gamma_p=0.1, kappa=0.001, scaling=1.0, ths=torch.arange(-3,3.1,0.5)): 
        super(AUC_ROC_Penalty_Hinge_VR, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling 
        self.auc_loss_fn = AUC_Loss(scaling=scaling)
        
        ### for constraints
        self.ths = ths.to(self.device)
        self.u_tpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_tpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_arr = [self.u_tpr0, self.u_fpr0, self.u_tpr1, self.u_fpr1]
        self.gamma = gamma
        self.gamma_p = gamma_p
        self.kappa = kappa
        self.beta = beta

    def forward(self, y_pred, y_true, z_attr, y_pred_b): 
        ### auc objective
        auc_loss = self.auc_loss_fn(y_pred, y_true)
        
        ### roc constraints
        y_pred = y_pred.reshape(-1, 1)
        y_pred_b = y_pred_b.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        rate_arr_b = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred[mask]-self.ths, c = self.scaling).mean(dim=0)
                rate_arr_b[i] = approx_indicator(y_pred_b[mask]-self.ths, c = self.scaling).mean(dim=0)
                ### update estimators
                if self.u_arr[i].sum()==0:
                    self.u_arr[i][:] = rate_arr[i].detach() + self.gamma_p*(rate_arr[i]-rate_arr_b[i]).detach()
                else:
                    self.u_arr[i][:] = (1-self.gamma)*self.u_arr[i][:]+ self.gamma*rate_arr[i].detach() +  \
                                        self.gamma_p*(rate_arr[i]-rate_arr_b[i]).detach()
                
        # print(torch.abs(self.u_tpr0-self.u_tpr1))
        ### constraints
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            tpr_loss = torch.where( torch.abs(self.u_tpr0-self.u_tpr1)-self.kappa >0 ,
                                            self.beta, 0).detach() * torch.abs(rate_arr[0]-rate_arr[2])
        else:
            tpr_loss = torch.tensor(0.)
            
        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            fpr_loss = torch.where( torch.abs(self.u_fpr0-self.u_fpr1)-self.kappa >0 ,
                                            self.beta, 0).detach() * torch.abs(rate_arr[1]-rate_arr[3])
        else:
            fpr_loss = torch.tensor(0.)

        
        return auc_loss + (tpr_loss.mean() + fpr_loss.mean())/2 
    
class AUC_ROC_Penalty_SH(torch.nn.Module): 
    def __init__(self, beta, gamma=0.8, kappa=0.001, scaling=1.0, ths=torch.arange(-3,3.1,0.5)): 
        super(AUC_ROC_Penalty_SH, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling 
        self.auc_loss_fn = AUC_Loss(scaling=scaling)
        
        ### for constraints
        self.ths = ths.to(self.device)
        self.u_tpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_tpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_arr = [self.u_tpr0, self.u_fpr0, self.u_tpr1, self.u_fpr1]
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta

    def forward(self, y_pred, y_true, z_attr): 
        ### auc objective
        auc_loss = self.auc_loss_fn(y_pred, y_true)
        
        ### roc constraints
        y_pred = y_pred.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred[mask]-self.ths, c = self.scaling).mean(dim=0)
                ### update estimators
                if self.u_arr[i].sum()==0:
                    self.u_arr[i][:] = rate_arr[i].detach()
                else:
                    self.u_arr[i][:] = (1-self.gamma)*self.u_arr[i][:]+ self.gamma*rate_arr[i].detach()
                
        # print(torch.abs(self.u_tpr0-self.u_tpr1))
        ### constraints  quadratic penalty 
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            tpr_loss = self.beta* torch.maximum(torch.abs(self.u_tpr0-self.u_tpr1)-self.kappa,
                                        torch.zeros_like(rate_arr[0]) ).detach() * torch.abs(rate_arr[0]-rate_arr[2])
        else:
            tpr_loss = torch.tensor(0.)
            
        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            fpr_loss = self.beta* torch.maximum( torch.abs(self.u_fpr0-self.u_fpr1)-self.kappa,
                                        torch.zeros_like(rate_arr[1]) ).detach() * torch.abs(rate_arr[1]-rate_arr[3])
        else:
            fpr_loss = torch.tensor(0.)

        
        return auc_loss + (tpr_loss.mean() + fpr_loss.mean())
    

class AUC_ROC_Penalty(torch.nn.Module): 
    def __init__(self, beta, gamma=0.8, kappa=0.001, scaling=1.0, ths=torch.arange(-3,3.1,0.5)): 
        super(AUC_ROC_Penalty, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling 
        self.auc_loss_fn = AUC_Loss(scaling=scaling)
        
        ### for constraints
        self.ths = ths.to(self.device)
        self.u_tpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr0 = torch.zeros(len(self.ths),).to(self.device)
        self.u_tpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_fpr1 = torch.zeros(len(self.ths),).to(self.device)
        self.u_arr = [self.u_tpr0, self.u_fpr0, self.u_tpr1, self.u_fpr1]
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta

    def forward(self, y_pred, y_true, z_attr): 
        ### auc objective
        auc_loss = self.auc_loss_fn(y_pred, y_true)
        
        ### roc constraints
        y_pred = y_pred.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred[mask]-self.ths, c = self.scaling).mean(dim=0)
                ### update estimators
                if self.u_arr[i].sum()==0:
                    self.u_arr[i][:] = rate_arr[i].detach()
                else:
                    self.u_arr[i][:] = (1-self.gamma)*self.u_arr[i][:]+ self.gamma*rate_arr[i].detach()
                
        # print(torch.abs(self.u_tpr0-self.u_tpr1))
        ### constraints
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            tpr_loss = torch.where( torch.abs(self.u_tpr0-self.u_tpr1)-self.kappa >0 ,
                                            self.beta, 0).detach() * torch.abs(rate_arr[0]-rate_arr[2])
        else:
            tpr_loss = torch.tensor(0.)
            
        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            fpr_loss = torch.where( torch.abs(self.u_fpr0-self.u_fpr1)-self.kappa >0 ,
                                            self.beta, 0).detach() * torch.abs(rate_arr[1]-rate_arr[3])
        else:
            fpr_loss = torch.tensor(0.)

        
        return auc_loss + (tpr_loss.mean() + fpr_loss.mean())/2 
    
class ConEx(torch.nn.Module): 
    def __init__(self, params, tau=10, mu=1e-3, theta=0.1, kappa=0.001, scaling=1.0, ths=torch.arange(-3,3.1,0.5)): 
        super(ConEx, self).__init__() 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaling = scaling 
        self.auc_loss_fn = AUC_Loss(scaling=scaling)
        
        ### for constraints
        self.ths = ths.to(self.device)
        ### for estimators
        self.l_tpr = torch.zeros(len(self.ths),).to(self.device)
        self.l_fpr = torch.zeros(len(self.ths),).to(self.device)
        self.s_tpr = torch.zeros(len(self.ths),).to(self.device)
        self.s_fpr = torch.zeros(len(self.ths),).to(self.device)

        self.y_tpr = torch.zeros(len(self.ths),).to(self.device)
        self.y_fpr = torch.zeros(len(self.ths),).to(self.device)

        ### paras
        self.theta = theta
        self.tau = tau
        self.mu = mu
        self.kappa = kappa

        #### reference 
        self.params = list(params)
        self.model_ref = self.__init_model_ref__(self.params)
        self.model_acc = self.__init_model_acc__(self.params)
        self.T = 0
    
    def __init_model_ref__(self, params):
         model_ref = []
         if not isinstance(params, list):
            params = list(params)
         for var in params: 
            if var is not None:
                model_ref.append(var.detach().clone().to(self.device))
            #    model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
         return model_ref
    
    def __init_model_acc__(self, params):
        model_acc = []
        if not isinstance(params, list):
           params = list(params)
        for var in params: 
            if var is not None:
               model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return model_acc

    def update_s_t(self, model, model_b, X_batch, y_true, z_attr):
        y_pred_b = model_b(X_batch)

        ### roc constraints
        y_pred_b = y_pred_b.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred_b[mask]-self.ths, c = self.scaling).mean(dim=0)

        ### constraints
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            tpr_loss = torch.abs(rate_arr[0]-rate_arr[2])-self.kappa  
        else:
            tpr_loss = None

        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            fpr_loss = torch.abs(rate_arr[1]-rate_arr[3])-self.kappa
        else:
            fpr_loss = None

        ### update s
        if tpr_loss is not None:
            for k in range(len(tpr_loss)):
                grads = torch.autograd.grad(tpr_loss[k], model_b.parameters(), retain_graph= True )
                l_f_sum = torch.tensor(0.)
                #### f(x(t-1)) + mu *||x(t-1)-x0||^2 + ( f'(x(t-1))+2*mu*(x(t-1)-x_0) )*(x(t)- x(t-1))
                params, params_b = list(model.parameters()), list(model_b.parameters())
                with torch.no_grad():
                    for i in range( len(params_b) ):
                        l_f_sum = l_f_sum + ( self.mu*torch.square(params_b[i].data - self.model_ref[i].data) ).sum()    \
                                          + ( (grads[i].detach()+2*self.mu*(params_b[i].data - self.model_ref[i].data))*(params[i].data - params_b[i].data) ).sum()


                    # for grad, param, param_b, param_0 in zip(grads, model.parameters(), model_b.parameters(),   ):
                    #     approx_sum += grad.detach()*(param.data - param_b.data).sum()

                l_f = tpr_loss[k] + l_f_sum
                self.s_tpr[k] = l_f + self.theta * (l_f - self.l_tpr[k]) * int(self.T>0)
                self.l_tpr[k] = l_f

        if fpr_loss is not None:
            for k in range(len(fpr_loss)):
                grads = torch.autograd.grad(fpr_loss[k], model_b.parameters(), retain_graph= (k!= len(fpr_loss)-1))
                l_f_sum = torch.tensor(0.)
                #### f(x(t-1)) + mu *||x(t-1)-x0||^2 + ( f'(x(t-1))+2*mu*(x(t-1)-x_0) )*(x(t)- x(t-1))
                params, params_b = list(model.parameters()), list(model_b.parameters())
                with torch.no_grad():
                    for i in range( len(params_b) ):
                        l_f_sum = l_f_sum + ( self.mu*torch.square(params_b[i].data - self.model_ref[i].data) ).sum()    \
                                          + ( (grads[i].detach()+2*self.mu*(params_b[i].data - self.model_ref[i].data))*(params[i].data - params_b[i].data) ).sum()


                    # for grad, param, param_b, param_0 in zip(grads, model.parameters(), model_b.parameters(),   ):
                    #     approx_sum += grad.detach()*(param.data - param_b.data).sum()

                l_f = fpr_loss[k] + l_f_sum
                self.s_fpr[k] = l_f + self.theta * (l_f - self.l_fpr[k]) * int(self.T>0)
                self.l_fpr[k] = l_f


        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                self.model_acc[i].data = self.model_acc[i].data + param.data
        self.T+=1

    def forward(self, y_pred, y_true, z_attr): 
        ### auc objective
        auc_loss = self.auc_loss_fn(y_pred, y_true)
        
        ### roc constraints
        y_pred = y_pred.reshape(-1, 1)
        pos_mask = (y_true == 1).squeeze() 
        z0_mask = (z_attr==0).squeeze() 
        
        # mask_tpr0 = 
        rate_arr = [None, None, None, None] #[tpr_0, fpr_0, tpr_1, fpr_1]
        mask_arr= [pos_mask&z0_mask, ~pos_mask&z0_mask, pos_mask&~z0_mask, ~pos_mask&~z0_mask]
        for i, mask in enumerate(mask_arr):
            if mask.sum()>0:
                rate_arr[i] = approx_indicator(y_pred[mask]-self.ths, c = self.scaling).mean(dim=0)
 
        # print(torch.abs(self.u_tpr0-self.u_tpr1))
        ### constraints
        if rate_arr[0] is not None and rate_arr[2] is not None: ### tpr_0, tpr_1
            tpr_loss = torch.abs(rate_arr[0]-rate_arr[2])-self.kappa  
        else:
            tpr_loss = torch.tensor(0.)
            
        if rate_arr[1] is not None and rate_arr[3] is not None: ### fpr_0, fpr_1
            fpr_loss = torch.abs(rate_arr[1]-rate_arr[3])-self.kappa

        else:
            fpr_loss = torch.tensor(0.)

        ## update y_t
        self.y_tpr = torch.maximum(self.y_tpr+ 1/self.tau*self.s_tpr,  torch.zeros_like( self.y_tpr)).detach()
        self.y_fpr = torch.maximum(self.y_fpr+ 1/self.tau*self.s_fpr,  torch.zeros_like( self.y_fpr)).detach()

        norm_sum = torch.tensor(0.)
        #### f(x(t)) + mu *||x(t)-x0||^2 
        for i in range( len(self.params) ):
            norm_sum = norm_sum + ( self.mu*torch.square(self.params[i] - self.model_ref[i].data) ).sum() 

        return auc_loss+norm_sum + ( (tpr_loss+norm_sum)*self.y_tpr ).sum() + ( (fpr_loss+norm_sum)*self.y_fpr).sum()
    
    def update_ref(self):
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0