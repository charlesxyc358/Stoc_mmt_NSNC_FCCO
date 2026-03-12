import numpy as np
import torch
from sklearn.isotonic import isotonic_regression

def inv_permutation(p):
    ret = np.zeros(len(p), dtype=int)
    ret[p] = np.arange(len(p))
    return ret
def project_onto_capped_simplex(theta:torch.Tensor, tau: float, device:torch.device):
    """
    Efficient bregman projections onto the permutahedron and related polytopes.
    C. H. Lim and S. J. Wright.
    In Proc. of AISTATS, pages 1205–1213, 2016
    and its implementation from
    https://github.com/mblondel/projection-losses/blob/master/polytopes.py
    """
    assert len(theta.shape) == 1
    theta = theta.detach().clone().cpu().numpy()
    k = np.ceil(1/tau).astype(int) - 1
    w = np.zeros_like(theta)
    w[:k] = tau
    w[k] = 1 - k * tau
    perm = np.argsort(theta)[::-1]
    theta = theta[perm]
    dual_sol = isotonic_regression(theta - w, increasing=False)
    primal_sol = theta - dual_sol
    return torch.tensor(primal_sol[inv_permutation(perm)]).to(device)

def grad_CVaR(x:torch.Tensor, alpha:float):
    return torch.where(x < 0, 0, 1.0/alpha)
def grad_smCVaR(x:torch.Tensor, lamda:float, alpha:float):
    return torch.where(x < 0, 0.0, torch.where(x < lamda/alpha, x/lamda, 1.0/alpha))

def generate_step_iter(total_iters, in_iter):
    out_step_iter = list(range(in_iter, total_iters+1, in_iter))
    last_loop = total_iters - out_step_iter[-1]
    if  last_loop < 0.5 * in_iter or last_loop < 5:
        out_step_iter[-1] = total_iters
    else:
        out_step_iter.append(total_iters)
    return out_step_iter