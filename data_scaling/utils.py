import torch
import numpy as np
from typing import Union
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def generate_metrics(estimates: Union[np.ndarray, torch.Tensor],
                     ground_truth: Union[np.ndarray, torch.Tensor]):
    '''
    Generate metrics comparing estimates to ground truth.

    Args:
      estimates: estimated values, e.g., scaling law predictions for \psi_k(z).
      ground_truth: true values, e.g., \psi_k(z).
    '''
    if isinstance(estimates, np.ndarray):
        error = np.mean((estimates - ground_truth) ** 2)
        expl_var = 1 - error / np.mean((ground_truth.mean() - ground_truth) ** 2)
        r2 = r2_score(ground_truth, estimates)
        corr = np.corrcoef(estimates, ground_truth)[0, 1]
        spearman = spearmanr(estimates, ground_truth)[0]
        sign = np.mean(np.sign(estimates) == np.sign(ground_truth))
    else:
        error = torch.mean((estimates - ground_truth) ** 2).item()
        expl_var = 1 - error / torch.mean((ground_truth.mean() - ground_truth) ** 2).item()
        r2 = r2_score(ground_truth.cpu().detach(), estimates.cpu().detach())
        corr = torch.corrcoef(torch.stack([estimates, ground_truth]))[0, 1].item()
        spearman = spearmanr(estimates.cpu().detach(), ground_truth.cpu().detach())[0]
        sign = torch.mean((torch.sign(estimates) == torch.sign(ground_truth)).float()).item()
    return {
        'error': error,
        'expl_var': expl_var,
        'r2': r2,
        'corr': corr,
        'spearman': spearman,
        'sign_agreement': sign
    }
