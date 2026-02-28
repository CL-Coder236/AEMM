import torch
import numpy as np
from scipy.linalg import sqrtm
from torch.distributions.multivariate_normal import MultivariateNormal

class FeatureDistance:
    
    @staticmethod
    def mmd(x1, x2, kernel='rbf', sigma=1.0, c=1.0, d=3):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        n1, n2 = x1.size(0), x2.size(0)
        
        def rbf_kernel(x, y, gamma):
            dist = torch.cdist(x, y)**2
            return torch.exp(-gamma * dist)
            
        def linear_kernel(x, y):
            return x @ y.T
            
        def poly_kernel(x, y):
            return (x @ y.T + c)**d
        
        kernels = {
            'rbf': lambda x,y: rbf_kernel(x,y,1/(2*sigma**2)),
            'linear': linear_kernel,
            'poly': poly_kernel
        }
        
        k_func = kernels.get(kernel)
        if not k_func:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        k11 = k_func(x1, x1)
        k22 = k_func(x2, x2)
        k12 = k_func(x1, x2)
        
        mask = torch.ones_like(k11, dtype=torch.bool)
        mask.fill_diagonal_(False) if hasattr(mask, 'fill_diagonal_') else \
            torch.diagonal(mask).fill_(False)
            
        k11 = k11 * mask
        k22 = k22 * mask
        
        term1 = k11.sum() / (n1*(n1-1))
        term2 = k22.sum() / (n2*(n2-1))
        term3 = k12.sum() * 2 / (n1*n2)
        
        return term1 + term2 - term3
    
    @staticmethod
    def kl_divergence(x1, x2, eps=1e-5):

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        mu1 = x1.mean(dim=0)
        mu2 = x2.mean(dim=0)
        cov1 = FeatureDistance.compute_covariance(x1, eps)
        cov2 = FeatureDistance.compute_covariance(x2, eps)

        try:
            u, s, v = torch.svd(cov2)
            s = torch.clamp(s, min=eps)
            cov2_inv = v @ torch.diag(1.0/s) @ u.T
        except Exception:
            cov2_inv = torch.linalg.pinv(cov2)

        trace_term = torch.trace(cov2_inv @ cov1)
        mean_diff = mu2 - mu1
        mean_term = mean_diff @ cov2_inv @ mean_diff

        sign1, logdet1 = torch.linalg.slogdet(cov1)
        sign2, logdet2 = torch.linalg.slogdet(cov2)

        if sign1 <= 0 or sign2 <= 0:
            logdet_term = torch.log(torch.clamp(cov2.diag(), min=eps)).sum() - \
                          torch.log(torch.clamp(cov1.diag(), min=eps)).sum()
        else:
            logdet_term = logdet2 - logdet1

        return 0.5 * (trace_term + mean_term + logdet_term - x1.size(1))

    @staticmethod
    def compute_distance(x1, x2, metric='mmd', **kwargs):
        """统一接口计算不同类型的距离"""
        if not isinstance(x1, torch.Tensor) or not isinstance(x2, torch.Tensor):
            raise ValueError("Input x1 and x2 must be torch.Tensors.")
        if x1.dim() != 2 or x2.dim() != 2:
            raise ValueError("Input x1 and x2 must be 2D tensors.")
        if x1.size(1) != x2.size(1):
            raise ValueError("Input x1 and x2 must have the same number of features.")

        if metric == 'mmd':
            kernel = kwargs.get('kernel', 'rbf')
            if kernel not in ['rbf', 'linear', 'poly']:
                raise ValueError(f"Unsupported kernel: {kernel}")
            sigma = kwargs.get('sigma', 1.0)
            return FeatureDistance.mmd(x1, x2, kernel=kernel, sigma=sigma)
        elif metric == 'kl':
            eps = kwargs.get('eps', 1e-5)
            return FeatureDistance.kl_divergence(x1, x2, eps=eps)
        elif metric == 'mse':
            return torch.nn.functional.mse_loss(x1, x2)
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
