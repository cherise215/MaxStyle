import torch
import torch.nn as nn
import random


class MixStyle(nn.Module):
    """MixStyle.
    code is adapted from the orginal mixstyle paper.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-8, mix='random', lmda=None, zero_init=False, coefficient_sampler=None):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          lmda (float): weight for mix (intrapolation or extrapolation). If not set, will sample from [0,1) (from beta distribution)
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.eps = eps
        self.mu = None
        self.std = None
        self.zero_init = zero_init
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.lmda = lmda
        self.coeficient_sampler = None
        self.beta = torch.distributions.Beta(alpha, alpha)

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def get_perm(self):
        return self.perm

    def forward(self, x, perm=None):
        p = torch.rand(1)
        if p > self.p:
            # print("not operating")
            return x

        B = x.size(0)
        C = x.size(1)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        if self.lmda is None:
            if self.coeficient_sampler is None:
                # use default setting: for intrapolatio, e.g., mixstyle,  use beta distribution and for extrapolation, gaussian uses uniform distribution
                # print(f"lamda: {lmda}")
                lmda = self.beta.sample((B, 1, 1, 1))
            else:
                if self.coeficient_sampler == 'beta':
                    lmda = self.beta.sample((B, 1, 1, 1))
                elif self.coeficient_sampler == 'uniform':
                    lmda = torch.rand(B, 1, 1, 1)
                elif self.coeficient_sampler == 'gaussian':
                    lmda = torch.randn(B, 1, 1, 1)
                else:
                    raise ValueError

        else:
            lmda = torch.ones(B, 1, 1, 1) * self.lmda
        lmda = lmda.to(x.device)

        if self.mix in ['random', 'crossdomain']:
            if perm is not None:
                mu2, sig2 = mu[perm], sig[perm]
            else:
                if self.mix == 'random':
                    # random shuffle
                    perm = torch.randperm(B)
                    mu2, sig2 = mu[perm], sig[perm]
                elif self.mix == 'crossdomain':
                    # split into two halves and swap the order
                    perm = torch.arange(B - 1, -1, -1)  # inverse index
                    perm_b, perm_a = perm.chunk(2)
                    perm_b = perm_b[torch.randperm(B // 2)]
                    perm_a = perm_a[torch.randperm(B // 2)]
                    perm = torch.cat([perm_b, perm_a], 0)
                    mu2, sig2 = mu[perm], sig[perm]
            mu_mix = mu * (1 - lmda) + mu2 * lmda
            sig_mix = sig * (1 - lmda) + sig2 * lmda
            self.perm = perm
            # print(perm)
            return x_normed * sig_mix + mu_mix
        elif self.mix == 'gaussian':
            ## DSU: adding gaussian noise
            gaussian_mu = torch.randn(B, C, 1, 1, device=x.device) * torch.std(mu, dim=0, keepdim=True)
            gaussian_mu.requires_grad = False
            gaussian_std = torch.randn(B, C, 1, 1, device=x.device) * torch.std(sig, dim=0, keepdim=True)
            gaussian_std.requires_grad = False
            mu_mix = mu + gaussian_mu
            sig_mix = sig + gaussian_std
            return x_normed * sig_mix + mu_mix
        else:
            raise NotImplementedError

        
       


