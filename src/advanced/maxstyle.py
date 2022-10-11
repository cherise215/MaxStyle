import torch
import torch.nn as nn
import random


class MaxStyle(nn.Module):
    """MaxStyle
    Official implementation of MaxStyle: [MICCAI 2022] MaxStyle: Adversarial Style Composition for Robust Medical Image Segmentation
    code is adapted based on the orginal mixstyle implementation: https://github.com/KaiyangZhou/mixstyle-release
    Reference:
    Chen et al., MaxStyle: Adversarial Style Composition for Robust Medical Image Segmentation. MICCAI, 2022.
    """

    def __init__(self, batch_size, num_feature, p=0.5, mix_style=True,no_noise=False,
    mix_learnable=True, noise_learnable=True,always_use_beta=False,alpha=0.1,eps=1e-6,use_gpu=True, debug=False):
        """
        Args:
            batch_size (int): _description_
            num_feature (int): _description_
            p (float, optional): _description_. Defaults to 0.5.
            mix_style (bool, optional): whether to apply style mixing. Defaults to True.
            no_noise (bool, optional): whether to disable style noise perturbation. Defaults to False.
            mix_learnable (bool, optional): make style mixing parameters learnable. Defaults to True.
            noise_learnable (bool, optional): make style noise parameters learnable.. Defaults to True.
            always_use_beta (bool, optional): use beta distribution to sample linear mixing weight lmda, otherwise, sample from uniform distribution. Defaults to False.
            alpha (float, optional): beta(alpha, alpha). control the shape of beta distribution. Defaults to 0.1.
            eps (float, optional): scaling parameter to avoid numerical issues.
            use_gpu (bool, optional): whether use gpu or not. Defaults to True.
            debug (bool, optional): whether to print debug information. Defaults to False.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_feature = num_feature
        self.p = p
        self.mix_style = mix_style
        self.no_noise = no_noise  
        self.mix_learnable = mix_learnable
        self.noise_learnable = noise_learnable
        self.always_use_beta = always_use_beta
        self.alpha = alpha
        self.eps = eps
        self.use_gpu = use_gpu
        self.debug=debug
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.data = None
        self.init_parameters()

    def init_parameters(self):
        '''
        init permutation order, lmda for style mixing, gamma_noise, beta_noise for noise perturbation 
        '''
        batch_size  = self.batch_size
        num_feature = self.num_feature
    
        self.perm = torch.randperm(batch_size) 
        while torch.allclose(self.perm, torch.arange(batch_size)):
            # avoid identical permutation order
            self.perm = torch.randperm(batch_size)  
       
        if self.debug: print ('permutation index',self.perm)

        self.rand_p = torch.rand(1)

        if self.rand_p >= self.p:
            ##not performing 
            if self.debug: print("not performing maxstyle") 
            self.gamma_noise = torch.zeros(batch_size, num_feature, 1, 1, device=self.device).float()
            self.beta_noise = torch.zeros(batch_size, num_feature, 1, 1, device=self.device).float()
            self.lmda = torch.zeros(batch_size, 1, 1, 1, device=self.device).float()

            self.gamma_noise.requires_grad = False
            self.beta_noise.requires_grad = False
            self.lmda.requires_grad = False
        else:
            if self.no_noise:
                gamma_noise = torch.randn(batch_size, num_feature, 1, 1, device=self.device).float()
                beta_noise = torch.randn(batch_size, num_feature, 1, 1, device=self.device).float()
            else:
                gamma_noise = torch.zeros(batch_size, num_feature, 1, 1, device=self.device).float()
                beta_noise = torch.zeros(batch_size, num_feature, 1, 1, device=self.device).float()
            self.gamma_noise = None
            self.beta_noise = None

            if self.noise_learnable:
                assert self.no_noise is False, 'turn no_noise=False to enable the optimization of noise'

                self.gamma_noise = nn.Parameter(torch.empty(batch_size, num_feature, 1, 1, device=self.device))
                self.beta_noise = nn.Parameter(torch.empty(batch_size, num_feature, 1, 1, device=self.device))
                
                nn.init.normal_(self.gamma_noise)
                nn.init.normal_(self.beta_noise)
                self.gamma_noise.requires_grad = True
                self.beta_noise.requires_grad = True
            else:
                self.gamma_noise = gamma_noise
                self.beta_noise = beta_noise
                self.gamma_noise.requires_grad = False
                self.beta_noise.requires_grad = False
            if self.mix_style is False:
                self.lmda = torch.zeros(batch_size, 1, 1, 1, dtype =torch.float32,device = self.device)
                self.lmda.requires_grad = False
            else:
                self.lmda=None
                if self.always_use_beta:
                    self.beta_sampler = torch.distributions.Beta(self.alpha, self.alpha)
                    lmda = self.beta_sampler.sample((batch_size, 1, 1, 1)).to(self.device)
                    self.lmda = nn.Parameter(lmda.float())
                else: 
                    lmda = torch.rand(batch_size, 1, 1, 1, dtype =torch.float32,device = self.device)
                    self.lmda = nn.Parameter(lmda.float())
                    
                if self.mix_learnable:
                    self.lmda.requires_grad = True
                else:
                    self.lmda.requires_grad = False
        self.gamma_std = None
        self.beta_std = None
        if self.debug: 
            print("lmda:",self.lmda)
            print("gamma_noise:",self.gamma_noise)
            print("beta_noise:",self.beta_noise)
            print("perm:",self.perm)

  

    def __repr__(self):
        if self.p >=self.rand_p:
            return f'MaxStyle: \
                 mean of gamma noise: {torch.mean(self.gamma_noise)}, std:{torch.std(self.gamma_noise)} ,\
                 mean of beta noise: {torch.mean(self.beta_noise)}, std: {torch.std(self.beta_noise)}, \
                 mean of mix coefficient: {torch.mean(self.lmda)}, std: {torch.std(self.lmda)}'

        else:
            return 'diffuse style not applied'

    def reset(self):
        ## this function must be called before each forward pass if the underlying input data is changed
        self.init_parameters()
        if self.debug: print('reinitializing parameters')
    def forward(self, x):
        self.data = x
        B = x.size(0)
        C = x.size(1)
        flatten_feature = x.view(B, C, -1)

        if (self.rand_p >= self.p) or (not self.mix_style and self.no_noise) or B<=1 or flatten_feature.size(2)==1:
            # print("not operating")
            if B<=1:
                Warning('MaxStyle: B<=1, not performing maxstyle')
            if flatten_feature.size(2)==1:
                Warning('MaxStyle: spatial dim=1, not performing maxstyle')
            return x

        assert self.batch_size == B and self.num_feature == C, f"check input dim, expect ({self.batch_size}, {self.num_feature}, *,*) , got {B}{C}"

        # style normalization, making it consistent with MixStyle implementation
        mu = x.mean(dim=[2, 3], keepdim=True) ## [B,C,1,1]
        var = x.var(dim=[2, 3], keepdim=True) ## [B,C,1,1]
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        if self.debug:print ('normed', x_normed.mean())

        # estimate noise from the given distribution
        if self.gamma_std is None:
            self.gamma_std = torch.std(sig, dim=0, keepdim=True).detach() ## [1,C,1,1]
        if self.beta_std is None:
            self.beta_std = torch.std(mu, dim=0, keepdim=True).detach()   ## [1,C,1,1]

        # base style statistics from interpolated styles
        if B > 1:
            if self.mix_style:
                clipped_lmda = torch.clamp(self.lmda, 0, 1)
                mu2, sig2 = mu[self.perm], sig[self.perm] ## [B,C,1,1]
                sig_mix = sig * (1 - clipped_lmda) + sig2 * clipped_lmda  ## [B,C,1,1]
                mu_mix = mu * (1 - clipped_lmda) + mu2 * clipped_lmda
            else:
                sig_mix = sig
                mu_mix = mu
            # augment style w/o or w/ noise 
            if self.no_noise:
                x_aug = sig_mix* x_normed +mu_mix
            else:
                x_aug = (sig_mix +  self.gamma_noise * self.gamma_std) * \
                                x_normed + (mu_mix + self.beta_noise * self.beta_std)
        else:
             x_aug = (sig +  self.gamma_noise * self.gamma_std) * \
                                x_normed + (mu + self.beta_noise * self.beta_std)
        return x_aug



if __name__ =="__main__":
    ## test MaxStyle function
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(43)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = (3*torch.arange(32, device=device,dtype=torch.float32)+5).view(4,2,2,2)
    
    print ('set up maxstyle')
    style_augmentor = MaxStyle(batch_size=features.size(0), num_feature=features.size(1), p=0.5, 
                                mix_style=True, mix_learnable=True, noise_learnable=True, 
                                always_use_beta=False, no_noise=False, use_gpu=torch.cuda.is_available(),
                                debug=False)
    # print ("test Maxstyle's stability with the same input data")
    # augmented_features = style_augmentor(features)
    # print ('after maxstyle 1', augmented_features.mean())
    
    # #test MaxStyle with the same input data
    # augmented_features_2 = style_augmentor(features)
    # print ('after maxstyle 2', augmented_features_2.mean())
    # print("augmented feature changed?", not torch.allclose(augmented_features, augmented_features_2, equal_nan=True))
    # print ('reset parameters')

    # #test Maxstyle's behavior with different input data, i.e., auto reset parameters
    # features_2 = (3*torch.arange(32, device=device,dtype=torch.float32)+5+1e-6).view(4,2,2,2)
    # augmented_features_4 = style_augmentor(features_2)
    # print("augmented feature changed?", not torch.allclose(augmented_features, augmented_features_4, equal_nan=True))
    
    ## test the differentiability of MaxStyle. Note that this is only a demo for functional testing. Not for practical use.
    gt = torch.ones_like(features)
    lr = 0.1
    num_step = 5
    print ('start optimization')
    print ('list of parameters',list(style_augmentor.parameters()))
    if list(style_augmentor.parameters()) == []:
        print ('no parameters to optimize')
    else:
        optimizer = torch.optim.Adam(list(style_augmentor.parameters()), lr=lr)
        loss_fn  = torch.nn.MSELoss(reduction='mean')  ## loss function to optimize maxstyle parameters.here we use MSE for simplicity. In practice, we can use other loss functions, such as CrossEntropyLoss for the segmentation task, etc
        for i in range(num_step):
            output = style_augmentor(features)
            loss = loss_fn(output, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'<---------------after {i} step optimization--------------->')
            print(style_augmentor)
            print(f'loss {loss.item()}')
        