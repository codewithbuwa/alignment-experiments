from tkinter import N

from utils import *
class GaussianMixturePolicy(nn.Module):
    def __init__(self, n_components=2, mu_init=None, log_sigma_init=None, logits_init=None):
        super().__init__()
        self.n_components = n_components
        # Means: initialize around reference mean or given values
        if mu_init is None:
            mu_init = torch.linspace(REF_MU - 2, REF_MU + 2, n_components)
        self.mus = nn.Parameter(mu_init.clone().detach().float())
        # Log scales
        if log_sigma_init is None:
            log_sigma_init = math.log(REF_SIGMA) * torch.ones(n_components)
        self.log_sigmas = nn.Parameter(log_sigma_init.clone().detach().float())
        # Mixture weights (logits)
        if logits_init is None:
            logits_init = torch.zeros(n_components)
        self.logits = nn.Parameter(logits_init.clone().detach().float())

    def sigmas(self):
        return torch.exp(self.log_sigmas)

    def probs(self):
        return torch.softmax(self.logits, dim=-1)

    def log_prob(self, y):
        """Compute log probability for each y (vectorized)"""

        y = y.unsqueeze(-1)  
        mus = self.mus.unsqueeze(0)  
        sigmas = self.sigmas().unsqueeze(0) 
        log_probs = -0.5 * (((y - mus) / sigmas) ** 2 + \
            2 * self.log_sigmas + math.log(2 * math.pi))
        # log_probs shape: (batch, K)
        log_weights = torch.log_softmax(self.logits, dim=-1).unsqueeze(0)
        return torch.logsumexp(log_probs + log_weights, dim=-1)

    def sample(self, n):
        """Draw n samples from the mixture"""
        with torch.no_grad():
            probs = self.probs()
            comp = torch.multinomial(probs, n, replacement=True)  
            mus_comp = self.mus[comp]
            sigmas_comp = self.sigmas()[comp]
            return mus_comp + sigmas_comp * torch.randn(n)
        
    def kl_to_ref(self, n_samples=1000):
        """
        Monte Carlo estimate of KL(π || π_ref)
        """
        ref_policy = REF_POLICY
        with torch.no_grad():
            # Sample from current policy
            y = self.sample(n_samples)
            
            # Compute log probabilities under both policies
            log_p = self.log_prob(y)          
            log_ref = ref_policy.log_prob(y)
            
            kl = torch.mean(log_p - log_ref).detach()
            
            return kl
        

REF_POLICY = GaussianMixturePolicy(n_components= N_COMPONENTS,
    mu_init = torch.linspace(5, 8, N_COMPONENTS),
    log_sigma_init = math.log(REF_SIGMA) * torch.ones(N_COMPONENTS), 
    logits_init = torch.zeros(N_COMPONENTS)
).to(DEVICE)