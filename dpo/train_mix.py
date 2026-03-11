from dataset.dataset_mix import *
from policy.gaussian_mixture import *
from policy.gaussian import *
from utils import *
import experiments_single.imp_reward as ir

y_w, y_l = build_mixture_dpo_dataset()

def train_dpo_mixture(ref_policy: GaussianMixturePolicy = REF_POLICY, 
                      beta = BETA, n_components=2, steps=STEPS, lr=LR):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    sigmas = []  # track average sigma for monitoring
    history = []
    for i in range(steps):
        optimizer.zero_grad()
        h_w = ir.implicit_reward(policy, ref_policy, y_w, beta)
        h_l = ir.implicit_reward(policy, ref_policy, y_l, beta)

        loss = -torch.mean(torch.log(torch.sigmoid(h_w - h_l)))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            avg_sigma = policy.sigmas().mean().item()
            sigmas.append(avg_sigma)
        if i+1%500==0:
            mu = policy.mus
            history.append((mu, avg_sigma))
    for j in history:
        print(j[0], j[1], "Hello!")
    return policy, sigmas
