from __init__ import *
import experiments_single.imp_reward as ir
import dataset.dataset_mix as data

def train_kto_mixture(ref_policy: GaussianMixturePolicy = REF_POLICY, beta = BETA, delta=1.5, good_ratio=None,
                      estimation_mode="batch", n_components=2, alpha = 0.3):
    policy = GaussianMixturePolicy(n_components=n_components).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    y_fixed, labels_fixed = data.build_mixture_kto_dataset(delta = delta)
    sigmas = []
    running_kl = torch.tensor(0.0).to(DEVICE)
    
    for step in range(STEPS):
        optimizer.zero_grad()
        
        log_p = policy.log_prob(y_fixed)
        log_ref = ref_policy.log_prob(y_fixed)
        h = ir.implicit_reward(policy, ref_policy, y_fixed, beta)
        
        # KL(pi || pi_ref) – pi_ref is the original single Gaussian
        if estimation_mode == "batch":
            with torch.no_grad():
                kl = policy.kl_to_ref(n_samples=DATASET_SIZE)
        elif estimation_mode == "running_avg":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)
                log_p_sample = policy.log_prob(y_sample)
                log_ref_sample = ref_policy.log_prob(y_sample)
                batch_kl = torch.mean(log_p_sample - log_ref_sample).detach()
                running_kl = (1 - alpha) * running_kl + alpha * batch_kl                
                kl = running_kl
        elif estimation_mode == "fixed":
            kl = 0.2
        else:
            raise ValueError("Unknown estimation_mode")
        
        z = h - kl
        v = torch.zeros_like(z)
        v[labels_fixed == 1.0] = torch.sigmoid(z[labels_fixed == 1.0])
        v[labels_fixed == 0.0] = torch.sigmoid(-LAMBDA * z[labels_fixed == 0.0])
        
        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            avg_sigma = policy.sigmas().mean().item()
            sigmas.append(avg_sigma)
    
    return policy, sigmas