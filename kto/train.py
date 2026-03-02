from __init__ import *
import experiments_single.imp_reward as ir
import dataset.dataset as data

def train_kto(beta, delta=1.5, good_ratio=None, estimation_mode="analytical", alpha=0.5):
    policy = GaussianPolicy().to(DEVICE)
    ref_policy = GaussianPolicy(REF_MU, math.log(REF_SIGMA)).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    torch.manual_seed(42)
    y_fixed, labels_fixed = data.build_kto_dataset(delta, good_ratio=good_ratio)  # fixed dataset
    sigmas = []
    running_kl = torch.tensor(0.0).to(DEVICE)

    for step in range(STEPS):
        optimizer.zero_grad()

        # Use the fixed dataset for the main loss (y_fixed, labels_fixed)
        log_p = policy.log_prob(y_fixed)
        log_p_ref = ref_policy.log_prob(y_fixed)
        h = ir.implicit_reward(policy, ref_policy, y_fixed, beta)

        # Estimate KL(pi || pi_ref) correctly
        if estimation_mode == "analytical":
            kl = policy.kl_to_ref().detach()          # closed form
        elif estimation_mode == "batch":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)   # sample from current policy
                log_p_sample = policy.log_prob(y_sample)
                log_ref_sample = ref_policy.log_prob(y_sample)
                kl = torch.mean(log_p_sample - log_ref_sample).detach()   # E_pi[log pi - log pi_ref] = KL(pi || pi_ref)
        elif estimation_mode == "running_avg":
            with torch.no_grad():
                y_sample = policy.sample(DATASET_SIZE)
                log_p_sample = policy.log_prob(y_sample)
                log_ref_sample = ref_policy.log_prob(y_sample)
                batch_kl = torch.mean(log_p_sample - log_ref_sample).detach()
                running_kl = (1 - alpha) * running_kl + alpha * batch_kl
                kl = running_kl
        elif estimation_mode == "fixed":
            kl = 0.3   # constant

        z = h - kl

        v = torch.zeros_like(z)
        v[labels_fixed == 1.0] = torch.sigmoid(z[labels_fixed == 1.0])
        v[labels_fixed == 0.0] = torch.sigmoid(-LAMBDA * z[labels_fixed == 0.0])

        loss = torch.mean(1 - v)
        loss.backward()
        optimizer.step()

        sigmas.append(policy.sigma().item())

    return policy, sigmas