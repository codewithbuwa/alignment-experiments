from utils import *
from policy.gaussian_mixture import *

def build_mixture_dpo_dataset(ref_policy: GaussianMixturePolicy = REF_POLICY, target=TARGET, good_ratio=None):
    """
    Build DPO dataset by sampling pairs from a mixture reference policy.
    """
    # Sample pairs from the mixture policy
    y1 = ref_policy.sample(DATASET_SIZE)
    y2 = ref_policy.sample(DATASET_SIZE)
    
    # Determine winner/loser based on distance to target
    dist1 = torch.abs(y1 - target)
    dist2 = torch.abs(y2 - target)
    
    winners = torch.where(dist1 < dist2, y1, y2)
    losers = torch.where(dist1 < dist2, y2, y1)
    
    if good_ratio is not None:
        # Keep only pairs with largest margin (strongest preference signals)
        margin = torch.abs(dist1 - dist2)
        k = int(good_ratio * DATASET_SIZE)
        topk_indices = torch.topk(margin, k=k).indices
        return winners[topk_indices], losers[topk_indices]
    
    return winners, losers


def build_mixture_kto_dataset(ref_policy: GaussianMixturePolicy = REF_POLICY, delta = 1.5, good_ratio=None, target=TARGET):
    """
    Build KTO dataset by sampling from a mixture reference policy.
    """
    zone = (ZONE[0] - delta, ZONE[1] + delta)
    
    # Sample from mixture policy
    y = ref_policy.sample(DATASET_SIZE)

    
    labels = ((y >= zone[0]) & (y <= zone[1])).float()
    
    if good_ratio is not None:
        # Resample to achieve desired good_ratio
        n_good_desired = int(DATASET_SIZE * good_ratio)
        n_bad_desired = DATASET_SIZE - n_good_desired
        
        # Separate current samples
        good_mask = (labels == 1)
        bad_mask = (labels == 0)
        good_samples = y[good_mask].tolist()
        bad_samples = y[bad_mask].tolist()
        
        # Keep sampling until we have enough
        while len(good_samples) < n_good_desired or len(bad_samples) < n_bad_desired:
            # Sample a batch
            y_batch = ref_policy.sample(DATASET_SIZE)
            labels_batch = ((y_batch >= zone[0]) & (y_batch <= zone[1]))
            
            # Add to respective lists
            good_samples.extend(y_batch[labels_batch].tolist())
            bad_samples.extend(y_batch[~labels_batch].tolist())
        
        # Take exactly what we need
        good_tensor = torch.tensor(good_samples[:n_good_desired])
        bad_tensor = torch.tensor(bad_samples[:n_bad_desired])
        
        # Combine
        y = torch.cat([good_tensor, bad_tensor])
        labels = torch.cat([
            torch.ones(n_good_desired),
            torch.zeros(n_bad_desired)
        ])
        
        # Shuffle
        perm = torch.randperm(DATASET_SIZE)
        y = y[perm]
        labels = labels[perm]
    
    return y, labels

