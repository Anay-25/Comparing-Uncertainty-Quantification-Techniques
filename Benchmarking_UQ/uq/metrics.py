# uq/metrics.py
import torch

def predictive_entropy(probs, eps=1e-8):
    # probs: (N, C)
    return -(probs * (probs + eps).log()).sum(dim=1)

def max_probability(probs):
    # probs: (N, C)
    return probs.max(dim=1).values

def expected_calibration_error(probs, labels, n_bins=15):
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop = in_bin.float().mean()
        if prop > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_bin - acc_bin) * prop

    return ece.item()
