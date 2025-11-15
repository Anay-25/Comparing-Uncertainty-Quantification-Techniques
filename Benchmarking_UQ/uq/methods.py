# uq/methods.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

def _enable_dropout(model):
    """Enable dropout layers during test time for MC Dropout."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

@torch.no_grad()
def predict_vanilla(model, dataloader, device):
    """Single forward pass, no stochasticity."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(dataloader, desc="Vanilla", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)

@torch.no_grad()
def predict_mc_dropout(model, dataloader, device, T=10):
    """MC Dropout with T stochastic passes."""
    model.eval()
    _enable_dropout(model)
    all_pass_probs = []

    for t in range(T):
        probs_list, labels_list = [], []
        for images, labels in tqdm(dataloader, desc=f"MC Dropout {t+1}/{T}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)
            if t == 0:
                labels_list.append(labels)
        all_pass_probs.append(torch.cat(probs_list))
        if t == 0:
            all_labels = torch.cat(labels_list)

    all_pass_probs = torch.stack(all_pass_probs)     # (T, N, C)
    mean_probs = all_pass_probs.mean(dim=0)          # (N, C)
    return mean_probs, all_pass_probs, all_labels

@torch.no_grad()
def predict_tta(model, dataloader, device, T=10):
    """TTA: Repeat forward passes with augmented dataloader."""
    model.eval()
    all_pass_probs = []

    for t in range(T):
        probs_list, labels_list = [], []
        for images, labels in tqdm(dataloader, desc=f"TTA {t+1}/{T}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs)
            if t == 0:
                labels_list.append(labels)
        all_pass_probs.append(torch.cat(probs_list))
        if t == 0:
            all_labels = torch.cat(labels_list)

    all_pass_probs = torch.stack(all_pass_probs)     # (T, N, C)
    mean_probs = all_pass_probs.mean(dim=0)
    return mean_probs, all_pass_probs, all_labels
