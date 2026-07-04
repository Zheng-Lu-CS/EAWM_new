import torch
from torch import Tensor


def compute_uncertainty_weights(
    uncertainty: Tensor,
    beta: float = 1.0,
    min_weight: float = 0.05,
) -> Tensor:
    if beta <= 0:
        return torch.ones_like(uncertainty)
    return torch.exp(-beta * uncertainty).clamp(min=min_weight, max=1.0)


def robust_tree_backup(
    edge_returns: Tensor,
    mode: str = "lcb",
    lcb_alpha: float = 0.5,
    cvar_fraction: float = 0.5,
    trim_ratio: float = 0.25,
) -> Tensor:
    assert edge_returns.ndim == 2, f"Expected (nodes, branches), got {edge_returns.shape}"
    mode = mode.lower()
    if mode == "mean":
        return edge_returns.mean(dim=1)
    if mode == "lcb":
        return edge_returns.mean(dim=1) - lcb_alpha * edge_returns.std(dim=1, unbiased=False)
    if mode == "cvar":
        num_branches = edge_returns.shape[1]
        k = max(1, int(num_branches * cvar_fraction))
        return edge_returns.sort(dim=1).values[:, :k].mean(dim=1)
    if mode in {"trimmed_mean", "trimmed", "trim_mean"}:
        num_branches = edge_returns.shape[1]
        trim = int(num_branches * trim_ratio)
        if trim * 2 >= num_branches:
            trim = 0
        return edge_returns.sort(dim=1).values[:, trim : num_branches - trim].mean(dim=1)
    raise ValueError(f"Unknown TreeCF backup mode: {mode}")


def normalize_tree_advantages(
    advantages: Tensor,
    edge_returns: Tensor,
    eps: float = 1e-6,
    clip: float = 5.0,
) -> Tensor:
    assert advantages.shape == edge_returns.shape
    scale = edge_returns.std(dim=1, unbiased=False, keepdim=True)
    normalized = advantages / (scale + eps)
    if clip > 0:
        normalized = normalized.clamp(-clip, clip)
    return normalized
