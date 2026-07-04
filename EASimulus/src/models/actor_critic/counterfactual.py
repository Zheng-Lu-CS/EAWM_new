import torch
from torch import Tensor


def compute_counterfactual_group_baseline(
    branch_returns: Tensor,
    baseline: str = "median",
    trim_ratio: float = 0.25,
) -> Tensor:
    assert branch_returns.ndim == 2, f"Expected (batch, branches), got {branch_returns.shape}"
    baseline = baseline.lower()
    if baseline == "median":
        return branch_returns.median(dim=1, keepdim=True).values
    if baseline in {"trimmed_mean", "trimmed", "trim_mean"}:
        num_branches = branch_returns.shape[1]
        trim = int(num_branches * trim_ratio)
        if trim * 2 >= num_branches:
            trim = 0
        sorted_returns = branch_returns.sort(dim=1).values
        return sorted_returns[:, trim : num_branches - trim].mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown counterfactual baseline: {baseline}")


def compute_counterfactual_group_advantages(
    branch_returns: Tensor,
    baseline: str = "median",
    trim_ratio: float = 0.25,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    group_baseline = compute_counterfactual_group_baseline(
        branch_returns, baseline=baseline, trim_ratio=trim_ratio
    )
    group_std = branch_returns.std(dim=1, unbiased=False, keepdim=True)
    advantages = (branch_returns - group_baseline) / (group_std + eps)
    return advantages, group_baseline, group_std


def compute_uncertainty_weights(
    uncertainty: Tensor,
    beta: float = 1.0,
    min_weight: float = 0.05,
) -> Tensor:
    if beta <= 0:
        return torch.ones_like(uncertainty)
    return torch.exp(-beta * uncertainty).clamp(min=min_weight, max=1.0)
