from typing import Optional

import torch
from torch import Tensor


def robust_center(
    values: Tensor,
    mode: str = "median",
    trim_ratio: float = 0.25,
) -> Tensor:
    assert values.ndim == 2, f"Expected (anchors, branches), got {values.shape}"
    mode = mode.lower()
    if mode == "mean":
        return values.mean(dim=1, keepdim=True)
    if mode == "median":
        return values.median(dim=1, keepdim=True).values
    if mode in {"trimmed_mean", "trimmed", "trim_mean"}:
        num_branches = values.shape[1]
        trim = int(num_branches * trim_ratio)
        if trim * 2 >= num_branches:
            trim = 0
        return values.sort(dim=1).values[:, trim : num_branches - trim].mean(
            dim=1, keepdim=True
        )
    raise ValueError(f"Unknown robust center mode: {mode}")


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


def normalize_counterfactual_advantages(
    advantages: Tensor,
    eps: float = 1e-6,
    clip: float = 5.0,
    scale: str = "std",
) -> Tensor:
    assert advantages.ndim == 2, f"Expected (anchors, branches), got {advantages.shape}"
    scale = scale.lower()
    if scale == "std":
        denom = advantages.std(dim=1, unbiased=False, keepdim=True)
    elif scale == "mad":
        med = advantages.median(dim=1, keepdim=True).values
        denom = (advantages - med).abs().median(dim=1, keepdim=True).values * 1.4826
    elif scale in {"none", "identity"}:
        normalized = advantages
        if clip > 0:
            normalized = normalized.clamp(-clip, clip)
        return normalized
    else:
        raise ValueError(f"Unknown advantage scale: {scale}")
    normalized = advantages / (denom + eps)
    if clip > 0:
        normalized = normalized.clamp(-clip, clip)
    return normalized


def select_counterfactual_actions(
    logits: Tensor,
    branching: int,
    mode: str,
    sample_temperature: float = 1.0,
    force_actions: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert logits.ndim == 2, f"Expected (anchors, actions), got {logits.shape}"
    if sample_temperature <= 0:
        raise ValueError("sample_temperature must be > 0")
    num_actions = logits.shape[-1]
    branching = min(int(branching), int(num_actions))
    if branching < 1:
        raise ValueError("branching must be >= 1")

    policy_log_probs = torch.log_softmax(logits, dim=-1)
    policy_probs = torch.softmax(logits, dim=-1)
    proposal_probs = torch.softmax(logits / sample_temperature, dim=-1)
    mode = mode.lower()

    if mode == "topk":
        actions = proposal_probs.topk(branching, dim=-1).indices
    elif mode in {"sample", "sample_without_replacement"}:
        actions = torch.multinomial(proposal_probs, branching, replacement=False)
    elif mode == "sample_with_replacement":
        actions = torch.multinomial(proposal_probs, branching, replacement=True)
    elif mode == "mixed":
        top1 = proposal_probs.argmax(dim=-1, keepdim=True)
        if branching == 1:
            actions = top1
        else:
            sample_probs = proposal_probs.scatter(1, top1, 0.0)
            sample_probs = sample_probs / sample_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sampled = torch.multinomial(sample_probs, branching - 1, replacement=False)
            actions = torch.cat([top1, sampled], dim=1)
    else:
        raise ValueError(f"Unknown counterfactual candidate mode: {mode}")

    if force_actions is not None:
        force_actions = force_actions.long().reshape(-1, 1)
        assert force_actions.shape[0] == actions.shape[0]
        has_force = actions.eq(force_actions).any(dim=1, keepdim=True)
        actions = torch.where(
            has_force,
            actions,
            torch.cat([actions[:, :-1], force_actions], dim=1),
        )

    selected_log_probs = policy_log_probs.gather(1, actions)
    selected_policy_probs = policy_probs.gather(1, actions)
    selected_proposal_probs = proposal_probs.gather(1, actions)
    forced_mask = (
        actions.eq(force_actions)
        if force_actions is not None
        else torch.zeros_like(actions, dtype=torch.bool)
    )
    return (
        actions,
        selected_log_probs,
        selected_policy_probs,
        selected_proposal_probs,
        forced_mask,
    )


def apply_hybrid_first_residual(
    model_residuals: Tensor,
    real_residuals: Tensor,
    replay_action_mask: Tensor,
) -> Tensor:
    assert model_residuals.ndim == 2
    if real_residuals.ndim == 1:
        real_residuals = real_residuals.reshape(-1, 1)
    assert real_residuals.shape == model_residuals[:, :1].shape
    assert replay_action_mask.shape == model_residuals.shape
    return torch.where(replay_action_mask, real_residuals.expand_as(model_residuals), model_residuals)
