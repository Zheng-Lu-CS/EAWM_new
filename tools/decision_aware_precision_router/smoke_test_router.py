#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "EASimulus" / "src"))

    from mechanisms.decision_aware_precision_router import DecisionAwarePrecisionRouter

    torch.manual_seed(0)
    router = DecisionAwarePrecisionRouter(
        embed_dim=32,
        tokens_per_obs=64,
        enabled=True,
        target_keep_ratio=0.5,
        warmup_epochs=0,
        feature_dim=4,
        budget_loss_weight=0.01,
    )
    router.train()

    x = torch.randn(4, 3, 64, 32, requires_grad=True)
    features = torch.randn(4, 1, 64, 2)
    y = router(x, features, epoch=1)
    assert y.shape == x.shape, (y.shape, x.shape)
    losses = router.regularization_losses()
    assert "dapr_budget_loss" in losses
    loss = y.square().mean() + sum(losses.values())
    loss.backward()
    assert x.grad is not None
    assert any(p.grad is not None for p in router.parameters())

    router.eval()
    with torch.no_grad():
        y_eval = router(x.detach(), features, epoch=1)
    assert y_eval.shape == x.shape
    info = router.info()
    assert "dapr_keep_ratio" in info and "dapr_keep_prob" in info
    print("DAPR smoke test passed.")


if __name__ == "__main__":
    main()
