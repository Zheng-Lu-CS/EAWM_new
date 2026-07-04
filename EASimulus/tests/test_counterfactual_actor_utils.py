import sys
import unittest
from pathlib import Path

import torch


ACTOR_CRITIC_DIR = Path(__file__).resolve().parents[1] / "src" / "models" / "actor_critic"
sys.path.insert(0, str(ACTOR_CRITIC_DIR))

from counterfactual import (  # noqa: E402
    compute_uncertainty_weights,
    normalize_tree_advantages,
    robust_tree_backup,
)


class CounterfactualActorUtilsTest(unittest.TestCase):
    def test_lcb_backup(self):
        returns = torch.tensor([[1.0, 2.0, 3.0]])
        backup = robust_tree_backup(returns, mode="lcb", lcb_alpha=1.0)
        expected = returns.mean(dim=1) - returns.std(dim=1, unbiased=False)
        self.assertTrue(torch.allclose(backup, expected))

    def test_cvar_backup_uses_low_tail(self):
        returns = torch.tensor([[1.0, 2.0, 100.0, 200.0]])
        backup = robust_tree_backup(returns, mode="cvar", cvar_fraction=0.5)
        self.assertTrue(torch.equal(backup, torch.tensor([1.5])))

    def test_trimmed_mean_backup(self):
        returns = torch.tensor([[0.0, 1.0, 2.0, 100.0]])
        backup = robust_tree_backup(returns, mode="trimmed_mean", trim_ratio=0.25)
        self.assertTrue(torch.equal(backup, torch.tensor([1.5])))

    def test_normalized_advantage_is_finite_for_equal_returns(self):
        returns = torch.ones(2, 3)
        advantages = torch.zeros_like(returns)
        normalized = normalize_tree_advantages(advantages, returns)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))

    def test_normalized_advantage_clip(self):
        returns = torch.tensor([[0.0, 1.0, 2.0]])
        advantages = torch.tensor([[100.0, 0.0, -100.0]])
        normalized = normalize_tree_advantages(advantages, returns, clip=2.0)
        self.assertLessEqual(normalized.max().item(), 2.0)
        self.assertGreaterEqual(normalized.min().item(), -2.0)

    def test_uncertainty_weights_clamp(self):
        uncertainty = torch.tensor([[0.0, 1.0, 100.0]])
        weights = compute_uncertainty_weights(uncertainty, beta=1.0, min_weight=0.05)
        self.assertTrue(torch.equal(weights[:, :1], torch.ones(1, 1)))
        self.assertGreaterEqual(weights.min().item(), 0.05)
        self.assertLessEqual(weights.max().item(), 1.0)

    def test_uncertainty_weights_beta_zero(self):
        uncertainty = torch.tensor([[0.0, 10.0]])
        weights = compute_uncertainty_weights(uncertainty, beta=0.0)
        self.assertTrue(torch.equal(weights, torch.ones_like(uncertainty)))

    def test_unknown_backup_mode_raises(self):
        with self.assertRaises(ValueError):
            robust_tree_backup(torch.ones(1, 2), mode="unknown")


if __name__ == "__main__":
    unittest.main()
