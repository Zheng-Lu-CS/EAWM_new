import sys
import unittest
from pathlib import Path

import torch


ACTOR_CRITIC_DIR = Path(__file__).resolve().parents[1] / "src" / "models" / "actor_critic"
sys.path.insert(0, str(ACTOR_CRITIC_DIR))

from counterfactual import (  # noqa: E402
    compute_counterfactual_group_advantages,
    compute_counterfactual_group_baseline,
    compute_uncertainty_weights,
)


class CounterfactualActorUtilsTest(unittest.TestCase):
    def test_median_baseline(self):
        returns = torch.tensor([[1.0, 3.0, 2.0], [10.0, -1.0, 0.0]])
        baseline = compute_counterfactual_group_baseline(returns, baseline="median")
        self.assertTrue(torch.equal(baseline, torch.tensor([[2.0], [0.0]])))

    def test_trimmed_mean_baseline(self):
        returns = torch.tensor([[0.0, 1.0, 2.0, 100.0]])
        baseline = compute_counterfactual_group_baseline(
            returns, baseline="trimmed_mean", trim_ratio=0.25
        )
        self.assertTrue(torch.equal(baseline, torch.tensor([[1.5]])))

    def test_zero_std_advantage_is_finite_zero(self):
        returns = torch.ones(2, 4)
        advantages, baseline, std = compute_counterfactual_group_advantages(returns)
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))
        self.assertTrue(torch.equal(baseline, torch.ones(2, 1)))
        self.assertTrue(torch.equal(std, torch.zeros(2, 1)))

    def test_uncertainty_weights_clamp(self):
        uncertainty = torch.tensor([[0.0, 1.0, 100.0]])
        weights = compute_uncertainty_weights(uncertainty, beta=1.0, min_weight=0.05)
        self.assertTrue(torch.equal(weights[:, :1], torch.ones(1, 1)))
        self.assertGreaterEqual(weights.min().item(), 0.05)
        self.assertLessEqual(weights.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
