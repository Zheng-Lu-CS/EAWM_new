import sys
import unittest
from pathlib import Path

import torch


ACTOR_CRITIC_DIR = Path(__file__).resolve().parents[1] / "src" / "models" / "actor_critic"
sys.path.insert(0, str(ACTOR_CRITIC_DIR))

from counterfactual import (  # noqa: E402
    apply_hybrid_first_residual,
    compute_uncertainty_weights,
    normalize_tree_advantages,
    normalize_counterfactual_advantages,
    robust_center,
    robust_tree_backup,
    scale_uncertainty_for_weights,
    select_counterfactual_actions,
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

    def test_relative_uncertainty_scaling_penalizes_high_risk_branches(self):
        uncertainty = torch.tensor([[0.01, 0.02, 0.05], [0.03, 0.03, 0.03]])
        absolute = scale_uncertainty_for_weights(uncertainty, mode="absolute")
        relative = scale_uncertainty_for_weights(uncertainty, mode="relative")
        self.assertTrue(torch.equal(absolute, uncertainty))
        self.assertEqual(relative[0, 0].item(), 0.0)
        self.assertEqual(relative[0, 1].item(), 0.0)
        self.assertGreater(relative[0, 2].item(), 0.0)
        self.assertTrue(torch.equal(relative[1], torch.zeros_like(relative[1])))

    def test_unknown_backup_mode_raises(self):
        with self.assertRaises(ValueError):
            robust_tree_backup(torch.ones(1, 2), mode="unknown")

    def test_robust_center_modes(self):
        values = torch.tensor([[1.0, 2.0, 100.0], [0.0, 4.0, 8.0]])
        self.assertTrue(torch.equal(robust_center(values, "median"), torch.tensor([[2.0], [4.0]])))
        self.assertTrue(torch.equal(robust_center(values, "mean"), values.mean(dim=1, keepdim=True)))
        trimmed = robust_center(torch.tensor([[0.0, 1.0, 2.0, 100.0]]), "trimmed_mean", trim_ratio=0.25)
        self.assertTrue(torch.equal(trimmed, torch.tensor([[1.5]])))

    def test_normalize_counterfactual_advantages_is_finite_and_clipped(self):
        advantages = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, -100.0]])
        normalized = normalize_counterfactual_advantages(advantages, clip=2.0)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertLessEqual(normalized.max().item(), 2.0)
        self.assertGreaterEqual(normalized.min().item(), -2.0)

    def test_normalize_counterfactual_advantages_none_scale(self):
        advantages = torch.tensor([[1.0, -2.0, 3.0]])
        normalized = normalize_counterfactual_advantages(
            advantages, scale="none", clip=0.0
        )
        self.assertTrue(torch.equal(normalized, advantages))

    def test_select_counterfactual_actions_topk(self):
        logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
        actions, log_probs, policy_probs, proposal_probs, forced = select_counterfactual_actions(
            logits, branching=2, mode="topk"
        )
        self.assertTrue(torch.equal(actions, torch.tensor([[1, 3]])))
        self.assertEqual(log_probs.shape, actions.shape)
        self.assertEqual(policy_probs.shape, actions.shape)
        self.assertEqual(proposal_probs.shape, actions.shape)
        self.assertFalse(forced.any())

    def test_select_counterfactual_actions_force_replay_action(self):
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        actions, _, _, _, forced = select_counterfactual_actions(
            logits,
            branching=2,
            mode="topk",
            force_actions=torch.tensor([3]),
        )
        self.assertTrue(torch.equal(actions, torch.tensor([[0, 3]])))
        self.assertTrue(torch.equal(forced, torch.tensor([[False, True]])))

    def test_select_counterfactual_actions_sample_shape(self):
        torch.manual_seed(0)
        logits = torch.zeros(3, 5)
        actions, log_probs, _, _, _ = select_counterfactual_actions(
            logits, branching=4, mode="sample"
        )
        self.assertEqual(actions.shape, (3, 4))
        self.assertEqual(log_probs.shape, actions.shape)
        for row in actions:
            self.assertEqual(row.unique().numel(), row.numel())

    def test_hybrid_first_residual_only_replaces_replay_branch(self):
        model = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        real = torch.tensor([[10.0], [20.0]])
        mask = torch.tensor([[False, True, False], [True, False, False]])
        corrected = apply_hybrid_first_residual(model, real, mask)
        expected = torch.tensor([[1.0, 10.0, 3.0], [20.0, 5.0, 6.0]])
        self.assertTrue(torch.equal(corrected, expected))


if __name__ == "__main__":
    unittest.main()
