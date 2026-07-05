import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    from agent import Agent  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env deps
    if exc.name != "hydra":
        raise
    Agent = None


class AgentLoadingTest(unittest.TestCase):
    @unittest.skipIf(Agent is None, "hydra is not installed in this environment")
    def test_non_strict_module_load_allows_missing_new_head(self):
        agent = Agent(None, None, None)
        module = nn.Linear(2, 1)
        incoming = {"weight": torch.ones_like(module.weight)}
        agent._load_module_state_dict(
            module,
            incoming,
            "actor_critic",
            Path("checkpoint.pt"),
            strict=False,
        )
        self.assertTrue(torch.equal(module.weight, incoming["weight"]))

    @unittest.skipIf(Agent is None, "hydra is not installed in this environment")
    def test_strict_module_load_rejects_missing_key(self):
        agent = Agent(None, None, None)
        module = nn.Linear(2, 1)
        incoming = {"weight": torch.ones_like(module.weight)}
        with self.assertRaises(RuntimeError):
            agent._load_module_state_dict(
                module,
                incoming,
                "actor_critic",
                Path("checkpoint.pt"),
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
