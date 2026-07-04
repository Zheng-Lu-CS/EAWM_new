from pathlib import Path

from loguru import logger
import torch
from torch import Tensor
import torch.nn as nn

from models.actor_critic import ActorCriticLS
from models.tokenizer import MultiModalTokenizer
from models.world_model import POPWorldModel
from utils import extract_state_dict
from utils.types import MultiModalObs, ObsModality


class Agent(nn.Module):
    def __init__(self, tokenizer: MultiModalTokenizer, world_model: POPWorldModel, actor_critic: ActorCriticLS):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return next(self.parameters()).device

    def _summarize_state_dict_mismatch(
        self,
        module: nn.Module,
        incoming_state_dict: dict,
        module_name: str,
        max_items: int = 20,
    ) -> list[str]:
        current_state_dict = module.state_dict()
        current_keys = set(current_state_dict.keys())
        incoming_keys = set(incoming_state_dict.keys())
        missing = sorted(current_keys - incoming_keys)
        unexpected = sorted(incoming_keys - current_keys)
        mismatched = []
        for key in sorted(current_keys & incoming_keys):
            current_value = current_state_dict[key]
            incoming_value = incoming_state_dict[key]
            if (
                hasattr(current_value, "shape")
                and hasattr(incoming_value, "shape")
                and tuple(current_value.shape) != tuple(incoming_value.shape)
            ):
                mismatched.append(
                    f"{key}: checkpoint{tuple(incoming_value.shape)} != model{tuple(current_value.shape)}"
                )

        def sample(items: list[str]) -> str:
            if not items:
                return "none"
            suffix = "" if len(items) <= max_items else f" ... (+{len(items) - max_items} more)"
            return ", ".join(items[:max_items]) + suffix

        return [
            f"[state-dict][{module_name}] missing={len(missing)}: {sample(missing)}",
            f"[state-dict][{module_name}] unexpected={len(unexpected)}: {sample(unexpected)}",
            f"[state-dict][{module_name}] shape_mismatch={len(mismatched)}: {sample(mismatched)}",
        ]

    def _load_module_state_dict(
        self,
        module: nn.Module,
        incoming_state_dict: dict,
        module_name: str,
        path_to_checkpoint: Path,
    ) -> None:
        try:
            module.load_state_dict(incoming_state_dict)
        except RuntimeError:
            logger.error(
                f"Failed to load {module_name} from checkpoint: {path_to_checkpoint}"
            )
            for line in self._summarize_state_dict_mismatch(
                module, incoming_state_dict, module_name
            ):
                logger.error(line)
            raise

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device, weights_only=True)
        if load_tokenizer:
            self._load_module_state_dict(
                self.tokenizer,
                extract_state_dict(agent_state_dict, 'tokenizer'),
                'tokenizer',
                path_to_checkpoint,
            )
        if load_world_model:
            self._load_module_state_dict(
                self.world_model,
                extract_state_dict(agent_state_dict, 'world_model'),
                'world_model',
                path_to_checkpoint,
            )
        if load_actor_critic:
            self._load_module_state_dict(
                self.actor_critic,
                extract_state_dict(agent_state_dict, 'actor_critic'),
                'actor_critic',
                path_to_checkpoint,
            )

    def act(self, obs: MultiModalObs, should_sample: bool = True, temperature: float = 1.0) ->Tensor:
        assert isinstance(self.actor_critic, ActorCriticLS)
        input_ac = self._embed_obs(obs)
        actions_dist = self.actor_critic(inputs=input_ac)[0].get_actions_distributions(temperature)
        action = actions_dist.sample()[:, -1] if should_sample else actions_dist.mode[:, -1]
        if self.actor_critic.include_action_inputs:
            self.actor_critic.process_action(action)
        return action

    def reset_actor_critic(self, n, burnin_observations: MultiModalObs, mask_padding, actions=None):
        assert isinstance(self.actor_critic, ActorCriticLS)
        b_o = burnin_observations
        if burnin_observations is not None:
            b_o = self._embed_obs(burnin_observations)

        ac_actions_embs = None
        if actions is not None:
            assert actions.dim() in [2, 3]
            # actions_emb = self.world_model.action_embeddings(actions)
            ac_actions_embs = self.actor_critic.embed_action(actions)
            for actions_emb in ac_actions_embs:
                assert actions_emb is None or actions_emb.dim() == 3, f"Got {actions_emb.dim()}"

        return self.actor_critic.reset(n=n, burnin_observations=b_o, mask_padding=mask_padding, ac_actions_embs=ac_actions_embs)

    def _embed_obs(self, obs: MultiModalObs):
        encoded = self.tokenizer.encode(obs, should_preprocess=True)
        encoded = {k: v.z_quantized for k, v in encoded.items()}

        if ObsModality.vector in encoded:
            encoded[ObsModality.vector] = self.tokenizer.tokenizers[ObsModality.vector.name].decode(encoded[ObsModality.vector])

        return encoded
