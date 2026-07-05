from abc import abstractmethod, ABC
from math import ceil
from typing import Any, Optional
import sys

from loguru import logger
from einops import rearrange
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from yet_another_retnet.retnet import RetNetDecoder, RetNetDecoderLayer
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import POPWorldModelEnv
from models.actor_critic.counterfactual import (
    apply_hybrid_first_residual,
    compute_uncertainty_weights,
    normalize_counterfactual_advantages,
    normalize_tree_advantages,
    robust_center,
    robust_tree_backup,
    scale_uncertainty_for_weights,
    select_counterfactual_actions,
)
from models.actor_critic.encoders import ObsEncoderBase
from models.actor_critic.types import *
from models.tokenizer import MultiModalTokenizer
from models.world_model import POPWorldModel
from models.embedding import make_mlp, MultiDiscreteEmbedding
from utils import (
    compute_lambda_returns,
    LossWithIntermediateLosses,
    QuantizedContinuousDistribution,
    LSTMCellWrapper,
    ObsModality,
    HLGaussCategoricalRegressionHead,
    RecurrentState,
)
from utils.types import MultiModalObs
from utils.preprocessing import BufferScaler
from utils.distributions import (
    CategoricalDistribution,
    SquashedDiagNormalDistribution,
    MultiCategoricalDistribution,
)


class ActorCriticLS(nn.Module):
    """
    This version works in latent space. it receives the token codes of the observation frame (embed_dim, w, h)
    and applies a CNN + dense to map it to a latent vector which is the input to the LSTM.
    Importantly, the token codes are the learned codes of the tokenizer.
    """

    def __init__(
        self,
        obs_encoders: dict[ObsModality, ObsEncoderBase],
        obs_mlp_layer_sizes: list[int],
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.obs_encoders = nn.ModuleDict({k.name: v for k, v in obs_encoders.items()})
        self.separate_networks = separate_networks
        self.obs_latent_fuser = make_mlp(
            layer_dims=[sum([enc.out_dim for enc in self.obs_encoders.values()])]
            + obs_mlp_layer_sizes,
            linear_out=False,
            device=device,
        )
        self._ordered_modalities = [
            modality for modality in ObsModality if modality.name in self.obs_encoders
        ]
        self.device = device
        self.name = name
        self.include_action_inputs = include_action_inputs
        self.context_len = context_len
        self.real_reward_weight = real_reward_weight
        self.intrinsic_reward_weight = intrinsic_reward_weight

        self.lstm_dim = obs_mlp_layer_sizes[-1]
        self.embed_dim = obs_mlp_layer_sizes[-1]

        self.rnn_type = rnn_type
        self.actor, self.actor_state = self._build_rnn_model()

        if self.separate_networks:
            self.critic_encoders = nn.ModuleDict(
                {
                    k.name: v.build_another() if self.separate_networks else None
                    for k, v in obs_encoders.items()
                }
            )
            self.critic_latent_fuser = make_mlp(
                layer_dims=[sum([enc.out_dim for enc in self.obs_encoders.values()])]
                + obs_mlp_layer_sizes,
                linear_out=False,
                device=device,
            )
            self.critic, self.critic_state = self._build_rnn_model()
        else:
            self.critic_encoders = {k.name: None for k, v in obs_encoders.items()}
            self.critic_latent_fuser = None
            self.critic, self.critic_state = None, None

        self.critic_v_head = self._build_critic_head()
        self.actor_linear = self._build_actor_head()
        self.dr_q_head = None

        # self.action_emb_map = nn.Linear(token_embed_dim, lstm_latent_dim) if token_embed_dim != lstm_latent_dim else None
        self.action_emb_map = None

        self.return_scaler = BufferScaler()

        logger.info(
            f"Initialized ActorCriticLS (separate networks: {separate_networks})"
        )
        logger.info(
            f"reward weights: real: {self.real_reward_weight}, intrinsic: {self.intrinsic_reward_weight}"
        )

    def _build_rnn_model(self):
        if self.rnn_type == "lstm":
            lstm = nn.LSTM(
                self.embed_dim,
                self.lstm_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
                device=self.device,
            )
            lstm = LSTMCellWrapper(lstm)
            # lstm = nn.LSTMCell(self.lstm_dim, self.lstm_dim, device=self.device)
            lstm_state = (None, None)
            return lstm, lstm_state

        elif self.rnn_type == "gru":
            gru = nn.GRUCell(self.embed_dim, self.lstm_dim, device=self.device)
            gru_state = None
            return gru, gru_state

    @abstractmethod
    def _build_actor_head(self) -> nn.Module:
        pass

    def _build_critic_head(self) -> nn.Module:
        return nn.Sequential(
            # nn.LayerNorm(self.lstm_dim),
            nn.Linear(self.lstm_dim, 1, device=self.device)
        )

    def __repr__(self) -> str:
        return self.name

    def clear(self) -> None:
        if self.rnn_type == "lstm":
            self.actor_state = (None, None)
            self.critic_state = (None, None) if self.separate_networks else None
        elif self.rnn_type == "gru":
            self.actor_state = None
            self.critic_state = None

    def get_zero_rnn_state(self, n, device, rnn_type: str = None):
        if rnn_type is None:
            rnn_type = self.rnn_type

        if rnn_type == "lstm":
            return torch.zeros(n, self.lstm_dim, device=device), torch.zeros(
                n, self.lstm_dim, device=device
            )
        elif rnn_type == "gru":
            return torch.zeros(n, self.lstm_dim, device=device)
        else:
            assert False, f"rnn type '{rnn_type}' not supported"

    def embed_obs(self, observation: MultiModalObs) -> tuple[Tensor, Tensor]:
        assert set([k.name for k in observation.keys()]) == set(
            self.obs_encoders.keys()
        ), f"{set(observation.keys())} != {set(self.obs_encoders.keys())}"

        actor_emb = torch.cat(
            [
                self.obs_encoders[k.name](observation[k])
                for k in self._ordered_modalities
            ],
            dim=-1,
        )
        actor_emb = self.obs_latent_fuser(actor_emb)
        critic_emb = None
        if self.separate_networks:
            critic_emb = torch.cat(
                [
                    self.critic_encoders[k.name](observation[k])
                    for k in self._ordered_modalities
                ],
                dim=-1,
            )
            critic_emb = self.critic_latent_fuser(critic_emb)

        return actor_emb, critic_emb

    @abstractmethod
    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        pass

    def process_action(self, action: Tensor, mask_padding: Tensor = None):
        assert self.include_action_inputs
        assert action is not None

        x = self.embed_action(action)
        return self.process_action_emb(x, mask_padding)

    def process_action_emb(
        self, ac_action_embs: tuple[Tensor, Tensor], mask_padding: Tensor = None
    ):
        assert self.include_action_inputs and ac_action_embs[0].dim() == 2 and (
            not self.separate_networks or ac_action_embs[1].dim() == 2
        ), f"Got {ac_action_embs[0].dim()} ({ac_action_embs[0].shape})"
        self.actor_state = self._rnn_forward(
            ac_action_embs[0], mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            ac_action_embs[1], mask_padding, self.critic, self.critic_state
        )

    def reset(
        self,
        n: int,
        burnin_observations: Optional[MultiModalObs] = None,
        mask_padding: Optional[Tensor] = None,
        ac_actions_embs: tuple[Tensor, Tensor] = None,
    ) -> None:
        assert ac_actions_embs is None or (
            ac_actions_embs[0].dim() == 3
            and (not self.separate_networks or ac_actions_embs[1].dim() == 3)
        )  # (b, t, e)

        device = self.device
        self.actor_state = self.get_zero_rnn_state(n, device)
        self.critic_state = (
            self.get_zero_rnn_state(n, device) if self.separate_networks else None
        )
        if burnin_observations is not None:
            batch_dims = set(v.shape[:2] for v in burnin_observations.values())
            assert len(batch_dims) == 1
            batch_dims = batch_dims.pop()
            assert (
                batch_dims[0] == n
                and mask_padding is not None
                and batch_dims == mask_padding.shape
            )
            assert batch_dims == ac_actions_embs[0].shape[:2]
            for i in range(batch_dims[1]):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(
                            {k: v[:, i] for k, v in burnin_observations.items()},
                            mask_padding[:, i],
                        )
                        # if self.include_action_inputs:
                        #     cur_actions_embs = (ac_actions_embs[0][:, i],
                        #                         ac_actions_embs[1][:, i] if ac_actions_embs[1] is not None else None)
                        #     self.process_action_emb(ac_action_embs=cur_actions_embs, mask_padding=mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        if self.rnn_type == "lstm":
            hx, cx = self.actor_state
            hx = hx[mask]
            cx = cx[mask]
            self.actor_state = (hx, cx)
        elif self.rnn_type == "gru":
            self.actor_state = self.actor_state[mask]

    def _rnn_forward(self, x, mask_padding, model, rnn_state):
        if model is None:
            # Shared actor-critic network
            return None

        if mask_padding is None:
            rnn_state = model(x, rnn_state)
        else:
            if self.rnn_type == "lstm":
                hx, cx = rnn_state
                hx[mask_padding], cx[mask_padding] = model(
                    x[mask_padding], (hx[mask_padding], cx[mask_padding])
                )
                rnn_state = (hx, cx)

            elif self.rnn_type == "gru":
                rnn_state[mask_padding] = model(x, rnn_state[mask_padding])
            else:
                assert False, f"rnn type '{self.rnn_type}' not supported"
        return rnn_state

    def rnn_forward(self, x, mask_padding):
        self.actor_state = self._rnn_forward(
            x, mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            x, mask_padding, self.critic, self.critic_state
        )

    def get_rnn_output(self):
        if self.rnn_type == "lstm":
            return self.actor_state[0]
        elif self.rnn_type == "gru":
            return self.actor_state
        else:
            assert False, f"rnn type '{self.rnn_type}' not supported"

    def get_critic_rnn_output(self):
        if not self.separate_networks:
            return self.get_rnn_output()

        if self.rnn_type == "lstm":
            return self.critic_state[0]
        elif self.rnn_type == "gru":
            return self.critic_state
        else:
            assert False, f"rnn type '{self.rnn_type}' not supported"

    def forward(
        self, inputs: MultiModalObs, mask_padding: Optional[torch.BoolTensor] = None
    ) -> tuple[ActorOutput, CriticOutput]:
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.any())
        x_actor, x_critic = self.embed_obs(inputs)  # (b, d_lstm)
        self.actor_state = self._rnn_forward(
            x_actor, mask_padding, self.actor, self.actor_state
        )
        self.critic_state = self._rnn_forward(
            x_critic, mask_padding, self.critic, self.critic_state
        )

        return (
            self._compute_actor_output(self.get_rnn_output()),
            self._compute_critic_output(self.get_critic_rnn_output()),
        )

    @abstractmethod
    def _compute_actor_output(self, actor_latent) -> ActorOutput:
        pass

    @abstractmethod
    def _compute_critic_output(self, critic_latent) -> CriticOutput:
        pass

    def _get_actions_distribution(self, outputs: ImagineOutput):
        pass

    def _get_values_means(self, values_info: ValuesInfo) -> Tensor:
        return values_info.value_means

    def compute_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        imagine_horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        epoch: int,
        actor_start_epoch: int,
        imagine: bool,
        **kwargs: Any,
    ) -> tuple[LossWithIntermediateLosses, dict]:
        actor_loss_mode = kwargs.get("actor_loss_mode", "dreamer")
        if imagine and actor_loss_mode == "tree_counterfactual":
            return self._compute_tree_counterfactual_loss(
                batch=batch,
                tokenizer=tokenizer,
                world_model=world_model,
                imagine_horizon=imagine_horizon,
                gamma=gamma,
                lambda_=lambda_,
                entropy_weight=entropy_weight,
                epoch=epoch,
                actor_start_epoch=actor_start_epoch,
                **kwargs,
            )
        if imagine and actor_loss_mode == "doubly_robust_counterfactual":
            return self._compute_doubly_robust_counterfactual_loss(
                batch=batch,
                tokenizer=tokenizer,
                world_model=world_model,
                imagine_horizon=imagine_horizon,
                gamma=gamma,
                lambda_=lambda_,
                entropy_weight=entropy_weight,
                epoch=epoch,
                actor_start_epoch=actor_start_epoch,
                **kwargs,
            )
        if imagine and actor_loss_mode != "dreamer":
            raise ValueError(f"Unknown actor_loss_mode: {actor_loss_mode}")

        if imagine:
            outputs = self.imagine(
                batch, tokenizer, world_model, horizon=imagine_horizon
            )
            info_prefix = "imagined_"
        else:
            outputs = self.get_action_distribution_and_values(
                batch, tokenizer, world_model, horizon=imagine_horizon
            )
            info_prefix = "real_"
        # outputs = self.play_env(batch, tokenizer, world_model, horizon=imagine_horizon)

        values_means = self._get_values_means(outputs.values_info)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=values_means,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]
        self.return_scaler.update(lambda_returns)
        returns_scale = torch.maximum(
            torch.ones_like(self.return_scaler.scale), self.return_scaler.scale * 0.5
        )

        values = values_means[:, :-1]

        d = outputs.actions_distributions
        log_probs = d.log_prob(outputs.actions)[:, :-1]
        advantage = (lambda_returns - values).detach() / returns_scale
        loss_actions = -(log_probs * advantage.detach()).mean()

        loss_actor = loss_actions
        loss_entropy = -entropy_weight * d.entropy().mean()
        if epoch < actor_start_epoch:
            loss_actor = torch.zeros_like(loss_actor)
            loss_entropy = torch.zeros_like(loss_entropy)

        loss_values = self._compute_critic_loss(outputs.values_info, lambda_returns)

        info = {
            info_prefix + "rewards": outputs.rewards.detach().clone(),
            info_prefix + "returns": lambda_returns.detach().clone(),
            info_prefix + "values": values.detach().clone(),
            info_prefix + "normalized_advantage": advantage.detach().clone(),
            info_prefix + "log_probs": log_probs.detach().clone(),
            info_prefix + "returns_scale": returns_scale.item(),
        }
        if imagine:
            intermediatelosses = LossWithIntermediateLosses(
                imagined_loss_actor=loss_actor,
                imagined_loss_values=loss_values,
                imagined_loss_entropy=loss_entropy,
            )
        else:
            intermediatelosses = LossWithIntermediateLosses(
                real_loss_actor=loss_actor,
                real_loss_values=loss_values,
                real_loss_entropy=loss_entropy,
            )
        return intermediatelosses, info

    def _compute_tree_counterfactual_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        imagine_horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        epoch: int,
        actor_start_epoch: int,
        **kwargs: Any,
    ) -> tuple[LossWithIntermediateLosses, dict]:
        loss_actions, tree_entropy, tree_info = self._build_tree_counterfactual_actor_loss(
            batch=batch,
            tokenizer=tokenizer,
            world_model=world_model,
            horizon=imagine_horizon,
            gamma=gamma,
            entropy_weight=entropy_weight,
            **kwargs,
        )

        if epoch < actor_start_epoch:
            loss_actions = torch.zeros_like(loss_actions)
            tree_entropy = torch.zeros_like(tree_entropy)

        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)
        values_means = self._get_values_means(outputs.values_info)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=values_means,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]
        self.return_scaler.update(lambda_returns)
        returns_scale = torch.maximum(
            torch.ones_like(self.return_scaler.scale), self.return_scaler.scale * 0.5
        )

        values = values_means[:, :-1]
        loss_values = self._compute_critic_loss(outputs.values_info, lambda_returns)

        info = {
            "imagined_rewards": outputs.rewards.detach().clone(),
            "imagined_returns": lambda_returns.detach().clone(),
            "imagined_values": values.detach().clone(),
            "imagined_normalized_advantage": (
                (lambda_returns - values).detach() / returns_scale
            ).detach().clone(),
            "imagined_log_probs": outputs.actions_distributions.log_prob(
                outputs.actions
            )[:, :-1].detach().clone(),
            "imagined_returns_scale": returns_scale.item(),
            **tree_info,
        }
        intermediatelosses = LossWithIntermediateLosses(
            imagined_loss_actor=loss_actions,
            imagined_loss_values=loss_values,
            imagined_loss_entropy=tree_entropy,
        )
        return intermediatelosses, info

    def _compute_doubly_robust_counterfactual_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        imagine_horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        epoch: int,
        actor_start_epoch: int,
        **kwargs: Any,
    ) -> tuple[LossWithIntermediateLosses, dict]:
        loss_actions, dr_entropy, q_loss, dr_info = (
            self._build_doubly_robust_counterfactual_actor_loss(
                batch=batch,
                tokenizer=tokenizer,
                world_model=world_model,
                horizon=imagine_horizon,
                gamma=gamma,
                lambda_=lambda_,
                entropy_weight=entropy_weight,
                **kwargs,
            )
        )

        if epoch < actor_start_epoch:
            loss_actions = torch.zeros_like(loss_actions)
            dr_entropy = torch.zeros_like(dr_entropy)

        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)
        values_means = self._get_values_means(outputs.values_info)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=values_means,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]
        self.return_scaler.update(lambda_returns)
        returns_scale = torch.maximum(
            torch.ones_like(self.return_scaler.scale), self.return_scaler.scale * 0.5
        )

        values = values_means[:, :-1]
        loss_values = self._compute_critic_loss(outputs.values_info, lambda_returns)
        q_loss_weight = float(kwargs.get("dr_q_loss_weight", 0.5))

        info = {
            "imagined_rewards": outputs.rewards.detach().clone(),
            "imagined_returns": lambda_returns.detach().clone(),
            "imagined_values": values.detach().clone(),
            "imagined_normalized_advantage": (
                (lambda_returns - values).detach() / returns_scale
            ).detach().clone(),
            "imagined_log_probs": outputs.actions_distributions.log_prob(
                outputs.actions
            )[:, :-1].detach().clone(),
            "imagined_returns_scale": returns_scale.item(),
            **dr_info,
        }
        intermediatelosses = LossWithIntermediateLosses(
            imagined_loss_actor=loss_actions,
            imagined_loss_values=loss_values,
            imagined_loss_entropy=dr_entropy,
            imagined_loss_q=q_loss * q_loss_weight,
        )
        return intermediatelosses, info

    def _values_to_column(self, values_info: ValuesInfo) -> Tensor:
        values = self._get_values_means(values_info)
        return values.reshape(values.shape[0], -1)[:, :1]

    def _compute_dr_q_values(
        self, critic_latent: Tensor, detach_latent: bool = False
    ) -> Tensor:
        q_head = getattr(self, "dr_q_head", None)
        if q_head is None:
            raise ValueError(
                "doubly_robust_counterfactual requires training.actor_critic.dr_q_head=True"
            )
        if detach_latent:
            critic_latent = critic_latent.detach()
        return q_head(critic_latent)

    def _build_doubly_robust_counterfactual_actor_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        gamma: float,
        lambda_: float,
        entropy_weight: float,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        if not hasattr(self, "num_actions"):
            raise ValueError(
                "doubly_robust_counterfactual currently supports discrete Atari actors only."
            )
        if getattr(self, "dr_q_head", None) is None:
            raise ValueError(
                "doubly_robust_counterfactual requires training.actor_critic.dr_q_head=True"
            )

        effective_horizon = horizon - self.context_len + 1
        rollout_horizon = min(
            int(kwargs.get("dr_rollout_horizon", 4)), effective_horizon
        )
        branching = min(int(kwargs.get("dr_branching", 4)), int(self.num_actions))
        if rollout_horizon < 1:
            raise ValueError("dr_rollout_horizon must be >= 1")
        if branching < 2:
            raise ValueError("dr_branching must be >= 2")

        candidate_mode = kwargs.get("dr_candidate_mode", "topk")
        center_mode = kwargs.get("dr_center", "median")
        sample_temperature = float(kwargs.get("dr_sample_temperature", 1.0))
        adv_eps = float(kwargs.get("dr_adv_eps", 1e-6))
        adv_clip = float(kwargs.get("dr_adv_clip", 5.0))
        adv_scale = kwargs.get("dr_adv_scale", "std")
        uncertainty_beta = float(kwargs.get("dr_uncertainty_beta", 1.0))
        uncertainty_min_weight = float(kwargs.get("dr_uncertainty_weight_min", 0.05))
        trim_ratio = float(kwargs.get("dr_trim_ratio", 0.25))
        is_clip = float(kwargs.get("dr_is_clip", 2.0))
        q_loss_type = kwargs.get("dr_q_loss_type", "mse").lower()
        q_huber_delta = float(kwargs.get("dr_q_huber_delta", 1.0))
        q_target_clip = float(kwargs.get("dr_q_target_clip", 0.0))
        q_detach_latent = bool(kwargs.get("dr_q_detach_latent", False))
        real_first_residual = bool(kwargs.get("dr_real_first_residual", False))
        force_replay_action = bool(
            kwargs.get("dr_force_replay_action", real_first_residual)
        )
        anchor_index = -2 if real_first_residual else -1

        wm_env, obs_tokens = self._imagination_set_initial_state(
            batch, tokenizer, world_model, anchor_index=anchor_index
        )
        batch_size = batch["mask_padding"].shape[0]
        current_wm_snapshot = self._snapshot_wm_env(wm_env, detach=True)
        current_actor_state = self._clone_rnn_state(self.actor_state, detach=True)
        current_critic_state = self._clone_rnn_state(self.critic_state, detach=True)

        self.actor_state = current_actor_state
        self.critic_state = current_critic_state
        self._restore_wm_env(wm_env, current_wm_snapshot)

        obs_codes = self._to_codes(obs_tokens, world_model, tokenizer)
        actor_outs, critic_outs = self(inputs=obs_codes)
        actions_dist = actor_outs.get_actions_distributions()
        logits = actions_dist.logits[:, 0]
        replay_actions = (
            batch["actions"][:, anchor_index].long()
            if force_replay_action or real_first_residual
            else None
        )
        actions, log_probs, policy_probs, proposal_probs, replay_action_mask = (
            select_counterfactual_actions(
                logits=logits,
                branching=branching,
                mode=candidate_mode,
                sample_temperature=sample_temperature,
                force_actions=replay_actions,
            )
        )
        v0 = self._values_to_column(critic_outs.get_value_info())
        q0_all = self._compute_dr_q_values(
            self.get_critic_rnn_output(), detach_latent=q_detach_latent
        )
        q0_selected = q0_all.gather(1, actions)
        anchor_entropy = actions_dist.entropy().reshape(batch_size, -1)[:, 0]

        post_anchor_actor_state = self._clone_rnn_state(self.actor_state, detach=True)
        post_anchor_critic_state = self._clone_rnn_state(self.critic_state, detach=True)

        real_first_residuals = None
        real_first_targets = None
        if real_first_residual:
            with torch.no_grad():
                saved_actor_state = self._clone_rnn_state(self.actor_state, detach=True)
                saved_critic_state = self._clone_rnn_state(
                    self.critic_state, detach=True
                )
                self.actor_state = self._clone_rnn_state(
                    post_anchor_actor_state, detach=True
                )
                self.critic_state = self._clone_rnn_state(
                    post_anchor_critic_state, detach=True
                )
                self.process_action(replay_actions)
                real_next_index = (
                    batch["mask_padding"].shape[1] + anchor_index + 1
                    if anchor_index < 0
                    else anchor_index + 1
                )
                real_next_obs = {
                    k: v[:, real_next_index : real_next_index + 1]
                    for k, v in batch["observations"].items()
                }
                real_next_tokens = world_model.get_obs_tokens(
                    real_next_obs, tokenizer=tokenizer
                )
                real_next_tokens = {
                    k: v[:, 0] for k, v in real_next_tokens.items()
                }
                real_next_codes = self._to_codes(
                    real_next_tokens, world_model, tokenizer
                )
                _, real_next_critic_outs = self(inputs=real_next_codes)
                real_next_values = self._values_to_column(
                    real_next_critic_outs.get_value_info()
                )
                self.actor_state = saved_actor_state
                self.critic_state = saved_critic_state

                real_rewards = batch["rewards"][:, anchor_index].reshape(-1, 1)
                real_ends = batch["ends"][:, anchor_index].reshape(-1, 1).bool()
                real_q = q0_all.gather(1, replay_actions.reshape(-1, 1))
                real_first_targets = (
                    real_rewards
                    + gamma * real_ends.logical_not().float() * real_next_values
                ).detach()
                real_first_residuals = (real_first_targets - real_q).detach()

        self.actor_state = self._repeat_rnn_state(post_anchor_actor_state, branching)
        self.critic_state = self._repeat_rnn_state(post_anchor_critic_state, branching)
        self._restore_wm_env(
            wm_env, self._repeat_wm_snapshot(current_wm_snapshot, branching)
        )

        current_actions = actions.reshape(-1, 1)
        current_q = q0_selected.reshape(-1)
        alive = torch.ones(
            batch_size * branching, dtype=torch.bool, device=current_actions.device
        )
        residual_sum = torch.zeros(
            batch_size, branching, dtype=q0_selected.dtype, device=q0_selected.device
        )
        uncertainty_sum = torch.zeros_like(residual_sum)
        uncertainty_count = torch.zeros_like(residual_sum)
        q_preds = []
        q_targets = []
        q_masks = []

        for step in range(rollout_horizon):
            self.process_action(current_actions.squeeze(1))
            next_obs_tokens, reward, done, step_info = wm_env.step(
                current_actions,
                should_predict_next_obs=True,
                return_tokens=True,
            )
            reward = reward.reshape(-1)
            done = done.reshape(-1).bool()
            uncertainty = None
            if isinstance(step_info, dict):
                uncertainty = step_info.get("uncertainty")
            if uncertainty is None:
                uncertainty = torch.zeros_like(reward.reshape(-1, 1))
            uncertainty = uncertainty.reshape(-1)

            next_codes = self._to_codes(next_obs_tokens, world_model, tokenizer)
            next_actor_outs, next_critic_outs = self(inputs=next_codes)
            next_values = self._values_to_column(
                next_critic_outs.get_value_info()
            ).reshape(-1)

            td_targets = (
                reward + gamma * done.logical_not().float() * next_values.detach()
            )
            residuals = td_targets - current_q
            residuals = residuals.reshape(batch_size, branching)
            td_targets = td_targets.reshape(batch_size, branching)
            current_q_2d = current_q.reshape(batch_size, branching)
            active = alive.reshape(batch_size, branching)

            if step == 0 and real_first_residual:
                residuals = apply_hybrid_first_residual(
                    residuals,
                    real_first_residuals,
                    replay_action_mask,
                )
                td_targets = torch.where(
                    replay_action_mask,
                    real_first_targets.expand_as(td_targets),
                    td_targets,
                )

            discount = (gamma * lambda_) ** step
            residual_sum = residual_sum + discount * residuals.detach() * active.float()
            uncertainty_sum = uncertainty_sum + uncertainty.reshape(
                batch_size, branching
            ).detach() * active.float()
            uncertainty_count = uncertainty_count + active.float()

            q_preds.append(current_q_2d.reshape(-1))
            q_target_for_loss = td_targets.detach()
            if q_target_clip > 0:
                q_target_for_loss = q_target_for_loss.clamp(
                    min=-q_target_clip, max=q_target_clip
                )
            q_targets.append(q_target_for_loss.reshape(-1))
            q_masks.append(active.reshape(-1))

            alive = alive & done.logical_not()
            if step < rollout_horizon - 1:
                next_actions_dist = next_actor_outs.get_actions_distributions()
                next_actions = next_actions_dist.sample().reshape(-1)
                next_q_all = self._compute_dr_q_values(
                    self.get_critic_rnn_output(), detach_latent=q_detach_latent
                )
                current_q = next_q_all.gather(1, next_actions.reshape(-1, 1)).reshape(-1)
                current_actions = next_actions.reshape(-1, 1)

        q_preds = torch.cat(q_preds)
        q_targets = torch.cat(q_targets)
        q_masks = torch.cat(q_masks)
        if q_masks.any().item():
            q_preds_active = q_preds[q_masks]
            q_targets_active = q_targets[q_masks]
            if q_loss_type in {"huber", "smooth_l1"}:
                q_loss = F.smooth_l1_loss(
                    q_preds_active,
                    q_targets_active,
                    beta=max(q_huber_delta, 1e-6),
                )
            elif q_loss_type == "mse":
                q_loss = F.mse_loss(q_preds_active, q_targets_active)
            else:
                raise ValueError(f"Unknown dr_q_loss_type: {q_loss_type}")
        else:
            q_loss = torch.zeros_like(q_preds.mean())

        dr_advantages = (q0_selected.detach() - v0.detach()) + residual_sum
        centers = robust_center(
            dr_advantages,
            mode=center_mode,
            trim_ratio=trim_ratio,
        )
        centered_advantages = dr_advantages - centers
        normalized_advantages = normalize_counterfactual_advantages(
            centered_advantages,
            eps=adv_eps,
            clip=adv_clip,
            scale=adv_scale,
        )

        branch_uncertainty = uncertainty_sum / uncertainty_count.clamp_min(1.0)
        weights = compute_uncertainty_weights(
            branch_uncertainty,
            beta=uncertainty_beta,
            min_weight=uncertainty_min_weight,
        )
        is_ratio = policy_probs / proposal_probs.clamp_min(1e-8)
        if is_clip > 0:
            is_ratio = is_ratio.clamp(max=is_clip)
        weights = weights * is_ratio.detach()

        loss_actions = -(
            weights.detach() * normalized_advantages.detach() * log_probs
        ).mean()
        loss_entropy = -entropy_weight * anchor_entropy.mean()
        self.clear()

        dr_info = {
            "imagined_dr_advantages": normalized_advantages.detach().reshape(-1),
            "imagined_dr_raw_advantages": dr_advantages.detach().reshape(-1),
            "imagined_dr_centered_advantages": centered_advantages.detach().reshape(-1),
            "imagined_dr_residuals": residual_sum.detach().reshape(-1),
            "imagined_dr_q_values": q0_selected.detach().reshape(-1),
            "imagined_dr_v_values": v0.detach().expand_as(q0_selected).reshape(-1),
            "imagined_dr_weights": weights.detach().reshape(-1),
            "imagined_dr_uncertainty": branch_uncertainty.detach().reshape(-1),
            "imagined_dr_log_probs": log_probs.detach().reshape(-1),
            "imagined_dr_policy_probs": policy_probs.detach().reshape(-1),
            "imagined_dr_is_ratio": is_ratio.detach().reshape(-1),
            "imagined_dr_replay_action_mask": replay_action_mask.float().reshape(-1),
            "imagined_dr_q_preds": q_preds.detach(),
            "imagined_dr_q_targets": q_targets.detach(),
        }
        return loss_actions, loss_entropy, q_loss, dr_info

    def _compute_critic_loss(
        self, values_info: ContinuousValuesInfo, targets
    ) -> Tensor:
        values = values_info.value_means
        return F.mse_loss(values[:, :-1], targets)

    def _imagination_set_initial_state(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        anchor_index: int = -1,
    ) -> tuple[POPWorldModelEnv, MultiModalObs]:
        device = batch["mask_padding"].device

        wm_env = POPWorldModelEnv(
            tokenizer,
            world_model,
            device=device,
            real_reward_weight=self.real_reward_weight,
            intrinsic_reward_weight=self.intrinsic_reward_weight,
        )

        batch_size, seq_len = batch["mask_padding"].shape[:2]
        mask_padding = batch["mask_padding"]
        if anchor_index < 0:
            anchor_index = seq_len + anchor_index
        assert 0 <= anchor_index < seq_len, f"anchor_index={anchor_index}, seq_len={seq_len}"
        assert mask_padding[:, anchor_index].all()

        # set the initial state of the actor-critic:
        with torch.no_grad():
            obs_tokens = world_model.get_obs_tokens(
                batch["observations"], tokenizer=tokenizer
            )
        obs_quantized = self._to_codes(obs_tokens, world_model, tokenizer)

        # Ignore last obs as it could be the last obs (obtained after termination signal)
        burnin_observations = (
            {k: v[:, :anchor_index] for k, v in obs_quantized.items()}
            if anchor_index > 0
            else None
        )
        ac_actions_embs = (
            self.embed_action(batch["actions"][:, :anchor_index])
            if anchor_index > 0
            else None
        )
        self.reset(
            n=batch_size,
            burnin_observations=burnin_observations,
            mask_padding=mask_padding[:, :anchor_index] if anchor_index > 0 else None,
            ac_actions_embs=ac_actions_embs,
        )

        # reset WM env:
        ctx_len = self.context_len
        action_seq_len = world_model.tokens_per_action
        ctx_start = anchor_index - ctx_len
        assert ctx_start >= 0, f"anchor_index={anchor_index} must be >= context_len={ctx_len}"
        ctx = world_model.get_tokens_emb(
            {k: obs_tokens[k][:, ctx_start:anchor_index] for k in obs_tokens.keys()},
            batch["actions"][:, ctx_start:anchor_index],
            tokenizer=tokenizer,
        ).flatten(1, 2)[:, :-action_seq_len]

        wm_env.reset_from_initial_observations(
            ctx,
            return_tokens=True,
        )

        return wm_env, {k: v[:, anchor_index] for k, v in obs_tokens.items()}

    @abstractmethod
    def _concat_distributions(
        self, all_actions_dists
    ) -> torch.distributions.Distribution:
        pass

    def _concat_values(
        self, values_info_list: list[ContinuousValuesInfo]
    ) -> ContinuousValuesInfo:
        values_means = [v.values for v in values_info_list]
        return ContinuousValuesInfo(
            rearrange(torch.stack(values_means, dim=1), "b t 1 -> b t")
        )

    def _to_codes(
        self,
        obs_tokens: MultiModalObs,
        world_model: POPWorldModel,
        tokenizer: MultiModalTokenizer,
    ) -> MultiModalObs:
        codes = tokenizer.to_codes(obs_tokens)
        if ObsModality.vector in codes:
            codes[ObsModality.vector] = tokenizer.tokenizers[
                ObsModality.vector.name
            ].decode(codes[ObsModality.vector])
        return codes

    @staticmethod
    def _clone_tensor(x: Optional[Tensor], detach: bool = True) -> Optional[Tensor]:
        if x is None:
            return None
        x = x.detach() if detach else x
        return x.clone()

    @staticmethod
    def _repeat_first_dim(x: Optional[Tensor], repeats: int) -> Optional[Tensor]:
        if x is None:
            return None
        return x.repeat_interleave(repeats, dim=0)

    @staticmethod
    def _clone_multimodal_batch(items: Optional[MultiModalObs], detach: bool = True):
        if items is None:
            return None
        return {k: ActorCriticLS._clone_tensor(v, detach=detach) for k, v in items.items()}

    @staticmethod
    def _repeat_multimodal_batch(items: MultiModalObs, repeats: int) -> MultiModalObs:
        return {k: v.repeat_interleave(repeats, dim=0) for k, v in items.items()}

    @staticmethod
    def _clone_rnn_state(state, detach: bool = True):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(ActorCriticLS._clone_rnn_state(s, detach=detach) for s in state)
        return ActorCriticLS._clone_tensor(state, detach=detach)

    @staticmethod
    def _repeat_rnn_state(state, repeats: int):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(ActorCriticLS._repeat_rnn_state(s, repeats) for s in state)
        return state.repeat_interleave(repeats, dim=0)

    @staticmethod
    def _clone_wm_state_tensor(state, detach: bool = True):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(
                ActorCriticLS._clone_wm_state_tensor(s, detach=detach) for s in state
            )
        if isinstance(state, list):
            return [
                ActorCriticLS._clone_wm_state_tensor(s, detach=detach) for s in state
            ]
        return ActorCriticLS._clone_tensor(state, detach=detach)

    @staticmethod
    def _repeat_wm_state_tensor(state, repeats: int, batch_dim: Optional[int] = None):
        if state is None:
            return None
        if isinstance(state, tuple):
            return tuple(
                ActorCriticLS._repeat_wm_state_tensor(s, repeats, batch_dim=0)
                for s in state
            )
        if isinstance(state, list):
            return [
                ActorCriticLS._repeat_wm_state_tensor(s, repeats, batch_dim=0)
                for s in state
            ]
        if batch_dim is None:
            batch_dim = 1 if state.dim() >= 5 else 0
        return state.repeat_interleave(repeats, dim=batch_dim)

    @staticmethod
    def _clone_wm_recurrent_state(
        recurrent_state: Optional[RecurrentState], detach: bool = True
    ) -> Optional[RecurrentState]:
        if recurrent_state is None:
            return None
        n = recurrent_state.n
        if isinstance(n, torch.Tensor):
            n = ActorCriticLS._clone_tensor(n, detach=detach)
        return RecurrentState(
            ActorCriticLS._clone_wm_state_tensor(recurrent_state.state, detach=detach),
            n,
        )

    @staticmethod
    def _repeat_wm_recurrent_state(
        recurrent_state: Optional[RecurrentState], repeats: int
    ) -> Optional[RecurrentState]:
        if recurrent_state is None:
            return None
        n = recurrent_state.n
        if isinstance(n, torch.Tensor):
            n = n.repeat_interleave(repeats, dim=0)
        return RecurrentState(
            ActorCriticLS._repeat_wm_state_tensor(recurrent_state.state, repeats),
            n,
        )

    def _snapshot_wm_env(self, wm_env: POPWorldModelEnv, detach: bool = True) -> dict:
        return {
            "prior_context": self._clone_tensor(wm_env.prior_context, detach=detach),
            "recurrent_state": self._clone_wm_recurrent_state(
                wm_env.recurrent_state, detach=detach
            ),
            "last_obs_tokens": self._clone_multimodal_batch(
                wm_env.last_obs_tokens, detach=detach
            ),
            "last_obs_tokens_emb": self._clone_tensor(
                getattr(wm_env, "last_obs_tokens_emb", None), detach=detach
            ),
            "last_uncertainty": self._clone_tensor(
                wm_env.last_uncertainty, detach=detach
            ),
        }

    def _restore_wm_env(self, wm_env: POPWorldModelEnv, snapshot: dict) -> None:
        wm_env.prior_context = snapshot["prior_context"]
        wm_env.recurrent_state = snapshot["recurrent_state"]
        wm_env.last_obs_tokens = snapshot["last_obs_tokens"]
        wm_env.last_obs_tokens_emb = snapshot["last_obs_tokens_emb"]
        wm_env.last_uncertainty = snapshot["last_uncertainty"]

    def _repeat_wm_snapshot(self, snapshot: dict, repeats: int) -> dict:
        return {
            "prior_context": self._repeat_first_dim(snapshot["prior_context"], repeats),
            "recurrent_state": self._repeat_wm_recurrent_state(
                snapshot["recurrent_state"], repeats
            ),
            "last_obs_tokens": self._repeat_multimodal_batch(
                snapshot["last_obs_tokens"], repeats
            )
            if snapshot["last_obs_tokens"] is not None
            else None,
            "last_obs_tokens_emb": self._repeat_first_dim(
                snapshot["last_obs_tokens_emb"], repeats
            ),
            "last_uncertainty": self._repeat_first_dim(
                snapshot["last_uncertainty"], repeats
            ),
        }

    def _select_treecf_actions(
        self,
        logits: Tensor,
        branching: int,
        mode: str,
        sample_temperature: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert logits.ndim == 2, f"Expected (nodes, actions), got {logits.shape}"
        num_actions = logits.shape[-1]
        branching = min(branching, num_actions)
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
            raise ValueError(f"Unknown treecf_candidate_mode: {mode}")

        return (
            actions,
            policy_log_probs.gather(1, actions),
            policy_probs.gather(1, actions),
        )

    def _build_tree_counterfactual_actor_loss(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        gamma: float,
        entropy_weight: float,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, dict]:
        if not hasattr(self, "num_actions"):
            raise ValueError("tree_counterfactual currently supports discrete Atari actors only.")

        effective_horizon = horizon - self.context_len + 1
        rollout_horizon = min(int(kwargs.get("treecf_depth", 3)), effective_horizon)
        branching = min(int(kwargs.get("treecf_branching", 3)), int(self.num_actions))
        if rollout_horizon < 1:
            raise ValueError("treecf_depth must be >= 1")
        if branching < 2:
            raise ValueError("treecf_branching must be >= 2")

        candidate_mode = kwargs.get("treecf_candidate_mode", "topk")
        backup_mode = kwargs.get("treecf_backup", "lcb")
        sample_temperature = float(kwargs.get("treecf_sample_temperature", 1.0))
        if sample_temperature <= 0:
            raise ValueError("treecf_sample_temperature must be > 0")
        adv_eps = float(kwargs.get("treecf_adv_eps", 1e-6))
        adv_clip = float(kwargs.get("treecf_adv_clip", 5.0))
        adv_scale = kwargs.get("treecf_adv_scale", "std")
        baseline_mode = kwargs.get("treecf_adv_baseline", "median").lower()
        uncertainty_beta = float(kwargs.get("treecf_uncertainty_beta", 1.0))
        uncertainty_mode = kwargs.get("treecf_uncertainty_mode", "absolute").lower()
        uncertainty_min_weight = float(kwargs.get("treecf_uncertainty_weight_min", 0.05))
        trim_ratio = float(kwargs.get("treecf_trim_ratio", 0.25))

        wm_env, obs_tokens = self._imagination_set_initial_state(
            batch, tokenizer, world_model
        )
        batch_size = batch["mask_padding"].shape[0]
        anchor_wm_snapshot = self._snapshot_wm_env(wm_env, detach=True)
        anchor_actor_state = self._clone_rnn_state(self.actor_state, detach=True)
        anchor_critic_state = self._clone_rnn_state(self.critic_state, detach=True)

        self.actor_state = anchor_actor_state
        self.critic_state = anchor_critic_state
        self._restore_wm_env(wm_env, anchor_wm_snapshot)

        obs_codes = self._to_codes(obs_tokens, world_model, tokenizer)
        actor_outs, _ = self(inputs=obs_codes)
        actions_dist = actor_outs.get_actions_distributions()
        logits = actions_dist.logits[:, 0]
        actions, log_probs, policy_probs, _, _ = select_counterfactual_actions(
            logits=logits,
            branching=branching,
            mode=candidate_mode,
            sample_temperature=sample_temperature,
        )
        anchor_entropy = actions_dist.entropy().reshape(batch_size, -1)[:, 0]

        post_anchor_actor_state = self._clone_rnn_state(self.actor_state, detach=True)
        post_anchor_critic_state = self._clone_rnn_state(self.critic_state, detach=True)
        self.actor_state = self._repeat_rnn_state(post_anchor_actor_state, branching)
        self.critic_state = self._repeat_rnn_state(post_anchor_critic_state, branching)
        self._restore_wm_env(
            wm_env, self._repeat_wm_snapshot(anchor_wm_snapshot, branching)
        )

        num_branches_total = batch_size * branching
        current_actions = actions.reshape(-1, 1)
        alive = torch.ones(
            num_branches_total, dtype=torch.bool, device=current_actions.device
        )
        returns = torch.zeros(
            num_branches_total, dtype=log_probs.dtype, device=log_probs.device
        )
        discounts = torch.ones_like(returns)
        uncertainty_sum = torch.zeros_like(returns)
        uncertainty_count = torch.zeros_like(returns)
        rollout_rewards = []
        rollout_ends = []

        with torch.no_grad():
            for step in range(rollout_horizon):
                self.process_action(current_actions.squeeze(1))
                next_obs_tokens, reward, done, step_info = wm_env.step(
                    current_actions,
                    should_predict_next_obs=True,
                    return_tokens=True,
                )
                reward = reward.reshape(-1)
                done = done.reshape(-1).bool()
                uncertainty = None
                if isinstance(step_info, dict):
                    uncertainty = step_info.get("uncertainty")
                if uncertainty is None:
                    uncertainty = torch.zeros_like(reward.reshape(-1, 1))
                uncertainty = uncertainty.reshape(-1)

                active = alive.float()
                returns = returns + discounts * reward * active
                uncertainty_sum = uncertainty_sum + uncertainty * active
                uncertainty_count = uncertainty_count + active
                rollout_rewards.append(reward.reshape(batch_size, branching))
                rollout_ends.append(done.reshape(batch_size, branching))

                alive = alive & done.logical_not()
                next_codes = self._to_codes(next_obs_tokens, world_model, tokenizer)
                next_actor_outs, next_critic_outs = self(inputs=next_codes)

                if step == rollout_horizon - 1:
                    leaf_values = self._get_values_means(
                        next_critic_outs.get_value_info()
                    ).reshape(-1)
                    returns = returns + discounts * gamma * alive.float() * leaf_values
                else:
                    discounts = discounts * gamma
                    next_actions_dist = next_actor_outs.get_actions_distributions()
                    current_actions = next_actions_dist.sample().reshape(-1, 1)

        branch_returns = returns.reshape(batch_size, branching)
        if baseline_mode == "policy":
            baseline_probs = policy_probs.detach()
            baseline_probs = baseline_probs / baseline_probs.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            baseline = (baseline_probs * branch_returns).sum(dim=1, keepdim=True)
        elif baseline_mode == "backup":
            baseline = robust_tree_backup(
                branch_returns,
                mode=backup_mode,
                lcb_alpha=float(kwargs.get("treecf_lcb_alpha", 0.5)),
                cvar_fraction=float(kwargs.get("treecf_cvar_fraction", 0.5)),
                trim_ratio=trim_ratio,
            ).reshape(-1, 1)
        elif baseline_mode in {"mean", "median", "trimmed_mean", "trimmed", "trim_mean"}:
            baseline = robust_center(
                branch_returns,
                mode=baseline_mode,
                trim_ratio=trim_ratio,
            )
        else:
            raise ValueError(f"Unknown treecf_adv_baseline: {baseline_mode}")

        raw_advantages = branch_returns - baseline
        advantages = normalize_counterfactual_advantages(
            raw_advantages,
            eps=adv_eps,
            clip=adv_clip,
            scale=adv_scale,
        )
        branch_uncertainty = (
            uncertainty_sum / uncertainty_count.clamp_min(1.0)
        ).reshape(batch_size, branching)
        uncertainty_for_weight = scale_uncertainty_for_weights(
            branch_uncertainty,
            mode=uncertainty_mode,
            eps=adv_eps,
        )
        weights = compute_uncertainty_weights(
            uncertainty_for_weight,
            beta=uncertainty_beta,
            min_weight=uncertainty_min_weight,
        )
        alive_2d = alive.reshape(batch_size, branching)
        weights = weights * (uncertainty_count.reshape(batch_size, branching) > 0).float()

        loss_actions = -(weights.detach() * advantages.detach() * log_probs).mean()
        loss_entropy = -entropy_weight * anchor_entropy.mean()
        rollout_depths = torch.full(
            (batch_size * branching,),
            rollout_horizon,
            device=branch_returns.device,
            dtype=branch_returns.dtype,
        )
        self.clear()

        tree_info = {
            "imagined_treecf_edge_returns": branch_returns.detach().reshape(-1),
            "imagined_treecf_branch_returns": branch_returns.detach().reshape(-1),
            "imagined_treecf_raw_advantages": raw_advantages.detach().reshape(-1),
            "imagined_treecf_advantages": advantages.detach().reshape(-1),
            "imagined_treecf_weights": weights.detach().reshape(-1),
            "imagined_treecf_uncertainty": branch_uncertainty.detach().reshape(-1),
            "imagined_treecf_uncertainty_risk": uncertainty_for_weight.detach().reshape(-1),
            "imagined_treecf_log_probs": log_probs.detach().reshape(-1),
            "imagined_treecf_policy_probs": policy_probs.detach().reshape(-1),
            "imagined_treecf_depths": rollout_depths.detach(),
            "imagined_treecf_alive": alive_2d.float().reshape(-1),
            "imagined_treecf_entropy": anchor_entropy.detach(),
            "imagined_treecf_root_values": baseline.detach().reshape(-1),
            "imagined_treecf_rollout_rewards": torch.stack(
                rollout_rewards, dim=1
            ).detach(),
            "imagined_treecf_rollout_ends": torch.stack(rollout_ends, dim=1).float(),
        }
        return loss_actions, loss_entropy, tree_info

    def imagine(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ) -> ImagineOutput:
        mask_padding, actions = batch["mask_padding"], batch["actions"]

        assert mask_padding[:, -1].all()
        device = self.device

        all_actions = []
        all_actions_dists = []
        all_values_info = []
        all_q_info = []
        all_rewards = []
        all_ends = []
        all_observations = []

        wm_env, obs_tokens = self._imagination_set_initial_state(
            batch, tokenizer, world_model
        )

        obs_codes = self._to_codes(obs_tokens, world_model, tokenizer)

        effective_horizon = horizon - self.context_len + 1
        for k in tqdm(
            range(effective_horizon),
            disable=not show_pbar,
            desc="Imagination",
            file=sys.stdout,
        ):

            all_observations.append(obs_codes)

            actor_outs, critic_outs = self(inputs=obs_codes)
            action = actor_outs.get_actions_distributions().sample()
            assert self.include_action_inputs
            q = self.process_action(action.squeeze(1))
            should_predict_next_obs = k < effective_horizon - 1
            obs_tokens, reward, done, _ = wm_env.step(
                action,
                should_predict_next_obs=should_predict_next_obs,
                return_tokens=True,
            )
            obs_codes = (
                self._to_codes(obs_tokens, world_model, tokenizer)
                if should_predict_next_obs
                else None
            )

            all_actions.append(action)
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            all_rewards.append(reward.reshape(-1, 1))
            all_ends.append(done.reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations={
                k: torch.stack([o[k] for o in all_observations], dim=1)
                for k in self._ordered_modalities
            },  # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )

    def play_env(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ) -> ImagineOutput:
        # initial_observations = batch['observations']
        mask_padding = batch["mask_padding"]
        # assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = next(iter(batch.values())).device
        from envs import make_dm_control

        env = make_dm_control("walker-run")

        all_actions = []
        all_actions_dists = []
        all_values_info = []
        all_q_info = []
        all_rewards = []
        all_ends = []
        all_observations = []

        self.reset(n=1)
        obs = torch.tensor([env.reset()[0]]).float().to(device)

        obs_codes = self._to_codes(tokenizer.encode(obs).tokens, world_model, tokenizer)

        # effective_horizon = horizon - self.context_len + 1
        effective_horizon = 500
        for k in tqdm(
            range(effective_horizon),
            disable=not show_pbar,
            desc="Imagination",
            file=sys.stdout,
        ):

            all_observations.append(obs_codes)

            actor_outs, critic_outs = self(inputs=obs_codes)
            action = actor_outs.get_actions_distributions().sample()
            if self.include_action_inputs:
                q = self.process_action(action.squeeze(1))
            obs, reward, terminated, truncated, _ = env.step(
                action[0].detach().cpu().numpy()
            )
            obs = torch.Tensor([obs]).float().to(action.device)
            if terminated:
                obs = torch.Tensor([env.reset()[0]]).float().to(device)
            reward = torch.Tensor([reward]).float().to(device)
            terminated = torch.Tensor([terminated]).bool().to(device)
            obs_codes = self._to_codes(
                tokenizer.encode(obs).tokens, world_model, tokenizer
            )

            all_actions.append(action)
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            # all_q_info.append(self._get_q_info(q))
            all_rewards.append(reward.reshape(-1, 1))
            all_ends.append(terminated.reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1),  # (B, T, C, H, W)
            actions=torch.cat(all_actions, dim=1),  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),  # (B, T)
        )

    def get_action_distribution_and_values(
        self,
        batch: Batch,
        tokenizer: MultiModalTokenizer,
        world_model: POPWorldModel,
        horizon: int,
        show_pbar: bool = False,
    ):
        device=batch["mask_padding"].device
        wm_env = POPWorldModelEnv(
            tokenizer,
            world_model,
            device=device,
            real_reward_weight=self.real_reward_weight,
            intrinsic_reward_weight=self.intrinsic_reward_weight,
        )

        mask_padding, actions, rewards, ends = (
            batch["mask_padding"],
            batch["actions"],
            batch["rewards"],
            batch["ends"],
        )
        batch_size, seq_len = batch["mask_padding"].shape[:2]
        # set the initial state of the actor-critic:
        with torch.no_grad():
            obs_tokens = world_model.get_obs_tokens(
                batch["observations"], tokenizer=tokenizer
            )
        obs_quantized = self._to_codes(obs_tokens, world_model, tokenizer)
        effective_horizon = horizon - self.context_len + 1
        # Ignore last obs as it could be the last obs (obtained after termination signal)
        burnin_observations = (
            {k: v[:, : -effective_horizon - 1] for k, v in obs_quantized.items()}
            if seq_len > 1
            else None
        )
        ac_actions_embs = self.embed_action(
            batch["actions"][:, : -effective_horizon - 1]
        )
        self.reset(
            n=batch_size,
            burnin_observations=burnin_observations,
            mask_padding=mask_padding[:, : -effective_horizon - 1],
            ac_actions_embs=ac_actions_embs,
        )
        ctx_len = self.context_len
        action_seq_len = world_model.tokens_per_action
        ctx = world_model.get_tokens_emb(
            {k: obs_tokens[k][:, -ctx_len - effective_horizon - 1 : - effective_horizon - 1] for k in obs_tokens.keys()},
            batch["actions"][:, -ctx_len - effective_horizon - 1 : - effective_horizon - 1],
            tokenizer=tokenizer,
        ).flatten(1, 2)[:, :-action_seq_len]

        wm_env.reset_from_initial_observations(
            ctx,
            return_tokens=True,
        )
        all_actions_dists = []
        all_values_info = []
        all_observations = []
        all_rewards=[]
        for i in tqdm(
            range(seq_len - effective_horizon - 1, seq_len - 1),
            disable=not show_pbar,
            desc="RealExperience",
            file=sys.stdout,
        ):
            total_rewards=wm_env._compute_total_reward_with_real_env(batch["actions"][:,i],batch["rewards"][:,i])
            obs_codes = {k: v[:, i] for k, v in obs_quantized.items()}
            all_observations.append(obs_codes)
            actor_outs, critic_outs = self(inputs=obs_codes)
            assert self.include_action_inputs
            all_actions_dists.append(actor_outs.get_actions_distributions())
            all_values_info.append(critic_outs.get_value_info())
            all_rewards.append(total_rewards.reshape(-1, 1))
        self.clear()
        return ImagineOutput(
            observations={
                k: torch.stack([o[k] for o in all_observations], dim=1)
                for k in self._ordered_modalities
            },  # (B, T, C, H, W)
            actions=actions[:, seq_len - effective_horizon - 1 : seq_len - 1],  # (B, T)
            actions_distributions=self._concat_distributions(
                all_actions_dists
            ),  # (B, T, #actions)
            values_info=self._concat_values(all_values_info),  # (B, T)
            q_values_info=None,  # self._concat_values(all_q_info),
            rewards=torch.cat(all_rewards, dim=1).to(device),  # (B, T)
            ends=ends[:, seq_len - effective_horizon - 1 : seq_len - 1],  # (B, T)
        )


class ActorCriticLS2(ActorCriticLS, ABC):

    def __init__(
        self,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.num_value_categories = num_value_bins + 1
        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

    def _build_critic_head(self):
        return HLGaussCategoricalRegressionHead(
            self.lstm_dim,
            self.num_value_categories,
            sym_log_normalize=True,
            device=self.device,
        )

    def _compute_critic_output(self, critic_latent) -> CategoricalCriticOutput:
        means_values = self.critic_v_head(critic_latent)

        return CategoricalCriticOutput(value_logits=critic_latent, values=means_values)

    def _compute_critic_loss(
        self, values_info: CategoricalValuesInfo, targets
    ) -> Tensor:
        loss = self.critic_v_head.compute_loss(
            values_info.values_logits[:, :-1], targets
        )
        return loss

    def _concat_values(self, values_info_list: list[CategoricalValuesInfo]):
        return CategoricalValuesInfo(
            values_logits=torch.stack(
                [vi.values_logits for vi in values_info_list], dim=1
            ),
            values=torch.stack([vi.values for vi in values_info_list], dim=1),
        )


class DiscreteActorCriticLS(ActorCriticLS2):

    def __init__(
        self,
        act_vocab_size,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        unknown_action=False,
        **kwargs,
    ) -> None:
        self.num_actions = act_vocab_size 

        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

        act_dim=self.num_actions if unknown_action else self.num_actions+1
        self.actor_actions_embeddings = nn.Embedding(
            act_dim, self.embed_dim, device=self.device
        )
        self.critic_actions_embeddings = (
            nn.Embedding(act_dim, self.embed_dim, device=self.device)
            if separate_networks
            else None
        )
        self.dr_q_head = (
            nn.Linear(self.lstm_dim, self.num_actions, device=self.device)
            if bool(kwargs.get("dr_q_head", False))
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Linear(self.lstm_dim, self.num_actions, device=self.device)

    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        actor_action_emb = self.actor_actions_embeddings(action.long())
        critic_action_emb = (
            self.critic_actions_embeddings(action.long())
            if self.separate_networks
            else None
        )

        return actor_action_emb, critic_action_emb

    def _compute_actor_output(self, actor_latent) -> DiscreteActorOutput:
        logits_actions = self.actor_linear(actor_latent)
        if logits_actions.dim() == 2:
            logits_actions = logits_actions.unsqueeze(1)

        return DiscreteActorOutput(logits_actions=logits_actions)

    def _concat_distributions(self, all_actions_dists) -> CategoricalDistribution:
        return CategoricalDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1)
        )


class MultiDiscreteActorCriticLS(ActorCriticLS2):

    def __init__(
        self,
        actions_nvec,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.actions_nvec = actions_nvec

        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

        self.actor_actions_embeddings = MultiDiscreteEmbedding(
            actions_nvec, self.embed_dim, device=self.device
        )
        self.critic_actions_embeddings = (
            MultiDiscreteEmbedding(actions_nvec, self.embed_dim, device=self.device)
            if separate_networks
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Linear(self.lstm_dim, sum(self.actions_nvec), device=self.device)

    def embed_action(self, action) -> tuple[Tensor, Tensor]:
        actor_action_emb = self.actor_actions_embeddings(action.long()).mean(dim=-2)
        critic_action_emb = (
            self.critic_actions_embeddings(action.long()).mean(dim=-2)
            if self.separate_networks
            else None
        )

        return actor_action_emb, critic_action_emb

    def _compute_actor_output(self, actor_latent) -> MultiDiscreteActorOutput:
        logits_actions = self.actor_linear(actor_latent)
        if logits_actions.dim() == 2:
            logits_actions = logits_actions.unsqueeze(1)

        return MultiDiscreteActorOutput(
            logits_actions=logits_actions, nvec=self.actions_nvec
        )

    def _concat_distributions(self, all_actions_dists) -> MultiCategoricalDistribution:
        return MultiCategoricalDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1),
            nvec=all_actions_dists[0].nvec,
        )


class ContinuousActorCriticLS(ActorCriticLS2):
    def __init__(
        self,
        action_dim: int,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 100,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        super().__init__(
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )
        self.actor_action_projection = nn.Linear(
            action_dim, self.embed_dim, device=device
        )
        self.critic_action_projection = (
            nn.Linear(action_dim, self.embed_dim, device=device)
            if separate_networks
            else None
        )

    def _build_actor_head(self) -> nn.Module:
        model = nn.Sequential(
            nn.Linear(
                self.lstm_dim, self.action_dim * 2, device=self.device
            )  # mean, std
        )
        model[-1].bias.data = torch.cat(
            [
                torch.zeros(self.action_dim, device=self.device),
                torch.ones(self.action_dim, device=self.device) * 2,
            ]
        )
        return model

    def embed_action(self, action: Tensor) -> tuple[Tensor, Tensor]:
        critic_emb = (
            self.critic_action_projection(action) if self.separate_networks else None
        )
        return self.actor_action_projection(action), critic_emb

    def _compute_actor_output(self, actor_latent) -> ContinuousActorOutput:
        actor_outs = self.actor_linear(actor_latent)
        if actor_latent.dim() == 2:
            actor_outs = actor_outs.unsqueeze(1)
        actions_means, actions_log_stds = torch.split(
            actor_outs, self.action_dim, dim=-1
        )
        # actions_means = torch.clamp(actions_means, -1, 1)
        min_std, max_std = (0.05, 1)
        actions_stds = min_std + torch.sigmoid(actions_log_stds) * (max_std - min_std)
        # actions_stds = torch.ones_like(actions_log_stds)*0.1

        return ContinuousActorOutput(
            actions_means=actions_means, actions_stds=actions_stds
        )

    def _concat_distributions(self, all_actions_dists) -> Normal:
        return SquashedDiagNormalDistribution(
            loc=torch.cat([d.loc for d in all_actions_dists], dim=1),
            scale=torch.cat([d.scale for d in all_actions_dists], dim=1),
        )


class DContinuousActorCriticLS(ContinuousActorCriticLS):

    def __init__(
        self,
        action_dim: int,
        obs_encoders,
        obs_mlp_layer_sizes: list[int],
        num_value_bins: int = 128,
        separate_networks: bool = True,
        real_reward_weight: float = 1.0,
        intrinsic_reward_weight: float = 1.0,
        name: str = "actor_critic",
        include_action_inputs: bool = True,
        context_len: int = 2,
        rnn_type: str = "lstm",
        device=None,
        n_action_quant_levels: int = 51,
        **kwargs,
    ):
        self.n_action_quant_levels = n_action_quant_levels
        super().__init__(
            action_dim=action_dim,
            obs_encoders=obs_encoders,
            obs_mlp_layer_sizes=obs_mlp_layer_sizes,
            num_value_bins=num_value_bins,
            separate_networks=separate_networks,
            real_reward_weight=real_reward_weight,
            intrinsic_reward_weight=intrinsic_reward_weight,
            name=name,
            include_action_inputs=include_action_inputs,
            context_len=context_len,
            rnn_type=rnn_type,
            device=device,
        )

    def _build_actor_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.lstm_dim,
                self.action_dim * self.n_action_quant_levels,
                device=self.device,
            )
        )

    def _compute_actor_output(self, actor_latent) -> QuantizedContinuousActorOutput:
        actions_logits = rearrange(
            self.actor_linear(actor_latent),
            "... (m n) -> ... m n",
            m=self.action_dim,
            n=self.n_action_quant_levels,
        )
        if actor_latent.dim() == 2:
            actions_logits = actions_logits.unsqueeze(1)

        return QuantizedContinuousActorOutput(logits_actions=actions_logits)

    def _concat_distributions(
        self, all_actions_dists
    ) -> QuantizedContinuousDistribution:
        return QuantizedContinuousDistribution(
            logits=torch.cat([d.logits for d in all_actions_dists], dim=1)
        )
