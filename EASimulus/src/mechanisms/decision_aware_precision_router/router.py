from __future__ import annotations

import math
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F


class DecisionAwarePrecisionRouter(nn.Module):
    """Fixed-slot precision router for spatial image tokens.

    The router keeps the token count constant and chooses, per spatial slot,
    whether the world model/controller should receive the high-precision token
    embedding or a local summary embedding. This avoids changing RetNet block
    structure while still testing decision-aware precision allocation.
    """

    def __init__(
        self,
        embed_dim: int,
        tokens_per_obs: int,
        enabled: bool = False,
        target_keep_ratio: float = 0.5,
        min_keep_ratio: float = 0.05,
        temperature: float = 1.0,
        hard: bool = True,
        router: str = "gumbel",
        summary: str = "local_mean",
        summary_kernel: int = 2,
        budget_loss_weight: float = 0.01,
        entropy_loss_weight: float = 0.0,
        feature_dim: int = 4,
        hidden_dim: int = 128,
        use_positional_features: bool = True,
        detach_summary: bool = True,
        warmup_epochs: int = 0,
        eps: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.embed_dim = int(embed_dim)
        self.tokens_per_obs = int(tokens_per_obs)
        self.target_keep_ratio = float(target_keep_ratio)
        self.min_keep_ratio = float(min_keep_ratio)
        self.temperature = float(temperature)
        self.hard = bool(hard)
        self.router = router
        self.summary = summary
        self.summary_kernel = int(summary_kernel)
        self.budget_loss_weight = float(budget_loss_weight)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_positional_features = bool(use_positional_features)
        self.detach_summary = bool(detach_summary)
        self.warmup_epochs = int(warmup_epochs)
        self.eps = float(eps)

        grid = int(math.sqrt(self.tokens_per_obs))
        if grid * grid != self.tokens_per_obs:
            raise ValueError(
                f"DecisionAwarePrecisionRouter expects a square image-token grid; "
                f"got tokens_per_obs={self.tokens_per_obs}."
            )
        self.grid_size = grid

        if self.summary not in {"local_mean", "global_mean", "learned"}:
            raise ValueError(f"Unsupported summary mode: {self.summary}")
        if self.router not in {"gumbel", "topk", "soft"}:
            raise ValueError(f"Unsupported router mode: {self.router}")

        if self.summary == "learned":
            self.summary_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
            nn.init.normal_(self.summary_token, std=0.02)
        else:
            self.summary_token = None

        pos_dim = 2 if self.use_positional_features else 0
        self.score_net = nn.Sequential(
            nn.LayerNorm(self.embed_dim + self.feature_dim + pos_dim),
            nn.Linear(self.embed_dim + self.feature_dim + pos_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.current_epoch = 0
        self._last_keep_probs: Optional[Tensor] = None
        self._last_keep_mask: Optional[Tensor] = None
        self._last_budget_loss: Optional[Tensor] = None
        self._last_entropy_loss: Optional[Tensor] = None

    @property
    def active(self) -> bool:
        return self.enabled and self.current_epoch >= self.warmup_epochs

    def set_epoch(self, epoch: Optional[int]) -> None:
        if epoch is not None:
            self.current_epoch = int(epoch)

    def forward(
        self,
        image_embeddings: Tensor,
        route_features: Optional[Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Tensor:
        self.set_epoch(epoch)
        if not self.active:
            self._clear_step_state(image_embeddings)
            return image_embeddings

        x = image_embeddings
        leading_shape = x.shape[:-2]
        k, e = x.shape[-2:]
        if k != self.tokens_per_obs or e != self.embed_dim:
            raise ValueError(
                f"Expected image embeddings (..., {self.tokens_per_obs}, {self.embed_dim}), "
                f"got {tuple(x.shape)}."
            )

        x_flat = x.reshape(-1, k, e)
        features = self._prepare_features(route_features, leading_shape, x.device, x.dtype)
        features = features.reshape(-1, k, self.feature_dim)

        if self.use_positional_features:
            pos = self._position_features(x.device, x.dtype)
            pos = pos.unsqueeze(0).expand(x_flat.shape[0], -1, -1)
            router_input = torch.cat([x_flat.detach(), features.detach(), pos], dim=-1)
        else:
            router_input = torch.cat([x_flat.detach(), features.detach()], dim=-1)

        logits = self.score_net(router_input).squeeze(-1)
        keep_probs = torch.sigmoid(logits)
        keep_mask = self._sample_mask(logits, keep_probs)
        summary = self._summary_embeddings(x_flat)
        routed = keep_mask.unsqueeze(-1) * x_flat + (1.0 - keep_mask).unsqueeze(-1) * summary

        self._last_keep_probs = keep_probs
        self._last_keep_mask = keep_mask
        mean_keep = keep_probs.mean()
        floor_penalty = F.relu(self.min_keep_ratio - mean_keep).pow(2)
        target_penalty = (mean_keep - self.target_keep_ratio).pow(2)
        self._last_budget_loss = self.budget_loss_weight * (target_penalty + floor_penalty)
        entropy = -(keep_probs * torch.log(keep_probs + self.eps) + (1.0 - keep_probs) * torch.log(1.0 - keep_probs + self.eps))
        self._last_entropy_loss = -self.entropy_loss_weight * entropy.mean()

        return routed.reshape(*leading_shape, k, e)

    def regularization_losses(self) -> dict[str, Tensor]:
        losses = {}
        if self._last_budget_loss is not None and self.budget_loss_weight != 0.0:
            losses["dapr_budget_loss"] = self._last_budget_loss
        if self._last_entropy_loss is not None and self.entropy_loss_weight != 0.0:
            losses["dapr_entropy_loss"] = self._last_entropy_loss
        return losses

    def info(self) -> dict[str, Tensor]:
        if self._last_keep_probs is None or self._last_keep_mask is None:
            return {}
        with torch.no_grad():
            return {
                "dapr_keep_prob": self._last_keep_probs.mean().detach(),
                "dapr_keep_ratio": self._last_keep_mask.mean().detach(),
                "dapr_keep_prob_std": self._last_keep_probs.std().detach(),
                "dapr_budget_target": torch.tensor(
                    self.target_keep_ratio,
                    device=self._last_keep_probs.device,
                    dtype=self._last_keep_probs.dtype,
                ),
            }

    def _clear_step_state(self, ref: Tensor) -> None:
        zero = ref.new_tensor(0.0)
        self._last_keep_probs = None
        self._last_keep_mask = None
        self._last_budget_loss = zero
        self._last_entropy_loss = zero

    def _prepare_features(
        self,
        route_features: Optional[Tensor],
        leading_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        target = (*leading_shape, self.tokens_per_obs, self.feature_dim)
        if route_features is None:
            return torch.zeros(target, device=device, dtype=dtype)

        f = route_features.to(device=device, dtype=dtype)
        if f.shape[-1] > self.feature_dim:
            f = f[..., : self.feature_dim]
        elif f.shape[-1] < self.feature_dim:
            pad = torch.zeros(*f.shape[:-1], self.feature_dim - f.shape[-1], device=device, dtype=dtype)
            f = torch.cat([f, pad], dim=-1)

        if f.shape[-2] != self.tokens_per_obs:
            raise ValueError(
                f"Route features must have {self.tokens_per_obs} spatial entries, got {f.shape[-2]}."
            )
        while f.dim() < len(target):
            f = f.unsqueeze(-3)
        return f.expand(target)

    def _position_features(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        coords = torch.linspace(-1.0, 1.0, self.grid_size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([yy, xx], dim=-1).reshape(self.tokens_per_obs, 2)

    def _sample_mask(self, logits: Tensor, keep_probs: Tensor) -> Tensor:
        if self.router == "soft":
            return keep_probs

        if self.router == "topk":
            keep_count = max(1, int(round(self.target_keep_ratio * self.tokens_per_obs)))
            keep_count = min(self.tokens_per_obs, keep_count)
            topk = torch.topk(logits, keep_count, dim=-1).indices
            hard = torch.zeros_like(logits)
            hard.scatter_(dim=-1, index=topk, value=1.0)
            return hard.detach() - keep_probs.detach() + keep_probs if self.training else hard

        if self.training:
            noise = torch.rand_like(logits).clamp_(self.eps, 1.0 - self.eps)
            logistic = torch.log(noise) - torch.log1p(-noise)
            soft = torch.sigmoid((logits + logistic) / max(self.temperature, self.eps))
        else:
            soft = keep_probs

        if not self.hard:
            return soft
        hard = (soft >= 0.5).to(soft.dtype)
        return hard.detach() - soft.detach() + soft if self.training else hard

    def _summary_embeddings(self, x: Tensor) -> Tensor:
        if self.summary == "learned":
            assert self.summary_token is not None
            return self.summary_token.expand(x.shape[0], self.tokens_per_obs, self.embed_dim)

        if self.summary == "global_mean":
            summary = x.mean(dim=1, keepdim=True).expand_as(x)
            return summary.detach() if self.detach_summary else summary

        h = w = self.grid_size
        grid = rearrange(x, "b (h w) e -> b e h w", h=h, w=w)
        pooled = F.avg_pool2d(
            grid,
            kernel_size=self.summary_kernel,
            stride=self.summary_kernel,
            ceil_mode=True,
        )
        summary_grid = F.interpolate(pooled, size=(h, w), mode="nearest")
        summary = rearrange(summary_grid, "b e h w -> b (h w) e")
        return summary.detach() if self.detach_summary else summary
