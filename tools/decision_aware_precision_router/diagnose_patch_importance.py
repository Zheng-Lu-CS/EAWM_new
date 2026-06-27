#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf


def _prepare_imports(repo_root: Path) -> Path:
    src_dir = repo_root / "EASimulus" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def _load_cfg(run_dir: Path):
    candidates = [
        run_dir / "config" / "base.yaml",
        run_dir / ".hydra" / "config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return OmegaConf.load(path)
    raise FileNotFoundError(f"No Hydra config found under {run_dir}")


def _make_env(cfg):
    from envs import SingleProcessEnv

    env_partial = instantiate(cfg.env.train)
    env_fn = partial(env_partial, tokenizer_config=cfg.tokenizer)
    return SingleProcessEnv(env_fn)


def _load_agent(cfg, run_dir: Path, checkpoint_name: str, device):
    from main import build_agent

    env = _make_env(cfg)
    agent = build_agent(env=env, cfg=cfg, device=device)
    ckpt_path = run_dir / "checkpoints" / f"{checkpoint_name}.pt"
    if not ckpt_path.exists():
        env.close()
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    agent.load(ckpt_path, device=device)
    agent.eval()
    return agent, env


def _load_dataset(cfg, run_dir: Path):
    dataset_dir = run_dir / "checkpoints" / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset checkpoint not found: {dataset_dir}")
    ds = instantiate(cfg.datasets.train)
    ds.load_disk_checkpoint(dataset_dir)
    return ds


def _to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, dict):
            batch[k] = _to_device(v, device)
        else:
            batch[k] = v.to(device)
    return batch


def _local_summary(x, grid_size: int, kernel: int):
    grid = rearrange(x, "b t (h w) e -> (b t) e h w", h=grid_size, w=grid_size)
    pooled = F.avg_pool2d(grid, kernel_size=kernel, stride=kernel, ceil_mode=True)
    summary = F.interpolate(pooled, size=(grid_size, grid_size), mode="nearest")
    return rearrange(summary, "(b t) e h w -> b t (h w) e", b=x.shape[0], t=x.shape[1])


def _image_logits_and_losses(agent, batch, image_embeddings=None):
    from utils import ObsModality

    wm = agent.world_model
    tokenizer = agent.tokenizer
    obs_tokens = wm.get_obs_tokens(batch["observations"], tokenizer)

    if image_embeddings is None:
        tokens_emb = wm.get_tokens_emb(obs_tokens, batch["actions"], tokenizer=tokenizer)
    else:
        actions_emb = wm.embed_actions(batch["actions"])
        tokens_emb = torch.cat([image_embeddings, actions_emb], dim=2)

    outputs = wm(tokens_emb)
    pred_mask = batch["mask_padding"].clone()
    pred_mask[:, : wm.context_length] = 0
    outputs = rearrange(outputs, "b (t k) e -> b t k e", k=wm.tokens_per_block)
    image_idx = wm.ordered_modalities.index(ObsModality.image)
    start = sum(wm.segment_lengths[:image_idx])
    stop = start + wm.segment_lengths[image_idx]
    image_outputs = outputs[:, :, start:stop]
    logits = wm.head_observations[ObsModality.image.name](image_outputs[pred_mask])
    labels = obs_tokens[ObsModality.image][pred_mask]
    losses = F.cross_entropy(
        logits.flatten(0, 1),
        labels.flatten(),
        reduction="none",
    ).reshape(-1, wm.tokens_per_obs_dict[ObsModality.image])
    return obs_tokens, pred_mask, image_outputs, losses


def _score_to_mask(scores, keep_ratio: float, context_length: int):
    b, t, k = scores.shape
    keep = max(1, min(k, int(round(k * keep_ratio))))
    flat_scores = scores.reshape(b * t, k)
    topk = torch.topk(flat_scores, keep, dim=-1).indices
    mask = torch.zeros_like(flat_scores)
    mask.scatter_(1, topk, 1.0)
    mask = mask.reshape(b, t, k)
    mask[:, :context_length] = 1.0
    return mask


def _random_mask(shape, keep_ratio: float, context_length: int, device):
    b, t, k = shape
    scores = torch.rand(b, t, k, device=device)
    return _score_to_mask(scores, keep_ratio, context_length)


def _expand_pred_scores(scores_pred, pred_mask, full_shape):
    scores = torch.zeros(full_shape, device=scores_pred.device)
    scores[pred_mask] = scores_pred
    return scores


def _reward_gradient_scores(agent, batch, base_embeddings):
    wm = agent.world_model
    emb = base_embeddings.detach().clone().requires_grad_(True)
    actions_emb = wm.embed_actions(batch["actions"]).detach()
    outputs = wm(torch.cat([emb, actions_emb], dim=2))
    rewards, _ = wm.sample_rewards_ends(outputs)
    objective = rewards.abs().mean()
    grad = torch.autograd.grad(objective, emb, retain_graph=False, create_graph=False)[0]
    return grad.detach().norm(dim=-1)


def _value_gradient_scores(agent, base_embeddings):
    from utils import ObsModality

    ac = agent.actor_critic
    wm = agent.world_model
    b, t, k, e = base_embeddings.shape
    h = int(k ** 0.5)
    obs_codes = rearrange(
        base_embeddings.detach().clone().requires_grad_(True),
        "b t (h w) e -> b t e h w",
        h=h,
    )
    ac.clear()
    ac.reset(n=b)
    values = []
    for step in range(t):
        _, critic_out = ac(inputs={ObsModality.image: obs_codes[:, step]})
        values.append(critic_out.get_value_info().value_means)
    value_objective = torch.stack(values, dim=1).abs().mean()
    grad = torch.autograd.grad(
        value_objective, obs_codes, retain_graph=False, create_graph=False
    )[0]
    ac.clear()
    return rearrange(grad.detach().norm(dim=2), "b t h w -> b t (h w)")


def run_diagnostic(args):
    repo_root = Path(args.repo_root).resolve()
    _prepare_imports(repo_root)

    from dataset import get_dataloader
    from utils import ObsModality

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_cfg(run_dir)
    cfg.common.device = args.device
    if OmegaConf.select(cfg, "mechanism.decision_aware_precision_router.enabled") is not None:
        cfg.mechanism.decision_aware_precision_router.enabled = False

    device = torch.device(args.device)
    agent, env = _load_agent(cfg, run_dir, args.checkpoint, device)
    try:
        if ObsModality.image not in agent.tokenizer.modalities:
            raise RuntimeError("Patch-importance diagnostic requires an image tokenizer.")

        dataset = _load_dataset(cfg, run_dir)
        dataloader = get_dataloader(
            dataset,
            cfg.world_model.context_length,
            cfg.common.sequence_length,
            args.batch_size,
            shuffle=True,
            padding_strategy="right",
            obs_modalities=agent.tokenizer.modalities,
            num_workers=0,
            drop_last=True,
        )
        data_iter = iter(dataloader)
        rows = []
        h = int(agent.world_model.tokens_per_obs_dict[ObsModality.image] ** 0.5)

        for batch_idx in range(args.num_batches):
            batch = _to_device(next(data_iter), device)
            with torch.no_grad():
                obs_tokens, pred_mask, image_outputs, base_losses = _image_logits_and_losses(agent, batch)
                base_ce = base_losses.mean().item()

            image_tokens = obs_tokens[ObsModality.image]
            image_tokenizer = agent.tokenizer.tokenizers[ObsModality.image.name]
            with torch.no_grad():
                base_embeddings = image_tokenizer.to_codes(
                    image_tokens.flatten(0, 1)
                )
                base_embeddings = rearrange(
                    base_embeddings,
                    "(b t) e h w -> b t (h w) e",
                    b=image_tokens.shape[0],
                    t=image_tokens.shape[1],
                )
                summary = _local_summary(base_embeddings, h, args.summary_kernel)

            oracle_scores = _expand_pred_scores(
                base_losses, pred_mask, image_tokens.shape
            )
            event_scores = torch.zeros_like(oracle_scores)
            event_scores[:, 1:] = (image_tokens[:, 1:] != image_tokens[:, :-1]).float()

            uncertainty_pred = agent.world_model.curiosity_head[ObsModality.image.name].estimate_uncertainty(
                image_outputs[pred_mask]
            )[0]
            uncertainty_scores = _expand_pred_scores(
                uncertainty_pred, pred_mask, image_tokens.shape
            )
            reward_grad_scores = _reward_gradient_scores(agent, batch, base_embeddings)
            value_grad_scores = _value_gradient_scores(agent, base_embeddings)

            strategies = {
                "random": None,
                "event_change": event_scores,
                "uncertainty": uncertainty_scores,
                "reward_grad": reward_grad_scores,
                "value_grad": value_grad_scores,
                "oracle_ce": oracle_scores,
                "decision_score": event_scores
                + uncertainty_scores
                + reward_grad_scores
                + value_grad_scores,
            }

            for keep_ratio in args.keep_ratios:
                for name, scores in strategies.items():
                    if scores is None:
                        mask = _random_mask(
                            image_tokens.shape,
                            keep_ratio,
                            agent.world_model.context_length,
                            device,
                        )
                    else:
                        mask = _score_to_mask(
                            scores,
                            keep_ratio,
                            agent.world_model.context_length,
                        )
                    routed = mask.unsqueeze(-1) * base_embeddings + (1.0 - mask).unsqueeze(-1) * summary
                    with torch.no_grad():
                        _, _, _, losses = _image_logits_and_losses(agent, batch, image_embeddings=routed)
                    rows.append(
                        {
                            "batch": batch_idx,
                            "strategy": name,
                            "keep_ratio": keep_ratio,
                            "base_image_ce": base_ce,
                            "compressed_image_ce": losses.mean().item(),
                            "delta_image_ce": losses.mean().item() - base_ce,
                            "actual_keep_ratio": mask.mean().item(),
                        }
                    )

        out_dir = Path(args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "patch_importance.csv"
        json_path = out_dir / "patch_importance_summary.json"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary = {}
        for row in rows:
            key = f"{row['strategy']}@{row['keep_ratio']}"
            summary.setdefault(key, []).append(row["delta_image_ce"])
        summary = {
            key: {
                "mean_delta_image_ce": float(torch.tensor(vals).mean()),
                "num_batches": len(vals),
            }
            for key, vals in summary.items()
        }
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {csv_path}")
        print(f"Wrote {json_path}")
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="last", choices=["last", "best"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--summary-kernel", type=int, default=2)
    parser.add_argument("--output-dir", default="tools/decision_aware_precision_router/results")
    return parser.parse_args()


if __name__ == "__main__":
    run_diagnostic(parse_args())
