#!/usr/bin/env python
"""Report structural compute differences between original POP and dense Atari WM."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def timed_cuda(fn, device, warmup=3, steps=10):
    import torch

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(steps):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = 0
    return (time.perf_counter() - start) / steps, peak


def build_agent(easimulus_dir: Path, device):
    import gymnasium
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    src_dir = easimulus_dir / "src"
    sys.path.insert(0, str(src_dir))

    from main import build_agent
    from utils import ObsModality

    class DummyAtariEnv:
        modalities = {ObsModality.image}
        action_space = gymnasium.spaces.Discrete(18)
        num_actions = 18
        observation_space = {
            ObsModality.image: gymnasium.spaces.Box(
                low=0, high=255, shape=(1, 64, 64, 3), dtype="uint8"
            )
        }

    cfg = OmegaConf.create(
        {
            "common": {"device": str(device)},
            "initialization": {"video": {"path": None}},
            "env": {"train": {"hsize": 64, "wsize": 64}},
            "tokenizer": {
                "image": OmegaConf.load(easimulus_dir / "config/tokenizer/image/default.yaml")
            },
            "actor_critic": OmegaConf.load(easimulus_dir / "config/actor_critic/atari.yaml"),
            "world_model": OmegaConf.load(easimulus_dir / "config/world_model/atari.yaml"),
        }
    )
    cfg.tokenizer.image.with_lpips = False
    cfg.tokenizer.image.vgg_lpips_ckpt_path = str(
        easimulus_dir / "cache/rem/tokenizer_pretrained_vgg"
    )
    OmegaConf.resolve(cfg)
    return build_agent(DummyAtariEnv(), cfg, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easimulus-dir", type=Path, default=Path("EASimulus"))
    parser.add_argument("--output", type=Path, default=Path("logs/dense_compute_report.md"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:
        write_fallback_report(args.output, f"torch import failed: {type(exc).__name__}: {exc}")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    try:
        agent = build_agent(args.easimulus_dir.resolve(), device)
    except Exception as exc:
        write_fallback_report(
            args.output,
            f"agent instantiation failed: {type(exc).__name__}: {exc}",
        )
        return
    agent.to(device)
    tokenizer = agent.tokenizer
    wm = agent.world_model
    ac = agent.actor_critic

    tokenizer_params = count_params(tokenizer)
    dense_wm_params = count_params(wm)
    ac_params = count_params(ac)
    embed_dim = wm.config.embed_dim
    old_tokens_per_block = wm.tokens_per_obs + wm.tokens_per_action
    dense_tokens_per_block = wm.tokens_per_block
    placeholder_params = old_tokens_per_block * embed_dim
    cls_params = wm.cls_embedding.numel()
    pos_params = wm.image_pos_embedding.numel() if wm.image_pos_embedding is not None else 0
    adaln_params = count_params(wm.action_adaln)
    old_wm_params_est = dense_wm_params - cls_params - pos_params - adaln_params + placeholder_params

    b, t = args.batch_size, args.seq_len
    observations = {
        next(iter(tokenizer.modalities)): torch.rand(b, t, 3, 64, 64, device=device)
    }
    actions = torch.randint(0, 18, (b, t), device=device)
    rewards = torch.randn(b, t, device=device)
    ends = torch.zeros(b, t, dtype=torch.long, device=device)
    mask_padding = torch.ones(b, t, dtype=torch.bool, device=device)
    batch = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "ends": ends,
        "mask_padding": mask_padding,
    }

    def wm_forward_loss():
        with torch.no_grad():
            losses, _ = wm.compute_loss(batch, tokenizer)
            return losses.loss_total

    try:
        wm_time, wm_peak = timed_cuda(wm_forward_loss, device)
        timing_note = f"{wm_time * 1000:.2f} ms"
        memory_note = f"{wm_peak / (1024 ** 2):.1f} MiB" if wm_peak else "N/A on CPU"
    except Exception as exc:
        timing_note = f"failed: {type(exc).__name__}: {exc}"
        memory_note = "N/A"

    original_effective_tokens = old_tokens_per_block + wm.tokens_per_obs
    dense_effective_tokens = dense_tokens_per_block
    token_path_ratio = dense_effective_tokens / original_effective_tokens

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# Dense Atari Compute Report",
                "",
                "## Parameter Count",
                "",
                "| Component | Original POP estimate | Dense model | Delta |",
                "|---|---:|---:|---:|",
                f"| Tokenizer | {tokenizer_params:,} | {tokenizer_params:,} | 0 |",
                f"| World model | {old_wm_params_est:,} | {dense_wm_params:,} | {dense_wm_params - old_wm_params_est:,} |",
                f"| Actor critic | {ac_params:,} | {ac_params:,} | 0 |",
                f"| Total | {tokenizer_params + old_wm_params_est + ac_params:,} | {tokenizer_params + dense_wm_params + ac_params:,} | {dense_wm_params - old_wm_params_est:,} |",
                "",
                "## Token Path",
                "",
                f"- Original RetNet block length: `{old_tokens_per_block}` = `{wm.tokens_per_obs}` obs tokens + `{wm.tokens_per_action}` action token.",
                f"- Dense RetNet block length: `{dense_tokens_per_block}` = `{wm.tokens_per_obs}` obs tokens + `1` CLS token.",
                f"- Original POP training additionally runs `{wm.tokens_per_obs}` prediction tokens per block; effective token path is approximately `{original_effective_tokens}` tokens/block.",
                f"- Dense training uses approximately `{dense_effective_tokens}` tokens/block, ratio `{token_path_ratio:.3f}` of the original POP token path.",
                "",
                "## Added / Removed Parameters",
                "",
                f"- Removed POP placeholder embedding estimate: `{placeholder_params:,}` params.",
                f"- Added CLS embedding: `{cls_params:,}` params.",
                f"- Added learnable 2D patch position embedding: `{pos_params:,}` params.",
                f"- Added action-to-AdaLN MLP: `{adaln_params:,}` params.",
                "- Tokenizer LPIPS is structurally unchanged, but dense tokenizer training calls LPIPS once per grayscale channel, so LPIPS loss compute is about `3x` the previous single image LPIPS call.",
                "",
                "## Smoke Timing",
                "",
                f"- Device: `{device}`",
                f"- Dummy batch: batch size `{b}`, sequence length `{t}`",
                f"- Dense world-model forward+loss time: {timing_note}",
                f"- Dense peak CUDA memory: {memory_note}",
                "",
                "Note: original POP timing is reported analytically because this code path is removed from the dense implementation. Use the token-path ratio above for the RetNet training-forward comparison.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[compute] wrote {args.output}")


def write_fallback_report(output: Path, reason: str):
    embed_dim = 256
    num_layers = 10
    tokens_per_obs = 64
    tokens_per_action = 1
    old_tokens_per_block = tokens_per_obs + tokens_per_action
    dense_tokens_per_block = tokens_per_obs + 1
    placeholder_params = old_tokens_per_block * embed_dim
    cls_params = embed_dim
    pos_params = tokens_per_obs * embed_dim
    adaln_params = (
        embed_dim * (4 * embed_dim)
        + (4 * embed_dim)
        + (4 * embed_dim) * (num_layers * 2 * 2 * embed_dim)
        + (num_layers * 2 * 2 * embed_dim)
    )
    original_effective_tokens = old_tokens_per_block + tokens_per_obs
    token_path_ratio = dense_tokens_per_block / original_effective_tokens
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# Dense Atari Compute Report",
                "",
                f"Full runtime profiling was not available: `{reason}`.",
                "",
                "## Analytic Structural Estimate",
                "",
                f"- Original RetNet block length: `{old_tokens_per_block}` = `{tokens_per_obs}` obs tokens + `{tokens_per_action}` action token.",
                f"- Dense RetNet block length: `{dense_tokens_per_block}` = `{tokens_per_obs}` obs tokens + `1` CLS token.",
                f"- Original POP training additionally runs `{tokens_per_obs}` prediction tokens per block; approximate effective token path `{original_effective_tokens}`.",
                f"- Dense approximate effective token path `{dense_tokens_per_block}`, ratio `{token_path_ratio:.3f}` of original.",
                "",
                "## Added / Removed Parameters",
                "",
                f"- Removed POP placeholder embedding estimate: `{placeholder_params:,}` params.",
                f"- Added CLS embedding: `{cls_params:,}` params.",
                f"- Added learnable 2D patch position embedding: `{pos_params:,}` params.",
                f"- Added action-to-AdaLN MLP estimate: `{adaln_params:,}` params.",
                f"- Net structural parameter delta estimate: `{cls_params + pos_params + adaln_params - placeholder_params:,}` params.",
                "- Tokenizer LPIPS is structurally unchanged, but dense tokenizer training calls LPIPS once per grayscale channel, so LPIPS loss compute is about `3x` the previous single image LPIPS call.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[compute] wrote fallback report {output}")


if __name__ == "__main__":
    main()
