#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys


METRIC_RE = re.compile(
    r"'([^']+)': ([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
EPOCH_RE = re.compile(r"Epoch (\d+) / (\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter verbose EASimulus logs into per-epoch monitor lines."
    )
    parser.add_argument("--task", required=True)
    return parser.parse_args()


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 1000:
        return f"{value:.4g}"
    if abs(value) >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def print_metric(task: str, epoch: int | None, name: str, pairs: dict[str, float]) -> None:
    prefix = f"[metrics][{task}]"
    if epoch is not None:
        prefix += f"[epoch {epoch}]"

    if name == "train":
        print(
            f"{prefix} train_return={fmt(pairs.get('train_dataset/return'))} "
            f"episodes={fmt(pairs.get('train_dataset/#episodes'))} "
            f"steps={fmt(pairs.get('train_dataset/#steps'))}",
            flush=True,
        )
    elif name == "test":
        print(
            f"{prefix} test_return={fmt(pairs.get('test_dataset/return'))} "
            f"episodes={fmt(pairs.get('test_dataset/#episodes'))} "
            f"steps={fmt(pairs.get('test_dataset/#steps'))}",
            flush=True,
        )
    elif name == "tokenizer":
        print(
            f"{prefix} tokenizer_loss={fmt(pairs.get('tokenizers/train/total_loss'))} "
            f"recon={fmt(pairs.get('tokenizers/train/reconstruction_loss'))} "
            f"perceptual={fmt(pairs.get('tokenizers/train/perceptual_loss'))} "
            f"codebook_epoch={fmt(pairs.get('tokenizers/train/Epoch codebook usage'))} "
            f"codebook_global={fmt(pairs.get('tokenizers/train/Global codebook usage'))}",
            flush=True,
        )
    elif name == "world_model":
        print(
            f"{prefix} wm_loss={fmt(pairs.get('world_model/train/total_loss'))} "
            f"obs={fmt(pairs.get('world_model/train/loss_obs'))} "
            f"reward={fmt(pairs.get('world_model/train/loss_rewards'))} "
            f"event={fmt(pairs.get('world_model/train/loss_events'))} "
            f"curiosity={fmt(pairs.get('world_model/train/curiosity_loss'))}",
            flush=True,
        )
    elif name == "actor_critic":
        print(
            f"{prefix} ac_loss={fmt(pairs.get('actor_critic/train/total_loss'))} "
            f"imagined_return={fmt(pairs.get('actor_critic/train/imagined_returns_mean'))} "
            f"imagined_reward={fmt(pairs.get('actor_critic/train/imagined_rewards_mean'))} "
            f"logp={fmt(pairs.get('actor_critic/train/imagined_log_probs_mean'))}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    task = args.task
    current_epoch: int | None = None

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            print(f"[metrics][{task}] Epoch {current_epoch} / {total}", flush=True)
            continue

        if "Successfully loaded model" in line:
            print(f"[metrics][{task}] {line}", flush=True)
            continue

        if "Saving checkpoint at epoch" in line:
            print(f"[metrics][{task}] {line}", flush=True)
            continue

        if "Best epoch" in line:
            print(f"[metrics][{task}] {line}", flush=True)
            continue

        pairs = {key: float(value) for key, value in METRIC_RE.findall(line)}
        if not pairs:
            continue

        if "train_dataset/return" in pairs:
            print_metric(task, current_epoch, "train", pairs)
        if "test_dataset/return" in pairs:
            print_metric(task, current_epoch, "test", pairs)
        if "tokenizers/train/total_loss" in pairs:
            print_metric(task, current_epoch, "tokenizer", pairs)
        if "world_model/train/total_loss" in pairs:
            print_metric(task, current_epoch, "world_model", pairs)
        if "actor_critic/train/total_loss" in pairs:
            print_metric(task, current_epoch, "actor_critic", pairs)


if __name__ == "__main__":
    main()
