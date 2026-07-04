#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys


METRIC_RE = re.compile(
    r"'([^']+)': ([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
EPOCH_RE = re.compile(r"Epoch (\d+) / (\d+)")
TRACEBACK_END_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception|Interrupt|Warning):|^AssertionError:|^RuntimeError:|^ValueError:|^FileNotFoundError:|^OSError:|^KeyError:|^IndexError:|^TypeError:"
)


def is_tqdm_line(line: str) -> bool:
    return "%|" in line or "it/s]" in line or line.startswith("\r")


def should_forward_status(line: str) -> bool:
    prefixes = (
        "[task]",
        "[train]",
        "[resume]",
        "[checkpoint]",
        "[runtime]",
        "[worker]",
        "[record]",
        "[actor-launch]",
    )
    if line.startswith(prefixes):
        return True
    return any(
        marker in line
        for marker in (
            "Successfully loaded model",
            "Saving checkpoint at epoch",
            "Best epoch",
            "CUDA out of memory",
            "Traceback",
        )
    )


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
            f"logp={fmt(pairs.get('actor_critic/train/imagined_log_probs_mean'))} "
            f"cf_uncert={fmt(pairs.get('actor_critic/train/imagined_counterfactual_uncertainty_mean'))} "
            f"cf_weight={fmt(pairs.get('actor_critic/train/imagined_counterfactual_weights_mean'))} "
            f"cf_group_std={fmt(pairs.get('actor_critic/train/imagined_counterfactual_group_std_mean'))}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    task = args.task
    current_epoch: int | None = None
    in_traceback = False

    for raw_line in sys.stdin:
        for line in raw_line.replace("\r", "\n").splitlines():
            line = line.rstrip("\n")
            if not line or is_tqdm_line(line):
                continue

            if "Traceback (most recent call last):" in line:
                in_traceback = True
                print(f"[traceback][{task}] {line}", flush=True)
                continue

            if in_traceback:
                print(f"[traceback][{task}] {line}", flush=True)
                if TRACEBACK_END_RE.search(line):
                    in_traceback = False
                continue

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total = int(epoch_match.group(2))
                print(f"[metrics][{task}] Epoch {current_epoch} / {total}", flush=True)
                continue

            if should_forward_status(line):
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
