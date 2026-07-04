#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys


METRIC_RE = re.compile(
    r"'([^']+)': ([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)"
)
EPOCH_RE = re.compile(r"Epoch (\d+) / (\d+)")
TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
ERROR_LINE_RE = re.compile(
    r"(RuntimeError|ValueError|KeyError|AssertionError|Exception|Error executing job|"
    r"Missing key|Unexpected key|size mismatch|Error\(s\) in loading state_dict|"
    r"ConfigCompositionException|ConfigAttributeError|InstantiationException|"
    r"ModuleNotFoundError|ImportError|CUDA out of memory|OutOfMemoryError|"
    r"\[state-dict\]|Failed to load)",
    re.IGNORECASE,
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
            f"tree_adv={fmt(pairs.get('actor_critic/train/imagined_treecf_advantages_mean'))} "
            f"tree_w={fmt(pairs.get('actor_critic/train/imagined_treecf_weights_mean'))} "
            f"tree_u={fmt(pairs.get('actor_critic/train/imagined_treecf_uncertainty_mean'))} "
            f"tree_root={fmt(pairs.get('actor_critic/train/imagined_treecf_root_values_mean'))} "
            f"dr_adv={fmt(pairs.get('actor_critic/train/imagined_dr_advantages_mean'))} "
            f"dr_res={fmt(pairs.get('actor_critic/train/imagined_dr_residuals_mean'))} "
            f"dr_q={fmt(pairs.get('actor_critic/train/imagined_dr_q_values_mean'))} "
            f"dr_w={fmt(pairs.get('actor_critic/train/imagined_dr_weights_mean'))} "
            f"dr_u={fmt(pairs.get('actor_critic/train/imagined_dr_uncertainty_mean'))} "
            f"q_loss={fmt(pairs.get('actor_critic/train/imagined_loss_q'))}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    task = args.task
    current_epoch: int | None = None
    in_traceback = False

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")

        if TRACEBACK_START_RE.search(line):
            in_traceback = True
            print(f"[traceback][{task}] {line}", flush=True)
            continue

        if in_traceback and METRIC_RE.findall(line):
            in_traceback = False

        if in_traceback:
            print(f"[traceback][{task}] {line}", flush=True)
            if not line:
                in_traceback = False
            continue

        if ERROR_LINE_RE.search(line):
            print(f"[traceback][{task}] {line}", flush=True)
            continue

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
