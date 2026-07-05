#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


EAWM_ATARI_BASELINES = {
    "Alien": {"Simulus": 691.8, "EASimulus": 740.0},
    "Assault": {"Simulus": 1528.8, "EASimulus": 2112.7},
    "Asterix": {"Simulus": 1477.3, "EASimulus": 1590.3},
    "Breakout": {"Simulus": 153.4, "EASimulus": 236.5},
}

TEST_RE = re.compile(
    r"\[metrics\]\[(?P<task>[^\]]+)\]\[epoch (?P<epoch>\d+)\] "
    r"test_return=(?P<return>[-+0-9.eE]+)"
)
ACTOR_RE = re.compile(
    r"\[metrics\]\[(?P<task>[^\]]+)\]\[epoch (?P<epoch>\d+)\] "
    r"ac_loss=(?P<loss>[-+0-9.eE]+).*?"
    r"imagined_return=(?P<imagined_return>[-+0-9.eE]+).*?"
    r"tree_adv=(?P<tree_adv>[-+0-9.eE]+|NA).*?"
    r"tree_w=(?P<tree_w>[-+0-9.eE]+|NA).*?"
    r"tree_u=(?P<tree_u>[-+0-9.eE]+|NA)"
)


@dataclass
class TaskSummary:
    run: str
    task: str
    train_log: Path
    last_actor_epoch: int | None = None
    last_imagined_return: float | None = None
    last_tree_adv: float | None = None
    last_tree_w: float | None = None
    last_tree_u: float | None = None
    last_test_epoch: int | None = None
    last_test_return: float | None = None
    best_test_epoch: int | None = None
    best_test_return: float | None = None

    @property
    def game(self) -> str:
        return self.task.split("__", 1)[0]


def parse_float(value: str) -> float | None:
    if value == "NA":
        return None
    return float(value)


def parse_train_log(path: Path) -> TaskSummary:
    task_from_path = path.parent.name
    run = path.parent.parent.name
    summary = TaskSummary(run=run, task=task_from_path, train_log=path)
    text = path.read_text(errors="ignore")

    for match in ACTOR_RE.finditer(text):
        summary.task = match.group("task")
        summary.last_actor_epoch = int(match.group("epoch"))
        summary.last_imagined_return = parse_float(match.group("imagined_return"))
        summary.last_tree_adv = parse_float(match.group("tree_adv"))
        summary.last_tree_w = parse_float(match.group("tree_w"))
        summary.last_tree_u = parse_float(match.group("tree_u"))

    for match in TEST_RE.finditer(text):
        summary.task = match.group("task")
        epoch = int(match.group("epoch"))
        value = float(match.group("return"))
        summary.last_test_epoch = epoch
        summary.last_test_return = value
        if summary.best_test_return is None or value > summary.best_test_return:
            summary.best_test_epoch = epoch
            summary.best_test_return = value

    return summary


def iter_train_logs(roots: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "train.log":
            logs.append(root)
        elif root.is_dir():
            logs.extend(root.glob("**/train.log"))
    return sorted(set(logs))


def ratio(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0:
        return "NA"
    return f"{value / baseline:.3f}"


def fmt(value: float | int | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4f}".rstrip("0").rstrip(".")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize TreeCF actor logs against EAWM Atari baselines."
    )
    parser.add_argument("paths", nargs="*", default=["logs"], help="Log roots or train.log files.")
    parser.add_argument("--min-epoch", type=int, default=200)
    args = parser.parse_args()

    summaries = [parse_train_log(path) for path in iter_train_logs([Path(p) for p in args.paths])]
    summaries = [s for s in summaries if s.task and "treecf" in s.task]
    summaries.sort(key=lambda s: (s.run, s.game, s.task))

    header = [
        "run",
        "task",
        "actor_ep",
        "last_test",
        "best_test",
        "easimulus",
        "best/easim",
        "simulus",
        "best/sim",
        "tree_w",
        "tree_u",
        "status",
    ]
    print("\t".join(header))
    for s in summaries:
        baselines = EAWM_ATARI_BASELINES.get(s.game, {})
        easim = baselines.get("EASimulus")
        sim = baselines.get("Simulus")
        reached = (s.last_actor_epoch or 0) >= args.min_epoch
        weak = reached and s.best_test_return is not None and easim is not None and s.best_test_return < 0.5 * easim
        status = "stop_candidate" if weak else ("watch" if reached else "early")
        row = [
            s.run,
            s.task,
            fmt(s.last_actor_epoch),
            f"{fmt(s.last_test_return)}@{fmt(s.last_test_epoch)}",
            f"{fmt(s.best_test_return)}@{fmt(s.best_test_epoch)}",
            fmt(easim),
            ratio(s.best_test_return, easim),
            fmt(sim),
            ratio(s.best_test_return, sim),
            fmt(s.last_tree_w),
            fmt(s.last_tree_u),
            status,
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
