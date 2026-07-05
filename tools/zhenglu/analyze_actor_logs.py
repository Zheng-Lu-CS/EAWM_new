#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


EAWM_ATARI_BASELINES = {
    "Alien": {"Simulus": 691.8, "EASimulus": 740.0},
    "Assault": {"Simulus": 1528.8, "EASimulus": 2112.7},
    "Asterix": {"Simulus": 1477.3, "EASimulus": 1590.3},
    "Breakout": {"Simulus": 153.4, "EASimulus": 236.5},
}

METRIC_RE = re.compile(
    r"\[metrics\]\[(?P<task>[^\]]+)\]\[epoch (?P<epoch>\d+)\] (?P<body>.*)"
)
TASK_KV_RE = re.compile(r"\[task\]\s+(?P<key>[A-Za-z0-9_.-]+)=(?P<value>.*)")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_.-]+)=(?P<value>[-+0-9.eE]+|NA)")
ERROR_RE = re.compile(
    r"Traceback|RuntimeError|AssertionError|ImportError|ValueError|Error\(s\)|"
    r"\[(?:dr|treecf|actor)-launch\]\[(?:error|fail|worker_fail)\]",
    re.IGNORECASE,
)


@dataclass
class LogSummary:
    run: str
    task: str
    train_log: Path
    source_run_dir: str | None = None
    last_actor_epoch: int | None = None
    last_test_epoch: int | None = None
    last_test_return: float | None = None
    best_test_epoch: int | None = None
    best_test_return: float | None = None
    errors: int = 0
    first_error: str | None = None
    last_metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def game(self) -> str:
        return self.task.split("__", 1)[0]

    @property
    def variant(self) -> str:
        return self.task.split("__", 1)[1] if "__" in self.task else self.task

    @property
    def source_kind(self) -> str:
        source = (self.source_run_dir or "").lower()
        if not source:
            return "unknown"
        if "decision_aware_precision_router" in source or "/dapr" in source:
            return "bad:dapr"
        if any(token in source for token in ("eadense", "edense", "dense_ablation")):
            return "bad:dense"
        if "easimulus_atari_" in source:
            return "easimulus"
        return "other"


def parse_float(value: str) -> float | None:
    if value == "NA":
        return None
    return float(value)


def parse_train_log(path: Path) -> LogSummary:
    task_from_path = path.parent.name
    run = path.parent.parent.name
    summary = LogSummary(run=run, task=task_from_path, train_log=path)
    text_parts = [path.read_text(errors="ignore")]
    raw_path = path.with_name("train.raw.log")
    if raw_path.exists():
        text_parts.insert(0, raw_path.read_text(errors="ignore"))

    for raw_line in "\n".join(text_parts).splitlines():
        line = raw_line.strip()
        if ERROR_RE.search(line):
            summary.errors += 1
            if summary.first_error is None:
                summary.first_error = line[:180]

        task_match = TASK_KV_RE.search(line)
        if task_match and task_match.group("key") == "source_run_dir":
            summary.source_run_dir = task_match.group("value").strip()

        metric_match = METRIC_RE.search(line)
        if not metric_match:
            continue

        summary.task = metric_match.group("task")
        epoch = int(metric_match.group("epoch"))
        body = metric_match.group("body")
        metrics = {
            match.group("key"): parse_float(match.group("value"))
            for match in KV_RE.finditer(body)
        }

        if "test_return" in metrics:
            value = metrics["test_return"]
            summary.last_test_epoch = epoch
            summary.last_test_return = value
            if value is not None and (
                summary.best_test_return is None or value > summary.best_test_return
            ):
                summary.best_test_epoch = epoch
                summary.best_test_return = value
        else:
            summary.last_actor_epoch = epoch
            summary.last_metrics = metrics

    return summary


def iter_train_logs(roots: list[Path]) -> list[Path]:
    logs: list[Path] = []
    for root in roots:
        if root.is_file() and root.name == "train.log":
            logs.append(root)
        elif root.is_dir():
            logs.extend(root.glob("**/train.log"))
    return sorted(set(logs))


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


def ratio(value: float | None, baseline: float | None) -> str:
    if value is None or baseline is None or baseline == 0:
        return "NA"
    return f"{value / baseline:.3f}"


def score_status(summary: LogSummary, min_epoch: int, stop_ratio: float) -> str:
    if summary.errors and summary.best_test_return is None:
        return "error"
    baselines = EAWM_ATARI_BASELINES.get(summary.game, {})
    simulus = baselines.get("Simulus")
    easimulus = baselines.get("EASimulus")
    best = summary.best_test_return
    reached = (summary.last_actor_epoch or summary.last_test_epoch or 0) >= min_epoch
    if best is not None and easimulus is not None and best >= easimulus:
        return "beats_easimulus"
    if best is not None and simulus is not None and best >= simulus:
        return "beats_simulus"
    if reached and best is not None and easimulus is not None and best < stop_ratio * easimulus:
        return "stop_candidate"
    if reached:
        return "watch"
    return "early"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize EASimulus actor logs against Atari baselines."
    )
    parser.add_argument("paths", nargs="*", default=["logs"], help="Log roots or train.log files.")
    parser.add_argument("--min-epoch", type=int, default=200)
    parser.add_argument("--stop-ratio", type=float, default=0.5)
    parser.add_argument(
        "--contains",
        default="",
        help="Only include task/run paths containing this case-insensitive substring.",
    )
    args = parser.parse_args()

    summaries = [parse_train_log(path) for path in iter_train_logs([Path(p) for p in args.paths])]
    if args.contains:
        needle = args.contains.lower()
        summaries = [
            s
            for s in summaries
            if needle in s.task.lower() or needle in s.run.lower() or needle in str(s.train_log).lower()
        ]
    summaries.sort(key=lambda s: (s.run, s.game, s.variant, str(s.train_log)))

    header = [
        "run",
        "task",
        "source",
        "actor_ep",
        "last_test",
        "best_test",
        "easimulus",
        "best/easim",
        "simulus",
        "best/sim",
        "dr_adv",
        "dr_w",
        "q_loss",
        "imag_ret",
        "tree_adv",
        "tree_w",
        "tree_u",
        "tree_risk",
        "tree_ent",
        "tree_ret_std",
        "errors",
        "status",
    ]
    print("\t".join(header))
    for summary in summaries:
        baselines = EAWM_ATARI_BASELINES.get(summary.game, {})
        easim = baselines.get("EASimulus")
        sim = baselines.get("Simulus")
        metrics = summary.last_metrics
        row = [
            summary.run,
            summary.task,
            summary.source_kind,
            fmt(summary.last_actor_epoch),
            f"{fmt(summary.last_test_return)}@{fmt(summary.last_test_epoch)}",
            f"{fmt(summary.best_test_return)}@{fmt(summary.best_test_epoch)}",
            fmt(easim),
            ratio(summary.best_test_return, easim),
            fmt(sim),
            ratio(summary.best_test_return, sim),
            fmt(metrics.get("dr_adv")),
            fmt(metrics.get("dr_w")),
            fmt(metrics.get("q_loss")),
            fmt(metrics.get("imagined_return")),
            fmt(metrics.get("tree_adv")),
            fmt(metrics.get("tree_w")),
            fmt(metrics.get("tree_u")),
            fmt(metrics.get("tree_risk")),
            fmt(metrics.get("tree_ent")),
            fmt(metrics.get("tree_return_std")),
            str(summary.errors),
            score_status(summary, args.min_epoch, args.stop_ratio),
        ]
        print("\t".join(row))


if __name__ == "__main__":
    main()
