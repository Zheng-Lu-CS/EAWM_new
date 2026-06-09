#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
EXP_TIMESTAMP="${EXP_TIMESTAMP:-20260609_001420}"
VIDEO_SECONDS="${VIDEO_SECONDS:-360}"
FPS="${FPS:-15}"
SEED="${SEED:-0}"
TASKS="${TASKS:-Breakout Boxing Seaquest RoadRunner}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
VIDEO_ROOT="${PROJECT_ROOT}/videos/easimulus_atari_training_results_${EXP_TIMESTAMP}_${TIMESTAMP}"
MASTER_LOG="${LOG_DIR}/record_easimulus_atari_training_results_${EXP_TIMESTAMP}_${TIMESTAMP}.log"
MANIFEST="${VIDEO_ROOT}/manifest.tsv"

mkdir -p "${LOG_DIR}" "${VIDEO_ROOT}" \
  "${PROJECT_ROOT}/cache/pip" "${PROJECT_ROOT}/cache/torch" \
  "${PROJECT_ROOT}/cache/huggingface" "${PROJECT_ROOT}/cache/xdg" \
  "${PROJECT_ROOT}/cache/matplotlib"
exec > >(tee -a "${MASTER_LOG}") 2>&1

export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[record][error] conda is not available in PATH."
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

print_runtime_info() {
  echo "[runtime] hostname: $(hostname)"
  echo "[runtime] date: $(date -Is)"
  echo "[runtime] pwd: $(pwd)"
  echo "[runtime] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
  echo "[runtime] PROJECT_ROOT=${PROJECT_ROOT}"
  echo "[runtime] EASIMULUS_DIR=${EASIMULUS_DIR}"
  echo "[runtime] EXP_TIMESTAMP=${EXP_TIMESTAMP}"
  echo "[runtime] VIDEO_ROOT=${VIDEO_ROOT}"
  echo "[runtime] VIDEO_SECONDS=${VIDEO_SECONDS}"
  echo "[runtime] FPS=${FPS}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "[runtime] nvidia-smi: not found"
  fi
  python - <<'PY'
import sys, torch
print(f"[runtime] python: {sys.version}")
print(f"[runtime] torch: {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
PY
}

check_inputs() {
  if [[ ! -d "${EASIMULUS_DIR}" ]]; then
    echo "[record][error] EASimulus directory not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -d "${LOG_DIR}" ]]; then
    echo "[record][error] Log directory not found: ${LOG_DIR}"
    exit 1
  fi
  local count
  count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( count < 4 )); then
    echo "[record][error] Need at least 4 visible CUDA devices; found ${count}."
    exit 1
  fi
}

build_manifest() {
  TASKS="${TASKS}" PROJECT_ROOT="${PROJECT_ROOT}" EXP_TIMESTAMP="${EXP_TIMESTAMP}" SEED="${SEED}" MANIFEST="${MANIFEST}" python - <<'PY'
from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
logs_dir = project_root / "logs"
exp_timestamp = os.environ["EXP_TIMESTAMP"]
seed = os.environ["SEED"]
tasks = os.environ["TASKS"].replace(",", " ").split()
manifest = Path(os.environ["MANIFEST"])

epoch_re = re.compile(r"Epoch (\d+) /")
best_re = re.compile(r"Best epoch (\d+), best score: ([-+0-9.eE]+)")
metric_re = re.compile(r"'([^']+)': ([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)")


def parse_run_dir(log_path: Path, task: str) -> Path | None:
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("[task] command:"):
            command = line.split(":", 1)[1].strip()
            for token in shlex.split(command):
                if token.startswith("hydra.run.dir="):
                    return Path(token.split("=", 1)[1])
    master = logs_dir / f"train_easimulus_atari_4gpu_seed{seed}_{exp_timestamp}.log"
    if master.exists():
        pattern = re.compile(rf"\[train\]\[resume\]\s+{re.escape(task)}:\s+(.+)$")
        for line in master.read_text(encoding="utf-8", errors="replace").splitlines():
            m = pattern.search(line)
            if m:
                return Path(m.group(1).strip())
    return None


def parse_metrics(log_path: Path):
    current_epoch = None
    best_epoch = ""
    best_score = ""
    eval16 = ""
    final_eval = ""
    final_epoch = ""
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
        m = best_re.search(line)
        if m:
            best_epoch = m.group(1)
            best_score = m.group(2)
        if current_epoch is not None and "'test_dataset/return'" in line:
            pairs = {k: float(v) for k, v in metric_re.findall(line)}
            if "test_dataset/return" not in pairs:
                continue
            episodes = int(pairs.get("test_dataset/#episodes", 0))
            final_epoch = str(current_epoch)
            if current_epoch == 600 and episodes == 16:
                eval16 = f"{pairs['test_dataset/return']:.6g}"
            elif current_epoch == 600 and episodes > 16:
                final_eval = f"{pairs['test_dataset/return']:.6g}"
    return best_epoch, best_score, final_epoch, eval16, final_eval


rows = []
for task in tasks:
    log_path = logs_dir / f"train_easimulus_atari_{task}_seed{seed}_{exp_timestamp}_resume.log"
    if not log_path.exists():
        raise SystemExit(f"[manifest][error] Missing task log: {log_path}")

    run_dir = parse_run_dir(log_path, task)
    if run_dir is None:
        raise SystemExit(f"[manifest][error] Could not resolve hydra run dir for {task} from {log_path}")

    ckpt_dir = run_dir / "checkpoints"
    best_ckpt = ckpt_dir / "best.pt"
    final_ckpt = ckpt_dir / "last.pt"
    for path in (best_ckpt, final_ckpt):
        if not path.exists():
            raise SystemExit(f"[manifest][error] Missing checkpoint for {task}: {path}")

    best_epoch, best_score, final_epoch, eval16, final_eval = parse_metrics(log_path)
    if not best_epoch or not best_score:
        raise SystemExit(f"[manifest][error] Could not parse best score for {task}: {log_path}")
    if not final_epoch or not eval16:
        raise SystemExit(f"[manifest][error] Could not parse final eval16 for {task}: {log_path}")
    if not final_eval:
        final_eval = eval16

    env_id = f"{task}NoFrameskip-v4"
    rows.append([
        task,
        env_id,
        str(run_dir),
        str(best_ckpt),
        str(final_ckpt),
        best_epoch,
        f"{float(best_score):.6g}",
        final_epoch,
        eval16,
        final_eval,
    ])

manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("w", encoding="utf-8") as f:
    f.write("task\tenv_id\trun_dir\tbest_ckpt\tfinal_ckpt\tbest_epoch\tbest_score\tfinal_epoch\tfinal_eval16\tfinal_eval120\n")
    for row in rows:
        f.write("\t".join(row) + "\n")

print(f"[manifest] wrote {manifest}")
for row in rows:
    print("[manifest] " + " | ".join(row))
PY
}

run_one() {
  local gpu="$1"
  local task="$2"
  local env_id="$3"
  local run_dir="$4"
  local best_ckpt="$5"
  local final_ckpt="$6"
  local best_epoch="$7"
  local best_score="$8"
  local final_epoch="$9"
  local final_eval16="${10}"
  local final_eval120="${11}"
  local video="${VIDEO_ROOT}/${task}_best_epoch_${best_epoch}_vs_final_${final_epoch}.mp4"
  local log_file="${LOG_DIR}/record_easimulus_atari_${task}_${EXP_TIMESTAMP}_${TIMESTAMP}.log"

  {
    echo "[task] hostname: $(hostname)"
    echo "[task] date: $(date -Is)"
    echo "[task] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[task] task=${task}"
    echo "[task] env_id=${env_id}"
    echo "[task] run_dir=${run_dir}"
    echo "[task] best_ckpt=${best_ckpt}"
    echo "[task] final_ckpt=${final_ckpt}"
    echo "[task] video=${video}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" \
      python "${PROJECT_ROOT}/tools/zhenglu/record_easimulus_atari_demo.py" \
        --easimulus-dir "${EASIMULUS_DIR}" \
        --env-id "${env_id}" \
        --checkpoint "${best_ckpt}" \
        --output "${video}" \
        --seconds "${VIDEO_SECONDS}" \
        --fps "${FPS}" \
        --seed "${SEED}" \
        --min-width 640 \
        --overlay-line "${task} | best checkpoint playback" \
        --overlay-line "best ckpt: epoch ${best_epoch}, eval16 ${best_score}" \
        --overlay-line "final ckpt: epoch ${final_epoch}, eval16 ${final_eval16}, final120 ${final_eval120}" \
        --overlay-line "experiment ${EXP_TIMESTAMP}"
  } >> "${log_file}" 2>&1
}

activate_conda
cd "${EASIMULUS_DIR}"
print_runtime_info
check_inputs
build_manifest

echo "[record] manifest: ${MANIFEST}"
echo "[record] videos: ${VIDEO_ROOT}"
echo "[record] master log: ${MASTER_LOG}"

declare -a pids=()
declare -a names=()
gpu=0
while IFS=$'\t' read -r task env_id run_dir best_ckpt final_ckpt best_epoch best_score final_epoch final_eval16 final_eval120; do
  if [[ "${task}" == "task" ]]; then
    continue
  fi
  echo "[record] starting ${task} on GPU ${gpu}"
  (run_one "${gpu}" "${task}" "${env_id}" "${run_dir}" "${best_ckpt}" "${final_ckpt}" "${best_epoch}" "${best_score}" "${final_epoch}" "${final_eval16}" "${final_eval120}") &
  pids+=("$!")
  names+=("${task}")
  gpu=$((gpu + 1))
done < "${MANIFEST}"

failures=0
for i in "${!pids[@]}"; do
  set +e
  wait "${pids[$i]}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[record][fail] ${names[$i]} exited with ${rc}"
    failures=$((failures + 1))
  else
    echo "[record][ok] ${names[$i]}"
  fi
done

if (( failures > 0 )); then
  echo "[record][error] ${failures} recording task(s) failed."
  echo "[record][error] Master log: ${MASTER_LOG}"
  echo "[record][error] Partial videos: ${VIDEO_ROOT}"
  exit 1
fi

echo "[record] All recordings completed."
echo "[record] Videos: ${VIDEO_ROOT}"
echo "[record] Manifest: ${MANIFEST}"
echo "[record] Master log: ${MASTER_LOG}"
