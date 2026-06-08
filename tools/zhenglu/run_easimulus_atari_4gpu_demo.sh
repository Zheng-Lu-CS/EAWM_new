#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/ckpt}"
VIDEO_SECONDS="${VIDEO_SECONDS:-180}"
FPS="${FPS:-15}"
SEED="${SEED:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
VIDEO_ROOT="${PROJECT_ROOT}/videos/easimulus_atari_demo_${TIMESTAMP}"
MASTER_LOG="${LOG_DIR}/demo_easimulus_atari_4gpu_${TIMESTAMP}.log"

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

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[demo][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "[demo][error] Checkpoint directory not found: ${CKPT_DIR}"
  echo "[demo][hint] Create it and put official checkpoints such as Breakout.pt there."
  exit 1
fi

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[demo][error] conda is not available in PATH."
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
  echo "[runtime] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[runtime] CKPT_DIR=${CKPT_DIR}"
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

check_4gpu() {
  local count
  count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( count < 4 )); then
    echo "[demo][error] Need at least 4 visible CUDA devices for this 4-GPU demo script; found ${count}."
    echo "[demo][error] A 1-GPU setup node is normal, but demo recording should run inside a 4-GPU compute job."
    exit 1
  fi
}

task_to_env_id() {
  local task="$1"
  echo "${task}NoFrameskip-v4"
}

collect_tasks() {
  TASKS_LIST=()
  if [[ -n "${TASKS:-}" ]]; then
    local raw="${TASKS//,/ }"
    local task
    for task in ${raw}; do
      task="${task%.pt}"
      TASKS_LIST+=("${task}")
    done
  else
    local ckpt
    while IFS= read -r ckpt; do
      TASKS_LIST+=("$(basename "${ckpt}" .pt)")
    done < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.pt" | sort)
  fi

  if (( ${#TASKS_LIST[@]} == 0 )); then
    echo "[demo][error] No checkpoints found in ${CKPT_DIR}."
    echo "[demo][hint] Expected files like ${CKPT_DIR}/Breakout.pt and ${CKPT_DIR}/RoadRunner.pt."
    exit 1
  fi
}

run_one() {
  local gpu="$1"
  local task="$2"
  local env_id="$3"
  local ckpt="${CKPT_DIR}/${task}.pt"
  local video="${VIDEO_ROOT}/${task}.mp4"
  local log_file="${LOG_DIR}/demo_easimulus_atari_${task}_${TIMESTAMP}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[demo][error] Missing checkpoint for ${task}: ${ckpt}" | tee -a "${log_file}"
    return 1
  fi

  {
    echo "[task] hostname: $(hostname)"
    echo "[task] date: $(date -Is)"
    echo "[task] pwd: $(pwd)"
    echo "[task] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
    echo "[task] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[task] task=${task}"
    echo "[task] env_id=${env_id}"
    echo "[task] ckpt=${ckpt}"
    echo "[task] video=${video}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" \
      python "${PROJECT_ROOT}/tools/zhenglu/record_easimulus_atari_demo.py" \
        --easimulus-dir "${EASIMULUS_DIR}" \
        --env-id "${env_id}" \
        --checkpoint "${ckpt}" \
        --output "${video}" \
        --seconds "${VIDEO_SECONDS}" \
        --fps "${FPS}" \
        --seed "${SEED}"
  } >> "${log_file}" 2>&1
}

run_batch() {
  local start="$1"
  local -a pids=()
  local -a names=()
  local i task env_id gpu

  for gpu in 0 1 2 3; do
    i=$((start + gpu))
    if (( i >= ${#TASKS_LIST[@]} )); then
      break
    fi
    task="${TASKS_LIST[$i]}"
    env_id="$(task_to_env_id "${task}")"
    echo "[demo] starting ${task} (${env_id}) on GPU ${gpu}"
    (run_one "${gpu}" "${task}" "${env_id}") &
    pids+=("$!")
    names+=("${task}")
  done

  local failures=0
  for i in "${!pids[@]}"; do
    set +e
    wait "${pids[$i]}"
    local rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[demo][fail] ${names[$i]} exited with ${rc}; log=${LOG_DIR}/demo_easimulus_atari_${names[$i]}_${TIMESTAMP}.log"
      failures=$((failures + 1))
    else
      echo "[demo][ok] ${names[$i]} video=${VIDEO_ROOT}/${names[$i]}.mp4"
    fi
  done
  return "${failures}"
}

activate_conda
cd "${EASIMULUS_DIR}"
print_runtime_info
check_4gpu
collect_tasks

echo "[demo] tasks (${#TASKS_LIST[@]}): ${TASKS_LIST[*]}"
echo "[demo] master log: ${MASTER_LOG}"

total_failures=0
for ((start=0; start<${#TASKS_LIST[@]}; start+=4)); do
  set +e
  run_batch "${start}"
  batch_failures=$?
  set -e
  total_failures=$((total_failures + batch_failures))
done

if (( total_failures > 0 )); then
  echo "[demo][error] ${total_failures} demo task(s) failed."
  echo "[demo][error] Master log: ${MASTER_LOG}"
  echo "[demo][error] Videos written under: ${VIDEO_ROOT}"
  exit 1
fi

echo "[demo] All demo recordings completed."
echo "[demo] Videos: ${VIDEO_ROOT}"
echo "[demo] Master log: ${MASTER_LOG}"
