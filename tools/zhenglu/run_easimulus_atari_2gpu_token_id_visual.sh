#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${EASIMULUS_DIR:-${PROJECT_ROOT}/EASimulus}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/ckpt/EASimulus/Atari}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/ouputs/token_id_visual}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/token_id_visual}"
VIDEO_SECONDS="${VIDEO_SECONDS:-600}"
FPS="${FPS:-1}"
SEED="${SEED:-0}"
GPU_IDS="${GPU_IDS:-0 1}"
FORCE="${FORCE:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${LOG_DIR}/token_id_visual_2gpu_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}" \
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
  if [[ "${USE_CONDA:-1}" != "1" ]]; then
    return
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[token-vis][warn] conda is not available; continuing with current python."
    return
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
  echo "[runtime] project_root=${PROJECT_ROOT}"
  echo "[runtime] easimulus_dir=${EASIMULUS_DIR}"
  echo "[runtime] ckpt_dir=${CKPT_DIR}"
  echo "[runtime] output_root=${OUTPUT_ROOT}"
  echo "[runtime] log_dir=${LOG_DIR}"
  echo "[runtime] video_seconds=${VIDEO_SECONDS}, fps=${FPS}"
  echo "[runtime] gpu_ids=${GPU_IDS}"
  echo "[runtime] force=${FORCE}"
  echo "[runtime] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "[runtime] nvidia-smi: not found"
  fi
  PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" python - <<'PY'
import importlib
import sys
import torch

for name in ["torch", "gymnasium", "ale_py", "hydra", "omegaconf", "PIL", "imageio", "imageio_ffmpeg"]:
    importlib.import_module(name)
    print(f"[runtime] imported {name}")
print(f"[runtime] python: {sys.version}")
print(f"[runtime] torch: {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Need at least 2 visible CUDA devices for this script.")
PY
}

check_inputs() {
  if [[ ! -d "${EASIMULUS_DIR}" ]]; then
    echo "[token-vis][error] EASimulus directory not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "[token-vis][error] Checkpoint directory not found: ${CKPT_DIR}"
    exit 1
  fi
  if [[ "${FPS}" != "1" ]]; then
    echo "[token-vis][error] FPS must be 1 because the requested video records one observed transition per second."
    exit 1
  fi
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
    echo "[token-vis][error] No checkpoints found in ${CKPT_DIR}."
    exit 1
  fi
}

task_to_env_id() {
  local task="$1"
  echo "${task}NoFrameskip-v4"
}

run_one() {
  local gpu="$1"
  local task="$2"
  local env_id="$3"
  local ckpt="${CKPT_DIR}/${task}.pt"
  local video="${OUTPUT_ROOT}/${task}.mp4"
  local tmp_video="${OUTPUT_ROOT}/${task}.tmp.${TIMESTAMP}.mp4"
  local log_file="${LOG_DIR}/${task}_${TIMESTAMP}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[token-vis][error] Missing checkpoint for ${task}: ${ckpt}" | tee -a "${log_file}"
    return 1
  fi
  if [[ -s "${video}" && "${FORCE}" != "1" ]]; then
    echo "[token-vis][skip] ${task}: existing output ${video}"
    return 0
  fi

  {
    echo "[task] date: $(date -Is)"
    echo "[task] gpu=${gpu}"
    echo "[task] task=${task}"
    echo "[task] env_id=${env_id}"
    echo "[task] ckpt=${ckpt}"
    echo "[task] video=${video}"
    rm -f "${tmp_video}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" \
      python "${PROJECT_ROOT}/tools/zhenglu/record_easimulus_atari_token_id_visual.py" \
        --easimulus-dir "${EASIMULUS_DIR}" \
        --env-id "${env_id}" \
        --checkpoint "${ckpt}" \
        --output "${tmp_video}" \
        --seconds "${VIDEO_SECONDS}" \
        --fps "${FPS}" \
        --seed "${SEED}"
    mv -f "${tmp_video}" "${video}"
  } >> "${log_file}" 2>&1
}

run_batch() {
  local start="$1"
  local -a gpus=(${GPU_IDS})
  local -a pids=()
  local -a names=()
  local slot gpu idx task env_id

  for slot in "${!gpus[@]}"; do
    idx=$((start + slot))
    if (( idx >= ${#TASKS_LIST[@]} )); then
      break
    fi
    gpu="${gpus[$slot]}"
    task="${TASKS_LIST[$idx]}"
    env_id="$(task_to_env_id "${task}")"
    echo "[token-vis] starting ${task} (${env_id}) on GPU ${gpu}"
    (run_one "${gpu}" "${task}" "${env_id}") &
    pids+=("$!")
    names+=("${task}")
  done

  local failures=0
  for idx in "${!pids[@]}"; do
    set +e
    wait "${pids[$idx]}"
    local rc=$?
    set -e
    if (( rc != 0 )); then
      echo "[token-vis][fail] ${names[$idx]} exited with ${rc}; log=${LOG_DIR}/${names[$idx]}_${TIMESTAMP}.log"
      failures=$((failures + 1))
    else
      echo "[token-vis][ok] ${names[$idx]} video=${OUTPUT_ROOT}/${names[$idx]}.mp4"
    fi
  done
  return "${failures}"
}

activate_conda
check_inputs
cd "${EASIMULUS_DIR}"
print_runtime_info
collect_tasks

echo "[token-vis] tasks (${#TASKS_LIST[@]}): ${TASKS_LIST[*]}"
echo "[token-vis] master log: ${MASTER_LOG}"

gpu_count="$(wc -w <<< "${GPU_IDS}")"
total_failures=0
for ((start=0; start<${#TASKS_LIST[@]}; start+=gpu_count)); do
  set +e
  run_batch "${start}"
  batch_failures=$?
  set -e
  total_failures=$((total_failures + batch_failures))
done

if (( total_failures > 0 )); then
  echo "[token-vis][error] ${total_failures} task(s) failed."
  echo "[token-vis][error] Master log: ${MASTER_LOG}"
  exit 1
fi

echo "[token-vis] All token-id visualizations completed."
echo "[token-vis] Videos: ${OUTPUT_ROOT}"
echo "[token-vis] Master log: ${MASTER_LOG}"
