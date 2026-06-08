#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/smoke_easimulus_atari_4gpu_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/outputs" \
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[smoke][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[smoke][error] conda is not available in PATH."
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
    echo "[smoke][error] Need at least 4 visible CUDA devices for this 4-GPU smoke script; found ${count}."
    echo "[smoke][error] A 1-GPU environment setup node is normal, but this script must run inside a 4-GPU compute job."
    exit 1
  fi
}

run_one() {
  local gpu="$1"
  local game="$2"
  local media_count="$3"
  local log_file="$4"
  local task_output="$5"
  shift 5
  local cmd=("$@")

  {
    echo "[task] hostname: $(hostname)"
    echo "[task] date: $(date -Is)"
    echo "[task] pwd: $(pwd)"
    echo "[task] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
    echo "[task] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[task] game=${game}"
    echo "[task] output=${task_output}"
    echo "[task] media_episode_count=${media_count}"
    echo "[task] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } >> "${log_file}" 2>&1
}

start_task() {
  local gpu="$1"
  local game="$2"
  local game_short="${game%NoFrameskip-v4}"
  local log_file="${LOG_DIR}/smoke_easimulus_atari_${game_short}_${TIMESTAMP}.log"
  local task_output="${PROJECT_ROOT}/outputs/easimulus_atari_4gpu_smoke_${TIMESTAMP}/${game_short}"
  mkdir -p "${task_output}"
  local media_count="${MEDIA_EPISODES_TO_SAVE:-0}"

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "common.seed=${SEED:-0}"
    world_model.event_pred=True
    world_model.ges=True
    wandb.mode=disabled
    "wandb.name=${game_short}-smoke-seed${SEED:-0}"
    "outputs_dir_path=${task_output}"
    common.epochs=3
    collection.train.stop_after_epochs=3
    collection.train.config.num_steps=200
    training.tokenizer.start_after_epochs=0
    training.tokenizer.steps_per_epoch=1
    training.world_model.start_after_epochs=1
    training.world_model.steps_per_epoch=1
    training.actor_critic.real_start_after_epochs=1
    training.actor_critic.start_after_epochs=2
    training.actor_critic.steps_per_epoch=1
    evaluation.every=1
    collection.test.config.num_episodes=1
    collection.test.config.num_episodes_end=1
    env.test.max_episode_steps=1000
    common.do_checkpoint=False
    "collection.train.num_episodes_to_save=${media_count}"
    "collection.test.num_episodes_to_save=${media_count}"
    evaluation.tokenizer.save_reconstructions=False
  )

  (
    set +e
    run_one "${gpu}" "${game}" "${media_count}" "${log_file}" "${task_output}" "${cmd[@]}"
    local rc=$?
    if (( rc != 0 )) && [[ "${media_count}" == "0" ]] && [[ "${RETRY_MEDIA_SAVE_ONE:-1}" == "1" ]]; then
      echo "[task] first attempt failed with media save count 0; retrying once with count 1" >> "${log_file}" 2>&1
      cmd=("${cmd[@]/collection.train.num_episodes_to_save=0/collection.train.num_episodes_to_save=1}")
      cmd=("${cmd[@]/collection.test.num_episodes_to_save=0/collection.test.num_episodes_to_save=1}")
      run_one "${gpu}" "${game}" "1" "${log_file}" "${task_output}" "${cmd[@]}"
      rc=$?
    fi
    exit "${rc}"
  ) &

  PIDS+=("$!")
  NAMES+=("${game_short}")
  LOGS+=("${log_file}")
  echo "[smoke] started ${game} on GPU ${gpu}; log=${log_file}"
}

activate_conda
cd "${EASIMULUS_DIR}"
print_runtime_info
check_4gpu

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()

start_task 0 BreakoutNoFrameskip-v4
start_task 1 BoxingNoFrameskip-v4
start_task 2 SeaquestNoFrameskip-v4
start_task 3 RoadRunnerNoFrameskip-v4

failures=0
for i in "${!PIDS[@]}"; do
  set +e
  wait "${PIDS[$i]}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[smoke][fail] ${NAMES[$i]} exited with ${rc}; log=${LOGS[$i]}"
    failures=$((failures + 1))
  else
    echo "[smoke][ok] ${NAMES[$i]} completed; log=${LOGS[$i]}"
  fi
done

if (( failures > 0 )); then
  echo "[smoke][error] ${failures} Atari smoke task(s) failed. Master log: ${MASTER_LOG}"
  exit 1
fi

echo "[smoke] All Atari smoke tasks completed successfully. Master log: ${MASTER_LOG}"
