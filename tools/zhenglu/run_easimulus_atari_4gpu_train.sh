#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-60}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/easimulus_atari_4gpu_${TIMESTAMP}"
MASTER_LOG="${LOG_DIR}/train_easimulus_atari_4gpu_seed${SEED}_${TIMESTAMP}.log"

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
export WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

declare -a PIDS=()
declare -a NAMES=()
declare -a LOGS=()
declare -a STATUS=()
HEARTBEAT_PID=""

print_child_status() {
  local i
  for i in "${!PIDS[@]}"; do
    if [[ "${STATUS[$i]:-running}" == "running" ]] && kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "[train][status] running name=${NAMES[$i]} pid=${PIDS[$i]} log=${LOGS[$i]}"
    else
      echo "[train][status] done_or_missing name=${NAMES[$i]:-unknown} pid=${PIDS[$i]:-unknown} status=${STATUS[$i]:-unknown} log=${LOGS[$i]:-unknown}"
    fi
  done
}

heartbeat_loop() {
  while true; do
    echo "[train][heartbeat] $(date -Is) master_pid=$$ heartbeat_pid=${BASHPID}"
    print_child_status
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
    sleep "${HEARTBEAT_INTERVAL}"
  done
}

handle_signal() {
  local sig="$1"
  local rc="$2"
  echo "[train][signal] received ${sig} at $(date -Is); forwarding TERM to child tasks."
  print_child_status
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill -TERM "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  echo "[train][signal] master log: ${MASTER_LOG}"
  echo "[train][signal] output root: ${OUTPUT_ROOT}"
  exit "${rc}"
}

trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[train][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

if ! [[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_EVERY < 1 )); then
  echo "[train][error] CHECKPOINT_EVERY must be a positive integer; got ${CHECKPOINT_EVERY}"
  exit 1
fi

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[train][error] conda is not available in PATH."
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
  echo "[runtime] WANDB_MODE=${WANDB_MODE}"
  echo "[runtime] SEED=${SEED}"
  echo "[runtime] HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL}"
  echo "[runtime] LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS}"
  echo "[runtime] AUTO_RESUME=${AUTO_RESUME}"
  echo "[runtime] CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  echo "[runtime] OUTPUT_ROOT=${OUTPUT_ROOT}"
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
    echo "[train][error] Need at least 4 visible CUDA devices for this 4-GPU training script; found ${count}."
    echo "[train][error] A 1-GPU environment setup node is normal, but this script must run inside a 4-GPU compute job."
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
    echo "[task] WANDB_MODE=${WANDB_MODE}"
    echo "[task] SEED=${SEED}"
    echo "[task] game=${game}"
    echo "[task] output=${task_output}"
    echo "[task] media_episode_count=${media_count}"
    echo "[task] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } >> "${log_file}" 2>&1
}

repair_interrupted_checkpoint() {
  local run_dir="$1"
  local tmp_dir="${run_dir}/checkpoints_tmp"
  local ckpt_dir="${run_dir}/checkpoints"
  if [[ ! -d "${tmp_dir}" ]]; then
    return 0
  fi

  echo "[train][resume] found interrupted checkpoint save: ${tmp_dir}" >&2
  mkdir -p "${ckpt_dir}"
  local item
  for item in last.pt best.pt run_metadata.pt optimizer.pt num_seen_episodes_test_dataset.pt; do
    if [[ -f "${tmp_dir}/${item}" ]]; then
      cp -f "${tmp_dir}/${item}" "${ckpt_dir}/${item}"
      echo "[train][resume] restored ${item} from checkpoints_tmp" >&2
    fi
  done
  mv "${tmp_dir}" "${run_dir}/checkpoints_tmp.restored_${TIMESTAMP}" 2>/dev/null || true
}

is_valid_resume_dir() {
  local run_dir="$1"
  repair_interrupted_checkpoint "${run_dir}"
  [[ -f "${run_dir}/checkpoints/run_metadata.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/optimizer.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/num_seen_episodes_test_dataset.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
  return 0
}

find_resume_run_dir() {
  local game="$1"
  local game_short="$2"
  if [[ "${AUTO_RESUME}" != "1" ]]; then
    return 1
  fi
  if [[ ! -d "${PROJECT_ROOT}/outputs" ]]; then
    return 1
  fi

  local line
  local metadata
  local run_dir
  while IFS= read -r line; do
    metadata="${line#* }"
    run_dir="$(dirname "$(dirname "${metadata}")")"
    if is_valid_resume_dir "${run_dir}"; then
      echo "${run_dir}"
      return 0
    fi
  done < <(
    find "${PROJECT_ROOT}/outputs" \
      -type f \
      -path "*/${game_short}_seed${SEED}/${game}/*/*-seed-${SEED}/checkpoints/run_metadata.pt" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr
  )
  return 1
}

start_task() {
  local gpu="$1"
  local game="$2"
  local game_short="${game%NoFrameskip-v4}"
  local log_file="${LOG_DIR}/train_easimulus_atari_${game_short}_seed${SEED}_${TIMESTAMP}.log"
  local task_output="${OUTPUT_ROOT}/${game_short}_seed${SEED}"
  local resume_run_dir=""
  if resume_run_dir="$(find_resume_run_dir "${game}" "${game_short}")"; then
    task_output="$(dirname "$(dirname "$(dirname "${resume_run_dir}")")")"
    log_file="${LOG_DIR}/train_easimulus_atari_${game_short}_seed${SEED}_${TIMESTAMP}_resume.log"
    echo "[train][resume] ${game_short}: ${resume_run_dir}"
  fi
  mkdir -p "${task_output}"
  local media_count="${MEDIA_EPISODES_TO_SAVE:-0}"

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "common.seed=${SEED}"
    world_model.event_pred=True
    world_model.ges=True
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${game_short}-seed${SEED}"
    "wandb.group=easimulus_atari_4gpu_${TIMESTAMP}"
    "outputs_dir_path=${task_output}"
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "collection.train.num_episodes_to_save=${media_count}"
    "collection.test.num_episodes_to_save=${media_count}"
    evaluation.tokenizer.save_reconstructions=False
  )

  if [[ -n "${resume_run_dir}" ]]; then
    cmd+=(
      common.resume=True
      "hydra.run.dir=${resume_run_dir}"
      hydra.output_subdir=null
    )
  fi

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
  STATUS+=("running")
  echo "[train] started ${game} on GPU ${gpu}; log=${log_file}; output=${task_output}"
}

activate_conda
cd "${EASIMULUS_DIR}"
print_runtime_info
check_4gpu

start_task 0 BreakoutNoFrameskip-v4
echo "[train] sleeping ${LAUNCH_STAGGER_SECONDS}s before launching Boxing to reduce concurrent compile/init pressure"
sleep "${LAUNCH_STAGGER_SECONDS}"
start_task 1 BoxingNoFrameskip-v4
echo "[train] sleeping ${LAUNCH_STAGGER_SECONDS}s before launching Seaquest to reduce concurrent compile/init pressure"
sleep "${LAUNCH_STAGGER_SECONDS}"
start_task 2 SeaquestNoFrameskip-v4
echo "[train] sleeping ${LAUNCH_STAGGER_SECONDS}s before launching RoadRunner to reduce concurrent compile/init pressure"
sleep "${LAUNCH_STAGGER_SECONDS}"
start_task 3 RoadRunnerNoFrameskip-v4

heartbeat_loop &
HEARTBEAT_PID="$!"
echo "[train] heartbeat started pid=${HEARTBEAT_PID}; interval=${HEARTBEAT_INTERVAL}s"

failures=0
for i in "${!PIDS[@]}"; do
  set +e
  wait "${PIDS[$i]}"
  rc=$?
  set -e
  STATUS[$i]="${rc}"
  if (( rc != 0 )); then
    echo "[train][fail] ${NAMES[$i]} exited with ${rc}; log=${LOGS[$i]}"
    failures=$((failures + 1))
  else
    echo "[train][ok] ${NAMES[$i]} completed; log=${LOGS[$i]}"
  fi
done

if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
  kill -TERM "${HEARTBEAT_PID}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" 2>/dev/null || true
fi

if (( failures > 0 )); then
  echo "[train][error] ${failures} Atari training task(s) failed. Master log: ${MASTER_LOG}"
  echo "[train][error] Output root: ${OUTPUT_ROOT}"
  exit 1
fi

echo "[train] All Atari training tasks completed successfully."
echo "[train] Master log: ${MASTER_LOG}"
echo "[train] Output root: ${OUTPUT_ROOT}"
