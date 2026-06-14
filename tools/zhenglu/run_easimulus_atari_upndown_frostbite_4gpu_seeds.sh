#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
WANDB_MODE="${WANDB_MODE:-offline}"
SERVER_NAME="${SERVER_NAME:-upndown_frostbite_4gpu_seeds}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-60}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"
RETRY_MEDIA_SAVE_ONE="${RETRY_MEDIA_SAVE_ONE:-1}"
EXP_TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TOOLS_DIR="${PROJECT_ROOT}/tools/zhenglu"
MONITOR_SCRIPT="${TOOLS_DIR}/monitor_easimulus_metrics.py"
LOG_ROOT="${PROJECT_ROOT}/logs"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/easimulus_atari_${SERVER_NAME}_${EXP_TIMESTAMP}"
MASTER_LOG="${LOG_ROOT}/train_easimulus_atari_${SERVER_NAME}_${EXP_TIMESTAMP}.log"

TASKS=(UpNDown UpNDown Frostbite Frostbite)
SEEDS=(24 48 24 48)
GPUS=(0 1 2 3)

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" \
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
HEARTBEAT_PID=""

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
  echo "[runtime] SERVER_NAME=${SERVER_NAME}"
  echo "[runtime] AUTO_RESUME=${AUTO_RESUME}"
  echo "[runtime] CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  echo "[runtime] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[runtime] MASTER_LOG=${MASTER_LOG}"
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
    echo "[train][error] EASimulus directory not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
    echo "[train][error] Monitor script not found: ${MONITOR_SCRIPT}"
    exit 1
  fi
  if ! [[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_EVERY < 1 )); then
    echo "[train][error] CHECKPOINT_EVERY must be a positive integer; got ${CHECKPOINT_EVERY}"
    exit 1
  fi
  local count
  count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( count < 4 )); then
    echo "[train][error] Need at least 4 visible CUDA devices; found ${count}."
    exit 1
  fi
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
  mv "${tmp_dir}" "${run_dir}/checkpoints_tmp.restored_${EXP_TIMESTAMP}" 2>/dev/null || true
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
  local seed="$3"
  if [[ "${AUTO_RESUME}" != "1" ]] || [[ ! -d "${PROJECT_ROOT}/outputs" ]]; then
    return 1
  fi

  local line metadata run_dir
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
      -path "*/${game_short}_seed${seed}/${game}/*/*-seed-${seed}/checkpoints/run_metadata.pt" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr
  )
  return 1
}

checkpoint_epoch() {
  local run_dir="$1"
  python - "${run_dir}/checkpoints/run_metadata.pt" <<'PY'
import sys, torch
metadata = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(metadata.get("epoch", 0)))
PY
}

run_one() {
  local gpu="$1"
  local game_short="$2"
  local seed="$3"
  local game="${game_short}NoFrameskip-v4"
  local log_dir="${LOG_ROOT}/${game_short}"
  mkdir -p "${log_dir}"

  local log_file="${log_dir}/train_easimulus_atari_${game_short}_seed${seed}_${EXP_TIMESTAMP}.log"
  local task_output="${OUTPUT_ROOT}/${game_short}_seed${seed}"
  local resume_run_dir=""

  if resume_run_dir="$(find_resume_run_dir "${game}" "${game_short}" "${seed}")"; then
    task_output="$(dirname "$(dirname "$(dirname "${resume_run_dir}")")")"
    log_file="${log_dir}/train_easimulus_atari_${game_short}_seed${seed}_${EXP_TIMESTAMP}_resume.log"
    echo "[train][resume] ${game_short} seed=${seed}: ${resume_run_dir}"
    local resume_epoch
    resume_epoch="$(checkpoint_epoch "${resume_run_dir}")"
    if (( resume_epoch >= 600 )); then
      echo "[train][skip] ${game_short} seed=${seed} already has checkpoint epoch ${resume_epoch}; skipping."
      return 0
    fi
  fi

  mkdir -p "${task_output}"
  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "common.seed=${seed}"
    world_model.event_pred=True
    world_model.ges=True
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${game_short}-seed${seed}"
    "wandb.group=easimulus_atari_${SERVER_NAME}_${EXP_TIMESTAMP}"
    "outputs_dir_path=${task_output}"
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "collection.train.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    "collection.test.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    evaluation.tokenizer.save_reconstructions=False
  )

  if [[ -n "${resume_run_dir}" ]]; then
    cmd+=(common.resume=True "hydra.run.dir=${resume_run_dir}" hydra.output_subdir=null)
  fi

  echo "[train][start] ${game_short} seed=${seed} gpu=${gpu}; log=${log_file}; output=${task_output}"
  set +e
  {
    echo "[task] hostname: $(hostname)"
    echo "[task] date: $(date -Is)"
    echo "[task] pwd: $(pwd)"
    echo "[task] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
    echo "[task] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[task] WANDB_MODE=${WANDB_MODE}"
    echo "[task] game=${game}"
    echo "[task] seed=${seed}"
    echo "[task] output=${task_output}"
    echo "[task] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${game_short}_seed${seed}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )) && [[ "${MEDIA_EPISODES_TO_SAVE}" == "0" ]] && [[ "${RETRY_MEDIA_SAVE_ONE}" == "1" ]]; then
    echo "[train][retry] ${game_short} seed=${seed} failed with media save count 0; retrying once with count 1."
    cmd=("${cmd[@]/collection.train.num_episodes_to_save=0/collection.train.num_episodes_to_save=1}")
    cmd=("${cmd[@]/collection.test.num_episodes_to_save=0/collection.test.num_episodes_to_save=1}")
    set +e
    {
      echo "[task] retry date: $(date -Is)"
      echo "[task] retry command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${game_short}_seed${seed}"
    rc=${PIPESTATUS[0]}
    set -e
  fi

  if (( rc != 0 )); then
    echo "[train][fail] ${game_short} seed=${seed} exited with ${rc}; log=${log_file}"
    return "${rc}"
  fi

  echo "[train][ok] ${game_short} seed=${seed}; log=${log_file}"
}

heartbeat_loop() {
  while true; do
    echo "[train][heartbeat] $(date -Is) master_pid=$$ heartbeat_pid=${BASHPID}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    fi
    sleep "${HEARTBEAT_INTERVAL}"
  done
}

handle_signal() {
  local sig="$1"
  local rc="$2"
  echo "[train][signal] received ${sig} at $(date -Is); forwarding TERM to workers."
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

activate_conda
cd "${EASIMULUS_DIR}"
check_inputs
print_runtime_info

heartbeat_loop &
HEARTBEAT_PID="$!"
echo "[train] heartbeat started pid=${HEARTBEAT_PID}; interval=${HEARTBEAT_INTERVAL}s"

for i in "${!TASKS[@]}"; do
  (
    if (( i > 0 )); then
      sleep "$((i * LAUNCH_STAGGER_SECONDS))"
    fi
    run_one "${GPUS[$i]}" "${TASKS[$i]}" "${SEEDS[$i]}"
  ) &
  PIDS+=("$!")
  NAMES+=("${TASKS[$i]}_seed${SEEDS[$i]}")
  LOGS+=("${LOG_ROOT}/${TASKS[$i]}/train_easimulus_atari_${TASKS[$i]}_seed${SEEDS[$i]}_${EXP_TIMESTAMP}.log")
  echo "[train] launched ${NAMES[$i]} on gpu=${GPUS[$i]} pid=${PIDS[$i]}"
done

failures=0
for i in "${!PIDS[@]}"; do
  set +e
  wait "${PIDS[$i]}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[train][worker_fail] ${NAMES[$i]} pid=${PIDS[$i]} rc=${rc} log=${LOGS[$i]}"
    failures=$((failures + 1))
  else
    echo "[train][worker_ok] ${NAMES[$i]} pid=${PIDS[$i]}"
  fi
done

if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
  kill -TERM "${HEARTBEAT_PID}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" 2>/dev/null || true
fi

if (( failures > 0 )); then
  echo "[train][error] ${failures} worker(s) failed."
  echo "[train][error] Master log: ${MASTER_LOG}"
  echo "[train][error] Output root: ${OUTPUT_ROOT}"
  exit 1
fi

echo "[train] All 4 runs completed successfully."
echo "[train] Master log: ${MASTER_LOG}"
echo "[train] Output root: ${OUTPUT_ROOT}"
