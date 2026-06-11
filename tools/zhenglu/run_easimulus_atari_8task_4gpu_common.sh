#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${TASKS:-}" ]]; then
  echo "[train][error] TASKS must contain exactly 8 Atari game names, for example: TASKS=\"Alien Amidar ...\""
  exit 1
fi

read -r -a TASK_ARRAY <<< "${TASKS}"
if (( ${#TASK_ARRAY[@]} != 8 )); then
  echo "[train][error] Expected exactly 8 tasks; got ${#TASK_ARRAY[@]}: ${TASKS}"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
SERVER_NAME="${SERVER_NAME:-atari_8task}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-60}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
VIDEO_SECONDS="${VIDEO_SECONDS:-600}"
VIDEO_FPS="${VIDEO_FPS:-15}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"
RETRY_MEDIA_SAVE_ONE="${RETRY_MEDIA_SAVE_ONE:-1}"
RESUME_OUTPUT_PREFIX="${RESUME_OUTPUT_PREFIX:-easimulus_atari_${SERVER_NAME}_}"

EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TOOLS_DIR="${PROJECT_ROOT}/tools/zhenglu"
MONITOR_SCRIPT="${TOOLS_DIR}/monitor_easimulus_metrics.py"
RECORD_SCRIPT="${TOOLS_DIR}/record_easimulus_atari_demo.py"
TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${PROJECT_ROOT}/logs"
MASTER_LOG="${LOG_ROOT}/train_easimulus_atari_${SERVER_NAME}_seed${SEED}_${TIMESTAMP}.log"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/easimulus_atari_${SERVER_NAME}_${TIMESTAMP}"
VIDEO_ROOT="${PROJECT_ROOT}/videos/easimulus_atari_${SERVER_NAME}_${TIMESTAMP}"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${VIDEO_ROOT}" \
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

declare -a WORKER_PIDS=()
HEARTBEAT_PID=""
RUN_TRAINING_RUN_DIR=""

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
  echo "[runtime] SERVER_NAME=${SERVER_NAME}"
  echo "[runtime] TASKS=${TASKS}"
  echo "[runtime] AUTO_RESUME=${AUTO_RESUME}"
  echo "[runtime] RESUME_OUTPUT_PREFIX=${RESUME_OUTPUT_PREFIX}"
  echo "[runtime] CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  echo "[runtime] VIDEO_SECONDS=${VIDEO_SECONDS}"
  echo "[runtime] OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "[runtime] VIDEO_ROOT=${VIDEO_ROOT}"
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
  if [[ ! -f "${RECORD_SCRIPT}" ]]; then
    echo "[train][error] Record script not found: ${RECORD_SCRIPT}"
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

task_log_dir() {
  local game_short="$1"
  echo "${LOG_ROOT}/${game_short}"
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

find_latest_run_dir() {
  local game="$1"
  local game_short="$2"
  if [[ ! -d "${PROJECT_ROOT}/outputs" ]]; then
    return 1
  fi

  local line metadata run_dir search_root
  while IFS= read -r line; do
    metadata="${line#* }"
    run_dir="$(dirname "$(dirname "${metadata}")")"
    if is_valid_resume_dir "${run_dir}"; then
      echo "${run_dir}"
      return 0
    fi
  done < <(
    while IFS= read -r search_root; do
      find "${search_root}" \
        -type f \
        -path "*/${game_short}_seed${SEED}/${game}/*/*-seed-${SEED}/checkpoints/run_metadata.pt" \
        -printf '%T@ %p\n' 2>/dev/null
    done < <(find "${PROJECT_ROOT}/outputs" -maxdepth 1 -type d -name "${RESUME_OUTPUT_PREFIX}*" 2>/dev/null) | sort -nr
  )
  return 1
}

checkpoint_epoch() {
  local run_dir="$1"
  python - "${run_dir}/checkpoints/run_metadata.pt" <<'PY'
import sys, torch
path = sys.argv[1]
metadata = torch.load(path, map_location="cpu", weights_only=False)
print(int(metadata.get("epoch", 0)))
PY
}

run_training() {
  local gpu="$1"
  local game_short="$2"
  local game="${game_short}NoFrameskip-v4"
  RUN_TRAINING_RUN_DIR=""
  local log_dir
  log_dir="$(task_log_dir "${game_short}")"
  mkdir -p "${log_dir}"

  local log_file="${log_dir}/train_easimulus_atari_${game_short}_seed${SEED}_${TIMESTAMP}.log"
  local task_output="${OUTPUT_ROOT}/${game_short}_seed${SEED}"
  local resume_run_dir=""

  if [[ "${AUTO_RESUME}" == "1" ]] && resume_run_dir="$(find_latest_run_dir "${game}" "${game_short}")"; then
    task_output="$(dirname "$(dirname "$(dirname "${resume_run_dir}")")")"
    log_file="${log_dir}/train_easimulus_atari_${game_short}_seed${SEED}_${TIMESTAMP}_resume.log"
    echo "[train][resume] ${game_short}: ${resume_run_dir}"
    local resume_epoch
    resume_epoch="$(checkpoint_epoch "${resume_run_dir}")"
    if (( resume_epoch >= 600 )); then
      echo "[train][skip] ${game_short} already has checkpoint epoch ${resume_epoch}; skipping training."
      RUN_TRAINING_RUN_DIR="${resume_run_dir}"
      return 0
    fi
  fi

  mkdir -p "${task_output}"

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
    "wandb.group=easimulus_atari_${SERVER_NAME}_${TIMESTAMP}"
    "outputs_dir_path=${task_output}"
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "collection.train.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    "collection.test.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    evaluation.tokenizer.save_reconstructions=False
  )

  if [[ -n "${resume_run_dir}" ]]; then
    cmd+=(
      common.resume=True
      "hydra.run.dir=${resume_run_dir}"
      hydra.output_subdir=null
    )
  fi

  echo "[train][start] ${game_short} on GPU ${gpu}; log=${log_file}; output=${task_output}"
  set +e
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
    echo "[task] media_episode_count=${MEDIA_EPISODES_TO_SAVE}"
    echo "[task] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${game_short}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )) && [[ "${MEDIA_EPISODES_TO_SAVE}" == "0" ]] && [[ "${RETRY_MEDIA_SAVE_ONE}" == "1" ]]; then
    echo "[train][retry] ${game_short} failed with media save count 0; retrying once with count 1."
    cmd=("${cmd[@]/collection.train.num_episodes_to_save=0/collection.train.num_episodes_to_save=1}")
    cmd=("${cmd[@]/collection.test.num_episodes_to_save=0/collection.test.num_episodes_to_save=1}")
    set +e
    {
      echo "[task] retry date: $(date -Is)"
      echo "[task] retry command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${game_short}"
    rc=${PIPESTATUS[0]}
    set -e
  fi

  if (( rc != 0 )); then
    echo "[train][fail] ${game_short} exited with ${rc}; log=${log_file}"
    return "${rc}"
  fi

  local final_run_dir
  if ! final_run_dir="$(find_latest_run_dir "${game}" "${game_short}")"; then
    echo "[train][fail] ${game_short} completed but no valid checkpoint run dir was found under ${PROJECT_ROOT}/outputs/${RESUME_OUTPUT_PREFIX}*"
    return 1
  fi
  echo "[train][ok] ${game_short} completed; run_dir=${final_run_dir}; log=${log_file}"
  RUN_TRAINING_RUN_DIR="${final_run_dir}"
}

record_video() {
  local gpu="$1"
  local game_short="$2"
  local run_dir="$3"
  local game="${game_short}NoFrameskip-v4"
  local ckpt="${run_dir}/checkpoints/last.pt"
  local log_dir
  log_dir="$(task_log_dir "${game_short}")"
  mkdir -p "${log_dir}" "${VIDEO_ROOT}"

  local epoch
  epoch="$(checkpoint_epoch "${run_dir}")"
  local video="${VIDEO_ROOT}/${game_short}_seed${SEED}_epoch${epoch}_final_10min.mp4"
  local log_file="${log_dir}/record_easimulus_atari_${game_short}_seed${SEED}_${TIMESTAMP}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[record][error] Missing final checkpoint for ${game_short}: ${ckpt}"
    return 1
  fi

  echo "[record][start] ${game_short} on GPU ${gpu}; checkpoint=${ckpt}; video=${video}"
  {
    echo "[record] hostname: $(hostname)"
    echo "[record] date: $(date -Is)"
    echo "[record] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[record] task=${game_short}"
    echo "[record] env_id=${game}"
    echo "[record] run_dir=${run_dir}"
    echo "[record] checkpoint=${ckpt}"
    echo "[record] video=${video}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" \
      python "${RECORD_SCRIPT}" \
        --easimulus-dir "${EASIMULUS_DIR}" \
        --env-id "${game}" \
        --checkpoint "${ckpt}" \
        --output "${video}" \
        --seconds "${VIDEO_SECONDS}" \
        --fps "${VIDEO_FPS}" \
        --seed "${SEED}" \
        --min-width 640 \
        --overlay-line "${game_short} | final 100k policy" \
        --overlay-line "seed ${SEED}, checkpoint epoch ${epoch}" \
        --overlay-line "experiment ${SERVER_NAME}_${TIMESTAMP}"
  } >> "${log_file}" 2>&1
  echo "[record][ok] ${game_short}; video=${video}; log=${log_file}"
}

run_worker() {
  local gpu="$1"
  shift
  local task
  for task in "$@"; do
    echo "[worker][gpu${gpu}] next task=${task}"
    if ! run_training "${gpu}" "${task}"; then
      echo "[worker][gpu${gpu}][fail] training failed for ${task}"
      return 1
    fi
    local run_dir="${RUN_TRAINING_RUN_DIR}"
    if [[ -z "${run_dir}" ]]; then
      echo "[worker][gpu${gpu}][fail] could not resolve run dir for ${task}"
      return 1
    fi
    if ! record_video "${gpu}" "${task}" "${run_dir}"; then
      echo "[worker][gpu${gpu}][fail] recording failed for ${task}"
      return 1
    fi
    echo "[worker][gpu${gpu}][ok] ${task}"
  done
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
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  echo "[train][signal] master log: ${MASTER_LOG}"
  echo "[train][signal] output root: ${OUTPUT_ROOT}"
  echo "[train][signal] video root: ${VIDEO_ROOT}"
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

for gpu in 0 1 2 3; do
  first_index="${gpu}"
  second_index="$((gpu + 4))"
  (
    if (( gpu > 0 )); then
      sleep "$((gpu * LAUNCH_STAGGER_SECONDS))"
    fi
    run_worker "${gpu}" "${TASK_ARRAY[$first_index]}" "${TASK_ARRAY[$second_index]}"
  ) &
  WORKER_PIDS+=("$!")
  worker_pid_index="$((${#WORKER_PIDS[@]} - 1))"
  echo "[train] started worker gpu=${gpu}; tasks=${TASK_ARRAY[$first_index]},${TASK_ARRAY[$second_index]}; pid=${WORKER_PIDS[$worker_pid_index]}"
done

failures=0
for pid in "${WORKER_PIDS[@]}"; do
  set +e
  wait "${pid}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[train][worker_fail] pid=${pid} rc=${rc}"
    failures=$((failures + 1))
  else
    echo "[train][worker_ok] pid=${pid}"
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
  echo "[train][error] Video root: ${VIDEO_ROOT}"
  exit 1
fi

echo "[train] All 8 Atari tasks completed successfully for ${SERVER_NAME}."
echo "[train] Master log: ${MASTER_LOG}"
echo "[train] Output root: ${OUTPUT_ROOT}"
echo "[train] Video root: ${VIDEO_ROOT}"
