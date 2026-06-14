#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${TASK:-}" ]]; then
  echo "[train][error] TASK must be set, e.g. TASK=Assault"
  exit 1
fi

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
SERVER_NAME="${SERVER_NAME:-${TASK,,}_dense_ablation_2gpu}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-60}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-60}"
AUTO_RESUME="${AUTO_RESUME:-1}"
ARCH_TAG="${ARCH_TAG:-easimulus_dense_v1}"
ALLOW_UNTAGGED_RESUME="${ALLOW_UNTAGGED_RESUME:-0}"
ARCH_MARKER_FILE="${ARCH_MARKER_FILE:-architecture.txt}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"
RETRY_MEDIA_SAVE_ONE="${RETRY_MEDIA_SAVE_ONE:-1}"
EXP_TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TOOLS_DIR="${PROJECT_ROOT}/tools/zhenglu"
MONITOR_SCRIPT="${TOOLS_DIR}/monitor_easimulus_metrics.py"
LOG_ROOT="${PROJECT_ROOT}/logs"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/easimulus_atari_${SERVER_NAME}_${EXP_TIMESTAMP}"
MASTER_LOG="${LOG_ROOT}/train_easimulus_atari_${SERVER_NAME}_seed${SEED}_${EXP_TIMESTAMP}.log"

VARIANTS=(dense eadense)
GPUS=(0 1)

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
export EASIMULUS_ARCH_TAG="${ARCH_TAG}"
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

check_inputs() {
  if [[ ! -d "${EASIMULUS_DIR}" ]]; then
    echo "[train][error] EASimulus directory not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
    echo "[train][error] Monitor script not found: ${MONITOR_SCRIPT}"
    exit 1
  fi
  local count
  count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( count < 2 )); then
    echo "[train][error] Need at least 2 visible CUDA devices; found ${count}."
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
  for item in last.pt best.pt run_metadata.pt optimizer.pt num_seen_episodes_test_dataset.pt "${ARCH_MARKER_FILE}"; do
    if [[ -f "${tmp_dir}/${item}" ]]; then
      cp -f "${tmp_dir}/${item}" "${ckpt_dir}/${item}"
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
  local arch_file="${run_dir}/checkpoints/${ARCH_MARKER_FILE}"
  if [[ -f "${arch_file}" ]]; then
    local saved_arch
    saved_arch="$(tr -d '[:space:]' < "${arch_file}")"
    if [[ "${saved_arch}" != "${ARCH_TAG}" ]]; then
      echo "[train][resume_skip] ${run_dir}: architecture tag '${saved_arch}' != '${ARCH_TAG}'." >&2
      return 1
    fi
  elif [[ "${ALLOW_UNTAGGED_RESUME}" != "1" ]]; then
    echo "[train][resume_skip] ${run_dir}: missing ${ARCH_MARKER_FILE}; set ALLOW_UNTAGGED_RESUME=1 to override." >&2
    return 1
  fi
  return 0
}

find_resume_run_dir() {
  local game="$1"
  local output_name="$2"
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
      -path "*/${output_name}/${game}/*/*-seed-${SEED}/checkpoints/run_metadata.pt" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr
  )
  return 1
}

checkpoint_epoch() {
  python - "$1/checkpoints/run_metadata.pt" <<'PY'
import sys, torch
metadata = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(metadata.get("epoch", 0)))
PY
}

run_variant() {
  local gpu="$1"
  local variant="$2"
  local game="${TASK}NoFrameskip-v4"
  local use_weighting="False"
  if [[ "${variant}" == "eadense" ]]; then
    use_weighting="True"
  fi

  local log_dir="${LOG_ROOT}/${TASK}"
  local output_name="${TASK}_${variant}_seed${SEED}"
  local task_output="${OUTPUT_ROOT}/${output_name}"
  local log_file="${log_dir}/train_easimulus_atari_${TASK}_${variant}_seed${SEED}_${EXP_TIMESTAMP}.log"
  local resume_run_dir=""
  mkdir -p "${log_dir}" "${task_output}"

  if resume_run_dir="$(find_resume_run_dir "${game}" "${output_name}")"; then
    task_output="$(dirname "$(dirname "$(dirname "${resume_run_dir}")")")"
    log_file="${log_dir}/train_easimulus_atari_${TASK}_${variant}_seed${SEED}_${EXP_TIMESTAMP}_resume.log"
    echo "[train][resume] ${TASK} ${variant}: ${resume_run_dir}"
    local epoch
    epoch="$(checkpoint_epoch "${resume_run_dir}")"
    if (( epoch >= 600 )); then
      echo "[train][skip] ${TASK} ${variant} already reached epoch ${epoch}."
      return 0
    fi
  fi

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "common.seed=${SEED}"
    world_model.event_pred=True
    world_model.ges=True
    world_model.use_dense_ssl=True
    "world_model.use_unchanged_patch_weighting=${use_weighting}"
    world_model.unchanged_patch_loss_weight=0.2
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=${TASK}-${variant}-seed${SEED}"
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

  echo "[train][start] ${TASK} ${variant} gpu=${gpu}; log=${log_file}; output=${task_output}"
  set +e
  {
    echo "[task] date: $(date -Is)"
    echo "[task] CUDA_VISIBLE_DEVICES=${gpu}"
    echo "[task] game=${game}"
    echo "[task] variant=${variant}"
    echo "[task] seed=${SEED}"
    echo "[task] command: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${TASK}_${variant}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )) && [[ "${MEDIA_EPISODES_TO_SAVE}" == "0" ]] && [[ "${RETRY_MEDIA_SAVE_ONE}" == "1" ]]; then
    cmd=("${cmd[@]/collection.train.num_episodes_to_save=0/collection.train.num_episodes_to_save=1}")
    cmd=("${cmd[@]/collection.test.num_episodes_to_save=0/collection.test.num_episodes_to_save=1}")
    set +e
    {
      echo "[task] retry date: $(date -Is)"
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    } 2>&1 | tee -a "${log_file}" | python "${MONITOR_SCRIPT}" --task "${TASK}_${variant}"
    rc=${PIPESTATUS[0]}
    set -e
  fi
  return "${rc}"
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

cleanup() {
  local rc="$1"
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill -TERM "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  exit "${rc}"
}
trap 'cleanup 143' TERM
trap 'cleanup 129' HUP
trap 'cleanup 130' INT

activate_conda
cd "${EASIMULUS_DIR}"
check_inputs
echo "[runtime] TASK=${TASK} SEED=${SEED} ARCH_TAG=${ARCH_TAG} AUTO_RESUME=${AUTO_RESUME} ALLOW_UNTAGGED_RESUME=${ALLOW_UNTAGGED_RESUME} OUTPUT_ROOT=${OUTPUT_ROOT} MASTER_LOG=${MASTER_LOG}"

heartbeat_loop &
HEARTBEAT_PID="$!"

for i in "${!VARIANTS[@]}"; do
  (
    if (( i > 0 )); then
      sleep "$((i * LAUNCH_STAGGER_SECONDS))"
    fi
    run_variant "${GPUS[$i]}" "${VARIANTS[$i]}"
  ) &
  PIDS+=("$!")
  NAMES+=("${TASK}_${VARIANTS[$i]}")
  LOGS+=("${LOG_ROOT}/${TASK}/train_easimulus_atari_${TASK}_${VARIANTS[$i]}_seed${SEED}_${EXP_TIMESTAMP}.log")
done

failures=0
for i in "${!PIDS[@]}"; do
  set +e
  wait "${PIDS[$i]}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[train][fail] ${NAMES[$i]} rc=${rc} log=${LOGS[$i]}"
    failures=$((failures + 1))
  else
    echo "[train][ok] ${NAMES[$i]}"
  fi
done

if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
  kill -TERM "${HEARTBEAT_PID}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" 2>/dev/null || true
fi

if (( failures > 0 )); then
  echo "[train][error] ${failures} run(s) failed. Master log: ${MASTER_LOG}"
  exit 1
fi

echo "[train] ${TASK} dense/eadense completed. Master log: ${MASTER_LOG}"
