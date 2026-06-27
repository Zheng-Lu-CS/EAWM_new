#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"

EXP_NAME="${EXP_NAME:-dapr_atari_validation}"
TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-Breakout Seaquest Frostbite Kangaroo RoadRunner PrivateEye}"
SEEDS="${SEEDS:-0 1 2}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
WANDB_MODE="${WANDB_MODE:-offline}"
ENV_NAME="${ENV_NAME:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
ROUTER="${ROUTER:-gumbel}"
SUMMARY="${SUMMARY:-local_mean}"
SUMMARY_KERNEL="${SUMMARY_KERNEL:-2}"
BUDGET_LOSS_WEIGHT="${BUDGET_LOSS_WEIGHT:-0.01}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-25}"
FEATURE_DROPOUT_PROB="${FEATURE_DROPOUT_PROB:-0.5}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"

LOG_ROOT="${PROJECT_ROOT}/logs/decision_aware_precision_router/atari/${EXP_NAME}_${TIMESTAMP}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/decision_aware_precision_router/atari/${EXP_NAME}"
mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"

activate_env() {
  if [[ -n "${ENV_NAME}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      echo "[dapr-atari][error] ENV_NAME=${ENV_NAME}, but conda is not available."
      exit 1
    fi
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
  fi
}

is_valid_checkpoint() {
  local run_dir="$1"
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/optimizer.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/run_metadata.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
}

checkpoint_epoch() {
  local run_dir="$1"
  python - "${run_dir}/checkpoints/run_metadata.pt" <<'PY'
import sys, torch
metadata = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(metadata.get("epoch", 0)))
PY
}

prepare_run_dir() {
  local run_dir="$1"
  if [[ -d "${run_dir}" ]] && ! is_valid_checkpoint "${run_dir}"; then
    local broken="${run_dir}.broken_${TIMESTAMP}"
    echo "[dapr-atari][repair] moving incomplete run dir ${run_dir} -> ${broken}"
    mv "${run_dir}" "${broken}"
  fi
}

run_one() {
  local gpu="$1"
  local task="$2"
  local seed="$3"
  local env_id="${task}NoFrameskip-v4"
  local run_dir="${OUTPUT_ROOT}/${task}_seed${seed}"
  local log_dir="${LOG_ROOT}/${task}"
  local log_file="${log_dir}/seed${seed}.log"
  mkdir -p "${log_dir}"
  prepare_run_dir "${run_dir}"

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${env_id}"
    "common.seed=${seed}"
    world_model.event_pred=True
    world_model.ges=True
    mechanism.decision_aware_precision_router.enabled=True
    "mechanism.decision_aware_precision_router.target_keep_ratio=${KEEP_RATIO}"
    "mechanism.decision_aware_precision_router.router=${ROUTER}"
    "mechanism.decision_aware_precision_router.summary=${SUMMARY}"
    "mechanism.decision_aware_precision_router.summary_kernel=${SUMMARY_KERNEL}"
    "mechanism.decision_aware_precision_router.budget_loss_weight=${BUDGET_LOSS_WEIGHT}"
    "mechanism.decision_aware_precision_router.warmup_epochs=${WARMUP_EPOCHS}"
    "mechanism.decision_aware_precision_router.feature_dropout_prob=${FEATURE_DROPOUT_PROB}"
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=dapr-${task}-seed${seed}"
    "wandb.group=${EXP_NAME}_${TIMESTAMP}"
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "collection.train.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    "collection.test.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    evaluation.tokenizer.save_reconstructions=False
    "hydra.run.dir=${run_dir}"
  )

  if is_valid_checkpoint "${run_dir}"; then
    local epoch
    epoch="$(checkpoint_epoch "${run_dir}")"
    if (( epoch >= 600 )); then
      echo "[dapr-atari][skip] ${task} seed ${seed} already reached epoch ${epoch}: ${run_dir}"
      return 0
    fi
    cmd+=(common.resume=True hydra.output_subdir=null)
    echo "[dapr-atari][resume] ${task} seed ${seed}: ${run_dir}"
  fi

  {
    echo "[dapr-atari][start] date=$(date -Is) gpu=${gpu} task=${task} seed=${seed}"
    echo "[dapr-atari][cmd] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    cd "${EASIMULUS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    echo "[dapr-atari][done] date=$(date -Is) task=${task} seed=${seed}"
  } 2>&1 | tee -a "${log_file}"
}

main() {
  activate_env
  if [[ ! -d "${EASIMULUS_DIR}" ]]; then
    echo "[dapr-atari][error] EASimulus dir not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  read -r -a TASK_ARRAY <<< "${TASKS}"
  read -r -a SEED_ARRAY <<< "${SEEDS}"
  read -r -a GPU_ARRAY <<< "${GPU_IDS}"
  if (( ${#GPU_ARRAY[@]} == 0 )); then
    echo "[dapr-atari][error] GPU_IDS is empty."
    exit 1
  fi

  declare -a JOBS=()
  local task seed
  for seed in "${SEED_ARRAY[@]}"; do
    for task in "${TASK_ARRAY[@]}"; do
      JOBS+=("${task}:${seed}")
    done
  done

  echo "[dapr-atari] jobs=${#JOBS[@]} tasks=${TASKS} seeds=${SEEDS} gpus=${GPU_IDS}"
  echo "[dapr-atari] logs=${LOG_ROOT}"
  echo "[dapr-atari] outputs=${OUTPUT_ROOT}"

  declare -a PIDS=()
  local gpu_index gpu
  for gpu_index in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$gpu_index]}"
    (
      local i pair t s
      for (( i=gpu_index; i<${#JOBS[@]}; i+=${#GPU_ARRAY[@]} )); do
        pair="${JOBS[$i]}"
        t="${pair%%:*}"
        s="${pair##*:}"
        run_one "${gpu}" "${t}" "${s}"
      done
    ) &
    PIDS+=("$!")
  done

  local failures=0 pid rc
  for pid in "${PIDS[@]}"; do
    set +e
    wait "${pid}"
    rc=$?
    set -e
    if (( rc != 0 )); then
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "[dapr-atari][error] ${failures} worker(s) failed. See ${LOG_ROOT}"
    exit 1
  fi
  echo "[dapr-atari] all jobs completed."
}

main "$@"
