#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"

EXP_NAME="${EXP_NAME:-dapr_craftax_noop_check}"
TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SEEDS:-0 1 2 3}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
WANDB_MODE="${WANDB_MODE:-offline}"
ENV_NAME="${ENV_NAME:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"

LOG_ROOT="${PROJECT_ROOT}/logs/decision_aware_precision_router/craftax/${EXP_NAME}_${TIMESTAMP}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/decision_aware_precision_router/craftax/${EXP_NAME}"
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
      echo "[dapr-craftax][error] ENV_NAME=${ENV_NAME}, but conda is not available."
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
    echo "[dapr-craftax][repair] moving incomplete run dir ${run_dir} -> ${broken}"
    mv "${run_dir}" "${broken}"
  fi
}

run_one() {
  local gpu="$1"
  local seed="$2"
  local run_dir="${OUTPUT_ROOT}/Craftax_seed${seed}"
  local log_file="${LOG_ROOT}/seed${seed}.log"
  prepare_run_dir "${run_dir}"

  local cmd=(
    python src/main.py
    benchmark=craftax
    "common.seed=${seed}"
    actor_critic.intrinsic_reward_weight=1
    mechanism.decision_aware_precision_router.enabled=True
    "wandb.mode=${WANDB_MODE}"
    "wandb.name=dapr-craftax-seed${seed}"
    "wandb.group=${EXP_NAME}_${TIMESTAMP}"
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    evaluation.tokenizer.save_reconstructions=False
    "hydra.run.dir=${run_dir}"
  )

  if is_valid_checkpoint "${run_dir}"; then
    local epoch
    epoch="$(checkpoint_epoch "${run_dir}")"
    if (( epoch >= 10000 )); then
      echo "[dapr-craftax][skip] seed ${seed} already reached epoch ${epoch}: ${run_dir}"
      return 0
    fi
    cmd+=(common.resume=True hydra.output_subdir=null)
    echo "[dapr-craftax][resume] seed ${seed}: ${run_dir}"
  fi

  {
    echo "[dapr-craftax][start] date=$(date -Is) gpu=${gpu} seed=${seed}"
    echo "[dapr-craftax][note] Craftax uses vector observations in this repo, so DAPR is expected to be a no-op sanity path."
    echo "[dapr-craftax][cmd] CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    cd "${EASIMULUS_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    echo "[dapr-craftax][done] date=$(date -Is) seed=${seed}"
  } 2>&1 | tee -a "${log_file}"
}

main() {
  activate_env
  read -r -a SEED_ARRAY <<< "${SEEDS}"
  read -r -a GPU_ARRAY <<< "${GPU_IDS}"
  declare -a PIDS=()
  local idx gpu seed
  for idx in "${!SEED_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$((idx % ${#GPU_ARRAY[@]}))]}"
    seed="${SEED_ARRAY[$idx]}"
    run_one "${gpu}" "${seed}" &
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
    echo "[dapr-craftax][error] ${failures} worker(s) failed. See ${LOG_ROOT}"
    exit 1
  fi
  echo "[dapr-craftax] all jobs completed."
}

main "$@"
