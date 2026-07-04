#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
SEED="${SEED:-0}"
SERVER_SET="${SERVER_SET:-A}"
WANDB_MODE="${WANDB_MODE:-offline}"
TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-10}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
EVALUATION_EVERY="${EVALUATION_EVERY:-10}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"
VARIANTS="${VARIANTS:-dreamer_fixedwm cf_median_k8_u1}"
SOURCE_OUTPUT_PREFIX="${SOURCE_OUTPUT_PREFIX:-easimulus_atari_}"

TASKS_A_DEFAULT="Alien Amidar Assault Asterix BankHeist BattleZone Breakout ChopperCommand CrazyClimber DemonAttack Freeway Frostbite"
TASKS_B_DEFAULT="Gopher Jamesbond Kangaroo Krull KungFuMaster MsPacman Pong PrivateEye Qbert RoadRunner UpNDown"

if [[ -z "${TASKS:-}" ]]; then
  if [[ "${SERVER_SET}" == "A" ]]; then
    TASKS="${TASKS_A_DEFAULT}"
  elif [[ "${SERVER_SET}" == "B" ]]; then
    TASKS="${TASKS_B_DEFAULT}"
  else
    echo "[actor-launch][error] SERVER_SET must be A or B when TASKS is not provided; got ${SERVER_SET}" >&2
    exit 1
  fi
fi

EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TOOLS_DIR="${PROJECT_ROOT}/tools/zhenglu"
MONITOR_SCRIPT="${TOOLS_DIR}/monitor_easimulus_metrics.py"
LOG_ROOT="${PROJECT_ROOT}/logs/actor"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/fixedwm_actor_atari_${SERVER_SET}_seed${SEED}"
MASTER_LOG="${LOG_ROOT}/launcher_${SERVER_SET}_seed${SEED}.log"

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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

declare -a TASK_ARRAY=()
declare -a VARIANT_ARRAY=()
declare -a RUN_UNITS=()
declare -a WORKER_PIDS=()

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[actor-launch][error] conda is not available in PATH."
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
    echo "[actor-launch][error] Missing EASimulus dir: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -f "${MONITOR_SCRIPT}" ]]; then
    echo "[actor-launch][error] Missing monitor script: ${MONITOR_SCRIPT}"
    exit 1
  fi
  if ! [[ "${TASKS_PER_GPU}" =~ ^[0-9]+$ ]] || (( TASKS_PER_GPU < 1 )); then
    echo "[actor-launch][error] TASKS_PER_GPU must be a positive integer; got ${TASKS_PER_GPU}"
    exit 1
  fi
  local gpu_count
  gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( gpu_count < 4 )); then
    echo "[actor-launch][error] Need at least 4 visible CUDA devices; found ${gpu_count}."
    exit 1
  fi
}

normalize_inputs() {
  local task variant
  for task in ${TASKS}; do
    if [[ "${task}" == "Hero" ]]; then
      echo "[actor-launch][skip] Hero is explicitly disabled for this experiment."
      continue
    fi
    TASK_ARRAY+=("${task}")
  done
  for variant in ${VARIANTS}; do
    case "${variant}" in
      dreamer_fixedwm|cf_median_k8_u1|cf_trim_k8_u1)
        VARIANT_ARRAY+=("${variant}")
        ;;
      *)
        echo "[actor-launch][error] Unsupported variant '${variant}'. Allowed: dreamer_fixedwm cf_median_k8_u1 cf_trim_k8_u1"
        exit 1
        ;;
    esac
  done
  if (( ${#VARIANT_ARRAY[@]} > 3 )); then
    echo "[actor-launch][error] At most three variants per task are allowed."
    exit 1
  fi
  for task in "${TASK_ARRAY[@]}"; do
    for variant in "${VARIANT_ARRAY[@]}"; do
      RUN_UNITS+=("${task}|${variant}")
    done
  done
}

repair_interrupted_checkpoint() {
  local run_dir="$1"
  local tmp_dir="${run_dir}/checkpoints_tmp"
  local prev_dir="${run_dir}/checkpoints_prev"
  local ckpt_dir="${run_dir}/checkpoints"
  local item

  if [[ -d "${tmp_dir}" ]]; then
    echo "[actor-launch][resume] repairing interrupted save from ${tmp_dir}"
    mkdir -p "${ckpt_dir}"
    for item in last.pt best.pt run_metadata.pt optimizer.pt num_seen_episodes_test_dataset.pt; do
      if [[ -f "${tmp_dir}/${item}" ]]; then
        cp -f "${tmp_dir}/${item}" "${ckpt_dir}/${item}"
      fi
    done
    mv "${tmp_dir}" "${run_dir}/checkpoints_tmp.restored_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
  fi

  if ! is_valid_actor_run_dir "${run_dir}" "no_repair" && [[ -d "${prev_dir}" ]]; then
    echo "[actor-launch][resume] falling back to previous checkpoint from ${prev_dir}"
    mkdir -p "${ckpt_dir}"
    for item in last.pt best.pt run_metadata.pt optimizer.pt num_seen_episodes_test_dataset.pt; do
      if [[ -f "${prev_dir}/${item}" ]]; then
        cp -f "${prev_dir}/${item}" "${ckpt_dir}/${item}"
      fi
    done
  fi
}

is_valid_actor_run_dir() {
  local run_dir="$1"
  local repair_mode="${2:-repair}"
  if [[ "${repair_mode}" != "no_repair" ]]; then
    repair_interrupted_checkpoint "${run_dir}"
  fi
  [[ -f "${run_dir}/checkpoints/run_metadata.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/optimizer.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/num_seen_episodes_test_dataset.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
  return 0
}

is_valid_source_run_dir() {
  local run_dir="$1"
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
  return 0
}

find_source_run_dir() {
  local game_short="$1"
  local game="${game_short}NoFrameskip-v4"
  local search_base="${FIXED_WM_SOURCE_ROOT:-${PROJECT_ROOT}/outputs}"
  local line metadata run_dir

  while IFS= read -r line; do
    metadata="${line#* }"
    run_dir="$(dirname "$(dirname "${metadata}")")"
    if [[ "${run_dir}" == *"/fixedwm_actor_atari_"* ]]; then
      continue
    fi
    if is_valid_source_run_dir "${run_dir}"; then
      echo "${run_dir}"
      return 0
    fi
  done < <(
    find "${search_base}" \
      -type f \
      -path "*/${game_short}_seed${SEED}/${game}/*/*-seed-${SEED}/checkpoints/run_metadata.pt" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr
  )

  while IFS= read -r line; do
    metadata="${line#* }"
    run_dir="$(dirname "$(dirname "${metadata}")")"
    if [[ "${run_dir}" == *"/fixedwm_actor_atari_"* ]]; then
      continue
    fi
    if [[ "${run_dir}" == *"/${SOURCE_OUTPUT_PREFIX}"* ]] && is_valid_source_run_dir "${run_dir}"; then
      echo "${run_dir}"
      return 0
    fi
  done < <(
    find "${PROJECT_ROOT}/outputs" \
      -type f \
      -path "*/${game_short}_seed${SEED}/${game}/*/*-seed-${SEED}/checkpoints/last.pt" \
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

variant_args() {
  local variant="$1"
  case "${variant}" in
    dreamer_fixedwm)
      echo "training.actor_critic.actor_loss_mode=dreamer training.actor_critic.batch_num_samples=${DREAMER_ACTOR_BATCH_NUM_SAMPLES:-128}"
      ;;
    cf_median_k8_u1)
      echo "training.actor_critic.actor_loss_mode=counterfactual_group training.actor_critic.counterfactual_branches=8 training.actor_critic.counterfactual_baseline=median training.actor_critic.counterfactual_uncertainty_beta=1.0 training.actor_critic.batch_num_samples=${CF_ACTOR_BATCH_NUM_SAMPLES:-16}"
      ;;
    cf_trim_k8_u1)
      echo "training.actor_critic.actor_loss_mode=counterfactual_group training.actor_critic.counterfactual_branches=8 training.actor_critic.counterfactual_baseline=trimmed_mean training.actor_critic.counterfactual_trim_ratio=0.25 training.actor_critic.counterfactual_uncertainty_beta=1.0 training.actor_critic.batch_num_samples=${CF_ACTOR_BATCH_NUM_SAMPLES:-16}"
      ;;
  esac
}

acquire_lock() {
  local lock_dir="$1"
  if mkdir "${lock_dir}" 2>/dev/null; then
    echo "${BASHPID}" > "${lock_dir}/pid"
    return 0
  fi
  local pid=""
  if [[ -f "${lock_dir}/pid" ]]; then
    pid="$(cat "${lock_dir}/pid" 2>/dev/null || true)"
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[actor-launch][error] Active lock exists: ${lock_dir} pid=${pid}"
    return 1
  fi
  echo "[actor-launch][resume] Removing stale lock: ${lock_dir}"
  rm -rf "${lock_dir}"
  mkdir "${lock_dir}"
  echo "${BASHPID}" > "${lock_dir}/pid"
}

write_manifest() {
  local manifest="$1"
  local task_name="$2"
  local game_short="$3"
  local variant="$4"
  local gpu="$5"
  local run_dir="$6"
  local source_run_dir="$7"
  local resume_flag="$8"
  local cmd_text="$9"
  cat > "${manifest}" <<EOF
task_name: ${task_name}
game: ${game_short}
variant: ${variant}
seed: ${SEED}
gpu: ${gpu}
server_set: ${SERVER_SET}
resume: ${resume_flag}
hydra_run_dir: ${run_dir}
source_run_dir: ${source_run_dir}
source_checkpoint: ${source_run_dir}/checkpoints/last.pt
source_dataset: ${source_run_dir}/checkpoints/dataset
log_file: ${LOG_ROOT}/${task_name}/train.log
command: ${cmd_text}
updated_at: $(date -Is)
EOF
}

run_one() {
  local gpu="$1"
  local game_short="$2"
  local variant="$3"
  local game="${game_short}NoFrameskip-v4"
  local task_name="${game_short}__${variant}"
  local log_dir="${LOG_ROOT}/${task_name}"
  local log_file="${log_dir}/train.log"
  local manifest="${log_dir}/run_manifest.yaml"
  local lock_dir="${log_dir}/run.lock"
  local run_dir="${OUTPUT_ROOT}/${task_name}/hydra_run"
  local source_run_dir
  local resume_flag="false"

  mkdir -p "${log_dir}" "$(dirname "${run_dir}")"
  if ! acquire_lock "${lock_dir}"; then
    return 1
  fi

  if ! source_run_dir="$(find_source_run_dir "${game_short}")"; then
    echo "[actor-launch][error] No source fixed-WM checkpoint found for ${game_short}" | python "${MONITOR_SCRIPT}" --task "${task_name}" >> "${log_file}"
    rm -rf "${lock_dir}"
    return 1
  fi

  if [[ "${AUTO_RESUME}" == "1" ]] && is_valid_actor_run_dir "${run_dir}"; then
    resume_flag="true"
    local epoch
    epoch="$(checkpoint_epoch "${run_dir}")"
    echo "[actor-launch][resume] ${task_name} from epoch ${epoch}: ${run_dir}" | python "${MONITOR_SCRIPT}" --task "${task_name}" >> "${log_file}"
  fi

  if [[ "${resume_flag}" == "false" && -d "${run_dir}" ]]; then
    local broken_run_dir="${run_dir}.broken_$(date +%Y%m%d_%H%M%S)"
    echo "[actor-launch][resume] moving invalid existing run dir to ${broken_run_dir}" | python "${MONITOR_SCRIPT}" --task "${task_name}" >> "${log_file}"
    mv "${run_dir}" "${broken_run_dir}"
  fi

  local extra_args_text
  extra_args_text="$(variant_args "${variant}")"
  read -r -a extra_args <<< "${extra_args_text}"

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "env.test.id=${game}"
    "common.seed=${SEED}"
    common.disable_tqdm=True
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "evaluation.every=${EVALUATION_EVERY}"
    training.fixed_world_model=True
    "initialization.agent.path_to_checkpoint=${source_run_dir}/checkpoints/last.pt"
    initialization.agent.load_tokenizer=True
    initialization.agent.load_world_model=True
    initialization.agent.load_actor_critic=False
    "initialization.dataset.path=${source_run_dir}/checkpoints/dataset"
    "wandb.mode=${WANDB_MODE}"
    "wandb.id=${task_name}-seed${SEED}"
    wandb.resume=allow
    "wandb.name=${task_name}-seed${SEED}"
    "wandb.group=fixedwm_actor_${SERVER_SET}_seed${SEED}"
    "outputs_dir_path=${OUTPUT_ROOT}/${task_name}"
    "hydra.run.dir=${run_dir}"
    "collection.train.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    "collection.test.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    evaluation.tokenizer.save_reconstructions=False
    "${extra_args[@]}"
  )

  if [[ "${resume_flag}" == "true" ]]; then
    cmd+=(common.resume=True hydra.output_subdir=null)
  fi

  write_manifest "${manifest}" "${task_name}" "${game_short}" "${variant}" "${gpu}" "${run_dir}" "${source_run_dir}" "${resume_flag}" "CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"

  set +e
  {
    echo "[task] date=$(date -Is)"
    echo "[task] task_name=${task_name}"
    echo "[task] game=${game}"
    echo "[task] variant=${variant}"
    echo "[task] gpu=${gpu}"
    echo "[task] source_run_dir=${source_run_dir}"
    echo "[task] hydra_run_dir=${run_dir}"
    echo "[task] resume=${resume_flag}"
    echo "[task] command=CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } 2>&1 | python "${MONITOR_SCRIPT}" --task "${task_name}" >> "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )); then
    echo "[actor-launch][fail] ${task_name} rc=${rc}; log=${log_file}"
  else
    echo "[actor-launch][ok] ${task_name}; log=${log_file}"
  fi
  rm -rf "${lock_dir}"
  return "${rc}"
}

worker_loop() {
  local worker_id="$1"
  local gpu="$2"
  local total_workers="$3"
  local i unit game_short variant

  for ((i = worker_id; i < ${#RUN_UNITS[@]}; i += total_workers)); do
    unit="${RUN_UNITS[$i]}"
    game_short="${unit%%|*}"
    variant="${unit##*|}"
    run_one "${gpu}" "${game_short}" "${variant}" || return 1
  done
}

handle_signal() {
  local sig="$1"
  local rc="$2"
  echo "[actor-launch][signal] received ${sig}; forwarding TERM to workers."
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  exit "${rc}"
}

trap 'handle_signal TERM 143' TERM
trap 'handle_signal HUP 129' HUP
trap 'handle_signal INT 130' INT

activate_conda
cd "${EASIMULUS_DIR}"
check_inputs
normalize_inputs

echo "[actor-launch] server_set=${SERVER_SET} seed=${SEED}"
echo "[actor-launch] tasks=${TASK_ARRAY[*]}"
echo "[actor-launch] variants=${VARIANT_ARRAY[*]}"
echo "[actor-launch] tasks_per_gpu=${TASKS_PER_GPU}"
echo "[actor-launch] output_root=${OUTPUT_ROOT}"
echo "[actor-launch] log_root=${LOG_ROOT}"
echo "[actor-launch] Hero is disabled."

total_workers=$((4 * TASKS_PER_GPU))
worker_id=0
for gpu in 0 1 2 3; do
  for ((slot = 0; slot < TASKS_PER_GPU; slot++)); do
    (
      sleep "$((worker_id * LAUNCH_STAGGER_SECONDS))"
      worker_loop "${worker_id}" "${gpu}" "${total_workers}"
    ) &
    WORKER_PIDS+=("$!")
    worker_pid_index="$((${#WORKER_PIDS[@]} - 1))"
    echo "[actor-launch] worker=${worker_id} gpu=${gpu} pid=${WORKER_PIDS[$worker_pid_index]}"
    worker_id=$((worker_id + 1))
  done
done

failures=0
for pid in "${WORKER_PIDS[@]}"; do
  set +e
  wait "${pid}"
  rc=$?
  set -e
  if (( rc != 0 )); then
    echo "[actor-launch][worker_fail] pid=${pid} rc=${rc}"
    failures=$((failures + 1))
  else
    echo "[actor-launch][worker_ok] pid=${pid}"
  fi
done

if (( failures > 0 )); then
  echo "[actor-launch][error] ${failures} worker(s) failed."
  exit 1
fi

echo "[actor-launch] all fixed-WM actor runs completed."
