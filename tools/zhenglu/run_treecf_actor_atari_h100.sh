#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
SEED="${SEED:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
TIMESTAMP="${EXP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-Alien Assault Asterix Breakout}"
VARIANTS="${VARIANTS:-treecf_lcb_topk_d3b3 treecf_cvar_sample_d2b4}"
SOURCE_WORLD_MODEL_OVERRIDES="${SOURCE_WORLD_MODEL_OVERRIDES:-world_model.event_pred=True world_model.ges=True}"
FIXED_WM_SOURCE_ROOT="${FIXED_WM_SOURCE_ROOT:-${PROJECT_ROOT}/outputs}"
ALLOW_SOURCE_ROOT_FALLBACK="${ALLOW_SOURCE_ROOT_FALLBACK:-1}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-10}"
AUTO_RESUME="${AUTO_RESUME:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
EVALUATION_EVERY="${EVALUATION_EVERY:-10}"
MEDIA_EPISODES_TO_SAVE="${MEDIA_EPISODES_TO_SAVE:-0}"
EPOCHS="${EPOCHS:-600}"
ACTOR_STEPS_PER_EPOCH="${ACTOR_STEPS_PER_EPOCH:-80}"
COLLECT_REAL_PREFIX="${COLLECT_REAL_PREFIX:-0}"
SOURCE_OUTPUT_PREFIX="${SOURCE_OUTPUT_PREFIX:-easimulus_atari_}"
RESUME_OUTPUT_PREFIX="${RESUME_OUTPUT_PREFIX:-treecf_actor_atari_seed${SEED}_}"
DRY_RUN="${DRY_RUN:-0}"

EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TOOLS_DIR="${PROJECT_ROOT}/tools/zhenglu"
MONITOR_SCRIPT="${TOOLS_DIR}/monitor_easimulus_metrics.py"
LOG_ROOT="${PROJECT_ROOT}/logs/treecf_actor_atari_seed${SEED}_${TIMESTAMP}"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/treecf_actor_atari_seed${SEED}_${TIMESTAMP}"
MASTER_LOG="${LOG_ROOT}/launcher.log"

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
declare -a WORKER_PIDS=()

activate_conda() {
  if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[treecf-launch][error] conda is not available in PATH."
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
}

check_inputs() {
  read -r -a TASK_ARRAY <<< "${TASKS}"
  read -r -a VARIANT_ARRAY <<< "${VARIANTS}"
  if (( ${#TASK_ARRAY[@]} != 4 )); then
    echo "[treecf-launch][error] TASKS must contain exactly 4 Atari game names; got ${#TASK_ARRAY[@]}: ${TASKS}"
    exit 1
  fi
  if (( ${#VARIANT_ARRAY[@]} != 2 )); then
    echo "[treecf-launch][error] VARIANTS must contain exactly 2 variants; got ${#VARIANT_ARRAY[@]}: ${VARIANTS}"
    exit 1
  fi
  if [[ ! -d "${EASIMULUS_DIR}" ]]; then
    echo "[treecf-launch][error] Missing EASimulus dir: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[treecf-launch] DRY_RUN=1: skipping CUDA device-count check."
    return 0
  fi
  local gpu_count
  gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  if (( gpu_count < 4 )); then
    echo "[treecf-launch][error] Need at least 4 visible CUDA devices; found ${gpu_count}."
    exit 1
  fi
}

is_valid_source_run_dir() {
  local run_dir="$1"
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
  return 0
}

is_valid_actor_run_dir() {
  local run_dir="$1"
  [[ -f "${run_dir}/checkpoints/run_metadata.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/last.pt" ]] || return 1
  [[ -f "${run_dir}/checkpoints/optimizer.pt" ]] || return 1
  [[ -d "${run_dir}/checkpoints/dataset" ]] || return 1
  return 0
}

checkpoint_epoch() {
  local run_dir="$1"
  python - "${run_dir}/checkpoints/run_metadata.pt" <<'PY'
import sys, torch
metadata = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(metadata.get("epoch", 0)))
PY
}

find_actor_run_dir() {
  local task_name="$1"
  local line metadata run_dir
  if [[ ! -d "${PROJECT_ROOT}/outputs" ]]; then
    return 1
  fi
  while IFS= read -r line; do
    metadata="${line#* }"
    run_dir="$(dirname "$(dirname "${metadata}")")"
    if is_valid_actor_run_dir "${run_dir}"; then
      echo "${run_dir}"
      return 0
    fi
  done < <(
    find "${PROJECT_ROOT}/outputs" \
      -type f \
      -path "*/${task_name}/hydra_run/checkpoints/run_metadata.pt" \
      -path "*/${RESUME_OUTPUT_PREFIX}*/*" \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr
  )
  return 1
}

source_override_var_name() {
  local game_short="$1"
  local key
  key="$(printf '%s' "${game_short}" | tr -c '[:alnum:]_' '_')"
  echo "SOURCE_RUN_DIR_${key}"
}

find_source_run_dir() {
  local game_short="$1"
  local game="${game_short}NoFrameskip-v4"
  local line ckpt run_dir search_root override_var override_value source_pass

  override_var="$(source_override_var_name "${game_short}")"
  override_value="$(printenv "${override_var}" 2>/dev/null || true)"
  if [[ -n "${override_value}" ]]; then
    if is_valid_source_run_dir "${override_value}"; then
      echo "${override_value}"
      return 0
    fi
    echo "[treecf-launch][error] ${override_var} is set but invalid: ${override_value}" >&2
    return 1
  fi

  for source_pass in prefixed fallback; do
    if [[ "${source_pass}" == "fallback" ]]; then
      if [[ "${ALLOW_SOURCE_ROOT_FALLBACK}" != "1" ]]; then
        break
      fi
      echo "[treecf-launch][warn] No valid ${SOURCE_OUTPUT_PREFIX}* source found for ${game_short}; falling back to recursive search under ${FIXED_WM_SOURCE_ROOT}" >&2
    fi

    while IFS= read -r line; do
      ckpt="${line#* }"
      run_dir="$(dirname "$(dirname "${ckpt}")")"
      [[ "${run_dir}" == *"/treecf_actor_atari_"* ]] && continue
      [[ "${run_dir}" == *"/fixedwm_actor_atari_"* ]] && continue
      if [[ "${ckpt}" != *"/${game_short}_seed${SEED}/"* && "${ckpt}" != *"/${game}/"* ]]; then
        continue
      fi
      if is_valid_source_run_dir "${run_dir}"; then
        echo "${run_dir}"
        return 0
      fi
    done < <(
      while IFS= read -r search_root; do
        find "${search_root}" \
          -type f \
          -path "*/checkpoints/last.pt" \
          -printf '%T@ %p\n' 2>/dev/null
      done < <(
        if [[ -d "${FIXED_WM_SOURCE_ROOT}" ]]; then
          if [[ "${source_pass}" == "prefixed" ]]; then
            find "${FIXED_WM_SOURCE_ROOT}" -maxdepth 1 -type d -name "${SOURCE_OUTPUT_PREFIX}*" 2>/dev/null
          else
            printf '%s\n' "${FIXED_WM_SOURCE_ROOT}"
          fi
        fi
      ) | sort -nr
    )
  done
  return 1
}

variant_args() {
  local variant="$1"
  case "${variant}" in
    treecf_lcb_topk_d3b3)
      echo "training.actor_critic.actor_loss_mode=tree_counterfactual training.actor_critic.batch_num_samples=${TREECF_LCB_BATCH_NUM_SAMPLES:-8} training.actor_critic.treecf_depth=3 training.actor_critic.treecf_branching=3 training.actor_critic.treecf_candidate_mode=topk training.actor_critic.treecf_backup=lcb training.actor_critic.treecf_lcb_alpha=0.5 training.actor_critic.treecf_adv_baseline=median training.actor_critic.treecf_uncertainty_beta=1.0 training.actor_critic.treecf_depth_decay=0.95 training.actor_critic.entropy_weight=0.001"
      ;;
    treecf_cvar_sample_d2b4)
      echo "training.actor_critic.actor_loss_mode=tree_counterfactual training.actor_critic.batch_num_samples=${TREECF_CVAR_BATCH_NUM_SAMPLES:-12} training.actor_critic.treecf_depth=2 training.actor_critic.treecf_branching=4 training.actor_critic.treecf_candidate_mode=sample training.actor_critic.treecf_backup=cvar training.actor_critic.treecf_cvar_fraction=0.5 training.actor_critic.treecf_sample_temperature=1.1 training.actor_critic.treecf_adv_baseline=trimmed_mean training.actor_critic.treecf_uncertainty_beta=0.5 training.actor_critic.treecf_depth_decay=1.0 training.actor_critic.entropy_weight=0.002"
      ;;
    *)
      echo "[treecf-launch][error] Unsupported variant '${variant}'." >&2
      exit 1
      ;;
  esac
}

monitor_cmd() {
  local task_name="$1"
  if [[ -f "${MONITOR_SCRIPT}" ]]; then
    python "${MONITOR_SCRIPT}" --task "${task_name}"
  else
    cat
  fi
}

run_one() {
  local gpu="$1"
  local game_short="$2"
  local variant="$3"
  local game="${game_short}NoFrameskip-v4"
  local task_name="${game_short}__${variant}"
  local log_dir="${LOG_ROOT}/${task_name}"
  local log_file="${log_dir}/train.log"
  local task_output="${OUTPUT_ROOT}/${task_name}"
  local run_dir="${task_output}/hydra_run"
  local source_run_dir
  local resume_run_dir
  local resume_flag="false"

  mkdir -p "${log_dir}" "$(dirname "${run_dir}")"
  if ! source_run_dir="$(find_source_run_dir "${game_short}")"; then
    if [[ "${DRY_RUN}" == "1" ]]; then
      source_run_dir="${FIXED_WM_SOURCE_ROOT}/MISSING_${game_short}_SOURCE_FOR_DRY_RUN"
      echo "[treecf-launch][warn] No source checkpoint+dataset found for ${game_short}; using dry-run placeholder ${source_run_dir}" | tee -a "${log_file}"
    else
      echo "[treecf-launch][error] No source EASimulus checkpoint+dataset found for ${game_short} under ${FIXED_WM_SOURCE_ROOT}" | tee -a "${log_file}"
      return 1
    fi
  fi

  if [[ "${AUTO_RESUME}" == "1" ]] && resume_run_dir="$(find_actor_run_dir "${task_name}")"; then
    run_dir="${resume_run_dir}"
    task_output="$(dirname "${run_dir}")"
    resume_flag="true"
    local resume_epoch
    resume_epoch="$(checkpoint_epoch "${run_dir}")"
    echo "[treecf-launch][resume] ${task_name} from epoch ${resume_epoch}: ${run_dir}" | tee -a "${log_file}"
    if (( resume_epoch >= EPOCHS )); then
      echo "[treecf-launch][skip] ${task_name} already reached epoch ${resume_epoch}." | tee -a "${log_file}"
      return 0
    fi
  elif [[ -d "${run_dir}" ]]; then
    local invalid_dir="${run_dir}.invalid_${TIMESTAMP}"
    echo "[treecf-launch][resume] moving invalid existing run dir to ${invalid_dir}" | tee -a "${log_file}"
    mv "${run_dir}" "${invalid_dir}"
  fi

  local variant_args_text
  variant_args_text="$(variant_args "${variant}")"
  read -r -a extra_args <<< "${variant_args_text}"
  read -r -a source_wm_args <<< "${SOURCE_WORLD_MODEL_OVERRIDES}"

  local train_stop_after
  train_stop_after="${COLLECT_REAL_PREFIX}"

  local cmd=(
    python src/main.py
    tokenizer.image.with_lpips=True
    benchmark=atari
    "env.train.id=${game}"
    "env.test.id=${game}"
    "common.seed=${SEED}"
    "common.epochs=${EPOCHS}"
    common.disable_tqdm=True
    "common.checkpoint_every=${CHECKPOINT_EVERY}"
    "evaluation.every=${EVALUATION_EVERY}"
    training.fixed_world_model=True
    "training.actor_critic.steps_per_epoch=${ACTOR_STEPS_PER_EPOCH}"
    "collection.train.stop_after_epochs=${train_stop_after}"
    "${source_wm_args[@]}"
    "initialization.agent.path_to_checkpoint=${source_run_dir}/checkpoints/last.pt"
    initialization.agent.load_tokenizer=True
    initialization.agent.load_world_model=True
    initialization.agent.load_actor_critic=False
    "initialization.dataset.path=${source_run_dir}/checkpoints/dataset"
    "wandb.mode=${WANDB_MODE}"
    "wandb.id=${task_name}-seed${SEED}"
    wandb.resume=allow
    "wandb.name=${task_name}-seed${SEED}"
    "wandb.group=treecf_actor_atari_seed${SEED}_${TIMESTAMP}"
    "outputs_dir_path=${task_output}"
    "hydra.run.dir=${run_dir}"
    "collection.train.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    "collection.test.num_episodes_to_save=${MEDIA_EPISODES_TO_SAVE}"
    evaluation.tokenizer.save_reconstructions=False
    "${extra_args[@]}"
  )

  if [[ "${resume_flag}" == "true" ]]; then
    cmd+=(common.resume=True hydra.output_subdir=null)
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    {
      echo "[task] dry_run=1"
      echo "[task] task_name=${task_name}"
      echo "[task] gpu=${gpu}"
      echo "[task] source_run_dir=${source_run_dir}"
      echo "[task] hydra_run_dir=${run_dir}"
      echo "[task] resume=${resume_flag}"
      echo "[task] command=CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    } | tee -a "${log_file}"
    return 0
  fi

  set +e
  {
    echo "[task] date=$(date -Is)"
    echo "[task] task_name=${task_name}"
    echo "[task] gpu=${gpu}"
    echo "[task] game=${game}"
    echo "[task] variant=${variant}"
    echo "[task] source_run_dir=${source_run_dir}"
    echo "[task] hydra_run_dir=${run_dir}"
    echo "[task] resume=${resume_flag}"
    echo "[task] command=CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
  } 2>&1 | monitor_cmd "${task_name}" >> "${log_file}"
  local rc=${PIPESTATUS[0]}
  set -e

  if (( rc != 0 )); then
    echo "[treecf-launch][fail] ${task_name} rc=${rc}; log=${log_file}"
    if [[ -f "${log_file}" ]]; then
      echo "[treecf-launch][fail_tail_begin] ${task_name}"
      tail -n "${FAIL_TAIL_LINES:-160}" "${log_file}" || true
      echo "[treecf-launch][fail_tail_end] ${task_name}"
    fi
  else
    echo "[treecf-launch][ok] ${task_name}; log=${log_file}"
  fi
  return "${rc}"
}

handle_signal() {
  local sig="$1"
  local rc="$2"
  echo "[treecf-launch][signal] received ${sig}; forwarding TERM to workers."
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

echo "[treecf-launch] timestamp=${TIMESTAMP}"
echo "[treecf-launch] tasks=${TASK_ARRAY[*]}"
echo "[treecf-launch] variants=${VARIANT_ARRAY[*]}"
echo "[treecf-launch] output_root=${OUTPUT_ROOT}"
echo "[treecf-launch] log_root=${LOG_ROOT}"
echo "[treecf-launch] source_root=${FIXED_WM_SOURCE_ROOT}"
echo "[treecf-launch] source_output_prefix=${SOURCE_OUTPUT_PREFIX}"
echo "[treecf-launch] source_world_model_overrides=${SOURCE_WORLD_MODEL_OVERRIDES}"
echo "[treecf-launch] allow_source_root_fallback=${ALLOW_SOURCE_ROOT_FALLBACK}"
echo "[treecf-launch] resume_output_prefix=${RESUME_OUTPUT_PREFIX}"
echo "[treecf-launch] collect_real_prefix=${COLLECT_REAL_PREFIX}"
echo "[treecf-launch] dry_run=${DRY_RUN}"

worker_id=0
for gpu in 0 1 2 3; do
  game_short="${TASK_ARRAY[$gpu]}"
  for variant in "${VARIANT_ARRAY[@]}"; do
    (
      sleep "$((worker_id * LAUNCH_STAGGER_SECONDS))"
      run_one "${gpu}" "${game_short}" "${variant}"
    ) &
    WORKER_PIDS+=("$!")
    worker_pid_index="$((${#WORKER_PIDS[@]} - 1))"
    echo "[treecf-launch] worker=${worker_id} gpu=${gpu} game=${game_short} variant=${variant} pid=${WORKER_PIDS[$worker_pid_index]}"
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
    echo "[treecf-launch][worker_fail] pid=${pid} rc=${rc}"
    failures=$((failures + 1))
  else
    echo "[treecf-launch][worker_ok] pid=${pid}"
  fi
done

if (( failures > 0 )); then
  echo "[treecf-launch][error] ${failures} worker(s) failed."
  exit 1
fi

echo "[treecf-launch] all TreeCF Atari actor runs completed."
