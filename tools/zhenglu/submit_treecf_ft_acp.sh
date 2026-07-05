#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
WORKSPACE_NAME="${WORKSPACE_NAME:-share-space}"
AEC2_NAME="${AEC2_NAME:-share-cluster}"
CONTAINER_IMAGE_URL="${CONTAINER_IMAGE_URL:-registry.cn-sh-01.sensecore.cn/zhicheng_ccr/eawm:memory-20260608113416}"
WORKER_SPEC="${WORKER_SPEC:-n6ls.iu.i40.4.32c512g}"
QUOTA_TYPE="${QUOTA_TYPE:-spot}"
STORAGE_MOUNT="${STORAGE_MOUNT:-01995892-d478-76d8-aec7-13fd8284477e:/data}"
JOB_NAME="${JOB_NAME:-treecf-ft-$(date +%m%d%H%M)}"

EXPERIMENT_PREFIX="${EXPERIMENT_PREFIX:-treecf_ft_actor_atari}"
TASKS="${TASKS:-Alien Assault Asterix Breakout}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
VARIANTS="${VARIANTS:-treecf_lcb_topk_d3b3 treecf_cvar_sample_d2b4}"
EPOCHS="${EPOCHS:-260}"
ACTOR_LEARNING_RATE="${ACTOR_LEARNING_RATE:-5e-5}"
TREECF_EXTRA_OVERRIDES="${TREECF_EXTRA_OVERRIDES:-training.actor_critic.treecf_uncertainty_mode=relative}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-tools/zhenglu/run_treecf_actor_atari_h100.sh}"

TRAIN_COMMAND="cd ${PROJECT_ROOT}; \
source /data/share/hxd/miniconda3/etc/profile.d/conda.sh; \
git config --global --add safe.directory ${PROJECT_ROOT}; \
echo '========== TREECF FT ACP ATTEMPT' \$(date -Is) '=========='; \
nvidia-smi; \
EXPERIMENT_PREFIX='${EXPERIMENT_PREFIX}' \
LOAD_SOURCE_ACTOR=1 \
ACTOR_START_AFTER_EPOCHS=0 \
ACTOR_LEARNING_RATE='${ACTOR_LEARNING_RATE}' \
TREECF_INTRINSIC_REWARD_WEIGHT=0.0 \
TREECF_EXTRA_OVERRIDES='${TREECF_EXTRA_OVERRIDES}' \
TASKS='${TASKS}' \
GPU_LIST='${GPU_LIST}' \
VARIANTS='${VARIANTS}' \
EPOCHS='${EPOCHS}' \
AUTO_RESUME=1 \
bash ${LAUNCH_SCRIPT}"

echo "[treecf-acp] job_name=${JOB_NAME}"
echo "[treecf-acp] quota_type=${QUOTA_TYPE}"
echo "[treecf-acp] project_root=${PROJECT_ROOT}"
echo "[treecf-acp] experiment_prefix=${EXPERIMENT_PREFIX}"
echo "[treecf-acp] tasks=${TASKS}"
echo "[treecf-acp] variants=${VARIANTS}"
echo "[treecf-acp] command=${TRAIN_COMMAND}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

sco acp jobs create \
  --workspace-name="${WORKSPACE_NAME}" \
  --aec2-name="${AEC2_NAME}" \
  --job-name="${JOB_NAME}" \
  --container-image-url="${CONTAINER_IMAGE_URL}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec="${WORKER_SPEC}" \
  --quota-type="${QUOTA_TYPE}" \
  --storage-mount="${STORAGE_MOUNT}" \
  --command="${TRAIN_COMMAND}"
