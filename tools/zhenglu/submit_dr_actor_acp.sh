#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
WORKSPACE_NAME="${WORKSPACE_NAME:-share-space}"
AEC2_NAME="${AEC2_NAME:-share-cluster}"
CONTAINER_IMAGE_URL="${CONTAINER_IMAGE_URL:-registry.cn-sh-01.sensecore.cn/zhicheng_ccr/eawm:memory-20260608113416}"
WORKER_SPEC="${WORKER_SPEC:-n6ls.iu.i40.4.32c512g}"
QUOTA_TYPE="${QUOTA_TYPE:-spot}"
STORAGE_MOUNT="${STORAGE_MOUNT:-01995892-d478-76d8-aec7-13fd8284477e:/data}"
JOB_NAME="${JOB_NAME:-dr-actor-$(date +%m%d%H%M)}"

BRANCH="${BRANCH:-exp/actor2}"
SYNC_GIT="${SYNC_GIT:-1}"
TASKS="${TASKS:-Alien Assault Asterix Breakout}"
VARIANTS="${VARIANTS:-dr_q_topk_l4k4 dr_hybrid_sample_l3k6}"
EPOCHS="${EPOCHS:-600}"
ACTOR_STEPS_PER_EPOCH="${ACTOR_STEPS_PER_EPOCH:-80}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-tools/zhenglu/run_dr_actor_atari_h100.sh}"
ANALYZE_LOGS="${ANALYZE_LOGS:-1}"

TRAIN_COMMAND="set -e; \
cd ${PROJECT_ROOT}; \
source /data/share/hxd/miniconda3/etc/profile.d/conda.sh; \
git config --global --add safe.directory ${PROJECT_ROOT}; \
if [ '${SYNC_GIT}' = '1' ]; then git fetch origin ${BRANCH} && git checkout ${BRANCH} && git pull --ff-only origin ${BRANCH}; fi; \
echo '========== DR ACTOR ACP ATTEMPT' \$(date -Is) '=========='; \
git log --oneline -5; \
nvidia-smi; \
set +e; \
TASKS='${TASKS}' \
VARIANTS='${VARIANTS}' \
EPOCHS='${EPOCHS}' \
ACTOR_STEPS_PER_EPOCH='${ACTOR_STEPS_PER_EPOCH}' \
AUTO_RESUME=1 \
bash ${LAUNCH_SCRIPT}; \
launcher_rc=\$?; \
set -e; \
if [ '${ANALYZE_LOGS}' = '1' ]; then python tools/zhenglu/analyze_actor_logs.py logs --contains dr_actor_atari --min-epoch 200 || true; fi; \
exit \${launcher_rc}"

echo "[dr-acp] job_name=${JOB_NAME}"
echo "[dr-acp] quota_type=${QUOTA_TYPE}"
echo "[dr-acp] project_root=${PROJECT_ROOT}"
echo "[dr-acp] branch=${BRANCH}"
echo "[dr-acp] tasks=${TASKS}"
echo "[dr-acp] variants=${VARIANTS}"
echo "[dr-acp] command=${TRAIN_COMMAND}"

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
