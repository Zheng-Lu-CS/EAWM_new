#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/setup_easimulus_atari_demo_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}" "${PROJECT_ROOT}/ckpt" "${PROJECT_ROOT}/videos" \
  "${PROJECT_ROOT}/cache/pip" "${PROJECT_ROOT}/cache/torch" \
  "${PROJECT_ROOT}/cache/huggingface" "${PROJECT_ROOT}/cache/xdg" \
  "${PROJECT_ROOT}/cache/matplotlib"
exec > >(tee -a "${LOG_FILE}") 2>&1

export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

echo "[demo-setup] log: ${LOG_FILE}"
echo "[demo-setup] hostname: $(hostname)"
echo "[demo-setup] date: $(date -Is)"
echo "[demo-setup] pwd: $(pwd)"
echo "[demo-setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[demo-setup] ENV_NAME=${ENV_NAME}"
echo "[demo-setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[demo-setup][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[demo-setup][error] conda is not available in PATH. Run tools/zhenglu/setup_easimulus_atari_env.sh first."
  exit 1
fi

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${EASIMULUS_DIR}"

echo "[demo-setup] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[demo-setup] nvidia-smi: not found"
fi

python -m pip install imageio imageio-ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple

PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" python - <<'PY'
import importlib
import shutil
import sys

mods = [
    "torch", "torchvision", "gymnasium", "ale_py", "cv2", "hydra",
    "omegaconf", "numpy", "PIL", "imageio", "imageio_ffmpeg",
]
for name in mods:
    importlib.import_module(name)
    print(f"[demo-setup] imported {name}")

import torch
print(f"[demo-setup] python: {sys.version}")
print(f"[demo-setup] torch: {torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
print(f"[demo-setup] ffmpeg path: {shutil.which('ffmpeg') or 'imageio-ffmpeg bundled binary will be used'}")

import gymnasium
import ale_py
try:
    gymnasium.register_envs(ale_py)
except Exception:
    pass
env = gymnasium.make("BreakoutNoFrameskip-v4")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
print("[demo-setup] BreakoutNoFrameskip-v4 reset/step OK")
PY

if [[ ! -d "${PROJECT_ROOT}/ckpt" ]]; then
  echo "[demo-setup][error] ckpt directory was not created."
  exit 1
fi

echo "[demo-setup] Done. Put official checkpoints such as Breakout.pt under:"
echo "  ${PROJECT_ROOT}/ckpt"
echo "[demo-setup] Then run:"
echo "  cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} ENV_NAME=${ENV_NAME} bash tools/zhenglu/run_easimulus_atari_4gpu_demo.sh"
