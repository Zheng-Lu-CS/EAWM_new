#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-zhenglu_eawm_craftax}"
EASIMULUS_DIR="${EASIMULUS_DIR:-${PROJECT_ROOT}/EASimulus}"
CKPT_PATH="${CKPT_PATH:-${PROJECT_ROOT}/ckpt/EASimulus/craftax.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/crftax/token_id_6_27}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
GPU_ID="${GPU_ID:-0}"
VIDEO_SECONDS="${VIDEO_SECONDS:-300}"
VIDEO_FPS="${VIDEO_FPS:-1}"
SEED="${SEED:-0}"
CELL_SIZE="${CELL_SIZE:-58}"
FORCE="${FORCE:-1}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/craftax_token_id_demo_${TIMESTAMP}.log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" \
  "${PROJECT_ROOT}/cache/pip" \
  "${PROJECT_ROOT}/cache/torch" \
  "${PROJECT_ROOT}/cache/huggingface" \
  "${PROJECT_ROOT}/cache/xdg" \
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
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

activate_conda() {
  if [[ "${USE_CONDA:-1}" != "1" ]]; then
    return
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[craftax-demo][error] conda is not available in PATH."
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
    echo "[craftax-demo][error] EASimulus directory not found: ${EASIMULUS_DIR}"
    exit 1
  fi
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[craftax-demo][error] Craftax checkpoint not found: ${CKPT_PATH}"
    exit 1
  fi
  if [[ "${VIDEO_FPS}" -lt 1 ]]; then
    echo "[craftax-demo][error] VIDEO_FPS must be >= 1."
    exit 1
  fi
}

print_runtime_info() {
  echo "[craftax-demo] hostname=$(hostname)"
  echo "[craftax-demo] date=$(date -Is)"
  echo "[craftax-demo] project_root=${PROJECT_ROOT}"
  echo "[craftax-demo] easimulus_dir=${EASIMULUS_DIR}"
  echo "[craftax-demo] checkpoint=${CKPT_PATH}"
  echo "[craftax-demo] output_dir=${OUTPUT_DIR}"
  echo "[craftax-demo] log_file=${LOG_FILE}"
  echo "[craftax-demo] gpu_id=${GPU_ID}"
  echo "[craftax-demo] video_seconds=${VIDEO_SECONDS}"
  echo "[craftax-demo] video_fps=${VIDEO_FPS}"
  echo "[craftax-demo] seed=${SEED}"
  echo "[craftax-demo] conda_env=${CONDA_DEFAULT_ENV:-<none>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "[craftax-demo] nvidia-smi not found"
  fi
  PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" python - <<'PY'
import importlib
import sys
import torch

for name in ["torch", "torchvision", "hydra", "omegaconf", "jax", "gymnax", "craftax", "PIL", "imageio"]:
    importlib.import_module(name)
    print(f"[runtime] imported {name}")
print(f"[runtime] python={sys.version}")
print(f"[runtime] torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")
PY
}

activate_conda
check_inputs
print_runtime_info

VIDEO_PATH="${OUTPUT_DIR}/craftax_token_id_demo.mp4"
if [[ -s "${VIDEO_PATH}" && "${FORCE}" != "1" ]]; then
  echo "[craftax-demo][skip] existing video kept: ${VIDEO_PATH}"
  exit 0
fi

cd "${EASIMULUS_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" \
  python "${PROJECT_ROOT}/tools/zhenglu/record_easimulus_craftax_token_id_visual.py" \
    --easimulus-dir "${EASIMULUS_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --video-name "craftax_token_id_demo.mp4" \
    --seconds "${VIDEO_SECONDS}" \
    --fps "${VIDEO_FPS}" \
    --seed "${SEED}" \
    --cell-size "${CELL_SIZE}"

echo "[craftax-demo] Done."
echo "[craftax-demo] video=${VIDEO_PATH}"
echo "[craftax-demo] report=${OUTPUT_DIR}/report.md"
echo "[craftax-demo] csv=${OUTPUT_DIR}/frame_stats.csv"
echo "[craftax-demo] log=${LOG_FILE}"
