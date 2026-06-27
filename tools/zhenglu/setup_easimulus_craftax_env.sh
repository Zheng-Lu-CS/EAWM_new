#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-zhenglu_eawm_craftax}"
RECREATE_ENV="${RECREATE_ENV:-1}"
EASIMULUS_DIR="${EASIMULUS_DIR:-${PROJECT_ROOT}/EASimulus}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/setup_easimulus_craftax_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[setup-craftax] log=${LOG_FILE}"
echo "[setup-craftax] hostname=$(hostname)"
echo "[setup-craftax] date=$(date -Is)"
echo "[setup-craftax] project_root=${PROJECT_ROOT}"
echo "[setup-craftax] easimulus_dir=${EASIMULUS_DIR}"
echo "[setup-craftax] env_name=${ENV_NAME}"
echo "[setup-craftax] recreate_env=${RECREATE_ENV}"

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[setup-craftax][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

mkdir -p \
  "${PROJECT_ROOT}/cache/pip" \
  "${PROJECT_ROOT}/cache/torch" \
  "${PROJECT_ROOT}/cache/huggingface" \
  "${PROJECT_ROOT}/cache/xdg" \
  "${PROJECT_ROOT}/cache/matplotlib" \
  "${PROJECT_ROOT}/outputs/crftax/token_id_6_27"

export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

APT_PACKAGES=(
  build-essential gcc git curl wget
  ffmpeg libsm6 libxext6 libgl1 libegl1
  libgl1-mesa-dev libosmesa6 libosmesa6-dev libglew-dev
  x11-xserver-utils xvfb mesa-utils zip rsync htop tmux
)

configure_ubuntu_apt_mirror() {
  if [[ ! -f /etc/os-release ]]; then
    return 0
  fi
  # shellcheck source=/dev/null
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    return 0
  fi

  echo "[setup-craftax] Ubuntu detected; trying Tsinghua apt mirror."
  local deb822="/etc/apt/sources.list.d/ubuntu.sources"
  local legacy="/etc/apt/sources.list"
  if [[ -f "${deb822}" ]]; then
    cp -n "${deb822}" "${deb822}.bak.${TIMESTAMP}" || true
    sed -i \
      -e 's#http://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#http://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#https://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#https://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      "${deb822}" || true
  elif [[ -f "${legacy}" ]]; then
    cp -n "${legacy}" "${legacy}.bak.${TIMESTAMP}" || true
    sed -i \
      -e 's#http://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#http://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#https://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      -e 's#https://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' \
      "${legacy}" || true
  fi
}

install_system_packages_or_hint() {
  if [[ "$(id -u)" == "0" ]]; then
    configure_ubuntu_apt_mirror
    echo "[setup-craftax] Installing system packages."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
    return 0
  fi

  echo "[setup-craftax] Current user is not root; system packages will not be installed automatically."
  if command -v dpkg-query >/dev/null 2>&1; then
    local missing=()
    local pkg
    for pkg in "${APT_PACKAGES[@]}"; do
      if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
        missing+=("${pkg}")
      fi
    done
    if (( ${#missing[@]} > 0 )); then
      echo "[setup-craftax][hint] Missing apt packages may need admin installation:"
      echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    else
      echo "[setup-craftax] Required apt packages appear to be installed."
    fi
  fi
}

activate_conda_base() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup-craftax][error] conda is not available in PATH."
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
}

print_runtime_info() {
  echo "[runtime] conda_env=${CONDA_DEFAULT_ENV:-<none>}"
  echo "[runtime] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "[runtime] nvidia-smi not found"
  fi
  python - <<'PY'
import sys
print(f"[runtime] python={sys.version}")
try:
    import torch
    print(f"[runtime] torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[runtime] device0={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[runtime] torch import failed: {exc}")
PY
}

run_smoke_checks() {
  echo "[setup-craftax] Running Craftax/EASimulus smoke checks."
  PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" python - <<'PY'
import importlib
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

modules = [
    "torch", "torchvision", "hydra", "omegaconf", "einops", "wandb",
    "numpy", "PIL", "imageio", "cv2", "pygame", "dm_env", "jax", "gymnax", "craftax",
]
for name in modules:
    importlib.import_module(name)
    print(f"[smoke] imported {name}")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
print(f"[smoke] CUDA OK: {torch.cuda.device_count()} visible device(s)")

from envs.wrappers.craftax import make_craftax
env = make_craftax("Craftax-Symbolic-v1")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
print("[smoke] Craftax-Symbolic-v1 reset/step OK")
print("[smoke] obs keys:", sorted(obs.keys()))
print("[smoke] map shape:", obs["token_2d"].shape)
print("[smoke] vector shape:", obs["vector"].shape)
print("[smoke] direction shape:", obs["token"].shape)

import main
print("[smoke] imported EASimulus src/main.py OK")
PY
}

install_system_packages_or_hint
activate_conda_base

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${RECREATE_ENV}" == "1" ]]; then
    echo "[setup-craftax] Removing existing conda env for zero-start setup: ${ENV_NAME}"
    conda env remove -y -n "${ENV_NAME}"
  else
    echo "[setup-craftax] Reusing existing conda env: ${ENV_NAME}"
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup-craftax] Creating conda env ${ENV_NAME} with Python 3.10"
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
cd "${EASIMULUS_DIR}"
print_runtime_info

cat > "${PIP_CACHE_DIR}/pip.conf" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
export PIP_CONFIG_FILE="${PIP_CACHE_DIR}/pip.conf"

echo "[setup-craftax] Upgrading pip tooling."
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[setup-craftax] Installing PyTorch 2.5.1 CUDA 12.4 wheels."
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[setup-craftax] Installing Craftax runtime dependencies plus official EASimulus import-time dependencies."
python -m pip install \
  "numpy==1.25.2" \
  "einops==0.8.0" \
  "gymnasium==1.0.0a2" \
  "hydra-core==1.3.2" \
  "opencv-python==4.7.0.72" \
  "protobuf==3.20.*" \
  "psutil==5.8.0" \
  "pygame>=2.1.2" \
  "tqdm>=4.62.3" \
  "wandb>=0.15.9" \
  "nltk==3.8.1" \
  "loguru" \
  "yet-another-retnet>=0.5.0" \
  "dm-env==1.6" \
  "imageio[ffmpeg]" \
  "scipy" \
  "jax[cpu]" \
  "gymnax" \
  "craftax" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[setup-craftax] Re-pinning numpy to the Craftax Dockerfile-compatible version."
python -m pip install "numpy==1.25.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

print_runtime_info
run_smoke_checks

echo "[setup-craftax] Done."
echo "[setup-craftax] Run demo:"
echo "  cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} ENV_NAME=${ENV_NAME} bash tools/zhenglu/run_easimulus_craftax_token_id_demo.sh"
