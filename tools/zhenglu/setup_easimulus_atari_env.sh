#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/share/hxd/zhenglu/eawm}"
ENV_NAME="${ENV_NAME:-zhenglu_easimulus}"
EASIMULUS_DIR="${PROJECT_ROOT}/EASimulus"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/setup_easimulus_atari_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[setup] log: ${LOG_FILE}"
echo "[setup] hostname: $(hostname)"
echo "[setup] date: $(date -Is)"
echo "[setup] pwd: $(pwd)"
echo "[setup] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[setup] ENV_NAME=${ENV_NAME}"
echo "[setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -d "${EASIMULUS_DIR}" ]]; then
  echo "[setup][error] EASimulus directory not found: ${EASIMULUS_DIR}"
  exit 1
fi

mkdir -p \
  "${PROJECT_ROOT}/cache/pip" \
  "${PROJECT_ROOT}/cache/torch" \
  "${PROJECT_ROOT}/cache/huggingface" \
  "${PROJECT_ROOT}/cache/xdg" \
  "${PROJECT_ROOT}/cache/matplotlib" \
  "${PROJECT_ROOT}/outputs"

export PIP_CACHE_DIR="${PROJECT_ROOT}/cache/pip"
export TORCH_HOME="${PROJECT_ROOT}/cache/torch"
export HF_HOME="${PROJECT_ROOT}/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${PROJECT_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PROJECT_ROOT}/cache/xdg"
export MPLCONFIGDIR="${PROJECT_ROOT}/cache/matplotlib"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

APT_PACKAGES=(
  ffmpeg libsm6 libxext6 libgl1 libegl1 libgl1-mesa-dev libosmesa6
  libosmesa6-dev libglew-dev x11-xserver-utils xvfb mesa-utils zip rsync
  bison flex htop tmux
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

  echo "[setup] Ubuntu detected; configuring Tsinghua apt mirror when possible."
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
    echo "[setup] Installing system packages: ${APT_PACKAGES[*]}"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
    return 0
  fi

  echo "[setup] Current user is not root; system packages will not be installed automatically."
  if command -v dpkg-query >/dev/null 2>&1; then
    local missing=()
    local pkg
    for pkg in "${APT_PACKAGES[@]}"; do
      if ! dpkg-query -W -f='${Status}' "${pkg}" 2>/dev/null | grep -q "install ok installed"; then
        missing+=("${pkg}")
      fi
    done
    if (( ${#missing[@]} > 0 )); then
      echo "[setup][hint] Missing apt packages may need admin installation:"
      echo "  sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    else
      echo "[setup] Required apt packages appear to be installed."
    fi
  else
    echo "[setup][hint] dpkg-query is unavailable; please verify ffmpeg/libGL/libSM/xvfb/mesa packages manually."
  fi
}

activate_conda() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup][error] conda is not available in PATH."
    exit 1
  fi
  local conda_base
  conda_base="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${conda_base}/etc/profile.d/conda.sh"
}

print_runtime_info() {
  echo "[runtime] hostname: $(hostname)"
  echo "[runtime] date: $(date -Is)"
  echo "[runtime] pwd: $(pwd)"
  echo "[runtime] conda env: ${CONDA_DEFAULT_ENV:-<none>}"
  echo "[runtime] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "[runtime] nvidia-smi: not found"
  fi
  python - <<'PY'
import sys
print(f"[runtime] python: {sys.version}")
try:
    import torch
    print(f"[runtime] torch: {torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[runtime] torch cuda: {torch.version.cuda}, device0={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"[runtime] torch import failed: {exc}")
PY
}

maybe_patch_ale_registration() {
  echo "[setup] Checking Atari NoFrameskip-v4 registration."
  set +e
python - <<'PY'
import gymnasium
import ale_py

def can_make(env_id):
    try:
        env = gymnasium.make(env_id)
        env.close()
        return True
    except Exception as exc:
        print(f"[ale-check] {env_id} failed: {exc}")
        return False

if can_make("BreakoutNoFrameskip-v4"):
    raise SystemExit(0)
try:
    gymnasium.register_envs(ale_py)
except Exception as exc:
    print(f"[ale-check] explicit ale_py registration failed: {exc}")
if can_make("BreakoutNoFrameskip-v4"):
    raise SystemExit(42)
if can_make("ALE/Breakout-v5"):
    raise SystemExit(42)
raise SystemExit(1)
PY
  local rc=$?
  set -e

  if [[ "${rc}" == "0" ]]; then
    echo "[setup] BreakoutNoFrameskip-v4 can be created."
    return 0
  fi
  if [[ "${rc}" != "42" ]]; then
    echo "[setup][error] Neither BreakoutNoFrameskip-v4 nor ALE/Breakout-v5 could be created. Check Atari ROM installation."
    return "${rc}"
  fi

  local target="${EASIMULUS_DIR}/src/envs/wrappers/atari.py"
  echo "[setup] Explicit ale_py registration is required; applying minimal compatibility patch to ${target}."
  python - "${target}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = "    import ale_py\n"
patch = (
    "    import ale_py\n"
    "    # Compatibility patch: some gymnasium/ale-py installs need explicit ALE registration for NoFrameskip-v4 ids.\n"
    "    gymnasium.register_envs(ale_py)\n"
)
if "gymnasium.register_envs(ale_py)" not in text:
    if needle not in text:
        raise SystemExit(f"Could not find patch location in {path}")
    path.write_text(text.replace(needle, patch, 1))
print(f"[ale-check] patched={path}")
PY

  python - <<'PY'
import gymnasium
import ale_py
gymnasium.register_envs(ale_py)
env = gymnasium.make("BreakoutNoFrameskip-v4")
env.close()
print("[ale-check] BreakoutNoFrameskip-v4 works after registration patch.")
PY
}

run_smoke_checks() {
  echo "[setup] Running Python import and Atari environment smoke checks."
  PYTHONPATH="${EASIMULUS_DIR}/src:${PYTHONPATH:-}" python - <<'PY'
import importlib
import sys

modules = ["torch", "torchvision", "gymnasium", "ale_py", "cv2", "hydra", "omegaconf", "wandb", "numpy", "PIL"]
for name in modules:
    importlib.import_module(name)
    print(f"[smoke] imported {name}")

import torch
if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.is_available() is False")
print(f"[smoke] CUDA OK: {torch.cuda.device_count()} visible device(s)")

import gymnasium
import ale_py
try:
    env = gymnasium.make("BreakoutNoFrameskip-v4")
except Exception:
    gymnasium.register_envs(ale_py)
    env = gymnasium.make("BreakoutNoFrameskip-v4")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
print("[smoke] BreakoutNoFrameskip-v4 reset/step OK")

import main
print("[smoke] imported src/main.py dependencies OK")
PY
}

install_system_packages_or_hint
activate_conda

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Reusing existing conda env: ${ENV_NAME}"
else
  echo "[setup] Creating conda env ${ENV_NAME} with Python 3.10"
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
cd "${EASIMULUS_DIR}"

print_runtime_info

mkdir -p "${PIP_CACHE_DIR}"
cat > "${PIP_CACHE_DIR}/pip.conf" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
export PIP_CONFIG_FILE="${PIP_CACHE_DIR}/pip.conf"

echo "[setup] Upgrading pip tooling via Tsinghua PyPI mirror."
python -m pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[setup] Installing PyTorch 2.5.1 CUDA 12.4 wheels from the official PyTorch index."
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124

echo "[setup] Installing EASimulus requirements via Tsinghua PyPI mirror."
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "[setup] Installing AutoROM. Running AutoROM --accept-license means the user confirms the required Atari ROM research-use/license acceptance."
python -m pip install "autorom[accept-rom-license]" -i https://pypi.tuna.tsinghua.edu.cn/simple
AutoROM --accept-license

maybe_patch_ale_registration

echo "[setup] Downloading LPIPS/VGG weights with official EASimulus helper."
python get_lpips.py

print_runtime_info
run_smoke_checks

echo "[setup] Done."
echo "[setup] 4-GPU smoke:"
echo "  cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} ENV_NAME=${ENV_NAME} bash tools/zhenglu/run_easimulus_atari_4gpu_smoke.sh"
echo "[setup] 4-GPU full training:"
echo "  cd ${PROJECT_ROOT} && PROJECT_ROOT=${PROJECT_ROOT} ENV_NAME=${ENV_NAME} bash tools/zhenglu/run_easimulus_atari_4gpu_train.sh"
