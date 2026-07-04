#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_SET="${SERVER_SET:-B}"
exec bash "${SCRIPT_DIR}/run_fixedwm_actor_atari_h100.sh"
