#!/usr/bin/env bash
# Draft-OPD installer.
#
# This is now a thin wrapper around install_npu.sh, which adaptively skips
# already-installed components (vllm / vllm-ascend / sglang / verl) and only
# installs what is missing. See install_npu.sh for the full documentation.
#
# Quick reference:
#   bash install.sh                         # adaptive install (recommended)
#   INSTALL_SGLANG_DFLASH=1 bash install.sh # also install the sglang engine
#   FORCE_INSTALL_ENGINES=1 bash install.sh # reinstall engines from submodules
set -euo pipefail
cd "$(dirname "$0")"

exec bash ./install_npu.sh "$@"
