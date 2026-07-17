#!/usr/bin/env bash
# Draft-OPD installer.
#
# Installs, in dependency order:
#   1. vllm          (submodule, curnane-lab/vllm @ domino_npu_v0.21.0)
#   2. vllm-ascend   (submodule, curnane-lab/vllm-ascend @ domino_npu_v0.21.0rc1)
#   3. sglang-dflash (vendored SGLang rollout engine)
#   4. verl          (training stack)
#
# Prerequisites (Ascend NPU): CANN toolkit + torch + torch-npu matching the
# vllm-ascend version, and triton-ascend. See verl/requirements-npu.txt and
# the vllm-ascend installation guide. Python dependencies besides the engines
# are in verl/requirements.txt / verl/requirements-npu.txt.
#
# If you cloned without --recursive, first run:
#   git submodule update --init --recursive
#
# Container images that already ship vllm/vllm-ascend (e.g. editable installs
# under /vllm-workspace) should SKIP the engine steps to avoid clobbering
# them:
#   INSTALL_VLLM=0 INSTALL_VLLM_ASCEND=0 bash install.sh
#
# NOTE: always install verl WITHOUT extras (plain `pip install -e ./verl`).
# The [vllm]/[sglang] extras pin old engine versions (vllm<=0.12, torch==2.9.1)
# and would break an NPU/vllm-ascend v0.21 environment.
set -euo pipefail
cd "$(dirname "$0")"

INSTALL_VLLM=${INSTALL_VLLM:-1}
INSTALL_VLLM_ASCEND=${INSTALL_VLLM_ASCEND:-1}
INSTALL_SGLANG_DFLASH=${INSTALL_SGLANG_DFLASH:-1}

# Pull submodule contents when the user cloned without --recursive.
git submodule update --init --recursive

# 1. vLLM (CPU-only build of the frontend; NPU kernels come from vllm-ascend).
if [ "$INSTALL_VLLM" = "1" ]; then
    VLLM_TARGET_DEVICE=empty pip install -e ./vllm
fi

# 2. vLLM-Ascend (provides the `dflash` speculative method on NPU).
#    Its own nested submodules carry custom operators for Atlas A3.
if [ "$INSTALL_VLLM_ASCEND" = "1" ]; then
    (
        cd ./vllm-ascend
        git submodule update --init --recursive
        pip install -e .
    )
fi

# 3. SGLang rollout engine (alternative to the vLLM path).
if [ "$INSTALL_SGLANG_DFLASH" = "1" ]; then
    cd ./sglang-dflash
    pip install -e "./python"
    pip install cachetools
    cd ..
fi

# 4. verl training stack.
cd ./verl
pip install -e .

