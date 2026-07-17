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
set -euo pipefail
cd "$(dirname "$0")"

# Pull submodule contents when the user cloned without --recursive.
git submodule update --init --recursive

# 1. vLLM (CPU-only build of the frontend; NPU kernels come from vllm-ascend).
VLLM_TARGET_DEVICE=empty pip install -e ./vllm

# 2. vLLM-Ascend (provides the `dflash` speculative method on NPU).
#    Its own nested submodules carry custom operators for Atlas A3.
(
    cd ./vllm-ascend
    git submodule update --init --recursive
    pip install -e .
)

# 3. SGLang rollout engine (alternative to the vLLM path).
cd ./sglang-dflash
pip install -e "./python"
pip install cachetools
cd ..

# 4. verl training stack.
cd ./verl
pip install -e .
