#!/usr/bin/env bash
# Draft-OPD NPU installer (adaptive).
#
# Detects already-installed components and skips them, so the same script
# works in:
#   - pre-provisioned NPU containers (vllm/vllm-ascend already installed,
#     e.g. editable installs under /vllm-workspace): engines are skipped;
#   - bare environments: engines are installed from the in-repo submodules
#     (git submodule update --init --recursive runs only when needed).
#
# Components:
#   1. vllm          skip if `pip show vllm` exists (warns if not v0.21.x)
#   2. vllm_ascend   skip if `pip show vllm_ascend` exists (warns if not 0.21.0rc*)
#   3. triton-ascend skip if installed, else install ==3.2.1 (Huawei mirror)
#   4. sglang        SKIPPED BY DEFAULT — the vLLM(vllm-ascend) path does not
#                    use it, and the vendored CUDA-ecosystem sglang-dflash can
#                    disturb the container's torch/torch_npu stack at install
#                    time. Opt in with INSTALL_SGLANG_DFLASH=1 only if you need
#                    the SGLang reference path; skip if `pip show sglang` exists.
#   5. verl          skip if already editable-installed from THIS repo
#
# Overrides:
#   FORCE_INSTALL_ENGINES=1  install engines from submodules even if detected
#   INSTALL_SGLANG_DFLASH=1  also install the vendored sglang-dflash engine
#   PYTHON=python3           interpreter to probe (default: python)
#
# NOTE: verl is installed WITHOUT extras on purpose. verl[vllm] pins
# vllm<=0.12.0 and verl[sglang] pins torch==2.9.1 — both break an
# NPU / vllm-ascend v0.21 stack.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
FORCE_INSTALL_ENGINES=${FORCE_INSTALL_ENGINES:-0}
INSTALL_SGLANG_DFLASH=${INSTALL_SGLANG_DFLASH:-0}

log()  { echo -e "\033[1;32m[install_npu]\033[0m $*"; }
warn() { echo -e "\033[1;33m[install_npu][warn]\033[0m $*"; }
die()  { echo -e "\033[1;31m[install_npu][error]\033[0m $*" >&2; exit 1; }

dist_version() {
    "$PYTHON" - "$1" <<'PY'
import importlib.metadata as md
import sys
try:
    print(md.version(sys.argv[1]))
except md.PackageNotFoundError:
    pass
PY
}

need_submodules=0
ensure_submodules() {
    [ "$need_submodules" = "1" ] || return 0
    log "fetching engine submodules (git submodule update --init --recursive)"
    git submodule update --init --recursive
}

# ---------------------------------------------------------------------------
# 0. Sanity checks: NPU stack presence (never auto-installed — must match CANN)
# ---------------------------------------------------------------------------
torch_ver=$(dist_version torch || true)
if [ -z "$torch_ver" ]; then
    warn "torch not found; install torch + torch_npu matching your CANN toolkit first."
fi
torch_npu_ver=$(dist_version torch_npu || true)
if [ -n "$torch_npu_ver" ]; then
    log "torch_npu detected: $torch_npu_ver"
else
    warn "torch_npu not found; the vLLM(vllm-ascend) rollout path requires it."
fi

# ---------------------------------------------------------------------------
# 1. vllm
# ---------------------------------------------------------------------------
vllm_ver=$(dist_version vllm || true)
if [ -n "$vllm_ver" ] && [ "$FORCE_INSTALL_ENGINES" != "1" ]; then
    log "vllm already installed ($vllm_ver) — skip"
    case "$vllm_ver" in
        0.21.*) ;;
        *) warn "the OPD vLLM patches target vllm v0.21.x; found $vllm_ver — hook compatibility not guaranteed" ;;
    esac
else
    need_submodules=1
    ensure_submodules
    log "installing vllm from ./vllm (VLLM_TARGET_DEVICE=empty)"
    VLLM_TARGET_DEVICE=empty pip install -e ./vllm
fi

# ---------------------------------------------------------------------------
# 2. vllm-ascend
# ---------------------------------------------------------------------------
vllm_ascend_ver=$(dist_version vllm_ascend || true)
if [ -n "$vllm_ascend_ver" ] && [ "$FORCE_INSTALL_ENGINES" != "1" ]; then
    log "vllm_ascend already installed ($vllm_ascend_ver) — skip"
    case "$vllm_ascend_ver" in
        0.21.0rc*) ;;
        *) warn "the OPD patches target vllm-ascend v0.21.0rc1; found $vllm_ascend_ver — hook compatibility not guaranteed" ;;
    esac
else
    need_submodules=1
    ensure_submodules
    log "installing vllm-ascend from ./vllm-ascend (with nested submodules)"
    (
        cd ./vllm-ascend
        git submodule update --init --recursive
        pip install -e .
    )
fi

# ---------------------------------------------------------------------------
# 3. triton-ascend (NPU triton kernels; required by vllm-ascend)
# ---------------------------------------------------------------------------
triton_ascend_ver=$(dist_version triton_ascend || true)
if [ -n "$triton_ascend_ver" ]; then
    log "triton-ascend already installed ($triton_ascend_ver) — skip"
else
    log "installing triton-ascend==3.2.1"
    pip install triton-ascend==3.2.1 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
fi

# ---------------------------------------------------------------------------
# 4. sglang (skipped by default: the vLLM path does not use it)
# ---------------------------------------------------------------------------
sglang_ver=$(dist_version sglang || true)
if [ "$INSTALL_SGLANG_DFLASH" != "1" ]; then
    log "sglang engine not needed on the vLLM path — skip (INSTALL_SGLANG_DFLASH=1 to opt in)"
elif [ -n "$sglang_ver" ]; then
    log "sglang already installed ($sglang_ver) — skip"
else
    log "installing vendored sglang-dflash"
    (
        cd ./sglang-dflash
        pip install -e "./python"
        pip install cachetools
    )
fi

# ---------------------------------------------------------------------------
# 5. verl (training stack) — always WITHOUT extras
# ---------------------------------------------------------------------------
verl_location=$(pip show verl 2>/dev/null | grep -i "Editable project location" | awk '{print $NF}' || true)
repo_verl_dir="$(cd ./verl && pwd)"
if [ -n "$verl_location" ] && [ "$verl_location" = "$repo_verl_dir" ]; then
    log "verl already editable-installed from this repo — skip"
else
    [ -n "$verl_location" ] && warn "verl is editable-installed from another location ($verl_location); reinstalling from this repo"
    log "installing verl (editable, no extras)"
    pip install -e ./verl
fi

log "done. Next steps:"
log "  - verify DFlash serving baseline (e.g. vllm-ascend test_dflash, K=8)"
log "  - then: bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_vllm_npu.sh"
