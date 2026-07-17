# Draft-OPD: On-Policy Distillation for Speculative Draft Models

<p align="center">
  <img src="fig/overview-draft-opd.jpg" alt="Draft-OPD overview" width="92%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.29343"><img src="https://img.shields.io/badge/arXiv-2605.29343-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/collections/bingyang-lei/draft-opd"><img src="https://img.shields.io/badge/Hugging%20Face-Models-yellow.svg" alt="Hugging Face models"></a>
  <a href="https://github.com/bingyang-lei/Draft-OPD"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project page"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.29343">Paper</a> |
  <a href="https://huggingface.co/collections/bingyang-lei/draft-opd">Models</a> |
  <a href="https://github.com/bingyang-lei/Draft-OPD">Project Page</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#evaluation">Evaluation</a> |
  <a href="#acknowledgements">Acknowledgements</a> |
  <a href="#citation">Citation</a>
</p>

## News

- [July 2026] The training data for Draft-OPD has been released and is available at `/data`.
- [May 2026] Draft-OPD is available on arXiv: [2605.29343](https://arxiv.org/abs/2605.29343).
- [May 2026] Released Draft-OPD model checkpoints are available in the [Hugging Face collection](https://huggingface.co/collections/bingyang-lei/draft-opd).

## Introduction

Draft-OPD trains speculative draft models with on-policy target feedback. Instead of only learning from fixed target-generated trajectories, the drafter is supervised on draft-induced states exposed during speculative verification, including the positions where draft proposals are rejected.

This repository contains the public training and evaluation code for Draft-OPD. The main training stack is built on `verl` and `sglang-dflash`, while the evaluation utilities live under `diffusion/`.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `verl/` | Training code and the public OPD DFlash training entrypoint. |
| `sglang-dflash/` | DFlash / SGLang runtime code used by training and evaluation. |
| `vllm/` | vLLM submodule ([curnane-lab/vllm](https://github.com/curnane-lab/vllm), `domino_npu_v0.21.0`), frontend of the vLLM rollout path. |
| `vllm-ascend/` | vLLM-Ascend submodule ([curnane-lab/vllm-ascend](https://github.com/curnane-lab/vllm-ascend), `domino_npu_v0.21.0rc1`), provides the `dflash` speculative method on Ascend NPU. |
| `diffusion/` | Draft-OPD evaluation utilities, with the main benchmark workflow in `diffusion/dflash/`. |

## Training Entry Point

This repository provides a single public training entrypoint:

```bash
verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh
```

The script launches DFlash on-policy distillation through `verl`. It wraps `run_qwen_gsm8k.sh`, so run it from the repository root after installing the `verl` training environment.

## Install

Clone with submodules (the `vllm/` and `vllm-ascend/` directories are git submodules pinned to tested commits):

```bash
git clone --recursive git@github.com:curnane-lab/Draft-OPD.git
# or, if you already cloned without --recursive:
git submodule update --init --recursive
```

From the repository root, run:

```bash
bash install.sh
```

`install.sh` wraps `install_npu.sh`, which **auto-detects already-installed components and skips them** (`vllm` / `vllm-ascend` / `triton-ascend` / `sglang` / `verl`). In a pre-provisioned NPU container this only installs `verl`; in a bare environment it also installs the engines from the submodules. The SGLang engine is skipped by default (the vLLM path does not use it); opt in with `INSTALL_SGLANG_DFLASH=1`, and force engine reinstall with `FORCE_INSTALL_ENGINES=1`.

> **Note**: `verl` is always installed **without extras**. `verl[vllm]` pins `vllm<=0.12.0` and `verl[sglang]` pins `torch==2.9.1` — both break an NPU / vllm-ascend v0.21 stack.

### Ascend NPU notes

- The vLLM rollout path requires a matching CANN / torch / torch-npu stack for the pinned vllm-ascend version (see `vllm-ascend/docs/source/installation.md`), plus `triton-ascend` (already listed in `verl/requirements-npu.txt`). `install_npu.sh` checks and warns but never auto-installs torch/torch_npu.
- `install.sh` builds vLLM with `VLLM_TARGET_DEVICE=empty` when the engine is not already present; NPU kernels are provided by `vllm-ascend`.
- To run the NPU + vLLM variant of the OPD trainer, use `verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_vllm_npu.sh` (the SGLang reference path remains `run_qwen_gsm8k_forward-ins.sh`).

## Quick Start

Set your local model and data paths, then run:

```bash
cd /path/to/opd

MAIN_MODEL_PATH=/path/to/main/model \
DRAFT_MODEL_PATH=/path/to/draft/model \
TRAIN_JSONL=/path/to/train.jsonl \
bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh \
  "data.val_files=['/path/to/aime24.jsonl','/path/to/gsm8k.jsonl','/path/to/math500.jsonl','/path/to/mbpp.jsonl']"
```

Required paths:

- `MAIN_MODEL_PATH`: target/main model. For example, use [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B).
- `DRAFT_MODEL_PATH`: initialized draft model used for speculative decoding and training. For example, download [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) from Hugging Face. Alternatively, you can train your own draft model with [SpecForge](https://github.com/sgl-project/SpecForge).
- `TRAIN_JSONL`: training data in JSONL format.
- `data.val_files`: validation JSONL files, passed as a Hydra override.

Common optional overrides:

```bash
LR=3e-4 \
train_epochs=8 \
STUDENT_WORLD_SIZE=7 \
TEACHER_WORLD_SIZE=1 \
TRAIN_PROMPT_BSZ=21 \
PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
MAX_PROMPT=512 \
MAX_RESPONSE_LENGTH=4096 \
ENABLE_THINKING=False \
bash verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh
```

Useful DFlash-specific options:

- `DFLASH_LM_HEAD_CHUNK_SIZE`: LM-head chunk size, default `512`.
- `RANDOM_RESPONSE_ANCHOR_ENABLED`: enable random response-anchor ablation, default `False`.
- `RANDOM_RESPONSE_ANCHOR_SEED`: random-anchor seed, default `42`.
- `TEACHER_GPU_MEMORY_UTILIZATION`: teacher inference memory fraction, default `0.1`.

Checkpoints are saved under:

```bash
verl/checkpoints/verl-dflash-opd/
```

Use `verl/scripts/fsdp_to_dflash.sh` from the repository root to extract the draft model from saved actor weights.

## Evaluation

Draft-OPD evaluation utilities live under `diffusion/`, with the main benchmark workflow in `diffusion/dflash/`.

The `diffusion/dflash/` folder is adapted from an early version of [DFlash](https://github.com/z-lab/dflash). You can also directly use the DFlash repository to evaluate DFlash draft models.

See [diffusion/dflash/README.md](diffusion/dflash/README.md) for the DFlash evaluation entrypoints and links to the English / Chinese usage guides.

## Acknowledgements

We thank [DFlash](https://github.com/z-lab/dflash) and [EAGLE3](https://github.com/SafeAILab/EAGLE) for their inspiring work on speculative decoding and draft-model training. We also thank [SpecForge](https://github.com/sgl-project/SpecForge), [SGLang](https://github.com/sgl-project/sglang), and [verl](https://github.com/volcengine/verl) for the open-source infrastructure that this repository builds on.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{lei2026draftopdonpolicydistillationspeculative,
      title={Draft-OPD: On-Policy Distillation for Speculative Draft Models}, 
      author={Haodi Lei and Yafy Li and Haoran Zhang and Shunkai Zhang and Qianjia Cheng and Xiaoye Qu and Ganqu Cui and Bowen Zhou and Ning Ding and Yun Luo and Yu Cheng},
      year={2026},
      eprint={2605.29343},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.29343}, 
}
```
