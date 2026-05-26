# OPD DFlash Training

This repository provides a single public training entrypoint:

```bash
verl/examples/on_policy_distillation_trainer/run_qwen_gsm8k_forward-ins.sh
```

The script launches DFlash on-policy distillation through `verl`. It wraps `run_qwen_gsm8k.sh`, so run it from the repository root after installing the `verl` training environment.

## Install

From the repository root, run:

```bash
bash install.sh
```

This installs the editable `sglang-dflash` and `verl` packages and their dependencies. No other manual setup is required.

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
- `DRAFT_MODEL_PATH`: initialized draft model used for speculative decoding and training. For example, download [`z-lab/Qwen3-4B-DFlash-b16`](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16) from Hugging Face.
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

## Evaluation

Draft-OPD evaluation utilities live under `diffusion/`, with the main benchmark workflow in `diffusion/dflash/`.

See [diffusion/dflash/README.md](diffusion/dflash/README.md) for the DFlash evaluation entrypoints and links to the English / Chinese usage guides.


