# DFlash Benchmark Usage

This document covers the two benchmark entrypoints in this directory:

- `run_benchmark.sh`: offline multi-GPU evaluation through `torchrun + benchmark.py`.
- `sglang_run_bench.sh`: YAML-driven SGLang server benchmark launcher built around `benchmark_sglang.py`.

## 1. `run_benchmark.sh`

`run_benchmark.sh` runs `benchmark.py` with one or more draft models and benchmark tasks.

Minimal example:

```bash
cd diffusion/dflash

TARGET_MODEL_PATH=/path/to/target/model \
DRAFT_MODELS_CSV=/path/to/draft_a,/path/to/draft_b \
bash run_benchmark.sh
```

Required environment variables:

- `TARGET_MODEL_PATH`: target model path or Hugging Face model name.
- `DRAFT_MODELS_CSV`: comma-separated draft model paths.

Common optional environment variables:

- `LOG_DIR`: output log directory. Default: `logs/draft-opd-eval`.
- `TASKS_CSV`: comma-separated task list. Default: `gsm8k:128,aime24:30,math500:128,mbpp:128,humaneval:164,mt-bench:80,mgsm_zh:32`.
- `MAX_NEW_TOKENS`: maximum generated tokens per prompt. Default: `8192`.
- `BLOCK_SIZE`: speculative block size. Default: `16`.
- `TEMPERATURE`: sampling temperature. Default: `0.0`.
- `CUDA_VISIBLE_DEVICES`: GPU list. If unset, set `NUM_GPU` manually or the script uses `NUM_GPU=1`.
- `CONDA_ACTIVATE`: optional path to a conda activation script.
- `CONDA_ENV_NAME`: conda environment name if `CONDA_ACTIVATE` is set. Default: `python312`.
- `HF_HOME`: optional Hugging Face cache path.

Example with more explicit settings:

```bash
cd diffusion/dflash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_HOME=/path/to/hf_cache \
TARGET_MODEL_PATH=Qwen/Qwen3-4B \
DRAFT_MODELS_CSV=../../verl/checkpoints/verl-dflash-opd/YOUR_RUN/global_step_5000/draft_model \
TASKS_CSV=gsm8k:128,math500:128,mbpp:128 \
TEMPERATURE=0.0 \
bash run_benchmark.sh
```

The script forwards these core arguments to `benchmark.py`:

- `--dataset`
- `--max-samples`
- `--model-name-or-path`
- `--draft-name-or-path`
- `--max-new-tokens`
- `--block-size`
- `--temperature`
- `--skip-base`
- `--enable-thinking` when `ENABLE_THINKING=true`
- `--case` when `PRINT_CASE=true`

## 2. `sglang_run_bench.sh`

`sglang_run_bench.sh` launches benchmark jobs from a YAML config.

Minimal example:

```bash
cd diffusion/dflash
bash sglang_run_bench.sh --config-yaml eval_config/two_gpu_example.yaml
```

If no config is provided, it uses:

```bash
eval_config/four_gpu_final.yaml
```

Dry-run example:

```bash
bash sglang_run_bench.sh --config-yaml eval_config/two_gpu_example.yaml --dry-run
```

Useful environment variables:

- `HF_HOME`: optional Hugging Face cache path.
- `CONDA_ACTIVATE`: optional path to a conda activation script.
- `CONDA_ENV_NAME`: conda environment name if `CONDA_ACTIVATE` is set. Default: `dflash-sglang-eval`.
- `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`: default `1`.
- `FLASHINFER_DISABLE_VERSION_CHECK`: default `1`.

## 3. YAML Config Format

The YAML config should contain a `jobs` list:

```yaml
jobs:
  - name: gpu0
    gpu: 0
    target_model: /path/to/target/model
    draft_model: /path/to/draft/model
    output_md: sglang-logs-final/gpu0.md
    dataset_name: ["aime24:30", "math500:128", "gsm8k:128"]
    concurrency_num: "1,2,4"
    enable_think: true
    attention_backends: fa3
    max_new_tokens: 8192
```

Important fields:

- `gpu`: GPU id or comma-separated GPU ids used by this job.
- `target_model`: target model path or Hugging Face model name.
- `draft_model`: draft model path. Leave empty to run baseline only.
- `output_md`: markdown report path.
- `dataset_name`: one dataset string or a list of dataset strings. `bench:num` limits sample count, for example `math500:128`.
- `concurrency_num`: one concurrency value or comma-separated values.
- `enable_think`: whether to enable thinking prompt formatting.
- `attention_backends`: SGLang attention backend, for example `fa3`.
- `max_new_tokens`: maximum generated tokens per request.

Example configs are available in:

```bash
eval_config/
```

