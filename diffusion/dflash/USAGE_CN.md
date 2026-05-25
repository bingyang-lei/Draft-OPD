# DFlash 基准脚本使用说明（`run_benchmark.sh` 与 `sglang_run_bench.sh`）

本文档只覆盖 `./dflash` 目录下与基准测试直接相关的两个入口脚本：

- `run_benchmark.sh`：基于 `torchrun + benchmark.py` 的多卡离线评测入口。
- `sglang_run_bench.sh`：基于 `benchmark_sglang.py` + SGLang 服务端的吞吐/延迟评测入口。

---

## 1. `run_benchmark.sh` 用法

### 1.1 最小用法

```bash
cd dflash
bash run_benchmark.sh
```

脚本目前只支持一个命令行参数：

```bash
bash run_benchmark.sh --log-dir /path/to/logs
```

### 1.2 脚本参数说明

#### `--log-dir PATH`
- **作用**：指定日志输出目录。
- **默认值**：脚本内置 `log_dir="logs/verl-opd-mathcode-16k-8gpu"`。
- **行为**：脚本会 `mkdir -p "$log_dir"`，并按 draft 模型+thinking 模式生成日志文件。

> 除 `--log-dir` 外，其它运行配置都在脚本体中通过变量/数组写死（例如 `TASKS`、`DRAFT_MODELS`、`max-new-tokens` 等），需要手动改脚本。

### 1.3 脚本内部关键配置（非 CLI 参数）

这些不是 `run_benchmark.sh` 的命令行参数，但会显著影响行为：

- `CUDA_VISIBLE_DEVICES`
  - 若已设置：按逗号个数推断 `num_gpu`。
  - 若未设置：通过 `nvidia-smi -L` 自动统计 GPU 数。
- `TASKS`
  - 形如 `"数据集名:样本数"`，例如 `"gsm8k:128"`、`"math500:128"`。
  - 每个任务都会触发一次 `benchmark.py`。
- `DRAFT_MODELS`
  - draft 模型路径列表，脚本会逐个跑。
- `thinking_mode`
  - 当前脚本只跑 `"on"`，会映射到 `--enable-thinking`。
- `print_case`
  - 为 `true` 时会附加 `--case`，打印每条样本的生成内容。

### 1.4 `run_benchmark.sh` 最终传给 `benchmark.py` 的参数含义

脚本核心命令是：

```bash
torchrun --nproc_per_node="${num_gpu}" --master_port=29600 ./benchmark.py ...
```

其中参数语义如下：

- `--dataset`
  - 数据集名（来自 `TASKS` 的冒号前半部分）。
- `--max-samples`
  - 本次最多评测多少条样本（来自 `TASKS` 的冒号后半部分）。
- `--model-name-or-path`
  - 目标模型（target model）路径。
- `--draft-name-or-path`
  - DFlash draft 模型路径。
- `--max-new-tokens`
  - 单条请求最多生成 token 数（当前脚本设为 `8192`）。
- `--block-size`
  - speculative block 大小（当前脚本设为 `16`）。
- `--temperature`
  - 采样温度（当前脚本设为 `0.0`，近似贪心）。
- `--skip-base`
  - 跳过 baseline（仅 target）的评测。
- `--enable-thinking`
  - 启用 thinking prompt 模式（由 `thinking_mode` 控制）。
- `--case`（可选）
  - 输出逐样本内容（由 `print_case` 控制）。

> `benchmark.py` 还支持 `--save-acc-len PATH`（保存 acceptance length CSV），在 `run_benchmark.sh` 中是注释状态，可按需打开。

---

## 2. `sglang_run_bench.sh` 用法

### 2.1 最小用法

```bash
cd dflash
bash sglang_run_bench.sh
```

该脚本通过 shell 变量拼装 `python benchmark_sglang.py ...` 命令，默认会：

- 跳过 baseline（`SKIP_BASE="--skip-base"`）
- 开启 thinking（`ENABLE_THINK="--enable-think"`）
- 保存 case 到 JSONL（`OUTPUT_CASE=...` 非空时）

### 2.2 脚本中可直接改的变量

- `SKIP_BASE`
  - `"--skip-base"`：仅跑 DFLASH。
  - `""`（空字符串）：同时跑 baseline + DFLASH。
- `OUTPUT_CASE`
  - 非空时会展开为 `--output-case <path>`，把 completions 追加写入 JSONL。
  - 为空时不输出 case 文件。
- `ENABLE_THINK`
  - `"--enable-think"`：构造 prompt 时启用 thinking。
  - `""`：关闭 thinking prompt。

---

## 3. `sglang_run_bench.sh` 中 `benchmark_sglang.py` 参数详解

以下参数是 `sglang_run_bench.sh` 当前实际传入的核心参数，同时也可在你手工调用 `benchmark_sglang.py` 时复用。

- `--target-model`
  - target 模型路径或 HuggingFace 名称。
- `--draft-model`
  - DFlash draft 模型路径或名称。
- `--concurrencies`
  - 并发度列表（逗号分隔）。
  - 例如 `1,2,4` 会分别在不同并发下压测。
- `--dataset-name`
  - 数据集规格，支持：
    - 单数据集：`math500`
    - 显式计数：`math500:7`
    - 多数据集：`math500:10 aime25:5 gsm8k:32`
- `--attention-backends`
  - 候选后端列表（逗号分隔），如 `flashinfer,fa3,fa4`。
  - 脚本会按 GPU 架构自动过滤（如非 SM90 会去掉 fa3，<SM100 会去掉 fa4）。
- `--tp-size`
  - Tensor Parallel 大小。
- `--max-new-tokens`
  - 最大生成 token 数。
- `--skip-base` / `--skip-baseline`
  - 跳过 baseline，只报告 DFLASH（baseline/speedup 列会是 N/A）。
- `--enable-think`
  - 在 `tokenizer.apply_chat_template` 时启用 thinking。
- `--mamba-scheduler-strategy`
  - 透传给 sglang server，例如 `extra_buffer`。
- `--output-case PATH`
  - 输出（追加）每次测量的 completion JSONL。
- `--output-md`
  - 输出 markdown 报告文件路径。

### 3.1 其他常用但脚本里未显式设置的参数

`benchmark_sglang.py` 还支持下列常用调参项：

- `--batch-requests`
  - 使用服务端批量 `/generate`，而不是客户端并发请求。
- `--timeout-s`
  - 单请求超时秒数（默认 3600）。
- `--mem-fraction-static`
  - SGLang 静态显存占用比例（默认 0.75）。
- `--disable-radix-cache`
  - 关闭 radix cache。
- `--dtype`
  - 推理 dtype（默认 `bfloat16`）。
- `--max-running-requests`
  - 服务端最大并行请求数。
- `--questions-per-concurrency-base`
  - 每个并发度的样本基数，实际样本数约为 `base * concurrency`。
- `--max-questions-per-config`
  - 每个配置（backend+并发）样本上限。

---

## 4. 两个脚本怎么选

- 想验证 `benchmark.py` 逻辑、跑多卡 `torchrun`、关注 DFlash 内部 acceptance 行为：用 `run_benchmark.sh`。
- 想基于 SGLang 服务看并发吞吐、端到端延迟、不同 attention backend 对比：用 `sglang_run_bench.sh`。


# 总结

主要就是，运行diffusion/dflash/run_benchmark.sh，和diffusion/dflash/run_benchmark-8b.sh，测transformer上的推理速度。

运行diffusion/dflash/sglang_run_bench.sh，测sglang上的推理速度。

主要需要调的参数其实只有温度一个，因为我们都是测推理模型，都是思考模式，长度也都是8192.