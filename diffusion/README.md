# Draft-OPD Evaluation

This folder contains utilities for evaluating Draft-OPD models.

The main evaluation workflow lives in:

```bash
./dflash
```

Useful entrypoints:

- `./dflash/sglang_run_bench.sh`: launch benchmark jobs from a YAML config.
- `./dflash/launch_sglang_bench_jobs.py`: YAML-driven multi-job launcher.
- `./dflash/eval_config/`: example benchmark configs.

The following files are small local smoke-test helpers:

- `run-diffusion-server.sh`: start one local SGLang DFlash server.
- `sglang-metrics.py`: query an already-running SGLang server and compute simple throughput / acceptance metrics.
- `run-sglang-metric.sh`: run `sglang-metrics.py` against one or more local ports.

Before running any script, set local model, data, cache, and environment paths with environment variables or by editing the example configs. Public files intentionally avoid hard-coded machine-specific paths.
