from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    raise RuntimeError(f"Invalid boolean value: {value!r}")


def _as_list(value: Any, *, field_name: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        return shlex.split(value)
    if value is None:
        raise RuntimeError(f"Missing required field: {field_name}")
    return [str(value)]


def _as_concurrencies(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if value is None:
        raise RuntimeError("Missing required field: concurrency_num")
    return str(value)


def _as_gpu(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if value is None:
        raise RuntimeError("Missing required field: gpu")
    text = str(value).strip()
    if text.lower().startswith("gpu"):
        text = text[3:]
    if not text:
        raise RuntimeError("Missing required field: gpu")
    return text


def _get_field(job: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in job:
            return job[name]
    return default


def _expand_string(value: str, variables: dict[str, str]) -> str:
    def replace_var(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise RuntimeError(f"Unknown YAML variable: {name}")
        return variables[name]

    return VAR_RE.sub(replace_var, value)


def _expand_value(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _expand_string(value, variables)
    if isinstance(value, list):
        return [_expand_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item, variables) for key, item in value.items()}
    return value


def _load_variables(data: Any) -> dict[str, str]:
    if not isinstance(data, dict):
        return {}
    raw_vars = data.get("vars", {})
    if raw_vars is None:
        return {}
    if not isinstance(raw_vars, dict):
        raise RuntimeError("YAML `vars` must be a mapping.")
    return {str(key): str(value) for key, value in raw_vars.items()}


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    variables = _load_variables(data)
    if isinstance(data, dict) and "jobs" in data:
        jobs = data["jobs"]
    elif isinstance(data, list):
        jobs = data
    elif isinstance(data, dict):
        jobs = []
        for key, value in data.items():
            if not str(key).lower().startswith("gpu"):
                continue
            if not isinstance(value, dict):
                raise RuntimeError(
                    "Top-level gpu keys must map to a job object. "
                    "Prefer `jobs: [{gpu: 0, ...}]`."
                )
            job = dict(value)
            job.setdefault("gpu", str(key)[3:])
            jobs.append(job)
    else:
        raise RuntimeError("YAML must contain a `jobs:` list.")

    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("YAML must contain at least one job.")
    normalized: list[dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise RuntimeError(f"jobs[{idx}] must be a mapping.")
        normalized.append(_expand_value(dict(job), variables))
    return normalized


def _infer_speculative_algorithm(draft_model: str, job: dict[str, Any]) -> str:
    algorithm = _get_field(job, "speculative_algorithm", default=None)
    if not _is_empty(algorithm):
        return str(algorithm).strip().upper()
    if "eagle3" in draft_model.lower():
        return "EAGLE3"
    return "DFLASH"


def _append_optional_value(
    cmd: list[str],
    job: dict[str, Any],
    field_name: str,
    flag: str,
) -> None:
    value = _get_field(job, field_name, default=None)
    if not _is_empty(value):
        cmd.extend([flag, str(value)])


def _append_optional_flag(
    cmd: list[str],
    job: dict[str, Any],
    field_name: str,
    flag: str,
) -> None:
    if _as_bool(_get_field(job, field_name, default=False)):
        cmd.append(flag)


def _append_speculative_value(cmd: list[str], field_name: str, value: Any) -> None:
    flag = "--" + field_name.replace("_", "-")
    _append_optional_value(cmd, {field_name: value}, field_name, flag)


def _build_command(
    *,
    job: dict[str, Any],
    benchmark_script: Path,
    python_executable: str,
) -> tuple[list[str], str, Path]:
    gpu = _as_gpu(_get_field(job, "gpu", default=None))
    target_model = _get_field(job, "target_model", "target", default=None)
    output_md = _get_field(job, "output_md", "output", default=None)
    dataset_name = _get_field(job, "dataset_name", "dataset", "datasets", default=None)
    concurrency_num = _get_field(
        job,
        "concurrency_num",
        "concurrencies",
        "concurrency",
        default=None,
    )
    if _is_empty(target_model):
        raise RuntimeError("Missing required field: target_model")
    if _is_empty(output_md):
        raise RuntimeError("Missing required field: output_md")

    cmd = [
        python_executable,
        str(benchmark_script),
        "--target-model",
        str(target_model),
        "--output-md",
        str(output_md),
        "--dataset-name",
        *_as_list(dataset_name, field_name="dataset_name"),
        "--concurrencies",
        _as_concurrencies(concurrency_num),
        "--tp-size",
        str(_get_field(job, "tp_size", "tp", default=1)),
        "--mem-fraction-static",
        str(_get_field(job, "mem_fraction_static", default=0.75)),
    ]

    for field_name, flag in [
        ("max_new_tokens", "--max-new-tokens"),
        ("temp", "--temp"),
        ("timeout_s", "--timeout-s"),
        ("dtype", "--dtype"),
        ("max_running_requests", "--max-running-requests"),
        ("questions_per_concurrency_base", "--questions-per-concurrency-base"),
        ("max_questions_per_config", "--max-questions-per-config"),
        ("attention_backends", "--attention-backends"),
        ("mamba_scheduler_strategy", "--mamba-scheduler-strategy"),
        ("output_case", "--output-case"),
    ]:
        _append_optional_value(cmd, job, field_name, flag)

    for field_name, flag in [
        ("enable_think", "--enable-think"),
        ("batch_requests", "--batch-requests"),
        ("disable_radix_cache", "--disable-radix-cache"),
    ]:
        _append_optional_flag(cmd, job, field_name, flag)

    draft_model = _get_field(job, "draft_model", "draft", default=None)
    only_baseline = _as_bool(_get_field(job, "only_baseline", default=False))
    if _is_empty(draft_model) or only_baseline:
        cmd.append("--only-baseline")
    else:
        draft_model = str(draft_model).strip()
        algorithm = _infer_speculative_algorithm(draft_model, job)
        cmd.extend(
            [
                "--draft-model",
                draft_model,
                "--speculative-algorithm",
                algorithm,
            ]
        )
        if _as_bool(_get_field(job, "add_base", default=False)):
            cmd.append("--add-base")

        eagle_defaults = {
            "speculative_num_steps": 15,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 16,
        }
        if algorithm == "EAGLE3":
            for field_name, default_value in eagle_defaults.items():
                value = _get_field(job, field_name, default=default_value)
                _append_speculative_value(cmd, field_name, value)
        else:
            for field_name in eagle_defaults:
                value = _get_field(job, field_name, default=None)
                if not _is_empty(value):
                    _append_speculative_value(cmd, field_name, value)

    log_path = Path(str(output_md) + ".run.log")
    return cmd, gpu, log_path


def _print_dry_run(job_id: str, gpu: str, log_path: Path, cmd: list[str]) -> None:
    print(f"[dry-run] {job_id}")
    print(f"  CUDA_VISIBLE_DEVICES={gpu}")
    print(f"  log={log_path}")
    print(f"  cmd={shlex.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch YAML-defined SGLang benchmark jobs.")
    parser.add_argument("--config-yaml", "--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_sglang.py",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = _load_jobs(args.config_yaml)
    launched: list[tuple[str, subprocess.Popen, Any, Path]] = []

    for idx, job in enumerate(jobs):
        job_id = str(_get_field(job, "name", "id", default=f"job{idx}"))
        cmd, gpu, log_path = _build_command(
            job=job,
            benchmark_script=args.benchmark_script,
            python_executable=str(args.python),
        )
        if args.dry_run:
            _print_dry_run(job_id, gpu, log_path, cmd)
            continue

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"[launch] {job_id}: CUDA_VISIBLE_DEVICES={gpu} log={log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        launched.append((job_id, proc, log_f, log_path))

    if args.dry_run:
        return 0

    failures: list[tuple[str, int, Path]] = []
    try:
        for job_id, proc, log_f, log_path in launched:
            return_code = proc.wait()
            log_f.close()
            if return_code != 0:
                failures.append((job_id, return_code, log_path))
            print(f"[done] {job_id}: returncode={return_code} log={log_path}")
    except KeyboardInterrupt:
        for _, proc, log_f, _ in launched:
            if proc.poll() is None:
                proc.terminate()
            log_f.close()
        raise

    if failures:
        print("[error] Some benchmark jobs failed:", file=sys.stderr)
        for job_id, return_code, log_path in failures:
            print(f"  {job_id}: returncode={return_code} log={log_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
