from __future__ import annotations

import argparse
import json
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from transformers import AutoTokenizer
from model import load_and_process_dataset

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


def _build_sampling_params(max_new_tokens: int, temp: float) -> dict:
    if temp == 0.0:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": int(max_new_tokens),
        }
    return {
        "temperature": float(temp),
        "top_p": 0.95,
        "top_k": 20,
        "max_new_tokens": int(max_new_tokens),
    }


def _is_blackwell() -> bool:
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    temp: float,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params = _build_sampling_params(max_new_tokens, temp)
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


def _extract_completion_text(out: dict) -> str:
    """Best-effort parse of /generate JSON completion text."""
    if not isinstance(out, dict):
        return ""
    t = out.get("text")
    if isinstance(t, str):
        return t
    return ""


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temp: float,
    stop: list[str],
    timeout_s: int,
) -> list[dict]:
    if not prompts:
        return []
    sampling_params = _build_sampling_params(max_new_tokens, temp)
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": sampling_params,
        },
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    out = resp.json()
    if not isinstance(out, list):
        raise RuntimeError(
            "Expected a list response for batched /generate, but got "
            f"type={type(out).__name__}."
        )
    return out


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


@dataclass(frozen=True)
class DatasetPromptGroup:
    name: str
    requested_count: Optional[int]
    prompts: list[str]

    @property
    def spec_text(self) -> str:
        return (
            f"{self.name}:{self.requested_count}"
            if self.requested_count is not None
            else self.name
        )


def _run_bench_requests(
    base_url: str,
    *,
    prompts: list[str],
    max_new_tokens: int,
    temp: float,
    concurrency: int,
    batch_requests: bool,
    stop: list[str],
    timeout_s: int,
    expect_speculative: bool,
    output_case_path: Optional[str] = None,
    output_case_label: Optional[str] = None,
) -> BenchMetrics:
    # Drop the first batch from metrics to exclude one-time JIT/cuda-graph overhead
    bs = max(int(concurrency), 1)
    if len(prompts) > bs:
        warmup_prompts = prompts[:bs]
        if batch_requests:
            _send_generate_batch(
                base_url,
                warmup_prompts,
                max_new_tokens=max_new_tokens,
                temp=temp,
                stop=stop,
                timeout_s=timeout_s,
            )
        else:
            with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
                futures = [
                    pool.submit(
                        _send_generate,
                        base_url,
                        prompt,
                        max_new_tokens=max_new_tokens,
                        temp=temp,
                        stop=stop,
                        timeout_s=timeout_s,
                    )
                    for prompt in warmup_prompts
                ]
                for fut in as_completed(futures):
                    fut.result()

        prompts = prompts[bs:]

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []

    completions_ordered: list[str] = []

    if batch_requests:
        bs = max(int(concurrency), 1)
        for start_idx in range(0, len(prompts), bs):
            chunk_prompts = prompts[start_idx : start_idx + bs]
            outs = _send_generate_batch(
                base_url,
                chunk_prompts,
                max_new_tokens=max_new_tokens,
                temp=temp,
                stop=stop,
                timeout_s=timeout_s,
            )
            if len(outs) != len(chunk_prompts):
                raise RuntimeError(
                    "Batched /generate output length mismatch: "
                    f"got {len(outs)} outputs for {len(chunk_prompts)} prompts."
                )

            for j, out in enumerate(outs):
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
                completions_ordered.append(_extract_completion_text(out))
    else:
        with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
            futures = {
                pool.submit(
                    _send_generate,
                    base_url,
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temp=temp,
                    stop=stop,
                    timeout_s=timeout_s,
                ): i
                for i, prompt in enumerate(prompts)
            }
            ordered_outs: list[Optional[dict]] = [None] * len(prompts)
            for fut in as_completed(futures):
                i = futures[fut]
                ordered_outs[i] = fut.result()
            for out in ordered_outs:
                if out is None:
                    raise RuntimeError("Missing /generate response for a prompt index.")
                meta = out.get("meta_info", {}) or {}
                total_tokens += int(meta.get("completion_tokens", 0))
                spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
                if "spec_accept_length" in meta:
                    try:
                        spec_accept_lengths.append(float(meta["spec_accept_length"]))
                    except (TypeError, ValueError):
                        pass
                completions_ordered.append(_extract_completion_text(out))

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_speculative and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "Speculative decoding sanity check failed: did not observe any "
            "`spec_verify_ct` in responses (speculative decoding may not have been enabled)."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    if output_case_path and output_case_label and completions_ordered:
        path = Path(output_case_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"type": "section", "label": output_case_label},
                    ensure_ascii=False,
                )
                + "\n"
            )
            for i, (prm, comp) in enumerate(zip(prompts, completions_ordered)):
                rec = {"idx": i, "prompt": prm, "completion": comp}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    concurrencies: list[int],
    values: dict[int, Optional[float]],
    float_fmt: str,
) -> str:
    header = ["conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    row = ["value"]
    for c in concurrencies:
        v = values.get(c, None)
        row.append("N/A" if v is None else format(v, float_fmt))
    lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _parse_dataset_specs(raw_specs: list[str]) -> tuple[list[tuple[str, Optional[int]]], bool]:
    specs: list[tuple[str, Optional[int]]] = []
    has_explicit_count = False
    for raw in raw_specs:
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            name, count_text = text.split(":", 1)
            name = name.strip()
            count_text = count_text.strip()
            if not name:
                raise RuntimeError(f"Invalid dataset spec: {raw!r}")
            try:
                count = int(count_text)
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid dataset count in spec {raw!r}; expected name:count."
                ) from e
            if count <= 0:
                raise RuntimeError(f"Dataset count must be positive in spec {raw!r}.")
            specs.append((name, count))
            has_explicit_count = True
        else:
            specs.append((text, None))
    if not specs:
        raise RuntimeError("No dataset specs provided.")

    explicit_mode = has_explicit_count or len(specs) > 1
    if explicit_mode:
        missing = [name for name, count in specs if count is None]
        if missing:
            raise RuntimeError(
                "When passing multiple datasets or explicit counts, every dataset must "
                "use the `name:count` form. Missing counts for: "
                + ", ".join(missing)
            )
    return specs, explicit_mode


def _build_prompt_groups(
    *,
    dataset_specs: list[tuple[str, Optional[int]]],
    explicit_mode: bool,
    tokenizer: AutoTokenizer,
    enable_think: bool,
    max_questions: int,
) -> tuple[list[DatasetPromptGroup], str]:
    prompt_groups: list[DatasetPromptGroup] = []
    dataset_spec_text = " ".join(
        f"{name}:{count}" if count is not None else name for name, count in dataset_specs
    )

    if explicit_mode:
        for dataset_name, requested_count in dataset_specs:
            dataset = load_and_process_dataset(dataset_name)
            assert requested_count is not None
            prompts: list[str] = []
            actual_count = min(int(requested_count), len(dataset))
            if actual_count < int(requested_count):
                print(
                    f"Warning: dataset {dataset_name} only has {len(dataset)} items; "
                    f"requested {requested_count}, using {actual_count}."
                )
            print(
                f"Loading dataset: {dataset_name} "
                f"(requested={requested_count}, used={actual_count})..."
            )
            for i in range(actual_count):
                item = dataset[i]
                user_content = item["turns"][0]
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_think,
                )
                prompts.append(prompt_text)

            if not prompts:
                raise RuntimeError(f"No prompts available for dataset {dataset_name}.")
            prompt_groups.append(
                DatasetPromptGroup(
                    name=dataset_name,
                    requested_count=requested_count,
                    prompts=prompts,
                )
            )

        if not prompt_groups:
            raise RuntimeError("No prompts available after applying dataset counts.")
    else:
        dataset_name = dataset_specs[0][0]
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_and_process_dataset(dataset_name)
        prompts: list[str] = []
        if len(dataset) < max_questions:
            print(
                f"Warning: Dataset has {len(dataset)} items, but need up to {max_questions}. "
                "Reusing items."
            )
        for i in range(max_questions):
            item = dataset[i % len(dataset)]
            user_content = item["turns"][0]
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_think,
            )
            prompts.append(prompt_text)

        if not prompts:
            raise RuntimeError(f"No prompts available for dataset {dataset_name}.")
        prompt_groups.append(
            DatasetPromptGroup(
                name=dataset_name,
                requested_count=None,
                prompts=prompts,
            )
        )

    return prompt_groups, dataset_spec_text


def _build_warmup_prompts(prompts: list[str], max_concurrency: int) -> list[str]:
    return [prompts[i % len(prompts)] for i in range(max(max_concurrency, 1))]


def _build_speculative_server_args(args: argparse.Namespace) -> list[str]:
    if not args.draft_model:
        raise RuntimeError("A draft model is required for speculative decoding.")
    spec_args = [
        "--speculative-algorithm",
        str(args.speculative_algorithm).upper(),
        "--speculative-draft-model-path",
        args.draft_model,
    ]
    optional_args = [
        ("--speculative-num-steps", args.speculative_num_steps),
        ("--speculative-eagle-topk", args.speculative_eagle_topk),
        ("--speculative-num-draft-tokens", args.speculative_num_draft_tokens),
    ]
    for flag, value in optional_args:
        if value is not None:
            spec_args.extend([flag, str(value)])
    return spec_args


def _write_markdown_lines(path: str, lines: list[str], mode: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _build_markdown_header(
    *,
    args: argparse.Namespace,
    dataset_spec_text: str,
    sampling_params_for_report: dict,
    attention_backends: list[str],
    tp: int,
    concurrencies: list[int],
    explicit_dataset_mode: bool,
    prompt_groups: list[DatasetPromptGroup],
    max_questions: int,
    device_sm: int,
    is_blackwell: bool,
) -> list[str]:
    spec_label = str(args.speculative_algorithm).upper()
    md_lines: list[str] = []
    title = (
        "Baseline Bench Report"
        if not args.run_speculative
        else f"{spec_label} Bench Report"
    )
    md_lines.append(f"# {title}")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- mode: `{args.bench_mode}`")
    md_lines.append(f"- dataset: `{dataset_spec_text}`")
    md_lines.append(f"- enable_think: `{bool(args.enable_think)}`")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(
        f"- draft_model: `{args.draft_model}`" if args.draft_model else "- draft_model: `(none)`"
    )
    if args.run_speculative:
        md_lines.append(f"- speculative_algorithm: `{spec_label}`")
        if args.speculative_num_steps is not None:
            md_lines.append(f"- speculative_num_steps: `{args.speculative_num_steps}`")
        if args.speculative_eagle_topk is not None:
            md_lines.append(f"- speculative_eagle_topk: `{args.speculative_eagle_topk}`")
        if args.speculative_num_draft_tokens is not None:
            md_lines.append(
                f"- speculative_num_draft_tokens: `{args.speculative_num_draft_tokens}`"
            )
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- temp: `{args.temp}`")
    md_lines.append(f"- top_p: `{sampling_params_for_report['top_p']}`")
    md_lines.append(f"- top_k: `{sampling_params_for_report['top_k']}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(
        f"- mamba_scheduler_strategy: `{args.mamba_scheduler_strategy}`"
        if args.mamba_scheduler_strategy
        else "- mamba_scheduler_strategy: `(server default)`"
    )
    md_lines.append(f"- tp_size: `{tp}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- questions_per_concurrency: `base={args.questions_per_concurrency_base}`")
    if explicit_dataset_mode:
        md_lines.append("- measured_prompts_per_concurrency: `per dataset spec`")
        md_lines.append(
            "- dataset_prompt_counts: `"
            + ", ".join(f"{group.spec_text}={len(group.prompts)}" for group in prompt_groups)
            + "`"
        )
    else:
        md_lines.append(
            f"- measured_prompts_per_concurrency: `varies with concurrency up to {max_questions}`"
        )
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append(f"- is_blackwell: `{is_blackwell}`")
    md_lines.append(f"- run_baseline: `{bool(args.run_baseline)}`")
    md_lines.append(f"- run_speculative: `{bool(args.run_speculative)}`")
    md_lines.append(
        f"- output_case: `{args.output_case}`" if args.output_case else "- output_case: `(disabled)`"
    )
    md_lines.append("- drop_first_batch: `true`")
    md_lines.append("")
    return md_lines


def _build_dataset_markdown_section(
    *,
    args: argparse.Namespace,
    backend: str,
    group: DatasetPromptGroup,
    concurrencies: list[int],
    baseline_toks: dict[tuple[str, str, int], Optional[float]],
    dflash_toks: dict[tuple[str, str, int], Optional[float]],
    dflash_accept_len: dict[tuple[str, str, int], Optional[float]],
) -> list[str]:
    spec_label = str(args.speculative_algorithm).upper()
    md_lines: list[str] = []
    md_lines.append(f"### Dataset: `{group.spec_text}`")
    md_lines.append("")
    md_lines.append(f"- measured_prompts: `{len(group.prompts)}`")
    md_lines.append("")

    baseline_values = {
        c: baseline_toks.get((backend, group.spec_text, c), None)
        for c in concurrencies
    }
    dflash_values = {
        c: dflash_toks.get((backend, group.spec_text, c), None)
        for c in concurrencies
    }
    speedup_values: dict[int, Optional[float]] = {}
    for c in concurrencies:
        b = baseline_values.get(c, None)
        d = dflash_values.get(c, None)
        speedup_values[c] = None if (b is None or d is None or b <= 0) else (d / b)

    if args.run_baseline:
        md_lines.append("#### Baseline output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=baseline_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

    if args.run_speculative:
        md_lines.append(f"#### {spec_label} output tok/s")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=dflash_values,
                float_fmt=",.2f",
            )
        )
        md_lines.append("")

    if args.run_baseline and args.run_speculative:
        md_lines.append(f"#### Speedup ({spec_label} / baseline)")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values=speedup_values,
                float_fmt=".3f",
            )
        )
        md_lines.append("")

    if args.run_speculative:
        md_lines.append(f"#### {spec_label} acceptance length")
        md_lines.append(
            _format_table(
                concurrencies=concurrencies,
                values={
                    c: dflash_accept_len.get((backend, group.spec_text, c), None)
                    for c in concurrencies
                },
                float_fmt=".3f",
            )
        )
        md_lines.append("")
    return md_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-md",
        type=str,
        default=None,
        help="Write a markdown report to this file (disabled by default).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        nargs="+",
        default=["gsm8k"],
        help=(
            "One or more dataset specs. Supports legacy single dataset usage like `math500`, "
            "or explicit multi-dataset specs like `math500:10 aime25:5 gsm8k:32`."
        ),
    )
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        default="DFLASH",
        help="Speculative decoding algorithm to pass to the SGLang server.",
    )
    parser.add_argument(
        "--speculative-num-steps",
        type=int,
        default=None,
        help="Optional SGLang --speculative-num-steps value.",
    )
    parser.add_argument(
        "--speculative-eagle-topk",
        type=int,
        default=None,
        help="Optional SGLang --speculative-eagle-topk value.",
    )
    parser.add_argument(
        "--speculative-num-draft-tokens",
        type=int,
        default=None,
        help="Optional SGLang --speculative-num-draft-tokens value.",
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Run target-model baseline only. If --draft-model is omitted, this mode is selected automatically.",
    )
    parser.add_argument(
        "--add-base",
        action="store_true",
        help="When --draft-model is set, also run the target-model baseline and report speedup.",
    )
    parser.add_argument(
        "--batch-requests",
        action="store_true",
        help="Send prompts as server-side batched /generate requests (batch size = concurrency) instead of client-side concurrent requests.",
    )
    parser.add_argument(
        "--enable-think",
        action="store_true",
        help="Pass enable_thinking=True to tokenizer.apply_chat_template when building prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature. 0 keeps deterministic settings "
            "(top_p=1.0, top_k=1); non-zero values use recommended sampling settings "
            "(top_p=0.95, top_k=20)."
        ),
    )
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size (single value, no sweep).",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of client concurrency levels.",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
        help="num_questions = base * concurrency (default matches the sweep plan).",
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
        help="Cap num_questions per (tp, concurrency) run (default: 1024).",
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3,fa4",
        help="Comma-separated list. Will auto-skip fa3 unless SM90 (Hopper), and fa4 unless SM100+ (Blackwell).",
    )
    parser.add_argument(
        "--mamba-scheduler-strategy",
        type=str,
        default=None,
        help=(
            "If set, forwarded to sglang server as "
            "`--mamba-scheduler-strategy <value>` (e.g. `extra_buffer`). "
            "Omit to leave server defaults unchanged."
        ),
    )
    parser.add_argument(
        "--output-case",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "If set, append speculative-decoding measured completions (JSON lines: idx, prompt, completion) "
            "to this path for each (backend, concurrency) run."
        ),
    )
    args = parser.parse_args()
    if args.draft_model is not None:
        args.draft_model = str(args.draft_model).strip()
        if args.draft_model.lower() in {"", "none", "null"}:
            args.draft_model = None
    if args.only_baseline and args.draft_model:
        print("Warning: --only-baseline was set, so --draft-model will be ignored.")
        args.draft_model = None
    if args.add_base and not args.draft_model:
        print("Warning: --add-base was set without --draft-model; running baseline only.")

    args.run_speculative = bool(args.draft_model) and not bool(args.only_baseline)
    args.run_baseline = bool(args.only_baseline) or bool(args.add_base) or not args.run_speculative
    if args.run_baseline and args.run_speculative:
        args.bench_mode = "speculative_with_baseline"
    elif args.run_speculative:
        args.bench_mode = "speculative_only"
    else:
        args.bench_mode = "baseline_only"
    spec_label = str(args.speculative_algorithm).upper()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    sampling_params_for_report = _build_sampling_params(
        int(args.max_new_tokens), float(args.temp)
    )
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    requested_num_questions_by_conc = {
        c: min(int(args.questions_per_concurrency_base) * int(c), int(args.max_questions_per_config))
        for c in concurrencies
    }
    max_questions = max(requested_num_questions_by_conc.values())
    max_concurrency = max(concurrencies)

    attention_backends = [s.strip() for s in args.attention_backends.split(",") if s.strip()]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if device_sm != 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    if device_sm < 100:
        attention_backends = [b for b in attention_backends if b != "fa4"]
    attention_backends = attention_backends or ["flashinfer"]
    print(f"attention_backends: {attention_backends}")

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    dataset_specs, explicit_dataset_mode = _parse_dataset_specs(args.dataset_name)
    prompt_groups, dataset_spec_text = _build_prompt_groups(
        dataset_specs=dataset_specs,
        explicit_mode=explicit_dataset_mode,
        tokenizer=tokenizer,
        enable_think=bool(args.enable_think),
        max_questions=max_questions,
    )
    if explicit_dataset_mode:
        total_selected = sum(len(group.prompts) for group in prompt_groups)
        print(
            f"Using explicit dataset mixture: {dataset_spec_text} "
            f"(total measured prompts={total_selected})"
        )
    else:
        dataset_name = dataset_specs[0][0]
        print(
            f"Using legacy single-dataset mode: {dataset_name} "
            f"(max measured prompts across concurrencies={max_questions})"
        )

    if args.output_case and args.run_speculative:
        Path(args.output_case).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_case, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "kind": "benchmark_sglang_case_outputs",
                        "dataset": dataset_spec_text,
                        "enable_think": bool(args.enable_think),
                        "speculative_algorithm": spec_label,
                        "note": (
                            f"{spec_label} measured prompts only "
                            "(warmup batch excluded); JSONL records follow."
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Results indexed by (backend, dataset_spec, concurrency).
    # Removed TP dimension from keys since we aren't sweeping it.
    baseline_toks: dict[tuple[str, str, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, str, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, str, int], Optional[float]] = {}

    tp = args.tp_size  # Fixed TP size

    if args.output_md:
        _write_markdown_lines(
            args.output_md,
            _build_markdown_header(
                args=args,
                dataset_spec_text=dataset_spec_text,
                sampling_params_for_report=sampling_params_for_report,
                attention_backends=attention_backends,
                tp=tp,
                concurrencies=concurrencies,
                explicit_dataset_mode=explicit_dataset_mode,
                prompt_groups=prompt_groups,
                max_questions=max_questions,
                device_sm=device_sm,
                is_blackwell=is_blackwell,
            ),
            mode="w",
        )

    for backend in attention_backends:
        port_base = find_available_port(20000)
        if args.output_md:
            _write_markdown_lines(
                args.output_md,
                [f"## Backend: `{backend}`", ""],
                mode="a",
            )

        common_server_args: list[str] = [
            "--trust-remote-code",
            "--attention-backend",
            backend,
            "--tp-size",
            str(tp),
            "--dtype",
            str(args.dtype),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--max-running-requests",
            str(args.max_running_requests),
        ]
        common_server_args.extend(
            ["--cuda-graph-bs", *[str(i) for i in range(1, 33)], "--cuda-graph-max-bs", "32"]
        )
        if args.disable_radix_cache:
            common_server_args.append("--disable-radix-cache")
        if args.mamba_scheduler_strategy:
            common_server_args.extend(
                ["--mamba-scheduler-strategy", str(args.mamba_scheduler_strategy).strip()]
            )

        if args.run_baseline:
            print(f"\n=== backend={backend} tp={tp} (baseline) ===")
            baseline_port = port_base
            baseline_url = f"http://127.0.0.1:{baseline_port}"
            baseline_proc = popen_launch_server(
                args.target_model,
                baseline_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=common_server_args,
            )
            try:
                # Warm up.
                _send_generate(
                    baseline_url,
                    "Hello",
                    max_new_tokens=8,
                    temp=float(args.temp),
                    stop=[],
                    timeout_s=min(int(args.timeout_s), 300),
                )

                for group in prompt_groups:
                    warmup_prompts = _build_warmup_prompts(
                        group.prompts, max_concurrency
                    )
                    for conc in concurrencies:
                        if explicit_dataset_mode:
                            n = len(group.prompts)
                        else:
                            n = min(
                                int(requested_num_questions_by_conc[conc]),
                                len(group.prompts),
                            )
                        _flush_cache(baseline_url)
                        print(
                            f"[warmup] dataset={group.spec_text} run 1 warmup batch "
                            f"(size={conc}) after /flush_cache; excluded from metrics."
                        )
                        metrics = _run_bench_requests(
                            baseline_url,
                            prompts=warmup_prompts[:conc] + group.prompts[:n],
                            max_new_tokens=int(args.max_new_tokens),
                            temp=float(args.temp),
                            concurrency=int(conc),
                            batch_requests=bool(args.batch_requests),
                            stop=[],
                            timeout_s=int(args.timeout_s),
                            expect_speculative=False,
                            output_case_path=None,
                            output_case_label=None,
                        )
                        baseline_toks[(backend, group.spec_text, conc)] = (
                            metrics.output_toks_per_s
                        )
                        print(
                            f"[baseline] dataset={group.spec_text} conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"latency={metrics.latency_s:.1f}s "
                        )
                    if args.output_md and not args.run_speculative:
                        _write_markdown_lines(
                            args.output_md,
                            _build_dataset_markdown_section(
                                args=args,
                                backend=backend,
                                group=group,
                                concurrencies=concurrencies,
                                baseline_toks=baseline_toks,
                                dflash_toks=dflash_toks,
                                dflash_accept_len=dflash_accept_len,
                            ),
                            mode="a",
                        )
            finally:
                kill_process_tree(baseline_proc.pid)
                try:
                    baseline_proc.wait(timeout=30)
                except Exception:
                    pass

        if not args.run_speculative:
            continue

        print(f"\n=== backend={backend} tp={tp} ({spec_label}) ===")
        dflash_port = find_available_port(port_base + 1)
        dflash_url = f"http://127.0.0.1:{dflash_port}"
        dflash_proc = popen_launch_server(
            args.target_model,
            dflash_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *common_server_args,
                *_build_speculative_server_args(args),
            ],
        )
        try:
            _send_generate(
                dflash_url,
                "Hello",
                max_new_tokens=8,
                temp=float(args.temp),
                stop=[],
                timeout_s=min(int(args.timeout_s), 300),
            )
            for group in prompt_groups:
                warmup_prompts = _build_warmup_prompts(group.prompts, max_concurrency)
                for conc in concurrencies:
                    if explicit_dataset_mode:
                        n = len(group.prompts)
                    else:
                        n = min(
                            int(requested_num_questions_by_conc[conc]),
                            len(group.prompts),
                        )
                    _flush_cache(dflash_url)
                    print(
                        f"[warmup] dataset={group.spec_text} run 1 warmup batch "
                        f"(size={conc}) after /flush_cache; excluded from metrics."
                    )
                    metrics = _run_bench_requests(
                        dflash_url,
                        prompts=warmup_prompts[:conc] + group.prompts[:n],
                        max_new_tokens=int(args.max_new_tokens),
                        temp=float(args.temp),
                        concurrency=int(conc),
                        batch_requests=bool(args.batch_requests),
                        stop=[],
                        timeout_s=int(args.timeout_s),
                        expect_speculative=True,
                        output_case_path=args.output_case,
                        output_case_label=(
                            f"dataset={group.spec_text} backend={backend} "
                            f"tp={tp} conc={conc} ({spec_label})"
                        ),
                    )
                    dflash_toks[(backend, group.spec_text, conc)] = (
                        metrics.output_toks_per_s
                    )
                    dflash_accept_len[(backend, group.spec_text, conc)] = (
                        metrics.spec_accept_length
                    )
                    accept_len_text = (
                        "N/A"
                        if metrics.spec_accept_length is None
                        else f"{metrics.spec_accept_length:.3f}"
                    )
                    print(
                        f"[{spec_label}]   dataset={group.spec_text} conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"latency={metrics.latency_s:.1f}s "
                        f"accept_len={accept_len_text} "
                        f"spec_verify_ct_sum={metrics.spec_verify_ct_sum}"
                    )
                if args.output_md:
                    _write_markdown_lines(
                        args.output_md,
                        _build_dataset_markdown_section(
                            args=args,
                            backend=backend,
                            group=group,
                            concurrencies=concurrencies,
                            baseline_toks=baseline_toks,
                            dflash_toks=dflash_toks,
                            dflash_accept_len=dflash_accept_len,
                        ),
                        mode="a",
                    )
        finally:
            kill_process_tree(dflash_proc.pid)
            try:
                dflash_proc.wait(timeout=30)
            except Exception:
                pass

    if args.output_md:
        print(f"\nWrote markdown report incrementally to: {args.output_md}")
    else:
        print("\nMarkdown report disabled (pass --output-md to write one).")

    if args.output_case and args.run_speculative:
        print(f"Wrote {spec_label} case outputs (JSONL) to: {args.output_case}")


if __name__ == "__main__":
    main()
