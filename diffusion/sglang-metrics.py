#!/usr/bin/env python3
"""
Standalone script to benchmark an already-started SGLang server via sglang.run_batch
(and compute metrics similar to SpecForge's compute_metrics).

Usage:
    python sglang_metrics.py --port 30000 --num 10 --batch-size 1 --context "Hello, world"
    python sglang_metrics.py --port 30000 --num 5 --batch-size 4 --context "Hello, world"
    python sglang_metrics.py --port 30000 --num 5 --batch-size 2 --context-file prompts.txt
"""
import os

hf_home = os.environ.get("HF_HOME", "")  # your Hugging Face cache path.
if hf_home:
    os.environ["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", f"{hf_home}/hub")
    os.environ["HF_DATASETS_CACHE"] = os.environ.get("HF_DATASETS_CACHE", f"{hf_home}/datasets")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
from datasets import load_dataset
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, List, Optional
import requests


@dataclass
class BenchmarkMetrics:
    latency: float
    output_throughput: float
    accept_length: float


DEFAULT_DAPO_PATH = os.environ.get("DAPO_PATH", "")  # your DAPO jsonl path.
TARGET_MODEL_PATH = os.environ.get("TARGET_MODEL_PATH", "")  # your target model path.
def _build_base_url(host: str, port: int) -> str:
    if host.startswith(("http://", "https://")):
        return f"{host}:{port}"
    return f"http://{host}:{port}"


def _build_sampling_params(temperature: float, max_new_tokens: int) -> dict:
    temperature = float(temperature)
    sampling_params: dict = {
        "temperature": temperature,
        "max_new_tokens": int(max_new_tokens),
    }
    if temperature == 0:
        sampling_params.update({
            "top_p": 1.0,
            "top_k": 1,
        })
    return sampling_params


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    temperature: float,
    timeout_s: int,
    ) -> dict:
    sampling_params = _build_sampling_params(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
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


def _send_generate_batch(
    base_url: str,
    prompts: list[str],
    *,
    max_new_tokens: int,
    stop: list[str],
    temperature: float,
    timeout_s: int,
    ) -> list[dict]:
    if not prompts:
        return []
    sampling_params = _build_sampling_params(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
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


def _flatten_outputs(outputs: List[Any]) -> List[dict]:
    """Normalize outputs to a flat list[dict] for metrics."""
    if isinstance(outputs[0], dict):
        return outputs

    flat_outputs: List[dict] = []
    for out in outputs:
        for item in out:
            if isinstance(item, dict):
                flat_outputs.append(item)
        # Ignore unexpected output item types silently.
    return flat_outputs


def compute_metrics_from_outputs(outputs: List[Any], total_latency: float) -> BenchmarkMetrics:
    if not outputs:
        return BenchmarkMetrics(
            latency=total_latency,
            output_throughput=0.0,
            accept_length=1.0,
        )

    num_output_tokens = sum(
        int((out.get("meta_info", {}) or {}).get("completion_tokens", 0))
        for out in outputs
    )
    output_throughput = num_output_tokens / total_latency if total_latency > 0 else 0.0

    first_meta = outputs[0].get("meta_info", {}) or {}
    has_verify = "spec_verify_ct" in first_meta
    if has_verify:
        num_verify_tokens = sum(
            int((out.get("meta_info", {}) or {}).get("spec_verify_ct", 0))
            for out in outputs
        )
        accept_length = num_output_tokens / num_verify_tokens if num_verify_tokens > 0 else 1.0
    else:
        accept_length = 1.0

    return BenchmarkMetrics(
        latency=total_latency,
        output_throughput=output_throughput,
        accept_length=accept_length,
    )

answers: List[str] = []
def load_math500_prompts(num: Optional[int]) -> List[str]:
    global answers
    prompts: List[str] = []
    answers = []
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompt_fmt = (
        "{problem}\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
        "Use standard LaTeX math formatting in your response: write fractions as "
        "\\frac{{a}}{{b}}, text labels as \\text{{...}}, and degree symbols as ^\\circ "
        "(for example, 90^\\circ)."
    )
    dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
        answers.append(item.get("answer", ""))
    return prompts

def load_aime24_prompts(num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
        answers.append(item.get("answer", ""))
    return prompts

def load_gsm8k_prompts(num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    dataset = load_dataset("gsm8k", "main", split="test")
    prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
    return prompts

def load_mbpp_prompts(num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
    return prompts

def load_humaneval_prompts(num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    dataset = load_dataset("openai/openai_humaneval", split="test")
    prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
    dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
    return prompts

def load_mtbench_prompts(num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    dataset = dataset.map(lambda x: {"turns": x["prompt"]})
    for i in range(min(num, len(dataset))):
        item = dataset[i]
        prompts.append(item["turns"][0])
    return prompts

def load_dapo_prompts(dapo_file: str, num: Optional[int]) -> List[str]:
    prompts: List[str] = []
    file_path = Path(dapo_file)
    if not file_path.exists():
        raise FileNotFoundError(f"DAPO data file not found: {dapo_file}")

    with file_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            message_list = obj.get("prompt") or []
            content = None
            if isinstance(message_list, list):
                for msg in message_list:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content")
                        break
                if content is None and message_list and isinstance(message_list[0], dict):
                    content = message_list[0].get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Invalid prompt content at line {line_idx} in {dapo_file}")
            prompts.append(content)
            if num is not None and len(prompts) >= num:
                break

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {dapo_file}")
    return prompts

def run_benchmark(
    port: int = 30000,
    num: int = 10,
    context: Optional[str] = None,
    context_file: Optional[str] = None,
    model: str = "None",  # kept for compatibility; not used in run_batch mode
    max_tokens: int = 2048,
    temperature: float = 0,
    host: str = "localhost",
    batch_size: int = 1,
    timeout_s: int = 3600,
    bench_list: str = "chat",
    dapo_file: str = DEFAULT_DAPO_PATH,
    enable_thinking: bool = False,
) -> tuple[BenchmarkMetrics, List[Any]]:
    """
    Run benchmark by calling sgl_func.run_batch repeatedly.

    Behavior:
    - If context_file is given: take at most num prompts (one prompt per run).
    - Else: repeat one context prompt for num runs.
    - In each run, send batch_size identical requests simultaneously.
    """
    _ = model  # keep existing parameter unchanged for CLI compatibility
    base_url = _build_base_url(host=host, port=port)

    if bench_list == "dapo":
        prompts = load_dapo_prompts(dapo_file=dapo_file, num=num)
    elif bench_list == "math500":
        prompts = load_math500_prompts(num=num)
    elif bench_list == "mbpp":
        prompts = load_mbpp_prompts(num=num)
    elif bench_list == "gsm8k":
        prompts = load_gsm8k_prompts(num=num)
    elif bench_list == "mt-bench":
        prompts = load_mtbench_prompts(num=num)
    elif bench_list == "aime24":
        prompts = load_aime24_prompts(num=num)
    elif bench_list == "humaneval":
        prompts = load_humaneval_prompts(num=num)
    else:
        if context_file:
            with open(context_file, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            prompts = prompts[:num] if num is not None else prompts
        else:
            if context is None:
                context = r"""Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\theta),$ where $r > 0$ and $0 \le \theta < 2 \pi.$
Please reason step by step, and put your final answer within \boxed{}."""
            prompts = [context] * num

    target_model = TARGET_MODEL_PATH
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    ) for prompt in prompts]

    print("answers:", answers)
    # print("======================")
    # print("After tokenization, prompts[0]:", prompts[0])
    # print("======================")

    all_outputs: List[dict] = []
    total_latency = 0.0
    if batch_size > 1:
        batch_num = (len(prompts) + batch_size - 1) // batch_size
        for batch_idx in range(batch_num):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))
            batch_prompts = prompts[start_idx:end_idx]
            tic = time.perf_counter()
            out = _send_generate_batch(
                base_url,
                batch_prompts,
                max_new_tokens=max_tokens,
                stop=[],
                temperature=temperature,
                timeout_s=timeout_s,
            )
            outs = [out]
            lat = time.perf_counter() - tic
            total_latency += lat
            all_outputs.extend(outs)
            print(f"[batch {batch_idx+1}] batch_size={len(batch_prompts)}, latency={lat:.4f}s")
    else:
        for i, prompt in enumerate(prompts):
            # one /generate call with batch payload of identical prompts
            tic = time.perf_counter()
            out = _send_generate(
                base_url,
                prompt,
                max_new_tokens=max_tokens,
                stop=[],
                temperature=temperature,
                timeout_s=timeout_s,
            )
            outs = [out]
            lat = time.perf_counter() - tic
            total_latency += lat
            all_outputs.extend(outs)

            # optional: quick per-round log
            print(f"[round {i+1}/{len(prompts)}] batch_size={batch_size}, latency={lat:.4f}s")
    all_outputs = _flatten_outputs(all_outputs)
    metrics = compute_metrics_from_outputs(all_outputs, total_latency)
    return metrics, all_outputs


def main():
    parser = argparse.ArgumentParser(description="Benchmark SGLang via run_batch (no requests)")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--num", type=int, default=1, help="Number of run_batch rounds")
    parser.add_argument("--context", type=str, default=None, help="Single prompt text")
    parser.add_argument("--context-file", type=str, default=None, help="File with one prompt per line")
    parser.add_argument("--model", type=str, default="None")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--output", type=str, default=None, help="Save metrics JSON to file")
    parser.add_argument(
        "--bench-list",
        type=str,
        choices=["chat", "dapo", "math500", "mbpp", "gsm8k", "mt-bench", "aime24", "humaneval"],
        default="chat",
        help="Benchmark task type: chat or dapo or math500 or mbpp or gsm8k or mt-bench or aime24 or humaneval",
    )
    parser.add_argument(
        "--dapo-file",
        type=str,
        default=DEFAULT_DAPO_PATH,
        help="Path to DAPO jsonl file (used when --bench-list dapo)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Requests per run_batch call (same prompt duplicated batch_size times)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking in the prompt (off by default)",
    )
    args = parser.parse_args()
    args.context = "hello, introduce yourself"
    metrics, outputs = run_benchmark(
        port=args.port,
        num=args.num,
        context=args.context,
        context_file=args.context_file,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        host=args.host,
        batch_size=args.batch_size,
        timeout_s=args.timeout_s,
        bench_list=args.bench_list,
        dapo_file=args.dapo_file,
        enable_thinking=args.enable_thinking,
    )

    total_completion_tokens = 0
    for out in outputs:
        meta_info = out.get("meta_info", {}) or {}
        completion_tokens = meta_info.get("completion_tokens", 0)
        try:
            total_completion_tokens += int(completion_tokens)
        except (TypeError, ValueError):
            continue
    print(f"Total completion_tokens: {total_completion_tokens}")

    sampling_params = _build_sampling_params(
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )
    config = {
        "num_rounds": args.num,
        "batch_size": args.batch_size,
        "total_requests": args.num * args.batch_size,
        "host": args.host,
        "port": args.port,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_s": args.timeout_s,
        "bench_list": args.bench_list,
        "dapo_file": args.dapo_file if args.bench_list == "dapo" else None,
    }
    if "top_p" in sampling_params:
        config["top_p"] = sampling_params["top_p"]
    if "top_k" in sampling_params:
        config["top_k"] = sampling_params["top_k"]

    result = {
        "metrics": [asdict(metrics)],
        "config": config,
    }

    print(f"Total requests: {len(outputs)}")

    # print(f"type of states[0]: {type(states[0])}")
    # print(f"states[0]: {states[0]}")

    print(json.dumps(result, indent=4, ensure_ascii=False))

    # optional: inspect one state meta
    if outputs and args.batch_size == 1:
        example_out = outputs[0]
        output = example_out.get("text", "")
        print("Example output(first 100 characters):", output[:100] + "...")
        # print("Example meta:", example_out.get("meta_info", {}))

    if args.output:
        with open(args.output, "a", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            f.write(f"Total completion_tokens: {total_completion_tokens}\n")
            f.write(f"the benchmark is: {args.bench_list}\n")
            # write example answer to a file
            if outputs and args.batch_size == 1:
                example_out = outputs[0]
                # output = example_out.get("text", "")
                # f.write(f"Example output(first 100 characters): {output[:100] + '...'}\n")
                # f.write(f"Example meta: {example_out.get('meta_info', {})}\n")
                for i, out in enumerate(outputs):
                    meta_info = out.get("meta_info", {}) or {}
                    completion_tokens = meta_info.get("completion_tokens", None)
                    f.write(
                        f"output[{i}]['meta_info']['completion_tokens']: {completion_tokens}\n"
                    )

            f.write("===================finished=================\n\n")
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
