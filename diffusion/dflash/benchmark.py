import argparse
import csv
import os
import time
import random
from collections import Counter
from itertools import chain
from types import SimpleNamespace
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import DFlashDraftModel, sample, load_and_process_dataset, extract_context_feature
import distributed as dist
# from pyinstrument import Profiler
# import os
# # 获取当前进程的 local_rank，如果是单机多卡，0 就是主进程
# local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# if local_rank == 0:
#     profiler = Profiler()
#     profiler.start()

def cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def tail_tokens_are_repetitive(token_ids: list[int]) -> bool:
    """Heuristic: last chunk is dominated by one token or a short repeating period."""
    n = len(token_ids)
    if n < 100:
        return False
    top_cnt = Counter(token_ids).most_common(1)[0][1]
    if top_cnt / n >= 0.95:
        return True
    max_p = min(20, n // 2)
    for p in range(1, max_p + 1):
        if all(token_ids[i] == token_ids[i % p] for i in range(n)):
            return True
    return False

# output_ids = torch.full(
    #     (1, max_length + block_size),
    #     mask_token_id,
    #     dtype=torch.long,
    #     device=model.device,
    # )
    # output_ids[:, :num_input_tokens] = input_ids
    # output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    # block_output_ids = output_ids[:, start : start + block_size].clone()
    # noise_embedding = target.model.embed_tokens(block_output_ids)

    # position_ids = torch.arange(max_length + block_size, device=model.device).unsqueeze(0)
    # start = input_ids.shape[1]
    # position_ids[:, past_key_values_draft.get_seq_length(): start + block_size]
@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    ) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )


    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    # print("shape of target_hidden: ", target_hidden.shape)
    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    stopped_by_eos = False

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            stopped_by_eos = True
            break

    hit_token_limit = not stopped_by_eos

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        hit_token_limit=hit_token_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking in the prompt (off by default)")
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip baseline evaluation for block-size=1 and related outputs.",
    )
    parser.add_argument(
        "--case",
        action="store_true",
        help="Print per-case generated response content.",
    )
    parser.add_argument(
        "--save-acc-len",
        type=str,
        default=None,
        metavar="PATH",
        help="If set, write per-draft acceptance lengths to this CSV path (columns: idx, acc_len).",
    )
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")
    if dist.is_main():
        print("now the dataset we evaluate is: ", args.dataset)

    def has_flash_attn():
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    if dist.is_main():
        print(f"Using block size: {block_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    case_outputs = [] if args.case else None
    indices = range(dist.rank(), len(dataset), dist.size())
    for idx in tqdm(indices, disable=not dist.is_main()):
        instance = dataset[idx]
        messages = []
        case_turn_outputs = [] if args.case else None
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            bs_list = [block_size] if args.skip_base else list(dict.fromkeys([1, block_size]))
            for bs in bs_list:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            spec_response.exclude_from_acc_len_csv = False
            if args.save_acc_len and spec_response.hit_token_limit and generated_ids.numel() >= 100:
                tail_ids = generated_ids[-100:].tolist()
                if tail_tokens_are_repetitive(tail_ids):
                    spec_response.exclude_from_acc_len_csv = True
                    tail_text = tokenizer.decode(
                        generated_ids[-100:], skip_special_tokens=False
                    )
                    logger.warning(
                        "[save-acc-len] Excluding sample from acc-len CSV: hit max-new-tokens "
                        f"with repetitive last-100 tokens (dataset idx={idx}, turn={turn_index})."
                    )
                    print(
                        f"[save-acc-len] dataset idx={idx} turn={turn_index} "
                        f"last_100_token_ids={tail_ids}\n"
                        f"[save-acc-len] last_100 decoded (raw): {tail_text!r}"
                    )
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            if args.case:
                case_turn_outputs.append(output_text)
            
            # Avoid sending large GPU tensors through gather_object.
            for bs in response:
                response[bs].output_ids = None

            responses.append(response)
        if args.case:
            case_outputs.append(
                {
                    "idx": idx,
                    "res": "\n".join(case_turn_outputs),
                }
            )

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if args.case:
            case_outputs = dist.gather(case_outputs, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        if args.case:
            case_outputs = list(chain(*case_outputs))

    if args.case:
        case_outputs.sort(key=lambda x: x["idx"])
        for item in case_outputs:
            print(f"dataset: {args.dataset}, idx: {item['idx']}.")
            print(f"res: {item['res']}\n\n")

    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    mean_num_output_tokens = np.mean([r[block_size].num_output_tokens for r in responses])
    max_num_output_tokens = np.max([r[block_size].num_output_tokens for r in responses])
    print(f"dflash max num output tokens: {max_num_output_tokens}")
    print(f"dflash mean num output tokens: {mean_num_output_tokens}")
    print(f"dflash time per output token: {tb}")
    print(f"dflash mean throughput: {1 / tb:.2f} tokens/s")

    if not args.skip_base:
        t1 = np.mean([r[1].time_per_output_token for r in responses])
        baseline_mean_num_output_tokens = np.mean([r[1].num_output_tokens for r in responses])
        max_baseline_num_output_tokens = np.max([r[1].num_output_tokens for r in responses])
        print(f"baseline max num output tokens: {max_baseline_num_output_tokens}")
        print(f"baseline mean num output tokens: {baseline_mean_num_output_tokens}")
        print(f"baseline time per output token: {t1}")
        print(f"baseline mean throughput: {1 / t1:.2f} tokens/s")
        print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if args.save_acc_len:
        per_gen_lengths = [
            r[block_size].acceptance_lengths
            for r in responses
            if not getattr(r[block_size], "exclude_from_acc_len_csv", False)
        ]
        n_excluded = sum(
            1 for r in responses if getattr(r[block_size], "exclude_from_acc_len_csv", False)
        )
        if n_excluded:
            print(f"[save-acc-len] Excluded {n_excluded} generations from CSV (repetitive tail at limit).")
        max_steps = max((len(L) for L in per_gen_lengths), default=0)
        out_path = os.path.abspath(os.path.expanduser(args.save_acc_len))
        parent = os.path.dirname(out_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "acc_len"])
            for i in range(max_steps):
                vals = [L[i] for L in per_gen_lengths if i < len(L)]
                acc_mean = float(np.mean(vals)) if vals else float("nan")
                writer.writerow([i, acc_mean])
        print(f"Saved per-draft-step mean acceptance lengths (over samples) to {out_path}")

if __name__ == "__main__":
    main()
    # if local_rank == 0:
    #     profiler.stop()
    #     # 将结果输出为 HTML，方便在浏览器中点开折叠查看
    #     profiler.write_html("benchmark_profile.html")
    #     print("性能分析报告已生成: benchmark_profile.html")