#!/usr/bin/env bash
# Draft-OPD on Ascend NPU with the vLLM(-Ascend) rollout engine.
#
# Differences from run_qwen_gsm8k_forward-ins.sh (the SGLang reference path):
#   1. ROLLOUT_NAME=vllm — student rollouts run on vLLM(vllm-ascend) with the
#      DFlash speculative-decoding method instead of sglang-dflash.
#   2. The DFlash draft checkpoint is passed via rollout.mtp (method=dflash),
#      which vLLMHttpServer turns into speculative_config.
#   3. Per-request reject metadata (positions / draft token ids / teacher
#      logprobs) is recorded by verl.utils.vllm.dflash_opd_patch, activated
#      automatically when the composed DFlash student is detected
#      (VERL_DFLASH_OPD=1 is set by the rollout server itself).
#
# Requirements (NPU):
#   - torch_npu + CANN matching the vllm-ascend version, and vllm-ascend
#     installed (provides the `dflash` speculative method).
#   - See requirements-npu.txt.
#
# Recommended NPU runtime env (prevents allocator fragmentation OOM):
#   export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:32
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export ROLLOUT_NAME="vllm"

# Default to NPUs 8-15 (8 cards total), matching the default split
# STUDENT_WORLD_SIZE=7 + TEACHER_WORLD_SIZE=1 below. Override by exporting
# ASCEND_RT_VISIBLE_DEVICES (and the world sizes) before launching.
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}
REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
REJECTED_DRAFT_USE_REVERSE_KL=${REJECTED_DRAFT_USE_REVERSE_KL:-True}
LR=${LR:-3e-4}
TEST_FREQ=${TEST_FREQ:-250}
SAVE_FREQ=${SAVE_FREQ:-500}
SAVE_START_STEP=${SAVE_START_STEP:-5000}
STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-7}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-21}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( PPO_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-$(( ROLLOUT_SPEED_TEST_WORKER_COUNT * 2 ))}
train_epochs=${train_epochs:-8}
stream_weight=${stream_weight:-1.0}
rejected_draft_stream_weight=${rejected_draft_stream_weight:-1.0}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.8}
RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
RANDOM_RESPONSE_ANCHOR_SEED=${RANDOM_RESPONSE_ANCHOR_SEED:-42}
DFLASH_LM_HEAD_CHUNK_SIZE=${DFLASH_LM_HEAD_CHUNK_SIZE:-512}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.2}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
# Graph mode (aclgraph capture) is the default now that the dflash rollout
# path is validated; eager runs each decode step kernel-by-kernel and costs
# 2-5x generation time. Set ROLLOUT_ENFORCE_EAGER=True only for bring-up or
# debugging to remove graph-mode failure modes.
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
# Keep rollout weights resident by default: skipping the HYBRID sleep/wake
# cycle avoids offloading and restoring the full weight set between NPU and
# host RAM on every step (two full copies per engine per step, plus the KV
# rebuild). Set ROLLOUT_FREE_CACHE_ENGINE=True if update_actor OOMs on
# activation memory (sleep level=1 still preserves the frozen DFlash target
# weights correctly, it just pays the copy cost).
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
# Graph capture footprint for the student rollout engine: PIECEWISE-only
# with a small size list. The verl default (FULL_AND_PIECEWISE with ~35
# sizes) exhausts NPU driver stream/queue resources when a student engine
# colocates with the FSDP trainer on one card (AclmdlRICaptureBegin 207005).
# JSON string consumed by vllm_async_server via json.loads.
VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG:-'{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[8,16,32,64]}'}
# Escape embedded double quotes so the value survives as ONE hydra string
# override (a raw JSON token fails hydra's override grammar).
_VLLM_COMPILATION_CONFIG_ESCAPED=${VLLM_COMPILATION_CONFIG//\"/\\\"}
ENABLE_THINKING=${ENABLE_THINKING:-False}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-""} # your DFlash draft model path.
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-7} # DFlash block size K.
TRAIN_JSONL=${TRAIN_JSONL:-""} # your data path.
TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"
TRAIN_FILES="['$TRAIN_JSONL']"
TODAY=$(date +"%m-%d")
EXP_NAME=${EXP_NAME:-"vllm-npu-ins-lr-${LR}-random_anchor-${RANDOM_RESPONSE_ANCHOR_ENABLED}/student-teacher-${TODAY}/${DISTILLATION_LOSS_MODE}/enable-thinking-${ENABLE_THINKING}/train-${TRAIN_JSONL_NAME}-update-accumulation-steps"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/verl-dflash-opd/${EXP_NAME}"}

exec "${SCRIPT_DIR}/run_qwen_gsm8k.sh" \
    data.train_files="${TRAIN_FILES}" \
    data.max_prompt_length="${MAX_PROMPT}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="${DRAFT_MODEL_PATH}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_lm_head_chunk_size="${DFLASH_LM_HEAD_CHUNK_SIZE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_enabled="${RANDOM_RESPONSE_ANCHOR_ENABLED}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_seed="${RANDOM_RESPONSE_ANCHOR_SEED}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_NUM_TOKENS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}" \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    ++actor_rollout_ref.rollout.mtp._target_=verl.workers.config.MtpConfig \
    ++actor_rollout_ref.rollout.mtp.enable=True \
    ++actor_rollout_ref.rollout.mtp.enable_rollout=True \
    ++actor_rollout_ref.rollout.mtp.method=dflash \
    ++actor_rollout_ref.rollout.mtp.num_speculative_tokens="${NUM_SPECULATIVE_TOKENS}" \
    ++actor_rollout_ref.rollout.mtp.draft_model_path="${DRAFT_MODEL_PATH}" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config=\"${_VLLM_COMPILATION_CONFIG_ESCAPED}\"" \
    distillation.n_gpus_per_node="${TEACHER_WORLD_SIZE}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_NUM_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.reverse_kl_weight="${REVERSE_KL_WEIGHT}" \
    distillation.distillation_loss.forward_kl_weight="${FORWARD_KL_WEIGHT}" \
    distillation.distillation_loss.rejected_draft_use_reverse_kl="${REJECTED_DRAFT_USE_REVERSE_KL}" \
    distillation.distillation_loss.response_stream_weight="${stream_weight}" \
    distillation.distillation_loss.rejected_draft_stream_weight="${rejected_draft_stream_weight}" \
    distillation.distillation_loss.rejected_draft_position_decay_enabled="${REJECTED_DRAFT_POSITION_DECAY_ENABLED}" \
    distillation.distillation_loss.rejected_draft_position_decay="${REJECTED_DRAFT_POSITION_DECAY}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_start_step="${SAVE_START_STEP}" \
    trainer.n_gpus_per_node="${STUDENT_WORLD_SIZE}" \
    trainer.rollout_speed_test_worker_count="${ROLLOUT_SPEED_TEST_WORKER_COUNT}" \
    trainer.rollout_speed_test_max_samples_per_benchmark="${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK}" \
    trainer.total_epochs="${train_epochs}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"
