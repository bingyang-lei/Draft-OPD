#!/usr/bin/env bash

unset RAY_ADDRESS
pkill -9 ray
pkill -9 python
sleep 1
export RAY_ADDRESS=local
set -xeuo pipefail

WORKDIR=${WORKDIR:-""} # TODO: write your own verl repo path, e.g. /path/to/verl
if [[ -z "$WORKDIR" ]]; then
    echo "Please set WORKDIR to your verl repo path." >&2
    exit 1
fi
cd "$WORKDIR"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

ROLLOUT_NAME=${ROLLOUT_NAME:-sglang}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-False}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
DISTILLATION_LOSS_MAX_CLAMP=${DISTILLATION_LOSS_MAX_CLAMP:-10.0}
DISTILLATION_LOG_PROB_MIN_CLAMP=${DISTILLATION_LOG_PROB_MIN_CLAMP:--10.0}
WANDB_PROJECT=${WANDB_PROJECT:-verl-dflash-opd}
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}
REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
REJECTED_DRAFT_USE_REVERSE_KL=${REJECTED_DRAFT_USE_REVERSE_KL:-True}
LR=${LR:-3e-4}

TEST_FREQ=${TEST_FREQ:-100}
SAVE_FREQ=${SAVE_FREQ:-500}
SAVE_START_STEP=${SAVE_START_STEP:-1500}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
INITIAL_VAL_REPEAT_TIMES=${INITIAL_VAL_REPEAT_TIMES:-2}
train_epochs=${train_epochs:-8}
stream_weight=${stream_weight:-1.0}
rejected_draft_stream_weight=${rejected_draft_stream_weight:-1.0}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.8}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-7}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-21}
UPDATE_ACCUMULATION_STEPS=${UPDATE_ACCUMULATION_STEPS:-10}

PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ACTOR_FSDP_MODEL_DTYPE=${ACTOR_FSDP_MODEL_DTYPE:-bf16}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
STUDENT_MAX_TOKEN_LEN_PER_GPU=${STUDENT_MAX_TOKEN_LEN_PER_GPU:-$(( PPO_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-False}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-True}

DFLASH_ATTENTION_IMPL=${DFLASH_ATTENTION_IMPL:-flex_attention}
DFLASH_LM_HEAD_CHUNK_SIZE=${DFLASH_LM_HEAD_CHUNK_SIZE:-1024}
DFLASH_RESPONSE_ANCHOR_STRIDE=${DFLASH_RESPONSE_ANCHOR_STRIDE:-1}
RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
RANDOM_RESPONSE_ANCHOR_SEED=${RANDOM_RESPONSE_ANCHOR_SEED:-42}

ENFORCE_EAGER=${ENFORCE_EAGER:-False}

ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-32}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$MAX_NUM_TOKENS}
TEACHER_MAX_NUM_BATCHED_TOKENS=${TEACHER_MAX_NUM_BATCHED_TOKENS:-$MAX_NUM_TOKENS}
ROLLOUT_SGLANG_MEM_FRACTION_STATIC=${ROLLOUT_SGLANG_MEM_FRACTION_STATIC:-0.45}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.45}

SEED=${SEED:-42}
SP=${SP:-1}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$ROLLOUT_AGENT_NUM_WORKERS}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-$(( ROLLOUT_SPEED_TEST_WORKER_COUNT * 2 ))}
ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE=${ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE:-True}

if (( STUDENT_WORLD_SIZE + TEACHER_WORLD_SIZE > GPUS_PER_NODE )); then
    echo "STUDENT_WORLD_SIZE + TEACHER_WORLD_SIZE must be <= GPUS_PER_NODE, got ${STUDENT_WORLD_SIZE}+${TEACHER_WORLD_SIZE}>${GPUS_PER_NODE}." >&2
    exit 1
fi

if (( TRAIN_PROMPT_BSZ % ROLLOUT_AGENT_NUM_WORKERS != 0 )); then
    echo "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ} must be divisible by ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}." >&2
    exit 1
fi

if (( ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK <= 0 || ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK % ROLLOUT_SPEED_TEST_WORKER_COUNT != 0 )); then
    echo "ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK} must be a positive multiple of ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT}." >&2
    exit 1
fi

if (( TEST_FREQ > 0 && TEST_FREQ % UPDATE_ACCUMULATION_STEPS != 0 )); then
    echo "TEST_FREQ=${TEST_FREQ} is not divisible by UPDATE_ACCUMULATION_STEPS=${UPDATE_ACCUMULATION_STEPS}; speed tests only run on update boundaries." >&2
fi

MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-""}       # TODO: write your own main model path
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"$MAIN_MODEL_PATH"}
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-""}     # TODO: write your own draft model path

TRAIN_JSONL=${TRAIN_JSONL:-""}               # TODO: write your own train jsonl path
TEST_JSONLS=(
    "" # TODO: write your own validation jsonl path
    "" # TODO: write your own validation jsonl path
    "" # TODO: write your own validation jsonl path
    "" # TODO: write your own validation jsonl path
)

for required_var in MAIN_MODEL_PATH TEACHER_MODEL_PATH DRAFT_MODEL_PATH TRAIN_JSONL; do
    if [[ -z "${!required_var}" ]]; then
        echo "Please set ${required_var} to your own path." >&2
        exit 1
    fi
done

for test_jsonl in "${TEST_JSONLS[@]}"; do
    if [[ -z "$test_jsonl" ]]; then
        echo "Please fill every entry in TEST_JSONLS with your own validation jsonl path." >&2
        exit 1
    fi
done

TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"

TODAY=$(date +"%m-%d")
EXP_NAME=${EXP_NAME:-"a3b-fsdp-draftmodel-0_lr-${LR}-decay-${REJECTED_DRAFT_POSITION_DECAY_ENABLED}-random_anchor-${RANDOM_RESPONSE_ANCHOR_ENABLED}/student-${STUDENT_WORLD_SIZE}-teacher-${TEACHER_WORLD_SIZE}-${TODAY}/bsz-${TRAIN_PROMPT_BSZ}-micro-${PPO_MICRO_BATCH_SIZE_PER_GPU}/train-${TRAIN_JSONL_NAME}-update-accumulation-steps"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/${WANDB_PROJECT}/${EXP_NAME}"}

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"./wandb_offline"}
export WANDB_PROJECT
export WANDB_NAME="$EXP_NAME"

TRAIN_FILES="['$TRAIN_JSONL']"
TEST_FILES="['${TEST_JSONLS[0]}','${TEST_JSONLS[1]}','${TEST_JSONLS[2]}','${TEST_JSONLS[3]}']"

DATA=(
    data.train_files="$TRAIN_FILES"
    data.val_files="$TEST_FILES"
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.train_batch_size=$TRAIN_PROMPT_BSZ
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.seed=$SEED
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path="$MAIN_MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.verl_composed_dflash_student=True
    +actor_rollout_ref.model.override_config.verl_dflash_main_model_path="$MAIN_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_dflash_attention_impl="$DFLASH_ATTENTION_IMPL"
    +actor_rollout_ref.model.override_config.verl_dflash_lm_head_chunk_size=$DFLASH_LM_HEAD_CHUNK_SIZE
    +actor_rollout_ref.model.override_config.verl_dflash_response_anchor_stride=$DFLASH_RESPONSE_ANCHOR_STRIDE
    +actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_enabled=$RANDOM_RESPONSE_ANCHOR_ENABLED
    +actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_seed=$RANDOM_RESPONSE_ANCHOR_SEED
    actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=$USE_TORCH_COMPILE
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=$TEACHER_WORLD_SIZE
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL_PATH"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_models.teacher_model.inference.name=$ROLLOUT_NAME
    distillation.teacher_models.teacher_model.inference.temperature=$TEACHER_TEMPERATURE
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_MEMORY_UTILIZATION
    distillation.teacher_models.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_NUM_BATCHED_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_NUM_SEQS
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.random_seed=$SEED
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
    distillation.distillation_loss.reverse_kl_weight=$REVERSE_KL_WEIGHT
    distillation.distillation_loss.forward_kl_weight=$FORWARD_KL_WEIGHT
    distillation.distillation_loss.rejected_draft_use_reverse_kl=$REJECTED_DRAFT_USE_REVERSE_KL
    distillation.distillation_loss.topk=1
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=$USE_POLICY_GRADIENT
    distillation.distillation_loss.response_stream_weight=$stream_weight
    distillation.distillation_loss.rejected_draft_stream_weight=$rejected_draft_stream_weight
    distillation.distillation_loss.rejected_draft_position_decay_enabled=$REJECTED_DRAFT_POSITION_DECAY_ENABLED
    distillation.distillation_loss.rejected_draft_position_decay=$REJECTED_DRAFT_POSITION_DECAY
    distillation.distillation_loss.loss_max_clamp=$DISTILLATION_LOSS_MAX_CLAMP
    distillation.distillation_loss.log_prob_min_clamp=$DISTILLATION_LOG_PROB_MIN_CLAMP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=$ROLLOUT_NAME
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE
    actor_rollout_ref.rollout.val_kwargs.top_k=1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS
    actor_rollout_ref.rollout.n_gpus_per_node=$STUDENT_WORLD_SIZE
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=DFLASH
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tp_size=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.dtype=bfloat16
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static=$ROLLOUT_SGLANG_MEM_FRACTION_STATIC
    +actor_rollout_ref.rollout.engine_kwargs.sglang.trust_remote_code=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=$SEED
)

STUDENT=(
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.data_loader_seed=$SEED
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_FSDP_MODEL_DTYPE
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD
    actor_rollout_ref.actor.fsdp_config.seed=$SEED
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$WANDB_PROJECT
    trainer.experiment_name=$EXP_NAME
    trainer.default_local_dir="$CKPT_DIR"
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.update_accumulation_steps=$UPDATE_ACCUMULATION_STEPS
    trainer.save_freq=$SAVE_FREQ
    trainer.save_start_step=$SAVE_START_STEP
    trainer.test_freq=$TEST_FREQ
    trainer.rollout_speed_test_only=True
    trainer.initial_val_repeat_times=$INITIAL_VAL_REPEAT_TIMES
    trainer.rollout_speed_test_max_samples_per_benchmark=$ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK
    trainer.rollout_speed_test_worker_count=$ROLLOUT_SPEED_TEST_WORKER_COUNT
    trainer.rollout_speed_test_clear_kv_cache=$ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE
    trainer.total_epochs=$train_epochs
    trainer.val_before_train=$VAL_BEFORE_TRAIN
    trainer.resume_mode=disable
    trainer.log_val_generations=10
)

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${DISTILLATION[@]}" \
    "${ROLLOUT[@]}" \
    "${STUDENT[@]}" \
    "${TRAINER[@]}" \
    "$@"
