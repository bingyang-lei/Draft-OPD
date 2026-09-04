#!/usr/bin/env bash
unset RAY_ADDRESS
pkill -9 ray
pkill -9 pythonq
sleep 1
export RAY_ADDRESS=local
set -xeuo pipefail

run_gpu_stress_test_on_exit() {
    local script_exit_code=$?
    set +e
    echo "[cleanup] Running gpu_stress_test.py before exiting..."
    python3 "your GPU stress test script path"
    local stress_exit_code=$?
    if [ ${stress_exit_code} -ne 0 ]; then
        echo "[cleanup] gpu_stress_test.py failed with exit code ${stress_exit_code}"
    fi
    trap - EXIT
    exit ${script_exit_code}
}
trap run_gpu_stress_test_on_exit EXIT

############################ Quick Config ############################
cd "your verl repository path"

ROLLOUT_NAME="sglang"

MAIN_MODEL_PATH="your main model path"
DRAFT_MODEL_PATH="your EAGLE3 draft model path"

TRAIN_JSONL="your training JSONL path"
TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"

TEST_JSONLS=(
    "your AIME 2024 validation JSONL path"
    "your GSM8K validation JSONL path"
    "your MATH-500 validation JSONL path"
    "your MBPP validation JSONL path"
)

STUDENT_MODEL="Qwen3-4B-eagle3"
TEACHER_MODEL="Qwen3-4B"
TEACHER_MODEL_PATH="$MAIN_MODEL_PATH"

USE_POLICY_GRADIENT=False
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-eagle3_native_target_distribution}
if [[ -z "${EAGLE3_RESPONSE_LOSS_MODE:-}" ]]; then
    if [[ "$DISTILLATION_LOSS_MODE" == "eagle3_native_target_distribution" ]]; then
        EAGLE3_RESPONSE_LOSS_MODE="native_target_distribution"
    else
        EAGLE3_RESPONSE_LOSS_MODE="scalar_k3"
    fi
fi
if [[ "$DISTILLATION_LOSS_MODE" == "eagle3_native_target_distribution" && "$EAGLE3_RESPONSE_LOSS_MODE" != "native_target_distribution" ]]; then
    echo "DISTILLATION_LOSS_MODE=eagle3_native_target_distribution requires EAGLE3_RESPONSE_LOSS_MODE=native_target_distribution." >&2
    exit 1
fi
if [[ "$DISTILLATION_LOSS_MODE" != "eagle3_native_target_distribution" && "$EAGLE3_RESPONSE_LOSS_MODE" == "native_target_distribution" ]]; then
    echo "EAGLE3_RESPONSE_LOSS_MODE=native_target_distribution requires DISTILLATION_LOSS_MODE=eagle3_native_target_distribution." >&2
    exit 1
fi
USE_FUSED_KERNELS=False
ROLLOUT_TEMPERATURE=0.0
TEACHER_TEMPERATURE=1.0
VAL_TEMPERATURE=0.0
train_epochs=8

if [[ -z "${DISTILLATION_LOSS_MAX_CLAMP:-}" ]]; then
    if [[ "$DISTILLATION_LOSS_MODE" == "eagle3_native_target_distribution" ]]; then
        DISTILLATION_LOSS_MAX_CLAMP=null
    else
        DISTILLATION_LOSS_MAX_CLAMP=10.0
    fi
fi
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

WANDB_PROJECT='verl-eagle3-opd'

MAX_PROMPT=512
MAX_NUM_TOKENS=4096
MAX_RESPONSE_LENGTH=$(( MAX_NUM_TOKENS - MAX_PROMPT - 1 ))
TRAIN_PROMPT_BSZ=24
UPDATE_ACCUMULATION_STEPS=10
SAVE_FREQ=${SAVE_FREQ:-1000}
SAVE_START_STEP=${SAVE_START_STEP:-4000}
save_freq=$SAVE_FREQ
TEST_FREQ=${TEST_FREQ:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
INITIAL_VAL_REPEAT_TIMES=${INITIAL_VAL_REPEAT_TIMES:-2}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-30}
ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE=${ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE:-True}
if (( TEST_FREQ > 0 && TEST_FREQ % UPDATE_ACCUMULATION_STEPS != 0 )); then
    echo "TEST_FREQ=${TEST_FREQ} is not divisible by UPDATE_ACCUMULATION_STEPS=${UPDATE_ACCUMULATION_STEPS}; speed tests only run on update boundaries." >&2
fi

# full_ttt backward recomputes one TTT step's eager attention at a time; its transient
# peak scales with micro batch size, so default to 1 sequence per micro batch.
STUDENT_MICRO_BATCH_SIZE_PER_GPU=${STUDENT_MICRO_BATCH_SIZE_PER_GPU:-1}
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * MAX_NUM_TOKENS ))
USE_DYNAMIC_BSZ=True

# SpecForge online EAGLE3 training uses --ttt-length default 7. The OPD replay
# block keeps one anchor plus that many future predictions.
EAGLE3_TTT_LENGTH=${EAGLE3_TTT_LENGTH:-7}
EAGLE3_REPLAY_BLOCK_SIZE=$(( EAGLE3_TTT_LENGTH + 1 ))
# full_ttt: SpecForge-aligned full-sequence TTT response stream (recommended).
# anchor_block: legacy per-anchor block implementation, kept for A/B comparison.
EAGLE3_RESPONSE_STREAM_IMPL=${EAGLE3_RESPONSE_STREAM_IMPL:-full_ttt}
EAGLE3_TTT_STEP_WEIGHT=${EAGLE3_TTT_STEP_WEIGHT:-0.8}
EAGLE3_SPECULATIVE_NUM_STEPS=${EAGLE3_SPECULATIVE_NUM_STEPS:-$EAGLE3_TTT_LENGTH}
EAGLE3_SPECULATIVE_EAGLE_TOPK=${EAGLE3_SPECULATIVE_EAGLE_TOPK:-1}
EAGLE3_SPECULATIVE_NUM_DRAFT_TOKENS=${EAGLE3_SPECULATIVE_NUM_DRAFT_TOKENS:-$(( EAGLE3_SPECULATIVE_NUM_STEPS + 1 ))}
EAGLE3_LM_HEAD_CHUNK_SIZE=2048
EAGLE3_RESPONSE_ANCHOR_STRIDE=1

RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
RANDOM_RESPONSE_ANCHOR_SEED=${RANDOM_RESPONSE_ANCHOR_SEED:-42}

LR=1e-4
SEED=42
STUDENT_WORLD_SIZE=6
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$STUDENT_WORLD_SIZE}
stream_weight=1.0
rejected_draft_stream_weight=${rejected_draft_stream_weight:-0.0}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=True
REJECTED_DRAFT_POSITION_DECAY=1.0
TEACHER_WORLD_SIZE=2

inference_max_num_seqs=256
rollout_max_num_seqs=256
SP=1

EAGLE3_SGLANG_ATTENTION_BACKEND=${EAGLE3_SGLANG_ATTENTION_BACKEND:-flashinfer}
ROLLOUT_SGLANG_MEM_FRACTION_STATIC=${ROLLOUT_SGLANG_MEM_FRACTION_STATIC:-0.3}
ROLLOUT_SGLANG_CUDA_GRAPH_MAX_BS=${ROLLOUT_SGLANG_CUDA_GRAPH_MAX_BS:-16}
TEACHER_SGLANG_CUDA_GRAPH_MAX_BS=${TEACHER_SGLANG_CUDA_GRAPH_MAX_BS:-16}

EXP_NAME="fsdp/student-teacher-eagle3/loss-${DISTILLATION_LOSS_MODE}/train-${TRAIN_JSONL_NAME}-update-accumulation-steps"
CKPT_DIR="checkpoints/${WANDB_PROJECT}/${EXP_NAME}"

ENFORCE_EAGER=False

export WANDB_MODE=offline
export WANDB_DIR=${WANDB_DIR:-"./wandb_offline"}
export WANDB_PROJECT
export WANDB_NAME="$EXP_NAME"

############################ Paths ############################

TRAIN_FILES="['$TRAIN_JSONL']"
TEST_FILES="['${TEST_JSONLS[0]}','${TEST_JSONLS[1]}','${TEST_JSONLS[2]}','${TEST_JSONLS[3]}']"

############################ Parameter Groups ############################

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

MODEL=(
    actor_rollout_ref.model.path="$MAIN_MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.verl_composed_eagle3_student=True
    +actor_rollout_ref.model.override_config.verl_eagle3_main_model_path="$MAIN_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_eagle3_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_eagle3_replay_block_size=$EAGLE3_REPLAY_BLOCK_SIZE
    +actor_rollout_ref.model.override_config.verl_eagle3_ttt_length=$EAGLE3_TTT_LENGTH
    +actor_rollout_ref.model.override_config.verl_eagle3_ttt_step_weight=$EAGLE3_TTT_STEP_WEIGHT
    +actor_rollout_ref.model.override_config.verl_eagle3_response_stream_impl=$EAGLE3_RESPONSE_STREAM_IMPL
    +actor_rollout_ref.model.override_config.verl_eagle3_lm_head_chunk_size=$EAGLE3_LM_HEAD_CHUNK_SIZE
    +actor_rollout_ref.model.override_config.verl_eagle3_response_anchor_stride=$EAGLE3_RESPONSE_ANCHOR_STRIDE
    +actor_rollout_ref.model.override_config.verl_eagle3_response_loss_mode=$EAGLE3_RESPONSE_LOSS_MODE
    +actor_rollout_ref.model.override_config.verl_eagle3_random_response_anchor_enabled=$RANDOM_RESPONSE_ANCHOR_ENABLED
    +actor_rollout_ref.model.override_config.verl_eagle3_random_response_anchor_seed=$RANDOM_RESPONSE_ANCHOR_SEED
    actor_rollout_ref.model.enable_gradient_checkpointing=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=$USE_FUSED_KERNELS
    actor_rollout_ref.actor.use_torch_compile=True
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
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.3
    distillation.teacher_models.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_seqs=$inference_max_num_seqs
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.random_seed=$SEED
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.cuda_graph_max_bs=$TEACHER_SGLANG_CUDA_GRAPH_MAX_BS
    distillation.distillation_loss.loss_mode=$DISTILLATION_LOSS_MODE
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

STUDENT=(
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.015
    actor_rollout_ref.actor.data_loader_seed=$SEED
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_BSZ
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$STUDENT_MAX_TOKEN_LEN_PER_GPU
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.seed=$SEED
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$STUDENT_MICRO_BATCH_SIZE_PER_GPU
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
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs
    actor_rollout_ref.rollout.n=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=EAGLE3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_steps=$EAGLE3_SPECULATIVE_NUM_STEPS
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_eagle_topk=$EAGLE3_SPECULATIVE_EAGLE_TOPK
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_num_draft_tokens=$EAGLE3_SPECULATIVE_NUM_DRAFT_TOKENS
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tp_size=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.dtype=bfloat16
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=$EAGLE3_SGLANG_ATTENTION_BACKEND
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static=$ROLLOUT_SGLANG_MEM_FRACTION_STATIC
    +actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=$ROLLOUT_SGLANG_CUDA_GRAPH_MAX_BS
    +actor_rollout_ref.rollout.engine_kwargs.sglang.trust_remote_code=True
    +actor_rollout_ref.rollout.engine_kwargs.sglang.random_seed=$SEED
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name=$WANDB_PROJECT
    trainer.experiment_name=$EXP_NAME
    trainer.default_local_dir="$CKPT_DIR"
    trainer.n_gpus_per_node=$STUDENT_WORLD_SIZE
    trainer.nnodes=1
    trainer.update_accumulation_steps=$UPDATE_ACCUMULATION_STEPS
    trainer.save_freq=$save_freq
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

############################ Launch ############################

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
