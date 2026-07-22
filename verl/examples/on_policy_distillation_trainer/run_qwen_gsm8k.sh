#!/usr/bin/env bash
unset RAY_ADDRESS
pkill -9 ray
pkill -9 python
sleep 1
export RAY_ADDRESS=local
set -xeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

run_gpu_stress_test_on_exit() {
    local script_exit_code=$?
    set +e
    if [ -n "${GPU_STRESS_TEST_SCRIPT:-}" ]; then
        echo "[cleanup] Running GPU_STRESS_TEST_SCRIPT before exiting..."
        python3 "${GPU_STRESS_TEST_SCRIPT}"
        local stress_exit_code=$?
        if [ ${stress_exit_code} -ne 0 ]; then
            echo "[cleanup] GPU_STRESS_TEST_SCRIPT failed with exit code ${stress_exit_code}"
        fi
    fi
    trap - EXIT
    exit ${script_exit_code}
}
trap run_gpu_stress_test_on_exit EXIT

############################ Quick Config ############################
cd "$REPO_ROOT"
# source /path/to/miniconda3/bin/activate verl
ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}" # sglang or vllm

MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-""} # your model path.

DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-""} # your draft model path.

TRAIN_JSONL=${TRAIN_JSONL:-""} # your data path.
TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"

TEST_JSONLS=(
    "" # your data path.
    "" # your data path.
    "" # your data path.
    "" # your data path.
)

STUDENT_MODEL="Qwen3-4B-dflash"
TEACHER_MODEL="Qwen3-4B"
TEACHER_MODEL_PATH="$MAIN_MODEL_PATH"

# USE_POLICY_GRADIENT=False
# DISTILLATION_LOSS_MODE="k3"
# DISTILLATION_LOSS_MODE="forward_kl_topk"
# USE_FUSED_KERNELS=False

USE_POLICY_GRADIENT=False
DISTILLATION_LOSS_MODE="k3"
USE_FUSED_KERNELS=False
ROLLOUT_TEMPERATURE=0.0
TEACHER_TEMPERATURE=1.0
VAL_TEMPERATURE=0.0
train_epochs=2

DISTILLATION_LOSS_MAX_CLAMP=10.0
DISTILLATION_LOG_PROB_MIN_CLAMP=-10.0

WANDB_PROJECT='verl-dflash-opd'

MAX_PROMPT=512
MAX_RESPONSE_LENGTH=4096
MAX_NUM_TOKENS=$(( MAX_PROMPT + MAX_RESPONSE_LENGTH + 1 ))
TRAIN_PROMPT_BSZ=24
UPDATE_ACCUMULATION_STEPS=10
SAVE_FREQ=${SAVE_FREQ:-500}
SAVE_START_STEP=${SAVE_START_STEP:-1600}
save_freq=$SAVE_FREQ
TEST_FREQ=${TEST_FREQ:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
INITIAL_VAL_REPEAT_TIMES=${INITIAL_VAL_REPEAT_TIMES:-2}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-30}
ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE=${ROLLOUT_SPEED_TEST_CLEAR_KV_CACHE:-True}
if (( TEST_FREQ > 0 && TEST_FREQ % UPDATE_ACCUMULATION_STEPS != 0 )); then
    echo "TEST_FREQ=${TEST_FREQ} is not divisible by UPDATE_ACCUMULATION_STEPS=${UPDATE_ACCUMULATION_STEPS}; speed tests only run on update boundaries." >&2
fi

STUDENT_MICRO_BATCH_SIZE_PER_GPU=2
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( STUDENT_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
USE_DYNAMIC_BSZ=True
DFLASH_ATTENTION_IMPL=${DFLASH_ATTENTION_IMPL:-"flex_attention"}
DFLASH_LM_HEAD_CHUNK_SIZE=512
DFLASH_RESPONSE_ANCHOR_STRIDE=1 # 控制 response 里用于构造 DFLASH anchor/segment 的密度, 1代表全部计算，2代表隔一个计算一个

# 用来做随机anchor的消融
RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
RANDOM_RESPONSE_ANCHOR_SEED=${RANDOM_RESPONSE_ANCHOR_SEED:-42}

LR=1e-5
SEED=42
STUDENT_WORLD_SIZE=6
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$STUDENT_WORLD_SIZE}
stream_weight=1.0
rejected_draft_stream_weight=2.0
REJECTED_DRAFT_POSITION_DECAY_ENABLED=True
REJECTED_DRAFT_POSITION_DECAY=0.9
TEACHER_WORLD_SIZE=2

inference_max_num_seqs=256
rollout_max_num_seqs=256
SP=1

EXP_NAME="fsdp/student-teacher-0503/loss-${DISTILLATION_LOSS_MODE}/train-${TRAIN_JSONL_NAME}-update-accumulation-steps"
CKPT_DIR="checkpoints/${WANDB_PROJECT}/${EXP_NAME}"

ENFORCE_EAGER=False # true for faster debugging

# Log to local wandb directory without syncing to wandb cloud.
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
    +actor_rollout_ref.model.override_config.verl_composed_dflash_student=True
    +actor_rollout_ref.model.override_config.verl_dflash_main_model_path="$MAIN_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.model.override_config.verl_dflash_attention_impl="$DFLASH_ATTENTION_IMPL"
    +actor_rollout_ref.model.override_config.verl_dflash_lm_head_chunk_size=$DFLASH_LM_HEAD_CHUNK_SIZE
    +actor_rollout_ref.model.override_config.verl_dflash_response_anchor_stride=$DFLASH_RESPONSE_ANCHOR_STRIDE
    +actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_enabled=$RANDOM_RESPONSE_ANCHOR_ENABLED
    +actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_seed=$RANDOM_RESPONSE_ANCHOR_SEED
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
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4
    distillation.teacher_models.teacher_model.inference.enforce_eager=$ENFORCE_EAGER
    distillation.teacher_models.teacher_model.inference.max_model_len=$MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$MAX_NUM_TOKENS
    distillation.teacher_models.teacher_model.inference.max_num_seqs=$inference_max_num_seqs
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.sglang.random_seed=$SEED
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
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
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
    # actor_rollout_ref.rollout.free_cache_engine=False
    
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE
    actor_rollout_ref.rollout.val_kwargs.top_k=1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.n=1

    actor_rollout_ref.rollout.load_format=auto
    # actor_rollout_ref.rollout.gpu_memory_utilization=0.9
    actor_rollout_ref.rollout.calculate_log_probs=False
    actor_rollout_ref.rollout.max_model_len=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_TOKENS
    actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs
    actor_rollout_ref.rollout.n=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_algorithm=DFLASH
    +actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="$DRAFT_MODEL_PATH"
    +actor_rollout_ref.rollout.engine_kwargs.sglang.tp_size=1
    +actor_rollout_ref.rollout.engine_kwargs.sglang.dtype=bfloat16
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=fa3
    +actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static=0.4
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
