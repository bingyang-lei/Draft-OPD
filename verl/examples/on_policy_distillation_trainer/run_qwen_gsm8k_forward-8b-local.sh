#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /mnt/shared-storage-user/leihaodi/opd/verl

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k3}

USE_TV_LOSS=${USE_TV_LOSS:-True}

REVERSE_KL_WEIGHT=${REVERSE_KL_WEIGHT:-0.0}
FORWARD_KL_WEIGHT=${FORWARD_KL_WEIGHT:-1.0}
REJECTED_DRAFT_USE_REVERSE_KL=${REJECTED_DRAFT_USE_REVERSE_KL:-True}
LR=${LR:-3e-4}
TEST_FREQ=${TEST_FREQ:-100}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
INITIAL_VAL_REPEAT_TIMES=${INITIAL_VAL_REPEAT_TIMES:-1}
SAVE_FREQ=${SAVE_FREQ:-500}
SAVE_START_STEP=${SAVE_START_STEP:-4000}
SAVE_BY_TEST_ACC_LENGTH_ENABLED=${SAVE_BY_TEST_ACC_LENGTH_ENABLED:-False}
SAVE_BY_TEST_ACC_LENGTH_THRESHOLD=${SAVE_BY_TEST_ACC_LENGTH_THRESHOLD:-4.65}
train_epochs=${train_epochs:-10}
stream_weight=${stream_weight:-1.0}
rejected_draft_stream_weight=${rejected_draft_stream_weight:-1.0}
REJECTED_DRAFT_POSITION_DECAY_ENABLED=${REJECTED_DRAFT_POSITION_DECAY_ENABLED:-True}
REJECTED_DRAFT_POSITION_DECAY=${REJECTED_DRAFT_POSITION_DECAY:-0.9}

ACCEPT_DRAFT_POSITION_DECAY_ENABLED=${ACCEPT_DRAFT_POSITION_DECAY_ENABLED:-False}
REJECTED_DRAFT_FIRST_TOKEN_ONLY=${REJECTED_DRAFT_FIRST_TOKEN_ONLY:-True}

STUDENT_WORLD_SIZE=${STUDENT_WORLD_SIZE:-7}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-21}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
STUDENT_MAX_TOKEN_LEN_PER_GPU=$(( PPO_MICRO_BATCH_SIZE_PER_GPU * (MAX_PROMPT + MAX_RESPONSE_LENGTH) ))
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_WORKER_COUNT=${ROLLOUT_SPEED_TEST_WORKER_COUNT:-$STUDENT_WORLD_SIZE}
ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK=${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK:-$(( ROLLOUT_SPEED_TEST_WORKER_COUNT * 2 ))}

if (( TRAIN_PROMPT_BSZ % ROLLOUT_AGENT_NUM_WORKERS != 0 )); then
    echo "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ} must be divisible by ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS}." >&2
    exit 1
fi

RANDOM_RESPONSE_ANCHOR_ENABLED=${RANDOM_RESPONSE_ANCHOR_ENABLED:-False}
RANDOM_RESPONSE_ANCHOR_SEED=${RANDOM_RESPONSE_ANCHOR_SEED:-42}
DFLASH_LM_HEAD_CHUNK_SIZE=${DFLASH_LM_HEAD_CHUNK_SIZE:-256}
TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.3}

MAIN_MODEL_PATH=${MAIN_MODEL_PATH:-"your Qwen3-8B path"}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"$MAIN_MODEL_PATH"}
# /mnt/shared-storage-user/leihaodi/opd/verl/checkpoints/verl-dflash-opd/fsdp/student-teacher-0503/loss-forward-kl-k3/train-16w_plus_test_user_prompt-update-accumulation-steps/global_step_1900/draft_model
# /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-4b-dflash_data/epoch_5_step_295000
# /mnt/shared-storage-user/leihaodi/imo/SpecForge/outputs/qwen3-8b-dflash/epoch_5_step_280000
DRAFT_MODEL_PATH=${DRAFT_MODEL_PATH:-"bingyang-lei/dflash-qwen3-8b-thinking"}

TRAIN_JSONL=${TRAIN_JSONL:-"your train data"}
TRAIN_JSONL_FILENAME="$(basename "$TRAIN_JSONL")"
TRAIN_JSONL_NAME="${TRAIN_JSONL_FILENAME%.jsonl}"
TRAIN_FILES="['$TRAIN_JSONL']"
TODAY=$(date +"%m-%d")
EXP_NAME=${EXP_NAME:-"exp-qwen3-8b-tvloss/reject-first-token-no-decay_${REJECTED_DRAFT_FIRST_TOKEN_ONLY}"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/verl-dflash-opd/${EXP_NAME}"}

exec "${SCRIPT_DIR}/run_qwen_gsm8k.sh" \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.model.path="${MAIN_MODEL_PATH}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_main_model_path="${MAIN_MODEL_PATH}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_draft_model_path="${DRAFT_MODEL_PATH}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_lm_head_chunk_size="${DFLASH_LM_HEAD_CHUNK_SIZE}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_enabled="${RANDOM_RESPONSE_ANCHOR_ENABLED}" \
    ++actor_rollout_ref.model.override_config.verl_dflash_random_response_anchor_seed="${RANDOM_RESPONSE_ANCHOR_SEED}" \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.speculative_draft_model_path="${DRAFT_MODEL_PATH}" \
    ++actor_rollout_ref.rollout.engine_kwargs.sglang.mem_fraction_static="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${STUDENT_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.agent.num_workers="${ROLLOUT_AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${STUDENT_WORLD_SIZE}" \
    trainer.n_gpus_per_node="${STUDENT_WORLD_SIZE}" \
    trainer.rollout_speed_test_worker_count="${ROLLOUT_SPEED_TEST_WORKER_COUNT}" \
    trainer.rollout_speed_test_max_samples_per_benchmark="${ROLLOUT_SPEED_TEST_MAX_SAMPLES_PER_BENCHMARK}" \
    distillation.n_gpus_per_node="${TEACHER_WORLD_SIZE}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL_PATH}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEMORY_UTILIZATION}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.use_tv_loss="${USE_TV_LOSS}" \
    distillation.distillation_loss.reverse_kl_weight="${REVERSE_KL_WEIGHT}" \
    distillation.distillation_loss.forward_kl_weight="${FORWARD_KL_WEIGHT}" \
    distillation.distillation_loss.rejected_draft_use_reverse_kl="${REJECTED_DRAFT_USE_REVERSE_KL}" \
    distillation.distillation_loss.response_stream_weight="${stream_weight}" \
    distillation.distillation_loss.rejected_draft_stream_weight="${rejected_draft_stream_weight}" \
    distillation.distillation_loss.rejected_draft_position_decay_enabled="${REJECTED_DRAFT_POSITION_DECAY_ENABLED}" \
    distillation.distillation_loss.rejected_draft_position_decay="${REJECTED_DRAFT_POSITION_DECAY}" \
    distillation.distillation_loss.accept_draft_position_decay_enabled="${ACCEPT_DRAFT_POSITION_DECAY_ENABLED}" \
    distillation.distillation_loss.rejected_draft_first_token_only="${REJECTED_DRAFT_FIRST_TOKEN_ONLY}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.initial_val_repeat_times="${INITIAL_VAL_REPEAT_TIMES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_start_step="${SAVE_START_STEP}" \
    trainer.save_by_test_metric_enabled="${SAVE_BY_TEST_ACC_LENGTH_ENABLED}" \
    trainer.save_by_test_metric_key="test_rollout/mbpp/sglang_spec_accept_length" \
    trainer.save_by_test_metric_threshold="${SAVE_BY_TEST_ACC_LENGTH_THRESHOLD}" \
    trainer.total_epochs="${train_epochs}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    "$@"
