#!/bin/bash
# DAPO baseline training on Qwen2.5-Math-7B.
# Usage:
#   conda activate rsp-dapo
#   bash train/train_dapo.sh

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASH_ATTN_VERSION=2
export NCCL_DEBUG=WARN
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_ENTITY="${WANDB_ENTITY:-anonymous}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Math-7B}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/simplelr_math_35}"
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/checkpoints/dapo_qwen25_math_7b_baseline}"
LOGS_DIR="${LOGS_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOGS_DIR"

RAY_TMPDIR="/tmp/ray_dapo_baseline"
mkdir -p "$RAY_TMPDIR"
ray start --head --port=7013 --dashboard-port=9013 --temp-dir="$RAY_TMPDIR" --num-gpus 4 2>/dev/null || true
NODE_IP=$(hostname -i | awk '{print $1}')
export RAY_ADDRESS="${NODE_IP}:7013"
sleep 5

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/test.parquet" \
  data.train_batch_size=1024 \
  data.val_batch_size=500 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  ++data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  algorithm.kl_ctrl.kl_coef=0.0 \
  ++actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  ++actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.top_k=50 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  ++actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
  ++actor_rollout_ref.rollout.micro_rollout_batch_size=384 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  custom_reward_function.path="${REPO_ROOT}/train/reward/custom_score.py" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  "trainer.logger=['console','wandb']" \
  trainer.project_name=RSP_DAPO \
  trainer.experiment_name=dapo_qwen25_math_7b_baseline \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=13 \
  trainer.resume_mode=auto \
  trainer.default_local_dir="${CKPTS_DIR}" \
  2>&1 | tee "${LOGS_DIR}/dapo_qwen25_math_7b_baseline.log"
