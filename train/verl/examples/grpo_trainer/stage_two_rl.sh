set -x


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/eurus_prime/train.parquet \
    data.val_files=$HOME/data/eurus_prime/val.parquet \
    reward_model.reward_manager=prime \
    data.shuffle=True \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32\
    data.truncation='error' \
    actor_rollout_ref.model.path= # entry your model path after stage1_rl
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path='/workspace/verl/verl/utils/reward_score/code_gen.py' \
    custom_reward_function.name='compute_score_prime' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='stage2_rl' \
    trainer.experiment_name='stage2_rl_execverify7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.total_epochs=5 $@