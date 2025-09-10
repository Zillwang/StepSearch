export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='/mnt/GeneralModel/zhengxuhui/data/stepsearch'

PROJECT_NAME='StepSearch'


export BASE_MODEL='/mnt/GeneralModel/share/model/Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=debug
export SWANLAB_API_KEY=6TDIcifjvpp01pN9Ywd09

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/musi_stepsearch_train.parquet \
    data.val_files=$DATA_DIR/musi_stepsearch_test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.max_start_length=2048 \
    data.max_obs_length=800 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=5e-7 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=4 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.grad_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=60 \
    trainer.test_freq=60 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=6 \
    trainer.total_training_steps=1120\
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/mnt/GeneralModel/zhengxuhui/data/stepsearch/$EXPERIMENT_NAME \
    trainer.save_predictions=true \
    trainer.answer_check_method=step \
    trainer.redundancy_penalty=true \
    trainer.information_gain=true \
    trainer.search_steps_reward=true \
    trainer.search_key_reward=true \
    max_turns=5 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee /mnt/GeneralModel/zhengxuhui/data/stepsearch/log/$EXPERIMENT_NAME.log