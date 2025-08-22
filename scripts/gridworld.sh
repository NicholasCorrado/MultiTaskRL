python ppo_discrete_action_dro.py \
  --env_ids GridWorldEnv1 GridWorldEnv2 GridWorldEnv3 GridWorldEnv4 \
  --total_timesteps 500000 \
  --dro_eps 0.01 \
  --dro_learning_rate 1.0 \
  --dro_num_steps 128 \
  --dro_success_ref \
  --linear \
  --num_steps 128