from stable_baselines3 import PPO, DQN
ppo_model = PPO.load("models/marl_ppo_ps_6int_260k")
dqn_model = DQN.load("models/marl_dqn_ps_6int_260k")

print(hasattr(dqn_model, "exploration_schedule"))  # DQN-specific
print(hasattr(ppo_model, "clip_range"))             # PPO-specific
print(dqn_model.policy)