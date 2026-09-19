from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoTSCEnv

config = {
    "sumocfg"      : "2int_files/2int.sumocfg",
    "tl_ids"       : [
        "30406197",
        "cluster_30406198_7161921427_7161921428"
    ],
    "link_ids"     : [
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        "-621271001_0", "669676574#0_0", "1190301745#0_0",
        "-669676574#8_0", "622835372_0", "622835372_1", "622835372_2"
    ],
    "cycle_length" : 90,
    "yellow_time"  : 6,
    "allred_time"  : 0,
    "init_split"   : 39,
    "s_lb"         : 15,
    "s_ub"         : 63,
    "delta_s"      : 5,
    "q_lc"         : 5,
    "q_hc"         : 15,
    "w_l"          : 1.0,
    "w_cp"         : 3.0,
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

env = SumoTSCEnv(config)

# ── DQN Agent ──────────────────────────────────────────────────────────
# Hyperparameters informed by Van der Pol (2016):
# - Replay buffer 50,000 (larger buffer more stable long term)
# - Low learning rate 0.0001 (higher rates cause oscillation)
# - Batch size 32
# - Target update interval 1000 (equivalent to freeze interval)
# - Epsilon decays from 1.0 to 0.1 over first 20% of training
model = DQN(
    policy                 = "MlpPolicy",
    env                    = env,
    learning_rate          = 0.0001,
    buffer_size            = 50000,
    learning_starts        = 1000,    # collect 1000 steps before learning
    batch_size             = 32,
    gamma                  = 0.99,
    target_update_interval = 1000,    # update target network every 1000 steps
    exploration_fraction   = 0.2,     # epsilon decays over first 20% of training
    exploration_initial_eps= 1.0,     # start fully random
    exploration_final_eps  = 0.05,    # end mostly greedy
    train_freq             = 4,       # update every 4 steps
    gradient_steps         = 1,
    verbose                = 1,
    tensorboard_log        = "./logs/"
)

print("Starting DQN training on 2-intersection network...")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}\n")

# ── Checkpointing ──────────────────────────────────────────────────────
checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_dqn_2int"
)

# ── Train ──────────────────────────────────────────────────────────────
model.learn(
    total_timesteps = 260000,    # same as PPO run for fair comparison
    callback        = checkpoint,
    tb_log_name     = "sarl_dqn_2int"
)

# ── Save ───────────────────────────────────────────────────────────────
model.save("sarl_dqn_2int_260k")
print("\nTraining complete. Model saved as sarl_dqn_2int_260k.zip")

env.close()