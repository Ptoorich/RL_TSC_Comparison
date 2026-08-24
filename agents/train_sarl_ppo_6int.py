from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env_6int import SumoTSCEnv6Int
import os

config = {
    "sumocfg"  : "6int_files/6int.sumocfg",

    # All 6 traffic light IDs from discovery script
    "tl_ids"   : [
        "30406195",
        "30406197",
        "54994558",
        "768241766",
        "cluster_30406198_7161921427_7161921428",
        "cluster_54994556_7161921429"
    ],

    # All approach lanes per intersection from discovery script
    "link_ids" : [
        # 30406195
        "7521010#0_0", "622835369_0", "622835369_1", "622835369_3",
        # 30406197
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        # 54994558
        "380081149#2_0",
        "42790161#2_0", "42790161#2_1", "42790161#2_2",
        # 768241766
        "42790161#0_0", "42790161#0_1", "42790161#0_2", "7521010#1_0",
        # cluster_30406198_7161921427_7161921428
        "-621271001_0", "669676574#2_0", "1190301745#0_0",
        "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2",
        # cluster_54994556_7161921429 (6-phase junction)
        "-621271003_0", "669676574#1_0",
        "623540942#0_0", "623540942#0_1", "623540942#0_2",
        "-669676574#2_0", "621271001_0"
    ],

    # Phase parameters — calibrated to real network
    "cycle_length" : 90,
    "yellow_time"  : 6,
    "allred_time"  : 0,
    "init_split"   : 39,
    "s_lb"         : 15,
    "s_ub"         : 63,
    "delta_s"      : 5,

    # Reward — identical to 2-intersection SARL PPO
    "q_lc"         : 5,
    "q_hc"         : 15,
    "w_l"          : 1.0,
    "w_cp"         : 3.0,
    "w_throughput" : 0.0,   # queue penalty only — same as 2int SARL PPO

    # Episode parameters
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

env = SumoTSCEnv6Int(config)

# PPO hyperparameters identical to 2-intersection SARL PPO
model = PPO(
    policy          = "MlpPolicy",
    env             = env,
    learning_rate   = 0.0003,
    n_steps         = 40,
    batch_size      = 10,
    n_epochs        = 10,
    gamma           = 0.99,
    verbose         = 1,
    tensorboard_log = "./logs/"
)

print("=== SARL PPO Training — 6 Intersection Network ===")
print(f"Intersections:   {len(config['tl_ids'])}")
print(f"Monitored lanes: {len(config['link_ids'])}")
print(f"Obs size:        {len(config['link_ids']) + len(config['tl_ids'])}")
print(f"Action space:    Discrete({3 * len(config['tl_ids'])})")
print(f"6-phase junction: cluster_54994556_7161921429")
print(f"Total timesteps: 260,000\n")

checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_ppo_6int"
)

model.learn(
    total_timesteps = 260000,
    callback        = checkpoint,
    tb_log_name     = "sarl_ppo_6int"
)

model.save("models/sarl_ppo_6int_260k")
print("\nTraining complete.")
print("Saved: models/sarl_ppo_6int_260k.zip")
env.close()