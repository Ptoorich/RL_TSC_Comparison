from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoTSCEnv
import os

# ── Change only this value between screening runs ──────────────────────
W_THROUGHPUT = 20.0   # candidate weight to test
RUN_NAME     = f"screen_dqn_wtp{int(W_THROUGHPUT)}"

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
    "w_throughput" : W_THROUGHPUT,
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)

env = SumoTSCEnv(config)

model = DQN(
    policy                  = "MlpPolicy",
    env                     = env,
    learning_rate           = 0.0001,
    buffer_size             = 50000,
    learning_starts         = 1000,
    batch_size              = 32,
    gamma                   = 0.99,
    target_update_interval  = 1000,
    exploration_fraction    = 0.2,
    exploration_initial_eps = 1.0,
    exploration_final_eps   = 0.05,
    train_freq              = 4,
    gradient_steps          = 1,
    verbose                 = 1,
    tensorboard_log         = "./logs/"
)

print(f"Screening run: w_throughput = {W_THROUGHPUT}")
print(f"Run name: {RUN_NAME}")
print(f"Training for 30,000 steps (screening only)\n")

model.learn(
    total_timesteps = 30000,
    tb_log_name     = RUN_NAME
)

model.save(f"screening/{RUN_NAME}")
print(f"\nScreening model saved to screening/{RUN_NAME}.zip")
env.close()