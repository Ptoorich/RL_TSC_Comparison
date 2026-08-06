from stable_baselines3 import PPO
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

# ── Load environment ───────────────────────────────────────────────────
env = SumoTSCEnv(config)

# ── Load existing model and attach TensorBoard log directory ───────────
print("Loading previous model...")
model = PPO.load(
    "sarl_ppo_2int",
    env             = env,
    tensorboard_log = "./logs/"
)
print("Loaded successfully. Continuing training for 250,000 steps...")

# ── Checkpoint every 10,000 steps ─────────────────────────────────────
checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_ppo_2int"
)

# ── Train ──────────────────────────────────────────────────────────────
model.learn(
    total_timesteps     = 250000,
    callback            = checkpoint,
    reset_num_timesteps = False,
    tb_log_name         = "sarl_ppo_2int_continued"
)

# ── Save final model ───────────────────────────────────────────────────
model.save("sarl_ppo_2int_260k")
print("Training complete. Model saved as sarl_ppo_2int_260k.zip")

env.close()