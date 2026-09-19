from sumo_env import SumoTSCEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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
    "use_gui"      : False,  # GUI off for faster training
}

# ── Step 1: Validate environment against Gymnasium standard ───────────
print("Checking environment...")
env = SumoTSCEnv(config)
check_env(env, warn=True)
print("Environment check passed.\n")

# ── Step 2: Create PPO agent ───────────────────────────────────────────
model = PPO(
    policy       = "MlpPolicy",   # standard feedforward network
    env          = env,
    learning_rate= 0.0003,
    n_steps      = 40,            # steps per update (matches ~1 episode)
    batch_size   = 10,
    n_epochs     = 10,
    gamma        = 0.99,
    verbose      = 1,             # prints training progress
    tensorboard_log = "./logs/",  # optional: track in tensorboard
)

# ── Step 3: Train ─────────────────────────────────────────────────────
print("Starting training...")
model.learn(total_timesteps=260000)  

# ── Step 4: Save the trained model ────────────────────────────────────
model.save("sarl_ppo_2int")
print("Model saved as sarl_ppo_2int.zip")

env.close()