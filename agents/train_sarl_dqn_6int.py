import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)

from sumo_env_6int import SumoTSCEnv6Int

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 260000


# ── Custom callback to log travel time and delay to TensorBoard ────────
class EpisodeMetricsCallback(BaseCallback):
    """
    Logs mean travel time and mean delay at the end of each episode.
    These are captured from tripinfo XML parsed in the environment.
    (Same pattern used in train_sarl_ppo_4int.py / train_sarl_dqn_4int.py —
    worth adding to train_sarl_ppo_6int.py too if you want tensorboard
    parity across all three 6int runs.)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        dones = self.locals.get("dones", [False])
        if any(dones):
            try:
                env = self.training_env.envs[0]
                self.logger.record(
                    "metrics/mean_travel_time",
                    env.mean_travel_time
                )
                self.logger.record(
                    "metrics/mean_delay",
                    env.mean_delay
                )
            except Exception:
                pass
        return True


# Run this script from agents/ — matches the same relative sumocfg path
# used by train_sarl_ppo_6int.py.
config = {
    "sumocfg"  : "6int_files/6int.sumocfg",
    "tl_ids"   : [
        "30406195",
        "30406197",
        "54994558",
        "768241766",
        "cluster_30406198_7161921427_7161921428",
        "cluster_54994556_7161921429"
    ],
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
    "w_throughput" : 0.0,   # queue penalty only — same as PPO 6int
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

# ── Step 1: Validate environment against Gymnasium standard ───────────
print("Checking environment...")
env = SumoTSCEnv6Int(config)
check_env(env, warn=True)
print("Environment check passed.\n")

# ── Step 2: Create DQN agent (same hyperparameters as sarl_dqn_2int/4int) ─
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
    tensorboard_log         = "./logs/",
)

# ── Step 3: Callbacks ──────────────────────────────────────────────────
checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_dqn_6int",
)
metrics_cb = EpisodeMetricsCallback()

callbacks = CallbackList([checkpoint, metrics_cb])

# ── Step 4: Train ─────────────────────────────────────────────────────
print("=== SARL DQN Training — 6 Intersection Network ===")
print(f"Intersections:   {len(config['tl_ids'])}")
print(f"Monitored lanes: {len(config['link_ids'])}")
print(f"Obs size:        {len(config['link_ids']) + len(config['tl_ids'])}")
print(f"Action space:    Discrete({3 * len(config['tl_ids'])})")
print(f"6-phase junction: cluster_54994556_7161921429")
print(f"Total timesteps: {TOTAL_TIMESTEPS}\n")

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = callbacks,
    tb_log_name     = "sarl_dqn_6int",
)

# ── Step 5: Save the trained model ────────────────────────────────────
final_name = (f"models/sarl_dqn_6int_{TOTAL_TIMESTEPS}"
              if TOTAL_TIMESTEPS != 260000 else "models/sarl_dqn_6int_260k")
model.save(final_name)
print(f"\nTraining complete. Model saved as {final_name}.zip")

env.close()