import os
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)

from sumo_env_4int import SumoTSCEnv

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 260000


# ── Custom callback to log travel time and delay to TensorBoard ────────
class EpisodeMetricsCallback(BaseCallback):
    """
    Logs mean travel time and mean delay at the end of each episode.
    These are captured from tripinfo XML parsed in the environment.
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


# Run this script from agents/ — matches the actual location of
# agents/4int_files/4int_osm.sumocfg (one level down, no ../.. needed).
config = {
    "sumocfg"      : "4int_files/4int_osm.sumocfg",
    "tl_ids"       : [
        "30406197",
        "cluster_30406198_7161921427_7161921428",
        "54994558",
        "cluster_54994556_7161921429",
    ],
    "link_ids"     : [
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        "-621271001_0", "669676574#2_0", "1190301745#0_0",
        "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2",
        "380081149#1_0", "42790161#2_0", "42790161#2_1", "42790161#2_2",
        "-621271003_0", "669676574#1_0", "623540942#0_0",
        "623540942#0_1", "623540942#0_2", "-669676574#2_0", "621271001_0",
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

# ── Step 2: Create DQN agent (same hyperparameters as sarl_dqn_2int) ──
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
    name_prefix = "sarl_dqn_4int",
)
metrics_cb = EpisodeMetricsCallback()

callbacks = CallbackList([checkpoint, metrics_cb])

# ── Step 4: Train ─────────────────────────────────────────────────────
print(f"Starting DQN training on 4-intersection network for "
      f"{TOTAL_TIMESTEPS} steps...")
print(f"Action space:      {env.action_space}")
print(f"Observation space: {env.observation_space}\n")

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = callbacks,
    tb_log_name     = "sarl_dqn_4int",
)

# ── Step 5: Save the trained model ────────────────────────────────────
final_name = (f"sarl_dqn_4int_{TOTAL_TIMESTEPS}"
              if TOTAL_TIMESTEPS != 260000 else "sarl_dqn_4int_260k")
model.save(final_name)
print(f"\nTraining complete. Model saved as {final_name}.zip")

env.close()