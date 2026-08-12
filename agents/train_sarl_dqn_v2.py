from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)
from sumo_env import SumoTSCEnv
import os

# ── Custom callback to log travel time and delay to TensorBoard ────────
class EpisodeMetricsCallback(BaseCallback):
    """
    Logs mean travel time and mean delay at the end of each episode.
    These are captured from tripinfo XML parsed in the environment.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        # Check if any episode just ended this step
        dones = self.locals.get("dones", [False])
        if any(dones):
            try:
                # Access the underlying env through VecEnv wrapper
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

# ── Config ─────────────────────────────────────────────────────────────
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
    "w_throughput" : 2.0,   # throughput incentive — new for v2
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)

env = SumoTSCEnv(config)

# ── DQN agent — same hyperparameters as v1 for fair comparison ─────────
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

print("Starting SARL DQN v2 training — throughput reward enabled")
print(f"w_throughput = {config['w_throughput']}")
print(f"Action space:      {env.action_space}")
print(f"Observation space: {env.observation_space}\n")

# ── Callbacks ──────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_dqn_v2_2int"
)
metrics_cb = EpisodeMetricsCallback()

callbacks = CallbackList([checkpoint_cb, metrics_cb])

# ── Train ──────────────────────────────────────────────────────────────
model.learn(
    total_timesteps = 260000,
    callback        = callbacks,
    tb_log_name     = "sarl_dqn_v2_2int"
)

model.save("sarl_dqn_v2_2int_260k")
print("Training complete. Model saved as sarl_dqn_v2_2int_260k.zip")
env.close()
