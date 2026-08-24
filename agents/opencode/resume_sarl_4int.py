import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList
)

from sumo_env_4int import SumoTSCEnv

TOTAL_TIMESTEPS = 260000

CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/sarl_ppo_4int_160000_steps.zip"


# ── Custom callback to log travel time and delay to TensorBoard ────────
class EpisodeMetricsCallback(BaseCallback):
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


config = {
    "sumocfg"      : "../../4int_files/4int_osm.sumocfg",
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
    "use_gui"      : False,
}

env = SumoTSCEnv(config)

print(f"Loading {CKPT_PATH} ...")
model = PPO.load(
    CKPT_PATH,
    env             = env,
    tensorboard_log = "./logs/",
)
remaining = TOTAL_TIMESTEPS - model.num_timesteps
print(f"Resumed at step {model.num_timesteps}; "
      f"{remaining} steps remaining.")

checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "sarl_ppo_4int",
)
metrics_cb = EpisodeMetricsCallback()

model.learn(
    total_timesteps     = remaining,
    callback            = CallbackList([checkpoint, metrics_cb]),
    reset_num_timesteps = False,
    tb_log_name         = "sarl_ppo_4int_resumed",
)

model.save("sarl_ppo_4int_260k")
print("Training complete. Model saved as sarl_ppo_4int_260k.zip")

env.close()
