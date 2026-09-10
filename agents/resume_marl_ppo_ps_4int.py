import os
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from sumo_env_marl_ps import SumoMARLEnvPS

TOTAL_TIMESTEPS = 260000

CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "checkpoints/marl_ppo_ps_4int_130000_steps.zip"


class SharedModelWrapper(gym.Env):
    """
    Gymnasium wrapper that gives the shared PPO model a standard
    gym interface. The shared model sees every agent's experience
    by alternating whose observation is presented each step.
    The wrapper coordinates joint actions internally.
    """

    def __init__(self, marl_env, primary_agent):
        super().__init__()
        self.marl_env      = marl_env
        self.primary_agent = primary_agent
        self.other_agents  = [a for a in marl_env.agent_ids
                               if a != primary_agent]

        self.observation_space = marl_env.observation_space
        self.action_space      = marl_env.action_space

        self._last_obs_dict = None
        self.shared_model   = [None]  # mutable ref set after model load

    def reset(self, seed=None, options=None):
        obs_dict            = self.marl_env.reset(seed=seed)
        self._last_obs_dict = obs_dict
        return obs_dict[self.primary_agent], {}

    def step(self, action):
        actions = {self.primary_agent: int(action)}

        # Other agents also use the SAME shared model
        for other in self.other_agents:
            if (self.shared_model[0] is not None
                    and self._last_obs_dict is not None):
                other_action, _ = self.shared_model[0].predict(
                    self._last_obs_dict[other],
                    deterministic=False
                )
                actions[other] = int(other_action)
            else:
                actions[other] = self.marl_env.action_space.sample()

        obs_dict, rewards, terminateds = self.marl_env.step(actions)
        self._last_obs_dict = obs_dict

        return (
            obs_dict[self.primary_agent],
            rewards[self.primary_agent],
            terminateds[self.primary_agent],
            False,
            {}
        )

    def close(self):
        pass


# ── Config ─────────────────────────────────────────────────────────────
config = {
    "sumocfg"  : "4int_files/4int_osm.sumocfg",
    "agent_ids": ["tl1", "tl2", "tl3", "tl4"],
    "tl_map"   : {
        "tl1": "30406197",
        "tl2": "cluster_30406198_7161921427_7161921428",
        "tl3": "54994558",
        "tl4": "cluster_54994556_7161921429",
    },
    "lane_map" : {
        "tl1": [
            "596638461#0_0",
            "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3"
        ],
        "tl2": [
            "-621271001_0", "669676574#2_0", "1190301745#0_0",
            "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2"
        ],
        "tl3": [
            "380081149#1_0", "42790161#2_0", "42790161#2_1", "42790161#2_2"
        ],
        "tl4": [
            "-621271003_0", "669676574#1_0",
            "623540942#0_0", "623540942#0_1", "623540942#0_2",
            "-669676574#2_0", "621271001_0"
        ],
    },
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

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("models",      exist_ok=True)

# ── Shared MARL-PS environment ─────────────────────────────────────────
marl_env = SumoMARLEnvPS(config)

# ── Wrapper — TL1 is the primary agent driving the SB3 training loop ──
wrapper = SharedModelWrapper(marl_env, primary_agent="tl1")

# ── Load from 130k checkpoint ──────────────────────────────────────────
print(f"Loading {CKPT_PATH} ...")
model = PPO.load(
    CKPT_PATH,
    env             = wrapper,
    tensorboard_log = "./logs/",
)
wrapper.shared_model[0] = model

remaining = TOTAL_TIMESTEPS - model.num_timesteps
print(f"Resumed at step {model.num_timesteps}; "
      f"{remaining} steps remaining.")

# ── Resume training ────────────────────────────────────────────────────
checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "marl_ppo_ps_4int",
)

model.learn(
    total_timesteps     = remaining,
    callback            = checkpoint,
    reset_num_timesteps = False,
    tb_log_name         = "marl_ppo_ps_4int_resumed",
)

# ── Save ───────────────────────────────────────────────────────────────
model.save("models/marl_ppo_ps_4int_260k")
print("\nTraining complete.")
print("Saved: models/marl_ppo_ps_4int_260k.zip")

marl_env.close()