import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env_marl import SumoMARLEnv
import os

class AgentEnvWrapper(gym.Env):
    def __init__(self, marl_env, agent_id, other_model_ref):
        super().__init__()
        self.marl_env        = marl_env
        self.agent_id        = agent_id
        self.other_model_ref = other_model_ref
        self.observation_space = marl_env.observation_space
        self.action_space      = marl_env.action_space
        self._last_obs_dict  = None
        self._terminated     = False

    def reset(self, seed=None, options=None):
        obs_dict             = self.marl_env.reset(seed=seed)
        self._last_obs_dict  = obs_dict
        self._terminated     = False
        return obs_dict[self.agent_id], {}

    def step(self, action):
        other_agents = [a for a in self.marl_env.agent_ids
                        if a != self.agent_id]
        other_agent  = other_agents[0]
        actions      = {self.agent_id: int(action)}

        if (self.other_model_ref[0] is not None
                and self._last_obs_dict is not None):
            other_action, _ = self.other_model_ref[0].predict(
                self._last_obs_dict[other_agent],
                deterministic=False
            )
            actions[other_agent] = int(other_action)
        else:
            actions[other_agent] = self.marl_env.action_space.sample()

        obs_dict, rewards, terminateds = self.marl_env.step(actions)
        self._last_obs_dict = obs_dict
        self._terminated    = terminateds[self.agent_id]

        return (
            obs_dict[self.agent_id],
            rewards[self.agent_id],
            terminateds[self.agent_id],
            False,
            {}
        )

    def close(self):
        pass


# ── Config ─────────────────────────────────────────────────────────────
config = {
    "sumocfg"  : "2int_files/2int.sumocfg",
    "agent_ids": ["tl1", "tl2"],
    "tl_map"   : {
        "tl1": "30406197",
        "tl2": "cluster_30406198_7161921427_7161921428"
    },
    "lane_map" : {
        "tl1": [
            "596638461#0_0",
            "4761588#1_0", "4761588#1_1",
            "4761588#1_2", "4761588#1_3"
        ],
        "tl2": [
            "-621271001_0", "669676574#0_0", "1190301745#0_0",
            "-669676574#8_0", "622835372_0",
            "622835372_1",   "622835372_2"
        ]
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
os.makedirs("models",      exist_ok=True)

# ── Shared MARL environment ────────────────────────────────────────────
marl_env = SumoMARLEnv(config)

# ── Model references ───────────────────────────────────────────────────
tl1_model_ref = [None]
tl2_model_ref = [None]

env_tl1 = AgentEnvWrapper(marl_env, "tl1", tl1_model_ref)
env_tl2 = AgentEnvWrapper(marl_env, "tl2", tl2_model_ref)

# ── Load from 60k checkpoints ──────────────────────────────────────────
print("Loading checkpoints from 60,000 steps...")
model_tl1 = PPO.load(
    "checkpoints/marl_ppo_tl1_60000",
    env             = env_tl1,
    tensorboard_log = "./logs/"
)
model_tl2 = PPO.load(
    "checkpoints/marl_ppo_tl2_60000",
    env             = env_tl2,
    tensorboard_log = "./logs/"
)
print("Checkpoints loaded. Continuing for remaining 200,000 steps...\n")

# ── Wire cross-references ──────────────────────────────────────────────
tl1_model_ref[0] = model_tl2
tl2_model_ref[0] = model_tl1

# ── Resume training ────────────────────────────────────────────────────
REMAINING_STEPS = 200000   # 260k - 60k
SEGMENT         = 40

steps_done = 0

while steps_done < REMAINING_STEPS:
    model_tl1.learn(
        total_timesteps     = SEGMENT,
        reset_num_timesteps = False,
        tb_log_name         = "marl_ppo_tl1_continued",
        progress_bar        = False
    )
    model_tl2.learn(
        total_timesteps     = SEGMENT,
        reset_num_timesteps = False,
        tb_log_name         = "marl_ppo_tl2_continued",
        progress_bar        = False
    )

    steps_done += SEGMENT

    if steps_done % 10000 == 0:
        model_tl1.save(f"checkpoints/marl_ppo_tl1_{60000 + steps_done}")
        model_tl2.save(f"checkpoints/marl_ppo_tl2_{60000 + steps_done}")
        print(f"Step {60000 + steps_done}/260000 — checkpoints saved")

# ── Save final models ──────────────────────────────────────────────────
model_tl1.save("models/marl_ppo_tl1_260k")
model_tl2.save("models/marl_ppo_tl2_260k")
print("\nTraining complete.")
print("Saved: models/marl_ppo_tl1_260k.zip")
print("Saved: models/marl_ppo_tl2_260k.zip")

marl_env.close()