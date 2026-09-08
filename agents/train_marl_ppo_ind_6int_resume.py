import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env_marl_ind import SumoMARLEnvInd
import os

RESUME_FROM     = 190000
TOTAL_TIMESTEPS = 260000
REMAINING       = TOTAL_TIMESTEPS - RESUME_FROM
SEGMENT         = 40  # matches n_steps, same as the original training run


# ── Single-agent Gymnasium wrapper (identical to train_marl_ppo_ind_6int.py) ─
class AgentEnvWrapper(gym.Env):
    def __init__(self, marl_env, agent_id, other_model_refs):
        super().__init__()
        self.marl_env         = marl_env
        self.agent_id         = agent_id
        self.other_model_refs = other_model_refs

        self.observation_space = marl_env.observation_space
        self.action_space      = marl_env.action_space

        self._last_obs_dict = None

    def reset(self, seed=None, options=None):
        obs_dict            = self.marl_env.reset(seed=seed)
        self._last_obs_dict = obs_dict
        return obs_dict[self.agent_id], {}

    def step(self, action):
        actions = {self.agent_id: int(action)}

        for other_agent, ref in self.other_model_refs.items():
            if ref[0] is not None and self._last_obs_dict is not None:
                other_action, _ = ref[0].predict(
                    self._last_obs_dict[other_agent],
                    deterministic=False
                )
                actions[other_agent] = int(other_action)
            else:
                actions[other_agent] = self.marl_env.action_space.sample()

        obs_dict, rewards, terminateds = self.marl_env.step(actions)
        self._last_obs_dict = obs_dict

        return (
            obs_dict[self.agent_id],
            rewards[self.agent_id],
            terminateds[self.agent_id],
            False,
            {}
        )

    def close(self):
        pass


# ── Config — identical to train_marl_ppo_ind_6int.py ────────────────────
config = {
    "sumocfg"  : "6int_files/6int.sumocfg",
    "agent_ids": ["tl1", "tl2", "tl3", "tl4", "tl5", "tl6"],
    "tl_map"   : {
        "tl1": "30406195",
        "tl2": "30406197",
        "tl3": "54994558",
        "tl4": "768241766",
        "tl5": "cluster_30406198_7161921427_7161921428",
        "tl6": "cluster_54994556_7161921429",
    },
    "lane_map" : {
        "tl1": [
            "7521010#0_0", "622835369_0", "622835369_1", "622835369_3"
        ],
        "tl2": [
            "596638461#0_0",
            "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3"
        ],
        "tl3": [
            "380081149#2_0", "42790161#2_0", "42790161#2_1", "42790161#2_2"
        ],
        "tl4": [
            "42790161#0_0", "42790161#0_1", "42790161#0_2", "7521010#1_0"
        ],
        "tl5": [
            "-621271001_0", "669676574#2_0", "1190301745#0_0",
            "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2"
        ],
        "tl6": [
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
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

marl_env = SumoMARLEnvInd(config)

model_refs = {agent: [None] for agent in config["agent_ids"]}

envs = {}
for agent in config["agent_ids"]:
    other_refs = {a: model_refs[a] for a in config["agent_ids"] if a != agent}
    envs[agent] = AgentEnvWrapper(marl_env, agent, other_refs)

# ── Load each agent's 190k checkpoint, attach its env ───────────────────
print("Loading checkpoints from step", RESUME_FROM, "...")
models = {}
for agent in config["agent_ids"]:
    checkpoint_path = f"checkpoints/marl_ppo_ind_6int_{agent}_{RESUME_FROM}"
    models[agent] = PPO.load(
        checkpoint_path,
        env             = envs[agent],
        tensorboard_log = "./logs/"
    )
    model_refs[agent][0] = models[agent]
    print(f"  Loaded {checkpoint_path}.zip")

print(f"\nResuming MARL PPO (independent) — 6 Intersection Network")
print(f"Resuming from step {RESUME_FROM}, training {REMAINING} more steps "
      f"to reach {TOTAL_TIMESTEPS}\n")

# ── Continue round-robin training ────────────────────────────────────────
steps_done = RESUME_FROM

while steps_done < TOTAL_TIMESTEPS:
    for agent in config["agent_ids"]:
        models[agent].learn(
            total_timesteps     = SEGMENT,
            reset_num_timesteps = False,
            tb_log_name         = f"marl_ppo_ind_6int_{agent}_continued",
            progress_bar        = False
        )

    steps_done += SEGMENT

    if steps_done % 10000 == 0:
        for agent in config["agent_ids"]:
            models[agent].save(f"checkpoints/marl_ppo_ind_6int_{agent}_{steps_done}")
        print(f"Step {steps_done}/{TOTAL_TIMESTEPS} — checkpoints saved")

# ── Save final models ─────────────────────────────────────────────────
for agent in config["agent_ids"]:
    models[agent].save(f"models/marl_ppo_ind_6int_{agent}_260k")
    print(f"Model saved: models/marl_ppo_ind_6int_{agent}_260k.zip")

print("\nTraining complete.")
marl_env.close()