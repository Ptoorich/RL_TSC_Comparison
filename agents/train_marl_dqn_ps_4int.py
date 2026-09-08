import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from sumo_env_marl_ps import SumoMARLEnvPS
import os


class SharedModelWrapper(gym.Env):
    """
    Gymnasium wrapper that gives the shared DQN model a standard
    gym interface. The shared model sees every agent's experience
    by alternating whose observation is presented each step.
    The wrapper coordinates joint actions internally.

    Other agents' actions are drawn via shared_model.predict(...,
    deterministic=False), which for DQN samples epsilon-greedily off
    the model's current exploration_rate — the same mechanism SB3
    uses internally during training, so the other agents follow the
    same exploration schedule as the primary agent.
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
        self.shared_model   = [None]  # mutable ref set after model creation

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
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

# ── Shared MARL-PS environment ─────────────────────────────────────────
marl_env = SumoMARLEnvPS(config)

# ── Wrapper — TL1 is the primary agent driving the SB3 training loop ──
# The other 3 agents use the same shared model via shared_model reference
wrapper = SharedModelWrapper(marl_env, primary_agent="tl1")

# ── Single shared DQN model — same hyperparameters as train_sarl_dqn.py /
# train_sarl_dqn_6int.py, for consistency across the SARL/MARL comparison ─
shared_model = DQN(
    policy                  = "MlpPolicy",
    env                     = wrapper,
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

# ── Wire shared model reference into wrapper ───────────────────────────
wrapper.shared_model[0] = shared_model

print("=== MARL DQN with Parameter Sharing — 4 Intersection Network ===")
print(f"Agents:          {config['agent_ids']}")
print(f"Obs size:        {marl_env.obs_size} (local + normalised ID)")
for agent in config["agent_ids"]:
    print(f"  {agent} ID:  {marl_env.agent_index[agent]:.4f}")
print(f"Action space:    Discrete(3) per agent")
print(f"Shared model:    ONE DQN network for all 4 agents")
print(f"Reward:          Global network queue penalty (shared)")
print(f"Total timesteps: 260,000\n")

# ── Train ──────────────────────────────────────────────────────────────
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint = CheckpointCallback(
    save_freq   = 10000,
    save_path   = "./checkpoints/",
    name_prefix = "marl_dqn_ps_4int"
)

shared_model.learn(
    total_timesteps = 260000,
    callback        = checkpoint,
    tb_log_name     = "marl_dqn_ps_4int"
)

# ── Save ───────────────────────────────────────────────────────────────
shared_model.save("models/marl_dqn_ps_4int_260k")
print("\nTraining complete.")
print("Saved: models/marl_dqn_ps_4int_260k.zip")

marl_env.close()