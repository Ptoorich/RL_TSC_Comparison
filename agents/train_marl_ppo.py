import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sumo_env_marl import SumoMARLEnv
import os

# ── Single-agent Gymnasium wrapper around the shared MARL environment ──
# SB3 requires a proper gym.Env — this wrapper gives each agent
# its own gym interface while sharing the underlying SUMO simulation
class AgentEnvWrapper(gym.Env):
    """
    Wraps one agent's view of the shared MARL environment
    into a standard Gymnasium interface for SB3 compatibility.

    The shared environment handles SUMO — this wrapper just
    exposes one agent's observation, action and reward to SB3.
    """

    def __init__(self, marl_env, agent_id, other_model_ref):
        super().__init__()
        self.marl_env       = marl_env
        self.agent_id       = agent_id
        self.other_model_ref= other_model_ref  # reference to other agent's model

        self.observation_space = marl_env.observation_space
        self.action_space      = marl_env.action_space

        self._last_obs_dict = None
        self._terminated    = False

    def reset(self, seed=None, options=None):
        obs_dict            = self.marl_env.reset(seed=seed)
        self._last_obs_dict = obs_dict
        self._terminated    = False
        return obs_dict[self.agent_id], {}

    def step(self, action):
        # Get the other agent's action using its current model
        other_agents = [a for a in self.marl_env.agent_ids
                        if a != self.agent_id]
        other_agent  = other_agents[0]

        actions = {self.agent_id: int(action)}

        # Other agent acts on its last observation
        if (self.other_model_ref[0] is not None
                and self._last_obs_dict is not None):
            other_action, _ = self.other_model_ref[0].predict(
                self._last_obs_dict[other_agent],
                deterministic=False
            )
            actions[other_agent] = int(other_action)
        else:
            # Random action before other model is initialised
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
        pass  # marl_env.close() called externally


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
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

# ── Shared MARL environment ────────────────────────────────────────────
marl_env = SumoMARLEnv(config)

# ── Model references — mutable list so wrappers can access updates ─────
# Each wrapper holds a reference to the OTHER agent's model
# so it can query it for joint actions during training
tl1_model_ref = [None]  # will hold tl2's model
tl2_model_ref = [None]  # will hold tl1's model

# ── Create per-agent Gymnasium wrappers ────────────────────────────────
env_tl1 = AgentEnvWrapper(marl_env, "tl1", tl1_model_ref)
env_tl2 = AgentEnvWrapper(marl_env, "tl2", tl2_model_ref)

# ── Create PPO models — identical hyperparameters to SARL PPO ──────────
model_tl1 = PPO(
    policy        = "MlpPolicy",
    env           = env_tl1,
    learning_rate = 0.0003,
    n_steps       = 40,
    batch_size    = 10,
    n_epochs      = 10,
    gamma         = 0.99,
    verbose       = 1,
    tensorboard_log= "./logs/"
)

model_tl2 = PPO(
    policy        = "MlpPolicy",
    env           = env_tl2,
    learning_rate = 0.0003,
    n_steps       = 40,
    batch_size    = 10,
    n_epochs      = 10,
    gamma         = 0.99,
    verbose       = 1,
    tensorboard_log= "./logs/"
)

# ── Wire cross-references so each wrapper can query the other model ────
tl1_model_ref[0] = model_tl2   # env_tl1 queries model_tl2 for joint action
tl2_model_ref[0] = model_tl1   # env_tl2 queries model_tl1 for joint action

print("=== MARL PPO Training — 2 Intersection Network ===")
print(f"Agents:          {config['agent_ids']}")
print(f"Obs size:        {marl_env.obs_size} per agent (padded)")
print(f"Action space:    Discrete(3) per agent")
print(f"Reward:          Global network queue penalty (shared)")
print(f"Hyperparameters: Identical to SARL PPO for fair comparison")
print(f"Total timesteps: 260,000 per agent\n")

# ── Train alternating between agents ──────────────────────────────────
# Each agent trains for n_steps then the other trains for n_steps
# This is the standard approach for independent MARL with SB3
TOTAL_TIMESTEPS = 260000
SEGMENT         = 40     # match n_steps so each learn() call = 1 rollout

print("Starting alternating training...")
steps_done = 0

while steps_done < TOTAL_TIMESTEPS:
    # Train tl1 for one segment
    model_tl1.learn(
        total_timesteps     = SEGMENT,
        reset_num_timesteps = False,
        tb_log_name         = "marl_ppo_tl1",
        progress_bar        = False
    )

    # Train tl2 for one segment
    model_tl2.learn(
        total_timesteps     = SEGMENT,
        reset_num_timesteps = False,
        tb_log_name         = "marl_ppo_tl2",
        progress_bar        = False
    )

    steps_done += SEGMENT

    # Checkpoint every 10,000 steps
    if steps_done % 10000 == 0:
        model_tl1.save(f"checkpoints/marl_ppo_tl1_{steps_done}")
        model_tl2.save(f"checkpoints/marl_ppo_tl2_{steps_done}")
        print(f"Step {steps_done}/{TOTAL_TIMESTEPS} — checkpoints saved")

# ── Save final models ──────────────────────────────────────────────────
model_tl1.save("models/marl_ppo_tl1_260k")
model_tl2.save("models/marl_ppo_tl2_260k")
print("\nTraining complete.")
print("Models saved: models/marl_ppo_tl1_260k.zip")
print("             models/marl_ppo_tl2_260k.zip")

marl_env.close()