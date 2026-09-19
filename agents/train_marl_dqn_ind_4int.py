import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback
from sumo_env_marl_ind import SumoMARLEnvInd
import os

# ── Single-agent Gymnasium wrapper around the shared MARL environment ──
class AgentEnvWrapper(gym.Env):
    """
    Wraps one agent's view of the shared MARL environment into a
    standard Gymnasium interface for SB3 compatibility. Generalized
    to N agents: queries every OTHER agent's current model (not just
    one) to build the joint action each step.
    """

    def __init__(self, marl_env, agent_id, other_model_refs):
        super().__init__()
        self.marl_env         = marl_env
        self.agent_id         = agent_id
        self.other_model_refs = other_model_refs  # {other_agent_id: [model_or_None]}

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
        pass  # marl_env.close() called externally


# ── Fix: exploration schedule tied to true cumulative progress ─────────
class TrueExplorationScheduleCallback(BaseCallback):
    """
    SB3 recomputes exploration progress against the total_timesteps
    argument of the CURRENT learn() call, not a grand total spanning
    many calls. Since round-robin training here calls learn() in
    SEGMENT=40-step chunks, the built-in schedule would collapse to
    exploration_final_eps almost immediately after the first segment,
    starving later segments of exploration.

    This callback overrides model.exploration_rate at every step,
    computed from the agent's true cumulative num_timesteps against
    TOTAL_TIMESTEPS, so epsilon decays smoothly across the full 260k
    run — matching the schedule used by the single-call SARL DQN and
    MARL-PS DQN scripts. Unlike overriding model.exploration_schedule
    directly, a callback is never stored on the model's own __dict__,
    so it has no effect on model.save() (which pickles the model's
    saved attributes, not the transient callback passed to learn()).
    """
    def __init__(self, total_timesteps, initial_eps, final_eps, fraction, verbose=0):
        super().__init__(verbose)
        self._schedule       = get_linear_fn(initial_eps, final_eps, fraction)
        self.total_timesteps = total_timesteps

    def _on_step(self):
        true_progress_remaining = 1.0 - (self.model.num_timesteps / self.total_timesteps)
        self.model.exploration_rate = self._schedule(true_progress_remaining)
        return True


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
    "use_gui"      : False,  # 
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs",        exist_ok=True)
os.makedirs("results",     exist_ok=True)
os.makedirs("models",      exist_ok=True)

TOTAL_TIMESTEPS = 260000
SEGMENT         = 40     # matches n_steps used by the PPO-independent scripts

# ── Shared MARL environment ────────────────────────────────────────────
marl_env = SumoMARLEnvInd(config)

# ── Model reference dict — each agent holds a mutable ref slot so
#    other agents' wrappers can look up its current model ─────────────
model_refs = {agent: [None] for agent in config["agent_ids"]}

# ── Create per-agent Gymnasium wrappers ────────────────────────────────
envs = {}
for agent in config["agent_ids"]:
    other_refs = {a: model_refs[a] for a in config["agent_ids"] if a != agent}
    envs[agent] = AgentEnvWrapper(marl_env, agent, other_refs)

# ── Create DQN models — same hyperparameters as SARL DQN / MARL-PS DQN ─
models           = {}
explore_callbacks = {}
for agent in config["agent_ids"]:
    models[agent] = DQN(
        policy                  = "MlpPolicy",
        env                     = envs[agent],
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
    model_refs[agent][0] = models[agent]

    # Corrected exploration schedule, applied via callback (see note above)
    explore_callbacks[agent] = TrueExplorationScheduleCallback(
        total_timesteps = TOTAL_TIMESTEPS,
        initial_eps     = 1.0,
        final_eps       = 0.05,
        fraction        = 0.2,
    )

print("=== MARL DQN (independent) Training — 4 Intersection Network ===")
print(f"Agents:          {config['agent_ids']}")
print(f"Obs size:        {marl_env.obs_size} per agent (padded)")
print(f"Action space:    Discrete(3) per agent")
print(f"Reward:          Global network queue penalty (shared)")
print(f"Hyperparameters: Identical to SARL DQN for fair comparison")
print(f"Total timesteps: {TOTAL_TIMESTEPS} per agent\n")

# ── Train — round-robin one segment per agent per round ────────────────
print("Starting alternating training...")
steps_done = 0

while steps_done < TOTAL_TIMESTEPS:
    for agent in config["agent_ids"]:
        models[agent].learn(
            total_timesteps     = SEGMENT,
            reset_num_timesteps = False,
            tb_log_name         = f"marl_dqn_ind_4int_{agent}",
            progress_bar        = False,
            callback            = explore_callbacks[agent]
        )

    steps_done += SEGMENT

    if steps_done % 10000 == 0:
        for agent in config["agent_ids"]:
            models[agent].save(f"checkpoints/marl_dqn_ind_4int_{agent}_{steps_done}")
        print(f"Step {steps_done}/{TOTAL_TIMESTEPS} — checkpoints saved")

# ── Save final models ──────────────────────────────────────────────────
for agent in config["agent_ids"]:
    models[agent].save(f"models/marl_dqn_ind_4int_{agent}_260k")
    print(f"Model saved: models/marl_dqn_ind_4int_{agent}_260k.zip")

print("\nTraining complete.")
marl_env.close()