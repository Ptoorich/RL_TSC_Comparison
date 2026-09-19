import numpy as np
from stable_baselines3 import DQN  # or PPO
from sumo_env_marl_ps import SumoMARLEnvPS

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
        "tl1": ["7521010#0_0", "622835369_0", "622835369_1", "622835369_3"],
        "tl2": ["596638461#0_0", "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3"],
        "tl3": ["380081149#2_0", "42790161#2_0", "42790161#2_1", "42790161#2_2"],
        "tl4": ["42790161#0_0", "42790161#0_1", "42790161#0_2", "7521010#1_0"],
        "tl5": ["-621271001_0", "669676574#2_0", "1190301745#0_0", "-669676574#9_0",
                "622835372_0", "622835372_1", "622835372_2"],
        "tl6": ["-621271003_0", "669676574#1_0", "623540942#0_0", "623540942#0_1",
                "623540942#0_2", "-669676574#2_0", "621271001_0"],
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
    "use_gui"      : True,   # <- the only functional change needed
}

model = DQN.load("models/marl_dqn_ps_6int_260k")
env   = SumoMARLEnvPS(config)

SEED = 42  # pick whichever seed you want to watch
obs_dict   = env.reset(seed=SEED)
terminated = False

while not terminated:
    actions = {
        agent: int(model.predict(obs_dict[agent], deterministic=True)[0])
        for agent in config["agent_ids"]
    }
    print(actions, "-> splits:", env.phase_splits)  # watch the splits collapse live in the console

    obs_dict, rewards, terminateds = env.step(actions)
    terminated = terminateds[config["agent_ids"][0]]

env.close()