import numpy as np
from stable_baselines3 import DQN  # or PPO
from sumo_env_marl_ps import SumoMARLEnvPS

config = {
    "sumocfg"  : "2int_files/2int.sumocfg",
    "agent_ids": ["tl1", "tl2"],
    "tl_map"   : {
        "tl1": "30406197",
        "tl2": "cluster_30406198_7161921427_7161921428",
    },
    "lane_map" : {
        "tl1": ["596638461#0_0", "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3"],
        "tl2": ["-621271001_0", "669676574#0_0", "1190301745#0_0", "-669676574#8_0",
                "622835372_0", "622835372_1", "622835372_2"],
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

model = DQN.load("models/marl_dqn_ps_2int_260k")
env   = SumoMARLEnvPS(config)

SEED = 42  # pick whichever seed you want to watch
obs_dict   = env.reset(seed=SEED)
terminated = False

while not terminated:
    actions = {
        agent: int(model.predict(obs_dict[agent], deterministic=True)[0])
        for agent in config["agent_ids"]
    }
    print(actions, "-> splits:", env.phase_splits)  # watch the splits move live in the console

    obs_dict, rewards, terminateds = env.step(actions)
    terminated = terminateds[config["agent_ids"][0]]

env.close()