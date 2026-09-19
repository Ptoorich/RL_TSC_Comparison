import numpy as np
from stable_baselines3 import DQN  # or DQN
from sumo_env import SumoTSCEnv

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
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : True,  # <- watch it live
}

model = DQN.load("models/sarl_dqn_2int_260k")   # swap for PPO.load("sarl_ppo_2int") for the other run
env   = SumoTSCEnv(config)

SEED = 42
obs, _     = env.reset(seed=SEED)
terminated = False

while not terminated:
    action, _ = model.predict(obs, deterministic=True)
    tl_id, delta = env._decode_action(action)
    print(f"action={int(action)} -> tl={tl_id}, delta={delta:+d}  |  splits: {env.phase_splits}")

    obs, reward, terminated, truncated, info = env.step(action)

env.close()