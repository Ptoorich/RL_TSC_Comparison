from sumo_env import SumoTSCEnv

config = {
    "sumocfg"  : "2int_files/2int.sumocfg",

    # Exact IDs from your network
    "tl_ids"   : [
        "30406197",
        "cluster_30406198_7161921427_7161921428"
    ],

    # All approach lanes from both intersections
    "link_ids" : [
        # TL1 - 30406197
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        # TL2 - cluster
        "-621271001_0", "669676574#0_0", "1190301745#0_0",
        "-669676574#8_0", "622835372_0", "622835372_1", "622835372_2"
    ],

    # Calibrated to your real Johannesburg network
    "cycle_length" : 90,
    "yellow_time"  : 6,
    "allred_time"  : 0,
    "init_split"   : 39,
    "s_lb"         : 15,
    "s_ub"         : 63,
    "delta_s"      : 5,

    # Reward thresholds
    "q_lc"  : 5,
    "q_hc"  : 15,
    "w_l"   : 1.0,
    "w_cp"  : 3.0,

    # Episode settings
    "sim_steps"   : 900,   # 10 cycles for a quick test
    "step_length" : 90,    # one decision per cycle
    "use_gui"     : True,
}

env = SumoTSCEnv(config)

print("=== Environment Test ===")
print(f"Observation space: {env.observation_space}")
print(f"Action space:      {env.action_space}")
print(f"Obs size:          {env.num_links} lanes + {env.num_tls} phase splits = {env.num_links + env.num_tls} elements")

obs, _ = env.reset()
print(f"\nInitial observation:\n  {obs}")

for step in range(5):
    action = env.action_space.sample()
    tl, delta = env._decode_action(action)
    obs, reward, terminated, _, _ = env.step(action)
    print(f"\nStep {step+1}:")
    print(f"  Action {action} → TL={tl}, delta={delta:+}s")
    print(f"  Phase splits: {env.phase_splits}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Obs:    {obs}")
    if terminated:
        break

env.close()
print("\nTest complete.")