import numpy as np
from stable_baselines3 import DQN
from sumo_env_6int import SumoTSCEnv6Int
import json
import os

# Run this script from agents/ — matches train_sarl_dqn_6int.py's config.
config = {
    "sumocfg"  : "6int_files/6int.sumocfg",
    "tl_ids"   : [
        "30406195",
        "30406197",
        "54994558",
        "768241766",
        "cluster_30406198_7161921427_7161921428",
        "cluster_54994556_7161921429"
    ],
    "link_ids" : [
        # 30406195
        "7521010#0_0", "622835369_0", "622835369_1", "622835369_3",
        # 30406197
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        # 54994558
        "380081149#2_0",
        "42790161#2_0", "42790161#2_1", "42790161#2_2",
        # 768241766
        "42790161#0_0", "42790161#0_1", "42790161#0_2", "7521010#1_0",
        # cluster_30406198_7161921427_7161921428
        "-621271001_0", "669676574#2_0", "1190301745#0_0",
        "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2",
        # cluster_54994556_7161921429 (6-phase junction)
        "-621271003_0", "669676574#1_0",
        "623540942#0_0", "623540942#0_1", "623540942#0_2",
        "-669676574#2_0", "621271001_0"
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
    "w_throughput" : 0.0,   # queue penalty only — same as PPO 6int
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

# ── Load trained model ──────────────────────────────────────────────────
# NOTE: train_sarl_dqn_6int.py saves to "models/sarl_dqn_6int_260k.zip".
# Update this path if the actual run landed somewhere else (worth
# double-checking, same as with the PPO 6int model earlier).
MODEL_PATH = "models/sarl_dqn_6int_260k"

model = DQN.load(MODEL_PATH)
env   = SumoTSCEnv6Int(config)

os.makedirs("results", exist_ok=True)

SEEDS = [42, 123, 256, 512, 999, 1337, 2024, 3141, 7777, 9999]

results = {
    "seeds"            : SEEDS,
    "total_reward"     : [],
    "avg_queue"        : [],
    "throughput"       : [],
    "teleports"        : [],
    "mean_travel_time" : [],
    "mean_delay"       : [],
}

print("=== Evaluating SARL DQN — 6 Intersection Network ===")
print(f"Model: {MODEL_PATH}.zip\n")

for ep, seed in enumerate(SEEDS):
    obs, _        = env.reset(seed=seed)
    ep_reward     = 0.0
    ep_queues     = []
    ep_throughput = 0
    ep_teleports  = 0
    terminated    = False

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        ep_reward     += reward
        ep_throughput += env.last_arrived
        ep_teleports  += env.last_teleports

        # Mean queue across all monitored lanes
        try:
            import traci
            all_q = [traci.lane.getLastStepHaltingNumber(l)
                      for l in config["link_ids"]]
            if all_q:
                ep_queues.append(np.mean(all_q))
        except Exception:
            pass

    results["total_reward"].append(ep_reward)
    results["avg_queue"].append(
        float(np.mean(ep_queues)) if ep_queues else 0.0
    )
    results["throughput"].append(ep_throughput)
    results["teleports"].append(ep_teleports)
    results["mean_travel_time"].append(env.mean_travel_time)
    results["mean_delay"].append(env.mean_delay)

    print(f"Episode {ep+1:2d} (seed={seed:4d}): "
          f"reward={ep_reward:8.1f}  "
          f"queue={results['avg_queue'][-1]:.2f}  "
          f"throughput={ep_throughput:3d}  "
          f"travel_time={env.mean_travel_time:.1f}s  "
          f"delay={env.mean_delay:.1f}s  "
          f"teleports={ep_teleports}")

env.close()

print("\n=== Summary across 10 episodes ===")
print(f"Mean reward:       {np.mean(results['total_reward']):.1f} "
      f"± {np.std(results['total_reward']):.1f}")
print(f"Mean avg queue:    {np.mean(results['avg_queue']):.2f} "
      f"± {np.std(results['avg_queue']):.2f} vehicles")
print(f"Mean throughput:   {np.mean(results['throughput']):.1f} "
      f"± {np.std(results['throughput']):.1f} vehicles")
print(f"Mean travel time:  {np.mean(results['mean_travel_time']):.1f} "
      f"± {np.std(results['mean_travel_time']):.1f} seconds")
print(f"Mean delay:        {np.mean(results['mean_delay']):.1f} "
      f"± {np.std(results['mean_delay']):.1f} seconds")
print(f"Mean teleports:    {np.mean(results['teleports']):.1f} "
      f"± {np.std(results['teleports']):.1f}")

with open("results/sarl_dqn_6int_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/sarl_dqn_6int_results.json")