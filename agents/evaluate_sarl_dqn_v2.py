import numpy as np
from stable_baselines3 import DQN
from sumo_env import SumoTSCEnv
import json
import os

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
    "w_throughput" : 2.0,
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

# ── Load trained model ─────────────────────────────────────────────────
model = DQN.load("sarl_dqn_v2_2int_260k")
env   = SumoTSCEnv(config)

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

print("=== Evaluating SARL DQN v2 — 2 Intersection Network ===")
print("Reward: queue penalty + throughput incentive (w=2.0)\n")

for ep, seed in enumerate(SEEDS):
    obs, _        = env.reset(seed=seed)
    ep_reward     = 0
    ep_queues     = []
    ep_teleports  = 0
    ep_throughput = 0
    terminated    = False

    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)

        ep_reward     += reward

        # Read metrics from env instance variables
        # — safe even after TraCI connection closes
        ep_throughput += env.last_arrived
        ep_teleports  += env.last_teleports

        ep_queues.append(
            sum(env._get_queue_lengths()) / max(env.num_links, 1)
        )

    # Travel time and delay set by env.step when terminated
    ep_travel_time = env.mean_travel_time
    ep_delay       = env.mean_delay

    results["total_reward"].append(ep_reward)
    results["avg_queue"].append(float(np.mean(ep_queues)))
    results["throughput"].append(ep_throughput)
    results["teleports"].append(ep_teleports)
    results["mean_travel_time"].append(ep_travel_time)
    results["mean_delay"].append(ep_delay)

    print(f"Episode {ep+1:2d} (seed={seed:4d}): "
          f"reward={ep_reward:8.1f}  "
          f"queue={np.mean(ep_queues):.2f}  "
          f"throughput={ep_throughput:3d}  "
          f"travel_time={ep_travel_time:.1f}s  "
          f"delay={ep_delay:.1f}s  "
          f"teleports={ep_teleports}")

env.close()

# ── Summary ────────────────────────────────────────────────────────────
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

# ── Save ───────────────────────────────────────────────────────────────
with open("results/sarl_dqn_v2_2int_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/sarl_dqn_v2_2int_results.json")