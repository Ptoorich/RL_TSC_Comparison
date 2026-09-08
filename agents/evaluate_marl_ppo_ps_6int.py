import numpy as np
from stable_baselines3 import PPO
from sumo_env_marl_ps import SumoMARLEnvPS
import json
import os

# Run this script from agents/ — matches train_marl_ppo_ps_6int.py's config.
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

# ── Load the single shared model — same model drives every agent ────────
shared_model = PPO.load("models/marl_ppo_ps_6int_260k")
env = SumoMARLEnvPS(config)

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

print("=== Evaluating MARL PPO (parameter sharing) — 6 Intersection Network ===")
print(f"Agents: {config['agent_ids']}\n")

for ep, seed in enumerate(SEEDS):
    obs_dict      = env.reset(seed=seed)
    ep_reward     = 0.0
    ep_queues     = []
    ep_throughput = 0
    ep_teleports  = 0
    terminated    = False

    while not terminated:
        actions = {
            agent: int(shared_model.predict(
                obs_dict[agent], deterministic=True
            )[0])
            for agent in config["agent_ids"]
        }

        obs_dict, rewards, terminateds = env.step(actions)

        terminated     = terminateds[config["agent_ids"][0]]
        ep_reward     += rewards[config["agent_ids"][0]]  # same global reward for all
        ep_throughput += env.last_arrived
        ep_teleports  += env.last_teleports

        # Mean queue across all monitored lanes (all agents combined)
        try:
            import traci
            all_q = [traci.lane.getLastStepHaltingNumber(l)
                      for l in env.all_lanes]
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

with open("results/marl_ppo_ps_6int_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/marl_ppo_ps_6int_results.json")