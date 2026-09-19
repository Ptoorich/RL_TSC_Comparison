import numpy as np
from stable_baselines3 import PPO
from sumo_env_marl_ps import SumoMARLEnvPS
import json
import os

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

# ── Load shared model ──────────────────────────────────────────────────
shared_model = PPO.load("models/marl_ppo_ps_2int_260k")
env          = SumoMARLEnvPS(config)

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

print("=== Evaluating MARL PPO with Parameter Sharing — 2 Intersection Network ===")
print(f"Shared model obs size: {env.obs_size} (local obs + normalised ID)")
print(f"Normalised IDs: TL1={env.agent_index['tl1']:.4f}  "
      f"TL2={env.agent_index['tl2']:.4f}\n")

for ep, seed in enumerate(SEEDS):
    obs_dict      = env.reset(seed=seed)
    ep_reward     = 0
    ep_queues     = []
    ep_throughput = 0
    ep_teleports  = 0
    terminated    = False

    while not terminated:
        # Both agents use the SAME shared model — distinguished by norm ID
        actions = {
            agent: int(shared_model.predict(
                obs_dict[agent], deterministic=True
            )[0])
            for agent in config["agent_ids"]
        }

        obs_dict, rewards, terminateds = env.step(actions)

        terminated     = terminateds[config["agent_ids"][0]]
        ep_reward     += rewards[config["agent_ids"][0]]
        ep_throughput += env.last_arrived
        ep_teleports  += env.last_teleports

        # Mean queue across all monitored lanes
        try:
            import traci
            all_q = []
            for agent in config["agent_ids"]:
                all_q.extend([
                    traci.lane.getLastStepHaltingNumber(l)
                    for l in config["lane_map"][agent]
                ])
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
    print(env.phase_splits)

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

with open("results/marl_ppo_ps_2int_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/marl_ppo_ps_2int_results.json")