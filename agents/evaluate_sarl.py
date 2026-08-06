import numpy as np
import traci
from stable_baselines3 import PPO
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
    "sim_steps"    : 3600,
    "step_length"  : 90,
    "use_gui"      : False,
}

# ── Load trained model ─────────────────────────────────────────────────
model = PPO.load("sarl_ppo_2int_260k")
env   = SumoTSCEnv(config)

# ── Create results folder if it doesn't exist ──────────────────────────
os.makedirs("results", exist_ok=True)

# ── Run 10 evaluation episodes ─────────────────────────────────────────
NUM_EPISODES = 10
results = {
    "avg_queue"    : [],
    "total_reward" : [],
    "throughput"   : [],
    "teleports"    : [],
}

print("=== Evaluating SARL Agent - 2 Intersection Network ===\n")

for ep in range(NUM_EPISODES):
    obs, _        = env.reset()
    ep_reward     = 0
    ep_queues     = []
    ep_teleports  = 0
    ep_throughput = 0        # accumulates vehicles completing trips
    terminated    = False

    while not terminated:
        # Greedy action — no exploration
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)

        # Accumulate reward
        ep_reward += reward

        # Accumulate vehicles that completed their trip this step
        ep_throughput += traci.simulation.getArrivedNumber()

        # Record mean queue length across all monitored lanes
        queues = [traci.lane.getLastStepHaltingNumber(l)
                  for l in config["link_ids"]]
        ep_queues.append(np.mean(queues))

        # Accumulate teleports this step
        ep_teleports += traci.simulation.getStartingTeleportNumber()

    # Store episode results
    results["total_reward"].append(ep_reward)
    results["avg_queue"].append(np.mean(ep_queues))
    results["throughput"].append(ep_throughput)
    results["teleports"].append(ep_teleports)

    print(f"Episode {ep+1:2d}: "
          f"reward={ep_reward:8.1f}  "
          f"avg_queue={np.mean(ep_queues):.2f}  "
          f"throughput={ep_throughput}  "
          f"teleports={ep_teleports}")

env.close()

# ── Summary statistics ─────────────────────────────────────────────────
print("\n=== Summary across 10 episodes ===")
print(f"Mean reward:     {np.mean(results['total_reward']):.1f} "
      f"± {np.std(results['total_reward']):.1f}")
print(f"Mean avg queue:  {np.mean(results['avg_queue']):.2f} "
      f"± {np.std(results['avg_queue']):.2f} vehicles")
print(f"Mean throughput: {np.mean(results['throughput']):.1f} "
      f"± {np.std(results['throughput']):.1f} vehicles")
print(f"Mean teleports:  {np.mean(results['teleports']):.1f} "
      f"± {np.std(results['teleports']):.1f}")

# ── Save results to file ───────────────────────────────────────────────
with open("results/sarl_2int_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/sarl_2int_results.json")