import numpy as np
from stable_baselines3 import DQN
from sumo_env import SumoTSCEnv
import os

# ── Screening evaluation — 3 seeds, fast ──────────────────────────────
SEEDS = [42, 123, 256]

def screen_model(model_path, w_throughput, label):
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
        "w_throughput" : w_throughput,
        "sim_steps"    : 3600,
        "step_length"  : 90,
        "use_gui"      : False,
    }

    model = DQN.load(model_path)
    env   = SumoTSCEnv(config)

    queues      = []
    throughputs = []
    travel_times= []
    delays      = []

    for seed in SEEDS:
        obs, _        = env.reset(seed=seed)
        ep_queue      = []
        ep_throughput = 0
        terminated    = False

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, _, _ = env.step(action)
            ep_throughput += env.last_arrived
            ep_queue.append(
                sum(env._get_queue_lengths()) / env.num_links
            )

        queues.append(np.mean(ep_queue))
        throughputs.append(ep_throughput)
        travel_times.append(env.mean_travel_time)
        delays.append(env.mean_delay)

    env.close()

    return {
        "label"       : label,
        "queue"       : np.mean(queues),
        "throughput"  : np.mean(throughputs),
        "travel_time" : np.mean(travel_times),
        "delay"       : np.mean(delays),
    }

# ── Run screening on all saved models ─────────────────────────────────
os.makedirs("screening", exist_ok=True)

candidates = [
    ("sarl_dqn_2int_260k",         0.0,  "DQN v1 (queue only)"),
    ("sarl_dqn_v2_2int_260k",      2.0,  "DQN v2 (w_tp=2.0)"),
    ("screening/screen_dqn_wtp5",  5.0,  "Screen (w_tp=5.0)"),
    ("screening/screen_dqn_wtp10", 10.0, "Screen (w_tp=10.0)"),
]

print("=" * 75)
print(f"{'Config':<28} {'Queue':>8} {'Throughput':>12} "
      f"{'Travel Time':>13} {'Delay':>8}")
print(f"{'':28} {'(<5 ✓)':>8} {'(higher ✓)':>12} "
      f"{'(lower ✓)':>13} {'(lower ✓)':>8}")
print("-" * 75)

results = []
for path, w_tp, label in candidates:
    if not os.path.exists(path + ".zip"):
        print(f"{label:<28} — model not found, skipping")
        continue
    r = screen_model(path, w_tp, label)
    results.append(r)
    queue_flag = "✓" if r["queue"] < 5.0 else "✗"
    print(f"{label:<28} {r['queue']:>6.2f}{queue_flag:>2} "
          f"{r['throughput']:>12.1f} "
          f"{r['travel_time']:>12.1f}s "
          f"{r['delay']:>7.1f}s")

print("=" * 75)
print("\nBest throughput:", max(results, key=lambda x: x["throughput"])["label"])
print("Best travel time:", min(
    [r for r in results if r["travel_time"] > 0],
    key=lambda x: x["travel_time"]
)["label"])
print("Best queue:", min(results, key=lambda x: x["queue"])["label"])