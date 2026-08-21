import numpy as np
import json
import matplotlib.pyplot as plt
import os

# ── Load your existing DQN v2 episode-level results ───────────────────
# We need per-step data so we re-run evaluation in logging mode
# For now we use the summary stats to estimate
# (see Step 2 for precise per-step analysis)

# ── Weight combinations to test ────────────────────────────────────────
# Format: (w_throughput, label)
weight_configs = [
    (0.0,  "Queue only (DQN v1)"),
    (0.5,  "w_tp=0.5  light nudge"),
    (2.0,  "w_tp=2.0  current v2"),
    (5.0,  "w_tp=5.0  moderate"),
    (10.0, "w_tp=10.0 strong"),
    (20.0, "w_tp=20.0 aggressive"),
]

# ── Known per-episode metrics from your evaluations ───────────────────
# From DQN v2 evaluation — approximate per-step values
# queue_sum  = mean_avg_queue × num_links × num_steps
# arrived    = throughput / num_episodes
# We reconstruct reward per episode under each weight

# DQN v2 evaluation constants
num_links  = 12
num_steps  = 40      # steps per episode (3600s / 90s)
q_lc       = 5.0
q_hc       = 15.0
w_l        = 1.0
w_cp       = 3.0

# From your DQN v2 evaluation results
mean_queue_per_lane = 1.67   # vehicles — average per lane per step
mean_throughput_ep  = 23.1   # vehicles per episode
mean_teleports_ep   = 0.0

print("=" * 65)
print("Retroactive Reward Analysis — DQN v2 Policy")
print("=" * 65)
print(f"{'Config':<30} {'Est. Reward':>12} {'Queue':>8} {'Throughput':>12}")
print("-" * 65)

rewards = []
labels  = []

for w_tp, label in weight_configs:
    # Estimate queue penalty per episode
    # queue < q_lc = free flow → no penalty
    # queue 1.67 < 5 → most steps are free flow for DQN v2
    # But some lanes do queue — use mean to estimate
    q = mean_queue_per_lane

    if q <= q_lc:
        queue_penalty_per_lane_step = 0.0
    elif q <= q_hc:
        queue_penalty_per_lane_step = -(w_l * q)
    else:
        queue_penalty_per_lane_step = -(w_cp * w_l * q)

    total_queue_penalty = queue_penalty_per_lane_step * num_links * num_steps
    total_tp_reward     = w_tp * mean_throughput_ep
    total_teleport_pen  = -(5.0 * mean_teleports_ep)

    est_reward = total_queue_penalty + total_tp_reward + total_teleport_pen

    rewards.append(est_reward)
    labels.append(label)

    print(f"{label:<30} {est_reward:>12.1f} {mean_queue_per_lane:>8.2f} "
          f"{mean_throughput_ep:>12.1f}")

print("=" * 65)
print("\nNote: This estimates reward for the SAME trained policy")
print("under different weight configurations.")
print("It shows which weights would have best rewarded this policy,")
print("NOT how a newly trained agent with those weights would behave.")

# ── Plot ───────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 5))
colors = ["steelblue" if r == max(rewards) else "lightsteelblue"
          for r in rewards]
bars = plt.bar(range(len(labels)), rewards, color=colors, edgecolor="navy")
plt.xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=9)
plt.ylabel("Estimated Episode Reward")
plt.title("Retroactive Reward Estimate — DQN v2 Policy Under Different Weights")
plt.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/retroactive_reward_analysis.png", dpi=150)
plt.show()
print("\nSaved to results/retroactive_reward_analysis.png")