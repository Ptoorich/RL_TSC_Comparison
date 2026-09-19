# Data Processing

## 1. Data Capture

Two independent data streams are captured during each simulated episode, both drawn from the SUMO/TraCI interface rather than post-hoc estimation:

- **Step-level traffic-state data**, read live via TraCI during the episode: per-lane queue length (`traci.lane.getLastStepHaltingNumber`) for every monitored lane, vehicles arrived at their destination (`traci.simulation.getArrivedNumber`), and vehicles forced into a teleport by SUMO's collision-avoidance heuristics (`traci.simulation.getStartingTeleportNumber`). These are read at the same decision cadence as the controller acts (every `step_length` = 90 simulated seconds), so the data reflects the traffic state the controller was actually reacting to.
- **Trip-level outcome data**, written by SUMO to a `tripinfo` XML file at the end of each episode and parsed afterward with `xml.etree.ElementTree`. Each vehicle's completed trip records a `duration` (total travel time) and `timeLoss` (delay relative to free-flow travel), which are averaged across all vehicles in the episode to give that episode's mean travel time and mean delay. The tripinfo file is deleted immediately after parsing to avoid stale data being picked up by a later run.

The reward signal used during training is itself a processed quantity, not raw data: it is computed every step from the same lane-queue readings, using a piecewise penalty (no penalty below a low-congestion threshold, a linear penalty in a mid-congestion band, and a steeper penalty beyond a high-congestion threshold), plus an additional teleport penalty. Because this reward is derived on the fly and never persisted independently, it is not treated as a data source for reporting — evaluation instead uses the step-level and trip-level data described above.

## 2. Processing Pipeline

Evaluation is separated from training and run deterministically: each trained model (or set of per-agent models, for the independent-learner configurations) is loaded and evaluated with `deterministic=True` action selection over a **fixed set of 10 seeds** (`42, 123, 256, 512, 999, 1337, 2024, 3141, 7777, 9999`). The same seed set is reused unchanged across every algorithm (SARL PPO/DQN, MARL independent, MARL parameter-sharing) and every network size (2/4/6 intersections), so that differences in the resulting metrics can be attributed to the controller rather than to the evaluation protocol itself — every configuration faces an identical set of ten traffic-demand realisations.

Within a single episode:
- The mean queue across all monitored lanes is recorded at every decision step and stored in a list; the episode's `avg_queue` is the mean of that list.
- `throughput` and `teleports` are accumulated as running sums over the episode's steps.
- `mean_travel_time` and `mean_delay` come from parsing that episode's tripinfo file once the episode terminates.
- `total_reward` is the sum of the (shared, network-wide) reward returned by the environment at each step.

Across the 10 episodes for a given configuration, each metric is reduced with NumPy to a **mean and standard deviation** (`np.mean`, `np.std`), printed as `mean ± std`. Critically, the raw per-seed values — not just the aggregate — are serialised to a JSON file per configuration (e.g. `results/marl_ppo_ps_6int_results.json`), alongside the seed list itself. This preserves the full record needed to recompute other statistics, build plots, or run significance tests later without re-running any simulation.

## 3. Validation and Verification

A few checks are built into the current pipeline, and a few gaps are worth flagging honestly for the report rather than glossed over:

- **Built-in defenses.** TraCI calls inside the environment's observation and reward functions are wrapped in `try/except`, falling back to a zero observation or zero reward on failure. This protects a run from crashing on a transient TraCI hiccup, but it also means a genuine fault could silently produce zeroed-out data rather than an error. As currently written, these fallbacks are not counted or logged, so there is no direct evidence in the results that they never fired during a reported run. Adding a simple fallback counter (and reporting "0 of N steps used the fallback path" per episode) would close this gap and is a cheap addition if the report needs to state data integrity affirmatively rather than by assumption.
- **Per-episode visibility.** Each episode's metrics are printed as they complete, rather than only at the end of all 10. This lets an obviously anomalous episode (e.g. a seed that produces a gridlock) be spotted and investigated before it is folded into an aggregate.
- **Cross-metric consistency.** Because the reward is itself a function of the same lane queues used to compute `avg_queue`, the two series should track each other within an episode (higher `avg_queue` correlating with lower — more negative — reward). Checking this correlation is a lightweight sanity check that the reward the controller was optimised against matches the congestion the evaluation is reporting, and is worth including explicitly as a validation step in the write-up rather than left implicit.
- **Held constant across comparisons.** The lane/junction maps, reward weights, and evaluation seeds are defined once per network size and reused verbatim by every algorithm's training and evaluation script. This is a deliberate control: it rules out configuration drift as an explanation for any performance difference observed between SARL, MARL-independent, and MARL-parameter-sharing.

## 4. Uncertainty Analysis

Reliability of the reported numbers rests on the spread across the 10 evaluation seeds, which is a modest sample for statistical purposes. Two sources contribute to that spread, and they are worth separating in the discussion:

1. **Traffic-demand variation.** Each seed produces a different realisation of vehicle departures/routes within SUMO, so a controller may perform differently purely because of which demand pattern it was tested against — this is genuine environmental variability the controller has to be robust to.
2. **Residual simulator stochasticity.** Even with `deterministic=True` action selection removing the policy's own action-sampling randomness, SUMO's internal handling of conflicts (e.g. teleports) is not fully eliminated by fixing the seed at the Python level, since some of SUMO's own randomness is seeded separately via the `--seed` command-line argument.

The current scripts report standard deviation but do not go further into inferential statistics. Given `n = 10`, a natural addition for the report — without needing any new simulation runs, since the raw per-seed values are already saved — would be the standard error of the mean (`std / √10`) and an approximate 95% confidence interval, to make explicit how much the reported means could plausibly shift with a different sample of seeds. This is worth adding before treating small differences between algorithms as meaningful, since with only 10 seeds a modest std can produce a wide interval.

## 5. Communication of Data and Methods

Results currently reach the reader through two channels: a human-readable console summary (per-episode lines followed by an aggregate `mean ± std` block for each metric), and a machine-readable JSON artifact per algorithm/network-size combination that retains the full raw record. For the report itself, the recommended next step is a small aggregation pass over these JSON files to produce a single comparison table (or grouped bar chart with error bars) spanning all algorithms and network sizes side by side — this hasn't been built yet in the current scripts, but since every run already writes its raw seed-level data to JSON, assembling it is a matter of reading those files rather than re-running any simulation.