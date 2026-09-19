# Chapter 3 — Methodology

## 3.1 Overview of the Experimental Approach

This investigation compares Single-Agent Reinforcement Learning (SARL) and Multi-Agent Reinforcement Learning (MARL) for traffic signal control (TSC) on real-world urban road networks. Because the research questions concern control-policy behaviour rather than physical phenomena, the "test equipment" is a co-simulation test bench: the Eclipse SUMO traffic microsimulator acts as the plant under test, a custom Gymnasium environment acts as the measurement and actuation interface, and deep reinforcement learning agents (implemented in Stable-Baselines3) act as the adaptive controllers under investigation.

The overall experimental strategy is:

1. Construct nested, real-world signalised networks (2, 4 and 6 intersections) imported from OpenStreetMap.
2. Generate reproducible peak-hour traffic demand stochastically with fixed random seeds.
3. Wrap SUMO in a custom Gymnasium environment implementing the MDP formulation of Li and Zhuang so that the state, action and reward definitions are held constant across all experiments.
4. Train and evaluate controllers spanning the full factorial of two learning algorithms (PPO, DQN), two controller architectures (SARL, MARL) and three network scales (2, 4, 6 intersections) — a total of **12 simulations** (Section 3.6).
5. Evaluate every trained controller identically over ten fixed random seeds and compare network-level performance metrics.

All code, configurations, trained models and logs are version-controlled in the GitHub repository `RL_TSC_Comparison`, so that an independent person can clone the repository, install the pinned dependencies and reproduce every result.

**Figure 3.1 (insert): Overall system architecture** — a block diagram showing: SUMO microsimulator ↔ TraCI interface ↔ custom Gymnasium environment (`sumo_env.py` / `sumo_env_marl_ps.py`) ↔ Stable-Baselines3 agent(s) (PPO/DQN), with TensorBoard logging, checkpoint storage and Git/GitHub on the side.

---

## 3.2 Test Equipment: The Co-Simulation Test Bench

### 3.2.1 Traffic microsimulator

SUMO (Simulation of Urban MObility) v1.27.1 was used as the traffic microsimulator for every run, installed identically on the university workstations used. SUMO is a space-continuous, time-discrete car-following simulator; within it each vehicle is tracked individually, which allows queue lengths, throughputs and travel times to be measured directly rather than estimated. Key simulation parameters:

| Parameter | Value |
|---|---|
| Simulation step length | 1 s (SUMO default) |
| Episode duration | 3600 s simulated time |
| Teleporting (gridlock escape) | Enabled (default), counted as a reported metric |
| Right-hand traffic rule | As imported from OSM |

### 3.2.2 Measurement and actuation interface

The interface between the learning agents and the simulator is the TraCI (Traffic Control Interface) Python API, over a TCP client–server connection: each client process launches `sumo` as a subprocess and exchanges one message per simulation step. All measurements are taken through TraCI calls:

| Quantity | TraCI call | Notes |
|---|---|---|
| Queue length per lane | `traci.lane.getLastStepHaltingNumber()` | Vehicles halted (speed < 0.1 m/s) in the last step |
| Throughput | `traci.simulation.getArrivedNumber()`, accumulated every step | Accumulation inside the step loop is required; querying after episode end returns only the final step's arrivals (this defect was detected and corrected during commissioning) |
| Teleportations | `traci.simulation.getStartingTeleportNumber()` | Gridlock indicator |
| Travel time / delay | Parsed from SUMO `tripinfo` XML output | Mean per completed vehicle |
| Signal state | `traci.trafficlight.setPhaseDuration()` / phase logic | Actuation of phase splits |

### 3.2.3 Custom Gymnasium environment

A custom environment class, `SumoTSCEnv`, was developed to expose SUMO as a standard Gymnasium (`gym.Env`) so that off-the-shelf Stable-Baselines3 algorithms could be used without modification. Its responsibilities are:

- launching and closing the SUMO subprocess per episode;
- assembling the normalised observation vector (Section 3.5);
- decoding the flat integer action into (intersection, adjustment) pairs;
- computing the tiered reward (Section 3.5);
- enforcing the decision interval and episode horizon;
- passing an evaluation seed to SUMO via the `--seed` command-line option at `reset()`.

The environment was validated with `stable_baselines3.common.env_checker.check_env`, followed by a scripted smoke test in which a uniformly random agent acted in the environment; the test confirmed correct observation dimensionality, correct action decoding (phase splits observed changing by ±Δs), and plausible negative rewards. A second MARL variant, `SumoEnvMARL`, wraps the same simulation for N learners under a parameter-sharing regime: each per-agent observation (its own approach-lane queues and phase split, zero-padded to a common length, plus a one-hot intersection identifier) is fed through one shared policy network, exposed behind a thin adapter so that SB3's algorithms can consume it unchanged.

**Figure 3.2 (insert): Functional block diagram of `SumoTSCEnv`** — reset(): launch SUMO with optional `--seed`; step(action): decode action → apply phase-split change → advance SUMO by decision interval → read queues → compute reward → return (obs, reward, terminated); close(): terminate TraCI connection.

---

## 3.3 Construction of the Test Networks

### 3.3.1 Development sequence

Network construction proceeded in two phases:

1. **Commissioning corridor (synthetic).** A straight three-intersection arterial with two lanes per direction and traffic signals at each junction was built manually in NETEDIT (bundled with SUMO). This network, with demand generated by `randomTrips.py`, served purely as a pipeline sanity check: vehicles were visually confirmed in `sumo-gui` to stop at red and depart at green, and a minimal TraCI health-check script was run to confirm that live queue data could be read into Python.
2. **Experimental networks (real-world).** The synthetic corridor was discarded for experimentation. Three nested networks were instead extracted from OpenStreetMap for a real grid of signalised intersections in Johannesburg, using SUMO's bundled `osmWebWizard.py` tool. Nesting guarantees that the 2-intersection network is a strict subset of the 4-intersection network, which is in turn a subset of the 6-intersection network — so scaling the network size scales *the same* streets rather than comparing unrelated areas.

### 3.3.2 Network files and configuration

Each generated scenario folder was renamed to size-specific filenames to prevent cross-referencing errors (`2int.net.xml`, `2int.sumocfg`, `2int.routes.xml`; likewise for `4int.*` and `6int.*`). Each `.sumocfg` references its own net-file and route-file pair and sets `<begin value="0"/> <end value="3600"/>`.

**Figure 3.3 (insert, asset `figures/osm_extract.png`): OSM map extract** — the selected Johannesburg grid (Jorissen Street / Braamfontein area) showing the nested 2-, 4- and 6-intersection selection boxes used for network extraction, annotated with street names.

For the 2-intersection network (the fully characterised baseline):

| Item | Value |
|---|---|
| Signalised intersections | 2 (OSM IDs `30406197` and `cluster_30406198_7161921427_7161921428`) |
| Monitored approach lanes | 12 (all signalised approach lanes of both intersections) |
| Existing signal program | Two-phase per intersection; cycle length 90 s = 39 s green + 6 s yellow + 39 s green + 6 s yellow |

**Figure 3.4 (insert, asset `figures/network_2int.png`): 2-intersection network** — NETEDIT/SUMO-GUI rendering of the extracted topology as simulated.

**Figure 3.5 (insert, asset `figures/network_4int.png`): 4-intersection network** — rendering of the nested 4-intersection extension.

**Figure 3.6 (insert, asset `figures/network_6int.png`): 6-intersection network** — rendering of the full 6-intersection network.

### 3.3.3 Traffic demand generation

Demand was generated stochastically with SUMO's `randomTrips.py` tool and routed with `duarouter`:

```
python "%SUMO_HOME%\tools\randomTrips.py" -n <net>.net.xml -o trips.xml \
       --period 0.5 --end 3600 --begin 0
duarouter -n <net>.net.xml -t trips.xml -o routes.xml --ignore-errors
```

`--period 0.5` inserts a passenger vehicle every 0.5 s on average (~120 vehicles/min), representative of peak urban flow and a deliberate increase over osmWebWizard's conservative default (~2 cars/30 s). The same route file is reused across all controllers on a given network so that demand never confounds comparisons. During evaluation, run-to-run variability is introduced in a controlled way by passing a distinct integer seed per episode to SUMO (`--seed`), which perturbs stochastic elements (departure jitter, routing tie-breaks) while holding the demand distribution constant.

---

## 3.4 Instrumentation, Computers and Interfaces

### 3.4.1 Software instrumentation

All software is free and open source; no equipment, material or tool purchases were required for this project.

| Component | Role | Version |
|---|---|---|
| Eclipse SUMO (+ NETEDIT, randomTrips, duarouter, osmWebWizard) | Traffic microsimulator and network tools | 1.27.1 |
| Python | Runtime | 3.x (venv-managed; see Appendix A for frozen `requirements.txt`) |
| Gymnasium | Environment API standard | per `requirements.txt` |
| Stable-Baselines3 (PyTorch backend) | PPO and DQN implementations | per `requirements.txt` |
| TraCI (`traci` python package) | Simulator interface | ships with SUMO 1.27.1 |
| TensorBoard | Training-curve visualisation | per `requirements.txt` |
| Visual Studio Code | IDE/editor | latest at time of use |
| Git + GitHub | Version control and backup | repo `RL_TSC_Comparison` |

### 3.4.2 Computing platform

Every one of the twelve simulations was executed on university PC laboratory workstations; no personal hardware or cloud computing was used for any reported result. This means the entire study can be reproduced on the same institutional hardware. The specification and serial number of the machine used are given in Appendix A.

All training and evaluation ran CPU-only and headless (`sumo`, not `sumo-gui`, with `use_gui=False` throughout), allowing the long training budgets to run unattended, with checkpoints every 10 000 steps providing restart points. The interface layer between the learning code and the simulator is exclusively TraCI/TCP; no GUI rendering was used in any training or evaluation run.

**Figure 3.7 (insert): Compute and data-flow diagram** — university PC workstation (VS Code + Python venv + SUMO + PyTorch) → training/evaluation → saved models, checkpoints and TensorBoard logs → Git/GitHub backup, with TensorBoard reading its event files on the same machine.

---

## 3.5 MDP Formulation (Held Constant Across Experiments)

The Markov Decision Process follows Li and Zhuang's formulation for adaptive phase-split control, chosen deliberately because it makes the *control variable* the green split of a fixed-cycle signal rather than arbitrary phase switching, keeping the action space small and identical in meaning across network sizes.

**State.** A vector of length L + M (L monitored approach lanes, M intersections), all elements normalised to [0, 1]:
- per-lane queue: q_i / 50 capped at 1 (50 veh ≈ saturation of an approach);
- per-intersection phase split: (s_j − s_lb)/(s_ub − s_lb).
For the 2-intersection network: 12 + 2 = **14 elements**, `Box(0, 1, (14,), float32)`.

In the MARL configuration this vector is decomposed per agent: each agent observes only the queue lengths of its own approach lanes and its own phase split, zero-padded to the largest lane count among that network's intersections so that all agents share one fixed-size observation, with a one-hot intersection identifier appended (following the parameter-sharing formulation of Kolat et al.). The SARL agent instead observes the full global vector.

**Action.** `Discrete(3·M)`; decoded as intersection index ⌊a/3⌋ and adjustment a mod 3 ∈ {−Δs, 0, +Δs} with Δs = 5 s. For M = 2: **6 actions**. Phase splits are clamped to [s_lb, s_ub] = [15, 63] s around the commissioned initial split of 39 s, preserving the 90 s cycle length, 6 s yellow and 0 s all-red of the imported programs.

**Reward.** Tiered per-lane queue penalty summed over all monitored lanes:

    r_t = Σ_i  0                                  if q_i ≤ q_lc        (free flow)
              −(w_l · q_i)                        if q_lc < q_i ≤ q_hc (light congestion)
              −(w_cp · w_l · q_i)                 if q_i > q_hc        (heavy congestion)

with thresholds q_lc = 5 veh, q_hc = 15 veh, link weight w_l = 1.0 and heavy-congestion multiplier w_cp = 3.0. 

**Timing.** One decision per 90 s (one full cycle): the agent observes, adjusts the split, and SUMO advances 90 s before the next decision. An episode is therefore 3600 s / 90 s = **40 decision steps**.

---

## 3.6 Approach to Testing: Experimental Design

### 3.6.1 Factors investigated

All twelve simulations share the same simulator, environment physics, demand files, reward function and training budget, and differ only in three factors:

| Factor | Levels |
|---|---|
| Learning algorithm | PPO, DQN |
| Controller architecture | SARL (one agent, global state, joint action) vs MARL (N agents, parameter sharing, local state + intersection index, shared global reward) |
| Network scale | 2, 4, 6 intersections |

The resulting full-factorial experiment matrix is:

| Controller | 2int | 4int | 6int |
|---|---|---|---|
| SARL PPO | ✔ | ✔ | ✔ |
| SARL DQN | ✔ | ✔ | ✔ |
| MARL PPO | ✔ | ✔ | ✔ |
| MARL DQN | ✔ | ✔ | ✔ |

Each cell is one independently trained controller, giving 12 simulations in total and allowing the SARL/MARL comparison to be repeated at every network size rather than at a single scale.

### 3.6.2 Parameters held constant

- Network geometry and signal program structure (cycle 90 s, yellow 6 s, all-red 0 s);
- Demand generation procedure and route files;
- Episode horizon (3600 s) and decision interval (90 s);
- Action semantics (three discrete phase-split adjustments, Δs = 5 s), split bounds [15, 63] s and initial split 39 s;
- Per-lane queue normalisation (q/50) and per-intersection split normalisation;
- Reward thresholds and weights (identical reward function for every controller, SARL and MARL);
- Evaluation protocol (Section 3.8): identical seeds, deterministic policies, identical metrics.

These choices follow the supervisor's directive to keep SARL and MARL as identical as possible so that observed differences are attributable to the architecture alone.

### 3.6.3 Parameters adjusted

Only the three factors above are varied; everything else is held constant. Algorithm hyperparameters are *not* tuned per experiment beyond what is standard for each algorithm family (Section 3.7), to avoid conflating tuning effort with architectural performance. The factorial structure allows the main effects of algorithm, architecture and scale to be separated, and ensures the SARL/MARL comparison is confirmed at every network size.

---

## 3.7 Training Protocol

A single uniform training budget of **260 000 timesteps** (~6 500 episodes) applies to every one of the twelve simulations. Checkpoints were written every 10 000 steps for all runs, providing a common restart point should a long run be interrupted and a common basis for the results analysis.

**SARL PPO (3 runs).** Stable-Baselines3 PPO, `MlpPolicy`: learning rate 3×10⁻⁴, n_steps 40, batch size 10, n_epochs 10, γ = 0.99. On the 2-intersection network an initial 10 000-step pilot run verified the pipeline end-to-end; training then resumed from the pilot weights (`reset_num_timesteps=False`) to the full 260 000-step budget. The same hyperparameters were used unchanged on the 4- and 6-intersection networks.

**SARL DQN (3 runs).** Hyperparameters informed by van der Pol's DQN formulation: learning rate 1×10⁻⁴, replay buffer 50 000, learning_starts 1 000, batch size 32, γ = 0.99, target-network update every 1 000 steps, ε-greedy exploration decaying 1.0 → 0.05 over the first 20 % of training, train_freq 4. Trained on the university PC workstations and resumed from checkpoints when runs were interrupted.

**MARL PPO (3 runs).** One shared PPO policy for all N agents on a network (parameter sharing, as in Kolat et al.), with agent identity encoded by the intersection index in each per-agent observation. At each decision point all agents ingest their local padded observations, select actions in parallel, and all receive the same network-wide reward before the simulation advances. Hyperparameters are identical to SARL PPO for comparability. Trained with the same checkpointing/resume regime as the other runs.

**MARL DQN (3 runs).** The same parameter-sharing MARL structure driven by DQN, using the SARL DQN hyperparameters and the same lock-step, shared-global-reward training loop. Trained on the university PC workstations under the same checkpointing/resume regime.

Training progress for every run is captured in TensorBoard event files; the primary convergence indicator is `rollout/ep_rew_mean`.

---

## 3.8 Evaluation Protocol

All twelve trained controllers are evaluated identically:

- **Repetitions:** 10 evaluation episodes per controller, using the fixed seed list `[42, 123, 256, 512, 999, 1337, 2024, 3141, 7777, 9999]`. Seeds are passed to SUMO at `reset()` so demand realisations differ between episodes but are exactly reproducible by anyone re-running the list.
- **Policy mode:** deterministic (`model.predict(..., deterministic=True)`), i.e. no exploration noise at evaluation.
- **Metrics recorded per episode:** cumulative episode reward; time-average of the mean queue across all monitored lanes; throughput (total arrived vehicles, accumulated per step); mean travel time; mean delay; teleport count.
- **Reporting:** mean ± standard deviation across the 10 seeds for every metric; results compared only between controllers evaluated on the same network and seed set.

An initial evaluation with a fully deterministic route file produced zero-variance repeats (σ = 0), which motivated the seeded-stochasticity protocol above; the throughput-metric bug described in Section 3.2.2 was also discovered and fixed during evaluation commissioning, and all reported results use the corrected accumulation.

---

## 3.9 Data Capture and Management

- **Code and configs:** committed to GitHub daily (repository `RL_TSC_Comparison`), giving a full audit trail of environment changes.
- **Models:** final models saved as `.zip` (e.g. `sarl_ppo_2int_260k.zip`); intermediate checkpoints every 10 000 steps provide restart points for long runs.
- **Logs:** TensorBoard event files per run under `agents/logs/<run_name>/`, retained for supervisor review and results-chapter figures.
- **Backups:** all models, checkpoints and logs were committed to Git/GitHub after each completed run.

---

## 3.10 Verification Measures

Functionality of the complete chain was verified incrementally before any result was accepted:

1. Visual verification in `sumo-gui` that vehicles obey signals on each new network;
2. TraCI health-check script printing per-lane halting counts during a manual drive-through of the simulation;
3. `check_env` compliance plus a random-agent smoke test confirming observation shape, action decoding and reward sign/magnitude;
4. Monitoring of teleports in every evaluation (zero teleports in all reported runs indicates no gridlock artefacts contaminate the metrics);
5. Cross-checking that identical seeds produce identical metric sets across controllers, confirming the harness is deterministic given a seed.

---

## Appendix A — Equipment and Instrumentation Inventory

*(Serial numbers / exact identifiers to be transcribed from the physical machines and `pip freeze` output before submission.)*

| Item | Description | Identifier / Serial |
|---|---|---|
| Computing platform | University PC laboratory workstation, CPU-only training | Model: ______ ; S/N: ______ ; CPU: ______ ; RAM: ______ ; Lab/room: ______ |
| Number of workstations used | Machines on which the 12 runs were performed | ______ |
| Software: SUMO | Eclipse SUMO microsimulator | v1.27.1 (all workstations) |
| Software: Python | CPython runtime in project venv | v______ (`py --version`) |
| Libraries | gymnasium, stable-baselines3, torch, tensorboard, traci, numpy | versions per frozen `requirements.txt` (Appendix B) |
| VCS | GitHub repository | `RL_TSC_Comparison` (URL: ______ ) |

## Appendix B — Reproduction Checklist

1. Clone `RL_TSC_Comparison`; create venv; `pip install -r requirements.txt`.
2. Install SUMO 1.27.1; set `SUMO_HOME`.
3. Rebuild or reuse committed networks (`2int_files/2int.sumocfg` etc.).
4. Train: the configuration-specific scripts (`train_*.py`) for SARL PPO, SARL DQN, MARL PPO and MARL DQN, with the network selected in each config — or load the committed `.zip` models.
5. Evaluate: the evaluation script with the fixed seed list; results print as mean ± σ tables and dump JSON.

## Appendix C — Configuration Constants per Network

```
# SARL MDP (L = monitored approach lanes, M = intersections)
obs = Box(0, 1, (L + M,), float32)     # 2int: 14 elements; 4int/6int per config
act = Discrete(3·M)                    # 2int: 6 actions; 4int/6int per config
cycle_length = 90 s   yellow = 6 s   all-red = 0 s   init_split = 39 s
delta_s = 5 s   s_lb = 15 s   s_ub = 63 s
q_lc = 5 veh   q_hc = 15 veh   w_l = 1.0   w_cp = 3.0
sim_steps = 3600   step_length = 90
training_budget = 260 000 timesteps (all 12 runs, checkpoint every 10 000)
# MARL: per-agent local obs = own lanes padded to max lane count + intersection index
eval_seeds = [42, 123, 256, 512, 999, 1337, 2024, 3141, 7777, 9999]
```
