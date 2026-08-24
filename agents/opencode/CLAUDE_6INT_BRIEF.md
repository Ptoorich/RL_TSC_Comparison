# Handoff Brief: SARL PPO Training on Multi-Stage (6-Phase) Junctions

Context for setting up **6-intersection (6int) SARL PPO training**, replicating what
was done for the 4int network. Everything below reflects working, tested code.

---

## 1. Project context

- Research project: RL Traffic Signal Control comparison (Li & Zhuang 2026 MDP formulation).
- Simulator: SUMO 1.27.1 via TraCI. RL: Stable-Baselines3 PPO/DQN, Gymnasium env.
- Custom environment: `agents/opencode/sumo_env_4int.py` (copy of original
  `agents/sumo_env.py` plus one backward-compatible fix - see section 3).
  - State: normalised queue length per approach lane + phase split per intersection
  - Action: `Discrete(3 * num_tls)` -> pick junction j, adjust its split by -delta_s / 0 / +delta_s
  - Reward: tiered queue penalty (q<=5 free, q<=15 -1*q, else -3*q) + teleport penalty (-5 each);
    optional throughput term disabled (w_throughput=0) for PPO runs
- Fixed signal params: cycle=90s, yellow=6s, allred=0, init_split=39, bounds [15,63], delta_s=5
- Episode: 3600 SUMO seconds = one decision per cycle (90s) = exactly 40 env steps/episode
- Training target: single continuous `model.learn(total_timesteps=260000)`,
  CheckpointCallback every 10000 steps, final model saved as `<name>_260k.zip`

## 2. The problem the fix solves

OSM-built networks contain signalised junctions whose tlLogic has MORE than 4 phases.
Example (4int net, junction `cluster_54994556_7161921429`):

    phase 0: 33s green   (mainline N-S, includes permissive turns + u-turns)
    phase 1:  6s yellow
    phase 2:  6s green   (protected turn stage)
    phase 3:  6s yellow
    phase 4: 33s green   (cross-street arm)
    phase 5:  6s yellow

The ORIGINAL `_apply_action` assumed every junction is a simple 4-phase
`[G, y, G, y]` program and rebuilt the logic as:

    new_phases = [Phase(new_split, p0.state), p1, Phase(ew_green, p2.state), p3]
    traci.trafficlight.setProgramLogic(tl_id, new_logic)

Applied to a 6-phase junction this SILENTLY DELETES phases 4 and 5 on the agent's
first action touching that junction. One entire approach never gets green again ->
unbounded queue -> tiered penalty fires forever -> reward signal from that junction
is garbage and poisons PPO updates. It does not crash; it just ruins training.
Additional corruption: the actual cycle length changes (90 -> 95s), breaking the
episode timing invariant, while the env's internal `phase_splits` tracker keeps
reporting values that no longer match reality.

## 3. The fix (tested, in production use)

Location: `agents/opencode/sumo_env_4int.py`, method `_apply_action`.
Semantics: the agent's MDP is UNCHANGED (same obs, same Discrete(12)-style actions,
same reward). Only how SUMO executes "+5s / -5s at junction j" differs:

- 4-phase junctions: byte-for-byte original behaviour (2int results stay comparable)
- N-phase junctions (N > 4):
  * primary green (phase 0) receives the new split s
  * intermediate green stages keep their net-default durations (fixed)
  * the FINAL main green (phase index N-2) absorbs the remainder so the cycle stays constant
  * yellows are never touched

```python
logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
phases = logic.phases

if len(phases) == 4:
    # Standard two-green cycle (2int formulation, unchanged)
    ew_green = (self.cycle_length - new_split
                - 2 * self.yellow_time - self.allred_time)
    new_phases = [
        traci.trafficlight.Phase(new_split, phases[0].state),
        phases[1],
        traci.trafficlight.Phase(ew_green,  phases[2].state),
        phases[3],
    ]
else:
    # Multi-stage cycle: primary green takes the split, intermediate
    # green stages keep their defaults, final main green absorbs remainder.
    middle_greens = sum(phases[i].duration
                        for i in range(2, len(phases) - 2, 2))
    yellows       = sum(phases[i].duration
                        for i in range(1, len(phases), 2))
    last_green    = (self.cycle_length - new_split
                     - middle_greens - yellows - self.allred_time)

    new_phases = [phases[i] for i in range(len(phases))]
    new_phases[0]               = traci.trafficlight.Phase(
        new_split,  phases[0].state)
    new_phases[len(phases) - 2] = traci.trafficlight.Phase(
        last_green, phases[len(phases) - 2].state)

new_logic = traci.trafficlight.Logic(
    logic.programID, logic.type, logic.currentPhaseIndex, new_phases)
traci.trafficlight.setProgramLogic(tl_id, new_logic)
```

Notes:
- `new_split` is already clamped to [s_lb, s_ub] before this block, and
  `self.phase_splits[tl_id]` updated, exactly as in the original code.
- For the example 6-phase junction: last_green = 90 - s - 6(middle) - 18(yellows)
  = 66 - s, which stays positive for s in [15,63] (range [3,51]).
- Verified via TraCI sanity checks: phase COUNT preserved, cycle sum always 90s,
  4-phase path produces identical durations to the original implementation.

## 4. Recipe to build the 6int scripts

Work in `agents/opencode/`. NEVER modify originals under `agents/` root
(project owner rule: copies only). Run everything with the repo venv:
`venv\Scripts\python.exe`, working directory = `agents/opencode`.

### Step A - identify controlled junctions and lanes

From `6int_files/6int_osm.net.xml`:

```powershell
Select-String -Path "6int_files\6int_osm.net.xml" `
  -Pattern '<junction id="[^"]*" type="traffic_light"' |
  ForEach-Object { $_.Line.Trim() }
```

Each match gives you:
- `tl_ids`     : the junction `id` attribute (must equal the tlLogic ids)
- `link_ids`   : that junction's `incLanes` attribute (space-separated lane ids)

`link_ids` = concatenation of all junctions' incLanes, grouped per junction,
in the SAME order as `tl_ids` (obs layout: queues of all link_ids, then splits).

### Step B - verify phase structures

Count tlLogic phases per junction (any count works with the fixed env, but log it):

```powershell
Select-String -Path "6int_files\6int_osm.net.xml" -Pattern '<tlLogic' -Context 0,30
```

Expect a mix of 4-phase and 6-phase programs across 6 intersections.

### Step C - create `train_sarl_ppo_6int.py`

Copy the structure of `train_sarl_4int.py` (same folder) and change ONLY:

1. `"sumocfg": "../../6int_files/6int_osm.sumocfg"`
2. `"tl_ids"` / `"link_ids"` from Step A
3. model/checkpoint/tensorboard names: `sarl_ppo_6int*`
4. Keep EXACTLY these PPO hyperparameters (identical to 2int/4int runs):
   MlpPolicy, learning_rate=0.0003, n_steps=40, batch_size=10,
   n_epochs=10, gamma=0.99, verbose=1, tensorboard_log="./logs/"
5. Keep: check_env validation, EpisodeMetricsCallback (travel time/delay to TB),
   CheckpointCallback(save_freq=10000, name_prefix="sarl_ppo_6int"),
   `model.learn(total_timesteps=260000, ...)`, final save `sarl_ppo_6int_260k`

### Step D - sanity check BEFORE long training

Copy `sanity_check_4int.py` pattern; assert for EVERY junction:
- obs shape == (len(link_ids) + 6,)
- action space == Discrete(18)
- baseline durations read back correctly
- applying +5s then -10s preserves phase count and cycle sum == 90s
  on BOTH a 4-phase and a 6-phase junction
- one full env.step() returns finite reward

### Step E - smoke test, then full run

```
python train_sarl_ppo_6int.py 120      # ~1 min smoke test
python train_sarl_ppo_6int.py          # full 260k run
```

Launch detached so terminal/editor closure doesn't kill it:

```powershell
Start-Process -FilePath "..\..\venv\Scripts\python.exe" `
  -ArgumentList "train_sarl_ppo_6int.py" -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput "$PWD\logs\train_6int_console.log" `
  -RedirectStandardError  "$PWD\logs\train_6int_err.log" `
  -WindowStyle Hidden -PassThru
```

Runtime expectation: 4int ran ~15 s/episode (~28 h total) headless on CPU.
6int will be slower (more lanes/vehicles) - measure one episode first and report ETA.

## 5. Known environment behaviours (not bugs)

- TraCI occasionally dies mid-training with
  `tcpip::Socket::recvAndCheck @ recv: Socket reset by peer`.
  Recovery: relaunch via a resume script loading the newest
  `checkpoints/sarl_ppo_6int_*_steps.zip` with `reset_num_timesteps=False`
  (see `resume_sarl_4int.py`). Max loss = 10k steps.
- Teleport warnings during training are normal (jam/yield teleports under
  fixed-time baseline while exploring).
- Demand files intentionally keep u-turn routes (~16% of trips) for consistency
  with previously recorded experiments - do not regenerate demand.
- Evaluation (AFTER training): mirror `evaluate_sarl_dqn_v2.py`: 10 seeds
  [42,123,256,512,999,1337,2024,3141,7777,9999], deterministic=True, record
  reward, avg queue, throughput, teleports, mean travel time, mean delay ->
  save JSON under `results/`.

## 6. Monitoring

TensorBoard already serves http://localhost:6006 (logdir = agents/opencode/logs).
Key charts: rollout/ep_rew_mean, metrics/mean_travel_time, metrics/mean_delay.
For the 6int run, either point a second TB instance at the same logdir parent
or restart TB with logdir covering both run families.
