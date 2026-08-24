# Handoff: 6-intersection SARL PPO setup (for Claude)

Goal: replicate the existing SARL PPO pipeline (currently training on the 4-int
network) for the **6-int network** in `../../6int_files/`. Everything lives in
`agents/opencode/`. Read this whole file before writing code.

---

## 1. The fix you must understand: mixed 4-phase / 6-phase junctions

### Background
The environment (`sumo_env_4int.py`, class `SumoTSCEnv`) implements the Li &
Zhuang MDP formulation:

- **State** = normalised queue per approach lane + normalised current phase
  split per junction (`obs_size = num_links + num_tls`)
- **Action** = pick one junction, adjust its split by {-Δs, 0, +Δs}
  → `Discrete(3 * num_tls)`
- **Reward** = tiered queue penalty (+ teleport penalty, optional throughput)
- One decision every `step_length=90s` (one full cycle), episode =
  `sim_steps=3600s`

The agent does **not** choose phases directly. It controls one scalar per
junction, the **split** (`phase_splits[tl]`, bounded `[s_lb=15, s_ub=63]`),
and `_apply_action()` translates that scalar into concrete SUMO phase
durations each step.

### The bug that was fixed
The original 2-int formulation assumed every TLS program has exactly 4
phases: `[main_green, yellow, cross_green, yellow]`:

```
phase[0].duration = new_split
phase[2].duration = cycle_length - new_split - 2*yellow - allred
```

In the 4-int network, one junction
(`cluster_54994556_7161921429`) has a **6-phase** program — it has an extra
protected/permissive left-turn stage pair in the middle:

```
phase[0] dur=33  GGggggGgrrrrrrrGggggggg   <- main street green  (primary)
phase[1] dur=6   yyygggyyrrrrrrryggggggg   <- yellow
phase[2] dur=6   rrrggGrrrrrrrrrrgggGggg   <- protected left stage (middle green)
phase[3] dur=6   rrryyyrrrrrrrrrryyyyyyy   <- yellow
phase[4] dur=33  rrrrrrrrGGGGGGGrrrrrrrr   <- main street green  (final)
phase[5] dur=6   rrrrrrrryyyyyyyrrrrrrrr   <- yellow
```

Naively applying the 4-phase formula there corrupts the program (overwrites a
yellow/middle stage, cycle length drifts off 90 s).

### The fix (implemented in `sumo_env_4int.py`, `_apply_action`, ~lines 157–204)

Branch on `len(logic.phases)`:

- **`len(phases) == 4`** → original formula, unchanged.
- **Any other N-phase cycle** → redistribute while preserving structure:
  - `phases[0]` (primary green) ← `new_split`
  - middle green stages (even indices `2 … N-3`, e.g. the protected-left 6 s
    stage) **keep their network-default durations** — they are NOT controllable
  - all yellows (odd indices, incl. the last phase) keep defaults
  - `phases[N-2]` (final main green) absorbs the remainder:
    `cycle_length - new_split - sum(middle_greens) - sum(yellows) - allred`
  - phase count and phase order are never changed; a brand-new
    `traci.trafficlight.Logic` is pushed via `setProgramLogic`.

For the 6-phase case the arithmetic works out to:
`final_green = 90 - split - 6 - 18 - 0 = 66 - split`, which stays positive over
the whole legal split range `[15, 63]` → `final_green ∈ [3, 51]`.

This was verified by `sanity_check_4int.py` (asserts): on the 6-phase junction,
`+Δs` gives durations `[44,6,6,6,22,6]`, then `-10` gives `[34,6,6,6,32,6]`,
cycle always == 90 s.

**Consequence for you:** the fix is already generalised — `SumoTSCEnv` needs
**no code changes** for 6int. The class is fully config-driven; all network
specifics live in the config dicts of the train/resume scripts.

---

## 2. Existing files to mirror (all in `agents/opencode/`)

| File | Role |
|---|---|
| `sumo_env_4int.py` | `SumoTSCEnv` gym env (generic, config-driven) |
| `train_sarl_4int.py` | fresh PPO training run |
| `resume_sarl_4int.py` | resume from checkpoint argv[1] → 260k |
| `sanity_check_4int.py` | env + split-redistribution assertions |

PPO hyperparameters (keep identical for comparability):
```python
PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=40, batch_size=10,
    n_epochs=10, gamma=0.99, verbose=1, tensorboard_log="./logs/")
```
Callbacks: `CheckpointCallback(save_freq=10000, save_path="./checkpoints/",
name_prefix=...)` + `EpisodeMetricsCallback` (logs `metrics/mean_travel_time`
and `metrics/mean_delay` from `env.mean_travel_time` / `env.mean_delay`,
parsed from per-episode `tripinfo_ep{N}.xml`). Total timesteps target:
**260000**, same as 2int/4int runs.

Shared config values (identical to 4int):
```python
cycle_length=90, yellow_time=6, allred_time=0, init_split=39,
s_lb=15, s_ub=63, delta_s=5, q_lc=5, q_hc=15, w_l=1.0, w_cp=3.0,
sim_steps=3600, step_length=90, use_gui=False
```

---

## 3. The 6-int network (already built, in `../../6int_files/`)

- Config: `6int_osm.sumocfg`; net: `6int_osm.net.xml`; routes:
  `6int_routes.xml` (referenced by the sumocfg — use relative path
  `"../../6int_files/6int_osm.sumocfg"` from `agents/opencode/`).
- All six tlLogics have `programID=0`, `type="actuated"`, and already sum to a
  90 s cycle:

| tl id | #phases | default durations |
|---|---|---|
| `30406195` | 4 | 39, 6, 39, 6 |
| `30406197` | 4 | 39, 6, 39, 6 |
| `54994558` | 4 | 39, 6, 39, 6 |
| `768241766` | 4 | 39, 6, 39, 6 |
| `cluster_30406198_7161921427_7161921428` | 4 | 39, 6, 39, 6 |
| `cluster_54994556_7161921429` | **6** | 33, 6, 6, 6, 33, 6 |

So 5 standard junctions + 1 six-phase junction — exactly the case the fix
handles.

### ⚠️ The one thing you MUST derive yourself: `link_ids`
`link_ids` are the observed approach lanes (queues are read with
`lane.getLastStepHaltingNumber`). The 4int list (23 lanes for 4 junctions) was
hand-picked from the net. For 6int you must build the equivalent list for all
6 junctions. Recommended method:

```python
import traci
links = traci.trafficlight.getControlledLinks(tl_id)
# collect incoming lane ids from those links, dedupe, order deterministically
```
or parse `<connection tl="..." from="..."/>` entries in `6int_osm.net.xml`.
Match the 4int convention: include the approach lanes of each junction
(multiple sub-lanes per approach where they exist, e.g. `4761588#1_0.._3`).
Observation size will be `len(link_ids) + 6`; sanity-check it prints what you
expect.

---

## 4. Deliverables to create (in `agents/opencode/`)

1. **`train_sarl_6int.py`** — clone of `train_sarl_4int.py` with:
   - `config["sumocfg"] = "../../6int_files/6int_osm.sumocfg"`
   - the 6 `tl_ids` listed above
   - your derived `link_ids`
   - `name_prefix="sarl_ppo_6int"`, `tb_log_name="sarl_ppo_6int"`
   - `TOTAL_TIMESTEPS` from argv, default 260000
   - final model saved as `sarl_ppo_6int_260k`
   - keep the `check_env(env, warn=True)` call
2. **`resume_sarl_6int.py`** — clone of `resume_sarl_4int.py`:
   - checkpoint path from `sys.argv[1]` (default latest 6int ckpt)
   - `reset_num_timesteps=False`, `tb_log_name="sarl_ppo_6int_resumed"`
   - continues to `TOTAL_TIMESTEPS=260000`
3. **`sanity_check_6int.py`** — clone of `sanity_check_4int.py`:
   - print obs shape and `Discrete(18)` action space
   - assert baseline programs load with cycles == 90
   - apply `+5` then `-10` on BOTH a 4-phase junction and
     `cluster_54994556_7161921429`; assert durations match the redistribution
     rules above and phase counts stay 4 / 6
4. Reuse `from sumo_env_4int import SumoTSCEnv` (or copy to
   `sumo_env_6int.py` if you prefer naming symmetry — either is fine, do NOT
   fork the logic).

## 5. Run conventions in this workspace (important)

- Python: repo-root `venv/Scripts/python.exe` (SUMO 1.27.1 + SB3 installed;
  `SUMO_HOME` already configured on this machine).
- Always run scripts with CWD = `agents/opencode/` (configs use relative
  paths; `tripinfo_ep*.xml` and checkpoints land in CWD).
- Long runs are launched **detached via WMI**, because interactive shells reap
  children when they exit:
  ```powershell
  Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = 'cmd.exe /c ""...\venv\Scripts\python.exe" train_sarl_6int.py > "logs\train_sarl_6int_console.log" 2> "logs\train_sarl_6int_err.log""'
    CurrentDirectory = "...\agents\opencode" }
  ```
  (`Start-Process` from the agent shell gets killed — don't use it.)
- Monitor progress: `Select-String logs\train_sarl_6int_console.log -Pattern "total_timesteps"`.
  TensorBoard is already running on port 6006 pointed at `agents/opencode/logs`.
- Each launch spawns 2 python.exe processes (venv launcher + real interpreter)
  plus one `sumo.exe` — that is normal, not a duplicate trainer.
