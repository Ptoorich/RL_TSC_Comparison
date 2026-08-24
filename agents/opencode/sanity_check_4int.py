import traci
from sumo_env_4int import SumoTSCEnv

TL_4PH   = "30406197"
TL_6PH   = "cluster_54994556_7161921429"

config = {
    "sumocfg"      : "../../4int_files/4int_osm.sumocfg",
    "tl_ids"       : [
        "30406197",
        "cluster_30406198_7161921427_7161921428",
        "54994558",
        "cluster_54994556_7161921429",
    ],
    "link_ids"     : [
        "596638461#0_0",
        "4761588#1_0", "4761588#1_1", "4761588#1_2", "4761588#1_3",
        "-621271001_0", "669676574#2_0", "1190301745#0_0",
        "-669676574#9_0", "622835372_0", "622835372_1", "622835372_2",
        "380081149#1_0", "42790161#2_0", "42790161#2_1", "42790161#2_2",
        "-621271003_0", "669676574#1_0", "623540942#0_0",
        "623540942#0_1", "623540942#0_2", "-669676574#2_0", "621271001_0",
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

def show(env, tl, label):
    logic = traci.trafficlight.getAllProgramLogics(tl)[0]
    durs  = [p.duration for p in logic.phases]
    total = sum(durs)
    ok    = "OK" if total == env.cycle_length else "!! CYCLE != 90"
    print(f"  {label}: {len(durs)} phases, durations={durs}, "
          f"cycle={total}s {ok}")

env = SumoTSCEnv(config)
obs, _ = env.reset()
print(f"obs shape: {obs.shape} (expect (27,))")
print(f"action space: {env.action_space} (expect Discrete(12))")

print("\n-- Baseline programs --")
for tl in config["tl_ids"]:
    show(env, tl, tl[:28])

print("\n-- 4-phase junction: +5s via _apply_action --")
env._apply_action(TL_4PH, +5)
show(env, TL_4PH, TL_4PH)
assert abs(env.phase_splits[TL_4PH] - 44) < 1e-9
logic = traci.trafficlight.getAllProgramLogics(TL_4PH)[0]
assert len(logic.phases) == 4
assert abs(logic.phases[0].duration - 44) < 1e-9
assert abs(logic.phases[2].duration - 34) < 1e-9, "ew_green should be 90-44-12"

print("\n-- 6-phase junction: +5s then -10s --")
env._apply_action(TL_6PH, +5)
show(env, TL_6PH, "after +5s")
logic = traci.trafficlight.getAllProgramLogics(TL_6PH)[0]
assert len(logic.phases) == 6, "phase count must stay 6"
d = [p.duration for p in logic.phases]
assert d[0] == 44 and d[2] == 6 and d[4] == 22, f"unexpected {d}"
env._apply_action(TL_6PH, -10)
show(env, TL_6PH, "after -10s")
d = [p.duration for p in traci.trafficlight.getAllProgramLogics(TL_6PH)[0].phases]
assert d[0] == 34 and d[2] == 6 and d[4] == 32, f"unexpected {d}"

print("\n-- Full MDP step: action 11 (= junction idx 3, +delta) --")
env.phase_splits = {tl: 39 for tl in config["tl_ids"]}
env._apply_action(TL_6PH, 0)
for _ in range(3):
    obs, r, term, trunc, info = env.step(11)
    print(f"  step reward={r:9.2f}  terminated={term}  "
          f"s[junction3]={env.phase_splits[TL_6PH]}")

env.close()

print("\nALL SANITY CHECKS PASSED")
