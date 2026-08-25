import os
import sys

os.environ["SUMO_HOME"] = r"C:\RL_TSC"
tools = os.path.join(os.environ["SUMO_HOME"], "tools")
bin_path = os.path.join(os.environ["SUMO_HOME"], "bin")

if tools not in sys.path:
    sys.path.append(tools)
if bin_path not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + bin_path

import gymnasium as gym
import numpy as np
import traci
import xml.etree.ElementTree as ET


class SumoTSCEnv6Int(gym.Env):
    """
    SARL environment for the 6-intersection network.
    MDP formulation identical to 2-intersection SARL PPO.

    Handles mixed phase structures:
    - Five standard 4-phase junctions: phase[0]=NS green, phase[2]=EW green
    - One 6-phase junction (cluster_54994556_7161921429):
        Phase 0: main NS green (33s)
        Phase 1: NS yellow    (6s)
        Phase 2: protected left green (6s)  -- kept at network default
        Phase 3: left yellow  (6s)          -- kept at network default
        Phase 4: EW green     (33s)         -- absorbs remainder
        Phase 5: EW yellow    (6s)
      Agent controls phase[0] split only; middle phases fixed;
      phase[4] = cycle - split - middle_greens - all_yellows

    NOTE: the 6-phase junction has more fixed overhead (protected left +
    3 yellows = 24s) than the standard 4-phase junctions (2 yellows = 12s).
    That leaves only 66s of cycle time to split between phase[0] and
    phase[4] on this junction, versus 78s on standard junctions. The
    global s_lb/s_ub action bounds are sized for the standard junctions,
    so _apply_action derives a tighter, junction-specific upper bound for
    the 6-phase TL to guarantee phase[4] never has to be force-clamped
    (which previously broke the 90s cycle-length invariant).
    """

    # Junction that requires special 6-phase handling
    SIX_PHASE_TL = "cluster_54994556_7161921429"

    def __init__(self, config):
        super().__init__()

        self.sumocfg   = config["sumocfg"]
        self.tl_ids    = config["tl_ids"]
        self.link_ids  = config["link_ids"]
        self.num_tls   = len(self.tl_ids)
        self.num_links = len(self.link_ids)
        self.use_gui   = config.get("use_gui", False)

        self.delta_s      = config.get("delta_s", 5)
        self.s_lb         = config.get("s_lb", 15)
        self.s_ub         = config.get("s_ub", 63)
        self.cycle_length = config.get("cycle_length", 90)
        self.yellow_time  = config.get("yellow_time", 6)
        self.allred_time  = config.get("allred_time", 0)
        self.init_split   = config.get("init_split", 39)

        self.q_lc         = config.get("q_lc", 5)
        self.q_hc         = config.get("q_hc", 15)
        self.w_l          = config.get("w_l", 1.0)
        self.w_cp         = config.get("w_cp", 3.0)
        self.w_throughput = config.get("w_throughput", 0.0)

        self.sim_steps     = config.get("sim_steps", 3600)
        self.step_length   = config.get("step_length", 90)
        self.current_step  = 0
        self.episode_count = 0

        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0

        obs_size = self.num_links + self.num_tls
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        # 3 actions × 6 intersections = 18 discrete actions
        self.action_space = gym.spaces.Discrete(3 * self.num_tls)

        print(f"SumoTSCEnv6Int initialised:")
        print(f"  Intersections: {self.num_tls}")
        print(f"  Monitored lanes: {self.num_links}")
        print(f"  Obs size: {obs_size}")
        print(f"  Action space: Discrete({3 * self.num_tls})")
        print(f"  6-phase junction: {self.SIX_PHASE_TL}")

    def _decode_action(self, action):
        tl_idx    = action // 3
        adj_idx   = action % 3
        tl_id     = self.tl_ids[tl_idx]
        delta_map = {0: -self.delta_s, 1: 0, 2: self.delta_s}
        return tl_id, delta_map[adj_idx]

    def _get_queue_lengths(self):
        try:
            return [traci.lane.getLastStepHaltingNumber(lane)
                    for lane in self.link_ids]
        except Exception:
            return [0] * self.num_links

    def _get_observation(self):
        queues = self._get_queue_lengths()
        splits = [self.phase_splits[tl] for tl in self.tl_ids]
        q_max  = 50.0
        norm_q = [min(q / q_max, 1.0) for q in queues]
        norm_s = [(s - self.s_lb) / (self.s_ub - self.s_lb) for s in splits]
        return np.array(norm_q + norm_s, dtype=np.float32)

    def _compute_reward(self):
        try:
            total = 0.0
            for q in self._get_queue_lengths():
                if q <= self.q_lc:
                    total += 0.0
                elif q <= self.q_hc:
                    total += -(self.w_l * q)
                else:
                    total += -(self.w_cp * self.w_l * q)
            arrived   = traci.simulation.getArrivedNumber()
            total    += self.w_throughput * arrived
            teleports = traci.simulation.getStartingTeleportNumber()
            total    += -(5.0 * teleports)
            return total
        except Exception:
            return 0.0

    def _apply_action_standard(self, tl_id, new_split):
        """Standard 4-phase junction: phase[0]=NS green, phase[2]=EW green."""
        ew_green = (self.cycle_length
                    - new_split
                    - 2 * self.yellow_time
                    - self.allred_time)

        logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = logic.phases

        new_phases = [
            traci.trafficlight.Phase(new_split, phases[0].state),
            phases[1],
            traci.trafficlight.Phase(ew_green,  phases[2].state),
            phases[3],
        ]
        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type,
            logic.currentPhaseIndex, new_phases
        )
        traci.trafficlight.setProgramLogic(tl_id, new_logic)

    def _apply_action_6phase(self, tl_id, new_split):
        """
        6-phase junction (cluster_54994556_7161921429):
          Phase 0: NS main green        <- agent controls (new_split)
          Phase 1: NS yellow            <- fixed (6s)
          Phase 2: protected left green <- fixed at network default (6s)
          Phase 3: left yellow          <- fixed (6s)
          Phase 4: EW green             <- absorbs remainder
          Phase 5: EW yellow            <- fixed (6s)

        middle_greens = phase[2].duration = 6s
        all_yellows   = phase[1] + phase[3] + phase[5] = 18s
        phase[4] = cycle - new_split - middle_greens - all_yellows

        new_split is guaranteed by _apply_action to already respect this
        junction's tighter bounds, so ew_green should always come out
        >= yellow_time here. No clamp is applied — if the invariant is
        ever violated it means the bound calculation itself is wrong,
        and the assertion below should surface that loudly rather than
        silently producing an infeasible phase plan.
        """
        logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = logic.phases

        assert len(phases) == 6, (
            f"Expected 6 phases for {tl_id}, got {len(phases)}"
        )

        middle_greens = phases[2].duration          # protected left (6s)
        all_yellows   = (phases[1].duration         # NS yellow
                         + phases[3].duration       # left yellow
                         + phases[5].duration)      # EW yellow

        ew_green = (self.cycle_length
                    - new_split
                    - middle_greens
                    - all_yellows)

        new_phases = [
            traci.trafficlight.Phase(new_split, phases[0].state),
            phases[1],                  # NS yellow — unchanged
            phases[2],                  # protected left — unchanged
            phases[3],                  # left yellow — unchanged
            traci.trafficlight.Phase(ew_green, phases[4].state),
            phases[5],                  # EW yellow — unchanged
        ]

        # Sanity check: total cycle should equal cycle_length
        total = sum(p.duration for p in new_phases)
        assert abs(total - self.cycle_length) < 1.0, (
            f"Cycle length mismatch: {total} != {self.cycle_length}"
        )

        new_logic = traci.trafficlight.Logic(
            logic.programID, logic.type,
            logic.currentPhaseIndex, new_phases
        )
        traci.trafficlight.setProgramLogic(tl_id, new_logic)

    def _apply_action(self, tl_id, delta):
        if tl_id == self.SIX_PHASE_TL:
            # This junction has more fixed overhead (protected left + 3
            # yellows) than the standard 4-phase junctions, so the global
            # s_ub is too permissive here. Derive the true max split from
            # this junction's own current phase durations so ew_green
            # never has to be force-clamped below yellow_time, which is
            # what previously broke the cycle-length invariant.
            logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            phases = logic.phases

            middle_greens = phases[2].duration
            all_yellows   = (phases[1].duration
                              + phases[3].duration
                              + phases[5].duration)

            max_split_for_junction = (self.cycle_length
                                       - middle_greens
                                       - all_yellows
                                       - self.yellow_time)  # leave EW >= 1 yellow's worth of green

            effective_ub = min(self.s_ub, max_split_for_junction)
            new_split = float(np.clip(
                self.phase_splits[tl_id] + delta,
                self.s_lb, effective_ub
            ))
        else:
            new_split = float(np.clip(
                self.phase_splits[tl_id] + delta,
                self.s_lb, self.s_ub
            ))

        self.phase_splits[tl_id] = new_split

        if tl_id == self.SIX_PHASE_TL:
            self._apply_action_6phase(tl_id, new_split)
        else:
            self._apply_action_standard(tl_id, new_split)

    def _parse_tripinfo(self):
        try:
            path = f"tripinfo_6int_ep{self.episode_count}.xml"
            if not os.path.exists(path):
                return 0.0, 0.0
            tree = ET.parse(path)
            root = tree.getroot()
            tt, dl = [], []
            for trip in root.findall('tripinfo'):
                d = trip.get('duration')
                t = trip.get('timeLoss')
                if d: tt.append(float(d))
                if t: dl.append(float(t))
            mean_tt  = float(np.mean(tt)) if tt else 0.0
            mean_del = float(np.mean(dl)) if dl else 0.0
            os.remove(path)
            return mean_tt, mean_del
        except Exception:
            return 0.0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass

        self.episode_count += 1
        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            binary, "-c", self.sumocfg,
            "--no-step-log", "true",
            "--tripinfo-output",
            f"tripinfo_6int_ep{self.episode_count}.xml"
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        traci.start(cmd)

        self.current_step     = 0
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0
        self.phase_splits     = {tl: self.init_split for tl in self.tl_ids}

        return self._get_observation(), {}

    def step(self, action):
        tl_id, delta = self._decode_action(action)
        self._apply_action(tl_id, delta)

        self.last_arrived   = 0
        self.last_teleports = 0

        for _ in range(self.step_length):
            if traci.simulation.getMinExpectedNumber() == 0:
                self.current_step = self.sim_steps
                break
            traci.simulationStep()
            self.current_step += 1

        obs        = self._get_observation()
        reward     = self._compute_reward()
        terminated = self.current_step >= self.sim_steps

        try:
            self.last_arrived   = traci.simulation.getArrivedNumber()
            self.last_teleports = traci.simulation.getStartingTeleportNumber()
        except Exception:
            pass

        if terminated:
            try:
                traci.close()
            except Exception:
                pass
            self.mean_travel_time, self.mean_delay = self._parse_tripinfo()

        return obs, reward, terminated, False, {}

    def close(self):
        try:
            traci.close()
        except Exception:
            pass