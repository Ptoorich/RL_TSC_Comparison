import gymnasium as gym
import numpy as np
import traci
import xml.etree.ElementTree as ET
import os

class SumoTSCEnv(gym.Env):
    """
    Custom Gymnasium environment for Traffic Signal Control.
    MDP formulation based on Li and Zhuang (2026).

    State:  normalised queue length per approach lane
            + normalised phase split per intersection
    Action: select intersection + adjust phase split by -Δs, 0, or +Δs
    Reward: tiered queue penalty + optional throughput incentive
    """

    def __init__(self, config):
        super().__init__()

        # ── Network config ─────────────────────────────────────────────
        self.sumocfg   = config["sumocfg"]
        self.tl_ids    = config["tl_ids"]
        self.link_ids  = config["link_ids"]
        self.num_tls   = len(self.tl_ids)
        self.num_links = len(self.link_ids)
        self.use_gui   = config.get("use_gui", False)

        # ── Phase split parameters ─────────────────────────────────────
        self.delta_s      = config.get("delta_s", 5)
        self.s_lb         = config.get("s_lb", 15)
        self.s_ub         = config.get("s_ub", 63)
        self.cycle_length = config.get("cycle_length", 90)
        self.yellow_time  = config.get("yellow_time", 6)
        self.allred_time  = config.get("allred_time", 0)
        self.init_split   = config.get("init_split", 39)

        # ── Reward parameters ──────────────────────────────────────────
        self.q_lc         = config.get("q_lc", 5)
        self.q_hc         = config.get("q_hc", 15)
        self.w_l          = config.get("w_l", 1.0)
        self.w_cp         = config.get("w_cp", 3.0)
        self.w_throughput = config.get("w_throughput", 0.0)

        # ── Episode parameters ─────────────────────────────────────────
        self.sim_steps     = config.get("sim_steps", 3600)
        self.step_length   = config.get("step_length", 90)
        self.current_step  = 0
        self.episode_count = 0

        # ── Internal phase split tracker ───────────────────────────────
        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        # ── Step-level metric storage ──────────────────────────────────
        # Read from env instance rather than TraCI directly
        # to avoid connection-closed errors in evaluation scripts
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0

        # ── Gymnasium spaces ───────────────────────────────────────────
        obs_size = self.num_links + self.num_tls
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3 * self.num_tls)

    # ── Decode flat action into (tl_id, delta) ─────────────────────────
    def _decode_action(self, action):
        tl_idx    = action // 3
        adj_idx   = action % 3
        tl_id     = self.tl_ids[tl_idx]
        delta_map = {0: -self.delta_s, 1: 0, 2: self.delta_s}
        return tl_id, delta_map[adj_idx]

    # ── Get queue lengths from SUMO ────────────────────────────────────
    def _get_queue_lengths(self):
        try:
            return [traci.lane.getLastStepHaltingNumber(lane)
                    for lane in self.link_ids]
        except Exception:
            return [0] * self.num_links

    # ── Build normalised observation vector ────────────────────────────
    def _get_observation(self):
        queues = self._get_queue_lengths()
        splits = [self.phase_splits[tl] for tl in self.tl_ids]

        q_max  = 50.0
        norm_q = [min(q / q_max, 1.0) for q in queues]
        norm_s = [(s - self.s_lb) / (self.s_ub - self.s_lb)
                  for s in splits]

        return np.array(norm_q + norm_s, dtype=np.float32)

    # ── Reward: queue penalty + optional throughput incentive ───────────
    def _compute_reward(self):
        try:
            total = 0.0

            # Li & Zhuang tiered queue penalty (Equation 3)
            for q in self._get_queue_lengths():
                if q <= self.q_lc:
                    total += 0.0
                elif q <= self.q_hc:
                    total += -(self.w_l * q)
                else:
                    total += -(self.w_cp * self.w_l * q)

            # Throughput incentive (0.0 = disabled, used in DQN v2)
            arrived = traci.simulation.getArrivedNumber()
            total  += self.w_throughput * arrived

            # Teleport penalty
            teleports = traci.simulation.getStartingTeleportNumber()
            total    += -(5.0 * teleports)

            return total
        except Exception:
            return 0.0

    # ── Parse tripinfo XML for mean travel time and delay ───────────────
    def _parse_tripinfo(self):
        try:
            path = f"tripinfo_ep{self.episode_count}.xml"
            if not os.path.exists(path):
                return 0.0, 0.0

            tree = ET.parse(path)
            root = tree.getroot()

            travel_times = []
            delays       = []

            for trip in root.findall('tripinfo'):
                duration = trip.get('duration')
                timeloss = trip.get('timeLoss')
                if duration is not None:
                    travel_times.append(float(duration))
                if timeloss is not None:
                    delays.append(float(timeloss))

            mean_tt    = float(np.mean(travel_times)) if travel_times else 0.0
            mean_delay = float(np.mean(delays))       if delays       else 0.0

            # Clean up file for next episode
            os.remove(path)

            return mean_tt, mean_delay

        except Exception:
            return 0.0, 0.0

    # ── Apply phase split change to SUMO ────────────────────────────────
    def _apply_action(self, tl_id, delta):
        new_split = np.clip(
            self.phase_splits[tl_id] + delta,
            self.s_lb, self.s_ub
        )
        self.phase_splits[tl_id] = float(new_split)

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
            logic.programID,
            logic.type,
            logic.currentPhaseIndex,
            new_phases
        )
        traci.trafficlight.setProgramLogic(tl_id, new_logic)

    # ── Reset: start a new episode ──────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except Exception:
            pass

        self.episode_count += 1

        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd    = [
            binary, "-c", self.sumocfg,
            "--no-step-log", "true",
            "--tripinfo-output", f"tripinfo_ep{self.episode_count}.xml"
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        traci.start(cmd)

        # Reset all trackers
        self.current_step     = 0
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0
        self.phase_splits     = {tl: self.init_split for tl in self.tl_ids}

        return self._get_observation(), {}

    # ── Step: one decision cycle ───────────────────────────────────────
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

        # Capture step metrics before potentially closing connection
        try:
            self.last_arrived   = traci.simulation.getArrivedNumber()
            self.last_teleports = traci.simulation.getStartingTeleportNumber()
        except Exception:
            pass

        # Parse tripinfo at episode end for travel time and delay
        if terminated:
            try:
                traci.close()
            except Exception:
                pass
            self.mean_travel_time, self.mean_delay = self._parse_tripinfo()

        return obs, reward, terminated, False, {}

    # ── Close SUMO ─────────────────────────────────────────────────────
    def close(self):
        try:
            traci.close()
        except Exception:
            pass