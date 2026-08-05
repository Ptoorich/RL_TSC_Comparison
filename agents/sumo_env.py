import gymnasium as gym
import numpy as np
import traci

class SumoTSCEnv(gym.Env):
    """
    Custom Gymnasium environment for Traffic Signal Control.
    MDP formulation based on Li and Zhuang (2026).

    State:  normalised queue length per approach lane 
            + normalised phase split per intersection
    Action: select intersection + adjust phase split by -Δs, 0, or +Δs
    Reward: tiered penalty based on queue length (Li & Zhuang Equation 3)
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

        # ── Phase split parameters (calibrated to your real network) ───
        self.delta_s      = config.get("delta_s", 5)
        self.s_lb         = config.get("s_lb", 15)
        self.s_ub         = config.get("s_ub", 63)
        self.cycle_length = config.get("cycle_length", 90)
        self.yellow_time  = config.get("yellow_time", 6)
        self.allred_time  = config.get("allred_time", 0)
        self.init_split   = config.get("init_split", 39)

        # ── Reward thresholds ──────────────────────────────────────────
        self.q_lc  = config.get("q_lc", 5)
        self.q_hc  = config.get("q_hc", 15)
        self.w_l   = config.get("w_l", 1.0)
        self.w_cp  = config.get("w_cp", 3.0)

        # ── Episode parameters ─────────────────────────────────────────
        self.sim_steps    = config.get("sim_steps", 3600)
        self.step_length  = config.get("step_length", 90)
        self.current_step = 0

        # ── Internal phase split tracker ───────────────────────────────
        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        # ── Gymnasium spaces ───────────────────────────────────────────
        obs_size = self.num_links + self.num_tls
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        # 3 actions per intersection: 0=decrease, 1=hold, 2=increase
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
        return [traci.lane.getLastStepHaltingNumber(lane)
                for lane in self.link_ids]

    # ── Build normalised observation vector ────────────────────────────
    def _get_observation(self):
        queues = self._get_queue_lengths()
        splits = [self.phase_splits[tl] for tl in self.tl_ids]

        q_max  = 50.0
        norm_q = [min(q / q_max, 1.0) for q in queues]
        norm_s = [(s - self.s_lb) / (self.s_ub - self.s_lb)
                  for s in splits]

        return np.array(norm_q + norm_s, dtype=np.float32)

    # ── Li & Zhuang tiered reward (Equation 3) ─────────────────────────
    def _compute_reward(self):
        total = 0.0
        for q in self._get_queue_lengths():
            if q <= self.q_lc:
                total += 0.0
            elif q <= self.q_hc:
                total += -(self.w_l * q)
            else:
                total += -(self.w_cp * self.w_l * q)
        return total

    # ── Apply phase split change to SUMO using setProgramLogic ─────────
    def _apply_action(self, tl_id, delta):
        # Update internal split with bounds check
        new_split = np.clip(
            self.phase_splits[tl_id] + delta,
            self.s_lb, self.s_ub
        )
        self.phase_splits[tl_id] = float(new_split)

        # Complementary green time for the other direction
        ew_green = (self.cycle_length
                    - new_split
                    - 2 * self.yellow_time
                    - self.allred_time)

        # Get existing program logic to preserve phase states
        logic  = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = logic.phases

        # Rebuild phases with updated green durations only
        # Phase 0 = NS green, Phase 1 = NS yellow (fixed)
        # Phase 2 = EW green, Phase 3 = EW yellow (fixed)
        new_phases = [
            traci.trafficlight.Phase(new_split, phases[0].state),
            phases[1],                 # yellow — unchanged
            traci.trafficlight.Phase(ew_green, phases[2].state),
            phases[3],                 # yellow — unchanged
        ]

        new_logic = traci.trafficlight.Logic(
            logic.programID,
            logic.type,
            logic.currentPhaseIndex,
            new_phases
        )
        traci.trafficlight.setProgramLogic(tl_id, new_logic)

    # ── Reset: start a new episode ─────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except Exception:
            pass

        binary  = "sumo-gui" if self.use_gui else "sumo"
        cmd     = [binary, "-c", self.sumocfg, "--no-step-log", "true"]
        traci.start(cmd)

        self.current_step = 0
        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        return self._get_observation(), {}

    # ── Step: one decision cycle ───────────────────────────────────────
    def step(self, action):
        tl_id, delta = self._decode_action(action)
        self._apply_action(tl_id, delta)

        for _ in range(self.step_length):
            traci.simulationStep()
            self.current_step += 1

        obs    = self._get_observation()
        reward = self._compute_reward()
        terminated = self.current_step >= self.sim_steps
        return obs, reward, terminated, False, {}

    # ── Close SUMO ─────────────────────────────────────────────────────
    def close(self):
        try:
            traci.close()
        except Exception:
            pass
        