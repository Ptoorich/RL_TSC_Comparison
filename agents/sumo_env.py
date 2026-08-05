from pathlib import Path
import gymnasium as gym
import numpy as np
import traci


class SumoTSCEnv(gym.Env):
    """Custom Gymnasium environment for Traffic Signal Control.

    MDP formulation based on Li and Zhuang (2026).

    State:  queue length per link + phase split per intersection (normalised)
    Action: select intersection + adjust phase split by -Δs, 0, or +Δs
    Reward: tiered penalty based on queue length congestion thresholds
    """

    def __init__(self, config):
        super().__init__()

        # ── Network configuration ──────────────────────────────────────
        self.sumocfg = config["sumocfg"]  # path to .sumocfg
        self.tl_ids = config["tl_ids"]  # list of traffic light IDs
        self.link_ids = config["link_ids"]  # list of lane IDs to monitor
        self.num_tls = len(self.tl_ids)  # M intersections
        self.num_links = len(self.link_ids)  # L links
        self.use_gui = config.get("use_gui", False)

        # ── Phase split parameters (Li & Zhuang fixed values) ──────────
        self.delta_s = config.get("delta_s", 5)  # Δs adjustment step (s)
        self.s_lb = config.get("s_lb", 20)  # lower bound phase split
        self.s_ub = config.get("s_ub", 70)  # upper bound phase split
        self.cycle_length = config.get("cycle_length", 100)
        self.yellow_time = config.get("yellow_time", 2)
        self.allred_time = config.get("allred_time", 2)
        self.init_split = config.get("init_split", 50)  # initial phase split

        # ── Reward thresholds (Li & Zhuang) ────────────────────────────
        self.q_lc = config.get("q_lc", 5)  # light congestion (vehicles)
        self.q_hc = config.get("q_hc", 15)  # heavy congestion (vehicles)
        self.w_l = config.get("w_l", 1.0)  # link importance weight
        self.w_cp = config.get("w_cp", 3.0)  # heavy congestion multiplier

        # ── Episode parameters ─────────────────────────────────────────
        self.sim_steps = config.get("sim_steps", 3600)  # 1 hour
        self.step_length = config.get("step_length", 10)  # decision every 10s
        self.current_step = 0

        # ── Track current phase splits per intersection ─────────────────
        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        # ── Gymnasium spaces ───────────────────────────────────────────
        # Observation: [queue_l1, ..., queue_lL, split_m1, ..., split_mM]
        obs_size = self.num_links + self.num_tls
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action: 3 choices per intersection (decrease, hold, increase)
        self.action_space = gym.spaces.Discrete(3 * self.num_tls)

    # ── Utility: decode flat action index into (intersection, adjustment)
    def _decode_action(self, action):
        tl_idx = action // 3
        adj_idx = action % 3
        tl_id = self.tl_ids[tl_idx]
        delta_map = {0: -self.delta_s, 1: 0, 2: self.delta_s}
        return tl_id, delta_map[adj_idx]

    # ── Utility: get queue lengths from SUMO via TraCI ──────────────────
    def _get_queue_lengths(self):
        queues = []
        for lane in self.link_ids:
            q = traci.lane.getLastStepHaltingNumber(lane)
            queues.append(q)
        return queues

    # ── Utility: get current phase splits ───────────────────────────────
    def _get_phase_splits(self):
        return [self.phase_splits[tl] for tl in self.tl_ids]

    # ── Utility: build normalised observation vector ─────────────────────
    def _get_observation(self):
        queues = self._get_queue_lengths()
        splits = self._get_phase_splits()

        q_max = 50.0  # max vehicles expected per lane
        norm_q = [min(q / q_max, 1.0) for q in queues]
        norm_s = [(s - self.s_lb) / (self.s_ub - self.s_lb) for s in splits]

        return np.array(norm_q + norm_s, dtype=np.float32)

    # ── Utility: tiered reward function ──────────────────────────────────
    def _compute_reward(self):
        queues = self._get_queue_lengths()
        total_reward = 0.0

        for q in queues:
            if q <= self.q_lc:
                r = 0.0
            elif q <= self.q_hc:
                r = -(self.w_l * q)
            else:
                r = -(self.w_cp * self.w_l * q)
            total_reward += r

        return total_reward

    # ── Apply action: adjust phase split in SUMO ────────────────────────
    def _apply_action(self, tl_id, delta):
        new_split = self.phase_splits[tl_id] + delta
        new_split = np.clip(new_split, self.s_lb, self.s_ub)
        self.phase_splits[tl_id] = new_split
        traci.trafficlight.setPhaseDuration(tl_id, float(new_split))

    # ── Reset: start a new episode ───────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except Exception:
            pass

        # Resolve config path to absolute string for cross-platform compatibility
        cfg_path = str(Path(self.sumocfg).resolve()).replace("\\", "/")

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [sumo_binary, "-c", cfg_path, "--no-step-log", "true"]
        traci.start(sumo_cmd)

        self.current_step = 0
        self.phase_splits = {tl: self.init_split for tl in self.tl_ids}

        obs = self._get_observation()
        return obs, {}

    # ── Step: advance simulation one decision step ───────────────────────
    def step(self, action):
        tl_id, delta = self._decode_action(action)
        self._apply_action(tl_id, delta)

        for _ in range(self.step_length):
            traci.simulationStep()
            self.current_step += 1

        obs = self._get_observation()
        reward = self._compute_reward()

        terminated = self.current_step >= self.sim_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    # ── Close: shut down SUMO ────────────────────────────────────────────
    def close(self):
        try:
            traci.close()
        except Exception:
            pass