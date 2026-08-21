import gymnasium as gym
import numpy as np
import traci
import xml.etree.ElementTree as ET
import os

class SumoMARLEnv:
    """
    Multi-Agent Traffic Signal Control Environment.
    Two independent PPO agents — one per intersection.
    
    Key differences from SumoTSCEnv (SARL):
    - Returns dict of local observations per agent
    - Accepts dict of actions per agent  
    - Returns same global reward to both agents
    - Observation per agent is local only (own lanes + own phase split)
    - TL1 obs padded to match TL2 obs size
    - Action space per agent is Discrete(3) not Discrete(6)
    
    MDP formulation consistent with Li and Zhuang (2026)
    and Kolat et al. (2023).
    """

    def __init__(self, config):

        # ── Network config ─────────────────────────────────────────────
        self.sumocfg  = config["sumocfg"]
        self.use_gui  = config.get("use_gui", False)

        # Agent IDs
        self.agent_ids = config["agent_ids"]   # e.g. ["tl1", "tl2"]

        # Map each agent to its traffic light ID in SUMO
        self.tl_map   = config["tl_map"]
        # e.g. {"tl1": "30406197",
        #        "tl2": "cluster_30406198_7161921427_7161921428"}

        # Map each agent to its local approach lane IDs
        self.lane_map = config["lane_map"]
        # e.g. {"tl1": ["596638461#0_0", ...],
        #        "tl2": ["-621271001_0", ...]}

        # ── Observation size ───────────────────────────────────────────
        # Pad all agents to the largest local observation size
        # TL1: 5 lanes + 1 split = 6  → padded to 8
        # TL2: 7 lanes + 1 split = 8  → no padding needed
        self.local_obs_sizes = {
            agent: len(self.lane_map[agent]) + 1
            for agent in self.agent_ids
        }
        self.obs_size = max(self.local_obs_sizes.values())

        # ── Phase split parameters ─────────────────────────────────────
        self.delta_s      = config.get("delta_s", 5)
        self.s_lb         = config.get("s_lb", 15)
        self.s_ub         = config.get("s_ub", 63)
        self.cycle_length = config.get("cycle_length", 90)
        self.yellow_time  = config.get("yellow_time", 6)
        self.allred_time  = config.get("allred_time", 0)
        self.init_split   = config.get("init_split", 39)

        # ── Reward parameters ──────────────────────────────────────────
        self.q_lc = config.get("q_lc", 5)
        self.q_hc = config.get("q_hc", 15)
        self.w_l  = config.get("w_l", 1.0)
        self.w_cp = config.get("w_cp", 3.0)

        # All monitored lanes across all agents for global reward
        self.all_lanes = [
            lane
            for agent in self.agent_ids
            for lane in self.lane_map[agent]
        ]

        # ── Episode parameters ─────────────────────────────────────────
        self.sim_steps     = config.get("sim_steps", 3600)
        self.step_length   = config.get("step_length", 90)
        self.current_step  = 0
        self.episode_count = 0

        # ── Internal phase split tracker ───────────────────────────────
        self.phase_splits = {
            agent: self.init_split for agent in self.agent_ids
        }

        # ── Step-level metrics ─────────────────────────────────────────
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0

        # ── Gymnasium spaces (same for all agents after padding) ───────
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        # Each agent controls only its own intersection: 3 actions
        self.action_space = gym.spaces.Discrete(3)

        print(f"MARL Environment initialised:")
        print(f"  Agents:       {self.agent_ids}")
        print(f"  Obs size:     {self.obs_size} (padded)")
        print(f"  Action space: Discrete(3) per agent")
        print(f"  All lanes monitored: {len(self.all_lanes)}")

    # ── Get local observation for one agent ────────────────────────────
    def _get_agent_obs(self, agent):
        try:
            lanes  = self.lane_map[agent]
            queues = [traci.lane.getLastStepHaltingNumber(l)
                      for l in lanes]
            split  = self.phase_splits[agent]

            q_max  = 50.0
            norm_q = [min(q / q_max, 1.0) for q in queues]
            norm_s = [(split - self.s_lb) / (self.s_ub - self.s_lb)]

            obs = np.array(norm_q + norm_s, dtype=np.float32)

            # Pad to uniform obs_size with zeros
            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)))

            return obs
        except Exception:
            return np.zeros(self.obs_size, dtype=np.float32)

    # ── Get observations for all agents ────────────────────────────────
    def _get_observations(self):
        return {agent: self._get_agent_obs(agent)
                for agent in self.agent_ids}

    # ── Global reward — network-wide queue penalty ─────────────────────
    # Identical to SARL reward function for fair comparison
    def _compute_reward(self):
        try:
            total = 0.0
            for lane in self.all_lanes:
                q = traci.lane.getLastStepHaltingNumber(lane)
                if q <= self.q_lc:
                    total += 0.0
                elif q <= self.q_hc:
                    total += -(self.w_l * q)
                else:
                    total += -(self.w_cp * self.w_l * q)

            teleports = traci.simulation.getStartingTeleportNumber()
            total    += -(5.0 * teleports)
            return total
        except Exception:
            return 0.0

    # ── Apply action for one agent ─────────────────────────────────────
    def _apply_action(self, agent, action_idx):
        tl_id     = self.tl_map[agent]
        delta_map = {0: -self.delta_s, 1: 0, 2: self.delta_s}
        delta     = delta_map[action_idx]

        new_split = float(np.clip(
            self.phase_splits[agent] + delta,
            self.s_lb, self.s_ub
        ))
        self.phase_splits[agent] = new_split

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

    # ── Parse tripinfo for travel time and delay ───────────────────────
    def _parse_tripinfo(self):
        try:
            path = f"tripinfo_marl_ep{self.episode_count}.xml"
            if not os.path.exists(path):
                return 0.0, 0.0

            tree = ET.parse(path)
            root = tree.getroot()

            travel_times = []
            delays       = []

            for trip in root.findall('tripinfo'):
                d = trip.get('duration')
                t = trip.get('timeLoss')
                if d: travel_times.append(float(d))
                if t: delays.append(float(t))

            mean_tt  = float(np.mean(travel_times)) if travel_times else 0.0
            mean_del = float(np.mean(delays))       if delays       else 0.0

            os.remove(path)
            return mean_tt, mean_del
        except Exception:
            return 0.0, 0.0

    # ── Reset ──────────────────────────────────────────────────────────
    def reset(self, seed=None):
        try:
            traci.close()
        except Exception:
            pass

        self.episode_count += 1

        binary = "sumo-gui" if self.use_gui else "sumo"
        cmd    = [
            binary, "-c", self.sumocfg,
            "--no-step-log", "true",
            "--tripinfo-output",
            f"tripinfo_marl_ep{self.episode_count}.xml"
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        traci.start(cmd)

        self.current_step     = 0
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0
        self.phase_splits     = {
            agent: self.init_split for agent in self.agent_ids
        }

        return self._get_observations()

    # ── Step ───────────────────────────────────────────────────────────
    def step(self, actions):
        """
        actions: dict {agent_id: action_index}
        Both agents act simultaneously before SUMO advances.
        """
        # Apply all agents' actions before advancing simulation
        for agent, action_idx in actions.items():
            self._apply_action(agent, int(action_idx))

        # Advance simulation by step_length seconds
        for _ in range(self.step_length):
            if traci.simulation.getMinExpectedNumber() == 0:
                self.current_step = self.sim_steps
                break
            traci.simulationStep()
            self.current_step += 1

        # Collect shared global reward
        reward     = self._compute_reward()
        terminated = self.current_step >= self.sim_steps

        # Capture step metrics
        try:
            self.last_arrived   = traci.simulation.getArrivedNumber()
            self.last_teleports = traci.simulation.getStartingTeleportNumber()
        except Exception:
            pass

        # Get new observations for all agents
        observations = self._get_observations()

        # Parse tripinfo at episode end
        if terminated:
            try:
                traci.close()
            except Exception:
                pass
            self.mean_travel_time, self.mean_delay = self._parse_tripinfo()

        # Return same reward to all agents
        rewards     = {agent: reward for agent in self.agent_ids}
        terminateds = {agent: terminated for agent in self.agent_ids}

        return observations, rewards, terminateds

    # ── Close ──────────────────────────────────────────────────────────
    def close(self):
        try:
            traci.close()
        except Exception:
            pass