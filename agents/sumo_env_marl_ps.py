import gymnasium as gym
import numpy as np
import traci
import xml.etree.ElementTree as ET
import os


class SumoMARLEnvPS:
    """
    Multi-Agent Traffic Signal Control Environment with Parameter Sharing.

    Key differences from SumoMARLEnv (independent learners):
    - Each agent's observation is augmented with a normalised intersection
      index (agent_index / num_agents) appended as the final element.
    - This allows a SINGLE shared PPO model to distinguish which
      intersection is acting while keeping observation space size
      constant regardless of network size (local_obs_size + 1).
    - Normalised index scales naturally to 4 and 6 intersection networks.

    Observation per agent: [local_queue_lanes..., phase_split, norm_id]
    All padded to uniform size: max_local_obs + 1
    """

    def __init__(self, config):

        # ── Network config ─────────────────────────────────────────────
        self.sumocfg   = config["sumocfg"]
        self.use_gui   = config.get("use_gui", False)
        self.agent_ids = config["agent_ids"]
        self.tl_map    = config["tl_map"]
        self.lane_map  = config["lane_map"]
        self.num_agents= len(self.agent_ids)

        # ── Normalised intersection index per agent ─────────────────────
        # agent_index / num_agents — unique, fixed size, scales to any N
        self.agent_index = {
            agent: i / self.num_agents
            for i, agent in enumerate(self.agent_ids)
        }

        # ── Observation size ───────────────────────────────────────────
        # local obs = lane queues + phase split
        # pad all to max local obs size, then append norm ID
        self.local_obs_sizes = {
            agent: len(self.lane_map[agent]) + 1
            for agent in self.agent_ids
        }
        self.max_local_obs = max(self.local_obs_sizes.values())
        self.obs_size      = self.max_local_obs + 1  # +1 for norm ID

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

        # ── Internal state ─────────────────────────────────────────────
        self.phase_splits = {
            agent: self.init_split for agent in self.agent_ids
        }
        self.last_arrived     = 0
        self.last_teleports   = 0
        self.mean_travel_time = 0.0
        self.mean_delay       = 0.0

        # ── Gymnasium spaces ───────────────────────────────────────────
        # Same for all agents — shared model requires uniform obs size
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)

        print(f"MARL-PS Environment initialised:")
        print(f"  Agents:            {self.agent_ids}")
        print(f"  Max local obs:     {self.max_local_obs}")
        print(f"  Obs size (+ ID):   {self.obs_size}")
        print(f"  Action space:      Discrete(3) per agent")
        print(f"  Normalised IDs:    {self.agent_index}")
        print(f"  All lanes:         {len(self.all_lanes)}")

    # ── Get observation for one agent (with normalised ID) ─────────────
    def _get_agent_obs(self, agent):
        try:
            lanes  = self.lane_map[agent]
            queues = [traci.lane.getLastStepHaltingNumber(l) for l in lanes]
            split  = self.phase_splits[agent]

            q_max  = 50.0
            norm_q = [min(q / q_max, 1.0) for q in queues]
            norm_s = [(split - self.s_lb) / (self.s_ub - self.s_lb)]

            # Local obs padded to max_local_obs
            local_obs = np.array(norm_q + norm_s, dtype=np.float32)
            if len(local_obs) < self.max_local_obs:
                local_obs = np.pad(local_obs,
                                   (0, self.max_local_obs - len(local_obs)))

            # Append normalised intersection ID
            norm_id = np.array([self.agent_index[agent]], dtype=np.float32)
            return np.concatenate([local_obs, norm_id])

        except Exception:
            return np.zeros(self.obs_size, dtype=np.float32)

    # ── Get observations for all agents ────────────────────────────────
    def _get_observations(self):
        return {agent: self._get_agent_obs(agent)
                for agent in self.agent_ids}

    # ── Global reward — network-wide queue penalty ─────────────────────
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
        delta     = delta_map[int(action_idx)]

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
            logic.programID, logic.type,
            logic.currentPhaseIndex, new_phases
        )
        traci.trafficlight.setProgramLogic(tl_id, new_logic)

    # ── Parse tripinfo ──────────────────────────────────────────────────
    def _parse_tripinfo(self):
        try:
            path = f"tripinfo_marl_ps_ep{self.episode_count}.xml"
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

    # ── Reset ──────────────────────────────────────────────────────────
    def reset(self, seed=None):
        try:
            traci.close()
        except Exception:
            pass

        self.episode_count += 1
        
        # Absolute binary path set to C:\RL_TSC\bin
        binary = r"C:\RL_TSC\bin\sumo-gui.exe" if self.use_gui else r"C:\RL_TSC\bin\sumo.exe"
        
        cmd = [
            binary, "-c", self.sumocfg,
            "--no-step-log", "true",
            "--tripinfo-output",
            f"tripinfo_marl_ps_ep{self.episode_count}.xml"
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
        for agent, action_idx in actions.items():
            self._apply_action(agent, action_idx)

        for _ in range(self.step_length):
            if traci.simulation.getMinExpectedNumber() == 0:
                self.current_step = self.sim_steps
                break
            traci.simulationStep()
            self.current_step += 1

        reward     = self._compute_reward()
        terminated = self.current_step >= self.sim_steps

        try:
            self.last_arrived   = traci.simulation.getArrivedNumber()
            self.last_teleports = traci.simulation.getStartingTeleportNumber()
        except Exception:
            pass

        observations = self._get_observations()
        rewards      = {agent: reward for agent in self.agent_ids}
        terminateds  = {agent: terminated for agent in self.agent_ids}

        if terminated:
            try:
                traci.close()
            except Exception:
                pass
            self.mean_travel_time, self.mean_delay = self._parse_tripinfo()

        return observations, rewards, terminateds

    # ── Close ──────────────────────────────────────────────────────────
    def close(self):
        try:
            traci.close()
        except Exception:
            pass