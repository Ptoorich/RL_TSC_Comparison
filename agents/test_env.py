from pathlib import Path
from sumo_env import SumoTSCEnv

# Calculate absolute path to sumocfg relative to this test file
BASE_DIR = Path(__file__).parent.resolve()
SUMO_CFG = BASE_DIR / "2int_files" / "2int.sumocfg"

config = {
    "sumocfg": str(SUMO_CFG),
    # Traffic Light IDs extracted from Jorissen Street network
    "tl_ids": ["30406197", "cluster_30406198_7161921427_7161921428"],
    # Approach Lane IDs (excluding internal junction lanes)
    "link_ids": [
        "-1190301745#0_0",
        "-621271001_0",
        "-669676574#1_0",
        "-669676574#8_0",
        "1190301745#0_0",
        "4761588#1_0",
        "4761588#1_1",
        "4761588#1_2",
        "4761588#1_3",
        "4761588#2_0",
        "4761588#2_1",
        "4761588#2_2",
        "4761588#2_3",
        "596638461#0_0",
        "596638461#1_0",
        "621271001_0",
        "622835372_0",
        "622835372_1",
        "622835372_2",
        "669676574#0_0",
        "669676574#3_0",
    ],
    "use_gui": True,  # Set to False if you want fast headless execution
    "sim_steps": 200,  # Test episode length
    "step_length": 10,
}

env = SumoTSCEnv(config)

obs, _ = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial observation vector: {obs}")
print(f"Observation space:        {env.observation_space}")
print(f"Action space:             {env.action_space}")

# Run 10 test steps with random actions
for step in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    print(f"Step {step+1}: action={action}, reward={reward:.2f}")
    if terminated:
        break

env.close()
print("Environment test completed successfully.")