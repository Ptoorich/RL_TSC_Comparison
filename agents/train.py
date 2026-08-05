import os
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from sumo_env import SumoTSCEnv

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
SUMO_CFG = BASE_DIR / "2int_files" / "2int.sumocfg"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "saved_models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Environment Config ──────────────────────────────────────────────────
config = {
    "sumocfg": str(SUMO_CFG),
    "tl_ids": ["30406197", "cluster_30406198_7161921427_7161921428"],
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
    "use_gui": False,  # Headless mode for fast execution
    "sim_steps": 3600,  # 1-hour simulation per episode
    "step_length": 10,  # Action decision every 10s -> 360 steps per episode
}

# ── Initialize & Wrap Environment ───────────────────────────────────────
env = SumoTSCEnv(config)
env = Monitor(env, filename=str(LOG_DIR / "train_monitor.csv"))

# ── Callbacks ───────────────────────────────────────────────────────────
# Save a model checkpoint every 5,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=str(MODEL_DIR / "checkpoints"),
    name_prefix="ppo_sumo",
)

# ── Hyperparameters & Model Setup ───────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=360,  # Collect 1 full episode before updating policy
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,  # Small entropy coefficient to encourage exploration
    verbose=1,
    tensorboard_log=str(LOG_DIR / "tensorboard"),
)

# ── Train Agent ─────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 36_000  # ~100 full episodes (increase for better performance)
print(f"Starting training for {TOTAL_TIMESTEPS} steps...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True,
)

# Save final model
final_model_path = MODEL_DIR / "ppo_sumo_final"
model.save(str(final_model_path))
print(f"Model saved to: {final_model_path}.zip")

env.close()