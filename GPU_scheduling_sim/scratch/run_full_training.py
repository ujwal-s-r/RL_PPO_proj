import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.getcwd())

from rl.config import PPOConfig
from rl.trainer import PPOTrainer

cfg = PPOConfig(
    total_timesteps=500000,
    num_envs=8,
    rollout_length=128,
    minibatch_size=64,
    epochs_per_update=4,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    eval_freq_steps=25600,
    eval_episodes=5,
    checkpoint_dir="checkpoints",
)

print("Starting Full 500k-Step PPO Training with Hierarchical Factorized Architecture on CUDA...", flush=True)
trainer = PPOTrainer(cfg)
history = trainer.train()
print("\n[SUCCESS] Full PPO Training Complete!", flush=True)
