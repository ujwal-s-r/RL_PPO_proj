"""Full PPO training orchestration loop with parallel environment rollouts and evaluation."""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import os
import time
import numpy as np
import torch

from rl.config import PPOConfig
from rl.ppo import PPO
from rl.buffer import RolloutBuffer
from rl.vector_env import SyncVectorEnv
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment


class PPOTrainer:
    """Orchestrates end-to-end PPO training with parallel environments and GPU batching."""

    def __init__(self, config: PPOConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)

        # Parallel environment vectorizer
        self.vec_env = SyncVectorEnv(
            num_envs=config.num_envs,
            cluster_config_path=config.cluster_config,
            reward_config_path=config.reward_config,
            scenario_name=config.scenario,
            base_seed=config.seed,
        )

        self.obs_dim = self.vec_env.obs_dim
        self.action_dim = self.vec_env.action_dim

        # PPO Agent & Rollout Buffer
        self.agent = PPO(config, self.obs_dim, self.action_dim)
        self.buffer = RolloutBuffer(
            rollout_length=config.rollout_length,
            num_envs=config.num_envs,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # Evaluation environment
        self.eval_env = GPUSchedulerEnvironment(
            cluster_config_path=config.cluster_config,
            reward_config_path=config.reward_config,
            scenario_name=config.scenario,
        )

    def train(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, List[Any]]:
        """
        Execute full PPO training loop.
        
        Returns:
            Dictionary containing logged training telemetry histories.
        """
        cfg = self.config
        total_steps = 0
        update_count = 0
        start_time = time.time()

        history: Dict[str, List[Any]] = {
            "timesteps": [],
            "mean_reward": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "eval_reward": [],
        }

        # Initialize vector environment
        obs, masks = self.vec_env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        masks_tensor = torch.tensor(masks, dtype=torch.float32, device=self.device)

        episode_rewards: List[float] = []
        current_ep_rewards = np.zeros(cfg.num_envs)

        print(f"Starting PPO Training on [{cfg.device.upper()}] with {cfg.num_envs} Parallel Envs...")
        print(f"Total target timesteps: {cfg.total_timesteps:,} (Batch size per update: {cfg.num_envs * cfg.rollout_length})\n")

        while total_steps < cfg.total_timesteps:
            # 1. Collect Rollout across all parallel environments
            for _ in range(cfg.rollout_length):
                with torch.no_grad():
                    actions, log_probs, values = self.agent.actor_critic.get_action(
                        obs=obs_tensor,
                        mask=masks_tensor,
                    )

                # Step environments on CPU
                next_obs, rewards, dones, next_masks, infos = self.vec_env.step(actions.cpu().numpy())

                current_ep_rewards += rewards
                for i, d in enumerate(dones):
                    if d:
                        episode_rewards.append(float(current_ep_rewards[i]))
                        current_ep_rewards[i] = 0.0

                # Convert to GPU tensors
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

                # Store in buffer
                self.buffer.add(
                    obs=obs_tensor,
                    action=actions,
                    reward=rewards_tensor,
                    done=dones_tensor,
                    value=values,
                    log_prob=log_probs,
                    mask=masks_tensor,
                )

                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
                masks_tensor = torch.tensor(next_masks, dtype=torch.float32, device=self.device)

                total_steps += cfg.num_envs

            # 2. Estimate value for final state and compute GAE
            with torch.no_grad():
                last_values = self.agent.actor_critic.get_value(obs_tensor)
                last_dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
                self.buffer.compute_gae(last_values, last_dones, cfg.gamma, cfg.gae_lambda)

            # 3. PPO Gradient Optimization Step
            loss_dict = self.agent.update(self.buffer)
            update_count += 1

            # Log Telemetry
            recent_reward = float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0
            history["timesteps"].append(total_steps)
            history["mean_reward"].append(recent_reward)
            history["policy_loss"].append(loss_dict["policy_loss"])
            history["value_loss"].append(loss_dict["value_loss"])
            history["entropy"].append(loss_dict["entropy"])
            history["approx_kl"].append(loss_dict["approx_kl"])

            fps = int(total_steps / max(1.0, time.time() - start_time))
            if update_count % 5 == 0 or total_steps >= cfg.total_timesteps:
                print(
                    f"Step {total_steps:06d}/{cfg.total_timesteps} | "
                    f"FPS: {fps} | "
                    f"Mean Rew: {recent_reward:+7.2f} | "
                    f"Loss(P): {loss_dict['policy_loss']:.4f} | "
                    f"Loss(V): {loss_dict['value_loss']:.4f} | "
                    f"Entropy: {loss_dict['entropy']:.3f} | "
                    f"KL: {loss_dict['approx_kl']:.4f}"
                )

            # 4. Periodic Checkpoint Saving
            if total_steps % cfg.save_freq_steps < (cfg.num_envs * cfg.rollout_length):
                ckpt_path = os.path.join(cfg.checkpoint_dir, f"ppo_step_{total_steps}.pt")
                self.agent.save_checkpoint(ckpt_path, {"total_steps": total_steps, "mean_reward": recent_reward})

            if progress_callback:
                progress_callback({"step": total_steps, "metrics": loss_dict, "mean_reward": recent_reward})

        # Save Final Checkpoint
        final_ckpt = os.path.join(cfg.checkpoint_dir, "ppo_final.pt")
        self.agent.save_checkpoint(final_ckpt, {"total_steps": total_steps, "mean_reward": recent_reward})
        print(f"\nTraining Complete! Saved final model to '{final_ckpt}'")

        return history
