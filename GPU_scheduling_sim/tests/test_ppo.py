"""Unit tests for Actor-Critic, Rollout Buffer, Action Masking, and PPO updates."""

import os
import torch
import numpy as np

from rl.config import PPOConfig
from rl.actor_critic import ActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo import PPO


def test_actor_critic_action_masking():
    obs_dim = 16
    action_dim = 4
    net = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn((1, obs_dim))
    
    # Mask where only action index 2 is feasible (1.0)
    mask = torch.tensor([[0.0, 0.0, 1.0, 0.0]], dtype=torch.float32)

    # In deterministic mode, must pick action 2
    action, log_prob, value = net.get_action(obs, mask=mask, deterministic=True)
    assert action.item() == 2

    # In sampling mode, must also pick action 2 100% of the time
    for _ in range(5):
        action, log_prob, value = net.get_action(obs, mask=mask, deterministic=False)
        assert action.item() == 2


def test_rollout_buffer_gae():
    device = torch.device("cpu")
    buffer = RolloutBuffer(rollout_length=4, num_envs=2, obs_dim=8, action_dim=4, device=device)

    for _ in range(4):
        buffer.add(
            obs=torch.zeros((2, 8)),
            action=torch.zeros(2, dtype=torch.long),
            reward=torch.ones(2),
            done=torch.zeros(2),
            value=torch.zeros(2),
            log_prob=torch.zeros(2),
            mask=torch.ones((2, 4)),
        )

    last_value = torch.zeros(2)
    last_done = torch.zeros(2)
    buffer.compute_gae(last_value, last_done, gamma=0.99, gae_lambda=0.95)

    assert not torch.isnan(buffer.advantages).any()
    assert not torch.isnan(buffer.returns).any()
    assert buffer.advantages.shape == (4, 2)


def test_checkpoint_save_and_load(tmp_path_str: str = "checkpoints/test_model.pt"):
    cfg = PPOConfig(hidden_dim=64, device="cpu")
    ppo = PPO(cfg, obs_dim=16, action_dim=4)

    # Save
    ppo.save_checkpoint(tmp_path_str, {"epoch": 1, "test_val": 42})
    assert os.path.exists(tmp_path_str)

    # Load into new agent
    ppo2 = PPO(cfg, obs_dim=16, action_dim=4)
    meta = ppo2.load_checkpoint(tmp_path_str)
    assert meta["epoch"] == 1
    assert meta["test_val"] == 42

    # Clean up test file
    if os.path.exists(tmp_path_str):
        os.remove(tmp_path_str)


def test_ppo_single_update():
    cfg = PPOConfig(
        hidden_dim=32,
        rollout_length=8,
        num_envs=2,
        minibatch_size=4,
        epochs_per_update=2,
        device="cpu",
    )
    obs_dim = 10
    action_dim = 4

    ppo = PPO(cfg, obs_dim, action_dim)
    buffer = RolloutBuffer(rollout_length=8, num_envs=2, obs_dim=obs_dim, action_dim=action_dim, device=torch.device("cpu"))

    for _ in range(8):
        buffer.add(
            obs=torch.randn((2, obs_dim)),
            action=torch.randint(0, action_dim, (2,)),
            reward=torch.ones(2),
            done=torch.zeros(2),
            value=torch.randn(2),
            log_prob=torch.zeros(2),
            mask=torch.ones((2, action_dim)),
        )

    buffer.compute_gae(torch.zeros(2), torch.zeros(2))
    loss_metrics = ppo.update(buffer)

    assert "policy_loss" in loss_metrics
    assert "value_loss" in loss_metrics
    assert "entropy" in loss_metrics
    assert not np.isnan(loss_metrics["policy_loss"])
