"""Unit tests for Actor-Critic, Rollout Buffer, Action Masking, and PPO updates."""

import os
import torch
import numpy as np

from rl.config import PPOConfig
from rl.actor_critic import ActorCritic
from rl.buffer import RolloutBuffer
from rl.ppo import PPO


def test_actor_critic_action_masking():
    obs_dim = 215
    action_dim = 128
    net = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn((1, obs_dim))
    
    # Mask where only action index 2 is feasible (1.0)
    mask = torch.zeros((1, action_dim), dtype=torch.float32)
    mask[0, 2] = 1.0

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
    cfg = PPOConfig(hidden_dim=64, device="cpu", max_nodes=8, max_queue_size=16)
    ppo = PPO(cfg, obs_dim=215, action_dim=128)

    # Save
    ppo.save_checkpoint(tmp_path_str, {"epoch": 1, "test_val": 42})
    assert os.path.exists(tmp_path_str)

    # Load into new agent
    ppo2 = PPO(cfg, obs_dim=215, action_dim=128)
    meta = ppo2.load_checkpoint(tmp_path_str)
    assert meta["epoch"] == 1
    assert meta["test_val"] == 42

    # Clean up test file
    if os.path.exists(tmp_path_str):
        os.remove(tmp_path_str)


def test_ppo_single_update():
    cfg = PPOConfig(
        hidden_dim=64,
        rollout_length=8,
        num_envs=2,
        minibatch_size=4,
        epochs_per_update=2,
        device="cpu",
        max_nodes=8,
        max_queue_size=16,
    )
    obs_dim = 215
    action_dim = 128

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


def test_gae_terminal_boundary_masking():
    """Verify that when transition t is terminal (done=1.0), next state value is NOT bootstrapped."""
    device = torch.device("cpu")
    buffer = RolloutBuffer(rollout_length=3, num_envs=1, obs_dim=4, action_dim=2, device=device)

    # Step 0: non-terminal, reward 1.0, value 0.5
    buffer.add(obs=torch.zeros((1, 4)), action=torch.zeros(1, dtype=torch.long), reward=torch.tensor([1.0]), done=torch.tensor([0.0]), value=torch.tensor([0.5]), log_prob=torch.zeros(1), mask=torch.ones((1, 2)))
    # Step 1: TERMINAL (done=1.0), reward 2.0, value 0.8
    buffer.add(obs=torch.zeros((1, 4)), action=torch.zeros(1, dtype=torch.long), reward=torch.tensor([2.0]), done=torch.tensor([1.0]), value=torch.tensor([0.8]), log_prob=torch.zeros(1), mask=torch.ones((1, 2)))
    # Step 2: start of next episode, reward 1.0, value 10.0 (high value in new episode)
    buffer.add(obs=torch.zeros((1, 4)), action=torch.zeros(1, dtype=torch.long), reward=torch.tensor([1.0]), done=torch.tensor([0.0]), value=torch.tensor([10.0]), log_prob=torch.zeros(1), mask=torch.ones((1, 2)))

    buffer.compute_gae(last_value=torch.tensor([0.0]), last_done=torch.tensor([0.0]), gamma=0.99, gae_lambda=0.95)

    # For step 1 (terminal): advantage delta must equal reward[1] - value[1] = 2.0 - 0.8 = 1.2 exactly (NO bootstrapping from step 2 value 10.0!)
    assert abs(buffer.advantages[1, 0].item() - 1.2) < 1e-4, f"Terminal transition bootstrapped improperly! Got {buffer.advantages[1, 0].item()}, expected 1.2"


def test_observation_lookahead_parity():
    """Verify that state.next_arrival correctly flows into _extract_observation."""
    from simulator.cluster import Cluster
    from simulator.simulator import Simulator
    from simulator.job import Job, WorkloadType
    from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment

    sim = Simulator(Cluster.from_yaml("configs/cluster_small.yaml"), max_queue_size=16)
    sim.reset()
    j1 = Job(job_id=1, arrival_time=0.0, gpu_count=2, vram_per_gpu_gb=24.0, estimated_runtime=60.0, actual_runtime=60.0, priority=5, workload_type=WorkloadType.INFERENCE)
    j2 = Job(job_id=2, arrival_time=45.0, gpu_count=4, vram_per_gpu_gb=40.0, estimated_runtime=100.0, actual_runtime=100.0, priority=8, workload_type=WorkloadType.TRAINING)
    sim.load_workload([j1, j2])
    sim.step_to_next_decision()

    state = sim.get_state()
    # Next arrival should be j2 at t=45.0
    assert abs(state.next_arrival[0] - 45.0) < 1e-3
    assert state.next_arrival[1] == 4.0
    assert state.next_arrival[2] == 40.0

    env = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml", max_nodes=8, max_queue_size=16)
    obs = env._extract_observation(state)
    # Check that next arrival features in obs are populated accurately
    # Global feature indices: next_dt_norm (idx 212), next_gpus_norm (idx 213), next_vram_norm (idx 214)
    assert abs(obs[212] - min(1.0, 45.0 / 60.0)) < 1e-3
    assert abs(obs[213] - (4.0 / 8.0)) < 1e-3
    assert abs(obs[214] - (40.0 / 80.0)) < 1e-3
