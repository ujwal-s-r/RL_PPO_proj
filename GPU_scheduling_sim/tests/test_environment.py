"""Unit tests for GPUSchedulerEnvironment and OpenEnv integration."""

import numpy as np
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment
from env.client import OpenEnvClient


def test_env_reset_and_obs_shape():
    env = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml")
    obs, info = env.reset(seed=42)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32
    assert "action_mask" in info
    assert len(info["action_mask"]) == env.action_dim
    assert not np.isnan(obs).any()


def test_env_step_valid_and_invalid():
    env = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml")
    obs, info = env.reset(seed=42)
    mask = info["action_mask"]

    # 1. Take a valid action (first index with mask == 1.0)
    valid_indices = np.where(mask > 0)[0]
    assert len(valid_indices) > 0, "Initial state must have at least one valid action"
    valid_action = int(valid_indices[0])

    next_obs, reward, terminated, truncated, step_info = env.step(valid_action)
    assert step_info["valid_action"] is True
    assert next_obs.shape == (env.obs_dim,)
    assert not np.isnan(reward)

    # 2. Take an invalid action (index where mask == 0.0)
    invalid_indices = np.where(step_info["action_mask"] == 0)[0]
    if len(invalid_indices) > 0:
        inv_action = int(invalid_indices[0])
        _, inv_reward, _, _, inv_info = env.step(inv_action)
        assert inv_info["valid_action"] is False
        assert inv_reward < 0.0  # Must incur penalty


def test_env_seed_determinism():
    env1 = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml")
    env2 = GPUSchedulerEnvironment(cluster_config_path="configs/cluster_small.yaml")

    obs1, info1 = env1.reset(seed=100)
    obs2, info2 = env2.reset(seed=100)

    np.testing.assert_allclose(obs1, obs2, atol=1e-5)
    np.testing.assert_allclose(info1["action_mask"], info2["action_mask"], atol=1e-5)


def test_openenv_client_flow():
    client = OpenEnvClient(cluster_config_path="configs/cluster_small.yaml")
    obs, info = client.reset(seed=42)
    assert obs.shape == (client.obs_dim,)

    done = False
    step_count = 0
    while not done and step_count < 20:
        mask = client.get_action_mask()
        valid_acts = np.where(mask > 0)[0]
        if len(valid_acts) == 0:
            break
        act = int(valid_acts[0])
        obs, reward, term, trunc, s_info = client.step(act)
        done = term or trunc
        step_count += 1

    assert step_count > 0
