"""Simple test runner using Python standard library unittest."""

import os
import sys
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

if __name__ == "__main__":
    from tests.test_gpu import test_gpu_lifecycle, test_node_scheduling
    from tests.test_cluster import test_cluster_initialization, test_cluster_invariants_during_execution
    from tests.test_queue import test_queue_push_pop
    from tests.test_simulator import test_simulator_basic_flow
    from tests.test_workloads import test_seed_determinism, test_all_scenarios_generation
    from tests.test_environment import (
        test_env_reset_and_obs_shape,
        test_env_step_valid_and_invalid,
        test_env_seed_determinism,
        test_openenv_client_flow,
    )
    from tests.test_baselines import (
        test_fifo_logic,
        test_sjf_logic,
        test_priority_logic,
        test_best_fit_logic,
        test_baseline_runner_episode,
    )
    from tests.test_ppo import (
        test_actor_critic_action_masking,
        test_rollout_buffer_gae,
        test_checkpoint_save_and_load,
        test_ppo_single_update,
    )
    from tests.test_evaluation import (
        test_ppo_policy_scheduler_inference,
        test_evaluator_with_ppo,
    )

    print("Running Complete Test Suite (Phases 1-6)...")
    
    test_gpu_lifecycle()
    print("[PASS] test_gpu_lifecycle")
    
    test_node_scheduling()
    print("[PASS] test_node_scheduling")
    
    test_cluster_initialization()
    print("[PASS] test_cluster_initialization")
    
    test_cluster_invariants_during_execution()
    print("[PASS] test_cluster_invariants_during_execution")
    
    test_queue_push_pop()
    print("[PASS] test_queue_push_pop")
    
    test_simulator_basic_flow()
    print("[PASS] test_simulator_basic_flow")

    test_seed_determinism()
    print("[PASS] test_seed_determinism")

    test_all_scenarios_generation()
    print("[PASS] test_all_scenarios_generation")

    test_env_reset_and_obs_shape()
    print("[PASS] test_env_reset_and_obs_shape")

    test_env_step_valid_and_invalid()
    print("[PASS] test_env_step_valid_and_invalid")

    test_env_seed_determinism()
    print("[PASS] test_env_seed_determinism")

    test_openenv_client_flow()
    print("[PASS] test_openenv_client_flow")

    test_fifo_logic()
    print("[PASS] test_fifo_logic")

    test_sjf_logic()
    print("[PASS] test_sjf_logic")

    test_priority_logic()
    print("[PASS] test_priority_logic")

    test_best_fit_logic()
    print("[PASS] test_best_fit_logic")

    test_baseline_runner_episode()
    print("[PASS] test_baseline_runner_episode")

    test_actor_critic_action_masking()
    print("[PASS] test_actor_critic_action_masking")

    test_rollout_buffer_gae()
    print("[PASS] test_rollout_buffer_gae")

    test_checkpoint_save_and_load()
    print("[PASS] test_checkpoint_save_and_load")

    test_ppo_single_update()
    print("[PASS] test_ppo_single_update")

    test_ppo_policy_scheduler_inference()
    print("[PASS] test_ppo_policy_scheduler_inference")

    test_evaluator_with_ppo()
    print("[PASS] test_evaluator_with_ppo")

    print("\nAll 23 tests passed successfully!")
