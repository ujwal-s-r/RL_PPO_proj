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

    print("Running Simulator Unit Tests...")
    
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

    print("\nAll 6 tests passed successfully!")
