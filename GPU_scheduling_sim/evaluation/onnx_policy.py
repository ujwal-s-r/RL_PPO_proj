"""ONNX Runtime PPO policy used by the lightweight hosted API."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from baselines.base import BaseScheduler
from env.server.gpu_scheduler_environment import GPUSchedulerEnvironment
from simulator.scheduler_state import SchedulerState


class ONNXPolicyScheduler(BaseScheduler):
    """Run the deterministic, action-masked PPO actor without PyTorch."""

    def __init__(
        self, checkpoint_path: str = "checkpoints/ppo_final.onnx", cluster_config_path: str = "configs/cluster_small.yaml"
    ) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ONNX PPO model not found at '{checkpoint_path}'. Run python scripts/export_onnx.py before starting the API."
            )
        self._temp_env = GPUSchedulerEnvironment(cluster_config_path=cluster_config_path)
        self._session = ort.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])
        self._input_names = [item.name for item in self._session.get_inputs()]

    @property
    def name(self) -> str:
        return "PPO"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        """Return a valid deterministic PPO action, with a safe scheduler fallback."""
        obs_vec = self._temp_env._extract_observation(state)
        mask_vec = state.get_action_mask(max_nodes=self._temp_env.max_nodes).flatten()
        if not np.any(mask_vec > 0):
            return self._first_valid_action(state)

        outputs = self._session.run(None, {
            self._input_names[0]: np.asarray(obs_vec, dtype=np.float32)[None, :],
            self._input_names[1]: np.asarray(mask_vec, dtype=np.float32)[None, :],
        })
        job_idx, node_idx = self._temp_env.decode_action(int(np.asarray(outputs[0]).reshape(-1)[0]))
        if state.is_action_valid(job_idx, node_idx):
            return job_idx, node_idx

        valid_indices = np.flatnonzero(mask_vec > 0)
        if len(valid_indices):
            return self._temp_env.decode_action(int(valid_indices[0]))
        return self._first_valid_action(state)

    @staticmethod
    def _first_valid_action(state: SchedulerState) -> Optional[Tuple[int, int]]:
        for job_idx in range(len(state.queue)):
            for node_idx in range(state.num_nodes):
                if state.is_action_valid(job_idx, node_idx):
                    return job_idx, node_idx
        return None
