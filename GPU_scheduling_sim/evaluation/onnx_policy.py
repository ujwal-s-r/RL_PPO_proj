"""ONNX Runtime PPO policy used by the lightweight hosted API."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort

from baselines.base import BaseScheduler
from simulator.job import WorkloadType
from simulator.scheduler_state import SchedulerState


WORKLOAD_TYPE_MAP = {
    WorkloadType.TRAINING: 0,
    WorkloadType.FINE_TUNING: 1,
    WorkloadType.EVALUATION: 2,
    WorkloadType.INFERENCE: 3,
    WorkloadType.EXPERIMENT: 4,
}


class ONNXPolicyScheduler(BaseScheduler):
    """Run the deterministic, action-masked PPO actor without PyTorch."""

    def __init__(
        self, checkpoint_path: str = "checkpoints/ppo_final.onnx", cluster_config_path: str = "configs/cluster_small.yaml"
    ) -> None:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"ONNX PPO model not found at '{checkpoint_path}'. Run python scripts/export_onnx.py before starting the API."
            )
        self._session = ort.InferenceSession(checkpoint_path, providers=["CPUExecutionProvider"])
        self._input_names = [item.name for item in self._session.get_inputs()]

    @property
    def name(self) -> str:
        return "PPO"

    def select_action(self, state: SchedulerState) -> Optional[Tuple[int, int]]:
        """Return a valid deterministic PPO action, with a safe scheduler fallback."""
        obs_vec = self._extract_observation(state)
        mask_vec = state.get_action_mask(max_nodes=8).flatten()
        if not np.any(mask_vec > 0):
            return self._first_valid_action(state)

        outputs = self._session.run(None, {
            self._input_names[0]: np.asarray(obs_vec, dtype=np.float32)[None, :],
            self._input_names[1]: np.asarray(mask_vec, dtype=np.float32)[None, :],
        })
        job_idx, node_idx = self._decode_action(int(np.asarray(outputs[0]).reshape(-1)[0]))
        if state.is_action_valid(job_idx, node_idx):
            return job_idx, node_idx

        valid_indices = np.flatnonzero(mask_vec > 0)
        if len(valid_indices):
            return self._decode_action(int(valid_indices[0]))
        return self._first_valid_action(state)

    @staticmethod
    def _decode_action(flat_action: int) -> Tuple[int, int]:
        return flat_action // 8, flat_action % 8

    @staticmethod
    def _extract_observation(state: SchedulerState) -> np.ndarray:
        """Match the feature layout used when the PPO actor was trained."""
        obs = np.zeros(215, dtype=np.float32)
        idx = 0
        for slot_idx in range(8):
            if slot_idx < state.cluster.num_nodes:
                node = state.cluster.nodes[slot_idx]
                features = [
                    node.available_gpu_count / max(1, node.gpu_count),
                    node.gpu_utilization,
                    node.vram_utilization,
                    node.free_vram_gb / 80.0,
                    node.vram_per_gpu_gb / 80.0,
                    node.gpu_count / 8.0,
                    min(1.0, len(node.running_jobs) / 16.0),
                    (node.node_id + 1.0) / 10.0,
                ]
            else:
                features = [0.0] * 8
            obs[idx : idx + 8] = features
            idx += 8

        for slot_idx in range(16):
            job = state.queue.get_at(slot_idx)
            if job is not None:
                remaining_deadline = max(0.0, job.deadline - state.current_time)
                features = [
                    1.0,
                    job.gpu_count / 8.0,
                    job.vram_per_gpu_gb / 80.0,
                    min(2.0, job.estimated_runtime / 600.0),
                    job.priority / 10.0,
                    min(2.0, (state.current_time - job.arrival_time) / 600.0),
                    min(2.0, remaining_deadline / 1200.0),
                    WORKLOAD_TYPE_MAP.get(job.workload_type, 0) / 5.0,
                    1.0 if remaining_deadline < job.estimated_runtime * 1.5 else 0.0,
                ]
            else:
                features = [0.0] * 9
            obs[idx : idx + 9] = features
            idx += 9

        next_dt, next_gpus, next_vram = state.next_arrival
        obs[idx : idx + 7] = [
            min(1.0, state.current_time / max(1.0, state.horizon_time)),
            len(state.queue) / 16.0,
            state.cluster.gpu_utilization,
            state.cluster.vram_utilization,
            min(1.0, next_dt / 60.0),
            next_gpus / 8.0,
            next_vram / 80.0,
        ]
        return obs

    @staticmethod
    def _first_valid_action(state: SchedulerState) -> Optional[Tuple[int, int]]:
        for job_idx in range(len(state.queue)):
            for node_idx in range(state.num_nodes):
                if state.is_action_valid(job_idx, node_idx):
                    return job_idx, node_idx
        return None
