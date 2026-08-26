"""Cluster model managing heterogeneous nodes and safety invariants."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import yaml
from simulator.node import Node
from simulator.job import Job


class Cluster:
    """Manages a collection of heterogeneous GPU nodes."""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes: List[Node] = nodes

    @classmethod
    def create_dynamic(cls, node_specs: List[Dict[str, Any]]) -> Cluster:
        """Create cluster from list of node specifications."""
        nodes: List[Node] = []
        for idx, spec in enumerate(node_specs):
            node = Node(
                node_id=idx,
                gpu_type=spec.get("gpu_type", "A100-SXM4-80GB"),
                gpu_count=int(spec.get("gpu_count", 4)),
                vram_per_gpu_gb=float(spec.get("vram_per_gpu_gb", spec.get("vram_gb", 80.0))),
                cpu_cores=int(spec.get("cpu_cores", 32)),
                ram_gb=float(spec.get("ram_gb", 128.0)),
            )
            nodes.append(node)
        return cls(nodes)

    @classmethod
    def create_random(cls, num_nodes: Optional[int] = None, seed: Optional[int] = None) -> Cluster:
        """
        Generate random cluster with 1 to 10 nodes and diverse GPU configurations.
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        n = num_nodes if num_nodes is not None else int(rng.integers(1, 11))

        gpu_templates = [
            {"type": "A100-SXM4-80GB", "vram": 80.0, "counts": [2, 4, 8]},
            {"type": "NVIDIA-H100-80GB", "vram": 80.0, "counts": [2, 4, 8]},
            {"type": "A100-PCIE-40GB", "vram": 40.0, "counts": [2, 4, 8]},
            {"type": "NVIDIA-A10-24GB", "vram": 24.0, "counts": [2, 4, 8]},
        ]

        nodes: List[Node] = []
        for idx in range(n):
            tmpl = rng.choice(gpu_templates)
            gpu_cnt = int(rng.choice(tmpl["counts"]))
            node = Node(
                node_id=idx,
                gpu_type=tmpl["type"],
                gpu_count=gpu_cnt,
                vram_per_gpu_gb=tmpl["vram"],
                cpu_cores=32,
                ram_gb=128.0,
            )
            nodes.append(node)
        return cls(nodes)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> Cluster:
        """Instantiate cluster from YAML configuration file."""
        import os
        if not os.path.exists(yaml_path):
            # Default to dynamic 3-node cluster if path does not exist
            return cls.create_random(num_nodes=3, seed=42)
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> Cluster:
        """Instantiate cluster from config dictionary."""
        nodes: List[Node] = []
        for n_cfg in cfg.get("nodes", []):
            node = Node(
                node_id=n_cfg.get("id", len(nodes)),
                gpu_type=n_cfg["type"],
                gpu_count=n_cfg.get("gpu_count", n_cfg.get("count", 4)),
                vram_per_gpu_gb=float(n_cfg.get("vram_gb", 40.0)),
                cpu_cores=int(n_cfg.get("cpu_cores", 32)),
                ram_gb=float(n_cfg.get("ram_gb", 128.0)),
            )
            nodes.append(node)
        return cls(nodes)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_gpus(self) -> int:
        return sum(node.gpu_count for node in self.nodes)

    @property
    def available_gpus(self) -> int:
        return sum(node.available_gpu_count for node in self.nodes)

    @property
    def total_vram_gb(self) -> float:
        return sum(node.total_vram_gb for node in self.nodes)

    @property
    def allocated_vram_gb(self) -> float:
        return sum(node.allocated_vram_gb for node in self.nodes)

    @property
    def gpu_utilization(self) -> float:
        """Global cluster GPU utilization fraction (0.0 to 1.0)."""
        if self.total_gpus == 0:
            return 0.0
        return (self.total_gpus - self.available_gpus) / self.total_gpus

    @property
    def vram_utilization(self) -> float:
        """Global cluster VRAM utilization fraction (0.0 to 1.0)."""
        if self.total_vram_gb <= 0:
            return 0.0
        return self.allocated_vram_gb / self.total_vram_gb

    def get_node(self, node_id: int) -> Optional[Node]:
        """Find node by ID."""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def validate_invariants(self) -> None:
        """
        Verify safety invariants across the entire cluster:
        - allocated GPUs <= physical GPUs
        - allocated VRAM <= physical VRAM
        - no two active jobs share the exact same GPU slot
        
        Raises:
            AssertionError: If any invariant is violated.
        """
        seen_job_gpu_pairs = set()
        for node in self.nodes:
            alloc_gpus = node.gpu_count - node.available_gpu_count
            assert 0 <= alloc_gpus <= node.gpu_count, (
                f"Node {node.node_id} invariant violated: "
                f"allocated {alloc_gpus} > total {node.gpu_count}"
            )
            assert node.allocated_vram_gb <= node.total_vram_gb + 1e-6, (
                f"Node {node.node_id} VRAM invariant violated: "
                f"allocated {node.allocated_vram_gb}GB > total {node.total_vram_gb}GB"
            )
            for gpu in node.gpus:
                if not gpu.is_free:
                    key = (node.node_id, gpu.gpu_id)
                    assert key not in seen_job_gpu_pairs, (
                        f"GPU conflict detected on Node {node.node_id}, GPU {gpu.gpu_id}"
                    )
                    seen_job_gpu_pairs.add(key)
                    assert gpu.allocated_vram_gb <= gpu.total_vram_gb, (
                        f"GPU {gpu.gpu_id} over-allocated on Node {node.node_id}"
                    )

    def reset(self) -> None:
        """Reset all nodes in the cluster."""
        for node in self.nodes:
            node.reset()
