"""Export the trained deterministic PPO actor for lightweight ONNX serving."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from rl.actor_critic import ActorCritic


class DeterministicPPOActor(nn.Module):
    """ONNX-friendly form of ``get_action(..., deterministic=True)``."""

    def __init__(self, actor_critic: ActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, observation: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        model = self.actor_critic
        reps = model._encode_representations(observation)
        batch_size = observation.shape[0]
        mask_2d = action_mask.view(batch_size, model.max_queue, model.num_nodes)

        job_logits = model._compute_job_logits(reps["q_refined"], reps["g_emb"])
        job_mask = mask_2d.sum(dim=-1) > 0.5
        chosen_job = torch.argmax(model._apply_mask_to_logits(job_logits, job_mask), dim=-1)

        node_logits = model._compute_conditional_node_logits(chosen_job, reps)
        batch_idx = torch.arange(batch_size, device=observation.device)
        node_mask = mask_2d[batch_idx, chosen_job, :]
        chosen_node = torch.argmax(model._apply_mask_to_logits(node_logits, node_mask), dim=-1)
        return chosen_job * model.num_nodes + chosen_node


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPO checkpoint for ONNX Runtime serving.")
    parser.add_argument("--checkpoint", default="checkpoints/ppo_final.pt")
    parser.add_argument("--output", default="checkpoints/ppo_final.onnx")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = ActorCritic(
        obs_dim=config.get("obs_dim", 215),
        action_dim=config.get("action_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        DeterministicPPOActor(model).eval(),
        (torch.zeros(1, model.obs_dim), torch.ones(1, model.action_dim)),
        output_path,
        input_names=["observation", "action_mask"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"observation": {0: "batch"}, "action_mask": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"Exported deterministic PPO actor to {output_path}")


if __name__ == "__main__":
    main()
