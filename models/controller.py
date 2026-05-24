import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class Zeros(nn.Module):
    """Placeholder used for transformer blocks that are not in edit_layers.
    Returns an all-zero delta so those blocks pass through unchanged without
    any conditional logic inside Controller.forward."""
    def forward(
        self,
        h: torch.Tensor,
        time_feat: torch.Tensor | None = None,
    ):
        return torch.zeros_like(h)


class Vector(nn.Module):
    def __init__(
        self,
        hidden_dim=(107, 1536),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Initialized to zero so the Controller has no effect at the start of training,
        # letting the base model converge before the delta grows.
        self.delta_weight = nn.Parameter(torch.zeros(hidden_dim))

    def __repr__(self):
        return f"MLP(hidden_dim={self.hidden_dim})"
    
    def forward(
        self,
        h: torch.Tensor,
        time_feat: torch.Tensor | None = None,
    ):
        B, T, D = h.shape
        delta = self.delta_weight.unsqueeze(0).expand(h.size(0), -1, -1)  # [B, T, D]
        return delta
    
class Transformer(nn.Module):
    """Down-project → TransformerEncoder → up-project bottleneck that produces a
    temporally-aware steering delta.  Captures relationships across all time positions
    in the hidden state, unlike Vector which applies a single shared delta."""
    def __init__(self, hidden_dim=(1024, 1536), bottleneck_dim=256, nhead=8, num_layers=1):
        super().__init__()
        # Down-project to bottleneck to keep parameter count manageable
        self.down_proj = nn.Linear(hidden_dim[1], bottleneck_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim[0], bottleneck_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_dim,
            nhead=nhead,
            dim_feedforward=bottleneck_dim * 2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim[1])
        self.norm = nn.LayerNorm(hidden_dim[1])

    def forward(
        self,
        h: torch.Tensor,
        time_feat: torch.Tensor | None = None,
    ):
        x = self.down_proj(h) + self.pos_embedding
        x = self.transformer(x)
        delta_h = self.up_proj(x)
        return self.norm(delta_h)

class Controller(nn.Module):
    def __init__(
        self,
        edit_layers: List[int],
        edit_steps: List[int] = None,    # None means all timesteps
        use_time: bool = False,
        total_layers: int = 24,
        time_emb_dim: int = 1536,
        hidden_dim: Tuple[int] = (107, 1536),
        control_type: str = "vector",
        inject_first_state: bool = False,
    ):
        super().__init__()

        if use_time:
            self.time_net = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim*2),
            )
            # Zero-init so the affine modulation starts as identity (gamma=0, beta=0),
            # keeping early training identical to use_time=False.
            torch.nn.init.zeros_(self.time_net[1].weight)
            torch.nn.init.zeros_(self.time_net[1].bias)
        else:
            self.time_net = None

        if control_type == "vector":
            self.controllers = nn.ModuleList([
                Vector(hidden_dim=hidden_dim)
                if i in edit_layers
                else Zeros()
                for i in range(total_layers)
            ])
        elif control_type == "transformer":
            self.controllers = nn.ModuleList([
                Transformer(hidden_dim=hidden_dim)
                if i in edit_layers
                else Zeros()
                for i in range(total_layers)
            ])
        else:
            raise NotImplementedError(control_type)

        # Position 0 of hidden_states is the prepended audio-duration global token.
        # Normally injection starts at position 1 to leave duration conditioning intact.
        self.inject_offset = 0 if inject_first_state else 1
        self.edit_steps = set(edit_steps) if edit_steps is not None else edit_steps
        # alpha is a runtime scaling factor applied on top of steer.  Both multiply the
        # delta, but alpha is set externally (e.g. controller.alpha = 1.0) while steer
        # is passed per-call by the pipeline.
        self.alpha = 1.0

    def forward(self, idx, hidden_states, steer, timestep_idx, time_emb=None):

        # timestep
        if self.edit_steps is not None and timestep_idx not in self.edit_steps:
            return hidden_states

        B = hidden_states.shape[0]
        # steer is a scalar weight applied to the controller delta
        if not torch.is_tensor(steer):
            if not isinstance(steer, (list, tuple)):
                steer = [steer]
            steer = torch.tensor(steer, device=hidden_states.device)
        if len(steer) == 1:
            steer = steer.expand(B)
        if len(steer) != B:
            # CFG doubles the batch (unconditional + conditional); steer must cover both halves.
            assert len(steer) * 2 == B, f"len(steer) = {len(steer)}, B = {B}"
            steer = torch.cat([steer, steer], dim=0)

        # Always exclude position 0 (audio-duration token) from the delta computation.
        # inject_offset controls where the result is written back, not what is read.
        delta = self.controllers[idx](hidden_states[:, 1:, :])

        # time_net
        if self.time_net is not None:
            time_feat = self.time_net(time_emb)
            gamma, beta = time_feat.chunk(2, dim=-1)
            gamma = torch.tanh(gamma)
            delta = delta * (1 + gamma[:, None, :]) + beta[:, None, :]

        # edit
        steer = steer.view(B, 1, 1)
        delta = delta * steer
        s = self.inject_offset
        e = s + delta.shape[1]
        parts = []
        if s > 0:
            parts.append(hidden_states[:, :s, :])
        parts.append(hidden_states[:, s:e, :] + delta * self.alpha)
        if e < hidden_states.shape[1]:
            parts.append(hidden_states[:, e:, :])
        return torch.cat(parts, dim=1)
