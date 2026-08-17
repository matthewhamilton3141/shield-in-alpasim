"""The Tier 2 RL policy net + checkpoint I/O + the shield-`policy` adapter.

Lives in the package (not a script) so training (`scripts/rl_scaled.py`) and deployment
(`ShieldedDriver`) import ONE definition of the network and the observation — the shield-trained
policy only transfers to AlpaSim if the net and its input match exactly.

The adapter is the key idea: the shield already consumes a `policy(state) -> (accel, steer)`
callable (`kitti_nav.shielded_rollout`), and a trained net *is* one — given the per-cycle obstacle
field, `RLPolicy.bound(field)` returns exactly that callable. So the learned policy slots into the
driver's existing rollout the same way the coast baseline does; the shield still filters it.

torch is imported here, so the driver defers importing this module until the RL mode is actually
selected (torch is absent on the Mac's non-AlpaSim path, present in the driver container).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from kitti_nav.vehicle import VehicleState
from shield_in_alpasim.rl_env import (
    ACTION_GRID,
    N_ACTIONS,
    EnvConfig,
    build_observation,
    observation_dim,
)

_OBS_SCALE_EGO = np.array([15.0, 0.52, 40.0], np.float32)


def obs_scale(obs_dim: int) -> torch.Tensor:
    """Per-feature scale (speeds/metres/radii -> ~unit) before the net; must match training."""
    n_disc = (obs_dim - 3) // 3
    disc = np.tile([40.0, 4.5, 1.0], n_disc).astype(np.float32)
    return torch.tensor(np.concatenate([_OBS_SCALE_EGO, disc]))


class ActorCritic(nn.Module):
    """Small shared-trunk MLP: policy logits over the discrete actions + a state value."""

    def __init__(self, obs_dim: int, n_actions: int = N_ACTIONS, hidden: int = 128):
        super().__init__()
        self.register_buffer("scale", obs_scale(obs_dim))
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)

    def forward(self, obs):
        h = self.trunk(obs / self.scale)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np):
        """Sample an action (training rollouts): returns (action_idx, log_prob, value)."""
        logits, value = self(torch.as_tensor(obs_np, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a), float(dist.log_prob(a)), float(value)


def save_policy(ac: ActorCritic, path: str, cfg: EnvConfig, meta: dict | None = None) -> None:
    """Checkpoint the net plus everything the loader needs to rebuild an identical obs + net."""
    torch.save({
        "state_dict": ac.state_dict(),
        "obs_dim": int(ac.scale.numel()),
        "hidden": ac.pi.in_features,
        "n_actions": ac.pi.out_features,
        # the obs-relevant EnvConfig fields (so deployment builds the SAME observation)
        "n_nearest": cfg.n_nearest,
        "goal_x": cfg.goal_x,
        "meta": meta or {},
    }, path)


@dataclass
class RLPolicy:
    """A trained net adapted to the shield's `policy(state) -> (accel, steer)` interface.

    `bound(field_circles)` returns the per-cycle callable the shield rollout expects; it builds the
    training observation from the rollout `state` and the (fixed, this-cycle) obstacle discs, runs
    the net, and returns the greedy discrete action as a continuous `(accel, steer)` command.
    """

    net: ActorCritic
    cfg: EnvConfig

    def bound(self, field_circles: np.ndarray):
        circles = np.asarray(field_circles, float).reshape(-1, 3)

        def policy(state: VehicleState) -> tuple[float, float]:
            obs = build_observation(state, circles, self.cfg)
            with torch.no_grad():
                logits, _ = self.net(torch.as_tensor(obs, dtype=torch.float32))
            return ACTION_GRID[int(torch.argmax(logits))]

        return policy


def load_policy(path: str, device: str = "cpu") -> RLPolicy:
    """Load a checkpoint saved by `save_policy` into a deployment-ready `RLPolicy`."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    # Rebuild the obs-relevant config so deployment reproduces the training observation exactly.
    cfg = EnvConfig(n_nearest=ckpt["n_nearest"], goal_x=ckpt["goal_x"])
    dim = observation_dim(cfg)
    if dim != ckpt["obs_dim"]:
        raise ValueError(f"checkpoint obs_dim {ckpt['obs_dim']} != rebuilt {dim} — config drift")
    return RLPolicy(net, cfg)
