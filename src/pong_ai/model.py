"""Нейронна мережа Actor-Critic для Pong.

Actor і Critic мають окремі ваги (як у SnakeAI v9), але повнозв'язну будову,
бо на вхід подається вектор ознак, а не зображення.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .env import ACTIONS, OBS_SIZE


def _layer(inp: int, out: int, gain: float = np.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(inp, out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int = OBS_SIZE, n_actions: int = len(ACTIONS), hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_size, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, n_actions, gain=0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_size, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, gain=1.0),
        )

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        logits = self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if greedy:
            return int(logits.argmax(-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())


def save_model(model: ActorCritic, path, meta: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)


def load_model(path) -> ActorCritic:
    data = torch.load(path, map_location="cpu", weights_only=False)
    hidden = data.get("meta", {}).get("hidden", 128)
    model = ActorCritic(hidden=hidden)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model
