import numpy as np
import torch

from pong_ai.env import ACTIONS, OBS_SIZE, make_env
from pong_ai.model import ActorCritic, load_model, save_model


def test_env_observation_shape_and_actions():
    env = make_env(seed=0, sticky=0.0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert env.action_space.n == len(ACTIONS) == 3
    for _ in range(200):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        assert "game_reward" in info and "hit" in info
        if term or trunc:
            break
    env.close()


def test_model_outputs():
    m = ActorCritic()
    x = torch.zeros(5, OBS_SIZE)
    assert m.actor(x).shape == (5, 3)
    assert m.value(x).shape == (5,)
    assert 0 <= m.act(np.zeros(OBS_SIZE, dtype=np.float32)) < 3


def test_save_load_roundtrip(tmp_path):
    m = ActorCritic(hidden=32)
    save_model(m, tmp_path / "m.pt", {"hidden": 32})
    m2 = load_model(tmp_path / "m.pt")
    x = torch.randn(4, OBS_SIZE)
    assert torch.allclose(m.actor(x), m2.actor(x))
