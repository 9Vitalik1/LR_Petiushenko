"""Показ гри навченого агента у вікні (потрібен pygame: pip install pygame).

Приклад:
    pong-play --checkpoint models/pong_ppo_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..env import make_env
from ..model import load_model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Показ гри агента Pong")
    p.add_argument("--checkpoint", type=Path, default=Path("models/pong_ppo_final.pt"))
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model = load_model(args.checkpoint)
    env = make_env(seed=args.seed, render_mode="human")
    for g in range(args.games):
        obs, _ = env.reset(seed=args.seed + g)
        me = cpu = 0
        while True:
            obs, _, term, trunc, info = env.step(model.act(obs))
            me += info["game_reward"] > 0
            cpu += info["game_reward"] < 0
            if term or trunc:
                break
        print(f"Гра {g + 1}: агент {me} : {cpu} CPU")
    env.close()


if __name__ == "__main__":
    main()
