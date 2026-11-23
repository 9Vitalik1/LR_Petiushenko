"""Запис гри агента у GIF/MP4 та кадрів для звіту.

Приклад:
    pong-record --checkpoint models/pong_ppo_final.pt --out docs/assets/pong_agent.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ..env import make_env
from ..model import load_model


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Запис гри агента Pong")
    p.add_argument("--checkpoint", type=Path, default=Path("models/pong_ppo_final.pt"))
    p.add_argument("--out", type=Path, default=Path("docs/assets/pong_agent.gif"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-frames", type=int, default=600, help="скільки кадрів писати у GIF")
    p.add_argument("--every", type=int, default=2, help="брати кожен N-й кадр")
    p.add_argument("--snapshot", type=Path, default=None, help="зберегти один кадр як PNG")
    args = p.parse_args(argv)

    model = load_model(args.checkpoint)
    env = make_env(seed=args.seed, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    frames, me, cpu, t = [], 0, 0, 0
    snap_saved = False
    while True:
        obs, _, term, trunc, info = env.step(model.act(obs))
        me, cpu = info["agent_points"], info["cpu_points"]  # табло гри
        if t % args.every == 0 and len(frames) < args.max_frames:
            frames.append(env.render())
        if args.snapshot and not snap_saved and t == 1200:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(args.snapshot, env.render())
            snap_saved = True
        t += 1
        if term or trunc:
            final = env.render()
            break
    frames += [final] * 30  # показати фінальний рахунок
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gif":
        imageio.mimsave(args.out, [np.asarray(f)[::2, ::2] for f in frames], duration=1 / 30, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=30)
    if args.snapshot:
        imageio.imwrite(args.snapshot.with_name(args.snapshot.stem + "_final.png"), final)
    print(f"Записано {args.out}; рахунок гри {me}:{cpu}")


if __name__ == "__main__":
    main()
