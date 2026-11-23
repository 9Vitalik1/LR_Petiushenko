"""Оцінювання навченої моделі на нових партіях (без навчання, без бонусів).

Очки рахуються за табло гри (RAM): кожне збільшення рахунку агента або CPU
вважається одним очком. Так оцінка не залежить від рідкісного збою ALE,
коли за один крок приходить винагорода +6 через скидання рахунку CPU.

Приклад:
    pong-evaluate --checkpoint models/pong_ppo_final.pt --episodes 50 --seed 900000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..env import make_env
from ..model import load_model

MAX_STEPS = 27000  # ліміт ALE/Pong-v5: 108 000 кадрів / frameskip 4


def play_game(model, seed: int, sticky: float, greedy: bool = True) -> dict:
    env = make_env(seed=seed, sticky=sticky)
    obs, _ = env.reset(seed=seed)
    me = cpu = steps = anomalies = last_point = 0
    prev_a = prev_c = 0
    while True:
        obs, _, term, trunc, info = env.step(model.act(obs, greedy=greedy))
        steps += 1
        a, c = info["agent_points"], info["cpu_points"]
        if a > prev_a:
            me += a - prev_a
            last_point = steps
        if c > prev_c:
            cpu += c - prev_c
            last_point = steps
        if abs(info["game_reward"]) > 1:
            anomalies += 1
        prev_a, prev_c = a, c
        if term or trunc:
            break
    env.close()
    return {
        "seed": seed, "agent": int(me), "cpu": int(cpu), "diff": int(me - cpu),
        "hits": int(info["hits"]), "steps": steps, "time_limit": bool(trunc and not term),
        "steps_since_last_point": steps - last_point, "reward_anomalies": anomalies,
    }


def summarize(games: list[dict], **meta) -> dict:
    d = np.array([g["diff"] for g in games])
    return {
        **meta,
        "episodes": len(games),
        "wins": int((d > 0).sum()),
        "draws": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
        "win_rate": float((d > 0).mean()),
        "mean_score_diff": float(d.mean()),
        "std_score_diff": float(d.std()),
        "min_score_diff": int(d.min()),
        "max_score_diff": int(d.max()),
        "mean_agent_points": float(np.mean([g["agent"] for g in games])),
        "mean_cpu_points": float(np.mean([g["cpu"] for g in games])),
        "games_reached_21": int(sum(g["agent"] == 21 for g in games)),
        "games_time_limit": int(sum(g["time_limit"] for g in games)),
        "games_with_reward_anomaly": int(sum(g["reward_anomalies"] > 0 for g in games)),
        "mean_hits": float(np.mean([g["hits"] for g in games])),
        "games": games,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Оцінювання агента Pong")
    p.add_argument("--checkpoint", type=Path, default=Path("models/pong_ppo_final.pt"))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=900000)
    p.add_argument("--sticky", type=float, default=0.25)
    p.add_argument("--random-policy", action="store_true", help="оцінити випадкового агента")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    torch.set_num_threads(1)
    if args.random_policy:
        rng = np.random.default_rng(args.seed)

        class _Rand:
            def act(self, obs, greedy=True):
                return int(rng.integers(3))

        model = _Rand()
    else:
        model = load_model(args.checkpoint)

    games = []
    for k in range(args.episodes):
        g = play_game(model, args.seed + k, args.sticky)
        games.append(g)
        note = "  (ліміт часу)" if g["time_limit"] else ""
        print(f"гра {k + 1:3d}: агент {g['agent']:2d} : {g['cpu']:2d} CPU   "
              f"(відбиттів {g['hits']}){note}", flush=True)

    s = summarize(games, checkpoint="random" if args.random_policy else str(args.checkpoint),
                  seed_start=args.seed, sticky_actions=args.sticky, greedy=True)
    print(
        f"\nПеремоги: {s['wins']}/{s['episodes']} ({s['win_rate']:.0%}); "
        f"середній рахунок {s['mean_agent_points']:.1f} : {s['mean_cpu_points']:.1f}; "
        f"різниця {s['mean_score_diff']:+.2f} ± {s['std_score_diff']:.2f}"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
