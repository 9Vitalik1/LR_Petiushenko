"""Навчання агента Pong методом PPO (Proximal Policy Optimization).

Приклад:
    pong-train --config configs/training.json --run-name main
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..model import ActorCritic, save_model
from ..vec_env import SubprocVecPong

DEFAULTS = {
    "total_steps": 2_000_000,
    "num_envs": 16,
    "num_workers": 2,
    "horizon": 128,
    "epochs": 4,
    "minibatch_size": 512,
    "learning_rate": 2.5e-4,
    "anneal_lr": True,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_clip": 0.2,
    "entropy_coefficient": 0.01,
    "value_coefficient": 0.5,
    "gradient_clip": 0.5,
    "hidden": 128,
    "hit_reward": 0.1,
    "hit_reward_anneal_fraction": 0.5,
    "sticky_actions": 0.25,
    "seed": 42,
    "save_every": 50,
}


def hit_coef(cfg: dict, step: int) -> float:
    """Бонус за відбиття лінійно зменшується до нуля (лише на старті навчання)."""
    end = cfg["hit_reward_anneal_fraction"] * cfg["total_steps"]
    if end <= 0:
        return 0.0
    return cfg["hit_reward"] * max(0.0, 1.0 - step / end)


def train(cfg: dict, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(1)

    n, T = cfg["num_envs"], cfg["horizon"]
    batch = n * T
    iterations = cfg["total_steps"] // batch

    envs = SubprocVecPong(n, cfg["num_workers"], cfg["seed"], cfg["sticky_actions"])
    model = ActorCritic(hidden=cfg["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    obs_buf = torch.zeros(T, n, envs.obs_size)
    act_buf = torch.zeros(T, n, dtype=torch.long)
    logp_buf = torch.zeros(T, n)
    rew_buf = torch.zeros(T, n)
    done_buf = torch.zeros(T, n)
    val_buf = torch.zeros(T, n)

    obs = torch.as_tensor(envs.observe(), dtype=torch.float32)
    done = torch.zeros(n)
    recent = deque(maxlen=50)
    best = -np.inf
    step = 0
    t0 = time.time()
    metrics_f = open(run_dir / "metrics.jsonl", "w")
    episodes_f = open(run_dir / "episodes.csv", "w")
    episodes_f.write("step,score,hits,length\n")

    for it in range(1, iterations + 1):
        if cfg["anneal_lr"]:
            opt.param_groups[0]["lr"] = cfg["learning_rate"] * (1.0 - (it - 1) / iterations)
        coef = hit_coef(cfg, step)

        # ---------------------------------------------------------- збір досвіду
        for t in range(T):
            obs_buf[t] = obs
            done_buf[t] = done
            with torch.no_grad():
                d = model.dist(obs)
                a = d.sample()
                val_buf[t] = model.value(obs)
            act_buf[t] = a
            logp_buf[t] = d.log_prob(a)
            o, r, h, dn, finished = envs.step(a.numpy())
            rew_buf[t] = torch.as_tensor(r + coef * h)
            obs = torch.as_tensor(o, dtype=torch.float32)
            done = torch.as_tensor(dn)
            step += n
            for score, hits, length in finished:
                recent.append((score, hits, length))
                episodes_f.write(f"{step},{score:.0f},{hits},{length}\n")

        # ------------------------------------------------------------------ GAE
        with torch.no_grad():
            next_val = model.value(obs)
        adv = torch.zeros(T, n)
        last = torch.zeros(n)
        for t in reversed(range(T)):
            if t == T - 1:
                nonterm, nv = 1.0 - done, next_val
            else:
                nonterm, nv = 1.0 - done_buf[t + 1], val_buf[t + 1]
            delta = rew_buf[t] + cfg["gamma"] * nv * nonterm - val_buf[t]
            last = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterm * last
            adv[t] = last
        ret = adv + val_buf

        b_obs = obs_buf.reshape(batch, -1)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret, b_val = adv.reshape(-1), ret.reshape(-1), val_buf.reshape(-1)

        # ----------------------------------------------------------- оновлення PPO
        stats = []
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(batch)
            for s in range(0, batch, cfg["minibatch_size"]):
                idx = perm[s : s + cfg["minibatch_size"]]
                d = model.dist(b_obs[idx])
                logp = d.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                mb_adv = b_adv[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - cfg["ppo_clip"], 1 + cfg["ppo_clip"]),
                ).mean()
                v = model.value(b_obs[idx])
                v_clip = b_val[idx] + (v - b_val[idx]).clamp(-cfg["ppo_clip"], cfg["ppo_clip"])
                vloss = 0.5 * torch.max((v - b_ret[idx]) ** 2, (v_clip - b_ret[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pg + cfg["value_coefficient"] * vloss - cfg["entropy_coefficient"] * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - (logp - b_logp[idx])).mean()
                    clipfrac = ((ratio - 1).abs() > cfg["ppo_clip"]).float().mean()
                stats.append([pg.item(), vloss.item(), ent.item(), kl.item(), clipfrac.item()])

        st = np.mean(stats, axis=0)
        mean_score = float(np.mean([e[0] for e in recent])) if recent else float("nan")
        mean_hits = float(np.mean([e[1] for e in recent])) if recent else float("nan")
        win_rate = float(np.mean([e[0] > 0 for e in recent])) if recent else float("nan")
        rec = {
            "iteration": it,
            "steps": step,
            "seconds": round(time.time() - t0, 1),
            "mean_score_50": mean_score,
            "win_rate_50": win_rate,
            "mean_hits_50": mean_hits,
            "episodes": len(recent),
            "hit_bonus": coef,
            "lr": opt.param_groups[0]["lr"],
            "policy_loss": st[0],
            "value_loss": st[1],
            "entropy": st[2],
            "approx_kl": st[3],
            "clip_fraction": st[4],
        }
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        episodes_f.flush()

        meta = {"hidden": cfg["hidden"], "iteration": it, "steps": step, "mean_score_50": mean_score}
        save_model(model, run_dir / "latest.pt", meta)
        if len(recent) >= 20 and mean_score > best:
            best = mean_score
            save_model(model, run_dir / "best.pt", meta)
        if it % cfg["save_every"] == 0:
            save_model(model, run_dir / "checkpoints" / f"iter_{it:05d}.pt", meta)
        if it % 5 == 0 or it == 1:
            fps = step / (time.time() - t0)
            print(
                f"it {it:4d}/{iterations} steps {step:8d} | score50 {mean_score:6.2f} "
                f"win {win_rate:4.2f} hits {mean_hits:5.1f} | ent {st[2]:.3f} kl {st[3]:.4f} "
                f"| bonus {coef:.3f} | {fps:.0f} steps/s",
                flush=True,
            )

    envs.close()
    metrics_f.close()
    episodes_f.close()
    print("Готово. Найкращий середній рахунок (50 ігор):", best)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="PPO-навчання агента Atari Pong")
    p.add_argument("--config", type=Path, default=Path("configs/training.json"))
    p.add_argument("--run-name", default="main")
    p.add_argument("--total-steps", type=int, default=None)
    args = p.parse_args(argv)

    cfg = dict(DEFAULTS)
    if args.config and args.config.exists():
        cfg.update(json.loads(args.config.read_text()))
    if args.total_steps:
        cfg["total_steps"] = args.total_steps
    train(cfg, Path("runs") / args.run_name)


if __name__ == "__main__":
    main()
