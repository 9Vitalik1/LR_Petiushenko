"""Простий векторизований запуск кількох ігор Pong у кількох процесах.

Кожен процес-робітник веде свою групу середовищ і сам перезапускає партію,
коли вона закінчилася. Головний процес отримує вже готові масиви numpy.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .env import OBS_SIZE, make_env


def _worker(conn, seeds, sticky):
    envs = [make_env(seed=s, sticky=sticky) for s in seeds]
    obs = np.stack([e.reset(seed=int(s))[0] for e, s in zip(envs, seeds)])
    ep_score = np.zeros(len(envs))
    ep_hits = np.zeros(len(envs))
    ep_len = np.zeros(len(envs), dtype=np.int64)
    while True:
        cmd, data = conn.recv()
        if cmd == "obs":
            conn.send(obs)
        elif cmd == "step":
            n = len(envs)
            rew = np.zeros(n, np.float32)
            hit = np.zeros(n, np.float32)
            done = np.zeros(n, np.float32)
            finished = []
            for i, (env, a) in enumerate(zip(envs, data)):
                o, _, term, trunc, info = env.step(a)
                rew[i] = info["game_reward"]
                hit[i] = info["hit"]
                ep_score[i] += info["game_reward"]
                ep_hits[i] += info["hit"]
                ep_len[i] += 1
                if term or trunc:
                    done[i] = 1.0
                    finished.append((float(ep_score[i]), int(ep_hits[i]), int(ep_len[i])))
                    ep_score[i] = ep_hits[i] = ep_len[i] = 0
                    o, _ = env.reset()
                obs[i] = o
            conn.send((obs.copy(), rew, hit, done, finished))
        elif cmd == "close":
            for e in envs:
                e.close()
            conn.close()
            break


class SubprocVecPong:
    def __init__(self, num_envs: int, num_workers: int, seed: int, sticky: float = 0.25):
        num_workers = max(1, min(num_workers, num_envs))
        chunks = np.array_split(np.arange(num_envs), num_workers)
        ctx = mp.get_context("spawn")
        self.conns, self.procs, self.sizes = [], [], []
        for chunk in chunks:
            parent, child = ctx.Pipe()
            seeds = [seed + int(i) for i in chunk]
            p = ctx.Process(target=_worker, args=(child, seeds, sticky), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
            self.sizes.append(len(chunk))
        self.num_envs = num_envs
        self.obs_size = OBS_SIZE

    def observe(self) -> np.ndarray:
        for c in self.conns:
            c.send(("obs", None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions: np.ndarray):
        start = 0
        for c, n in zip(self.conns, self.sizes):
            c.send(("step", actions[start : start + n].tolist()))
            start += n
        results = [c.recv() for c in self.conns]
        obs = np.concatenate([r[0] for r in results])
        rew = np.concatenate([r[1] for r in results])
        hit = np.concatenate([r[2] for r in results])
        done = np.concatenate([r[3] for r in results])
        finished = [f for r in results for f in r[4]]
        return obs, rew, hit, done, finished

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
