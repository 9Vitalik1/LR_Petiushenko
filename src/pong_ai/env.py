"""Середовище Atari Pong з компактним вектором ознак та системою винагород.

Мережа не отримує готових рішень: вона бачить лише положення та швидкості
м'яча й ракеток, прочитані з оперативної пам'яті (RAM) приставки Atari 2600.
Дію завжди обирає нейронна мережа.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

try:  # ale-py >= 0.9 реєструє ігри через gym.register_envs
    import ale_py

    gym.register_envs(ale_py)
except Exception:  # pragma: no cover
    pass

# Адреси RAM для Pong (джерело: AtariARI, Anand et al., 2019)
RAM_BALL_X = 49
RAM_BALL_Y = 54
RAM_PLAYER_Y = 51  # наша (права, зелена) ракетка
RAM_ENEMY_Y = 50  # ракетка вбудованого суперника
RAM_PLAYER_SCORE = 14
RAM_ENEMY_SCORE = 13

# Дії мережі -> дії ALE. FIRE не потрібен: подача в Pong автоматична.
ACTIONS = (0, 2, 3)  # NOOP, RIGHT (ракетка вгору), LEFT (ракетка вниз)
ACTION_NAMES = ("на місці", "вгору", "вниз")

OBS_SIZE = 12
PLAYER_X = 188  # горизонтальна координата нашої ракетки


class PongFeatures(gym.Wrapper):
    """Перетворює RAM Pong у 12 ознак і додає (за бажанням) бонус за відбиття.

    Ознаки (нормовані приблизно до [-1, 1]):
      0  м'яч на полі (0/1)
      1  x м'яча          2  y м'яча
      3  dx м'яча         4  dy м'яча
      5  y нашої ракетки  6  швидкість нашої ракетки
      7  y суперника      8  швидкість суперника
      9  y м'яча відносно нашої ракетки
      10 y м'яча відносно ракетки суперника
      11 прогноз y м'яча в момент, коли він долетить до нашої ракетки (відносно ракетки)
    """

    def __init__(self, env: gym.Env, hit_reward: float = 0.0):
        super().__init__(env)
        self.hit_reward = float(hit_reward)
        self.observation_space = gym.spaces.Box(-3.0, 3.0, (OBS_SIZE,), np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self._prev: np.ndarray | None = None
        self._prev_dx = 0
        self.hits = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _read(ram: np.ndarray) -> np.ndarray:
        return ram[[RAM_BALL_X, RAM_BALL_Y, RAM_PLAYER_Y, RAM_ENEMY_Y]].astype(np.int32)

    def _features(self, cur: np.ndarray) -> np.ndarray:
        bx, by, py, ey = cur
        prev = self._prev if self._prev is not None else cur
        visible = float(bx > 0 and by > 0)
        prev_visible = prev[0] > 0 and prev[1] > 0
        if visible and prev_visible:
            dx, dy = bx - prev[0], by - prev[1]
        else:
            dx, dy = 0, 0
        dpy, dey = py - prev[2], ey - prev[3]

        # Простий прогноз точки перетину з лінією нашої ракетки (з відбиттями від стін)
        if visible and dx > 0:
            steps = (PLAYER_X - bx) / dx
            ty = by + dy * steps
            lo, hi = 34.0, 194.0
            span = hi - lo
            t = (ty - lo) % (2 * span)
            ty = lo + (t if t <= span else 2 * span - t)
            pred = (ty - py) / 90.0
        else:
            pred = 0.0

        f = np.array(
            [
                visible,
                (bx - 130) / 70.0 * visible,
                (by - 120) / 90.0 * visible,
                dx / 8.0,
                dy / 8.0,
                (py - 120) / 90.0,
                dpy / 16.0,
                (ey - 120) / 90.0,
                dey / 16.0,
                (by - py) / 90.0 * visible,
                (by - ey) / 90.0 * visible,
                pred,
            ],
            dtype=np.float32,
        )
        return np.clip(f, -3.0, 3.0)

    # --------------------------------------------------------------------- API
    def reset(self, **kwargs):
        ram, info = self.env.reset(**kwargs)
        cur = self._read(ram)
        self._prev = None
        self._prev_dx = 0
        self.hits = 0
        obs = self._features(cur)
        self._prev = cur
        return obs, info

    def step(self, action):
        ram, reward, terminated, truncated, info = self.env.step(ACTIONS[int(action)])
        cur = self._read(ram)
        obs = self._features(cur)

        # Відбиття нашою ракеткою: м'яч летів праворуч і розвернувся біля нас
        prev = self._prev
        hit = False
        if prev is not None and prev[0] > 0 and cur[0] > 0:
            dx = int(cur[0] - prev[0])
            if self._prev_dx > 0 and dx < 0 and prev[0] > 160:
                self.hits += 1
                hit = True
            self._prev_dx = dx
        else:
            self._prev_dx = 0
        self._prev = cur
        shaped = self.hit_reward if hit else 0.0

        info = dict(info)
        info["game_reward"] = float(reward)
        info["hits"] = self.hits
        info["hit"] = float(hit)
        # Табло гри: справжні очки агента й суперника
        info["agent_points"] = int(ram[RAM_PLAYER_SCORE])
        info["cpu_points"] = int(ram[RAM_ENEMY_SCORE])
        return obs, float(reward) + shaped, terminated, truncated, info


def make_env(
    seed: int | None = None,
    hit_reward: float = 0.0,
    sticky: float = 0.25,
    render_mode: str | None = None,
) -> PongFeatures:
    """Створює ALE/Pong-v5 (frameskip=4, sticky actions 0.25 за замовчуванням)."""
    env = gym.make(
        "ALE/Pong-v5",
        obs_type="ram",
        repeat_action_probability=sticky,
        render_mode=render_mode,
    )
    env = PongFeatures(env, hit_reward=hit_reward)
    if seed is not None:
        env.action_space.seed(seed)
    return env
