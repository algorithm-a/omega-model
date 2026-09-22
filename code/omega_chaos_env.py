"""
omega_chaos_env.py

Omega-Chaos-Control: бенчмарк для RL-управления хаотической динамикой.

Версия 0.1 — одна конфигурация (easy).
Основано на OmegaUniverseEngine из omega_unified.py.

Задача: удерживать когерентность C и энтропию S
в заданной области фазового пространства.

Reward:
- 0, если C и S внутри области [C_low, C_high] × [S_low, S_high]
- -расстояние до ближайшей границы области, если вне

Запуск для проверки:
    python omega_chaos_env.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Импортируем движок из основного файла
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_unified import OmegaUniverseEngine


# ============================================================
# КОНФИГУРАЦИИ СРЕДЫ
# ============================================================

CONFIGS = {
    "easy": {
        "branching_factor": 2,
        "max_depth": 2,
        "consciousness_density": 3,
        "decoherence_noise": 0.03,
        "episode_length": 30,
        "target_c_low": 0.15,
        "target_c_high": 0.35,
        "target_s_low": 5.0,
        "target_s_high": 7.0,
        "initial_seed": 0,
    },
    "medium": {
        "branching_factor": 3,
        "max_depth": 3,
        "consciousness_density": 4,
        "decoherence_noise": 0.05,
        "episode_length": 30,
        "target_c_low": 0.15,
        "target_c_high": 0.35,
        "target_s_low": 5.0,
        "target_s_high": 7.0,
        "initial_seed": 0,
    },
    "hard": {
        "branching_factor": 4,
        "max_depth": 4,
        "consciousness_density": 4,
        "decoherence_noise": 0.10,
        "episode_length": 30,
        "target_c_low": 0.15,
        "target_c_high": 0.35,
        "target_s_low": 5.0,
        "target_s_high": 7.0,
        "initial_seed": 0,
    },
    "chaotic": {
        "branching_factor": 3,
        "max_depth": 3,
        "consciousness_density": 4,
        "decoherence_noise": 0.20,
        "episode_length": 30,
        "target_c_low": 0.15,
        "target_c_high": 0.35,
        "target_s_low": 5.0,
        "target_s_high": 7.0,
        "initial_seed": 0,
    },
    "adversarial": {
        "branching_factor": 5,
        "max_depth": 2,
        "consciousness_density": 3,
        "decoherence_noise": 0.15,
        "episode_length": 30,
        "target_c_low": 0.15,
        "target_c_high": 0.35,
        "target_s_low": 5.0,
        "target_s_high": 7.0,
        "initial_seed": 0,
    },
}


# ============================================================
# СРЕДА
# ============================================================

class OmegaChaosEnv(gym.Env):
    """
    Среда для RL-управления хаотической динамикой Omega-модели.

    Наблюдения (7 значений):
        [C, S, x, y, z, w, v]
        где (x, y, z, w, v) — state_vector корневой ветви

    Действия (1 значение):
        скаляр в [-1, 1] — управляющий сигнал micro_action

    Reward:
        0, если C и S внутри области
        -расстояние до области, если вне
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_name: str = "easy",
        frame_stack: int = 1,
    ):
        super().__init__()

        if config_name not in CONFIGS:
            raise ValueError(
                f"Неизвестная конфигурация: {config_name}. "
                f"Доступные: {list(CONFIGS.keys())}"
            )

        self.config_name = config_name
        self.cfg = CONFIGS[config_name]
        self.frame_stack = max(1, int(frame_stack))

        # Пространство действий: скаляр в [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Наблюдение: 7 значений × frame_stack
        obs_dim = 7 * self.frame_stack
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Внутреннее состояние
        self.engine: Optional[OmegaUniverseEngine] = None
        self.frame_buffer: list[np.ndarray] = []
        self.steps = 0

    def _get_single_observation(self) -> np.ndarray:
        """Собирает одно наблюдение (7 значений)."""
        assert self.engine is not None

        root = self.engine.multiversal_sphere.get("root")
        if root is None:
            vector = np.zeros(5, dtype=np.float32)
        else:
            vector = np.asarray(root.state_vector, dtype=np.float32)

        return np.concatenate(
            [
                np.asarray(
                    [
                        self.engine.global_coherence(),
                        self.engine.shannon_entropy(),
                    ],
                    dtype=np.float32,
                ),
                vector,
            ]
        )

    def _get_observation(self) -> np.ndarray:
        """Собирает frame-stacked наблюдение."""
        current = self._get_single_observation()
        self.frame_buffer.append(current)
        if len(self.frame_buffer) > self.frame_stack:
            self.frame_buffer.pop(0)
        # Если буфер ещё не полный — дублируем первое наблюдение
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.insert(0, current.copy())
        return np.concatenate(self.frame_buffer)

    def _compute_reward(self, c: float, s: float) -> float:
        """
        Reward = 0, если внутри области; -расстояние, если вне.
        """
        c_low = self.cfg["target_c_low"]
        c_high = self.cfg["target_c_high"]
        s_low = self.cfg["target_s_low"]
        s_high = self.cfg["target_s_high"]

        c_dist = max(0.0, c_low - c, c - c_high)
        s_dist = max(0.0, s_low - s, s - s_high)
        return -(c_dist + s_dist)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        actual_seed = (
            self.cfg["initial_seed"]
            if seed is None
            else int(seed)
        )

        # Создаём движок с параметрами из конфига
        self.engine = OmegaUniverseEngine(initial_seed=actual_seed)
        self.engine.branching_factor = self.cfg["branching_factor"]
        self.engine.max_depth = self.cfg["max_depth"]
        self.engine.noise = self.cfg["decoherence_noise"]

        self.engine.big_bang(self.engine.current_state)
        for branch_id in sorted(self.engine.multiversal_sphere):
            self.engine.simulate_experience(branch_id)

        self.frame_buffer = []
        self.steps = 0

        # Первое наблюдение
        obs = self._get_observation()
        c = self.engine.global_coherence()
        s = self.engine.shannon_entropy()
        info = {
            "coherence": c,
            "entropy": s,
            "inside_c": self.cfg["target_c_low"] <= c <= self.cfg["target_c_high"],
            "inside_s": self.cfg["target_s_low"] <= s <= self.cfg["target_s_high"],
        }
        return obs, info

    def step(self, action):
        assert self.engine is not None

        action = np.asarray(action, dtype=np.float64).flatten()
        theta = float(np.clip(action[0], -1.0, 1.0))

        # Воздействие агента
        self.engine.micro_action(theta)

        # Один цикл симуляции
        self.engine.run_cycle(
            max_cycles=1,
            verbose=False,
            enhance_on_no_omega=False,
        )

        # Метрики
        c = self.engine.global_coherence()
        s = self.engine.shannon_entropy()

        reward = self._compute_reward(c, s)

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.cfg["episode_length"]

        obs = self._get_observation()

        info = {
            "coherence": c,
            "entropy": s,
            "inside_c": self.cfg["target_c_low"] <= c <= self.cfg["target_c_high"],
            "inside_s": self.cfg["target_s_low"] <= s <= self.cfg["target_s_high"],
        }

        return obs, float(reward), terminated, truncated, info


# ============================================================
# БЫСТРАЯ ПРОВЕРКА
# ============================================================

def quick_test():
    """Прогон одного эпизода со случайной политикой."""
    print("=" * 72)
    print("OMEGA-CHAOS-ENV: быстрый тест (конфигурация 'easy')")
    print("=" * 72)

    env = OmegaChaosEnv(config_name="easy", frame_stack=1)
    obs, _ = env.reset(seed=42)

    print(f"Наблюдение при reset: shape={obs.shape}, dtype={obs.dtype}")
    print(f"Первые 3 значения: {obs[:3]}")

    total_reward = 0.0
    rng = np.random.default_rng(42)

    print("\nЭпизод из 30 шагов со случайной политикой:")
    for step in range(30):
        action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step < 5 or step >= 27:
            print(
                f"  step={step:2d}: "
                f"action={action[0]:+.3f}, "
                f"reward={reward:+.4f}, "
                f"C={info['coherence']:.4f}, "
                f"S={info['entropy']:.4f}, "
                f"inside_C={info['inside_c']}, "
                f"inside_S={info['inside_s']}"
            )

        if terminated or truncated:
            break

    print(f"\nСуммарный reward: {total_reward:+.4f}")
    print(f"Длина эпизода: {env.steps}")


if __name__ == "__main__":
    quick_test()