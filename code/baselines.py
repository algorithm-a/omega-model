"""
baselines.py

Baseline политики для Omega-Chaos-Env:
- RandomPolicy: случайное действие
- PIDPolicy: PID-контроллер для удержания C и S в области

Общая функция run_episode для прогона одного эпизода.

Запуск для проверки:
    python baselines.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_chaos_env import OmegaChaosEnv


# ============================================================
# ПОЛИТИКИ
# ============================================================

class RandomPolicy:
    """Случайное действие в [-1, 1]."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)

    def reset(self):
        pass


class PIDPolicy:
    """
    PID-контроллер для удержания C и S в области.

    Логика:
    - Считаем ошибки: err_C = C - target_C, err_S = S - target_S
    - target_C = середина [c_low, c_high]
    - target_S = середина [s_low, s_high]
    - action = -(kp_C * err_C + kd_C * d_err_C + kp_S * err_S * w_S)

    Знак минус: если C выше цели, мы хотим отрицательное действие,
    чтобы «сбить» когерентность. Если ниже — положительное.

    kp, kd — подбираются эмпирически.
    """

    name = "pid"

    def __init__(
        self,
        cfg: dict,
        kp_c: float = 5.0,
        kd_c: float = 1.0,
        kp_s: float = 5.0,
        kd_s: float = 1.0,
    ):
        self.cfg = cfg
        self.kp_c = kp_c
        self.kd_c = kd_c
        self.kp_s = kp_s
        self.kd_s = kd_s

        self.target_c = 0.5 * (cfg["target_c_low"] + cfg["target_c_high"])
        self.target_s = 0.5 * (cfg["target_s_low"] + cfg["target_s_high"])

        self.prev_err_c = 0.0
        self.prev_err_s = 0.0

    def reset(self):
        self.prev_err_c = 0.0
        self.prev_err_s = 0.0

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        c = info["coherence"]
        s = info["entropy"]

        # Нормализуем ошибки по ширине диапазонов
        c_range = self.cfg["target_c_high"] - self.cfg["target_c_low"]
        s_range = self.cfg["target_s_high"] - self.cfg["target_s_low"]

        err_c = (c - self.target_c) / max(c_range, 1e-6)
        err_s = (s - self.target_s) / max(s_range, 1e-6)

        d_err_c = err_c - self.prev_err_c
        d_err_s = err_s - self.prev_err_s

        self.prev_err_c = err_c
        self.prev_err_s = err_s

        # Управление: если C выше цели, давить вниз; если ниже — поднимать
        u_c = self.kp_c * err_c + self.kd_c * d_err_c
        u_s = self.kp_s * err_s + self.kd_s * d_err_s

        # action = -u (обратная связь)
        action_val = -u_c - u_s
        action_val = float(np.clip(action_val, -1.0, 1.0))

        return np.asarray([action_val], dtype=np.float32)


# ============================================================
# ПРОГОН ЭПИЗОДА
# ============================================================

def run_episode(env: OmegaChaosEnv, policy, seed: int) -> dict:
    """
    Прогон одного эпизода с заданной политикой.

    Возвращает dict с метриками.
    """
    obs, info = env.reset(seed=seed)
    policy.reset()

    total_reward = 0.0
    c_inside_count = 0
    s_inside_count = 0
    n_steps = 0
    actions = []
    c_values = []
    s_values = []

    for _ in range(env.cfg["episode_length"]):
        action = policy.act(obs, info)
        actions.append(float(action[0]))

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        n_steps += 1
        c_values.append(info["coherence"])
        s_values.append(info["entropy"])

        if info["inside_c"]:
            c_inside_count += 1
        if info["inside_s"]:
            s_inside_count += 1

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "n_steps": n_steps,
        "c_inside_frac": c_inside_count / max(n_steps, 1),
        "s_inside_frac": s_inside_count / max(n_steps, 1),
        "final_c": info["coherence"],
        "final_s": info["entropy"],
        "mean_c": float(np.mean(c_values)),
        "mean_s": float(np.mean(s_values)),
        "std_action": float(np.std(actions)),
    }


def run_many(
    env: OmegaChaosEnv,
    policy,
    seeds: range,
    verbose: bool = True,
) -> list[dict]:
    """Прогон на нескольких сидах."""
    results = []
    for seed in seeds:
        r = run_episode(env, policy, seed)
        r["seed"] = seed
        results.append(r)

        if verbose:
            print(
                f"  seed={seed:2d}: "
                f"R={r['total_reward']:+8.4f}, "
                f"C_in={r['c_inside_frac']:.2f}, "
                f"S_in={r['s_inside_frac']:.2f}"
            )
    return results


# ============================================================
# ПРОВЕРКА
# ============================================================

def main():
    print("=" * 72)
    print("BASELINES: random vs PID (конфигурация 'easy', 30 сидов)")
    print("=" * 72)

    env = OmegaChaosEnv(config_name="easy")
    cfg = env.cfg

    # --- Random ---
    print("\n--- RANDOM ---")
    random_policy = RandomPolicy(seed=0)
    random_results = run_many(env, random_policy, range(30))
    random_rewards = np.asarray([r["total_reward"] for r in random_results])

    # --- PID ---
    print("\n--- PID ---")
    pid_policy = PIDPolicy(cfg, kp_c=5.0, kd_c=1.0, kp_s=5.0, kd_s=1.0)
    pid_results = run_many(env, pid_policy, range(30))
    pid_rewards = np.asarray([r["total_reward"] for r in pid_results])

    # --- Сводка ---
    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(f"random: mean={random_rewards.mean():+.4f}, std={random_rewards.std():.4f}")
    print(f"PID:    mean={pid_rewards.mean():+.4f}, std={pid_rewards.std():.4f}")

    from scipy import stats
    t, p = stats.ttest_rel(pid_rewards, random_rewards)
    diff = pid_rewards.mean() - random_rewards.mean()
    print(f"\nПарный t-тест (PID vs random):")
    print(f"  mean_diff = {diff:+.4f}")
    print(f"  t = {t:+.3f}, p = {p:.4f}")
    if p < 0.05 and diff > 0:
        print("  → PID значимо ЛУЧШЕ random")
    elif p < 0.05 and diff < 0:
        print("  → PID значимо ХУЖЕ random")
    else:
        print("  → разница не значима")


if __name__ == "__main__":
    main()