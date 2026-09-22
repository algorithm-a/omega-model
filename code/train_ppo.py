"""
train_ppo.py

Обучение PPO на OmegaChaosEnv (конфигурация easy).

1. Создаёт среду.
2. Обучает PPO на 50000 шагов.
3. Сохраняет модель.
4. Оценивает на 30 сидах.
5. Сравнивает с baseline'ами (random, PID).
6. Сохраняет результаты в JSON + CSV.

Запуск:
    python train_ppo.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_chaos_env import OmegaChaosEnv
from baselines import RandomPolicy, PIDPolicy, run_episode


# ============================================================
# КОНФИГ
# ============================================================

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="easy", choices=["easy", "medium", "hard", "chaotic", "adversarial"])
args, _ = parser.parse_known_args()
CONFIG_NAME = args.config
TRAIN_STEPS = 50_000
SEEDS = list(range(30))
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results"
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# ОБЁРТКА PPO
# ============================================================

class PPOPolicy:
    """Обёртка PPO, совместимая с интерфейсом run_episode."""
    name = "ppo"

    def __init__(self, model):
        self.model = model

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).flatten()

    def reset(self):
        pass


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train():
    print("=" * 72)
    print(f"TRAIN PPO: config='{CONFIG_NAME}', steps={TRAIN_STEPS}")
    print("=" * 72)

    env = Monitor(OmegaChaosEnv(config_name=CONFIG_NAME))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=0,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
    )

    t0 = time.time()
    model.learn(total_timesteps=TRAIN_STEPS)
    t1 = time.time()

    print(f"\nОбучение заняло {t1 - t0:.1f} секунд")

    model_path = os.path.join(MODELS_DIR, f"ppo_{CONFIG_NAME}.zip")
    model.save(model_path)
    print(f"Модель сохранена: {model_path}")

    return model


# ============================================================
# ОЦЕНКА
# ============================================================

def evaluate(model, config_name: str, seeds: list[int]) -> list[dict]:
    print("\n" + "=" * 72)
    print(f"EVAL PPO: config='{config_name}', {len(seeds)} сидов")
    print("=" * 72)

    env = OmegaChaosEnv(config_name=config_name)
    policy = PPOPolicy(model)

    results = []
    for seed in seeds:
        r = run_episode(env, policy, seed)
        r["seed"] = seed
        results.append(r)
        print(
            f"  seed={seed:2d}: "
            f"R={r['total_reward']:+8.4f}, "
            f"C_in={r['c_inside_frac']:.2f}, "
            f"S_in={r['s_inside_frac']:.2f}"
        )
    return results


# ============================================================
# СРАВНЕНИЕ
# ============================================================

def compare(ppo_results, baselines_path):
    from scipy import stats

    with open(baselines_path) as f:
        baselines = json.load(f)

    random_rewards = np.asarray(
        [r["total_reward"] for r in baselines["random"]]
    )
    pid_rewards = np.asarray(
        [r["total_reward"] for r in baselines["pid"]]
    )
    ppo_rewards = np.asarray(
        [r["total_reward"] for r in ppo_results]
    )

    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(
        f"random: mean={random_rewards.mean():+8.4f}, "
        f"std={random_rewards.std():.4f}"
    )
    print(
        f"PID:    mean={pid_rewards.mean():+8.4f}, "
        f"std={pid_rewards.std():.4f}"
    )
    print(
        f"PPO:    mean={ppo_rewards.mean():+8.4f}, "
        f"std={ppo_rewards.std():.4f}"
    )

    print("\nПарные t-тесты (PPO vs baseline):")
    for name, baseline in [("random", random_rewards), ("PID", pid_rewards)]:
        t, p = stats.ttest_rel(ppo_rewards, baseline)
        diff = ppo_rewards.mean() - baseline.mean()
        print(
            f"  PPO vs {name:6s}: "
            f"mean_diff={diff:+8.4f}, t={t:+.3f}, p={p:.4f}"
        )
        if p < 0.05 and diff > 0:
            print(f"    → PPO значимо ЛУЧШЕ {name}")
        elif p < 0.05 and diff < 0:
            print(f"    → PPO значимо ХУЖЕ {name}")
        else:
            print(f"    → разница не значима")


# ============================================================
# MAIN
# ============================================================

def main():
    # Обучаем
    model = train()

    # Оцениваем
    ppo_results = evaluate(model, CONFIG_NAME, SEEDS)

    # Сохраняем PPO результаты
    ppo_path = os.path.join(RESULTS_DIR, f"ppo_{CONFIG_NAME}.json")
    with open(ppo_path, "w") as f:
        json.dump(ppo_results, f, indent=2)
    print(f"\nPPO результаты сохранены: {ppo_path}")

    # Сравниваем с baseline
    baselines_path = os.path.join(
        RESULTS_DIR, f"baselines_{CONFIG_NAME}.json"
    )
    if os.path.exists(baselines_path):
        compare(ppo_results, baselines_path)
    else:
        print(f"\nBaseline файл не найден: {baselines_path}")
        print("Сначала запустите сохранение baseline'ов.")


if __name__ == "__main__":
    main()