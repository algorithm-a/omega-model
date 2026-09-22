"""
run_all_baselines.py

Прогоняет random и PID на всех 5 конфигурациях.
Сохраняет результаты в results/baselines_<config>.json.

Запуск:
    python run_all_baselines.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_chaos_env import OmegaChaosEnv, CONFIGS
from baselines import RandomPolicy, PIDPolicy, run_many


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = list(range(30))

# Коэффициенты PID, отобранные ранее
PID_KWARGS = dict(kp_c=5.0, kd_c=1.0, kp_s=5.0, kd_s=1.0)


def main():
    print("=" * 72)
    print("BASELINES для всех 5 конфигураций")
    print("=" * 72)

    summary = {}

    for config_name in CONFIGS.keys():
        print(f"\n--- config: {config_name} ---")

        env = OmegaChaosEnv(config_name=config_name)
        cfg = env.cfg

        # Random
        t0 = time.time()
        random_results = run_many(
            env, RandomPolicy(seed=0), SEEDS, verbose=False
        )
        random_rewards = np.asarray(
            [r["total_reward"] for r in random_results]
        )

        # PID
        pid_results = run_many(
            env, PIDPolicy(cfg, **PID_KWARGS), SEEDS, verbose=False
        )
        pid_rewards = np.asarray(
            [r["total_reward"] for r in pid_results]
        )

        t1 = time.time()

        print(
            f"  random: mean={random_rewards.mean():+8.4f}, "
            f"std={random_rewards.std():.4f}"
        )
        print(
            f"  PID:    mean={pid_rewards.mean():+8.4f}, "
            f"std={pid_rewards.std():.4f}"
        )
        print(f"  время: {t1 - t0:.1f} сек")

        # Сохраняем
        out_path = os.path.join(
            RESULTS_DIR, f"baselines_{config_name}.json"
        )
        with open(out_path, "w") as f:
            json.dump(
                {"random": random_results, "pid": pid_results},
                f, indent=2,
            )
        print(f"  сохранено: {out_path}")

        summary[config_name] = {
            "random_mean": float(random_rewards.mean()),
            "random_std": float(random_rewards.std()),
            "pid_mean": float(pid_rewards.mean()),
            "pid_std": float(pid_rewards.std()),
        }

    # Сводная таблица
    print("\n" + "=" * 72)
    print("СВОДНАЯ ТАБЛИЦА BASELINES")
    print("=" * 72)
    print(f"{'config':>12} | {'random mean':>12} | {'PID mean':>10}")
    print("-" * 42)
    for name, s in summary.items():
        print(
            f"{name:>12} | {s['random_mean']:+12.4f} | "
            f"{s['pid_mean']:+10.4f}"
        )

    # Сохраняем сводку
    summary_path = os.path.join(RESULTS_DIR, "baselines_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nСводка сохранена: {summary_path}")


if __name__ == "__main__":
    main()