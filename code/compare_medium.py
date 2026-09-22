"""compare_medium.py — правильное сравнение PPO medium с baseline medium."""

import json
import os

import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results"
)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def main():
    baselines = load("baselines_medium.json")
    ppo = load("ppo_medium.json")

    random_r = np.asarray([r["total_reward"] for r in baselines["random"]])
    pid_r = np.asarray([r["total_reward"] for r in baselines["pid"]])
    ppo_r = np.asarray([r["total_reward"] for r in ppo])

    print("=" * 72)
    print("MEDIUM: правильное сравнение")
    print("=" * 72)
    print(f"random: mean={random_r.mean():+8.4f}, std={random_r.std():.4f}")
    print(f"PID:    mean={pid_r.mean():+8.4f}, std={pid_r.std():.4f}")
    print(f"PPO:    mean={ppo_r.mean():+8.4f}, std={ppo_r.std():.4f}")

    print("\nПарные t-тесты:")
    for name, baseline in [("random", random_r), ("PID", pid_r)]:
        t, p = stats.ttest_rel(ppo_r, baseline)
        diff = ppo_r.mean() - baseline.mean()
        if diff > 0 and p < 0.05:
            verdict = "PPO ЛУЧШЕ"
        elif diff < 0 and p < 0.05:
            verdict = "PPO ХУЖЕ"
        else:
            verdict = "разница не значима"
        print(
            f"  PPO vs {name:6s}: "
            f"mean_diff={diff:+8.4f}, t={t:+.3f}, p={p:.6f}  → {verdict}"
        )


if __name__ == "__main__":
    main()