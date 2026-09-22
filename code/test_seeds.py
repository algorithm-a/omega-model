"""test_seeds.py — проверка random policy на нескольких сидах."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omega_chaos_env import OmegaChaosEnv


def run_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    c_inside_count = 0
    s_inside_count = 0
    n_steps = 0

    for _ in range(30):
        action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        n_steps += 1
        if info["inside_c"]:
            c_inside_count += 1
        if info["inside_s"]:
            s_inside_count += 1
        if terminated or truncated:
            break

    return {
        "total_reward": total,
        "n_steps": n_steps,
        "c_inside_frac": c_inside_count / n_steps,
        "s_inside_frac": s_inside_count / n_steps,
        "final_c": info["coherence"],
        "final_s": info["entropy"],
    }


def main():
    print("=" * 72)
    print("RANDOM POLICY: 30 сидов")
    print("=" * 72)

    env = OmegaChaosEnv(config_name="easy")

    rewards = []
    for seed in range(30):
        r = run_episode(env, seed)
        rewards.append(r["total_reward"])
        print(
            f"seed={seed:2d}: "
            f"R={r['total_reward']:+8.4f}, "
            f"C_in={r['c_inside_frac']:.2f}, "
            f"S_in={r['s_inside_frac']:.2f}, "
            f"final_C={r['final_c']:.4f}, "
            f"final_S={r['final_s']:.4f}"
        )

    rewards = np.asarray(rewards)
    print("\n" + "=" * 72)
    print("СВОДКА")
    print("=" * 72)
    print(f"mean reward:  {rewards.mean():+.4f}")
    print(f"std reward:   {rewards.std():.4f}")
    print(f"min reward:   {rewards.min():+.4f}")
    print(f"max reward:   {rewards.max():+.4f}")


if __name__ == "__main__":
    main()