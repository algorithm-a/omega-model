"""
omega_quicktest.py

Быстрая проверка: даёт ли новая micro_action различимый сигнал в reward.

Не меняет omega_unified.py. Через monkey-patch подменяет метод
OmegaUniverseEngine.micro_action и прогоняет только два условия
на одном seed.
"""

import numpy as np
import math

import omega_unified as oa
from omega_unified import (
    OmegaLifeEnvControlled,
    _make_env_factory,
    make_policy,
    EPISODE_LENGTH,
)


# ============================================================
# НОВАЯ micro_action
# ============================================================

def new_micro_action(self, action: float) -> None:
    """
    Направленное воздействие: сдвигаем все компоненты вектора
    в сторону доминирующего знака среднего.
    """
    a = float(np.clip(action, -1.0, 1.0))
    for branch in self.multiversal_sphere.values():
        sv = np.asarray(branch.state_vector, dtype=float)
        mean_sv = float(sv.mean())
        bias = a * 0.7 * math.copysign(1.0, mean_sv)
        distorted = sv + bias
        norm = float(np.linalg.norm(distorted)) or 1.0
        branch.state_vector = tuple(float(x) for x in distorted / norm)


def old_micro_action(self, action: float) -> None:
    for branch in self.multiversal_sphere.values():
        sv = np.asarray(branch.state_vector, dtype=float)
        theta = float(action) * math.pi
        distorted = tuple(
            float(x * math.cos(theta) - math.sin(theta) * (1.0 - abs(x)))
            for x in sv
        )
        norm = math.sqrt(sum(x * x for x in distorted)) or 1.0
        branch.state_vector = tuple(x / norm for x in distorted)


# ============================================================
# ЗАПУСК
# ============================================================

def run_one(policy_type, reward_mode, seed, mode_label):
    env_factory = _make_env_factory(
        reward_mode=reward_mode,
        seed=seed,
        controls={},
    )
    eval_env = env_factory()
    obs, _ = eval_env.reset(seed=seed)
    policy = make_policy(env_factory, policy_type, seed)

    total_reward = 0.0
    c_vals = []
    actions = []
    for _ in range(EPISODE_LENGTH):
        action = policy(obs)
        actions.append(float(np.asarray(action).flatten()[0]))
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)
        c_vals.append(float(info["coherence"]))
        if terminated or truncated:
            break
    eval_env.close()
    print(f"  [{mode_label}] policy_type={policy_type}")
    print(f"  [{mode_label}] первые 10 actions: {[f'{a:+.4f}' for a in actions[:10]]}")
    print(f"  [{mode_label}] obs (первые 3): {obs[:3]}")
    return {
        "label": mode_label,
        "policy": policy_type,
        "reward": total_reward,
        "final_C": c_vals[-1] if c_vals else 0.0,
        "C_std": float(np.std(c_vals)) if c_vals else 0.0,
    }


def main():
    print("=" * 72)
    print("БЫСТРЫЙ ТЕСТ micro_action")
    print("=" * 72)

    # Уменьшаем бюджет обучения, чтобы тест шёл минуты, а не часы.
    oa.TRAIN_STEPS = 3000
    oa.EPISODE_LENGTH = 30
    print(f"[CFG] TRAIN_STEPS={oa.TRAIN_STEPS}, EPISODE_LENGTH={oa.EPISODE_LENGTH}")

    seed = 0

    # ---- Текущая micro_action ----
    print("\n--- СТАРАЯ micro_action ---")
    oa.OmegaUniverseEngine.micro_action = old_micro_action

    r_rand_old = run_one("random", "life", seed, "random_life (old)")
    r_train_old = run_one("trained", "life", seed, "trained_life (old)")
    print(
        f"random_life (old):  R={r_rand_old['reward']:+.4f}, "
        f"C_final={r_rand_old['final_C']:.4f}, C_std={r_rand_old['C_std']:.4f}"
    )
    print(
        f"trained_life (old): R={r_train_old['reward']:+.4f}, "
        f"C_final={r_train_old['final_C']:.4f}, C_std={r_train_old['C_std']:.4f}"
    )
    diff_old = r_train_old["reward"] - r_rand_old["reward"]
    print(f"  разница R: {diff_old:+.4f}")

    # ---- Новая micro_action ----
    print("\n--- НОВАЯ micro_action ---")
    oa.OmegaUniverseEngine.micro_action = new_micro_action

    r_rand_new = run_one("random", "life", seed, "random_life (new)")
    r_train_new = run_one("trained", "life", seed, "trained_life (new)")
    print(
        f"random_life (new):  R={r_rand_new['reward']:+.4f}, "
        f"C_final={r_rand_new['final_C']:.4f}, C_std={r_rand_new['C_std']:.4f}"
    )
    print(
        f"trained_life (new): R={r_train_new['reward']:+.4f}, "
        f"C_final={r_train_new['final_C']:.4f}, C_std={r_train_new['C_std']:.4f}"
    )
    diff_new = r_train_new["reward"] - r_rand_new["reward"]
    print(f"  разница R: {diff_new:+.4f}")

    # ---- Интерпретация ----
    print("\n" + "=" * 72)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 72)
    print(f"разница trained-random:")
    print(f"  старая micro_action: {diff_old:+.4f}")
    print(f"  новая  micro_action: {diff_new:+.4f}")

    if abs(diff_new) < 1e-6 and abs(diff_old) < 1e-6:
        print("=> ВЫВОД: обе версии дают нулевую разницу. PPO не обучается.")
        print("   Проблема глубже, чем micro_action.")
    elif abs(diff_new) > abs(diff_old) + 1e-3:
        print("=> ВЫВОД: новая micro_action даёт различимый сигнал.")
        print("   Можно править основной файл и запускать полный прогон.")
    else:
        print("=> ВЫВОД: разница есть, но не увеличилась.")
        print("   Нужно смотреть, что именно делает PPO — возможно, ещё мало шагов.")

    # Сравнение старой и новой по абсолютной величине
    print(f"\nАбсолютный R:")
    print(f"  старая random:  {r_rand_old['reward']:+.4f}")
    print(f"  новая  random:  {r_rand_new['reward']:+.4f}")
    print(f"  старая trained: {r_train_old['reward']:+.4f}")
    print(f"  новая  trained: {r_train_new['reward']:+.4f}")


if __name__ == "__main__":
    main()