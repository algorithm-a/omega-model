"""
probe_range.py
Замер достижимого диапазона C и S при разных стратегиях действий.

Запуск:
    D:\omega_backup\venv_omega\Scripts\python.exe D:\omega\probe_range.py
"""

import sys
import os

# Добавляем папку с omega_unified.py в путь импорта
sys.path.insert(0, r"D:\omega")

import numpy as np
from omega_unified import OmegaUniverseEngine


def run_probe(name, action_fn, n_steps=50, seed=42):
    """Прогоняет эпизод с заданной стратегией действий."""
    engine = OmegaUniverseEngine(initial_seed=seed)
    engine.big_bang(engine.current_state)

    # Начальная инициализация — как в reset() у OmegaLifeEnv
    for branch_id in sorted(engine.multiversal_sphere):
        engine.simulate_experience(branch_id)

    C0 = engine.global_coherence()
    S0 = engine.shannon_entropy()

    # Прогон эпизода
    trajectory = []
    for step in range(n_steps):
        action = action_fn(step, n_steps)
        engine.micro_action(action)
        engine.run_cycle(max_cycles=1, verbose=False, enhance_on_no_omega=False)

        C = engine.global_coherence()
        S = engine.shannon_entropy()
        trajectory.append((C, S))

    C_final, S_final = trajectory[-1]
    C_max = max(c for c, s in trajectory)
    C_min = min(c for c, s in trajectory)
    S_max = max(s for c, s in trajectory)
    S_min = min(s for c, s in trajectory)

    print(f"\n=== {name} ===")
    print(f"  C: start={C0:.4f}, final={C_final:.4f}, min={C_min:.4f}, max={C_max:.4f}")
    print(f"  S: start={S0:.4f}, final={S_final:.4f}, min={S_min:.4f}, max={S_max:.4f}")

    return {
        "name": name,
        "C0": C0, "S0": S0,
        "C_final": C_final, "S_final": S_final,
        "C_min": C_min, "C_max": C_max,
        "S_min": S_min, "S_max": S_max,
    }


def main():
    print("=" * 70)
    print("ЗАМЕР ДОСТИЖИМОГО ДИАПАЗОНА C и S")
    print("=" * 70)

    results = []

    # 1. Бездействие — action = 0
    results.append(run_probe(
        "no_action (action=0.0)",
        lambda step, n: 0.0,
    ))

    # 2. Случайные действия
    rng = np.random.default_rng(42)
    random_actions = rng.uniform(-1.0, 1.0, size=50)
    results.append(run_probe(
        "random (uniform [-1, 1])",
        lambda step, n: random_actions[step],
    ))

    # 3. Максимум вправо — action = +1.0
    results.append(run_probe(
        "max_action (action=+1.0)",
        lambda step, n: 1.0,
    ))

    # 4. Максимум влево — action = -1.0
    results.append(run_probe(
        "min_action (action=-1.0)",
        lambda step, n: -1.0,
    ))

    # 5. Знакопеременное — action = +1, -1, +1, -1...
    results.append(run_probe(
        "alternating (+1, -1, ...)",
        lambda step, n: 1.0 if step % 2 == 0 else -1.0,
    ))

    # Сводка
    print("\n" + "=" * 70)
    print("СВОДКА")
    print("=" * 70)
    print(f"{'Режим':<30} {'C_final':>10} {'S_final':>10} {'C_max':>10} {'S_min':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['C_final']:>10.4f} {r['S_final']:>10.4f} "
              f"{r['C_max']:>10.4f} {r['S_min']:>10.4f}")

    # Рекомендации по целям
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ ПО TARGET_C и TARGET_S")
    print("=" * 70)

    C_vals = [r["C_final"] for r in results]
    S_vals = [r["S_final"] for r in results]

    C_mean = float(np.mean(C_vals))
    S_mean = float(np.mean(S_vals))
    C_max = max(r["C_max"] for r in results)
    S_min = min(r["S_min"] for r in results)

    print(f"Среднее C_final по всем режимам: {C_mean:.4f}")
    print(f"Среднее S_final по всем режимам: {S_mean:.4f}")
    print(f"Максимум C по всем режимам:      {C_max:.4f}")
    print(f"Минимум S по всем режимам:       {S_min:.4f}")
    print()
    print("Предложение:")
    print(f"  TARGET_C = {C_mean + 0.3 * (C_max - C_mean):.3f}  "
          f"(между средним и максимумом)")
    print(f"  TARGET_S = {S_mean - 0.3 * (S_mean - S_min):.3f}  "
          f"(между средним и минимумом)")
    print()
    print("Это цели, которые агент, скорее всего, сможет достичь,")
    print("но которые требуют усилий (выше/ниже среднего).")


if __name__ == "__main__":
    main()