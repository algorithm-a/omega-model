"""
check_omega_max.py
Проверка: какой максимум когерентности C может дать Omega-модель
БЕЗ RL, только за счёт самонастройки (enhance_simulation).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from omega_unified import OmegaUniverseEngine

# Запускаем несколько прогонов с разными seed
seeds = [0, 1, 2, 3, 4]
max_cycles = 30  # больше циклов, чем по умолчанию

print("=" * 72)
print("ПРОВЕРКА МАКСИМАЛЬНОЙ КОГЕРЕНТНОСТИ OMEGA-МОДЕЛИ")
print("=" * 72)
print(f"Прогонов: {len(seeds)}")
print(f"Циклов на прогон: {max_cycles}")
print(f"Без RL — только самонастройка (enhance_simulation)")
print()

all_max_c = []
all_max_gen = []

for seed in seeds:
    eng = OmegaUniverseEngine(initial_seed=seed)
    
    # Сохраняем все C за все циклы
    c_values = []
    
    for cycle in range(max_cycles):
        eng.big_bang(eng.current_state)
        for bid in sorted(eng.multiversal_sphere):
            eng.simulate_experience(bid)
        
        c = eng.global_coherence()
        c_values.append(c)
        
        if eng.check_omega_state():
            # коллапс
            eng.current_state = eng.coherence_collapse()
        else:
            eng.enhance_simulation()
    
    max_c = max(c_values)
    max_gen = eng.current_state.generation
    all_max_c.append(max_c)
    all_max_gen.append(max_gen)
    
    print(f"seed={seed}: max_C={max_c:.4f}, "
          f"final_gen={max_gen}, "
          f"final_branching={eng.branching_factor}, "
          f"final_depth={eng.max_depth}, "
          f"final_noise={eng.noise:.4f}")

print()
print("=" * 72)
print(f"ИТОГ:")
print(f"  Максимум C по всем прогонам: {max(all_max_c):.4f}")
print(f"  Среднее максимума C:         {sum(all_max_c)/len(all_max_c):.4f}")
print(f"  Средняя генерация:           {sum(all_max_gen)/len(all_max_gen):.2f}")
print("=" * 72)