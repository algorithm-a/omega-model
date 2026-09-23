# Резюме эксперимента v8.2

## Постановка задачи

Проверить, может ли PPO (RecurrentPPO) превзойти baselines
(random, frozen, constant) на среде Omega-Chaos-Control (easy).

## Что изменилось по сравнению с v8.1

1. `action_mode: params → scalar`
2. `theta = 0.2 * theta`
3. `TARGET_S: 6.0 → 7.5`
4. Убрано глобальное `OMEGA_THRESHOLD = suggested`

## Результаты (30 сидов)

| Сравнение | mean_diff | t | p | Вывод |
|---|---|---|---|---|
| trained vs random | +1.78 | +1.72 | 0.096 | ⚠️ на грани |
| trained vs frozen | +0.68 | +0.56 | 0.580 | ❌ не значимо |
| trained vs constant | +3.20 | +5.39 | <0.001 | ✅ значимо |

## Средние значения

| Условие | total_reward | final_C | final_S |
|---|---|---|---|
| constant_reward | 0.000 | 0.0949 | 6.67 |
| frozen_computed | 2.517 | 0.3506 | 9.78 |
| random_growth | 1.421 | 0.3025 | 8.69 |
| trained_growth | 3.202 | 0.3783 | 8.11 |

## Вывод

PPO превосходит random (на грани значимости) и constant (значимо),
но не превосходит frozen. Это означает, что среда уже даёт высокую
C без управления, и PPO не находит способ её улучшить.

## Что дальше

1. Увеличить TRAIN_STEPS до 50000
2. Попробовать векторизацию действий (action_mode = vector)
3. Увеличить SEEDS до 100