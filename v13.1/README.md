# Omega-Model v13.1 — Multi-Agent (role-based, shared action)

**Коммит:** 8e4382e (обновляется)
**Дата:** 29.09.2026
**Статус:** паритет с MCTS, отдельные сиды превосходят v13.0

---

## Что нового vs v13.0

- **4 роли:** EXPLORER / STABILIZER / OPTIMIZER / OBSERVER
- **Динамическое назначение** роли каждой ветви (по энергии, расстоянию до среднего, локальной когерентности)
- **Shared action:** одно 5D-действие применяется ко всем ролям (вместо 20D per-role)
- **Ablation-флаг** `--no-role-features`: роли-фичи в наблюдении можно отключить
- **Роль-специфичные награды** `--use-role-rewards`
- **Штраф за плохое состояние** `--bad-state-penalty`

При `--multi-agent` OFF — поведение идентично v13.0.

---

## Ключевой результат

**Финальный прогон (10 сидов × 50k):**

| Метод | C_mean | std | медиана |
|---|---|---|---|
| **PPO (v13.1)** | **0.5209** | **0.0460** | **0.5250** |
| MCTS | 0.5278 | 0.0326 | 0.526 |
| random | 0.1704 | 0.0449 | 0.165 |

**Медиана PPO ≈ медиана MCTS — паритет.**
**4 из 10 сидов > v13.0 (0.542).**
**Max 0.592 — рекорд проекта.**

> Ablation A на 3 сидах дал 0.5634 — статистический выброс.
> На 10 сидах: 0.5209 — паритет с MCTS, не превосходство.

---

## Сравнение с историей

| Версия | C_mean | std |
|---|---|---|
| v13.0 single-agent | 0.5424 | 0.043 |
| v13.1 MARL (20D action) | 0.3128 | 0.035 |
| v13.1 shared action (роли-фичи ON, 10 сидов) | 0.5181 | 0.0687 |
| **v13.1 shared + no-role-features (10 сидов)** | **0.5209** | **0.0460** |
| v13.1 shared + no-role-features (3 сида) | 0.5634 | 0.0268 |
| MCTS | 0.5278 | 0.0326 |

**Реальность (10 сидов):**
- C_mean: −4% от v13.0 (0.521 vs 0.542)
- std: +7% (0.046 vs 0.043)
- Медиана: 0.525 ≈ MCTS (0.526)
- Max: 0.592 — выше v13.0 (0.542)

---

## Что сработало

| Фича | Эффект |
|---|---|
| **Shared action (5D вместо 20D)** | 0.313 → 0.518 — критично |
| **`--no-role-features`** | 0.5181 → 0.5209 (10 сидов) — небольшое улучшение |

## Что не сработало

| Фича | Эффект |
|---|---|
| 20D MARL (per-role actions) | C_mean 0.313 — провал |
| role-фичи в наблюдении (8D) | Шум, повышает std (0.069 vs 0.046) |
| `--bad-state-penalty 100.0` | Давит обучение (в тесте C 0.36 → 0.33) |
| `--use-role-rewards` | Не тестировался на полном прогоне |

---

## Главный вывод

**Multi-agent через роли не даёт системного выигрыша над single-agent (v13.0) и MCTS.**

- Медиана ≈ MCTS (0.525 vs 0.526)
- Среднее ниже v13.0 (0.521 vs 0.542)
- Отдельные сиды превосходят v13.0 (4 из 10)
- Max 0.592 — лучший результат проекта

**Роли как числа в наблюдении — вредят** (std растёт с 0.046 до 0.069).
**Роли как логика — нейтральны** (shared action без role-фич ≈ shared action с role-фичами).

---

## Команды

```cmd
cd /d D:\omega-model
.venv\Scripts\activate
cd v13.1

:: Smoke
python omega_v13_1.py --mode smoke

:: Финальный прогон 10 сидов
python omega_v13_1.py --mode train --branches 40 --seeds 10 --train-steps 50000 ^
  --complexity c --reward coherence --merge-split --event-features ^
  --event-penalty 0.5 --merge-split-every 1 --delta-c-bonus 10.0 ^
  --multi-agent --shared-action --no-role-features --output v13.1_final_10s

:: Ablation A (3 сида)
python omega_v13_1.py --mode train --branches 40 --seeds 3 --train-steps 50000 ^
  --complexity c --reward coherence --merge-split --event-features ^
  --event-penalty 0.5 --merge-split-every 1 --delta-c-bonus 10.0 ^
  --multi-agent --shared-action --no-role-features --output v13.1_abl_norolefeat