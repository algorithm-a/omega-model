# branching-control-benchmark



Компактный вычислительный прототип Omega-модели: RL-агент (PPO) учится

управлять 5D-мультивселенной, максимизируя «жизненную» reward-функцию —

баланс когерентности и энтропии.



## Что это



Это **игрушечная вычислительная модель**, а не физическая теория.

Она формализует метафору Omega-точки Тейяра де Шардена и Типлера:

мультивселенная эволюционирует через циклы, в каждом из которых

растёт когерентность, а на пороге Omega происходит коллапс в новую

сингулярность.



Модель состоит из трёх частей:



1\. **Ядро (`OmegaUniverseEngine`)** — 5D-мультивселенная с ветвлением,

   «кваантами сознания» (математическими сущностями, не частицами)

   и циклами коллапса.

2\. **Среда Gym (`OmegaLifeEnv`)** — обёртка над ядром, совместимая

   с Gymnasium API.

3\. **RL-агент (PPO)** — из stable-baselines3, обучается управлять

   системой через скалярное действие.



## Как запустить



```bash

pip install "numpy<2" pandas gymnasium stable-baselines3 scikit-learn matplotlib scipy

python omega_unified.py

Эксперимент идёт \~2 часа (30 сидов × 7 условий). Результаты

складываются в experiment_output/.



Что показал эксперимент

Проведено 30 сидов × 7 условий = 210 эпизодов. Парные t-тесты по сидам:



Сравнение	mean_diff	t	p	Вывод

trained_order vs random_order	+1.68	+3.09	0.0043	✅ значимо

trained_life vs random_life	+5.80	+7.41	< 0.0001	✅ высокозначимо

trained_life vs shuffled_life	−0.44	−0.57	0.5764	❌ контроль не работает

trained_life vs constant_reward	−32.26	−47.73	< 0.0001	✅ среда активна

Главный вывод: PPO статистически значимо улучшает результат

относительно случайной политики на обеих reward-функциях, причём

life-reward даёт в 2.4 раза более сильный сигнал, чем order-reward.



Reward-функции

life — удержание целевой точки в пространстве (C, S):



python

reward = -abs(new_coherence - 0.3) - 0.1 * abs(new_entropy - 6.0)

order — максимизация прироста когерентности:



python

reward = dC * 100.0

Условия эксперимента

Условие	Reward	Policy	Что проверяет

frozen	order	frozen (a=0)	Что даёт неподвижная система

random_order	order	random	Baseline для order

random_life	life	random	Baseline для life

trained_order	order	trained (PPO)	Работает ли RL на order

trained_life	life	trained (PPO)	Работает ли RL на life

shuffled_life	life	trained	Контроль на задержку reward

constant_reward	life	no_step	Нулевой reward

Ограничения

Модель честно позиционируется как экспериментальная:



Не физическая. «5D» — пространство состояния, не измерение.

«ConsciousnessQuanta» — математическая сущность.



C(t) не растёт к порогу Omega. PPO улучшает reward в основном

за счёт энтропии S, а не когерентности C. Это видно на

results/trajectories.png.



micro_action скалярный. Агент действует одинаково на все ветви

мультивселенной. Ветве-специфичное действие не реализовано.



shuffled_life неинформативен. Задержка reward на 1 шаг слишком

мала, чтобы повлиять на политику.



mode_count шумит. sklearn выдаёт ConvergenceWarning при

вырожденных данных (например, в frozen). Метрика работает, но

требует фильтрации.



Структура проекта

text

omega_unified.py        — основной код эксперимента

omega_quicktest.py      — быстрая проверка micro_action

fix_tail.py             — вспомогательный скрипт (восстановление файла)

fix_indent.py           — вспомогательный скрипт (исправление отступов)



results/

  experiment_results.csv — все эпизоды (210 строк)

  ttest_results.csv      — парные t-тесты

  trajectories.npz       — сырые траектории C(t), S(t), reward(t)

  trajectories.png       — графики усреднённых траекторий

  profile.txt            — cProfile одного эпизода



docs/

  summary.md             — краткое резюме результатов

  model.md               — описание модели и математики

Связанные работы

Концептуальные основы:



Тейяр де Шарден, «Феномен человека» (Omega Point)



Типлер, «Физика бессмертия» (Omega Point cosmology)



Penrose & Hameroff, Orch-OR (квантовое сознание)



Похожие GitHub-проекты:



CIEL/Ω — Quantum Consciousness System



ψ Net∞ OMEGA v4.0



The Omega (loning) — Reverse-Engineering Reality's Source Code



Aura (youngbryan97) — IIT 4.0 + 10 теорий сознания



RL и теория информации:



Schulman et al., PPO (2017)



Tononi, Integrated Information Theory



Автор и лицензия

Прототип создан в 2026 году как экспериментальная модель для

исследования связи RL, когерентности и эмерджентного поведения.

Лицензия: MIT (или на усмотрение автора).
notepad docs\model.md
