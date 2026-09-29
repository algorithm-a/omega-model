"""
Omega v13.4 — EVA (Evolving Agents + Branches).

Два уровня эволюции:
- Уровень 1: ВЕТВИ — 10 ветвей у каждого агента, мутируют внутри группы
- Уровень 2: АГЕНТЫ — 4 PPO, конкурируют, смещают Manager'а, умирают

Ключевые механики:
- 4 агента × 10 ветвей = 40 ветвей
- Агент выдаёт 5D action → применяется ко всем его 10 ветвям
- SPLIT/MERGE внутри группы (изоляция)
- Оценка агента = C его 10 ветвей (не глобальная)
- Турнирная таблица: 1-е ×1.0, 2-е ×0.6, 3-е ×0.3, 4-е ×0.0
- Смерть: 4-е место → агент + его 10 ветвей умирают
- Рождение: клон случайного (Manager/Worker 50/50) + мутация σ=0.01

Уроки из предыдущих версий:
- Mode НЕ в obs (шум, v13.2-A)
- Shared action 5D (не 20D, v13.1)
- σ=0.01 (не 0.05, v13.3)
- Наследование C (среднее Manager/умершего, v13.3)

Run:
    python eva.py --mode smoke
    python eva.py --mode train --generations 10 --agents 4 --branches-per-agent 10 \
        --seeds 3 --episodes-per-gen 3 --train-steps-per-gen 500 --sigma 0.01 \
        --output v13.4_results
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from omega_v13_1 import (  # noqa: E402
    OmegaConfig,
    OmegaV131Env,
    COMPLEXITY_PRESETS,
    apply_preset,
    train_ppo,
    EPS,
    STATE_DIM,
)

try:
    from stable_baselines3 import PPO
    import torch
except ImportError:
    PPO = None
    torch = None


# ============================================================
# BRANCH — ветвь (привязана к агенту)
# ============================================================

@dataclass
class Branch:
    """Ветвь — вектор 5D, принадлежит агенту.

    id: уникальный (например, "A1_b0", "A1_b1", ...)
    state: 5D вектор на сфере
    energy: 0..1.5
    coherence: локальная когерентность
    age: сколько шагов живёт
    """
    id: str
    state: np.ndarray
    energy: float = 0.5
    coherence: float = 0.0
    age: int = 0

    def clone(self) -> "Branch":
        return Branch(
            id=self.id,
            state=self.state.copy(),
            energy=self.energy,
            coherence=self.coherence,
            age=self.age,
        )


def mutate_branch_state(state: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Мутация state ветви: state + N(0, σ), нормировка на сферу."""
    x = state + rng.normal(0.0, sigma, size=STATE_DIM)
    n = float(np.linalg.norm(x))
    return x / (n if n > EPS else 1.0)


# ============================================================
# AGENT — PPO + 10 ветвей + метаданные
# ============================================================

@dataclass
class Agent:
    """Агент = PPO-политика + группа ветвей.

    id: "A1" / "A2" / ... / "G5" / ...
    role: "manager" / "worker" / "dead"
    model: PPO
    branches: список Branch (жёстко 10)
    points: накопленные очки (для турнирной таблицы)
    last_rank: 1..4 (место в турнирной таблице)
    last_c: последняя оценка C (по своим ветвям)
    generation: в каком поколении рождён
    parent: id родителя
    """
    id: str
    role: str = "worker"
    model: Optional[object] = None
    branches: list = field(default_factory=list)
    points: float = 0.0
    last_rank: int = 0
    last_c: float = 0.0
    generation: int = 0
    parent: Optional[str] = None
    c_history: list = field(default_factory=list)

    def n_branches(self) -> int:
        return len(self.branches)

    def states(self) -> np.ndarray:
        """Матрица состояний всех ветвей агента (n × 5)."""
        if not self.branches:
            return np.zeros((0, STATE_DIM), dtype=np.float64)
        return np.stack([b.state for b in self.branches])

    def coherence(self) -> float:
        """C агента = normalized_coherence его ветвей."""
        states = self.states()
        if len(states) == 0:
            return 0.0
        total = states.sum(axis=0)
        denom = len(states) * float(np.sum(states * states))
        if denom <= EPS:
            return 0.0
        raw = float(np.dot(total, total) / denom)
        return float(np.clip(raw, 0.0, 1.0))

    def add_points(self, c: float, weight: float) -> None:
        self.points += float(c) * float(weight)

    def clone_with_mutation(self, new_id: str, sigma: float = 0.01, rng=None) -> "Agent":
        """Клон агента: PPO + ветви с мутацией.

        НЕ наследует C автоматически — родитель передаёт last_c через
        replace_dead (там вычисляется среднее).
        """
        if self.model is None:
            raise RuntimeError(f"Agent {self.id} has no model to clone")
        if rng is None:
            rng = np.random.default_rng()

        new_model = _clone_ppo_with_noise(self.model, sigma=sigma)
        new_branches = []
        for b in self.branches:
            nb = b.clone()
            nb.energy = b.energy
            nb.age = 0
            new_branches.append(nb)

        new_agent = Agent(
            id=new_id,
            role="worker",
            model=new_model,
            branches=new_branches,
            points=0.0,
            generation=self.generation + 1,
            parent=self.id,
        )
        return new_agent


def _clone_ppo_with_noise(model, sigma: float):
    """Копия PPO с шумом в весах policy.

    НЕ через copy.deepcopy (не работает с torch).
    Через save/load на диск.
    """
    if PPO is None or torch is None:
        raise RuntimeError("stable-baselines3/torch not available")
    import tempfile
    import os
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "clone_model")
    try:
        model.save(tmp_path)
        new_model = PPO.load(tmp_path, env=model.env)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    with torch.no_grad():
        for p in new_model.policy.parameters():
            p.add_(torch.randn_like(p) * sigma)
    return new_model


# ============================================================
# POPULATION — 4 агента, турнирная таблица, смещение
# ============================================================

class Population:
    """Популяция агентов с динамической иерархией.

    Турнирная таблица: 1-е ×1.0, 2-е ×0.6, 3-е ×0.3, 4-е ×0.0
    Смещение: сильнейший по очкам → Manager
    Смерть: 4-е место → агент + его ветви умирают
    Рождение: клон случайного (Manager/Worker 50/50) + мутация
    """

    RANK_REWARDS = [1.0, 0.6, 0.3, 0.0]

    def __init__(self, agents: list):
        if len(agents) < 2:
            raise ValueError("Population needs >= 2 agents")
        self.agents = agents
        self.manager = agents[0]
        self.manager.role = "manager"
        for a in agents[1:]:
            a.role = "worker"
        self.next_id_counter = len(agents)

    def get_workers(self) -> list:
        return [a for a in self.agents if a.role == "worker"]

    def get_alive(self) -> list:
        return [a for a in self.agents if a.role != "dead"]

    def update_ranks(self) -> list:
        """Смещение Manager'а: сильнейший по очкам становится Manager'ом."""
        events = []
        alive = self.get_alive()
        if not alive:
            return events
        best = max(alive, key=lambda a: a.points)
        if best.id != self.manager.id:
            old = self.manager.id
            old_agent = self.manager
            old_agent.role = "worker"
            best.role = "manager"
            self.manager = best
            events.append(f"DISPLACE:{old}->{best.id}")
        return events

    def kill_weakest(self, exclude_ids: Optional[list] = None) -> Optional[str]:
        """Убить агента с последним рангом (4-е место).

        Manager неприкосновенен. exclude_ids — дополнительно не убивать
        (например, старого Manager'а в поколении смещения).
        Убивает агента + его ветви.
        """
        exclude = set(exclude_ids or [])
        exclude.add(self.manager.id)
        alive = self.get_alive()
        if len(alive) <= 2:
            return None
        sorted_alive = sorted(
            alive,
            key=lambda a: (a.last_rank if a.last_rank > 0 else 999),
            reverse=True,
        )
        victim = None
        for a in sorted_alive:
            if a.id not in exclude:
                victim = a
                break
        if victim is None:
            return None
        victim.role = "dead"
        victim.branches = []
        return victim.id

    def replace_dead(self, sigma: float = 0.01, rng=None) -> Optional[str]:
        """Заменить мёртвого: клон случайного родителя (Manager/Worker 50/50).

        Наследование C = среднее(C_родителя, C_умершего).
        """
        dead = [a for a in self.agents if a.role == "dead"]
        if not dead:
            return None
        slot = dead[0]

        # выбор родителя: Manager или случайный Worker (50/50)
        if random.random() < 0.5:
            parent = self.manager
        else:
            workers = self.get_workers()
            parent = random.choice(workers) if workers else self.manager

        # fallback: если у родителя нет модели — берём Manager'а
        if parent.model is None:
            parent = self.manager
        # если и у Manager'а нет модели — рождение невозможно
        if parent.model is None:
            return None

        self.next_id_counter += 1
        new_id = f"G{self.next_id_counter}"
        new_agent = parent.clone_with_mutation(new_id, sigma=sigma, rng=rng)

        # наследование C = среднее(C_родителя, C_умершего)
        c_parent = parent.last_c if parent.last_c > 0 else 0.0
        c_dead = slot.last_c if slot.last_c > 0 else 0.0
        if c_parent > 0 and c_dead > 0:
            inherited_c = 0.5 * (c_parent + c_dead)
        elif c_parent > 0:
            inherited_c = c_parent
        elif c_dead > 0:
            inherited_c = c_dead
        else:
            inherited_c = 0.0
        new_agent.last_c = inherited_c
        new_agent.c_history = [inherited_c]

        idx = self.agents.index(slot)
        self.agents[idx] = new_agent
        return new_id

    def reset_points(self) -> None:
        for a in self.agents:
            a.points = 0.0
            a.last_rank = 0

    def summary(self) -> str:
        lines = []
        for a in self.agents:
            lines.append(
                f"  {a.id} ({a.role:>7}): pts={a.points:6.2f} "
                f"C={a.last_c:.4f} rank={a.last_rank} "
                f"branches={a.n_branches()} gen={a.generation}"
            )
        return "\n".join(lines)
# ============================================================
# MULTI-AGENT ENV — 4 агента × 10 ветвей
# ============================================================

class MultiAgentEnv(OmegaV131Env):
    """Среда с 4 группами ветвей (по 10 у каждого агента).

    Ключевое отличие от OmegaV131Env:
    - Ветви не глобальные, а привязаны к агентам (agent.branches)
    - Каждый агент выдаёт 5D action → применяется к его 10 ветвям
    - SPLIT/MERGE внутри группы агента
    - C агента = normalized_coherence его ветвей
    - Manager влияет через приоритеты (не в obs)
    """

    def __init__(
        self,
        *args,
        agents: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agents = agents or []
        self._priorities = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        self.rng = np.random.default_rng(self.initial_seed)

    def set_manager_priorities(self, priorities: np.ndarray) -> None:
        p = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if p.size != 4:
            raise ValueError(f"priorities must be 4D, got {p.shape}")
        p = p - p.max()
        e = np.exp(p)
        self._priorities = (e / (e.sum() + EPS)).astype(np.float32)

    def get_manager_priorities(self) -> np.ndarray:
        return self._priorities.copy()

    # ---------- инициализация ветвей агента ----------

    def init_agent_branches(self, agent, n_branches: int = 10) -> None:
        """Создать n_branches ветвей для агента. Стартовый state — случайный."""
        agent.branches = []
        for i in range(n_branches):
            bid = f"{agent.id}_b{i:02d}"
            x = self.rng.normal(size=STATE_DIM)
            n = float(np.linalg.norm(x))
            state = x / (n if n > EPS else 1.0)
            agent.branches.append(Branch(
                id=bid,
                state=state,
                energy=float(self.rng.uniform(0.4, 0.8)),
                coherence=0.0,
                age=0,
            ))

    # ---------- управление группой ----------

    def apply_group_action(self, agent, action: np.ndarray) -> float:
        """Применить 5D action ко всем ветвям агента. Вернуть action_norm."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != STATE_DIM:
            raise ValueError(f"expected 5D, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        for b in agent.branches:
            x = b.state + self.config.action_scale * action
            noise = self.rng.normal(0.0, self.config.noise * 0.05, STATE_DIM)
            x = x + noise
            n = np.linalg.norm(x)
            b.state = x / (n if n > EPS else 1.0)

        return float(np.linalg.norm(action))

    def evolve_group(self, agent) -> dict:
        """Один шаг эволюции для группы агента (10 ветвей).

        v13.4 фикс 1: добавлены SPLIT/MERGE внутри группы.
        Ветви могут делиться (SPLIT) и сливаться (MERGE).
        Жёсткий лимит 10 сохраняется: если SPLIT даёт >10, обрезаем.
        """
        births = 0
        deaths = 0
        splits = 0
        merges = 0
        events = []

        if not agent.branches:
            return {"births": 0, "deaths": 0, "splits": 0, "merges": 0,
                    "events": tuple(events)}

        # 1. локальная когерентность группы
        group_states = np.stack([b.state for b in agent.branches])
        group_mean = group_states.mean(axis=0)
        nm = float(np.linalg.norm(group_mean))
        group_dir = group_mean / nm if nm > EPS else None

        for b in agent.branches:
            if group_dir is not None:
                cos = float(np.dot(b.state, group_dir))
                cos = max(-1.0, min(1.0, cos))
                b.coherence = float(np.clip((cos + 1.0) * 0.5, 0.0, 1.0))
            else:
                b.coherence = 0.0
            b.age += 1
            b.energy += self.config.energy_gain * b.coherence
            b.energy -= self.config.energy_decay
            b.energy = float(np.clip(b.energy, 0.0, 1.5))

        # 2. MERGE (внутри группы)
        if len(agent.branches) >= 2:
            items = list(agent.branches)
            consumed = set()
            for i in range(len(items)):
                a = items[i]
                if a.id in consumed:
                    continue
                for j in range(i + 1, len(items)):
                    b = items[j]
                    if b.id in consumed:
                        continue
                    dot = float(np.dot(a.state, b.state))
                    dist = 1.0 - dot
                    if dist < self.config.merge_distance:
                        # merge b into a
                        new_state = a.state + b.state
                        n = float(np.linalg.norm(new_state))
                        if n > EPS:
                            a.state = new_state / n
                        a.energy = float(np.clip(
                            self.config.merge_energy_gain * (a.energy + b.energy),
                            0.0, 1.5,
                        ))
                        b.energy = 0.0  # помечаем на удаление
                        consumed.add(b.id)
                        merges += 1
                        events.append(f"MERGE:{b.id}->{a.id}")
            if merges:
                agent.branches = [b for b in agent.branches if b.id not in consumed]

        # 3. SPLIT (внутри группы, но с лимитом max_branches на группу)
        capacity = max(0, self.config.max_branches - len(agent.branches))
        if capacity > 0:
            candidates = sorted(
                agent.branches,
                key=lambda b: b.energy,
                reverse=True,
            )
            for parent in candidates:
                if capacity <= 0:
                    break
                if parent.energy < self.config.split_threshold:
                    continue
                if self.rng.random() > self.config.split_probability:
                    continue
                # делим: создаём ребёнка
                self._spawn_counter = getattr(self, "_spawn_counter", 0) + 1
                child_id = f"{agent.id}_sp{self._spawn_counter:05d}"
                child_energy = parent.energy * (1.0 - self.config.split_energy_cost)
                parent.energy *= self.config.split_energy_cost
                x = parent.state + self.rng.normal(0.0, 0.1, STATE_DIM)
                n = float(np.linalg.norm(x))
                child_state = x / (n if n > EPS else 1.0)
                child = Branch(
                    id=child_id,
                    state=child_state,
                    energy=float(child_energy),
                    coherence=0.0,
                    age=0,
                )
                agent.branches.append(child)
                splits += 1
                capacity -= 1
                events.append(f"SPLIT:{parent.id}->{child_id}")

        # 4. смерти (energy < death_threshold)
        survivors = []
        for b in agent.branches:
            if b.energy < self.config.death_threshold:
                deaths += 1
                events.append(f"DEATH:{b.id}")
            else:
                survivors.append(b)
        agent.branches = survivors

        return {
            "births": births, "deaths": deaths,
            "splits": splits, "merges": merges,
            "events": tuple(events),
        }

    def evaluate_agent_episode(self, agent, n_steps: int = 20) -> float:
        """Оценить агента: среднее C за n_steps шагов.

        Прогоняем политику агента на его ветвях n_steps раз,
        усредняем C. Это стабильнее, чем мгновенный C.
        """
        if agent.model is None or not agent.branches:
            return agent.coherence()

        cs = []
        for _ in range(n_steps):
            action = agent.model.predict(
                np.zeros(self.observation_space.shape[0], dtype=np.float32),
                deterministic=True,
            )[0]
            self.apply_group_action(agent, action)
            self.evolve_group(agent)
            cs.append(agent.coherence())
        return float(np.mean(cs)) if cs else 0.0

    def make_agent_env(self, agent):
        """Обёртка: среда для одного агента.

        step() применяет action агента к его 10 ветвям и возвращает
        reward = C + ΔC × bonus.
        """
        import gymnasium as gym
        from gymnasium import spaces

        env_self = self
        agent_ref = agent

        class AgentEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = env_self.observation_space
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
                )
                self._steps = 0
                self._episode_len = 100
                self._prev_c = 0.0

            def reset(self, *, seed=None, options=None):
                self._steps = 0
                self._prev_c = agent_ref.coherence()
                obs = np.zeros(
                    self.observation_space.shape[0], dtype=np.float32
                )
                return obs, {}

            def step(self, action):
                action = np.asarray(action, dtype=np.float64).reshape(-1)
                env_self.apply_group_action(agent_ref, action)
                env_self.evolve_group(agent_ref)

                c_now = agent_ref.coherence()
                d_c = c_now - self._prev_c
                self._prev_c = c_now

                reward = (
                    env_self.config.reward_coherence * c_now
                    + env_self.config.delta_c_bonus * d_c
                )

                self._steps += 1
                terminated = False
                truncated = self._steps >= self._episode_len
                info = {"coherence": c_now, "d_c": d_c}

                obs = np.zeros(
                    self.observation_space.shape[0], dtype=np.float32
                )
                return obs, float(reward), terminated, truncated, info

        return AgentEnv()

    # ---------- шаг всей популяции ----------

    def step_agents(self, agent_actions: dict) -> dict:
        """Один шаг: каждое действие на свою группу.

        agent_actions: dict {agent_id: 5D action}
        Возвращает: info-словарь с C по каждому агенту.
        """
        info = {"agents": {}, "c_means": {}}

        for agent in self.agents:
            if agent.role == "dead":
                continue
            action = agent_actions.get(agent.id)
            if action is None:
                action = np.zeros(STATE_DIM, dtype=np.float64)

            norm = self.apply_group_action(agent, action)
            evo = self.evolve_group(agent)
            c = agent.coherence()
            agent.last_c = c

            # mean energy группы
            energies = [b.energy for b in agent.branches] if agent.branches else [0.0]
            mean_energy = float(np.mean(energies))

            info["agents"][agent.id] = {
                "coherence": c,
                "n_branches": len(agent.branches),
                "action_norm": norm,
                "mean_energy": mean_energy,
                "births": evo["births"],
                "deaths": evo["deaths"],
                "splits": evo["splits"],
                "merges": evo["merges"],
                "events": evo["events"],
            }
            info["c_means"][agent.id] = c

        # C_mean по всем живым агентам
        if info["c_means"]:
            info["c_mean_overall"] = float(
                np.mean(list(info["c_means"].values()))
            )
        else:
            info["c_mean_overall"] = 0.0

        return info

    # ---------- reward для обучения ----------

    def compute_reward(self, info: dict, prev_c_means: dict) -> float:
        """Общий reward для всех агентов: средний C + ΔC + приоритеты Manager'а."""
        c_now = info["c_mean_overall"]
        c_prev = float(np.mean(list(prev_c_means.values()))) if prev_c_means else 0.0
        d_c = c_now - c_prev

        dc_bonus = float(getattr(self.config, "delta_c_bonus", 0.0)) * d_c

        # event penalty
        total_events = 0
        for aid, a in info["agents"].items():
            total_events += a["splits"] + a["merges"]
        event_cost = 0.0
        if self.config.event_penalty > 0.0:
            event_cost = (
                self.config.event_penalty * float(total_events)
                / max(1, self.config.max_branches)
            )

        reward = (
            self.config.reward_coherence * c_now
            + dc_bonus
            - event_cost
        )
        return float(reward)


# ============================================================
# WORLD LOGGER — летопись мира
# ==========================================================

class WorldLogger:
    """Ведёт летопись: поколения, события, метрики."""

    def __init__(self):
        self.generations: list = []
        self.events: list = []
        self.t0 = time.time()

    def log_generation(
        self, gen: int, agents: list, manager_id: str, c_means: dict
    ) -> None:
        record = {
            "generation": gen,
            "time": time.time() - self.t0,
            "manager": manager_id,
            "agents": [
                {
                    "id": a.id,
                    "role": a.role,
                    "points": round(a.points, 3),
                    "c_last": round(a.last_c, 4),
                    "c_mean": round(c_means.get(a.id, a.last_c), 4),
                    "n_branches": a.n_branches(),
                    "generation": a.generation,
                    "parent": a.parent,
                }
                for a in agents
            ],
            "c_mean_overall": round(
                float(np.mean(list(c_means.values()))) if c_means else 0.0, 4
            ),
        }
        self.generations.append(record)

    def log_event(self, event_type: str, details: dict) -> None:
        self.events.append({
            "time": time.time() - self.t0,
            "type": event_type,
            **details,
        })

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"generations": self.generations, "events": self.events}
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_history(self) -> None:
        print(f"\n{'='*70}")
        print(f"ИСТОРИЯ МИРА ({len(self.generations)} поколений)")
        print(f"{'='*70}")
        for g in self.generations:
            print(
                f"  Поколение {g['generation']:>2}: "
                f"manager={g['manager']}  "
                f"C_mean={g['c_mean_overall']:.4f}"
            )
            for a in g["agents"]:
                marker = "★" if a["id"] == g["manager"] else " "
                status = "МЁРТВ" if a["role"] == "dead" else a["role"]
                print(
                    f"    {marker} {a['id']:>4} [{status:>7}] "
                    f"pts={a['points']:7.2f}  C={a['c_mean']:.4f}  "
                    f"br={a['n_branches']:>2}  gen={a['generation']}"
                )
        if self.events:
            print(f"\n  СОБЫТИЯ (последние 20):")
            for e in self.events[-20:]:
                t = e.get("time", 0.0)
                print(f"    [{t:6.1f}s] {e['type']}: "
                      f"{ {k: v for k, v in e.items() if k not in ('time', 'type')} }")
# ============================================================
# EVOLVE STEP — одно поколение
# ============================================================

def evolve_step(
    population: Population,
    env: MultiAgentEnv,
    logger: WorldLogger,
    generation: int,
    episodes_per_gen: int,
    train_steps: int,
    seeds: list,
    sigma: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Одно поколение:

    1. Оценка агентов (C по своим ветвям)
    2. Турнирная таблица: 1-е ×1.0, 2-е ×0.6, 3-е ×0.3, 4-е ×0.0
    3. Смещение Manager'а (сильнейший по очкам)
    4. Смерть 4-го места (агент + его ветви)
    5. Рождение: клон случайного (Manager/Worker 50/50) + мутация
    6. Дообучение всех (train_steps)
    """
    env.agents = population.get_alive()
    alive = population.get_alive()
    c_means = {}

    # 1. оценка (по эпизоду — среднее C за 20 шагов)
    if verbose:
        print(f"\n[GEN {generation}] Оценка агентов...")
    for agent in alive:
        c = env.evaluate_agent_episode(agent, n_steps=20)
        c_means[agent.id] = c
        agent.last_c = c
        agent.c_history.append(c)
        if verbose:
            print(
                f"  {agent.id} ({agent.role:>7}): "
                f"C={c:.4f}  branches={agent.n_branches()}"
            )

    # 2. турнирная таблица
    rank_rewards = Population.RANK_REWARDS
    ranked = sorted(alive, key=lambda a: c_means.get(a.id, 0.0), reverse=True)
    for i, agent in enumerate(ranked):
        rank = i + 1
        weight = rank_rewards[i] if i < len(rank_rewards) else 0.0
        agent.add_points(c_means.get(agent.id, 0.0), weight)
        agent.last_rank = rank
        if verbose:
            print(
                f"  #{rank} {agent.id} ({agent.role:>7}): "
                f"C={c_means.get(agent.id, 0.0):.4f} "
                f"→ points += {c_means.get(agent.id, 0.0) * weight:.4f}"
            )

    # 3. смещение — запомним старого Manager'а
    old_manager_id = population.manager.id
    events = population.update_ranks()
    for e in events:
        logger.log_event("DISPLACE", {"generation": generation, "detail": e})
        if verbose:
            print(f"  ⚡ {e}")

    # 4. смерть 4-го места (не убивать старого Manager'а)
    dead_id = population.kill_weakest(exclude_ids=[old_manager_id])
    if dead_id:
        logger.log_event("DEATH", {"generation": generation, "agent": dead_id})
        if verbose:
            print(f"  ☠ умер: {dead_id} (с ветвями)")

    # 5. рождение (клон Manager/Worker 50/50)
    new_id = population.replace_dead(sigma=sigma, rng=env.rng)
    if new_id:
        logger.log_event("BIRTH", {"generation": generation, "agent": new_id})
        if verbose:
            new_agent = population.get_alive()[-1]
            print(
                f"  ✚ рождён: {new_id} (parent={new_agent.parent}, "
                f"branches={new_agent.n_branches()})"
            )

    # 6. логирование
    logger.log_generation(
        generation, population.agents,
        population.manager.id, c_means,
    )

    # 7. дообучение
    if train_steps > 0 and verbose:
        print(f"  Дообучение ({train_steps} шагов на агента)...")
    for agent in population.get_alive():
        if agent.model is None:
            continue
        try:
            _fine_tune_agent(agent, train_steps, env)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] fine-tune {agent.id} failed: {ex}")

    population.reset_points()

    return {
        "generation": generation,
        "c_means": c_means,
        "manager": population.manager.id,
        "events": events,
        "dead": dead_id,
        "born": new_id,
    }


def _fine_tune_agent(agent: Agent, steps: int, env) -> None:
    """Дообучение агента PPO на кастомной среде (C + ΔC × bonus)."""
    model = agent.model
    if model is None:
        return
    agent_env = env.make_agent_env(agent)
    model.set_env(agent_env)
    model.learn(total_timesteps=int(steps), reset_num_timesteps=False)


# ============================================================
# TRAIN EVA — главный цикл
# ============================================================

def train_eva(
    config: OmegaConfig,
    seeds: list,
    generations: int,
    n_agents: int,
    branches_per_agent: int,
    episodes_per_gen: int,
    train_steps_per_gen: int,
    output_dir: str,
    sigma: float = 0.01,
) -> list:
    """Построить мир: n_agents агентов × branches_per_agent ветвей."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = WorldLogger()
    rng = np.random.default_rng(seeds[0])

    print(f"\n>>> EVA: {n_agents} агентов × {branches_per_agent} ветвей "
          f"= {n_agents * branches_per_agent} ветвей всего")
    print(f"    {generations} поколений, seeds={seeds}, σ={sigma}\n")

    # создаём стартовых агентов
    agents = []
    for i in range(n_agents):
        aid = f"A{i+1}"
        print(f"  Создаю агента {aid}...")

        # обучаем PPO на базовой среде (как в v13.1)
        env_i = OmegaV131Env(
            config=copy.deepcopy(config), seed=seeds[0],
            observation_mode="global", reward_mode="coherence",
        )
        train_ppo(
            env_i, seeds[0], train_steps_per_gen,
            save_path=output_path / "models" / f"init_{aid}",
            verbose_progress=False,
        )
        if PPO is not None:
            loaded = PPO.load(
                str(output_path / "models" / f"init_{aid}"),
                env=env_i,
            )
        else:
            loaded = None

        agent = Agent(id=aid, role="worker", model=loaded)

        # создаём группу ветвей
        env_tmp = MultiAgentEnv(
            config=config, seed=seeds[0],
            observation_mode="global", reward_mode="coherence",
        )
        env_tmp.rng = rng
        env_tmp.init_agent_branches(agent, branches_per_agent)
        env_i.close()
        env_tmp.close()

        agents.append(agent)

    population = Population(agents)
    population.manager = agents[0]
    agents[0].role = "manager"

    # общая среда для оценки и эволюции
    eval_env = MultiAgentEnv(
        config=copy.deepcopy(config), seed=seeds[0],
        observation_mode="global", reward_mode="coherence",
        agents=population.get_alive(),
    )
    eval_env.rng = rng

    rows = []
    for gen in range(1, generations + 1):
        t0 = time.time()
        result = evolve_step(
            population=population,
            env=eval_env,
            logger=logger,
            generation=gen,
            episodes_per_gen=episodes_per_gen,
            train_steps=train_steps_per_gen,
            seeds=seeds,
            sigma=sigma,
            verbose=True,
        )
        elapsed = time.time() - t0

        c_overall = float(np.mean(list(result["c_means"].values()))) if result["c_means"] else 0.0
        c_best = max(result["c_means"].values()) if result["c_means"] else 0.0

        rows.append({
            "generation": gen,
            "manager": result["manager"],
            "c_mean": round(c_overall, 4),
            "c_best": round(c_best, 4),
            "dead": result["dead"] or "",
            "born": result["born"] or "",
            "time_sec": round(elapsed, 1),
        })

        print(f"\n[GEN {gen}] C_mean={c_overall:.4f}  C_best={c_best:.4f}  "
              f"manager={result['manager']}  t={elapsed:.1f}s")

        # сохраняем модели
        for a in population.get_alive():
            if a.model is not None:
                try:
                    a.model.save(str(output_path / "models" / f"gen{gen}_{a.id}"))
                except Exception:
                    pass

    eval_env.close()

    # сохраняем историю
    logger.save(output_path / "world_history.json")
    logger.print_history()

    if rows:
        csv_path = output_path / "eva_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n[SAVE] {csv_path}")
        print(f"[SAVE] {output_path / 'world_history.json'}")

    return rows


# ============================================================
# SMOKE
# ============================================================

def smoke_test():
    print("[SMOKE v13.4] EVA — Evolving Agents + Branches")
    cfg = OmegaConfig(
        initial_branches=40, max_branches=100, episode_length=20,
        enable_merge_split=True, use_event_features=True,
    )
    apply_preset(cfg, "c")
    cfg.enable_merge_split = True
    cfg.use_event_features = True

    # среда
    env = MultiAgentEnv(cfg, seed=0, observation_mode="global",
                        reward_mode="coherence")
    print(f"[SMOKE]   env obs.shape={env.observation_space.shape}")

    # агент без модели (для проверки ветвей)
    a1 = Agent(id="A1", role="manager", model=None)
    env.init_agent_branches(a1, n_branches=10)
    print(f"[SMOKE]   agent A1 branches={a1.n_branches()}")
    print(f"[SMOKE]   A1 initial C={a1.coherence():.4f}")

    # применяем action
    action = np.array([0.5, -0.3, 0.1, -0.2, 0.0], dtype=np.float64)
    norm = env.apply_group_action(a1, action)
    print(f"[SMOKE]   apply_group_action norm={norm:.4f}")
    print(f"[SMOKE]   A1 C after action={a1.coherence():.4f}")

    # эволюция группы
    env.agents = [a1]
    info = env.step_agents({a1.id: action})
    print(f"[SMOKE]   step_agents: c_mean_overall={info['c_mean_overall']:.4f}")
    print(f"[SMOKE]   branches after step={a1.n_branches()}")

    # турнирная таблица (симуляция)
    a2 = Agent(id="A2", role="worker", model=None)
    a3 = Agent(id="A3", role="worker", model=None)
    a4 = Agent(id="A4", role="worker", model=None)
    for a in [a1, a2, a3, a4]:
        if a.n_branches() == 0:
            env.init_agent_branches(a, n_branches=10)
    pop = Population([a1, a2, a3, a4])

    # имитация очков
    a1.add_points(0.50, 1.0)
    a2.add_points(0.48, 0.6)
    a3.add_points(0.45, 0.3)
    a4.add_points(0.40, 0.0)
    a1.last_rank = 1
    a2.last_rank = 2
    a3.last_rank = 3
    a4.last_rank = 4
    events = pop.update_ranks()
    print(f"[SMOKE]   ranks updated: events={events}, manager={pop.manager.id}")

    # смерть 4-го
    dead = pop.kill_weakest()
    print(f"[SMOKE]   killed: {dead}")

    # рождение — только если есть модель для клонирования
    if all(a.model is None for a in pop.get_alive()):
        print("[SMOKE]   born: skipped (no models in smoke)")
    else:
        new_id = pop.replace_dead(sigma=0.01, rng=np.random.default_rng(42))
        if new_id:
            new_agent = pop.get_alive()[-1]
            print(f"[SMOKE]   born: {new_id} (parent={new_agent.parent}, "
                  f"branches={new_agent.n_branches()}, "
                  f"inherited_c={new_agent.last_c:.4f})")
        else:
            print("[SMOKE]   born: failed (no model)")

    env.close()
    print("[SMOKE v13.4] OK")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Omega v13.4 EVA")
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--branches", type=int, default=40,
                   help="initial_branches (для стартового PPO)")
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--branches-per-agent", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--episodes-per-gen", type=int, default=3)
    p.add_argument("--train-steps-per-gen", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--complexity", choices=list(COMPLEXITY_PRESETS), default="c")
    p.add_argument("--delta-c-bonus", type=float, default=10.0)
    p.add_argument("--event-penalty", type=float, default=0.5)
    p.add_argument("--output", default="v13.4_results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    if PPO is None:
        print("[ERROR] stable-baselines3 не установлен")
        return

    cfg = OmegaConfig(initial_branches=args.branches)
    apply_preset(cfg, args.complexity)
    cfg.enable_merge_split = True
    cfg.use_event_features = True
    cfg.event_penalty = float(args.event_penalty)
    cfg.delta_c_bonus = float(args.delta_c_bonus)
    cfg.__post_init__()

    seeds = list(range(args.seeds))

    train_eva(
        config=cfg,
        seeds=seeds,
        generations=args.generations,
        n_agents=args.agents,
        branches_per_agent=args.branches_per_agent,
        episodes_per_gen=args.episodes_per_gen,
        train_steps_per_gen=args.train_steps_per_gen,
        output_dir=args.output,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()