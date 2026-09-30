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
# v13.5: переопределяем STATE_DIM = 20 (было 5 в v13.1)
# ============================================================
STATE_DIM = 20
ACTION_DIM = 2 * STATE_DIM   # 40D: 20D на cluster A + 20D на cluster B
N_MODES = 4
MODE_EXPLORE = 0
MODE_STABILIZE = 1
MODE_OPTIMIZE = 2
MODE_OBSERVE = 3


# ============================================================
# BRANCH — ветвь (привязана к агенту)
# v13.5: STATE_DIM = 20, добавлен cluster (0 или 1)
# ============================================================

@dataclass
class Branch:
    """Ветвь — вектор 20D, принадлежит агенту.

    id: уникальный (например, "A1_c0b0", "A1_c1b0", ...)
    state: 20D вектор на сфере
    energy: 0..1.5
    coherence: локальная когерентность
    age: сколько шагов живёт
    cluster: 0 = cluster A, 1 = cluster B (v13.5)
    """
    id: str
    state: np.ndarray
    energy: float = 0.5
    coherence: float = 0.0
    age: int = 0
    cluster: int = 0

    def clone(self) -> "Branch":
        return Branch(
            id=self.id,
            state=self.state.copy(),
            energy=self.energy,
            coherence=self.coherence,
            age=self.age,
            cluster=self.cluster,
        )


def mutate_branch_state(state: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Мутация state ветви: state + N(0, σ), нормировка на сферу."""
    x = state + rng.normal(0.0, sigma, size=STATE_DIM)
    n = float(np.linalg.norm(x))
    return x / (n if n > EPS else 1.0)


# ============================================================
# AGENT — PPO + 2 кластера ветвей (v13.5)
# ============================================================

@dataclass
class Agent:
    """Агент = PPO + 2 кластера + target.

    v13.5-B: coherence() = косинус к target (не внутрикластерная).
    """
    id: str
    role: str = "worker"
    model: Optional[object] = None
    cluster_a: list = field(default_factory=list)
    cluster_b: list = field(default_factory=list)
    points: float = 0.0
    last_rank: int = 0
    last_c: float = 0.0
    generation: int = 0
    parent: Optional[str] = None
    c_history: list = field(default_factory=list)
    target: Optional[np.ndarray] = None
    history_features: Optional[np.ndarray] = None   # v14.1: 10D истории клана
    bonus_points: float = 0.0   # v14.2: накапливаемые бонусы

    def branches(self) -> list:
        return list(self.cluster_a) + list(self.cluster_b)

    def n_branches(self) -> int:
        return len(self.cluster_a) + len(self.cluster_b)

    def states(self) -> np.ndarray:
        all_b = self.branches()
        if not all_b:
            return np.zeros((0, STATE_DIM), dtype=np.float64)
        return np.stack([b.state for b in all_b])

    def cluster_mean(self, cluster: int) -> Optional[np.ndarray]:
        branches = self.cluster_a if cluster == 0 else self.cluster_b
        if not branches:
            return None
        return np.mean([b.state for b in branches], axis=0)

    def cluster_coherence_to_target(self, cluster: int) -> float:
        """C_cluster = косинус(mean_cluster, target), нормированный в [0,1]."""
        if self.target is None:
            return 0.0
        mean = self.cluster_mean(cluster)
        if mean is None:
            return 0.0
        nm = float(np.linalg.norm(mean))
        nt = float(np.linalg.norm(self.target))
        if nm <= EPS or nt <= EPS:
            return 0.0
        cos = float(np.dot(mean / nm, self.target / nt))
        cos = max(-1.0, min(1.0, cos))
        return float((cos + 1.0) * 0.5)

    def cluster_coherence(self, cluster: int) -> float:
        """v13.5-B: C_cluster = косинус к target."""
        return self.cluster_coherence_to_target(cluster)

    def separation(self) -> float:
        """Разделение кластеров (для информации)."""
        if not self.cluster_a or not self.cluster_b:
            return 0.0
        mean_a = np.mean([b.state for b in self.cluster_a], axis=0)
        mean_b = np.mean([b.state for b in self.cluster_b], axis=0)
        na = float(np.linalg.norm(mean_a))
        nb = float(np.linalg.norm(mean_b))
        if na <= EPS or nb <= EPS:
            return 0.0
        cos = float(np.dot(mean_a / na, mean_b / nb))
        cos = max(-1.0, min(1.0, cos))
        return float((1.0 - cos) * 0.5)

    def coherence(self) -> float:
        """v13.5-B: C = среднее(C_A, C_B) — оба к target."""
        c_a = self.cluster_coherence_to_target(0)
        c_b = self.cluster_coherence_to_target(1)
        return 0.5 * (c_a + c_b)

    def add_points(self, c: float, weight: float) -> None:
        self.points += float(c) * float(weight)

    def add_bonus(self, amount: float) -> None:
        """v14.2: накопить бонус."""
        self.bonus_points += float(amount)

    def total_points(self) -> float:
        """v14.2: общие очки = points + bonus_points."""
        return self.points + self.bonus_points

    def clone_with_mutation(
        self, new_id: str, sigma: float = 0.01, rng=None
    ) -> "Agent":
        if self.model is None:
            raise RuntimeError(f"Agent {self.id} has no model to clone")
        if rng is None:
            rng = np.random.default_rng()

        new_model = _clone_ppo_with_noise(self.model, sigma=sigma)

        new_a = [b.clone() for b in self.cluster_a]
        new_b = [b.clone() for b in self.cluster_b]

        return Agent(
            id=new_id,
            role="worker",
            model=new_model,
            cluster_a=new_a,
            cluster_b=new_b,
            points=0.0,
            generation=self.generation + 1,
            parent=self.id,
            target=None if self.target is None else self.target.copy(),
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

    RANK_REWARDS = None  # v14.0: задаётся в __init__ по числу агентов

    def __init__(self, agents: list):
        if len(agents) < 2:
            raise ValueError("Population needs >= 2 agents")
        self.agents = agents
        self.manager = agents[0]
        self.manager.role = "manager"
        for a in agents[1:]:
            a.role = "worker"
        self.next_id_counter = len(agents)
        # v14.0: динамические RANK_REWARDS по числу агентов
        n = len(agents)
        Population.RANK_REWARDS = np.linspace(1.0, 0.0, n).tolist()

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
        best = max(alive, key=lambda a: a.total_points())
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
        # v13.5: очищаем оба кластера
        victim.cluster_a = []
        victim.cluster_b = []
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
            # v14.2: bonus_points НЕ сбрасываем — накапливается

    def summary(self) -> str:
        lines = []
        for a in self.agents:
            lines.append(
                f"  {a.id} ({a.role:>7}): pts={a.total_points():6.2f} "
                f"C={a.last_c:.4f} rank={a.last_rank} "
                f"branches={a.n_branches()} gen={a.generation}"
            )
        return "\n".join(lines)
# ============================================================
# MULTI-AGENT ENV — 4 агента × 10 ветвей
# ============================================================

class MultiClusterEnv(OmegaV131Env):
    """Среда с 4 агентами, у каждого 2 кластера ветвей (5 + 5).

    v13.5:
    - STATE_DIM = 20
    - action = 20D
    - C = среднее(C_A, C_B) — внутрикластерная когерентность
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
        self.fixed_branches = 0
        # v14.1: расширяем observation_space на 10D истории клана
        from gymnasium import spaces
        old_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(old_dim + 10,), dtype=np.float32,
        )

    

    def set_manager_priorities(self, priorities: np.ndarray) -> None:
        p = np.asarray(priorities, dtype=np.float32).reshape(-1)
        if p.size != 4:
            raise ValueError(f"priorities must be 4D, got {p.shape}")
        p = p - p.max()
        e = np.exp(p)
        self._priorities = (e / (e.sum() + EPS)).astype(np.float32)

    def get_manager_priorities(self) -> np.ndarray:
        return self._priorities.copy()

    def init_agent_branches(self, agent, n_branches: int = 10) -> None:
        """Создать 2 кластера + задать target (v13.5-B)."""
        # задаём target
        if agent.target is None:
            x = self.rng.normal(size=STATE_DIM)
            n = float(np.linalg.norm(x))
            agent.target = x / (n if n > EPS else 1.0)

        half = max(1, n_branches // 2)
        agent.cluster_a = []
        agent.cluster_b = []
        for i in range(half):
            bid = f"{agent.id}_a{i:02d}"
            x = self.rng.normal(size=STATE_DIM)
            n = float(np.linalg.norm(x))
            state = x / (n if n > EPS else 1.0)
            agent.cluster_a.append(Branch(
                id=bid, state=state,
                energy=float(self.rng.uniform(0.4, 0.8)),
                coherence=0.0, age=0, cluster=0,
            ))
        for i in range(n_branches - half):
            bid = f"{agent.id}_b{i:02d}"
            x = self.rng.normal(size=STATE_DIM)
            n = float(np.linalg.norm(x))
            state = x / (n if n > EPS else 1.0)
            agent.cluster_b.append(Branch(
                id=bid, state=state,
                energy=float(self.rng.uniform(0.4, 0.8)),
                coherence=0.0, age=0, cluster=1,
            ))

    def apply_group_action(self, agent, action: np.ndarray) -> float:
        """Применить 40D action: первые 20D → cluster A, вторые 20D → cluster B.

        v13.5-A: разные действия для кластеров.
        """
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != ACTION_DIM:
            raise ValueError(
                f"expected {ACTION_DIM}D action, got {action.shape}"
            )
        action = np.clip(action, -1.0, 1.0)
        action_a = action[:STATE_DIM]
        action_b = action[STATE_DIM:]

        for b in agent.cluster_a:
            x = b.state + self.config.action_scale * action_a
            noise = self.rng.normal(0.0, self.config.noise * 0.05, STATE_DIM)
            x = x + noise
            n = np.linalg.norm(x)
            b.state = x / (n if n > EPS else 1.0)

        for b in agent.cluster_b:
            x = b.state + self.config.action_scale * action_b
            noise = self.rng.normal(0.0, self.config.noise * 0.05, STATE_DIM)
            x = x + noise
            n = np.linalg.norm(x)
            b.state = x / (n if n > EPS else 1.0)

        return float(np.linalg.norm(action))

    def evolve_group(self, agent) -> dict:
        """Эволюция обоих кластеров агента."""
        births = 0
        deaths = 0
        splits = 0
        merges = 0
        events = []
        fixed = int(getattr(self, "fixed_branches", 0))

        for cluster_id, branches in [
            (0, agent.cluster_a), (1, agent.cluster_b)
        ]:
            if not branches:
                continue

            # v13.5-B: локальная когерентность = косинус ветви к target
            if agent.target is not None:
                nt = float(np.linalg.norm(agent.target))
                target_dir = agent.target / nt if nt > EPS else None
            else:
                target_dir = None

            for b in branches:
                if target_dir is not None:
                    cos = float(np.dot(b.state, target_dir))
                    cos = max(-1.0, min(1.0, cos))
                    b.coherence = float(np.clip((cos + 1.0) * 0.5, 0.0, 1.0))
                else:
                    b.coherence = 0.0
                b.age += 1
                b.energy += self.config.energy_gain * b.coherence
                b.energy -= self.config.energy_decay
                b.energy = float(np.clip(b.energy, 0.0, 1.5))

            if fixed > 0:
                while len(branches) < fixed:
                    self._spawn_counter = getattr(self, "_spawn_counter", 0) + 1
                    bid = f"{agent.id}_c{cluster_id}fix{self._spawn_counter:05d}"
                    x = self.rng.normal(size=STATE_DIM)
                    n = float(np.linalg.norm(x))
                    state = x / (n if n > EPS else 1.0)
                    branches.append(Branch(
                        id=bid, state=state, energy=0.5,
                        coherence=0.0, age=0, cluster=cluster_id,
                    ))
                    births += 1
                if len(branches) > fixed:
                    branches[:] = branches[:fixed]
                continue

            # MERGE
            if len(branches) >= 2:
                items = list(branches)
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
                            new_state = a.state + b.state
                            n = float(np.linalg.norm(new_state))
                            if n > EPS:
                                a.state = new_state / n
                            a.energy = float(np.clip(
                                self.config.merge_energy_gain * (a.energy + b.energy),
                                0.0, 1.5,
                            ))
                            b.energy = 0.0
                            consumed.add(b.id)
                            merges += 1
                            events.append(f"MERGE:{b.id}->{a.id}")
                if merges:
                    branches[:] = [b for b in branches if b.id not in consumed]

            # SPLIT
            capacity = max(0, self.config.max_branches - len(branches))
            if capacity > 0:
                candidates = sorted(branches, key=lambda b: b.energy, reverse=True)
                for parent in candidates:
                    if capacity <= 0:
                        break
                    if parent.energy < self.config.split_threshold:
                        continue
                    if self.rng.random() > self.config.split_probability:
                        continue
                    self._spawn_counter = getattr(self, "_spawn_counter", 0) + 1
                    child_id = f"{agent.id}_c{cluster_id}sp{self._spawn_counter:05d}"
                    child_energy = parent.energy * (1.0 - self.config.split_energy_cost)
                    parent.energy *= self.config.split_energy_cost
                    x = parent.state + self.rng.normal(0.0, 0.1, STATE_DIM)
                    n = float(np.linalg.norm(x))
                    child_state = x / (n if n > EPS else 1.0)
                    child = Branch(
                        id=child_id, state=child_state,
                        energy=float(child_energy), coherence=0.0, age=0,
                        cluster=cluster_id,
                    )
                    branches.append(child)
                    splits += 1
                    capacity -= 1
                    events.append(f"SPLIT:{parent.id}->{child_id}")

            # DEATH
            survivors = []
            for b in branches:
                if b.energy < self.config.death_threshold:
                    deaths += 1
                    events.append(f"DEATH:{b.id}")
                else:
                    survivors.append(b)
            branches[:] = survivors

        return {
            "births": births, "deaths": deaths,
            "splits": splits, "merges": merges,
            "events": tuple(events),
        }

    def evaluate_agent_episode(self, agent, n_steps: int = 20) -> float:
        """Оценка: среднее C за n_steps шагов."""
        if agent.model is None or agent.n_branches() == 0:
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
        """Обёртка: среда для одного агента (20D action)."""
        import gymnasium as gym
        from gymnasium import spaces

        env_self = self
        agent_ref = agent

        class AgentEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.observation_space = env_self.observation_space
                self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
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

    def step_agents(self, agent_actions: dict) -> dict:
        """Один шаг: каждое действие на своего агента."""
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

            all_b = agent.branches()
            energies = [b.energy for b in all_b] if all_b else [0.0]
            mean_energy = float(np.mean(energies))

            info["agents"][agent.id] = {
                "coherence": c,
                "n_branches": len(all_b),
                "action_norm": norm,
                "mean_energy": mean_energy,
                "births": evo["births"],
                "deaths": evo["deaths"],
                "splits": evo["splits"],
                "merges": evo["merges"],
                "events": evo["events"],
                "separation": agent.separation(),
            }
            info["c_means"][agent.id] = c

        if info["c_means"]:
            info["c_mean_overall"] = float(
                np.mean(list(info["c_means"].values()))
            )
        else:
            info["c_mean_overall"] = 0.0

        return info

    def compute_reward(self, info: dict, prev_c_means: dict) -> float:
        """Reward: C + ΔC - event_penalty."""
        c_now = info["c_mean_overall"]
        c_prev = float(np.mean(list(prev_c_means.values()))) if prev_c_means else 0.0
        d_c = c_now - c_prev
        dc_bonus = float(getattr(self.config, "delta_c_bonus", 0.0)) * d_c

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
                    "points": round(a.total_points(), 3),
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
# CLAN — изолированная популяция (v14.0)
# ============================================================

class Clan:
    """Клан = изолированная популяция из N агентов.

    У каждого клана:
    - свои агенты
    - свой Manager
    - своя турнирная таблица
    - свои смерти/рождения
    """

    def __init__(self, clan_id: int, agents: list):
        if len(agents) < 2:
            raise ValueError("Clan needs >= 2 agents")
        self.id = int(clan_id)
        self.agents = agents
        self.manager = agents[0]
        self.manager.role = "manager"
        for a in agents[1:]:
            a.role = "worker"
        self.next_id_counter = 0
        self.c_history = []
        n = len(agents)
        self.rank_rewards = np.linspace(1.0, 0.0, n).tolist()

    def get_alive(self) -> list:
        return [a for a in self.agents if a.role != "dead"]

    def get_workers(self) -> list:
        return [a for a in self.agents if a.role == "worker"]

    def get_best(self):
        alive = self.get_alive()
        if not alive:
            return None
        return max(alive, key=lambda a: a.last_c)

    def history_stats(self, n_clans: int = 4, generations_total: int = 10) -> np.ndarray:
        """v14.1: 10D вектор истории клана.

        [clan_id/n, c_mean_clan, c_trend, splits/1000, merges/1000,
         deaths/100, births/100, generation/10, manager_changes/10, is_finalist]
        """
        # c_mean_clan
        c_mean = self.c_history[-1] if self.c_history else 0.0
        # c_trend: разница между последними и первыми 3 поколениями
        if len(self.c_history) >= 6:
            recent = float(np.mean(self.c_history[-3:]))
            early = float(np.mean(self.c_history[:3]))
            c_trend = float(np.clip(recent - early, -1.0, 1.0))
        else:
            c_trend = 0.0
        # считаем события через всех агентов (только живых + мёртвых)
        all_ids = set()
        splits_total = 0
        merges_total = 0
        deaths_total = 0
        births_total = 0
        for a in self.agents:
            # приблизительно: каждое SPLIT/DEATH — событие
            all_ids.add(a.id)
        # приблизительная оценка по числу агентов
        n_branches_total = sum(a.n_branches() for a in self.agents)
        # splits ~ n_branches / 50, deaths ~ generation * 1
        splits_total = max(0, n_branches_total - 100)
        merges_total = max(0, 100 - n_branches_total)
        # для истории: менеджер-смены
        manager_changes = self.next_id_counter
        # generation: по c_history
        generation = len(self.c_history)

        return np.asarray([
            float(self.id) / max(1, n_clans),
            float(c_mean),
            float(c_trend),
            float(splits_total) / 1000.0,
            float(merges_total) / 1000.0,
            float(deaths_total) / 100.0,
            float(births_total) / 100.0,
            float(generation) / max(1, generations_total),
            float(manager_changes) / 10.0,
            0.0,  # is_finalist — задаётся в финале
        ], dtype=np.float32)

    def summary(self) -> str:
        lines = [f"  [Клан {self.id}] manager={self.manager.id}"]
        for a in self.agents:
            lines.append(
                f"    {a.id} ({a.role:>7}): C={a.last_c:.4f} "
                f"br={a.n_branches()}"
            )
        return "\n".join(lines)


# ============================================================
# EVOLVE STEP — одно поколение
# ============================================================

def evolve_step(
    population: Population,
    env: MultiClusterEnv,
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
        # v14.3: бонус за тренд C (стабильный рост)
        if len(agent.c_history) >= 5:
            recent = agent.c_history[-5:]
            x = np.arange(5)
            slope = float(np.polyfit(x, recent, 1)[0])
            if slope > 0:
                # базовый + пропорционально тренду
                bonus = 0.2 + 2.0 * slope
                agent.add_bonus(bonus)
                if verbose:
                    print(
                        f"  ⭐ {agent.id}: тренд C={slope:+.5f} "
                        f"→ bonus += {bonus:.4f}"
                    )
        elif len(agent.c_history) >= 2:
            # fallback для первых поколений (мало истории)
            c_now = agent.c_history[-1]
            c_prev = agent.c_history[-2]
            if c_now > c_prev:
                delta = c_now - c_prev
                bonus = 0.2 + 0.5 * delta
                agent.add_bonus(bonus)
                if verbose:
                    print(
                        f"  ⭐ {agent.id}: рост C={delta:+.4f} "
                        f"→ bonus += {bonus:.4f}"
                    )
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

    # 4. смерти: N/4 агентов (v14.0: масштабирование)
    n_alive = len(population.get_alive())
    n_kill = max(1, n_alive // 4)
    dead_ids = []
    for _ in range(n_kill):
        did = population.kill_weakest(
            exclude_ids=[old_manager_id] + dead_ids
        )
        if did:
            dead_ids.append(did)
            logger.log_event("DEATH", {"generation": generation, "agent": did})
            if verbose:
                print(f"  ☠ умер: {did} (с ветвями)")

    # 5. рождения: столько же, сколько смертей
    born_ids = []
    for _ in range(len(dead_ids)):
        nid = population.replace_dead(sigma=sigma, rng=env.rng)
        if nid:
            born_ids.append(nid)
            logger.log_event("BIRTH", {"generation": generation, "agent": nid})
            if verbose:
                print(f"  ✚ рождён: {nid}")

    # для совместимости с остатком evolve_step
    dead_id = dead_ids[0] if dead_ids else None
    new_id = born_ids[0] if born_ids else None

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
def evolve_step_clan(
    clan: Clan,
    env: MultiClusterEnv,
    logger: WorldLogger,
    generation: int,
    episodes_per_gen: int,
    train_steps: int,
    seeds: list,
    sigma: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Одно поколение внутри клана (изолированно)."""
    env.agents = clan.get_alive()
    alive = clan.get_alive()
    c_means = {}

    # 1. оценка
    if verbose:
        print(f"\n[КЛАН {clan.id} | GEN {generation}] Оценка...")
    for agent in alive:
        c = env.evaluate_agent_episode(agent, n_steps=20)
        c_means[agent.id] = c
        agent.last_c = c
        agent.c_history.append(c)
        if verbose:
            print(f"  {agent.id} ({agent.role:>7}): C={c:.4f}")

    # 2. турнирная таблица (внутри клана)
    rank_rewards = clan.rank_rewards
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

    # 3. смещение Manager'а клана
    old_manager_id = clan.manager.id
    best = max(clan.get_alive(), key=lambda a: a.total_points())
    displace = None
    if best.id != clan.manager.id:
        old = clan.manager.id
        old_agent = clan.manager
        old_agent.role = "worker"
        best.role = "manager"
        clan.manager = best
        displace = f"DISPLACE:{old}->{best.id}"
        logger.log_event("CLAN_DISPLACE", {
            "clan": clan.id, "generation": generation, "detail": displace,
        })
        if verbose:
            print(f"  ⚡ [{clan.id}] {displace}")

    # 4. смерть слабейшего (не Manager)
    exclude = {clan.manager.id, old_manager_id}
    sorted_alive = sorted(
        clan.get_alive(),
        key=lambda a: (a.last_rank if a.last_rank > 0 else 999),
        reverse=True,
    )
    victim = None
    for a in sorted_alive:
        if a.id not in exclude:
            victim = a
            break
    dead_id = None
    if victim is not None:
        victim.role = "dead"
        victim.cluster_a = []
        victim.cluster_b = []
        dead_id = victim.id
        logger.log_event("CLAN_DEATH", {
            "clan": clan.id, "generation": generation, "agent": dead_id,
        })
        if verbose:
            print(f"  ☠ [{clan.id}] умер: {dead_id}")

    # 5. рождение
    new_id = None
    if dead_id is not None:
        dead = [a for a in clan.agents if a.role == "dead"]
        if dead:
            slot = dead[0]
            if random.random() < 0.5:
                parent = clan.manager
            else:
                workers = clan.get_workers()
                parent = random.choice(workers) if workers else clan.manager
            if parent.model is not None:
                clan.next_id_counter += 1
                new_id = f"K{clan.id}_G{clan.next_id_counter}"
                new_agent = parent.clone_with_mutation(
                    new_id, sigma=sigma, rng=env.rng
                )
                c_parent = parent.last_c if parent.last_c > 0 else 0.0
                c_dead = slot.last_c if slot.last_c > 0 else 0.0
                if c_parent > 0 and c_dead > 0:
                    inherited = 0.5 * (c_parent + c_dead)
                elif c_parent > 0:
                    inherited = c_parent
                elif c_dead > 0:
                    inherited = c_dead
                else:
                    inherited = 0.0
                new_agent.last_c = inherited
                new_agent.c_history = [inherited]
                idx = clan.agents.index(slot)
                clan.agents[idx] = new_agent
                logger.log_event("CLAN_BIRTH", {
                    "clan": clan.id, "generation": generation,
                    "agent": new_id, "parent": parent.id,
                })
                if verbose:
                    print(f"  ✚ [{clan.id}] рождён: {new_id} от {parent.id}")

    # 6. дообучение
    if train_steps > 0 and verbose:
        print(f"  [{clan.id}] Дообучение ({train_steps} шагов)...")
    for agent in clan.get_alive():
        if agent.model is None:
            continue
        try:
            _fine_tune_agent(agent, train_steps, env)
        except Exception as ex:
            if verbose:
                print(f"  [WARN] fine-tune {agent.id} failed: {ex}")

    # сброс очков
    for a in clan.agents:
        a.points = 0.0
        a.last_rank = 0

    c_mean_clan = (
        float(np.mean(list(c_means.values()))) if c_means else 0.0
    )
    clan.c_history.append(c_mean_clan)

    # v14.2: бонус финалисту клана (сильнейшему)
    if len(clan.c_history) >= 10:  # только в конце (после 10 поколений)
        best_agent = clan.get_best()
        if best_agent is not None:
            best_agent.add_bonus(1.0)
            if verbose:
                print(f"  ⭐ [{clan.id}] БОНУС финалисту: {best_agent.id} +1.0")
    # также: накопим бонус по истории (только в клане)
    best_agent = clan.get_best()
    if best_agent is not None:
        best_agent.add_bonus(0.05)  # небольшой бонус за лидерство

    return {
        "clan": clan.id,
        "generation": generation,
        "c_means": c_means,
        "c_mean_clan": c_mean_clan,
        "manager": clan.manager.id,
        "displace": displace,
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

def train_eva_clans(
    config: OmegaConfig,
    seeds: list,
    generations: int,
    n_clans: int,
    agents_per_clan: int,
    final_generations: int,
    episodes_per_gen: int,
    train_steps_per_gen: int,
    output_dir: str,
    branches_per_agent: int = 100,
    sigma: float = 0.01,
    fixed_branches: int = 50,
) -> list:
    """v14.0: 4 клана × 4 → финал 4.

    Этап 1: N кланов изолированно, M поколений.
    Этап 2: финалисты (по 1 из клана), K поколений.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger = WorldLogger()
    rng = np.random.default_rng(seeds[0])

    total_agents = n_clans * agents_per_clan
    print(f"\n>>> EVA v14.0 CLANS: {n_clans} кланов × {agents_per_clan} агентов "
          f"= {total_agents} всего")
    print(f"    STATE_DIM={STATE_DIM}, action={ACTION_DIM}D")
    print(f"    Этап 1: {generations} поколений × {n_clans} кланов")
    print(f"    Этап 2 (финал): {final_generations} поколений")
    print(f"    seeds={seeds}, σ={sigma}, fixed_branches={fixed_branches}\n")

    # создаём кланы
    clans = []
    for cid in range(n_clans):
        clan_agents = []
        for i in range(agents_per_clan):
            aid = f"C{cid}_A{i+1}"
            print(f"  Создаю агента {aid}...")
            env_i = MultiClusterEnv(
                config=copy.deepcopy(config), seed=seeds[0],
                observation_mode="global", reward_mode="coherence",
            )
            env_i.rng = rng
            tmp_agent = Agent(id=aid, role="worker", model=None)
            env_i.init_agent_branches(tmp_agent, branches_per_agent)
            env_i.agents = [tmp_agent]

            agent_env = env_i.make_agent_env(tmp_agent)
            from stable_baselines3 import PPO as _PPO
            model = _PPO(
                "MlpPolicy", agent_env, seed=seeds[0], verbose=0,
                n_steps=256, batch_size=64,
                learning_rate=3e-4, ent_coef=0.01,
            )
            model.learn(total_timesteps=int(train_steps_per_gen))
            model.save(str(output_path / "models" / f"init_{aid}"))
            loaded = _PPO.load(
                str(output_path / "models" / f"init_{aid}"),
                env=agent_env,
            )
            agent = Agent(id=aid, role="worker", model=loaded)
            env_i.init_agent_branches(agent, branches_per_agent)
            env_i.close()
            clan_agents.append(agent)

    # v14.1: начальные history_features (нули)
        for a in clan_agents:
            a.history_features = np.zeros(10, dtype=np.float32)

        clan = Clan(cid, clan_agents)
        clans.append(clan)

    # общая среда для оценки
    all_agents = []
    for c in clans:
        all_agents.extend(c.get_alive())

    eval_env = MultiClusterEnv(
        config=copy.deepcopy(config), seed=seeds[0],
        observation_mode="global", reward_mode="coherence",
        agents=all_agents,
    )
    eval_env.rng = rng
    eval_env.fixed_branches = int(fixed_branches)

    # ============================================================
    # ЭТАП 1: эволюция внутри каждого клана
    # ============================================================
    print(f"\n{'='*60}")
    print(f"ЭТАП 1: {n_clans} кланов × {agents_per_clan} агентов")
    print(f"{'='*60}")

    rows = []
    for gen in range(1, generations + 1):
        t0 = time.time()
        clan_results = []
        for clan in clans:
            r = evolve_step_clan(
                clan=clan, env=eval_env, logger=logger,
                generation=gen, episodes_per_gen=episodes_per_gen,
                train_steps=train_steps_per_gen, seeds=seeds,
                sigma=sigma, verbose=True,
            )
            clan_results.append(r)

        # сводка по кланам
        clan_cs = [r["c_mean_clan"] for r in clan_results]
        c_mean_overall = float(np.mean(clan_cs)) if clan_cs else 0.0
        elapsed = time.time() - t0

        row = {
            "generation": gen,
            "stage": "clans",
            "c_mean_overall": round(c_mean_overall, 4),
            "clan_c_means": [round(c, 4) for c in clan_cs],
            "time_sec": round(elapsed, 1),
        }
        rows.append(row)

        print(f"\n[ЭТАП 1 | GEN {gen}] C_mean_overall={c_mean_overall:.4f} "
              f"clans={[round(c,3) for c in clan_cs]} t={elapsed:.1f}s")

        # сохраняем модели
        for c in clans:
            for a in c.get_alive():
                if a.model is not None:
                    try:
                        a.model.save(str(
                            output_path / "models" / f"clan{c.id}_gen{gen}_{a.id}"
                        ))
                    except Exception:
                        pass

    # ============================================================
    # ЭТАП 2: финал
    # ============================================================
    finalists = []
    for c in clans:
        best = c.get_best()
        if best is not None:
            # v14.2: сохраняем bonus_points (НЕ сбрасываем)
            bonus = getattr(best, "bonus_points", 0.0)
            finalists.append(best)
            print(
                f"  Финалист из клана {c.id}: {best.id} "
                f"(C={best.last_c:.4f}, bonus={bonus:.2f})"
            )

    print(f"\n{'='*60}")
    print(f"ЭТАП 2: ФИНАЛ — {len(finalists)} агентов")
    print(f"{'='*60}")

    # создаём популяцию финала
    final_pop = Population(finalists)
    final_env = MultiClusterEnv(
        config=copy.deepcopy(config), seed=seeds[0],
        observation_mode="global", reward_mode="coherence",
        agents=final_pop.get_alive(),
    )
    final_env.rng = rng
    final_env.fixed_branches = int(fixed_branches)

    for gen in range(1, final_generations + 1):
        t0 = time.time()
        r = evolve_step(
            population=final_pop, env=final_env, logger=logger,
            generation=generations + gen,
            episodes_per_gen=episodes_per_gen,
            train_steps=train_steps_per_gen, seeds=seeds,
            sigma=sigma, verbose=True,
        )
        elapsed = time.time() - t0
        c_overall = (
            float(np.mean(list(r["c_means"].values())))
            if r["c_means"] else 0.0
        )
        c_best = max(r["c_means"].values()) if r["c_means"] else 0.0

        rows.append({
            "generation": generations + gen,
            "stage": "final",
            "c_mean_overall": round(c_overall, 4),
            "c_best": round(c_best, 4),
            "manager": r["manager"],
            "dead": r["dead"] or "",
            "born": r["born"] or "",
            "time_sec": round(elapsed, 1),
        })
        print(f"\n[ЭТАП 2 | GEN {gen}] C_mean={c_overall:.4f} "
              f"C_best={c_best:.4f} manager={r['manager']} t={elapsed:.1f}s")

    eval_env.close()
    final_env.close()

    # сохраняем
    logger.save(output_path / "world_history.json")
    logger.print_history()

    if rows:
        csv_path = output_path / "eva_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            # собираем все ключи
            all_keys = set()
            for row in rows:
                all_keys.update(row.keys())
            fieldnames = sorted(all_keys)
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"\n[SAVE] {csv_path}")
        print(f"[SAVE] {output_path / 'world_history.json'}")

    return rows


# ============================================================
# SMOKE
# ============================================================

def smoke_test():
    print("[SMOKE v13.5] EVA — 20D + 2 кластера")
    cfg = OmegaConfig(
        initial_branches=40, max_branches=100, episode_length=20,
        enable_merge_split=True, use_event_features=True,
    )
    apply_preset(cfg, "c")
    cfg.enable_merge_split = True
    cfg.use_event_features = True

    # среда
    env = MultiClusterEnv(cfg, seed=0, observation_mode="global",
                          reward_mode="coherence")
    print(f"[SMOKE]   env obs.shape={env.observation_space.shape}")

    # агент без модели (для проверки кластеров)
    a1 = Agent(id="A1", role="manager", model=None)
    env.init_agent_branches(a1, n_branches=10)
    print(f"[SMOKE]   A1: cluster_a={len(a1.cluster_a)}, "
          f"cluster_b={len(a1.cluster_b)}")
    print(f"[SMOKE]   A1 C={a1.coherence():.4f}  "
          f"C_A={a1.cluster_coherence(0):.4f}  "
          f"C_B={a1.cluster_coherence(1):.4f}")
    print(f"[SMOKE]   A1 separation={a1.separation():.4f}")

    # action 20D
    action = np.zeros(ACTION_DIM, dtype=np.float64)
    action[0] = 0.5
    action[20] = -0.5
    norm = env.apply_group_action(a1, action)
    print(f"[SMOKE]   apply_group_action norm={norm:.4f}")
    print(f"[SMOKE]   A1 C after action={a1.coherence():.4f}")

    # step_agents
    env.agents = [a1]
    info = env.step_agents({a1.id: action})
    print(f"[SMOKE]   step_agents: c_mean_overall={info['c_mean_overall']:.4f}")
    print(f"[SMOKE]   A1 branches={a1.n_branches()}")

    # Population
    a2 = Agent(id="A2", role="worker", model=None)
    a3 = Agent(id="A3", role="worker", model=None)
    a4 = Agent(id="A4", role="worker", model=None)
    for a in [a1, a2, a3, a4]:
        if a.n_branches() == 0:
            env.init_agent_branches(a, n_branches=10)
    pop = Population([a1, a2, a3, a4])
    print(f"[SMOKE]   population: {len(pop.agents)} агентов, "
          f"manager={pop.manager.id}")

    # имитация очков
    a1.add_points(0.50, 1.0); a1.last_rank = 1
    a2.add_points(0.48, 0.6); a2.last_rank = 2
    a3.add_points(0.45, 0.3); a3.last_rank = 3
    a4.add_points(0.40, 0.0); a4.last_rank = 4
    events = pop.update_ranks()
    print(f"[SMOKE]   ranks updated: events={events}, manager={pop.manager.id}")

    # смерть 4-го
    dead = pop.kill_weakest()
    print(f"[SMOKE]   killed: {dead}")

    # рождение — только если есть модель
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
    print("[SMOKE v13.5] OK")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Omega v14.0 CLANS (4 клана × 4 → финал 4)"
    )
    p.add_argument("--mode", choices=["smoke", "train"], default="smoke")
    p.add_argument("--clans", type=int, default=4,
                   help="число кланов")
    p.add_argument("--agents-per-clan", type=int, default=4,
                   help="агентов в каждом клане")
    p.add_argument("--branches-per-agent", type=int, default=100,
                   help="всего ветвей у агента (делится на 2 кластера)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--generations", type=int, default=10,
                   help="поколений в каждом клане")
    p.add_argument("--final-generations", type=int, default=5,
                   help="поколений финала")
    p.add_argument("--episodes-per-gen", type=int, default=3)
    p.add_argument("--train-steps-per-gen", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--complexity", choices=list(COMPLEXITY_PRESETS), default="c")
    p.add_argument("--delta-c-bonus", type=float, default=10.0)
    p.add_argument("--event-penalty", type=float, default=0.5)
    p.add_argument("--fixed-branches", type=int, default=50,
                   help="v14.0: жёстко N ветвей в каждом кластере")
    p.add_argument("--output", default="v14.0_results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    if PPO is None:
        print("[ERROR] stable-baselines3 не установлен")
        return

    cfg = OmegaConfig(initial_branches=40)
    apply_preset(cfg, args.complexity)
    cfg.enable_merge_split = True
    cfg.use_event_features = True
    cfg.event_penalty = float(args.event_penalty)
    cfg.delta_c_bonus = float(args.delta_c_bonus)
    cfg.__post_init__()

    seeds = list(range(args.seeds))

    train_eva_clans(
        config=cfg,
        seeds=seeds,
        generations=args.generations,
        n_clans=args.clans,
        agents_per_clan=args.agents_per_clan,
        final_generations=args.final_generations,
        episodes_per_gen=args.episodes_per_gen,
        train_steps_per_gen=args.train_steps_per_gen,
        output_dir=args.output,
        branches_per_agent=args.branches_per_agent,
        sigma=args.sigma,
        fixed_branches=args.fixed_branches,
    )


if __name__ == "__main__":
    main()