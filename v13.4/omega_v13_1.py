"""
Omega v13.1 — multi-agent (role-based).

Changes vs v13.0:
- Role enum: EXPLORER / STABILIZER / OPTIMIZER / OBSERVER
- Dynamic role assignment per branch (energy / distance-to-mean / local coherence)
- OmegaUniverse.apply_role_actions(actions: dict[Role, np.ndarray])
- OmegaV131Env: obs extended by role features (+8D)
- CLI: --multi-agent, --roles, --role-thresholds
- When enable_multi_agent=False -> behaviour identical to v13.0

Install (Python 3.13):
    pip install numpy gymnasium stable-baselines3 pandas matplotlib

Examples:
    python omega_v13_1.py --mode smoke
    python omega_v13_1.py --mode train --branches 40 --seeds 3 --train-steps 50000 ^
        --complexity c --reward coherence --merge-split --event-features ^
        --event-penalty 0.5 --merge-split-every 1 --delta-c-bonus 10.0 --multi-agent
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    PPO = None
    BaseCallback = object

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


EPS = 1e-12
STATE_DIM = 5


# ============================================================
# ROLES (v13.1)
# ============================================================

class Role(IntEnum):
    EXPLORER = 0      # high energy -> drives SPLIT
    STABILIZER = 1    # close to mean -> dampens noise
    OPTIMIZER = 2     # high local coherence -> pushes C up
    OBSERVER = 3      # passive / no action

    @property
    def name_lower(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_name(name: str) -> "Role":
        return Role[name.strip().upper()]

    @staticmethod
    def all_names() -> list[str]:
        return [r.name_lower for r in Role]


DEFAULT_ROLES = [Role.EXPLORER, Role.STABILIZER, Role.OPTIMIZER, Role.OBSERVER]


# ============================================================
# COMPLEXITY PRESETS (identical to v13.0)
# ============================================================

COMPLEXITY_PRESETS = {
    "baseline": {
        "energy_gain": 0.20,
        "spawn_threshold": 0.85,
        "target_coherence": None,
        "coherence_deviation": 0.0,
        "use_branch_signs": False,
        "uniform_action_penalty": 0.0,
        "similarity_penalty": 0.0,
    },
    "a": {
        "energy_gain": 0.05,
        "spawn_threshold": 0.90,
        "target_coherence": 0.5,
        "coherence_deviation": 2.0,
        "use_branch_signs": True,
        "uniform_action_penalty": 0.0,
        "similarity_penalty": 0.0,
    },
    "b": {
        "energy_gain": 0.05,
        "spawn_threshold": 0.90,
        "target_coherence": 0.5,
        "coherence_deviation": 2.0,
        "use_branch_signs": True,
        "uniform_action_penalty": 0.5,
        "similarity_penalty": 0.0,
    },
    "c": {
        "energy_gain": 0.05,
        "spawn_threshold": 0.90,
        "target_coherence": 0.5,
        "coherence_deviation": 2.0,
        "use_branch_signs": True,
        "uniform_action_penalty": 0.5,
        "similarity_penalty": 0.3,
    },
}


# ============================================================
# CONFIG (v13.0 fields + multi-agent fields)
# ============================================================

@dataclass
class OmegaConfig:
    # --- base (from v12.2) ---
    initial_branches: int = 40
    max_branches: int = 640
    episode_length: int = 100

    consciousness_density: int = 4
    noise: float = 0.05

    action_scale: float = 0.20
    energy_gain: float = 0.05
    energy_decay: float = 0.03
    spawn_threshold: float = 0.90
    death_threshold: float = 0.05
    spawn_probability: float = 0.35

    reward_coherence: float = 5.0
    reward_survival: float = 1.0
    reward_entropy: float = 0.02
    reward_action_cost: float = 0.01

    target_coherence: Optional[float] = 0.5
    coherence_deviation: float = 2.0
    use_branch_signs: bool = False
    uniform_action_penalty: float = 0.0
    similarity_penalty: float = 0.0

    mcts_simulations: int = 10
    mcts_horizon: int = 3

    output_dir: str = "v13.1_results"

    # --- v13.0: MERGE / SPLIT ---
    enable_merge_split: bool = False
    split_threshold: float = 0.5
    split_energy_cost: float = 0.5
    split_probability: float = 0.7
    merge_distance: float = 0.03
    merge_energy_gain: float = 2.0
    merge_max_pairs: int = 2000
    use_event_features: bool = False
    event_penalty: float = 0.0
    delta_c_bonus: float = 0.0
    alpha_std: float = 0.0
    merge_split_every: int = 1
    lr_start: float = 3e-4
    lr_end: float = 3e-4
    ppo_ent_coef: float = 0.01
    ppo_target_kl: float = 0.0
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    disable_merge: bool = False
    policy: str = "mlp"

    # --- v13.1: multi-agent ---
    enable_multi_agent: bool = False
    # v13.1.1: shared action — roles as context, single 5D action
    shared_action: bool = False
    # v13.1.2: ablation — hide role features from obs (8D -> 0D)
    no_role_features: bool = False
    # v13.1.2: role-specific rewards
    use_role_rewards: bool = False
    # v13.1.2: penalty for being in "bad state" (low coherence)
    bad_state_threshold: float = 0.30
    bad_state_penalty: float = 1.0
    # active roles (subset of Role names, order matters for action aggregation)
    active_roles: tuple = ("explorer", "stabilizer", "optimizer", "observer")
    # dynamic role assignment thresholds
    role_explorer_energy: float = 0.65
    role_stabilizer_dist: float = 0.15
    role_optimizer_coh: float = 0.55
    # aggregation: weight of each role's action when mixing
    role_action_weights: tuple = (1.0, 1.0, 1.0, 0.0)
    # v13.1.2: per-role reward weights
    role_reward_explorer_splits: float = 0.5
    role_reward_stabilizer_std: float = 1.0
    role_reward_optimizer_dc: float = 5.0
    role_reward_observer: float = 0.0

    def __post_init__(self):
        if self.max_branches < self.initial_branches:
            raise ValueError(
                f"max_branches ({self.max_branches}) must be >= "
                f"initial_branches ({self.initial_branches})"
            )
        if self.death_threshold >= self.spawn_threshold:
            raise ValueError(
                f"death_threshold ({self.death_threshold}) must be < "
                f"spawn_threshold ({self.spawn_threshold})"
            )
        if self.initial_branches < 2:
            raise ValueError("initial_branches must be >= 2")
        if self.episode_length < 1:
            raise ValueError("episode_length must be >= 1")
        if self.target_coherence is not None:
            if not 0.0 <= self.target_coherence <= 1.0:
                raise ValueError("target_coherence must be in [0, 1]")
        if self.split_threshold <= 0.0:
            raise ValueError("split_threshold must be > 0")
        if not 0.0 <= self.split_energy_cost < 1.0:
            raise ValueError("split_energy_cost must be in [0, 1)")
        if self.merge_distance < 0.0:
            raise ValueError("merge_distance must be >= 0")
        if self.merge_energy_gain <= 0.0:
            raise ValueError("merge_energy_gain must be > 0")
        # v13.1 validation
        valid_roles = {r.name_lower for r in Role}
        for r in self.active_roles:
            if r not in valid_roles:
                raise ValueError(
                    f"Unknown role '{r}'. Valid: {sorted(valid_roles)}"
                )
        if len(self.role_action_weights) != len(Role):
            raise ValueError(
                f"role_action_weights must have {len(Role)} entries"
            )


def apply_preset(cfg: OmegaConfig, preset_name: str) -> OmegaConfig:
    if preset_name not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(COMPLEXITY_PRESETS)}"
        )
    preset = COMPLEXITY_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg


# ============================================================
# CORE DATA
# ============================================================

@dataclass
class Branch:
    id: str
    state: np.ndarray
    depth: int = 0
    parent: Optional[str] = None
    energy: float = 0.5
    age: int = 0
    coherence: float = 0.0
    sign: Optional[np.ndarray] = None
    generation: int = 0
    entropy: float = 0.0
    alive: bool = True
    # --- v13.1 ---
    role: int = int(Role.OBSERVER)

    def clone(self) -> "Branch":
        return Branch(
            id=self.id,
            state=self.state.copy(),
            depth=self.depth,
            parent=self.parent,
            energy=self.energy,
            age=self.age,
            coherence=self.coherence,
            sign=None if self.sign is None else self.sign.copy(),
            generation=self.generation,
            entropy=self.entropy,
            alive=self.alive,
            role=self.role,
        )


@dataclass
class StepMetrics:
    coherence: float
    entropy: float
    alive: int
    births: int
    deaths: int
    mean_energy: float
    action_norm: float
    mean_pairwise_cosine: float = 0.0
    extinct: bool = False
    splits: int = 0
    merges: int = 0
    events: tuple = field(default_factory=tuple)
    # --- v13.1 ---
    role_counts: tuple = (0, 0, 0, 0)       # per-role branch counts
    role_energy: tuple = (0.0, 0.0, 0.0, 0.0)  # mean energy per role
    role_action_norm: tuple = (0.0, 0.0, 0.0, 0.0)  # mean |action| per role
# ============================================================
# UNIVERSE (v13.1: role-aware)
# ============================================================

class OmegaUniverse:
    def __init__(self, seed: int, config: OmegaConfig):
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.branches: dict[str, Branch] = {}
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
        self.total_splits = 0
        self.total_merges = 0
        self._spawn_counter = 0
        self._mean_state_cache: Optional[np.ndarray] = None
        self._pairwise_cosine_cache: Optional[float] = None
        self.reset(self.seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.branches.clear()
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
        self.total_splits = 0
        self.total_merges = 0
        self._spawn_counter = 0
        self._mean_state_cache = None
        self._pairwise_cosine_cache = None

        root_id = "root"
        root = self._random_unit_vector()
        self.branches[root_id] = Branch(
            id=root_id,
            state=root,
            depth=0,
            energy=0.75,
            generation=0,
            sign=self._sign_for(root_id) if self.config.use_branch_signs else None,
            role=int(Role.STABILIZER),
        )

        for i in range(max(0, self.config.initial_branches - 1)):
            bid = f"b{i:05d}"
            state = self._mutate(root, scale=0.65)
            self.branches[bid] = Branch(
                id=bid,
                state=state,
                depth=1,
                parent=root_id,
                energy=float(self.rng.uniform(0.35, 0.75)),
                generation=1,
                sign=self._sign_for(bid) if self.config.use_branch_signs else None,
                role=int(Role.OBSERVER),
            )

        self.invalidate_cache()
        self._assign_roles()

    def invalidate_cache(self) -> None:
        self._mean_state_cache = None
        self._pairwise_cosine_cache = None

    @staticmethod
    def _sign_for(branch_id: str) -> np.ndarray:
        h = zlib.crc32(branch_id.encode("utf-8")) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        return rng.choice([-1.0, 1.0], size=STATE_DIM).astype(np.float64)

    def _random_unit_vector(self) -> np.ndarray:
        x = self.rng.normal(size=STATE_DIM)
        n = np.linalg.norm(x)
        return x / (n if n > EPS else 1.0)

    def _mutate(self, state: np.ndarray, scale: float = 0.15) -> np.ndarray:
        x = state + self.rng.normal(0.0, scale, size=STATE_DIM)
        n = np.linalg.norm(x)
        return x / (n if n > EPS else 1.0)

    def mean_state(self) -> np.ndarray:
        if self._mean_state_cache is None:
            if not self.branches:
                self._mean_state_cache = np.zeros(STATE_DIM)
            else:
                self._mean_state_cache = np.mean(
                    [b.state for b in self.branches.values()], axis=0
                )
        return self._mean_state_cache

    def mean_pairwise_cosine(self) -> float:
        if self._pairwise_cosine_cache is not None:
            return self._pairwise_cosine_cache
        n = len(self.branches)
        if n < 2:
            self._pairwise_cosine_cache = 0.0
            return 0.0
        states = np.stack([b.state for b in self.branches.values()])
        gram = states @ states.T
        total = float(gram.sum()) - float(np.trace(gram))
        count = n * (n - 1)
        val = total / count if count > 0 else 0.0
        self._pairwise_cosine_cache = float(np.clip(val, -1.0, 1.0))
        return self._pairwise_cosine_cache

    def clone(self) -> "OmegaUniverse":
        other = OmegaUniverse.__new__(OmegaUniverse)
        other.config = copy.deepcopy(self.config)
        other.seed = self.seed
        other.rng = copy.deepcopy(self.rng)
        other.branches = {k: v.clone() for k, v in self.branches.items()}
        other.step_count = self.step_count
        other.total_births = self.total_births
        other.total_deaths = self.total_deaths
        other.total_splits = self.total_splits
        other.total_merges = self.total_merges
        other._spawn_counter = self._spawn_counter
        other._mean_state_cache = (
            None if self._mean_state_cache is None
            else self._mean_state_cache.copy()
        )
        other._pairwise_cosine_cache = self._pairwise_cosine_cache
        return other

    @staticmethod
    def normalized_coherence(states: np.ndarray) -> float:
        if len(states) == 0:
            return 0.0
        states = np.asarray(states, dtype=np.float64)
        total = states.sum(axis=0)
        denom = len(states) * float(np.sum(states * states))
        if denom <= EPS:
            return 0.0
        raw = float(np.dot(total, total) / denom)
        return float(np.clip(raw, 0.0, 1.0))

    def local_coherence(self, branch: Branch) -> float:
        if not self.branches:
            return 0.0
        mean = self.mean_state()
        nm = np.linalg.norm(mean)
        if nm <= EPS:
            return 0.0
        dot = float(np.dot(branch.state, mean / nm))
        return float(np.clip((dot + 1.0) * 0.5, 0.0, 1.0))

    def entropy(self) -> float:
        if not self.branches:
            return 0.0
        energies = np.asarray(
            [max(b.energy, 0.0) for b in self.branches.values()],
            dtype=np.float64,
        )
        total = float(energies.sum())
        if total <= EPS or len(energies) <= 1:
            return 0.0
        p = energies / total
        p = p[p > EPS]
        h = -float(np.sum(p * np.log(p)))
        hmax = math.log(len(energies))
        return float(h / hmax) if hmax > EPS else 0.0

    # ---------- v13.1: role assignment ----------

    def _assign_roles(self) -> None:
        """Dynamic role assignment per branch.

        Priority (checked in order):
          1. EXPLORER   if energy >= role_explorer_energy
          2. STABILIZER if cosine-dist(branch, mean) <= role_stabilizer_dist
          3. OPTIMIZER  if local coherence >= role_optimizer_coh
          4. OBSERVER   otherwise
        Root is always STABILIZER.
        """
        if not self.config.enable_multi_agent:
            for b in self.branches.values():
                b.role = int(Role.OBSERVER)
            return

        mean = self.mean_state()
        nm = float(np.linalg.norm(mean))
        mean_dir = mean / nm if nm > EPS else None

        cfg = self.config
        for b in self.branches.values():
            if b.id == "root":
                b.role = int(Role.STABILIZER)
                continue

            if b.energy >= cfg.role_explorer_energy:
                b.role = int(Role.EXPLORER)
                continue

            if mean_dir is not None:
                cos = float(np.dot(b.state, mean_dir))
                cos = max(-1.0, min(1.0, cos))
                dist = 1.0 - cos
            else:
                dist = 1.0

            if dist <= cfg.role_stabilizer_dist:
                b.role = int(Role.STABILIZER)
                continue

            if b.coherence >= cfg.role_optimizer_coh:
                b.role = int(Role.OPTIMIZER)
                continue

            b.role = int(Role.OBSERVER)

    def role_counts(self) -> tuple:
        counts = [0, 0, 0, 0]
        for b in self.branches.values():
            counts[int(b.role)] += 1
        return tuple(counts)

    def role_mean_energy(self) -> tuple:
        sums = [0.0, 0.0, 0.0, 0.0]
        counts = [0, 0, 0, 0]
        for b in self.branches.values():
            r = int(b.role)
            sums[r] += b.energy
            counts[r] += 1
        return tuple(
            (sums[i] / counts[i]) if counts[i] > 0 else 0.0
            for i in range(4)
        )

    # ---------- v13.1: role-aware action ----------

    def apply_role_actions(self, role_actions: dict) -> tuple:
        """Apply per-role actions to branches.

        role_actions: dict {Role or int or str: np.ndarray(5,)}
        Returns per-role mean action norm tuple (4,).
        Falls back to observer action (or zeros) for missing roles.
        """
        # normalize keys -> int
        norm: dict[int, np.ndarray] = {}
        for k, v in role_actions.items():
            if isinstance(k, Role):
                idx = int(k)
            elif isinstance(k, str):
                idx = int(Role.from_name(k))
            else:
                idx = int(k)
            arr = np.asarray(v, dtype=np.float64).reshape(-1)
            if arr.size != STATE_DIM:
                raise ValueError(
                    f"Role action must be {STATE_DIM}D, got {arr.shape}"
                )
            norm[idx] = np.clip(arr, -1.0, 1.0)

        default = norm.get(int(Role.OBSERVER), np.zeros(STATE_DIM))
        use_signs = self.config.use_branch_signs

        role_norm_sum = [0.0, 0.0, 0.0, 0.0]
        role_norm_cnt = [0, 0, 0, 0]

        for branch in self.branches.values():
            r = int(branch.role)
            action = norm.get(r, default)

            if use_signs and branch.sign is not None:
                effective = action * branch.sign
            else:
                effective = action

            x = branch.state + self.config.action_scale * effective
            noise = self.rng.normal(0.0, self.config.noise * 0.05, STATE_DIM)
            x = x + noise
            n = np.linalg.norm(x)
            branch.state = x / (n if n > EPS else 1.0)

            role_norm_sum[r] += float(np.linalg.norm(action))
            role_norm_cnt[r] += 1

        self.invalidate_cache()

        role_action_norm = tuple(
            (role_norm_sum[i] / role_norm_cnt[i]) if role_norm_cnt[i] > 0 else 0.0
            for i in range(4)
        )
        return role_action_norm

    def apply_action(self, action: np.ndarray) -> None:
        """v13.0-compatible single-action path."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != STATE_DIM:
            raise ValueError(f"Expected 5D action, got shape {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        use_signs = self.config.use_branch_signs
        for branch in self.branches.values():
            if use_signs and branch.sign is not None:
                effective = action * branch.sign
            else:
                effective = action
            x = branch.state + self.config.action_scale * effective
            noise = self.rng.normal(0.0, self.config.noise * 0.05, STATE_DIM)
            x = x + noise
            n = np.linalg.norm(x)
            branch.state = x / (n if n > EPS else 1.0)

        self.invalidate_cache()

    def _simulate_branch_coherence(self, branch: Branch) -> float:
        return self.local_coherence(branch)

    # ---------- SPLIT (v13.0, unchanged) ----------

    def _try_split(self, capacity: int) -> tuple[int, list[str]]:
        if capacity <= 0 or not self.branches:
            return 0, []

        if "root" in self.branches:
            root_state = self.branches["root"].state
            root_norm = float(np.linalg.norm(root_state))
            mean_dir = root_state / root_norm if root_norm > EPS else None
        else:
            mean_dir = None

        alpha = 0.05
        noise_scale = 0.05

        splits = 0
        events: list[str] = []
        candidates = sorted(
            self.branches.values(),
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

            self._spawn_counter += 1
            child_id = f"sp{self._spawn_counter:07d}"
            child_energy = parent.energy * (1.0 - self.config.split_energy_cost)
            parent.energy *= self.config.split_energy_cost

            if mean_dir is not None:
                x = (
                    parent.state
                    + alpha * (mean_dir - parent.state)
                    + self.rng.normal(0.0, noise_scale, size=STATE_DIM)
                )
            else:
                x = self._mutate(parent.state, scale=noise_scale)
            n = float(np.linalg.norm(x))
            child_state = x / (n if n > EPS else 1.0)

            child = Branch(
                id=child_id,
                state=child_state,
                depth=parent.depth + 1,
                parent=parent.id,
                energy=float(child_energy),
                generation=parent.generation + 1,
                sign=(
                    self._sign_for(child_id)
                    if self.config.use_branch_signs else None
                ),
                role=int(parent.role),
            )
            self.branches[child_id] = child
            splits += 1
            capacity -= 1
            events.append(f"SPLIT:{parent.id}->{child_id}")

        return splits, events

    # ---------- MERGE (v13.0, unchanged) ----------

    def _try_merge(self) -> tuple[int, list[str]]:
        if len(self.branches) < 2:
            return 0, []

        merges = 0
        events: list[str] = []
        evaluated = 0

        items = list(self.branches.values())
        consumed: set[str] = set()

        for i in range(len(items)):
            if evaluated >= self.config.merge_max_pairs:
                break
            a = items[i]
            if a.id in consumed or a.id == "root":
                continue
            for j in range(i + 1, len(items)):
                if evaluated >= self.config.merge_max_pairs:
                    break
                b = items[j]
                if b.id in consumed or b.id == "root":
                    continue
                if a.parent == b.id or b.parent == a.id:
                    continue
                evaluated += 1

                dot = float(np.dot(a.state, b.state))
                distance = 1.0 - dot
                if distance >= self.config.merge_distance:
                    continue

                new_state = a.state + b.state
                n = np.linalg.norm(new_state)
                if n > EPS:
                    a.state = new_state / n

                a.energy = float(np.clip(
                    self.config.merge_energy_gain * (a.energy + b.energy),
                    0.0,
                    1.5,
                ))
                a.age = max(a.age, b.age)
                a.generation = min(a.generation, b.generation)

                b.alive = False
                del self.branches[b.id]
                consumed.add(b.id)
                merges += 1
                events.append(f"MERGE:{b.id}->{a.id}")

        if merges:
            self.invalidate_cache()

        return merges, events

    # ---------- evolution (v13.0 + role reassign) ----------

    def evolve_population(self) -> dict:
        births = 0
        deaths = 0
        splits = 0
        merges = 0
        events: list[str] = []

        if not self.branches:
            self.step_count += 1
            return {
                "births": 0, "deaths": 0,
                "splits": 0, "merges": 0,
                "events": tuple(events),
            }

        for branch in list(self.branches.values()):
            c = self._simulate_branch_coherence(branch)
            branch.coherence = c
            branch.entropy = 0.0
            branch.age += 1
            branch.energy += self.config.energy_gain * c
            branch.energy -= self.config.energy_decay
            branch.energy = float(np.clip(branch.energy, 0.0, 1.5))

        for bid in list(self.branches.keys()):
            if bid == "root":
                continue
            branch = self.branches[bid]
            if branch.energy < self.config.death_threshold:
                branch.alive = False
                del self.branches[bid]
                deaths += 1
                events.append(f"DEATH:{bid}")

        if deaths:
            self.invalidate_cache()

        candidates = [
            b for b in self.branches.values()
            if b.energy >= self.config.spawn_threshold
        ]
        capacity = max(0, self.config.max_branches - len(self.branches))
        for parent in candidates[:capacity]:
            if capacity <= 0:
                break
            if self.rng.random() > self.config.spawn_probability:
                continue

            self._spawn_counter += 1
            child_id = f"s{self._spawn_counter:07d}"
            child = Branch(
                id=child_id,
                state=self._mutate(parent.state, scale=0.12),
                depth=parent.depth + 1,
                parent=parent.id,
                energy=float(parent.energy * 0.55),
                generation=parent.generation + 1,
                sign=(
                    self._sign_for(child_id)
                    if self.config.use_branch_signs else None
                ),
                role=int(parent.role),
            )
            self.branches[child_id] = child
            parent.energy *= 0.92
            births += 1
            capacity -= 1

        if self.config.enable_merge_split:
            k = max(1, int(self.config.merge_split_every))
            if self.step_count % k == 0:
                if not getattr(self.config, "disable_merge", False):
                    merges, merge_events = self._try_merge()
                    events.extend(merge_events)

                capacity = max(0, self.config.max_branches - len(self.branches))
                splits, split_events = self._try_split(capacity)
                events.extend(split_events)
                births += splits

        if births:
            self.invalidate_cache()

        # v13.1: reassign roles after population changes
        if self.config.enable_multi_agent:
            self._assign_roles()

        self.total_births += births
        self.total_deaths += deaths
        self.total_splits += splits
        self.total_merges += merges
        self.step_count += 1

        return {
            "births": births, "deaths": deaths,
            "splits": splits, "merges": merges,
            "events": tuple(events),
        }

    def step(self, action=None, role_actions=None) -> StepMetrics:
        """Two paths:
        - v13.0 mode: step(action) — single 5D action for all branches
        - v13.1 mode: step(role_actions={Role: 5D}) — per-role actions
        """
        role_norms = (0.0, 0.0, 0.0, 0.0)
        if role_actions is not None:
            role_norms = self.apply_role_actions(role_actions)
        elif action is not None:
            self.apply_action(action)
        else:
            raise ValueError("step() needs either action or role_actions")

        evo = self.evolve_population()

        states = (
            np.stack([b.state for b in self.branches.values()])
            if self.branches
            else np.zeros((0, STATE_DIM))
        )
        c = self.normalized_coherence(states)
        h = self.entropy()
        alive = len(self.branches)
        mean_energy = (
            float(np.mean([b.energy for b in self.branches.values()]))
            if self.branches else 0.0
        )
        mean_cos = self.mean_pairwise_cosine() if alive >= 2 else 0.0

        rc = self.role_counts()
        re = self.role_mean_energy()

        return StepMetrics(
            coherence=c,
            entropy=h,
            alive=alive,
            births=evo["births"],
            deaths=evo["deaths"],
            mean_energy=mean_energy,
            action_norm=float(sum(role_norms)) if role_actions is not None else 0.0,
            mean_pairwise_cosine=mean_cos,
            extinct=(alive == 0),
            splits=evo["splits"],
            merges=evo["merges"],
            events=evo["events"],
            role_counts=rc,
            role_energy=re,
            role_action_norm=role_norms,
        )
# ============================================================
# ENVIRONMENT (v13.1)
# ============================================================

class OmegaV131Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[OmegaConfig] = None,
        seed: int = 0,
        observation_mode: str = "global",
        reward_mode: str = "combined",
    ):
        super().__init__()
        self.config = config or OmegaConfig()
        if observation_mode not in {"local", "global"}:
            raise ValueError("observation_mode must be 'local' or 'global'")
        if reward_mode not in {"coherence", "survival", "combined"}:
            raise ValueError(
                "reward_mode must be 'coherence', 'survival', or 'combined'"
            )

        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.initial_seed = int(seed)

        # v13.1: action space depends on multi-agent + shared-action flags
        #   - single-agent:                          5D
        #   - multi-agent, shared_action=False:      5*N_roles D (20D)
        #   - multi-agent, shared_action=True:       5D (roles as context)
        if self.config.enable_multi_agent and not self.config.shared_action:
            n_roles = len(Role)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(STATE_DIM * n_roles,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
            )

        obs_dim = 7 if observation_mode == "local" else 13
        if observation_mode == "global" and self.config.use_event_features:
            obs_dim += 5
        if self.config.enable_multi_agent and not self.config.no_role_features:
            obs_dim += 8  # 4 role counts (normalized) + 4 role mean energies
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.world: Optional[OmegaUniverse] = None
        self.steps = 0
        self.prev_coherence = 0.0
        self.prev_alive = float(self.config.initial_branches)
        self._event_history: list[tuple[int, int]] = [(0, 0), (0, 0), (0, 0)]
        self._prev_alive_int = int(self.config.initial_branches)

    def _root(self) -> np.ndarray:
        if self.world is None or "root" not in self.world.branches:
            return np.zeros(STATE_DIM, dtype=np.float32)
        return self.world.branches["root"].state.astype(np.float32)

    def _role_features(self) -> np.ndarray:
        assert self.world is not None
        rc = self.world.role_counts()
        re = self.world.role_mean_energy()
        n = max(1, len(self.world.branches))
        counts_norm = np.asarray([c / n for c in rc], dtype=np.float32)
        energies = np.asarray(re, dtype=np.float32)
        return np.concatenate([counts_norm, energies]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        assert self.world is not None
        root = self._root()
        states = (
            np.stack([b.state for b in self.world.branches.values()])
            if self.world.branches
            else np.zeros((0, STATE_DIM))
        )
        c = self.world.normalized_coherence(states)
        h = self.world.entropy()

        if self.observation_mode == "local":
            base = np.concatenate(
                [np.asarray([c, h], dtype=np.float32), root]
            ).astype(np.float32)
            if self.config.enable_multi_agent and not self.config.no_role_features:
                return np.concatenate(
                    [base, self._role_features()]
                ).astype(np.float32)
            return base

        alive_norm = len(self.world.branches) / max(
            1, self.config.max_branches
        )
        birth_rate = self.world.total_births / max(1, self.steps)
        death_rate = self.world.total_deaths / max(1, self.steps)
        mean_energy = (
            float(np.mean([b.energy for b in self.world.branches.values()]))
            if self.world.branches else 0.0
        )
        std_energy = (
            float(np.std([b.energy for b in self.world.branches.values()]))
            if self.world.branches else 0.0
        )
        mean_local_c = (
            float(np.mean([b.coherence for b in self.world.branches.values()]))
            if self.world.branches else 0.0
        )
        base = np.concatenate(
            [
                np.asarray(
                    [
                        c,
                        h,
                        alive_norm,
                        min(birth_rate, 10.0) / 10.0,
                        min(death_rate, 10.0) / 10.0,
                        mean_energy,
                        std_energy,
                        mean_local_c,
                    ],
                    dtype=np.float32,
                ),
                root,
            ]
        ).astype(np.float32)

        if self.config.use_event_features:
            sp_last, mg_last = self._event_history[0]
            sp_recent = float(sum(hh[0] for hh in self._event_history))
            mg_recent = float(sum(hh[1] for hh in self._event_history))
            denom = max(1, self.config.max_branches)
            denom3 = max(1, 3 * self.config.max_branches)
            alive_delta = float(
                len(self.world.branches) - self._prev_alive_int
            ) / denom
            event_part = np.asarray(
                [
                    float(sp_last) / denom,
                    float(mg_last) / denom,
                    sp_recent / denom3,
                    mg_recent / denom3,
                    alive_delta,
                ],
                dtype=np.float32,
            )
            base = np.concatenate([base, event_part]).astype(np.float32)

        if self.config.enable_multi_agent and not self.config.no_role_features:
            base = np.concatenate([base, self._role_features()]).astype(
                np.float32
            )

        return base

    def _split_action(self, action: np.ndarray) -> dict:
        """Split flat action into per-role dict (multi-agent) or single 5D.

        v13.1.1: when shared_action=True, single 5D action is broadcast
        to all roles (roles act as context in obs, not as separate actors).
        """
        if not self.config.enable_multi_agent:
            return {"action": action}
        n = len(Role)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.config.shared_action:
            # v13.1.1: broadcast 5D to all roles
            if action.size != STATE_DIM:
                raise ValueError(
                    f"shared_action: expected {STATE_DIM}D, got {action.shape}"
                )
            return {int(r): action for r in Role}
        if action.size == STATE_DIM:
            return {int(r): action for r in Role}
        if action.size != STATE_DIM * n:
            raise ValueError(
                f"Multi-agent action must be {STATE_DIM * n}D, got {action.shape}"
            )
        return {
            int(r): action[i * STATE_DIM:(i + 1) * STATE_DIM]
            for i, r in enumerate(Role)
        }

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        actual_seed = self.initial_seed if seed is None else int(seed)
        self.world = OmegaUniverse(actual_seed, self.config)
        self.steps = 0

        states = np.stack(
            [b.state for b in self.world.branches.values()]
        )
        self.prev_coherence = self.world.normalized_coherence(states)
        self.prev_alive = float(len(self.world.branches))
        self._event_history = [(0, 0), (0, 0), (0, 0)]
        self._prev_alive_int = int(len(self.world.branches))

        return self._obs(), {}

    def step(self, action):
        assert self.world is not None
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        if self.config.enable_multi_agent:
            parts = self._split_action(action)
            metrics = self.world.step(role_actions=parts)
        else:
            metrics = self.world.step(action=action)

        d_coherence = metrics.coherence - self.prev_coherence
        survival = metrics.alive / max(1, self.config.initial_branches)

        event_cost = 0.0
        if self.config.event_penalty > 0.0:
            n_events = int(metrics.splits) + int(metrics.merges)
            event_cost = (
                self.config.event_penalty
                * float(n_events)
                / max(1, self.config.max_branches)
            )

        delta_c = float(d_coherence)
        dc_bonus = float(getattr(self.config, "delta_c_bonus", 0.0)) * delta_c

        std_penalty = 0.0
        alpha_std = float(getattr(self.config, "alpha_std", 0.0))
        if alpha_std > 0.0 and self.world is not None and self.world.branches:
            local_cs = np.asarray(
                [b.coherence for b in self.world.branches.values()],
                dtype=np.float64,
            )
            if local_cs.size > 1:
                std_penalty = alpha_std * float(local_cs.std())

                # v13.1.2: penalty for being in bad state (low coherence)
        bad_penalty = 0.0
        if (
            self.config.bad_state_penalty > 0.0
            and metrics.coherence < self.config.bad_state_threshold
        ):
            bad_penalty = self.config.bad_state_penalty * (
                self.config.bad_state_threshold - metrics.coherence
            )

        # v13.1.2: role-specific rewards
        role_reward = 0.0
        if (
            self.config.use_role_rewards
            and self.config.enable_multi_agent
            and self.world is not None
        ):
            rc = metrics.role_counts  # (explorer, stabilizer, optimizer, observer)
            n_total = max(1, sum(rc))

            # explorer: bonus per split, weighted by explorer fraction
            explorer_frac = rc[0] / n_total
            role_reward += (
                self.config.role_reward_explorer_splits
                * float(metrics.splits)
                * explorer_frac
            )

            # stabilizer: penalty for high std(local coherence), weighted by stabilizer fraction
            stabilizer_frac = rc[1] / n_total
            if self.world.branches:
                local_cs = np.asarray(
                    [b.coherence for b in self.world.branches.values()],
                    dtype=np.float64,
                )
                if local_cs.size > 1:
                    local_std = float(local_cs.std())
                    role_reward -= (
                        self.config.role_reward_stabilizer_std
                        * local_std
                        * stabilizer_frac
                    )

            # optimizer: bonus per delta_c, weighted by optimizer fraction
            optimizer_frac = rc[2] / n_total
            role_reward += (
                self.config.role_reward_optimizer_dc
                * float(d_coherence)
                * optimizer_frac
            )

            # observer: fixed (usually 0)
            role_reward += self.config.role_reward_observer

                # v13.1.2: penalty for being in bad state (low coherence)
        bad_penalty = 0.0
        if (
            self.config.bad_state_penalty > 0.0
            and metrics.coherence < self.config.bad_state_threshold
        ):
            bad_penalty = self.config.bad_state_penalty * (
                self.config.bad_state_threshold - metrics.coherence
            )

        # v13.1.2: role-specific rewards
        role_reward = 0.0
        if (
            self.config.use_role_rewards
            and self.config.enable_multi_agent
            and self.world is not None
        ):
            rc = metrics.role_counts  # (explorer, stabilizer, optimizer, observer)
            n_total = max(1, sum(rc))

            # explorer: bonus per split, weighted by explorer fraction
            explorer_frac = rc[0] / n_total
            role_reward += (
                self.config.role_reward_explorer_splits
                * float(metrics.splits)
                * explorer_frac
            )

            # stabilizer: penalty for high std(local coherence), weighted by stabilizer fraction
            stabilizer_frac = rc[1] / n_total
            if self.world.branches:
                local_cs = np.asarray(
                    [b.coherence for b in self.world.branches.values()],
                    dtype=np.float64,
                )
                if local_cs.size > 1:
                    local_std = float(local_cs.std())
                    role_reward -= (
                        self.config.role_reward_stabilizer_std
                        * local_std
                        * stabilizer_frac
                    )

            # optimizer: bonus per delta_c, weighted by optimizer fraction
            optimizer_frac = rc[2] / n_total
            role_reward += (
                self.config.role_reward_optimizer_dc
                * float(d_coherence)
                * optimizer_frac
            )

            # observer: fixed (usually 0)
            role_reward += self.config.role_reward_observer

        if self.reward_mode == "coherence":
            reward = (
                self.config.reward_coherence * metrics.coherence
                + dc_bonus
                - event_cost
                - std_penalty
                - bad_penalty
                + role_reward
            )
        elif self.reward_mode == "survival":
            reward = (
                self.config.reward_survival * survival
                - event_cost
                - bad_penalty
                + role_reward
            )
        else:
            reward = (
                self.config.reward_coherence * metrics.coherence
                + self.config.reward_survival * survival
                - self.config.reward_entropy * metrics.entropy
                + 0.5 * d_coherence
                - self.config.reward_action_cost * metrics.action_norm
                - event_cost
                - bad_penalty
                + role_reward
            )

        self.prev_coherence = metrics.coherence
        self.prev_alive = float(metrics.alive)
        self._prev_alive_int = int(metrics.alive)
        self._event_history = [
            (int(metrics.splits), int(metrics.merges)),
            self._event_history[0],
            self._event_history[1],
        ]
        self.steps += 1

        terminated = False
        truncated = self.steps >= self.config.episode_length

        info = {
            "coherence": metrics.coherence,
            "entropy": metrics.entropy,
            "alive": metrics.alive,
            "births": metrics.births,
            "deaths": metrics.deaths,
            "splits": metrics.splits,
            "merges": metrics.merges,
            "events": metrics.events,
            "mean_energy": metrics.mean_energy,
            "action_norm": metrics.action_norm,
            "d_coherence": d_coherence,
            "mean_pairwise_cosine": metrics.mean_pairwise_cosine,
            "extinct": metrics.extinct,
            "role_counts": metrics.role_counts,
            "role_energy": metrics.role_energy,
            "role_action_norm": metrics.role_action_norm,
        }

        return self._obs(), float(reward), terminated, truncated, info


# ============================================================
# POLICIES
# ============================================================

class RandomPolicy:
    def __init__(self, seed: int, n_roles: int = 1):
        self.rng = np.random.default_rng(seed)
        self.n_roles = int(n_roles)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        if self.n_roles > 1:
            return self.rng.uniform(
                -1.0, 1.0, size=STATE_DIM * self.n_roles
            ).astype(np.float32)
        return self.rng.uniform(-1.0, 1.0, size=STATE_DIM).astype(np.float32)


class MCTSPolicy:
    def __init__(
        self, env: OmegaV131Env, seed: int,
        simulations: int, horizon: int,
    ):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.simulations = int(simulations)
        self.horizon = int(horizon)
        if env.config.enable_multi_agent and not env.config.shared_action:
            self.n_roles = len(Role)
        else:
            self.n_roles = 1
    def _sample_action(self) -> np.ndarray:
        if self.n_roles > 1:
            return self.rng.uniform(
                -1.0, 1.0, size=STATE_DIM * self.n_roles
            )
        return self.rng.uniform(-1.0, 1.0, size=STATE_DIM)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        assert self.env.world is not None

        best_action = np.zeros(
            STATE_DIM * self.n_roles, dtype=np.float32
        )
        best_score = -float("inf")
        cfg = self.env.config

        for _ in range(self.simulations):
            candidate = self._sample_action()
            world = self.env.world.clone()
            score = 0.0

            for _ in range(self.horizon):
                if self.n_roles > 1:
                    parts = {
                        int(r): candidate[i * STATE_DIM:(i + 1) * STATE_DIM]
                        for i, r in enumerate(Role)
                    }
                    m = world.step(role_actions=parts)
                else:
                    m = world.step(action=candidate)
                s = (
                    5.0 * m.coherence
                    + 1.0 * (m.alive / max(1, world.config.initial_branches))
                    - 0.02 * m.entropy
                )
                if cfg.target_coherence is not None and cfg.coherence_deviation > 0:
                    dev = m.coherence - cfg.target_coherence
                    s -= cfg.coherence_deviation * dev * dev
                if cfg.uniform_action_penalty > 0:
                    s -= cfg.uniform_action_penalty * (
                        1.0 - float(np.std(candidate))
                    )
                if cfg.similarity_penalty > 0:
                    s -= cfg.similarity_penalty * m.mean_pairwise_cosine
                score += s

            score /= max(1, self.horizon)

            if score > best_score:
                best_score = score
                best_action = candidate.astype(np.float32)

        return best_action


# ============================================================
# PPO CALLBACK
# ============================================================

class ProgressCallback(BaseCallback):
    def __init__(self, total_steps: int, seed: int, print_every: int = 2000):
        super().__init__()
        self.total_steps = int(total_steps)
        self.seed = int(seed)
        self.print_every = int(print_every)
        self.next_print = self.print_every
        self.t0 = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_print:
            return True

        elapsed = time.time() - self.t0
        progress = 100.0 * self.num_timesteps / max(1, self.total_steps)

        mean_r = 0.0
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            mean_r = float(
                np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            )

        bar_len = 20
        filled = int(bar_len * progress / 100.0)
        bar = "#" * filled + "." * (bar_len - filled)

        print(
            f"  [PPO s={self.seed}] {bar} "
            f"{self.num_timesteps:>6}/{self.total_steps} "
            f"({progress:5.1f}%) "
            f"R={mean_r:8.2f} "
            f"t={elapsed:6.1f}s"
        )
        self.next_print += self.print_every
        return True


# ============================================================
# EVALUATION
# ============================================================

def evaluate_policy(env: OmegaV131Env, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)

    rewards = []
    coherence = []
    entropy = []
    alive = []
    births = []
    deaths = []
    splits = []
    merges = []
    extinct_flags = []
    cosine = []
    role_counts_acc = []
    t0 = time.perf_counter()

    for _ in range(env.config.episode_length):
        action = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        coherence.append(float(info["coherence"]))
        entropy.append(float(info["entropy"]))
        alive.append(int(info["alive"]))
        births.append(int(info["births"]))
        deaths.append(int(info["deaths"]))
        splits.append(int(info.get("splits", 0)))
        merges.append(int(info.get("merges", 0)))
        extinct_flags.append(int(info.get("extinct", False)))
        cosine.append(float(info.get("mean_pairwise_cosine", 0.0)))
        role_counts_acc.append(info.get("role_counts", (0, 0, 0, 0)))

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0

    if role_counts_acc:
        rc_arr = np.asarray(role_counts_acc, dtype=np.float64)
        role_means = rc_arr.mean(axis=0).tolist()
    else:
        role_means = [0.0, 0.0, 0.0, 0.0]

    return {
        "seed": seed,
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "final_coherence": float(coherence[-1]),
        "mean_coherence": float(np.mean(coherence)),
        "max_coherence": float(np.max(coherence)),
        "mean_entropy": float(np.mean(entropy)),
        "final_alive": int(alive[-1]),
        "mean_alive": float(np.mean(alive)),
        "births": int(np.sum(births)),
        "deaths": int(np.sum(deaths)),
        "splits": int(np.sum(splits)),
        "merges": int(np.sum(merges)),
        "extinct_steps": int(np.sum(extinct_flags)),
        "mean_pairwise_cosine": float(np.mean(cosine)),
        "runtime_sec": float(elapsed),
        "role_counts_mean": role_means,
    }


def train_ppo(
    env: OmegaV131Env,
    seed: int,
    train_steps: int,
    save_path: Optional[Path] = None,
    verbose_progress: bool = True,
):
    if PPO is None:
        raise RuntimeError(
            "stable-baselines3 is not installed. "
            "Run: pip install stable-baselines3"
        )

    lr_start = float(getattr(env.config, "lr_start", 3e-4))
    lr_end = float(getattr(env.config, "lr_end", 3e-4))

    def lr_schedule(progress_remaining: float) -> float:
        return lr_end + (lr_start - lr_end) * progress_remaining

    n_steps = int(getattr(env.config, "ppo_n_steps", 256))
    batch_size = int(getattr(env.config, "ppo_batch_size", 64))
    if batch_size > n_steps:
        batch_size = n_steps

    target_kl_cfg = float(getattr(env.config, "ppo_target_kl", 0.0))
    target_kl = target_kl_cfg if target_kl_cfg > 0.0 else None

    policy_type = str(getattr(env.config, "policy", "mlp")).lower()

    if policy_type == "gru":
        if RecurrentPPO is None:
            raise RuntimeError(
                "sb3-contrib is not installed. Run: pip install sb3-contrib"
            )
        rnn_batch = min(batch_size, n_steps)
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            seed=seed,
            verbose=0,
            n_steps=n_steps,
            batch_size=rnn_batch,
            learning_rate=lr_schedule,
            ent_coef=float(getattr(env.config, "ppo_ent_coef", 0.01)),
            target_kl=target_kl,
            gamma=float(getattr(env.config, "ppo_gamma", 0.99)),
            gae_lambda=float(getattr(env.config, "ppo_gae_lambda", 0.95)),
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
            ),
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=lr_schedule,
            ent_coef=float(getattr(env.config, "ppo_ent_coef", 0.01)),
            target_kl=target_kl,
            gamma=float(getattr(env.config, "ppo_gamma", 0.99)),
            gae_lambda=float(getattr(env.config, "ppo_gae_lambda", 0.95)),
        )

    if verbose_progress:
        callback = ProgressCallback(
            total_steps=train_steps, seed=seed, print_every=2000
        )
        model.learn(total_timesteps=int(train_steps), callback=callback)
    else:
        model.learn(total_timesteps=int(train_steps))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

    if policy_type == "gru":
        class GRUPolicy:
            def __init__(self):
                self._lstm_states = None
                self._episode_starts = np.ones((1,), dtype=bool)

            def predict(self, obs):
                action, self._lstm_states = model.predict(
                    obs,
                    state=self._lstm_states,
                    episode_start=self._episode_starts,
                    deterministic=True,
                )
                self._episode_starts = np.zeros((1,), dtype=bool)
                return action

        return GRUPolicy()

    class PPOPolicy:
        def predict(self, obs):
            return model.predict(obs, deterministic=True)[0]

    return PPOPolicy()


def run_condition(
    algorithm: str,
    seed: int,
    config: OmegaConfig,
    observation_mode: str,
    reward_mode: str,
    train_steps: int,
    complexity: str,
) -> dict:
    env = OmegaV131Env(
        config=config,
        seed=seed,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )

    if config.enable_multi_agent and not config.shared_action:
        n_roles = len(Role)
    else:
        n_roles = 1

    if algorithm == "random":
        policy = RandomPolicy(seed, n_roles=n_roles)
    elif algorithm == "ppo":
        out_dir = Path(config.output_dir)
        tag = "ms" if config.enable_merge_split else "noms"
        tag += "_ma" if config.enable_multi_agent else "_sa"
        save_path = out_dir / (
            f"ppo_{complexity}_{tag}_b{config.initial_branches}_s{seed}"
        )
        policy = train_ppo(env, seed, train_steps, save_path=save_path)
    elif algorithm == "mcts":
        policy = MCTSPolicy(
            env,
            seed,
            simulations=config.mcts_simulations,
            horizon=config.mcts_horizon,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result = evaluate_policy(env, policy, seed)
    env.close()

    result.update(
        {
            "algorithm": algorithm,
            "complexity": complexity,
            "branches": config.initial_branches,
            "observation_mode": observation_mode,
            "reward_mode": reward_mode,
            "train_steps": train_steps,
            "enable_merge_split": int(config.enable_merge_split),
            "enable_multi_agent": int(config.enable_multi_agent),
            "target_coherence": (
                -1.0 if config.target_coherence is None
                else float(config.target_coherence)
            ),
            "use_branch_signs": int(config.use_branch_signs),
            "uniform_action_penalty": float(config.uniform_action_penalty),
            "similarity_penalty": float(config.similarity_penalty),
            "energy_gain": float(config.energy_gain),
        }
    )
    return result


# ============================================================
# EXPERIMENTS
# ============================================================

def run_experiment(
    branches: int,
    seeds: list[int],
    algorithms: list[str],
    observation_mode: str,
    reward_mode: str,
    train_steps: int,
    output_dir: str,
    complexity: str,
    enable_merge_split: bool = False,
    use_event_features: bool = False,
    event_penalty: float = 0.0,
    merge_split_every: int = 1,
    lr_start: float = 3e-4,
    lr_end: float = 3e-4,
    disable_merge: bool = False,
    delta_c_bonus: float = 0.0,
    policy: str = "mlp",
    alpha_std: float = 0.0,
    enable_multi_agent: bool = False,
    active_roles: tuple = ("explorer", "stabilizer", "optimizer", "observer"),
    shared_action: bool = False,
    no_role_features: bool = False,
    use_role_rewards: bool = False,
    bad_state_penalty: float = 0.0,
    bad_state_threshold: float = 0.30,
) -> list[dict]:
    base_cfg = OmegaConfig(initial_branches=branches, output_dir=output_dir)
    apply_preset(base_cfg, complexity)
    base_cfg.enable_merge_split = enable_merge_split
    base_cfg.use_event_features = use_event_features
    base_cfg.event_penalty = float(event_penalty)
    base_cfg.merge_split_every = int(merge_split_every)
    base_cfg.lr_start = float(lr_start)
    base_cfg.lr_end = float(lr_end)
    base_cfg.disable_merge = bool(disable_merge)
    base_cfg.delta_c_bonus = float(delta_c_bonus)
    base_cfg.policy = str(policy)
    base_cfg.alpha_std = float(alpha_std)
    base_cfg.enable_multi_agent = bool(enable_multi_agent)
    base_cfg.active_roles = tuple(active_roles)
    base_cfg.shared_action = bool(shared_action)
    base_cfg.no_role_features = bool(no_role_features)
    base_cfg.use_role_rewards = bool(use_role_rewards)
    base_cfg.bad_state_penalty = float(bad_state_penalty)
    base_cfg.bad_state_threshold = float(bad_state_threshold)
    base_cfg.__post_init__()

    rows = []

    tag = "merge_split=ON" if enable_merge_split else "merge_split=OFF"
    tag += "  events=ON" if use_event_features else "  events=OFF"
    if event_penalty > 0.0:
        tag += f"  penalty={event_penalty:g}"
    if merge_split_every > 1:
        tag += f"  ms_every={merge_split_every}"
    if disable_merge:
        tag += "  no_merge"
    if delta_c_bonus > 0.0:
        tag += f"  dc_bonus={delta_c_bonus:g}"
    if policy != "mlp":
        tag += f"  policy={policy}"
    if alpha_std > 0.0:
        tag += f"  alpha_std={alpha_std:g}"
    if enable_multi_agent:
        tag += f"  multi_agent=ON roles={','.join(active_roles)}"
        if shared_action:
            tag += "  shared_action=ON"
    else:
        tag += "  multi_agent=OFF"
    print(f"\n>>> complexity={complexity}  N={branches}  seeds={len(seeds)}  "
          f"train_steps={train_steps}  {tag}\n")

    for algorithm in algorithms:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            print(
                f"[RUN] alg={algorithm:6s} N={branches:4d} seed={seed} "
                f"complexity={complexity} {tag}"
            )
            row = run_condition(
                algorithm=algorithm,
                seed=seed,
                config=cfg,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
                train_steps=train_steps,
                complexity=complexity,
            )
            rows.append(row)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tag_file = "ms" if enable_merge_split else "noms"
    if use_event_features:
        tag_file += "_ev"
    if event_penalty > 0.0:
        tag_file += f"_pen{event_penalty:g}".replace(".", "p")
    if merge_split_every > 1:
        tag_file += f"_me{merge_split_every}"
    if disable_merge:
        tag_file += "_nomg"
    if delta_c_bonus > 0.0:
        tag_file += f"_dc{delta_c_bonus:g}".replace(".", "p")
    if policy != "mlp":
        tag_file += f"_{policy}"
    if alpha_std > 0.0:
        tag_file += f"_as{alpha_std:g}".replace(".", "p")
    if enable_multi_agent:
        tag_file += "_ma"
        if shared_action:
            tag_file += "sa"
    path = Path(output_dir) / f"results_{complexity}_{branches}_{tag_file}.csv"
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row.get("complexity", "?"),
            row["branches"],
            row["algorithm"],
            row.get("enable_merge_split", 0),
            row.get("enable_multi_agent", 0),
        )
        groups.setdefault(key, []).append(row)

    stats: dict[tuple, dict] = {}
    for (cplx, branches, algorithm, ms, ma), items in groups.items():
        c = np.asarray([x["mean_coherence"] for x in items], dtype=np.float64)
        r = np.asarray([x["total_reward"] for x in items], dtype=np.float64)
        runtime = np.asarray([x["runtime_sec"] for x in items], dtype=np.float64)
        extinct = np.asarray([x["extinct_steps"] for x in items], dtype=np.float64)
        cos = np.asarray(
            [x.get("mean_pairwise_cosine", 0.0) for x in items],
            dtype=np.float64,
        )
        sp = np.asarray([x.get("splits", 0) for x in items], dtype=np.float64)
        mg = np.asarray([x.get("merges", 0) for x in items], dtype=np.float64)
        stats[(cplx, algorithm, ms, ma)] = {
            "complexity": cplx,
            "branches": branches,
            "algorithm": algorithm,
            "enable_merge_split": ms,
            "enable_multi_agent": ma,
            "C_mean": float(c.mean()),
            "C_std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
            "C_values": sorted(c.tolist()),
            "R_mean": float(r.mean()),
            "time_mean": float(runtime.mean()),
            "extinct_mean": float(extinct.mean()),
            "cos_mean": float(cos.mean()),
            "splits_mean": float(sp.mean()),
            "merges_mean": float(mg.mean()),
            "n_seeds": len(items),
        }

    by_complexity: dict[str, dict[str, dict]] = {}
    for (cplx, algorithm, ms, ma), s in stats.items():
        key = f"{algorithm}_ms{ms}_ma{ma}"
        by_complexity.setdefault(cplx, {})[key] = s

    for cplx, algs in by_complexity.items():
        branches = next(iter(algs.values()))["branches"]
        n_seeds = next(iter(algs.values()))["n_seeds"]

        print(f"\n{'=' * 88}")
        print(f"SUMMARY  complexity={cplx}  N={branches}  seeds={n_seeds}")
        print(f"{'=' * 88}\n")

        print(
            f"{'algorithm':<18} {'C_mean':>8} {'+-std':>8} "
            f"{'R_mean':>10} {'cos':>6} {'split':>7} {'merge':>7} "
            f"{'ext':>6} {'time':>9}"
        )
        print("-" * 88)
        for key in sorted(algs.keys()):
            s = algs[key]
            print(
                f"{key:<18} {s['C_mean']:>8.4f} {s['C_std']:>8.4f} "
                f"{s['R_mean']:>10.2f} {s['cos_mean']:>6.3f} "
                f"{s['splits_mean']:>7.1f} {s['merges_mean']:>7.1f} "
                f"{s['extinct_mean']:>6.1f} {s['time_mean']:>8.3f}s"
            )

        if algs:
            print(f"\n  Coherence (mean C, 0.0 .. 1.0):")
            print(f"  {'-' * 64}")
            max_c = max(s["C_mean"] for s in algs.values())
            bar_len = 40
            for key in sorted(algs.keys()):
                s = algs[key]
                filled = int(bar_len * s["C_mean"] / max(max_c, 1e-9))
                bar = "#" * filled + "." * (bar_len - filled)
                print(f"  {key:<18} |{bar}| {s['C_mean']:.4f}")

            print(f"\n  Per-seed C (sorted):")
            print(f"  {'-' * 64}")
            for key in sorted(algs.keys()):
                s = algs[key]
                vals = "  ".join(f"{v:.3f}" for v in s["C_values"])
                print(f"  {key:<18} [{vals}]")

        print(f"\n  {'-' * 64}")
        # v13.1 headline: multi-agent ON vs OFF (both merge_split ON)
        if "ppo_ms1_ma1" in algs and "ppo_ms1_ma0" in algs:
            c_off = algs["ppo_ms1_ma0"]["C_mean"]
            c_on = algs["ppo_ms1_ma1"]["C_mean"]
            gap = c_on - c_off
            rel = 100.0 * gap / max(abs(c_off), 1e-9)
            sign = "+" if gap >= 0 else ""
            print(f"  PPO multi-agent ON - OFF (ms=ON): "
                  f"{sign}{gap:.4f}  ({sign}{rel:.1f}%)")
        if "mcts_ms1_ma1" in algs and "ppo_ms1_ma1" in algs:
            gap = algs["mcts_ms1_ma1"]["C_mean"] - algs["ppo_ms1_ma1"]["C_mean"]
            rel = 100.0 * gap / max(algs["ppo_ms1_ma1"]["C_mean"], 1e-9)
            sign = "+" if gap >= 0 else ""
            print(f"  MCTS - PPO (ms=ON, ma=ON): {sign}{gap:.4f}  ({sign}{rel:.1f}%)")
        print(f"  {'-' * 64}")


def show_csv(path: str) -> None:
    path = Path(path)
    if not path.exists():
        print(f"[SHOW] File not found: {path}")
        return

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)

    print(f"[SHOW] Loaded {len(rows)} rows from {path}")
    print_summary(rows)


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test() -> None:
    print("[SMOKE] v13.1 — single-agent path (should match v13.0 behaviour)")

    cfg = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=20,
        enable_merge_split=True,
    )
    apply_preset(cfg, "c")
    cfg.enable_merge_split = True
    cfg.enable_multi_agent = False
    env = OmegaV131Env(cfg, seed=123, observation_mode="global",
                       reward_mode="coherence")
    obs, _ = env.reset(seed=123)
    assert obs.shape == (13,), f"single-agent OFF: expected 13D, got {obs.shape}"

    rng = np.random.default_rng(123)
    total_splits = 0
    for _ in range(20):
        a = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        total_splits += info["splits"]
        if term or trunc:
            break
    env.close()
    assert total_splits > 0, "SPLIT never fired"
    print(f"[SMOKE]   single-agent OK, splits={total_splits}")

    print("[SMOKE] v13.1 — multi-agent path (role features, per-role actions)")

    cfg2 = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=20,
        enable_merge_split=True,
        use_event_features=True,
        enable_multi_agent=True,
    )
    apply_preset(cfg2, "c")
    cfg2.enable_merge_split = True
    cfg2.use_event_features = True
    cfg2.enable_multi_agent = True
    env2 = OmegaV131Env(cfg2, seed=123, observation_mode="global",
                        reward_mode="coherence")
    obs2, _ = env2.reset(seed=123)
    # 13 (base) + 5 (events) + 8 (roles) = 26D
    assert obs2.shape == (26,), f"multi-agent: expected 26D, got {obs2.shape}"
    # action space must be 5*4 = 20D
    assert env2.action_space.shape == (20,), env2.action_space.shape

    rng2 = np.random.default_rng(123)
    role_seen = [0, 0, 0, 0]
    for _ in range(20):
        a = rng2.uniform(-1.0, 1.0, size=20).astype(np.float32)
        obs2, r, term, trunc, info2 = env2.step(a)
        rc = info2.get("role_counts", (0, 0, 0, 0))
        for i in range(4):
            role_seen[i] = max(role_seen[i], rc[i])
        if term or trunc:
            break
    env2.close()

    assert obs2.shape == (26,), obs2.shape
    print(f"[SMOKE]   multi-agent OK, obs.shape={obs2.shape}, "
          f"role_counts_max={role_seen}")
    # sanity: at least EXPLORER and STABILIZER/OPTIMIZER should appear
    assert sum(1 for x in role_seen if x > 0) >= 2, (
        f"too few roles active: {role_seen}"
    )

    print("[SMOKE] v13.1 OK")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Omega-model v13.1 (multi-agent)")
    p.add_argument(
        "--mode",
        choices=["smoke", "train", "scaling", "show"],
        default="smoke",
    )
    p.add_argument("--branches", type=int, default=40)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--train-steps", type=int, default=10000)
    p.add_argument(
        "--complexity",
        choices=list(COMPLEXITY_PRESETS),
        default="baseline",
    )
    p.add_argument("--branch-list", default="10,20,40,80")
    p.add_argument("--observation", choices=["local", "global"], default="global")
    p.add_argument(
        "--reward",
        choices=["coherence", "survival", "combined"],
        default="combined",
    )
    p.add_argument("--merge-split", action="store_true")
    p.add_argument("--event-features", action="store_true")
    p.add_argument("--event-penalty", type=float, default=0.0)
    p.add_argument("--merge-split-every", type=int, default=1)
    p.add_argument("--lr-start", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=3e-4)
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--delta-c-bonus", type=float, default=0.0)
    p.add_argument("--policy", choices=["mlp", "gru"], default="mlp")
    p.add_argument("--alpha-std", type=float, default=0.0)

    # --- v13.1 multi-agent ---
    p.add_argument(
        "--multi-agent",
        action="store_true",
        help="enable role-based multi-agent (default: OFF = v13.0 behaviour)",
    )
    p.add_argument(
        "--roles",
        default="explorer,stabilizer,optimizer,observer",
        help="comma-separated active roles",
    )
    p.add_argument(
        "--shared-action",
        action="store_true",
        help="roles as context only: single 5D action broadcast to all roles",
    )
    p.add_argument(
        "--no-role-features",
        action="store_true",
        help="ablation: hide 8D role features from obs",
    )
    p.add_argument(
        "--use-role-rewards",
        action="store_true",
        help="enable per-role reward shaping",
    )
    p.add_argument(
        "--bad-state-penalty",
        type=float,
        default=0.0,
        help="penalty when coherence < bad_state_threshold (0 = off)",
    )
    p.add_argument(
        "--bad-state-threshold",
        type=float,
        default=0.30,
        help="coherence threshold for bad-state penalty",
    )

    p.add_argument("--output", default="v13.1_results")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    if args.mode == "show":
        if not args.csv:
            print("[SHOW] Please provide --csv path")
            return
        show_csv(args.csv)
        return

    seeds = list(range(args.seeds))
    active_roles = tuple(
        r.strip().lower() for r in args.roles.split(",") if r.strip()
    )

    if args.mode == "train":
        rows = run_experiment(
            branches=args.branches,
            seeds=seeds,
            algorithms=["random", "ppo", "mcts"],
            observation_mode=args.observation,
            reward_mode=args.reward,
            train_steps=args.train_steps,
            output_dir=args.output,
            complexity=args.complexity,
            enable_merge_split=args.merge_split,
            use_event_features=args.event_features,
            event_penalty=args.event_penalty,
            merge_split_every=args.merge_split_every,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            disable_merge=args.no_merge,
            delta_c_bonus=args.delta_c_bonus,
            policy=args.policy,
            alpha_std=args.alpha_std,
            enable_multi_agent=args.multi_agent,
            active_roles=active_roles,
            shared_action=args.shared_action,
            no_role_features=args.no_role_features,
            use_role_rewards=args.use_role_rewards,
            bad_state_penalty=args.bad_state_penalty,
            bad_state_threshold=args.bad_state_threshold,
        )
        print_summary(rows)
        return

    if args.mode == "scaling":
        branch_list = [
            int(x.strip()) for x in args.branch_list.split(",") if x.strip()
        ]
        all_rows = []
        for b in branch_list:
            rows = run_experiment(
                branches=b,
                seeds=seeds,
                algorithms=["random", "ppo", "mcts"],
                observation_mode=args.observation,
                reward_mode=args.reward,
                train_steps=args.train_steps,
                output_dir=args.output,
                complexity=args.complexity,
                enable_merge_split=args.merge_split,
                use_event_features=args.event_features,
                event_penalty=args.event_penalty,
                merge_split_every=args.merge_split_every,
                lr_start=args.lr_start,
                lr_end=args.lr_end,
                disable_merge=args.no_merge,
                delta_c_bonus=args.delta_c_bonus,
                policy=args.policy,
                alpha_std=args.alpha_std,
                enable_multi_agent=args.multi_agent,
                active_roles=active_roles,
                shared_action=args.shared_action,
                no_role_features=args.no_role_features,
                use_role_rewards=args.use_role_rewards,
                bad_state_penalty=args.bad_state_penalty,
                bad_state_threshold=args.bad_state_threshold,
            )
            all_rows.extend(rows)
            print_summary(rows)
        return


if __name__ == "__main__":
    main()