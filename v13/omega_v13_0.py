
"""
Omega v13.0 — structural env changes (MERGE/SPLIT).

Changes vs v12.2:
- Branch: generation, entropy, alive
- OmegaConfig: enable_merge_split, split_threshold, split_energy_cost,
  merge_distance, merge_energy_gain
- StepMetrics: splits, merges, events
- OmegaUniverse: _try_split, _try_merge, total_splits, total_merges
- OmegaV13Env (renamed from OmegaV12Env); info adds splits/merges

When enable_merge_split=False -> behaviour identical to v12.2
(old spawn_branches path is used).
When enable_merge_split=True  -> spawn_branches disabled; only SPLIT/MERGE.

Install (Python 3.13):
    pip install numpy gymnasium stable-baselines3 pandas matplotlib

Examples:
    python omega_v13_0.py --mode smoke
    python omega_v13_0.py --mode train --branches 40 --seeds 3 --train-steps 20000 --complexity c --reward coherence
    python omega_v13_0.py --mode train --branches 40 --seeds 3 --train-steps 20000 --complexity c --reward coherence --merge-split
    python omega_v13_0.py --mode show --csv v13_results/results_c_40.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
import zlib
from dataclasses import dataclass, field
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
# COMPLEXITY PRESETS
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
# CONFIG
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

    output_dir: str = "v13_results"

    # --- v13.0: MERGE / SPLIT ---
    # split_probability 0.7 + merge_energy_gain 2.0 (energy conserved on merge)
    enable_merge_split: bool = False
    split_threshold: float = 0.5
    split_energy_cost: float = 0.5
    split_probability: float = 0.7
    merge_distance: float = 0.03
    merge_energy_gain: float = 2.0
    # merge scan limit — avoid O(N^2) blow-up when N is large
    merge_max_pairs: int = 2000
    # v13.0 event features in obs (splits/merges history)
    use_event_features: bool = False
    # v13.0 event penalty: subtract from reward for each SPLIT/MERGE event
    # reward -= event_penalty * (splits + merges) / max_branches
    event_penalty: float = 0.0
    # v13.0 delta C bonus: reward += delta_c_bonus * (C_t - C_{t-1})
    # positive when C grows, negative when C drops
    delta_c_bonus: float = 0.0
    # v13.0 std penalty: reward -= alpha_std * std(local_coherence over branches)
    # encourages branch-level coherence
    alpha_std: float = 0.0
    # v13.0 merge_split_every: run MERGE/SPLIT only every K steps (K>=1)
    # K=1 -> every step (original), K=5 -> once per 5 steps (less noise for PPO)
    merge_split_every: int = 1
    # v13.0 LR schedule: lr_end == lr_start -> constant LR (no decay)
    # set lr_end < lr_start to enable linear decay
    lr_start: float = 3e-4
    lr_end: float = 3e-4
    # v13.0 PPO params (baseline values that produced best ON result: 0.376)
    ppo_ent_coef: float = 0.01
    ppo_target_kl: float = 0.0   # 0.0 = disabled
    ppo_n_steps: int = 256
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    # v13.0 diagnostic: disable MERGE entirely (SPLIT-only mode)
    disable_merge: bool = False
    # v13.0 policy type: "mlp" or "gru" (RecurrentPPO with MlpLstmPolicy)
    policy: str = "mlp"

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
    # --- v13.0 ---
    generation: int = 0
    entropy: float = 0.0
    alive: bool = True

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
    # --- v13.0 ---
    splits: int = 0
    merges: int = 0
    events: tuple = field(default_factory=tuple)



# ============================================================
# UNIVERSE
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
            )

        self.invalidate_cache()

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

    def apply_action(self, action: np.ndarray) -> None:
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

    # ---------- v13.0: SPLIT ----------

    def _try_split(self, capacity: int) -> tuple[int, list[str]]:
        """Split branches with energy >= split_threshold.

        v13.0 design: child is born *toward* the global mean state
        (coherence-preserving mutation), not away from it. This keeps
        global C from collapsing when N grows.

        Parent loses split_energy_cost fraction of its energy,
        child inherits (1 - split_energy_cost) of parent's energy.
        """
        if capacity <= 0 or not self.branches:
            return 0, []

        # root direction used as attractor for children
        if "root" in self.branches:
            root_state = self.branches["root"].state
            root_norm = float(np.linalg.norm(root_state))
            mean_dir = root_state / root_norm if root_norm > EPS else None
        else:
            mean_dir = None

        alpha = 0.05       # pull toward mean
        noise_scale = 0.05 # small random component

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

            # child state: move toward mean, then small noise
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
            )
            self.branches[child_id] = child
            splits += 1
            capacity -= 1
            events.append(f"SPLIT:{parent.id}->{child_id}")

        return splits, events

    # ---------- v13.0: MERGE ----------

    def _try_merge(self) -> tuple[int, list[str]]:
        """Merge branches whose cosine distance < merge_distance.

        Pairs are examined in ascending energy order; merge_max_pairs caps
        the number of pair evaluations to avoid O(N^2) blow-up.
        """
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
                # v13.0 fix: do not merge parent with its direct child
                # (would create SPLIT->MERGE->SPLIT cycle)
                if a.parent == b.id or b.parent == a.id:
                    continue
                evaluated += 1

                dot = float(np.dot(a.state, b.state))
                distance = 1.0 - dot
                if distance >= self.config.merge_distance:
                    continue

                # merge b into a
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

                # kill b
                b.alive = False
                del self.branches[b.id]
                consumed.add(b.id)
                merges += 1
                events.append(f"MERGE:{b.id}->{a.id}")

        if merges:
            self.invalidate_cache()

        return merges, events

    # ---------- v13.0: evolution ----------

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

        # 1. update energy / coherence / age
        for branch in list(self.branches.values()):
            c = self._simulate_branch_coherence(branch)
            branch.coherence = c
            branch.entropy = 0.0  # per-branch entropy: reserved for v13.1
            branch.age += 1
            branch.energy += self.config.energy_gain * c
            branch.energy -= self.config.energy_decay
            branch.energy = float(np.clip(branch.energy, 0.0, 1.5))

        # 2. deaths
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

        # 3a. always: v12.2 spawn path (baseline behaviour)
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
            )
            self.branches[child_id] = child
            parent.energy *= 0.92
            births += 1
            capacity -= 1

        # 3b. optional: MERGE first (reduces N), then SPLIT (grows N)
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

    def step(self, action: np.ndarray) -> StepMetrics:
        self.apply_action(action)
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

        return StepMetrics(
            coherence=c,
            entropy=h,
            alive=alive,
            births=evo["births"],
            deaths=evo["deaths"],
            mean_energy=mean_energy,
            action_norm=float(np.linalg.norm(action)),
            mean_pairwise_cosine=mean_cos,
            extinct=(alive == 0),
            splits=evo["splits"],
            merges=evo["merges"],
            events=evo["events"],

        )



# ============================================================
# ENVIRONMENT
# ============================================================

class OmegaV13Env(gym.Env):
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

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        obs_dim = 7 if observation_mode == "local" else 13
        if observation_mode == "global" and self.config.use_event_features:
            obs_dim += 5  # splits/merges last, recent3, alive_delta
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.world: Optional[OmegaUniverse] = None
        self.steps = 0
        self.prev_coherence = 0.0
        self.prev_alive = float(self.config.initial_branches)
        # v13.0 event history: last 3 steps (splits, merges), newest first
        self._event_history: list[tuple[int, int]] = [(0, 0), (0, 0), (0, 0)]
        self._prev_alive_int = int(self.config.initial_branches)

    def _root(self) -> np.ndarray:
        if self.world is None or "root" not in self.world.branches:
            return np.zeros(STATE_DIM, dtype=np.float32)
        return self.world.branches["root"].state.astype(np.float32)

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
            return np.concatenate(
                [np.asarray([c, h], dtype=np.float32), root]
            ).astype(np.float32)

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
            sp_recent = float(sum(h[0] for h in self._event_history))
            mg_recent = float(sum(h[1] for h in self._event_history))
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
            return np.concatenate([base, event_part]).astype(np.float32)

        return base

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

        metrics = self.world.step(action)

        d_coherence = metrics.coherence - self.prev_coherence
        survival = metrics.alive / max(1, self.config.initial_branches)

        # v13.0 event penalty (applies to all reward modes)
        event_cost = 0.0
        if self.config.event_penalty > 0.0:
            n_events = int(metrics.splits) + int(metrics.merges)
            event_cost = (
                self.config.event_penalty
                * float(n_events)
                / max(1, self.config.max_branches)
            )

        # v13.0 delta C bonus
        delta_c = float(d_coherence)
        dc_bonus = float(getattr(self.config, "delta_c_bonus", 0.0)) * delta_c

        # v13.0 std penalty on local coherence across branches
        std_penalty = 0.0
        alpha_std = float(getattr(self.config, "alpha_std", 0.0))
        if alpha_std > 0.0 and self.world is not None and self.world.branches:
            local_cs = np.asarray(
                [b.coherence for b in self.world.branches.values()],
                dtype=np.float64,
            )
            if local_cs.size > 1:
                std_penalty = alpha_std * float(local_cs.std())

        if self.reward_mode == "coherence":
            reward = (
                self.config.reward_coherence * metrics.coherence
                + dc_bonus
                - event_cost
                - std_penalty
            )
        elif self.reward_mode == "survival":
            reward = (
                self.config.reward_survival * survival
                - event_cost
            )
        else:
            reward = (
                self.config.reward_coherence * metrics.coherence
                + self.config.reward_survival * survival
                - self.config.reward_entropy * metrics.entropy
                + 0.5 * d_coherence
                - self.config.reward_action_cost * metrics.action_norm
                - event_cost
            )

            if (
                self.config.target_coherence is not None
                and self.config.coherence_deviation > 0.0
            ):
                dev = metrics.coherence - self.config.target_coherence
                reward -= self.config.coherence_deviation * (dev * dev)

            if self.config.uniform_action_penalty > 0.0:
                action_std = float(np.std(action))
                reward -= self.config.uniform_action_penalty * (
                    1.0 - action_std
                )

            if self.config.similarity_penalty > 0.0:
                reward -= (
                    self.config.similarity_penalty
                    * metrics.mean_pairwise_cosine
                )

        self.prev_coherence = metrics.coherence
        self.prev_alive = float(metrics.alive)
        self._prev_alive_int = int(metrics.alive)
        # push newest event at the front, drop oldest
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
        }

        return self._obs(), float(reward), terminated, truncated, info


# ============================================================
# POLICIES
# ============================================================

class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, size=5).astype(np.float32)


class MCTSPolicy:
    def __init__(self, env: OmegaV13Env, seed: int, simulations: int, horizon: int):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.simulations = int(simulations)
        self.horizon = int(horizon)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        assert self.env.world is not None

        best_action = np.zeros(5, dtype=np.float32)
        best_score = -float("inf")
        cfg = self.env.config

        for _ in range(self.simulations):
            candidate = self.rng.uniform(-1.0, 1.0, size=5)
            world = self.env.world.clone()
            score = 0.0

            for _ in range(self.horizon):
                m = world.step(candidate)
                s = (
                    5.0 * m.coherence
                    + 1.0 * (m.alive / max(1, world.config.initial_branches))
                    - 0.02 * m.entropy
                )
                if cfg.target_coherence is not None and cfg.coherence_deviation > 0:
                    dev = m.coherence - cfg.target_coherence
                    s -= cfg.coherence_deviation * dev * dev
                if cfg.uniform_action_penalty > 0:
                    s -= cfg.uniform_action_penalty * (1.0 - float(np.std(candidate)))
                if cfg.similarity_penalty > 0:
                    s -= cfg.similarity_penalty * m.mean_pairwise_cosine
                score += s

            score /= max(1, self.horizon)

            if score > best_score:
                best_score = score
                best_action = candidate.astype(np.float32)

        return best_action


# ============================================================
# PPO PROGRESS CALLBACK
# ============================================================

class ProgressCallback(BaseCallback):
    """Print a progress bar every `print_every` timesteps during PPO training."""

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

def evaluate_policy(env: OmegaV13Env, policy, seed: int) -> dict:
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

        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0

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
    }


def train_ppo(
    env: OmegaV13Env,
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
        # progress_remaining goes 1.0 -> 0.0 over training
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
        # RecurrentPPO with MlpLstmPolicy; keep batch_size modest for stability
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
    env = OmegaV13Env(
        config=config,
        seed=seed,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    # config.use_event_features already carried via config

    if algorithm == "random":
        policy = RandomPolicy(seed)
    elif algorithm == "ppo":
        out_dir = Path(config.output_dir)
        tag = "ms" if config.enable_merge_split else "noms"
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
        )
        groups.setdefault(key, []).append(row)

    stats: dict[tuple, dict] = {}
    for (cplx, branches, algorithm, ms), items in groups.items():
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
        stats[(cplx, algorithm, ms)] = {
            "complexity": cplx,
            "branches": branches,
            "algorithm": algorithm,
            "enable_merge_split": ms,
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
    for (cplx, algorithm, ms), s in stats.items():
        key = f"{algorithm}_ms{ms}"
        by_complexity.setdefault(cplx, {})[key] = s

    for cplx, algs in by_complexity.items():
        branches = next(iter(algs.values()))["branches"]
        n_seeds = next(iter(algs.values()))["n_seeds"]

        print(f"\n{'=' * 84}")
        print(f"SUMMARY  complexity={cplx}  N={branches}  seeds={n_seeds}")
        print(f"{'=' * 84}\n")

        print(
            f"{'algorithm':<14} {'C_mean':>8} {'+-std':>8} "
            f"{'R_mean':>10} {'cos':>6} {'split':>7} {'merge':>7} "
            f"{'ext':>6} {'time':>9}"
        )
        print("-" * 84)
        for key in sorted(algs.keys()):
            s = algs[key]
            print(
                f"{key:<14} {s['C_mean']:>8.4f} {s['C_std']:>8.4f} "
                f"{s['R_mean']:>10.2f} {s['cos_mean']:>6.3f} "
                f"{s['splits_mean']:>7.1f} {s['merges_mean']:>7.1f} "
                f"{s['extinct_mean']:>6.1f} {s['time_mean']:>8.3f}s"
            )

        if algs:
            print(f"\n  Coherence (mean C, 0.0 .. 1.0):")
            print(f"  {'-' * 60}")
            max_c = max(s["C_mean"] for s in algs.values())
            bar_len = 40
            for key in sorted(algs.keys()):
                s = algs[key]
                filled = int(bar_len * s["C_mean"] / max(max_c, 1e-9))
                bar = "#" * filled + "." * (bar_len - filled)
                print(f"  {key:<14} |{bar}| {s['C_mean']:.4f}")

            print(f"\n  Per-seed C (sorted):")
            print(f"  {'-' * 60}")
            for key in sorted(algs.keys()):
                s = algs[key]
                vals = "  ".join(f"{v:.3f}" for v in s["C_values"])
                print(f"  {key:<14} [{vals}]")

        print(f"\n  {'-' * 60}")
        for base in ("ppo", "mcts"):
            key_off = f"{base}_ms0"
            key_on = f"{base}_ms1"
            if key_off in algs and key_on in algs:
                c_off = algs[key_off]["C_mean"]
                c_on = algs[key_on]["C_mean"]
                gap = c_on - c_off
                rel = 100.0 * gap / max(abs(c_off), 1e-9)
                sign = "+" if gap >= 0 else ""
                print(f"  {base.upper()} merge_split ON - OFF: "
                      f"{sign}{gap:.4f}  ({sign}{rel:.1f}%)")
        if "ppo_ms1" in algs and "mcts_ms1" in algs:
            gap = algs["mcts_ms1"]["C_mean"] - algs["ppo_ms1"]["C_mean"]
            rel = 100.0 * gap / max(algs["ppo_ms1"]["C_mean"], 1e-9)
            sign = "+" if gap >= 0 else ""
            print(f"  MCTS - PPO (ms=ON): {sign}{gap:.4f}  ({sign}{rel:.1f}%)")
        if "ppo_ms0" in algs and "random_ms0" in algs:
            gap = algs["ppo_ms0"]["C_mean"] - algs["random_ms0"]["C_mean"]
            rel = 100.0 * gap / max(algs["random_ms0"]["C_mean"], 1e-9)
            sign = "+" if gap >= 0 else ""
            print(f"  PPO - random (ms=OFF): {sign}{gap:.4f}  ({sign}{rel:.1f}%)")
        print(f"  {'-' * 60}")


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


def run_scaling(
    branch_list: list[int],
    seeds: list[int],
    train_steps: int,
    observation_mode: str,
    reward_mode: str,
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
):
    all_rows = []
    for branches in branch_list:
        rows = run_experiment(
            branches=branches,
            seeds=seeds,
            algorithms=["random", "ppo", "mcts"],
            observation_mode=observation_mode,
            reward_mode=reward_mode,
            train_steps=train_steps,
            output_dir=output_dir,
            complexity=complexity,
            enable_merge_split=enable_merge_split,
            use_event_features=use_event_features,
            event_penalty=event_penalty,
            merge_split_every=merge_split_every,
            lr_start=lr_start,
            lr_end=lr_end,
            disable_merge=disable_merge,
            delta_c_bonus=delta_c_bonus,
            policy=policy,
            alpha_std=alpha_std,
        )
        all_rows.extend(rows)
        print_summary(rows)

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
    path = Path(output_dir) / f"scaling_{complexity}_{tag_file}_results.csv"
    if all_rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n[SCALING] Saved combined results to {path}")


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test() -> None:
    print("[SMOKE] baseline (merge_split OFF) -- random actions -- should match v12.2...")

    cfg = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=20,
    )
    apply_preset(cfg, "baseline")
    cfg.enable_merge_split = False
    env = OmegaV13Env(cfg, seed=123, observation_mode="global", reward_mode="combined")
    obs, _ = env.reset(seed=123)
    assert obs.shape == (13,), obs.shape

    rng = np.random.default_rng(123)

    print("[SMOKE]   OFF step  C      alive  splits merges  mean_local_C  mean_energy")
    off_last_info = None
    for step_i in range(20):
        action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
        obs, reward, term, trunc, off_info = env.step(action)
        w = env.world
        if w.branches:
            lc = np.array([b.coherence for b in w.branches.values()])
            en = np.array([b.energy for b in w.branches.values()])
            mean_lc = float(lc.mean())
            mean_en = float(en.mean())
        else:
            mean_lc = mean_en = 0.0
        print(
            f"[SMOKE]   OFF {step_i+1:>3}  {off_info['coherence']:.4f} "
            f"{off_info['alive']:>5}  {off_info['splits']:>5}  {off_info['merges']:>5}   "
            f"{mean_lc:>10.4f}   {mean_en:>10.4f}"
        )
        off_last_info = off_info
        if term or trunc:
            break
    env.close()
    print(f"[SMOKE]   OFF final: C={off_last_info['coherence']:.4f} alive={off_last_info['alive']}")

    print("[SMOKE] merge_split ON -- random actions -- SPLIT should fire...")

    # v13.0: use exactly the same MERGE/SPLIT params as train (OmegaConfig defaults)
    cfg2 = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=20,
        enable_merge_split=True,
    )
    apply_preset(cfg2, "c")
    cfg2.enable_merge_split = True
    # split_threshold, split_energy_cost, split_probability, merge_distance
    # are NOT overwritten -> they come from OmegaConfig defaults (same as train)
    env2 = OmegaV13Env(cfg2, seed=123, observation_mode="global", reward_mode="coherence")
    obs, _ = env2.reset(seed=123)

    rng2 = np.random.default_rng(123)

    total_splits = 0
    total_merges = 0
    print("[SMOKE]   ON  step  C      alive  splits merges  mean_local_C  mean_energy")
    on_last_info = None
    for step_i in range(20):
        action = rng2.uniform(-1.0, 1.0, size=5).astype(np.float32)
        obs, reward, term, trunc, info2 = env2.step(action)
        total_splits += info2["splits"]
        total_merges += info2["merges"]
        w = env2.world
        if w.branches:
            lc = np.array([b.coherence for b in w.branches.values()])
            en = np.array([b.energy for b in w.branches.values()])
            mean_lc = float(lc.mean())
            mean_en = float(en.mean())
        else:
            mean_lc = mean_en = 0.0
        print(
            f"[SMOKE]   ON  {step_i+1:>3}  {info2['coherence']:.4f} "
            f"{info2['alive']:>5}  {info2['splits']:>5}  {info2['merges']:>5}   "
            f"{mean_lc:>10.4f}   {mean_en:>10.4f}"
        )
        on_last_info = info2
        if term or trunc:
            break
    env2.close()

    print(
        f"[SMOKE]   ON  final: C={on_last_info['coherence']:.4f} "
        f"alive={on_last_info['alive']} splits={total_splits} merges={total_merges}"
    )

    assert total_splits > 0, "SPLIT never fired -- check split_threshold / split_probability"

    print("[SMOKE] event features -- obs should be 18D...")
    cfg3 = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=5,
        enable_merge_split=True,
        split_threshold=0.6,
        split_energy_cost=0.5,
        split_probability=0.35,
        merge_distance=0.02,
        use_event_features=True,
    )
    apply_preset(cfg3, "c")
    cfg3.enable_merge_split = True
    cfg3.use_event_features = True
    env3 = OmegaV13Env(cfg3, seed=123, observation_mode="global", reward_mode="coherence")
    obs3, _ = env3.reset(seed=123)
    assert obs3.shape == (18,), f"expected 18D obs, got {obs3.shape}"
    for _ in range(5):
        a = np.asarray([0.3, -0.2, 0.1, -0.4, 0.0], dtype=np.float32)
        obs3, r, term, trunc, info3 = env3.step(a)
        assert obs3.shape == (18,), f"expected 18D obs, got {obs3.shape}"
        if term or trunc:
            break
    env3.close()
    print(f"[SMOKE]   events ON: obs.shape={obs3.shape}, last C={info3['coherence']:.4f}")

    print("[SMOKE] OK")


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Omega-model v13.0")
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
    p.add_argument(
        "--branch-list",
        default="10,20,40,80",
        help="comma-separated branch counts",
    )
    p.add_argument(
        "--observation",
        choices=["local", "global"],
        default="global",
    )
    p.add_argument(
        "--reward",
        choices=["coherence", "survival", "combined"],
        default="combined",
    )
    p.add_argument(
        "--merge-split",
        action="store_true",
        help="enable MERGE/SPLIT (default: OFF, v12.2 behaviour)",
    )
    p.add_argument(
        "--event-features",
        action="store_true",
        help="add splits/merges history to obs (13D -> 18D)",
    )
    p.add_argument(
        "--event-penalty",
        type=float,
        default=0.0,
        help="reward penalty per SPLIT/MERGE event (default 0.0)",
    )
    p.add_argument(
        "--merge-split-every",
        type=int,
        default=1,
        help="run MERGE/SPLIT only every K steps (default 1 = every step)",
    )
    p.add_argument(
        "--lr-start",
        type=float,
        default=3e-4,
        help="initial learning rate (default 3e-4)",
    )
    p.add_argument(
        "--lr-end",
        type=float,
        default=3e-4,
        help="final learning rate (default 3e-4 = no decay)",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="diagnostic: disable MERGE, keep SPLIT only",
    )
    p.add_argument(
        "--delta-c-bonus",
        type=float,
        default=0.0,
        help="reward bonus per unit of (C_t - C_{t-1}); default 0.0",
    )
    p.add_argument(
        "--policy",
        choices=["mlp", "gru"],
        default="mlp",
        help="PPO policy type: mlp (default) or gru (RecurrentPPO)",
    )
    p.add_argument(
        "--alpha-std",
        type=float,
        default=0.0,
        help="reward penalty on std(local_coherence); default 0.0",
    )
    p.add_argument("--output", default="v13_results")
    p.add_argument(
        "--csv",
        default=None,
        help="CSV path for --mode show",
    )
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
        )
        print_summary(rows)
        return

    if args.mode == "scaling":
        branch_list = [
            int(x.strip())
            for x in args.branch_list.split(",")
            if x.strip()
        ]
        run_scaling(
            branch_list=branch_list,
            seeds=seeds,
            train_steps=args.train_steps,
            observation_mode=args.observation,
            reward_mode=args.reward,
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
        )


if __name__ == "__main__":
    main()
