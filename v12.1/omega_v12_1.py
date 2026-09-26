"""
Omega v12.1 (rewritten) — reproducible RL experiment with task complexity.

Changes vs original v12.1:
- energy_gain reduced 0.20 -> 0.05 (population no longer saturates instantly)
- spawn_threshold raised 0.85 -> 0.90
- target_coherence + coherence_deviation: penalty for |C - target|^2
- use_branch_signs: each branch has a fixed +-1 sign vector, action is
  multiplied element-wise by it, so a single action has different effect
  on different branches
- uniform_action_penalty: penalize low std(action)
- similarity_penalty: penalize mean pairwise cosine between branches

Presets (--complexity):
    baseline : old v12.1 behaviour (energy_gain=0.20, no target, no signs)
    a        : energy_gain=0.05, target_coherence=0.5, use_branch_signs
    b        : a + uniform_action_penalty=0.5
    c        : b + similarity_penalty=0.3

Install (Python 3.13):
    pip install numpy gymnasium stable-baselines3 pandas matplotlib

Examples:
    python omega_v12_1.py --mode smoke
    python omega_v12_1.py --mode train --branches 40 --seeds 3 --complexity a
    python omega_v12_1.py --mode train --branches 40 --seeds 3 --complexity b
    python omega_v12_1.py --mode train --branches 40 --seeds 3 --complexity c
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


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

    # Reward weights.
    reward_coherence: float = 5.0
    reward_survival: float = 1.0
    reward_entropy: float = 0.02
    reward_action_cost: float = 0.01

    # v12.2 complexity knobs
    target_coherence: Optional[float] = 0.5
    coherence_deviation: float = 2.0
    use_branch_signs: bool = False
    uniform_action_penalty: float = 0.0
    similarity_penalty: float = 0.0

    # MCTS/random shooting.
    mcts_simulations: int = 10
    mcts_horizon: int = 3

    output_dir: str = "v12_results"

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


def apply_preset(cfg: OmegaConfig, preset_name: str) -> OmegaConfig:
    if preset_name not in COMPLEXITY_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available: {list(COMPLEXITY_PRESETS)}"
        )
    preset = COMPLEXITY_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(cfg, key, value)
    # Re-run validation after applying preset.
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
        self._spawn_counter = 0
        self._mean_state_cache: Optional[np.ndarray] = None
        self._pairwise_cosine_cache: Optional[float] = None
        self.reset(self.seed)

    # ---------- setup ----------

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.branches.clear()
        self.step_count = 0
        self.total_births = 0
        self.total_deaths = 0
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
                sign=self._sign_for(bid) if self.config.use_branch_signs else None,
            )

        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self._mean_state_cache = None
        self._pairwise_cosine_cache = None

    # ---------- helpers ----------

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
        # Gram matrix of normalized vectors; diagonal excluded.
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

    # ---------- dynamics ----------

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

    def evolve_population(self) -> tuple[int, int]:
        births = 0
        deaths = 0

        if not self.branches:
            self.step_count += 1
            return births, deaths

        for branch in list(self.branches.values()):
            c = self._simulate_branch_coherence(branch)
            branch.coherence = c
            branch.age += 1
            branch.energy += self.config.energy_gain * c
            branch.energy -= self.config.energy_decay
            branch.energy = float(np.clip(branch.energy, 0.0, 1.5))

        for bid in list(self.branches.keys()):
            if bid == "root":
                continue
            branch = self.branches[bid]
            if branch.energy < self.config.death_threshold:
                del self.branches[bid]
                deaths += 1

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
                sign=(
                    self._sign_for(child_id)
                    if self.config.use_branch_signs else None
                ),
            )
            self.branches[child_id] = child
            parent.energy *= 0.92
            births += 1
            capacity -= 1

        if births:
            self.invalidate_cache()

        self.total_births += births
        self.total_deaths += deaths
        self.step_count += 1
        return births, deaths

    def step(self, action: np.ndarray) -> StepMetrics:
        self.apply_action(action)
        births, deaths = self.evolve_population()

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
            births=births,
            deaths=deaths,
            mean_energy=mean_energy,
            action_norm=float(np.linalg.norm(action)),
            mean_pairwise_cosine=mean_cos,
            extinct=(alive == 0),
        )


# ============================================================
# ENVIRONMENT
# ============================================================

class OmegaV12Env(gym.Env):
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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.world: Optional[OmegaUniverse] = None
        self.steps = 0
        self.prev_coherence = 0.0
        self.prev_alive = float(self.config.initial_branches)

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
        return np.concatenate(
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

        return self._obs(), {}

    def step(self, action):
        assert self.world is not None
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, -1.0, 1.0)

        metrics = self.world.step(action)

        d_coherence = metrics.coherence - self.prev_coherence
        survival = metrics.alive / max(1, self.config.initial_branches)

        if self.reward_mode == "coherence":
            reward = self.config.reward_coherence * metrics.coherence
        elif self.reward_mode == "survival":
            reward = self.config.reward_survival * survival
        else:
            reward = (
                self.config.reward_coherence * metrics.coherence
                + self.config.reward_survival * survival
                - self.config.reward_entropy * metrics.entropy
                + 0.5 * d_coherence
                - self.config.reward_action_cost * metrics.action_norm
            )

            # Target-coherence deviation penalty.
            if (
                self.config.target_coherence is not None
                and self.config.coherence_deviation > 0.0
            ):
                dev = metrics.coherence - self.config.target_coherence
                reward -= self.config.coherence_deviation * (dev * dev)

            # Uniform-action penalty.
            if self.config.uniform_action_penalty > 0.0:
                action_std = float(np.std(action))
                reward -= self.config.uniform_action_penalty * (
                    1.0 - action_std
                )

            # Similarity penalty (mean pairwise cosine).
            if self.config.similarity_penalty > 0.0:
                reward -= (
                    self.config.similarity_penalty
                    * metrics.mean_pairwise_cosine
                )

        self.prev_coherence = metrics.coherence
        self.prev_alive = float(metrics.alive)
        self.steps += 1

        terminated = False
        truncated = self.steps >= self.config.episode_length

        info = {
            "coherence": metrics.coherence,
            "entropy": metrics.entropy,
            "alive": metrics.alive,
            "births": metrics.births,
            "deaths": metrics.deaths,
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
    """
    Lightweight Monte-Carlo random-shooting baseline.
    """

    def __init__(self, env: OmegaV12Env, seed: int, simulations: int, horizon: int):
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
# EVALUATION
# ============================================================

def evaluate_policy(env: OmegaV12Env, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)

    rewards = []
    coherence = []
    entropy = []
    alive = []
    births = []
    deaths = []
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
        "extinct_steps": int(np.sum(extinct_flags)),
        "mean_pairwise_cosine": float(np.mean(cosine)),
        "runtime_sec": float(elapsed),
    }


def train_ppo(env: OmegaV12Env, seed: int, train_steps: int, save_path: Optional[Path] = None):
    if PPO is None:
        raise RuntimeError(
            "stable-baselines3 is not installed. "
            "Run: pip install stable-baselines3"
        )

    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        verbose=0,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )
    model.learn(total_timesteps=int(train_steps))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

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
) -> dict:
    env = OmegaV12Env(
        config=config,
        seed=seed,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )

    if algorithm == "random":
        policy = RandomPolicy(seed)
    elif algorithm == "ppo":
        out_dir = Path(config.output_dir)
        save_path = out_dir / f"ppo_b{config.initial_branches}_s{seed}"
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
            "branches": config.initial_branches,
            "observation_mode": observation_mode,
            "reward_mode": reward_mode,
            "train_steps": train_steps,
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
) -> list[dict]:
    base_cfg = OmegaConfig(initial_branches=branches, output_dir=output_dir)
    apply_preset(base_cfg, complexity)
    rows = []

    for algorithm in algorithms:
        for seed in seeds:
            cfg = copy.deepcopy(base_cfg)
            print(
                f"[RUN] alg={algorithm:6s} N={branches:4d} seed={seed} "
                f"complexity={complexity}"
            )
            row = run_condition(
                algorithm=algorithm,
                seed=seed,
                config=cfg,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
                train_steps=train_steps,
            )
            rows.append(row)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = Path(output_dir) / f"results_{complexity}_{branches}.csv"

    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        return

    groups = {}
    for row in rows:
        key = (row["branches"], row["algorithm"], row.get("use_branch_signs", 0))
        groups.setdefault(key, []).append(row)

    print("\n=== SUMMARY ===")
    for (branches, algorithm, _signs), items in sorted(groups.items()):
        c = np.asarray([x["mean_coherence"] for x in items])
        r = np.asarray([x["total_reward"] for x in items])
        runtime = np.asarray([x["runtime_sec"] for x in items])
        extinct = np.asarray([x["extinct_steps"] for x in items])
        cos = np.asarray([x.get("mean_pairwise_cosine", 0.0) for x in items])

        c_mean = float(c.mean())
        c_std = float(c.std(ddof=1)) if len(c) > 1 else 0.0

        print(
            f"{algorithm:6s} N={branches:4d} "
            f"C={c_mean:.4f}±{c_std:.4f} "
            f"R={r.mean():.3f} "
            f"extinct={extinct.mean():.1f} "
            f"cos={cos.mean():.3f} "
            f"time={runtime.mean():.4f}s"
        )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test() -> None:
    print("[SMOKE] creating environment...")

    # Baseline preset (no complexity) — must behave like old v12.1.
    cfg = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=5,
    )
    apply_preset(cfg, "baseline")
    env = OmegaV12Env(cfg, seed=123, observation_mode="global", reward_mode="combined")
    obs, _ = env.reset(seed=123)
    assert obs.shape == (13,), obs.shape
    action = np.asarray([1, -1, 0.5, -0.5, 0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    assert obs.shape == (13,)
    assert 0.0 <= info["coherence"] <= 1.0
    assert info["alive"] >= 1
    assert "extinct" in info
    assert "mean_pairwise_cosine" in info
    env.close()

    # Preset "c" — all complexity knobs on.
    cfg_c = OmegaConfig(
        initial_branches=40,
        max_branches=100,
        episode_length=5,
    )
    apply_preset(cfg_c, "c")
    env_c = OmegaV12Env(cfg_c, seed=123, observation_mode="global", reward_mode="combined")
    obs, _ = env_c.reset(seed=123)
    for _ in range(5):
        obs, reward, _, _, info_c = env_c.step(
            np.asarray([1, -1, 0.5, -0.5, 0], dtype=np.float32)
        )
    assert 0.0 <= info_c["coherence"] <= 1.0
    assert "extinct" in info_c
    env_c.close()

    print(
        "[SMOKE] OK | baseline C={:.4f} alive={} | preset_c C={:.4f} alive={}".format(
            info["coherence"], info["alive"],
            info_c["coherence"], info_c["alive"],
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Omega-model v12.1 (rewritten)")
    p.add_argument(
        "--mode",
        choices=["smoke", "train", "scaling"],
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
    p.add_argument("--output", default="v12_results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
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
        )
        print_summary(rows)
        return

    if args.mode == "scaling":
        branch_list = [
            int(x.strip())
            for x in args.branch_list.split(",")
            if x.strip()
        ]
        for cplx in [args.complexity]:
            for branches in branch_list:
                rows = run_experiment(
                    branches=branches,
                    seeds=seeds,
                    algorithms=["random", "ppo", "mcts"],
                    observation_mode=args.observation,
                    reward_mode=args.reward,
                    train_steps=args.train_steps,
                    output_dir=args.output,
                    complexity=cplx,
                )
                print_summary(rows)


if __name__ == "__main__":
    main()