"""
omega_recurrent.py
Версия Omega-модели с RecurrentPPO (LSTM-память).

Отличия от omega_unified.py:
- Используется RecurrentPPO вместо обычного PPO
- Политика MlpLstmPolicy (с LSTM-слоем)
- LSTM-состояние сохраняется между шагами эпизода
- Результаты сохраняются в experiment_output_recurrent/

Запуск:
    python omega_recurrent.py

Зависимости:
    pip install "numpy<2" pandas gymnasium stable-baselines3 sb3-contrib scikit-learn matplotlib scipy
"""

from __future__ import annotations

import cProfile
import hashlib
import io
import math
import os
import pstats
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIG
# ============================================================

OMEGA_THRESHOLD = 0.75
BRANCHING_FACTOR = 3
MAX_DEPTH = 3
CONSCIOUSNESS_DENSITY = 4
DECOHERENCE_NOISE = 0.05

MAX_BRANCHING_FACTOR = 5
MAX_DEPTH_HARD = 5
MAX_TOTAL_BRANCHES = 1024
MAX_QUANTA_BUFFER = 5000

SEEDS = list(range(30))
TRAIN_STEPS = 10000
EPISODE_LENGTH = 50

PROFILE_FIRST_SEED = True

OUTDIR = "experiment_output_recurrent"
os.makedirs(OUTDIR, exist_ok=True)

EPS = 1e-12

TARGET_C = 0.3
TARGET_S = 6.0


# ============================================================
# CORE MODEL
# ============================================================

@dataclass
class Singularity:
    seed: int
    generation: int = 0
    entropy: float = 1.0

    def __repr__(self) -> str:
        return (
            f"<Singularity gen={self.generation} "
            f"seed={self.seed:#x} S={self.entropy:.3f}>"
        )


@dataclass
class ConsciousnessQuanta:
    branch_id: str
    amplitude: complex
    intensity: float = 0.0
    phase: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.phase = math.atan2(self.amplitude.imag, self.amplitude.real)
        self.intensity = abs(self.amplitude)


@dataclass
class Branch:
    id: str
    state_vector: tuple[float, ...]
    depth: int
    parent: Optional[str] = None
    impulses: list[ConsciousnessQuanta] = field(default_factory=list)
    local_field: list[ConsciousnessQuanta] = field(default_factory=list)
    sensitivity_threshold: float = 0.05
    coherence: float = 0.0

    @property
    def total_amplitude(self) -> complex:
        return sum((q.amplitude for q in self.impulses), 0j)

    @property
    def total_intensity(self) -> float:
        return float(sum(q.intensity for q in self.impulses))

    def absorb_impulses(self, incoming: list[ConsciousnessQuanta]) -> int:
        accepted = 0
        for q in incoming:
            if abs(q.amplitude) >= self.sensitivity_threshold:
                self.local_field.append(q)
                accepted += 1
        return accepted


class OmegaUniverseEngine:
    """Ядро циклической Omega-модели."""

    def __init__(self, initial_seed: Optional[int] = None):
        seed = (
            initial_seed
            if initial_seed is not None
            else random.SystemRandom().getrandbits(64)
        )
        self.current_state = Singularity(seed=seed)
        self.multiversal_sphere: dict[str, Branch] = {}
        self.consciousness_quanta: list[ConsciousnessQuanta] = []
        self.branching_factor = BRANCHING_FACTOR
        self.max_depth = MAX_DEPTH
        self.noise = DECOHERENCE_NOISE
        self.history: list[dict] = []
        self.rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.current_state.seed
        self.current_state = Singularity(seed=seed)
        self.multiversal_sphere.clear()
        self.consciousness_quanta.clear()
        self.history.clear()
        self.branching_factor = BRANCHING_FACTOR
        self.max_depth = MAX_DEPTH
        self.noise = DECOHERENCE_NOISE
        self.rng = random.Random(seed)

    def _trim_quanta_buffer(self) -> None:
        if len(self.consciousness_quanta) > MAX_QUANTA_BUFFER:
            self.consciousness_quanta = self.consciousness_quanta[
                -MAX_QUANTA_BUFFER:
            ]

    def big_bang(self, seed: Singularity) -> str:
        self.multiversal_sphere = self.generate_multiverse(seed)
        return (
            f"5D-Сфера активирована. "
            f"Веток: {len(self.multiversal_sphere)}, "
            f"глубина: {self.max_depth}"
        )

    def generate_multiverse(self, seed: Singularity) -> dict[str, Branch]:
        local_rng = random.Random(
            seed.seed ^ (seed.generation * 0x9E3779B97F4A7C15)
        )
        universe: dict[str, Branch] = {}
        root_vec = self._seed_to_vector(seed.seed)
        root = Branch(id="root", state_vector=root_vec, depth=0)
        universe[root.id] = root
        frontier = deque([root])

        while frontier and len(universe) < MAX_TOTAL_BRANCHES:
            node = frontier.popleft()
            if node.depth >= self.max_depth:
                continue
            for i in range(self.branching_factor):
                if len(universe) >= MAX_TOTAL_BRANCHES:
                    break
                child_vec = self._mutate(
                    node.state_vector, local_rng, node.depth, i
                )
                child_id = f"{node.id}.{i}"
                child = Branch(
                    id=child_id,
                    state_vector=child_vec,
                    depth=node.depth + 1,
                    parent=node.id,
                )
                universe[child_id] = child
                frontier.append(child)
        return universe

    @staticmethod
    def _seed_to_vector(seed: int) -> tuple[float, ...]:
        digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
        raw = [
            int.from_bytes(digest[i * 4:(i + 1) * 4], "big") / 2**32
            for i in range(5)
        ]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return tuple(x / norm for x in raw)

    @staticmethod
    def _mutate(
        vec: tuple[float, ...],
        rng: random.Random,
        depth: int,
        idx: int,
    ) -> tuple[float, ...]:
        scale = 1.0 / (1.0 + depth)
        new = list(vec)
        for k in range(5):
            new[k] += rng.gauss(0.0, scale) * (1.0 + 0.1 * idx)
        norm = math.sqrt(sum(x * x for x in new)) or 1.0
        return tuple(x / norm for x in new)

    def simulate_experience(self, branch_id: str) -> ConsciousnessQuanta:
        branch = self.multiversal_sphere[branch_id]
        quanta = self._simulate_consciousness(branch)
        branch.impulses.extend(quanta)
        branch.absorb_impulses(quanta)
        self.consciousness_quanta.extend(quanta)
        self._trim_quanta_buffer()
        branch.coherence = self._local_coherence(branch.impulses)
        return ConsciousnessQuanta(
            branch_id=branch_id,
            amplitude=branch.total_amplitude,
            intensity=branch.total_intensity,
            phase=0.0,
        )

    def _simulate_consciousness(
        self, branch: Branch
    ) -> list[ConsciousnessQuanta]:
        x, y, z, w, v = branch.state_vector
        result = []
        for n in range(CONSCIOUSNESS_DENSITY):
            theta = (
                x * 1.0 + y * 0.7 + z * 0.5 + w * 0.3 + v * 0.2
            ) * math.pi
            theta += n * (2 * math.pi / CONSCIOUSNESS_DENSITY)
            theta += self.rng.gauss(0.0, self.noise)
            intensity = (1.0 / (1.0 + branch.depth)) * (1.0 + 0.15 * n)
            amplitude = complex(
                intensity * math.cos(theta),
                intensity * math.sin(theta),
            )
            result.append(
                ConsciousnessQuanta(
                    branch_id=branch.id,
                    amplitude=amplitude,
                    meta={"observer": n, "depth": branch.depth},
                )
            )
        return result

    @staticmethod
    def _local_coherence(quanta: list[ConsciousnessQuanta]) -> float:
        if not quanta:
            return 0.0
        total = sum((q.amplitude for q in quanta), 0j)
        power = sum(abs(q.amplitude) ** 2 for q in quanta)
        if power <= 0:
            return 0.0
        return float(abs(total) ** 2 / power)

    def global_coherence(self) -> float:
        if not self.consciousness_quanta:
            return 0.0
        total = sum((q.amplitude for q in self.consciousness_quanta), 0j)
        power = sum(
            abs(q.amplitude) ** 2 for q in self.consciousness_quanta
        )
        if power <= 0:
            return 0.0
        return float(abs(total) ** 2 / power)

    def shannon_entropy(self, base: float = 2.0) -> float:
        if not self.consciousness_quanta:
            return 0.0
        intensities = np.asarray(
            [abs(q.amplitude) ** 2 for q in self.consciousness_quanta],
            dtype=float,
        )
        total = float(intensities.sum())
        if total <= 0:
            return 0.0
        probs = intensities / total
        probs = probs[probs > 1e-12]
        if len(probs) <= 1:
            return 0.0
        return float(-np.sum(probs * np.log(probs) / np.log(base)))

    def check_omega_state(self) -> bool:
        return self.global_coherence() >= OMEGA_THRESHOLD

    def find_perfect_branch(self) -> Branch:
        if not self.multiversal_sphere:
            raise RuntimeError("5D-Сфера пуста")
        branches = list(self.multiversal_sphere.values())
        if not self.consciousness_quanta:
            return max(branches, key=lambda b: b.total_intensity)
        if not any(b.impulses for b in branches):
            return self.multiversal_sphere.get("root", branches[0])
        global_mean = (
            sum(
                (q.amplitude for q in self.consciousness_quanta), 0j
            )
            / len(self.consciousness_quanta)
        )

        def score(branch: Branch) -> float:
            if not branch.impulses:
                return -1e9
            local_mean = branch.total_amplitude / len(branch.impulses)
            return float(
                (local_mean * global_mean.conjugate()).real
                * branch.total_intensity
            )

        return max(branches, key=score)

    def collapse_to_singularity(self, branch: Branch) -> Singularity:
        payload = (
            f"{self.current_state.seed}:"
            f"{branch.id}:"
            f"{branch.state_vector}:"
            f"{self.current_state.generation}"
        )
        new_seed = int.from_bytes(
            hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big"
        )
        return Singularity(
            seed=new_seed,
            generation=self.current_state.generation + 1,
            entropy=self.current_state.entropy * 1.05,
        )

    def coherence_collapse(self) -> Singularity:
        perfect = self.find_perfect_branch()
        new_singularity = self.collapse_to_singularity(perfect)
        self.multiversal_sphere.clear()
        self.consciousness_quanta.clear()
        self.big_bang(new_singularity)
        return new_singularity

    def enhance_simulation(self) -> None:
        self.branching_factor = min(
            self.branching_factor + 1, MAX_BRANCHING_FACTOR
        )
        self.max_depth = min(self.max_depth + 1, MAX_DEPTH_HARD)
        self.noise = max(self.noise * 0.7, 0.005)

    def micro_action(self, action: float) -> None:
        """Направленное воздействие: сдвиг всех компонент в сторону
        доминирующего знака среднего, масштаб 0.7."""
        a = float(np.clip(action, -1.0, 1.0))
        for branch in self.multiversal_sphere.values():
            sv = np.asarray(branch.state_vector, dtype=float)
            mean_sv = float(sv.mean())
            bias = a * 0.7 * math.copysign(1.0, mean_sv)
            distorted = sv + bias
            norm = float(np.linalg.norm(distorted)) or 1.0
            branch.state_vector = tuple(
                float(x) for x in distorted / norm
            )

    def run_cycle(
        self,
        max_cycles: int = 1,
        verbose: bool = False,
        enhance_on_no_omega: bool = True,
    ) -> Singularity:
        for cycle in range(max_cycles):
            if verbose:
                print(f"\n=== ЦИКЛ {cycle} === {self.current_state}")
            if not self.multiversal_sphere:
                msg = self.big_bang(self.current_state)
            else:
                msg = "5D-Сфера уже активна"
            if verbose:
                print(f"[BB] {msg}")
            for branch_id in sorted(self.multiversal_sphere):
                self.simulate_experience(branch_id)
            coherence = self.global_coherence()
            entropy = self.shannon_entropy()
            if verbose:
                print(
                    f"[Σ] impulses={len(self.consciousness_quanta)}, "
                    f"C={coherence:.4f}, S={entropy:.4f}"
                )
            collapsed = self.check_omega_state()
            self.history.append(
                {
                    "cycle": cycle,
                    "collapsed": collapsed,
                    "coherence": coherence,
                    "entropy": entropy,
                    "generation": self.current_state.generation,
                    "impulses_count": len(self.consciousness_quanta),
                    "branches_processed": len(self.multiversal_sphere),
                    "noise": self.noise,
                }
            )
            if collapsed:
                if verbose:
                    print("[Ω] ОМЕГА ДОСТИГНУТА → коллапс")
                self.current_state = self.coherence_collapse()
            elif enhance_on_no_omega:
                if verbose:
                    print("[→] Усиление симуляции")
                self.enhance_simulation()
        return self.current_state


# ============================================================
# METRICS
# ============================================================

def lyapunov_proxy(trajectory: list[float]) -> float:
    if len(trajectory) < 10:
        return float("-inf")
    arr = np.asarray(trajectory, dtype=np.float64)
    diffs = np.abs(np.diff(arr))
    diffs = diffs[diffs > EPS]
    if len(diffs) == 0:
        return float("-inf")
    return float(np.median(np.log(diffs)))


def spectral_entropy(signal: list[float]) -> float:
    if len(signal) < 16:
        return 0.0
    arr = np.asarray(signal, dtype=np.float64)
    arr = arr - arr.mean()
    if np.allclose(arr, 0.0, atol=EPS):
        return 0.0
    power = np.abs(np.fft.rfft(arr)) ** 2
    total = power.sum()
    if total < EPS:
        return 0.0
    p = power / total
    p = p[p > EPS]
    return float(-np.sum(p * np.log(p)))


def spectral_flatness(signal: list[float]) -> float:
    if len(signal) < 16:
        return 0.0
    arr = np.asarray(signal, dtype=np.float64)
    arr = arr - arr.mean()
    if np.allclose(arr, 0.0, atol=EPS):
        return 0.0
    power = np.abs(np.fft.rfft(arr)) ** 2 + EPS
    geometric = np.exp(np.mean(np.log(power)))
    arithmetic = np.mean(power)
    if arithmetic < EPS:
        return 0.0
    return float(geometric / arithmetic)


def mode_count(
    observations: np.ndarray,
    k_max: int = 8,
    n_bootstrap: int = 10,
    random_state: int = 42,
) -> int:
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score
    except ImportError:
        return -1

    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 2:
        return 1
    n_samples, n_features = observations.shape
    if n_samples < 20 or n_features == 0:
        return 1
    n_components = min(5, n_features, n_samples - 1)
    if n_components < 2:
        return 1
    try:
        X = PCA(
            n_components=n_components, random_state=random_state
        ).fit_transform(observations)
    except Exception:
        return 1

    rng = np.random.RandomState(random_state)
    selected = []
    for b in range(n_bootstrap):
        size = max(10, int(0.8 * n_samples))
        idx = rng.choice(n_samples, size=size, replace=False)
        Xb = X[idx]
        best_k = 1
        best_score = -1.0
        upper_k = min(k_max, len(Xb) - 1)
        for k in range(2, upper_k + 1):
            try:
                km = KMeans(
                    n_clusters=k, n_init=5, random_state=b
                ).fit(Xb)
                if len(set(km.labels_)) < 2:
                    continue
                score = silhouette_score(Xb, km.labels_)
                if score > best_score:
                    best_k = k
                    best_score = score
            except Exception:
                continue
        selected.append(best_k)
    return int(np.median(selected)) if selected else 1


def all_metrics(
    coherence_traj: list[float],
    entropy_traj: list[float],
    observations: np.ndarray,
) -> dict:
    return {
        "lyapunov": lyapunov_proxy(coherence_traj),
        "spectral_entropy": spectral_entropy(coherence_traj),
        "spectral_flatness": spectral_flatness(coherence_traj),
        "mode_count": mode_count(observations),
        "final_C": float(coherence_traj[-1]) if coherence_traj else 0.0,
        "final_S": float(entropy_traj[-1]) if entropy_traj else 0.0,
    }


# ============================================================
# RL ENVIRONMENT
# ============================================================

class OmegaLifeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_mode: str = "life",
        initial_seed: int = 42,
        action_mode: str = "scalar",
        episode_length: int = EPISODE_LENGTH,
    ):
        super().__init__()
        if reward_mode not in {"order", "life"}:
            raise ValueError(
                "reward_mode должен быть 'order' или 'life'"
            )
        if action_mode not in {"scalar", "vector"}:
            raise ValueError(
                "action_mode должен быть 'scalar' или 'vector'"
            )
        self.reward_mode = reward_mode
        self.initial_seed = initial_seed
        self.action_mode = action_mode
        self.episode_length = episode_length

        if action_mode == "scalar":
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        self.engine: Optional[OmegaUniverseEngine] = None
        self.state = None
        self.steps = 0

    def _get_observation(self) -> np.ndarray:
        assert self.engine is not None
        root = self.engine.multiversal_sphere.get("root")
        if root is None:
            vector = np.zeros(5, dtype=np.float32)
        else:
            vector = np.asarray(root.state_vector, dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(
                    [
                        self.engine.global_coherence(),
                        self.engine.shannon_entropy(),
                    ],
                    dtype=np.float32,
                ),
                vector,
            ]
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        actual_seed = self.initial_seed if seed is None else int(seed)
        self.engine = OmegaUniverseEngine(initial_seed=actual_seed)
        self.engine.big_bang(self.engine.current_state)
        for branch_id in sorted(self.engine.multiversal_sphere):
            self.engine.simulate_experience(branch_id)
        self.state = {
            "coherence_prev": self.engine.global_coherence(),
            "entropy_prev": self.engine.shannon_entropy(),
        }
        self.steps = 0
        return self._get_observation(), {}

    def step(self, action):
        assert self.engine is not None
        assert self.state is not None
        action = np.asarray(action, dtype=np.float64).flatten()
        if self.action_mode == "scalar":
            theta = float(np.clip(action[0], -1.0, 1.0))
        else:
            theta = float(np.clip(np.mean(action[:5]), -1.0, 1.0))

        self.engine.micro_action(theta)
        self.engine.run_cycle(
            max_cycles=1, verbose=False, enhance_on_no_omega=False
        )

        new_coherence = self.engine.global_coherence()
        new_entropy = self.engine.shannon_entropy()
        dC = new_coherence - self.state["coherence_prev"]
        dS = new_entropy - self.state["entropy_prev"]

        if self.reward_mode == "order":
            reward = dC * 100.0
        else:
            reward = (
                -abs(new_coherence - TARGET_C)
                - 0.1 * abs(new_entropy - TARGET_S)
            )

        self.state["coherence_prev"] = new_coherence
        self.state["entropy_prev"] = new_entropy
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.episode_length
        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            {"coherence": new_coherence, "entropy": new_entropy},
        )


class OmegaLifeEnvControlled(OmegaLifeEnv):
    def __init__(
        self,
        reward_mode: str = "life",
        initial_seed: int = 42,
        shuffle_reward: bool = False,
        freeze_action: bool = False,
        no_step: bool = False,
        action_mode: str = "scalar",
        episode_length: int = EPISODE_LENGTH,
    ):
        super().__init__(
            reward_mode=reward_mode,
            initial_seed=initial_seed,
            action_mode=action_mode,
            episode_length=episode_length,
        )
        self.shuffle_reward = shuffle_reward
        self.freeze_action = freeze_action
        self.no_step = no_step
        self._reward_buffer = None

    def reset(self, **kwargs):
        self._reward_buffer = None
        return super().reset(**kwargs)

    def step(self, action):
        assert self.engine is not None
        if self.no_step:
            self.steps += 1
            obs = self._get_observation()
            coh = self.engine.global_coherence()
            ent = self.engine.shannon_entropy()
            return (
                obs,
                0.0,
                False,
                self.steps >= self.episode_length,
                {"coherence": coh, "entropy": ent, "no_step": True},
            )
        if self.freeze_action:
            if self.action_mode == "scalar":
                action = np.zeros(1, dtype=np.float32)
            else:
                action = np.zeros(5, dtype=np.float32)
        obs, reward, terminated, truncated, info = super().step(action)
        if self.shuffle_reward:
            if self._reward_buffer is None:
                self._reward_buffer = 0.0
            delayed = self._reward_buffer
            self._reward_buffer = reward
            reward = delayed
        return obs, reward, terminated, truncated, info


# ============================================================
# EXPERIMENT
# ============================================================

CONDITIONS = [
    ("frozen", "order", "frozen", {"no_step": True}),
    ("random_order", "order", "random", {}),
    ("random_life", "life", "random", {}),
    ("trained_order", "order", "trained", {}),
    ("trained_life", "life", "trained", {}),
    ("shuffled_life", "life", "trained", {"shuffle_reward": True}),
    ("constant_reward", "life", "trained", {"no_step": True}),
]


def _make_env_factory(
    reward_mode: str,
    seed: int,
    controls: dict,
) -> Callable[[], OmegaLifeEnvControlled]:
    def factory() -> OmegaLifeEnvControlled:
        return OmegaLifeEnvControlled(
            reward_mode=reward_mode,
            initial_seed=seed,
            shuffle_reward=controls.get("shuffle_reward", False),
            freeze_action=controls.get("freeze_action", False),
            no_step=controls.get("no_step", False),
            action_mode="scalar",
            episode_length=EPISODE_LENGTH,
        )
    return factory


def make_policy(
    env_factory: Callable[[], OmegaLifeEnvControlled],
    policy_type: str,
    seed: int,
) -> Callable[[np.ndarray], np.ndarray]:
    if policy_type == "frozen":
        def frozen_policy(_obs):
            return np.zeros(1, dtype=np.float32)
        return frozen_policy

    if policy_type == "random":
        rng = np.random.default_rng(seed)
        def random_policy(_obs):
            return rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        return random_policy

    if policy_type == "trained":
        train_env = env_factory()
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            verbose=0,
            seed=seed,
            n_steps=256,
            batch_size=64,
            learning_rate=3e-4,
            ent_coef=0.01,
        )
        model.learn(total_timesteps=TRAIN_STEPS)
        train_env.close()

        # Для RecurrentPPO нужно хранить LSTM-состояние между шагами
        _lstm_state = {"state": None, "episode_start": True}

        def trained_policy(obs):
            action, _lstm_state["state"] = model.predict(
                obs,
                state=_lstm_state["state"],
                episode_start=np.array([_lstm_state["episode_start"]]),
                deterministic=True,
            )
            _lstm_state["episode_start"] = False
            return action
        return trained_policy

    raise ValueError(f"Неизвестная policy_type: {policy_type}")


def run_single(
    condition: str,
    reward_mode: str,
    policy_type: str,
    seed: int,
    controls: dict,
) -> dict:
    env_factory = _make_env_factory(reward_mode, seed, controls)
    eval_env = env_factory()
    obs, _ = eval_env.reset(seed=seed)
    policy = make_policy(env_factory, policy_type, seed)

    coherence_traj, entropy_traj, reward_traj, observations = [], [], [], []
    total_reward = 0.0

    for _ in range(EPISODE_LENGTH):
        action = policy(obs)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        coherence_traj.append(float(info["coherence"]))
        entropy_traj.append(float(info["entropy"]))
        reward_traj.append(float(reward))
        observations.append(np.asarray(obs, dtype=np.float32))
        total_reward += float(reward)
        if terminated or truncated:
            break

    eval_env.close()

    observations_array = (
        np.stack(observations)
        if observations
        else np.zeros((1, 7), dtype=np.float32)
    )

    metrics = all_metrics(coherence_traj, entropy_traj, observations_array)
    metrics.update(
        {
            "condition": condition,
            "seed": seed,
            "total_reward": total_reward,
            "reward_mean": (
                float(np.mean(reward_traj)) if reward_traj else 0.0
            ),
            "reward_std": (
                float(np.std(reward_traj)) if reward_traj else 0.0
            ),
            "episode_length": len(coherence_traj),
        }
    )
    return {
        "metrics": metrics,
        "coherence": np.asarray(coherence_traj, dtype=np.float32),
        "entropy": np.asarray(entropy_traj, dtype=np.float32),
        "reward": np.asarray(reward_traj, dtype=np.float32),
    }


def calibrate_omega_threshold(
    seeds=(0, 1, 2, 3, 4),
) -> float:
    values = []
    for s in seeds:
        eng = OmegaUniverseEngine(initial_seed=s)
        eng.big_bang(eng.current_state)
        for bid in sorted(eng.multiversal_sphere):
            eng.simulate_experience(bid)
        values.append(eng.global_coherence())

    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    maximum = float(values.max())
    suggested = median + 0.5 * (maximum - median)
    print(
        f"[CALIB] coherence: median={median:.4f}, "
        f"max={maximum:.4f}, suggested_threshold={suggested:.4f}"
    )
    return suggested


def profile_single_episode() -> None:
    print("\n[PROFILE] Прогон 1 эпизода (random policy)...")

    def _run():
        env = OmegaLifeEnvControlled(
            reward_mode="life",
            initial_seed=0,
            action_mode="scalar",
            episode_length=20,
        )
        obs, _ = env.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(20):
            action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()
    _run()
    t1 = time.time()
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(15)

    print(f"[PROFILE] wall_time={t1 - t0:.2f}s")
    print("[PROFILE] top-15 по cumulative:")
    print(s.getvalue())

    stats_path = os.path.join(OUTDIR, "profile.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"wall_time={t1 - t0:.4f}s\n\n")
        f.write(s.getvalue())
    print(f"[PROFILE] сохранено: {stats_path}")


def paired_ttest(
    df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    metric: str = "total_reward",
) -> dict:
    a = df[df["condition"] == cond_a].set_index("seed")[metric]
    b = df[df["condition"] == cond_b].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return {
            "cond_a": cond_a,
            "cond_b": cond_b,
            "metric": metric,
            "n": len(common),
            "t": float("nan"),
            "p": float("nan"),
            "mean_diff": float("nan"),
        }
    a_c = a.loc[common].to_numpy()
    b_c = b.loc[common].to_numpy()
    t, p = stats.ttest_rel(a_c, b_c)
    return {
        "cond_a": cond_a,
        "cond_b": cond_b,
        "metric": metric,
        "n": int(len(common)),
        "t": float(t),
        "p": float(p),
        "mean_diff": float(np.mean(a_c - b_c)),
    }


def plot_condition_summary(
    trajectories: dict,
    df: pd.DataFrame,
) -> None:
    conditions = sorted({k.split("_seed")[0] for k in trajectories})
    fig, axes = plt.subplots(
        len(conditions), 2,
        figsize=(10, 2.2 * len(conditions)),
        sharex=True,
    )
    if len(conditions) == 1:
        axes = np.array([axes])

    for i, cond in enumerate(conditions):
        c_list, s_list = [], []
        for key, data in trajectories.items():
            if key.startswith(f"{cond}_seed"):
                c_list.append(data["coherence"])
                s_list.append(data["entropy"])
        if not c_list:
            continue
        min_len = min(len(x) for x in c_list)
        C = np.stack([x[:min_len] for x in c_list])
        S = np.stack([x[:min_len] for x in s_list])
        t = np.arange(min_len)
        ax_c, ax_s = axes[i, 0], axes[i, 1]
        ax_c.plot(t, C.mean(axis=0), color="C0")
        ax_c.fill_between(
            t, C.mean(0) - C.std(0), C.mean(0) + C.std(0),
            color="C0", alpha=0.25,
        )
        ax_c.axhline(
            OMEGA_THRESHOLD, color="red", linestyle="--", linewidth=1
        )
        ax_c.set_title(f"{cond} - coherence")
        ax_c.set_ylim(0, 1.05)
        ax_s.plot(t, S.mean(axis=0), color="C1")
        ax_s.fill_between(
            t, S.mean(0) - S.std(0), S.mean(0) + S.std(0),
            color="C1", alpha=0.25,
        )
        ax_s.set_title(f"{cond} - entropy")
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    plt.tight_layout()
    path = os.path.join(OUTDIR, "trajectories.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[PLOT] сохранено: {path}")


def main():
    print("=" * 72)
    print("ОМЕГА-ЭКСПЕРИМЕНТ (v4: RecurrentPPO)")
    print("=" * 72)
    if PROFILE_FIRST_SEED:
        profile_single_episode()
    suggested = calibrate_omega_threshold()
    global OMEGA_THRESHOLD
    OMEGA_THRESHOLD = suggested
    print(f"[CFG] OMEGA_THRESHOLD = {OMEGA_THRESHOLD:.4f}")
    rows = []
    trajectories = {}
    for condition, mode, policy, controls in CONDITIONS:
        print(f"\n=== {condition} (reward={mode}, policy={policy}) ===")
        for seed in SEEDS:
            try:
                result = run_single(
                    condition, mode, policy, seed, controls
                )
            except Exception as exc:
                print(f"seed={seed}: ERROR {type(exc).__name__}: {exc}")
                continue
            metrics = result["metrics"]
            rows.append(metrics)
            key = f"{condition}_seed{seed}"
            trajectories[key] = {
                "coherence": result["coherence"],
                "entropy": result["entropy"],
                "reward": result["reward"],
            }
            lam = metrics["lyapunov"]
            lam_text = f"{lam:.4f}" if np.isfinite(lam) else "-inf"
            print(
                f"seed={seed}: "
                f"R={metrics['total_reward']:.4f}, "
                f"C={metrics['final_C']:.4f}, "
                f"S={metrics['final_S']:.4f}, "
                f"SE={metrics['spectral_entropy']:.4f}, "
                f"SF={metrics['spectral_flatness']:.4f}, "
                f"modes={metrics['mode_count']}, "
                f"lambda_proxy={lam_text}"
            )
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "experiment_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    npz_path = os.path.join(OUTDIR, "trajectories.npz")
    np.savez(
        npz_path,
        **{
            f"{key}_{name}": values
            for key, data in trajectories.items()
            for name, values in data.items()
        },
    )
    print("\n" + "=" * 72)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ (парный t-test по сидам)")
    print("=" * 72)
    test_pairs = [
        ("trained_order", "random_order"),
        ("trained_life", "random_life"),
        ("trained_life", "shuffled_life"),
        ("trained_life", "constant_reward"),
    ]
    test_rows = []
    for a, b in test_pairs:
        r = paired_ttest(df, a, b, metric="total_reward")
        test_rows.append(r)
        print(
            f"{a:>16} vs {b:<16} "
            f"n={r['n']:>2}, "
            f"mean_diff={r['mean_diff']:+.4f}, "
            f"t={r['t']:+.3f}, p={r['p']:.4f}"
        )
    pd.DataFrame(test_rows).to_csv(
        os.path.join(OUTDIR, "ttest_results.csv"),
        index=False, encoding="utf-8",
    )
    try:
        plot_condition_summary(trajectories, df)
    except Exception as exc:
        print(f"[PLOT] ошибка визуализации: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 72)
    print("ГОТОВО")
    print(f"Результаты: {csv_path}")
    print(f"Траектории: {npz_path}")
    print(f"t-тесты:    {os.path.join(OUTDIR, 'ttest_results.csv')}")
    print("=" * 72)
    if not df.empty:
        print("\nСредние значения по условиям:")
        summary = (
            df.groupby("condition")[
                [
                    "total_reward",
                    "final_C",
                    "final_S",
                    "spectral_entropy",
                    "spectral_flatness",
                    "mode_count",
                ]
            ]
            .mean()
            .round(4)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()