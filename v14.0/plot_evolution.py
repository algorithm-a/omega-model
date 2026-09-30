"""
Визуализация эволюции Omega-Model v14.0.

Читает world_history.json + eva_results.csv.
Строит:
1. C_mean по поколениям (кланы + финал)
2. C_best по поколениям
3. Heatmap кланов (C по поколениям)
4. Manager changes (кто когда был Manager)

Запуск:
    python plot_evolution.py --input v14.0_clans --output plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # без GUI


def load_history(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_c_mean(history: dict, output: Path) -> None:
    """График C_mean по поколениям."""
    gens = history.get("generations", [])
    if not gens:
        return
    g = [x["generation"] for x in gens]
    c_mean = [x["c_mean_overall"] for x in gens]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g, c_mean, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(y=max(c_mean), color="red", linestyle="--",
               alpha=0.5, label=f"max = {max(c_mean):.4f}")
    ax.set_xlabel("Поколение", fontsize=12)
    ax.set_ylabel("C_mean", fontsize=12)
    ax.set_title("Эволюция C_mean (Omega-Model v14.0)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = output / "c_mean.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_c_best(history: dict, output: Path) -> None:
    """График C_best по поколениям."""
    gens = history.get("generations", [])
    if not gens:
        return
    g = [x["generation"] for x in gens]
    c_best = [x.get("c_best", x["c_mean_overall"]) for x in gens]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g, c_best, "s-", color="darkorange", linewidth=2, markersize=6)
    ax.axhline(y=max(c_best), color="red", linestyle="--",
               alpha=0.5, label=f"max = {max(c_best):.4f}")
    ax.set_xlabel("Поколение", fontsize=12)
    ax.set_ylabel("C_best", fontsize=12)
    ax.set_title("Лучший агент по поколениям (C_best)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = output / "c_best.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_both(history: dict, output: Path) -> None:
    """C_mean и C_best на одном графике."""
    gens = history.get("generations", [])
    if not gens:
        return
    g = [x["generation"] for x in gens]
    c_mean = [x["c_mean_overall"] for x in gens]
    c_best = [x.get("c_best", x["c_mean_overall"]) for x in gens]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g, c_mean, "o-", color="steelblue",
            linewidth=2, markersize=6, label="C_mean")
    ax.plot(g, c_best, "s-", color="darkorange",
            linewidth=2, markersize=6, label="C_best")
    ax.set_xlabel("Поколение", fontsize=12)
    ax.set_ylabel("C", fontsize=12)
    ax.set_title("Эволюция C (Omega-Model v14.0)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = output / "c_mean_and_best.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_agents_heatmap(history: dict, output: Path) -> None:
    """Heatmap: агенты × поколения (их C)."""
    gens = history.get("generations", [])
    if not gens:
        return

    # собираем всех агентов
    all_ids = set()
    for gen in gens:
        for a in gen.get("agents", []):
            all_ids.add(a["id"])
    all_ids = sorted(all_ids)
    if not all_ids:
        return

    n_gens = len(gens)
    n_agents = len(all_ids)
    matrix = np.full((n_agents, n_gens), np.nan)

    for j, gen in enumerate(gens):
        for a in gen.get("agents", []):
            i = all_ids.index(a["id"])
            matrix[i, j] = a.get("c_mean", a.get("c_last", 0.0))

    fig, ax = plt.subplots(figsize=(12, max(4, n_agents * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_gens))
    ax.set_xticklabels([g["generation"] for g in gens])
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(all_ids)
    ax.set_xlabel("Поколение", fontsize=12)
    ax.set_ylabel("Агент", fontsize=12)
    ax.set_title("C агентов по поколениям", fontsize=14)
    plt.colorbar(im, ax=ax, label="C")
    plt.tight_layout()
    out = output / "agents_heatmap.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {out}")


def plot_manager_timeline(history: dict, output: Path) -> None:
    """Кто был Manager в каждом поколении."""
    gens = history.get("generations", [])
    if not gens:
        return

    managers = [g.get("manager", "?") for g in gens]
    unique = []
    for m in managers:
        if m not in unique:
            unique.append(m)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique))))

    fig, ax = plt.subplots(figsize=(10, 3))
    for i, g in enumerate(gens):
        m = g.get("manager", "?")
        idx = unique.index(m)
        ax.bar(i, 1, color=colors[idx], edgecolor="white")
        ax.text(i, 0.5, m, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([g["generation"] for g in gens])
    ax.set_yticks([])
    ax.set_xlabel("Поколение", fontsize=12)
    ax.set_title("Manager по поколениям", fontsize=14)
    plt.tight_layout()
    out = output / "manager_timeline.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[SAVE] {out}")


def main():
    p = argparse.ArgumentParser(description="Визуализация Omega-Model")
    p.add_argument("--input", default="v14.0_clans",
                   help="папка с world_history.json")
    p.add_argument("--output", default="plots",
                   help="папка для графиков")
    args = p.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    hist_path = input_dir / "world_history.json"
    if not hist_path.exists():
        print(f"[ERROR] не найден: {hist_path}")
        return

    print(f"[LOAD] {hist_path}")
    history = load_history(hist_path)

    print(f"[PLOT] C_mean")
    plot_c_mean(history, output_dir)

    print(f"[PLOT] C_best")
    plot_c_best(history, output_dir)

    print(f"[PLOT] C_mean + C_best")
    plot_both(history, output_dir)

    print(f"[PLOT] Heatmap агентов")
    plot_agents_heatmap(history, output_dir)

    print(f"[PLOT] Manager timeline")
    plot_manager_timeline(history, output_dir)

    print(f"\n[DONE] все графики в: {output_dir}")


if __name__ == "__main__":
    main()