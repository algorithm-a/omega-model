"""fix_tail.py - дописать хвост omega_unified.py"""

PATH = "omega_unified.py"

# Читаем файл
with open(PATH, encoding="utf-8") as f:
    s = f.read()

# Обрезаем по маркеру обрыва
marker = '        figsize=(10, 2.2'
idx = s.find(marker)
if idx == -1:
    raise SystemExit("marker not found")
s = s[:idx]

# Дописываем правильный хвост
tail = '''        figsize=(10, 2.2 * len(conditions)),
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
    print("ОМЕГА-ЭКСПЕРИМЕНТ (v3)")
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
        print(f"\\n=== {condition} (reward={mode}, policy={policy}) ===")
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
    print("\\n" + "=" * 72)
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
    print("\\n" + "=" * 72)
    print("ГОТОВО")
    print(f"Результаты: {csv_path}")
    print(f"Траектории: {npz_path}")
    print(f"t-тесты:    {os.path.join(OUTDIR, 'ttest_results.csv')}")
    print("=" * 72)
    if not df.empty:
        print("\\nСредние значения по условиям:")
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
'''

# Записываем
with open(PATH, "w", encoding="utf-8") as f:
    f.write(s + tail)

print("tail appended, file length:", len(s + tail))