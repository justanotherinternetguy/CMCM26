import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import LED_Simulation as sim

SEEDS = range(10)
PROFILES = (sim.CONCERT_PROFILE, sim.STUDENT_PROFILE)
EXTRA_SCENARIOS = {}                    # e.g. {"F3+F4": ("F3", "F4")}
BASELINE = "none"
OUTLIER_IQR_K = 1.5                     # outlier = outside [Q1 - k*IQR, Q3 + k*IQR] over the seeds
OUT_DIR = os.path.join("out", "closure_study")
RUNS_CSV = os.path.join(OUT_DIR, "runs.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "summary.csv")
METRICS = {"t90": "90% egress time (s)", "t100": "100% egress time (s)"}
PROFILE_COLORS = {"concert": "#d1495b", "students": "#1f77b4"}
WORKERS = max(1, (os.cpu_count() or 2) - 1)


def build_static(closed_labels) -> SimpleNamespace:
    grid = sim.Grid.from_model(cell_m=sim.CELL_M)
    paint = sim.GridPaint.load(sim.PAINT_PATH, grid)
    barrier_nodes = (paint.node_is_barrier(grid) if sim.BARRIERS_ENABLED
                     else np.zeros(grid.num_nodes, bool))

    fields = sim.compute_distance_fields(grid, barrier_nodes, closed_labels)
    labels = fields.exit_labels
    exit_costs = fields.cost_by_exit
    door_costs = sim.compute_door_costs(grid, fields.blocked_nodes)

    main_rows = np.array([labels.index(l) for l in sim.MAIN_ENTRANCE_LABELS])
    side_rows = np.array([labels.index(l) for l in sim.SIDE_EXIT_LABELS if l in labels], dtype=int)
    exit_node_masks = [sim.mask_to_node_mask(grid, grid.exit_mask(l)) for l in labels]
    door_node_masks = [sim.mask_to_node_mask(grid, grid.door_mask(d)) for d in sim.KNOWN_AUDITORIUM_DOOR_NAMES]

    return SimpleNamespace(
        grid=grid,
        fields=fields,
        exit_costs=exit_costs,
        door_costs=door_costs,
        door_to_exit_costs=sim.compute_door_to_exit_costs(door_costs, exit_costs),
        exit_goal_cost=sim.compute_exit_goal_costs(exit_costs),
        nearby_exit_rows=sim.nearby_open_exit_rows(grid, fields),
        led=sim.build_led_guidance(grid, paint, fields),
        managers=sim.place_crowd_managers(grid, fields, sim.CROWD_MANAGER_POSTS),
        seat_neighbors=sim.immediate_seat_neighbors(grid),
        tables=sim.build_movement_tables(grid),
        all_exit_rows=np.arange(len(labels)),
        uninformed_exit_rows=np.concatenate([main_rows, side_rows]),
        discovery_distance=sim.compute_discovery_distance(grid, fields, barrier_nodes),
        back_exit_distance=sim.compute_back_exit_distance(grid, fields, barrier_nodes),
        door_blocked_by_exit=np.array([[bool((d & e).any()) for e in exit_node_masks]
                                       for d in door_node_masks]),
        step_seconds=grid.cell_m / sim.MAX_WALKING_SPEED_M_PER_S,
    )


def run(profile, seed: int, s: SimpleNamespace):
    grid, fields = s.grid, s.fields
    rng = np.random.default_rng(seed)
    agents, occupant_at_node = sim.spawn_agents(grid, profile, rng, fields, s.door_costs,
                                                s.door_to_exit_costs, s.led)
    occupant_at_node[fields.closed_nodes] = -2

    if sim.RESPONSE_TIME_ENABLED:
        agents.response_time = sim.assign_response_times(
            grid, np.random.default_rng(np.random.SeedSequence([seed, 1])))
    manager_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    managers = s.managers if profile.has_crowd_managers else None

    for step in range(1, sim.MAX_STEPS + 1):
        sim.discover_closed_exits(agents, s.discovery_distance, s.door_blocked_by_exit, s.door_costs,
                                  s.door_to_exit_costs, s.exit_costs, s.all_exit_rows,
                                  s.uninformed_exit_rows, fields.exit_labels)
        sim.discover_back_exits(agents, s.back_exit_distance, s.door_costs, s.door_to_exit_costs,
                                s.exit_costs, fields.exit_labels)
        sim.rush_remaining_agents(agents, s.door_costs, s.door_to_exit_costs, s.exit_costs,
                                  s.all_exit_rows, s.uninformed_exit_rows, fields.exit_labels)
        sim.reevaluate_targets(agents, s.door_costs, s.door_to_exit_costs, s.exit_costs, s.exit_goal_cost,
                               s.all_exit_rows, s.uninformed_exit_rows, occupant_at_node, rng, step,
                               s.nearby_exit_rows)
        sim.apply_crowd_managers(agents, managers, s.door_costs, s.door_to_exit_costs, s.exit_costs,
                                 manager_rng, s.step_seconds)
        sim.move_agents_one_step(grid, agents, occupant_at_node, s.door_costs, s.exit_costs,
                                 s.exit_goal_cost, s.led, s.tables, rng, step, s.seat_neighbors,
                                 s.nearby_exit_rows)
        if agents.has_exited.all():
            break

    t90 = sim.time_to_fraction_out(agents, 0.9, s.step_seconds)
    t100 = sim.time_to_fraction_out(agents, 1.0, s.step_seconds)
    return (t90[1] if t90 else np.nan), (t100[1] if t100 else np.nan), int(agents.has_exited.sum())


def run_scenario(name: str, closed_labels) -> list:
    static = build_static(tuple(closed_labels))
    rows = []
    for profile in PROFILES:
        for seed in SEEDS:
            t90, t100, n_out = run(profile, seed, static)
            rows.append(dict(scenario=name, profile=profile.name, seed=seed, t90=t90, t100=t100, n_out=n_out))
    return rows


def run_all() -> list:
    scenarios = {BASELINE: ()}
    scenarios.update({label: (label,) for label in sim.Grid.from_model(cell_m=sim.CELL_M).exits})
    scenarios.update(EXTRA_SCENARIOS)
    order = {name: i for i, name in enumerate(scenarios)}

    rows = []
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(scenarios))) as pool:
        futures = [pool.submit(run_scenario, name, closed) for name, closed in scenarios.items()]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.extend(result)
            print(f"[{done}/{len(futures)}] closed = {result[0]['scenario']} finished", flush=True)
    rows.sort(key=lambda r: (order[r["scenario"]], r["profile"], r["seed"]))
    return rows


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs() -> list:
    with open(RUNS_CSV, newline="") as f:
        return [dict(scenario=r["scenario"], profile=r["profile"], seed=int(r["seed"]),
                     t90=float(r["t90"]), t100=float(r["t100"]), n_out=int(r["n_out"]))
                for r in csv.DictReader(f)]


def collect(rows: list) -> dict:
    grouped = {}
    for r in rows:
        for metric in METRICS:
            grouped.setdefault((r["scenario"], r["profile"], metric), []).append((r["seed"], r[metric]))
    return {key: np.array([v for _, v in sorted(items)], float) for key, items in grouped.items()}


def inlier_mask(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return finite
    q1, q3 = np.percentile(values[finite], [25, 75])
    spread = OUTLIER_IQR_K * (q3 - q1)
    return finite & (values >= q1 - spread) & (values <= q3 + spread)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if len(x) else np.nan


def _std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan


def trimmed_mean(data: dict, scenario: str, profile: str, metric: str) -> float:
    v = data[(scenario, profile, metric)]
    return _mean(v[inlier_mask(v)])


def summarize(data: dict, scenarios: list) -> list:
    summary = []
    for scenario in scenarios:
        for profile in PROFILES:
            for metric in METRICS:
                v = data[(scenario, profile.name, metric)]
                finite, keep = np.isfinite(v), inlier_mask(v)
                summary.append(dict(
                    scenario=scenario, profile=profile.name, metric=metric,
                    n=int(finite.sum()), n_unfinished=int((~finite).sum()),
                    mean_all=_mean(v[finite]), std_all=_std(v[finite]), median=float(np.median(v[finite])) if finite.any() else np.nan,
                    n_outliers=int(finite.sum() - keep.sum()),
                    mean_trimmed=_mean(v[keep]), std_trimmed=_std(v[keep])))
    return summary


def print_summary(summary: list, scenarios: list) -> None:
    index = {(r["scenario"], r["profile"], r["metric"]): r for r in summary}
    for profile in PROFILES:
        print(f"\n{profile.name}   (times in s; 'trim' = outliers removed)")
        print(f"{'closed':8s} | {'90% all':>8s} {'90% trim':>9s} {'out':>4s} | "
              f"{'100% all':>9s} {'100% trim':>10s} {'out':>4s} | unfinished")
        for scenario in scenarios:
            a, b = index[(scenario, profile.name, "t90")], index[(scenario, profile.name, "t100")]
            print(f"{scenario:8s} | {a['mean_all']:8.1f} {a['mean_trimmed']:9.1f} {a['n_outliers']:4d} | "
                  f"{b['mean_all']:9.1f} {b['mean_trimmed']:10.1f} {b['n_outliers']:4d} | {b['n_unfinished']}")


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=200)
    plt.close(fig)


def plot_means(data: dict, scenarios: list, metric: str) -> None:
    fig, axes = plt.subplots(1, len(PROFILES), figsize=(7 * len(PROFILES), 5), sharey=True)
    x = np.arange(len(scenarios))
    jitter_rng = np.random.default_rng(0)
    for ax, profile in zip(axes, PROFILES):
        means, stds, all_means = [], [], []
        for i, scenario in enumerate(scenarios):
            v = data[(scenario, profile.name, metric)]
            finite, keep = np.isfinite(v), inlier_mask(v)
            outliers = finite & ~keep
            jitter = jitter_rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(i + jitter[keep], v[keep], s=14, color="0.35", zorder=3)
            ax.scatter(i + jitter[outliers], v[outliers], s=40, marker="x", color="black", zorder=4)
            means.append(_mean(v[keep]))
            stds.append(_std(v[keep]))
            all_means.append(_mean(v[finite]))
        ax.bar(x, means, yerr=np.nan_to_num(stds), color=PROFILE_COLORS[profile.name], alpha=0.55,
               capsize=3, zorder=2)
        ax.scatter(x, all_means, marker="D", s=36, facecolor="white", edgecolor="black", zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")
        ax.set_xlabel("Exit closed")
        ax.set_title(profile.name.capitalize())
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_ylabel(METRICS[metric])
    axes[0].legend(handles=[
        Patch(facecolor="0.6", label="mean, outliers removed (±1 SD)"),
        Line2D([], [], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="black",
               label="mean, all runs"),
        Line2D([], [], marker="o", linestyle="", color="0.35", markersize=4, label="individual seed"),
        Line2D([], [], marker="x", linestyle="", color="black", label="removed outlier")],
        fontsize=8, loc="upper left")
    save_figure(fig, f"means_{metric}")


def plot_boxes(data: dict, scenarios: list) -> None:
    fig, axes = plt.subplots(len(METRICS), len(PROFILES), figsize=(7 * len(PROFILES), 4.5 * len(METRICS)),
                             sharex=True)
    for row, (metric, title) in enumerate(METRICS.items()):
        for col, profile in enumerate(PROFILES):
            ax = axes[row, col]
            groups = [data[(s, profile.name, metric)] for s in scenarios]
            groups = [g[np.isfinite(g)] for g in groups]
            ax.boxplot(groups, whis=OUTLIER_IQR_K, patch_artist=True,
                       boxprops=dict(facecolor=PROFILE_COLORS[profile.name], alpha=0.55),
                       medianprops=dict(color="black"))
            ax.set_xticks(range(1, len(scenarios) + 1))
            ax.set_xticklabels(scenarios, rotation=45, ha="right")
            ax.set_ylabel(title)
            ax.set_title(profile.name.capitalize())
            ax.grid(axis="y", alpha=0.3)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Exit closed")
    save_figure(fig, "boxplots")


def plot_impact(data: dict, scenarios: list) -> None:
    closed = [s for s in scenarios if s != BASELINE]
    if not closed or BASELINE not in scenarios:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(7 * len(METRICS), 0.6 * len(closed) + 2))
    y, height = np.arange(len(closed)), 0.38
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for k, profile in enumerate(PROFILES):
            base = trimmed_mean(data, BASELINE, profile.name, metric)
            change = [100 * (trimmed_mean(data, s, profile.name, metric) / base - 1) for s in closed]
            ax.barh(y + (k - 0.5) * height, change, height, color=PROFILE_COLORS[profile.name],
                    label=profile.name.capitalize())
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(closed)
        ax.invert_yaxis()
        ax.set_xlabel(f"Change in mean {title} vs. no closure (%)")
        ax.set_ylabel("Exit closed")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8)
    save_figure(fig, "closure_impact")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    if "--reuse" in sys.argv:
        rows = load_runs()
    else:
        rows = run_all()
        write_csv(RUNS_CSV, rows)

    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    data = collect(rows)
    summary = summarize(data, scenarios)
    write_csv(SUMMARY_CSV, summary)
    print_summary(summary, scenarios)

    for metric in METRICS:
        plot_means(data, scenarios, metric)
    plot_boxes(data, scenarios)
    plot_impact(data, scenarios)
    print(f"\nSaved runs, summary and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()