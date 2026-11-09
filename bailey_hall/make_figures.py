"""Figures for the write-up: scenario comparison and barrier layout."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "experiments")
INK, MUTED, SURFACE, GRID, BLUE, ORANGE = "#2b2b29", "#8a897f", "#ffffff", "#e4e3dd", "#2a78d6", "#eb6834"

ORDER = [  # (key, label)
    ("normal_egress", "Normal egress after a concert (Q1)"),
    ("students", "Students, lecture (Q3)"),
    ("concert_none", "Concert, no managers"),
    ("concert_baseline", "Concert, 6 baseline managers (Q2)"),
    ("concert_opt", "Concert, 6 optimised managers"),
    ("concert_opt_led", "  + LED strips"),
    ("concert_opt_barrier", "  + portico barriers"),
    ("concert_opt_led_barrier", "  + LED strips + barriers"),
]


def style(axis):
    axis.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=INK)


def scenarios():
    summary = json.load(open(os.path.join(OUT, "summary.json")))
    rows = [(label, summary[key]) for key, label in ORDER if key in summary]
    fig, axis = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2), facecolor=SURFACE)
    style(axis)
    y = np.arange(len(rows))[::-1]
    means = [r["mean"] for _, r in rows]
    axis.barh(y, means, height=0.6, color=BLUE, zorder=2)
    for yi, (_, r) in zip(y, rows):
        axis.plot([r["min"], r["max"]], [yi, yi], color=INK, lw=1, zorder=3)
        axis.text(r["max"] + 3, yi, f"{r['mean']:.0f} s", va="center", fontsize=9, color=INK)
    axis.axvline(180, color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=4)
    axis.text(180, len(rows) - 0.4, " 180 s target", color=ORANGE, fontsize=9, va="bottom")
    axis.set_yticks(y, [label for label, _ in rows])
    axis.set_xlabel("time for the last person to leave (s): bar = mean of 24 runs, line = min-max", color=INK)
    axis.set_xlim(0, max(r["max"] for _, r in rows) + 35)
    axis.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scenarios.png"), dpi=200)


def barriers():
    import LED_Simulation as sim
    from barriers import barrier_nodes
    from grid import FIRST_FLOOR, NODE_CELL, NODE_SEAT, NODE_WALKWAY
    scenario = sim.build_scenario(closed_exit_labels=(), barriers_enabled=False)
    grid = scenario.grid
    blocked = barrier_nodes(grid, "portico")
    fig, axis = plt.subplots(figsize=(7, 6.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    floor = grid.node_level == FIRST_FLOOR
    cells = floor & (grid.node_kind == NODE_CELL) & ~blocked
    seats = floor & np.isin(grid.node_kind, (NODE_SEAT, NODE_WALKWAY))
    axis.scatter(grid.node_x[cells], grid.node_y[cells], s=6, marker="s", c="#d9d8d2", lw=0)
    axis.scatter(grid.node_x[seats], grid.node_y[seats], s=2, marker="s", c="#b9b8b0", lw=0)
    axis.scatter(grid.node_x[blocked], grid.node_y[blocked], s=14, marker="s", c=ORANGE, lw=0, label="barrier")
    for name in sim.KNOWN_AUDITORIUM_DOOR_NAMES[:5]:
        nodes = sim.mask_to_node_mask(grid, grid.door_mask(name))
        axis.scatter(grid.node_x[nodes], grid.node_y[nodes], s=8, marker="s", c=BLUE, lw=0)
    for label in scenario.fields.exit_labels:
        nodes = sim.mask_to_node_mask(grid, grid.exit_mask(label)) & floor
        if nodes.any():
            axis.annotate(label, (grid.node_x[nodes].mean(), grid.node_y[nodes].mean()), fontsize=9,
                          color=INK, weight="bold", ha="center", va="center",
                          xytext=(0, -9 if label.startswith("F") else 9), textcoords="offset points")
    axis.scatter([], [], s=14, marker="s", c=BLUE, label="auditorium door")
    axis.legend(loc="upper center", frameon=False, ncol=2)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_barriers.png"), dpi=200)


if __name__ == "__main__":
    which = sys.argv[1:] or ["scenarios", "barriers"]
    for name in which:
        globals()[name]()
