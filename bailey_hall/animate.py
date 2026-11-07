"""
the idea is Simulation.py will save a snapshot (deep copy) of agents[] every timestep. we save the node of each agent, whether it has exited
the entire building, and whether it is still moving towards an auditorium door


to use:
animate.py [run.npz]
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

from grid import BALCONY, COST_STAIRS, FIRST_FLOOR, LEVEL_NAMES, NODE_SEAT, NODE_WALKWAY, Grid


HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PATH = os.path.join(HERE, "out", "run.npz")
FPS = 30

WALL_COLOR, FLOOR_COLOR, SEAT_COLOR, EXIT_COLOR, STAIR_COLOR = "#3a3f47", "#ece8df", "#bdb6a8", "#2ca02c", "#eff5c6"
TO_DOOR_COLOR, TO_EXIT_COLOR = "#1f77b4", "#d1495b"


def save_run(path: str, grid: Grid, frames: List[dict], step_seconds: float) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, cell_m=grid.cell_m, num_nodes=grid.num_nodes, step_seconds=step_seconds,
                        node=np.stack([frame["node"] for frame in frames]).astype(np.int32),
                        exited=np.stack([frame["exited"] for frame in frames]),
                        to_door=np.stack([frame["to_door"] for frame in frames]))
    return path


def load_run(path: str) -> dict:
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def animate(grid: Grid, run: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    node, exited, to_door = run["node"], run["exited"], run["to_door"]
    num_agents = node.shape[1]
    step_seconds = float(run["step_seconds"])
    _, num_rows, num_columns = grid.shape
    extent = (grid.x0, grid.x0 + num_columns * grid.cell, grid.y0 - num_rows * grid.cell, grid.y0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7.4), dpi=110, constrained_layout=True)
    agent_dots = []
    for level_index, axis in enumerate(axes):
        axis.imshow(np.where(grid.walkable[level_index], 1.0, np.nan), extent=extent, interpolation="nearest",
                    cmap=ListedColormap([FLOOR_COLOR]))
        
        stair_cells = np.where(grid.walkable[level_index] & (grid.cell_cost[level_index] == COST_STAIRS), 1.0, np.nan)
        exit_cells = np.where(grid.exit_id[level_index] >= 0, 1.0, np.nan)
        
        axis.imshow(stair_cells, extent=extent, interpolation="nearest", cmap=ListedColormap([STAIR_COLOR]))
        axis.imshow(exit_cells, extent=extent, interpolation="nearest", cmap=ListedColormap([EXIT_COLOR]))
        
        on_level = grid.node_level == level_index
        walkway = on_level & (grid.node_kind == NODE_WALKWAY)
        seat = on_level & (grid.node_kind == NODE_SEAT)
        
        axis.scatter(grid.node_x[walkway], grid.node_y[walkway], c=FLOOR_COLOR, s=9, marker="s", linewidths=0)
        axis.scatter(grid.node_x[seat], grid.node_y[seat], c=SEAT_COLOR, s=7, marker="s", linewidths=0)
        agent_dots.append(axis.scatter([], [], s=18, linewidths=0, zorder=3))
        axis.set_title(LEVEL_NAMES[level_index].replace("_", " "), fontsize=11)
        axis.set_xlabel("x east [m]")
        axis.set_ylabel("y north [m]")
        axis.set_aspect("equal")
        axis.set_facecolor(WALL_COLOR)
        axis.tick_params(length=0, labelsize=8)
        for spine in axis.spines.values():
            spine.set_visible(False)
    legend = [Line2D([], [], marker="o", linestyle="", color=TO_DOOR_COLOR, label="going to auditorium door"),
              Line2D([], [], marker="o", linestyle="", color=TO_EXIT_COLOR, label="going to building exit"),
              Line2D([], [], marker="s", linestyle="", color=EXIT_COLOR, label="exit")]
    figure.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
    clock = figure.suptitle("", fontsize=13)

    def draw_step(step_index):
        nodes_now, inside = node[step_index], ~exited[step_index]
        colors = np.where(to_door[step_index], TO_DOOR_COLOR, TO_EXIT_COLOR)
        node_levels = grid.node_level[nodes_now]
        for level_index, dots in ((FIRST_FLOOR, agent_dots[0]), (BALCONY, agent_dots[1])):
            shown = inside & (node_levels == level_index)
            dots.set_offsets(np.column_stack([grid.node_x[nodes_now[shown]], grid.node_y[nodes_now[shown]]]))
            dots.set_color(colors[shown])
        clock.set_text(f"t = {(step_index + 1) * step_seconds:6.1f} s    step {step_index + 1}    "
                       f"{int(exited[step_index].sum())} of {num_agents} out")
        return agent_dots

    animation = FuncAnimation(figure, draw_step, frames=len(node), blit=False)
    animation.save(path, writer=FFMpegWriter(fps=FPS),
                   progress_callback=lambda done, total: print(f"\r  frame {done + 1}/{total}", end="", flush=True))
    print()
    plt.close(figure)


if __name__ == "__main__":
    
    
    run_path = sys.argv[1] if len(sys.argv) > 1 else RUN_PATH
    run = load_run(run_path)
    output_path = os.path.splitext(run_path)[0] + ".mp4"
    animate(Grid.from_model(cell_m=float(run["cell_m"])), run, output_path)
    print(output_path)
