"""
Simulation.py saves a snapshot of the agents every timestep: the node of each agent, whether it has
exited the building, and whether it is still heading for an auditorium door.

to use:
animate.py [run.npz]
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

from grid import BALCONY, FIRST_FLOOR, LEVEL_NAMES, STAIR_INTERIOR, Grid


HERE = os.path.dirname(os.path.abspath(__file__))
RUN_PATH = os.path.join(HERE, "out", "run.npz")
FPS = 30
FRAME_STRIDE = 3    # render every Nth simulation step
DPI = 80

TO_DOOR_COLOR, TO_EXIT_COLOR = "#1f77b4", "#d1495b"
MANAGER_COLOR, TOLD_BY_MANAGER_COLOR = "#111111", "#8e44ad"
CLOSED_EXIT_COLOR = "#ff7f0e"


def save_run(path: str, grid: Grid, frames: List[dict], step_seconds: float,
             manager_nodes: Optional[np.ndarray] = None,
             closed_exit_cells: Optional[np.ndarray] = None) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    extra = {}
    if manager_nodes is not None:
        extra["manager_node"] = np.asarray(manager_nodes, np.int32)
    if closed_exit_cells is not None and np.any(closed_exit_cells):
        extra["closed_exit_cells"] = np.asarray(closed_exit_cells, bool)
    if frames and all("manager" in frame for frame in frames):
        extra["told_by_manager"] = np.stack([frame["manager"] for frame in frames])
    np.savez_compressed(path, cell_m=grid.cell_m, num_nodes=grid.num_nodes, step_seconds=step_seconds,
                        node=np.stack([frame["node"] for frame in frames]).astype(np.int32),
                        exited=np.stack([frame["exited"] for frame in frames]),
                        to_door=np.stack([frame["to_door"] for frame in frames]), **extra)
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
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

    agent_style = dict(s=18, edgecolors="white", linewidths=0.4, zorder=8)

    node, exited, to_door = run["node"], run["exited"], run["to_door"]
    told_by_manager = run.get("told_by_manager")
    manager_nodes = run.get("manager_node", np.empty(0, np.int32))
    closed_cells = run.get("closed_exit_cells")
    num_agents = node.shape[1]
    step_seconds = float(run["step_seconds"])
    _, num_rows, num_columns = grid.shape
    extent = (grid.x0, grid.x0 + num_columns * grid.cell, grid.y0 - num_rows * grid.cell, grid.y0)

    figure, axes = plt.subplots(1, 2, figsize=(14, 7.4), dpi=DPI, constrained_layout=True)
    agent_dots = []
    for level_index, axis in enumerate(axes):
        grid.draw_layout(axis, level_index, highlight_doors=True)
        if closed_cells is not None:
            axis.imshow(np.where(closed_cells[level_index], 1.0, np.nan), extent=extent,
                        interpolation="nearest", cmap=ListedColormap([CLOSED_EXIT_COLOR]), zorder=4)
        agent_dots.append(axis.scatter([], [], **agent_style))
        here = manager_nodes[grid.node_level[manager_nodes] == level_index]
        axis.scatter(grid.node_x[here], grid.node_y[here], c=MANAGER_COLOR, s=90, marker="*",
                     edgecolors="white", linewidths=0.6, zorder=9)
        axis.set_title(LEVEL_NAMES[level_index].replace("_", " "), fontsize=11)
    legend = grid.layout_legend(highlight_doors=True)
    legend += [Line2D([], [], marker="o", linestyle="", color=TO_DOOR_COLOR, label="going to auditorium door"),
               Line2D([], [], marker="o", linestyle="", color=TO_EXIT_COLOR, label="going to building exit")]
    if closed_cells is not None:
        legend.append(Line2D([], [], marker="s", linestyle="", color=CLOSED_EXIT_COLOR, label="closed exit"))
    if len(manager_nodes):
        legend += [Line2D([], [], marker="o", linestyle="", color=TOLD_BY_MANAGER_COLOR,
                          label="redirected by a crowd manager"),
                   Line2D([], [], marker="*", linestyle="", color=MANAGER_COLOR, markersize=10,
                          label="crowd manager")]
    figure.legend(handles=legend, loc="outside lower center", ncol=min(len(legend), 5), frameon=False, fontsize=9)
    clock = figure.suptitle("", fontsize=13)

    def draw_step(step_index):
        nodes_now, inside = node[step_index], ~exited[step_index]
        colors = np.where(to_door[step_index], TO_DOOR_COLOR, TO_EXIT_COLOR)
        if told_by_manager is not None:
            colors = np.where(told_by_manager[step_index], TOLD_BY_MANAGER_COLOR, colors)
        # Agents on a balcony stair stand on stair-chain nodes with their own level; draw them on the
        # first-floor panel at their place along the flight, or they would vanish mid-stair.
        node_levels = np.where(grid.node_level[nodes_now] == STAIR_INTERIOR, FIRST_FLOOR, grid.node_level[nodes_now])
        for level_index, dots in ((FIRST_FLOOR, agent_dots[0]), (BALCONY, agent_dots[1])):
            shown = inside & (node_levels == level_index)
            dots.set_offsets(np.column_stack([grid.node_x[nodes_now[shown]], grid.node_y[nodes_now[shown]]]))
            dots.set_color(colors[shown])
        clock.set_text(f"t = {(step_index + 1) * step_seconds:6.1f} s    step {step_index + 1}    "
                       f"{int(exited[step_index].sum())} of {num_agents} out")
        return agent_dots

    frame_ids = sorted(set(range(0, len(node), FRAME_STRIDE)) | {len(node) - 1})
    animation = FuncAnimation(figure, draw_step, frames=frame_ids, blit=False)
    animation.save(
        path,
        writer=FFMpegWriter(fps=FPS, codec="libx264",
                            extra_args=["-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]),
        progress_callback=lambda done, total: print(f"\r  frame {done + 1}/{total}", end="", flush=True),
    )
    print()
    plt.close(figure)


if __name__ == "__main__":
    run_path = sys.argv[1] if len(sys.argv) > 1 else RUN_PATH
    run = load_run(run_path)
    output_path = os.path.splitext(run_path)[0] + ".mp4"
    animate(Grid.from_model(cell_m=float(run["cell_m"])), run, output_path)
    print(output_path)