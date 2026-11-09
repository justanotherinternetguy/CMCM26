"""Temporary corridor barriers (Question 4a).

A barrier is a short line of floor cells across the semicircular corridor. `corridor_barrier(angle)` blocks
the corridor along the ray from the plan centre at `angle` degrees (counter-clockwise from east), for the
corridor radii only, so the auditorium doors and the seating are untouched.
"""
from __future__ import annotations

import numpy as np

from grid import FIRST_FLOOR, NODE_CELL, Grid

CORRIDOR_R_MIN_M = 15.2      # just outside the loge back wall (48.5 ft = 14.8 m)
CORRIDOR_R_MAX_M = 19.5      # past the outer corridor wall (60.5 ft = 18.4 m)
HALF_THICKNESS_M = 0.5       # two cells thick, so there is no diagonal gap through it


def corridor_barrier(grid: Grid, angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    ux, uy = np.cos(theta), np.sin(theta)
    x, y = grid.node_x, grid.node_y
    along = x * ux + y * uy
    across = np.abs(-x * uy + y * ux)
    return ((grid.node_kind == NODE_CELL) & (grid.node_level == FIRST_FLOOR)
            & (along >= CORRIDOR_R_MIN_M) & (along <= CORRIDOR_R_MAX_M) & (across <= HALF_THICKNESS_M))


# Barriers between each portico door (PW at about 210 deg, PE at about 320 deg) and the vestibule in front of
# the foyer. People leaving by the side doors of the loge are kept on the portico side, and the foyer is
# left to the aisle and centre doors and the balcony stairs.
BARRIER_SETS = {
    "none": (),
    "portico": (226.0, 314.0),
}


def barrier_nodes(grid: Grid, name: str) -> np.ndarray:
    mask = np.zeros(grid.num_nodes, bool)
    for angle in BARRIER_SETS[name]:
        mask |= corridor_barrier(grid, angle)
    return mask
