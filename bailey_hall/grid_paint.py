"""
Paint LEDs and barriers onto the Bailey Hall hybrid grid, and keep several named layouts.

    python bailey_hall/grid_paint.py                       # open the layout "default"
    python bailey_hall/grid_paint.py --layout night_show   # open (or start) the layout "night_show"
    python bailey_hall/grid_paint.py --cell-m 0.25
    python bailey_hall/grid_paint.py --grid out/grid_hybrid_0.5m.npz --layouts-dir out/my_layouts

The painter is a small local web page (grid_paint.html, same folder).  This file builds the grid, serves
the page, and saves what you paint.  Put both files next to grid.py and model.py.

Layouts
-------
Every layout is one file pair in the layouts folder (default out/layouts_<cell>m/):

    default.npz    default.json
    night_show.npz night_show.json
    ...

The server always has one CURRENT layout.  The page's normal Save button writes to it.  A small
"Layout" bar is added to the bottom right of the page (see LAYOUT_BAR_SNIPPET) to
    Open        switch to another saved layout (the page reloads and shows it)
    Save as...  choose a new name; the next press of the page's Save button writes to that name
    Rename      rename the selected layout
    Delete      delete the selected layout
If you build the same controls into grid_paint.html yourself, give the element id="layoutBar" and the
server will not add its own.

An older single paint file (out/grid_paint_<cell>m.npz) is copied into the layouts folder as "default"
the first time this version starts, if there is no "default" layout yet.  The old file is left alone.

What can be painted
-------------------
Every item has two INDEPENDENT yes/no flags, so something can be an LED and a barrier at the same time:

    has_led       the item carries LEDs
    is_barrier    the item is blocked (agents cannot use it)

Items that carry the flags:

    floor cells        the 0.5 m lattice tiles                       (levels, rows, columns)
    seats              one node per seat                             (seats,)
    row walkways       the walkway node in front of each seat        (seats,)

Walls are different: a wall already blocks, so a wall only has `has_led`.  A wall is a cell EDGE, either
the outline of the walkable area or a thin wall between two walkable cells (the lattice has no step across
it).  Edges are stored in two arrays, like the lines of a chessboard:

    wall_has_led_horizontal   (levels, rows + 1, columns)    edge on the NORTH side of cell (row, column)
    wall_has_led_vertical     (levels, rows, columns + 1)    edge on the WEST side of cell (row, column)

Using a painted file in a simulation
------------------------------------
    from grid import Grid
    from grid_paint import GridPaint
    grid = Grid.load("out/grid_hybrid_0.5m.npz")
    paint = GridPaint.load("out/layouts_0.5m/night_show.npz", grid)
    paint.cell_has_led                    # bool (2, H, W)
    paint.wall_led_segments(grid)         # list of dicts with both end points in metres
    blocked = paint.node_is_barrier(grid) # bool (nodes,), accepted by grid.node_distance_field(blocked=...)
    field = grid.node_distance_field(blocked=blocked)

Saving writes <layout>.npz (arrays) and <layout>.json (a human readable list of every LED and barrier).

When the grid changes
---------------------
Paint is stored per lattice cell, so a file only fits the lattice it was painted on.  GridPaint.load
therefore compares the saved lattice with the current grid:

    same rows and columns, same corner     used as it is.  A different NODE COUNT (stair nodes added, a few
                                           cells now walkable or not) does not matter: node arrays are rebuilt
                                           from the cell, seat and walkway flags every time.
    rows/columns/corner changed            every flag is moved to the cell at the same position in metres
                                           (a one-line note says how far).  Flags that fall off the new lattice
                                           are lost; flags on cells that stopped being floor are ignored.
    different cell size                    cannot be moved; paint it again.

Files saved by this version store the lattice corner (origin_x_m, origin_y_m).  For older files the west edge
is taken as -(columns / 2) * cell, the way grid.py places it, and the north edge is found by sliding the paint
north/south until it lies on the new floor and walls; if that is not clear-cut it stops with an error.  Give the
old corner yourself with GridPaint.load(path, grid, old_origin_m=(x0, y0)) or  --old-origin X Y.
Per-seat flags are kept only if the number of seats is unchanged.  Open the moved layout in the painter and
press Save once to store it in the new grid's format.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np

from grid import COST_FLOOR, EDGE_LATTICE, LEVEL_NAMES, Grid

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "grid_paint.html")

PAINT_FORMAT_VERSION = 1

DEFAULT_LAYOUT_NAME = "default"

LAYER_NAMES = ("cell_has_led", "cell_is_barrier",
               "wall_has_led_horizontal", "wall_has_led_vertical",
               "seat_has_led", "seat_is_barrier",
               "walkway_has_led", "walkway_is_barrier")


# ------------------------------------------------------------------- helpers --
def camel_case(snake_name: str) -> str:
    """cell_has_led -> cellHasLed (the names the web page uses)."""
    first_word, *other_words = snake_name.split("_")
    return first_word + "".join(word.title() for word in other_words)


def encode_flags(flags: np.ndarray) -> str:
    """Bool array -> base64 text, one byte per flag."""
    return base64.b64encode(np.asarray(flags).astype(np.uint8).tobytes()).decode("ascii")


def decode_flags(text: str, shape: Tuple[int, ...]) -> np.ndarray:
    raw_bytes = np.frombuffer(base64.b64decode(text), np.uint8)
    expected_size = int(np.prod(shape))
    if raw_bytes.size != expected_size:
        raise ValueError(f"expected {expected_size} flags for shape {shape}, got {raw_bytes.size}")
    return (raw_bytes != 0).reshape(shape)


def layer_shapes(grid: Grid) -> Dict[str, Tuple[int, ...]]:
    num_levels, num_rows, num_columns = grid.shape
    num_seats = len(grid.spawn)
    return {
        "cell_has_led": (num_levels, num_rows, num_columns),
        "cell_is_barrier": (num_levels, num_rows, num_columns),
        "wall_has_led_horizontal": (num_levels, num_rows + 1, num_columns),
        "wall_has_led_vertical": (num_levels, num_rows, num_columns + 1),
        "seat_has_led": (num_seats,),
        "seat_is_barrier": (num_seats,),
        "walkway_has_led": (num_seats,),
        "walkway_is_barrier": (num_seats,),
    }


# --------------------------------------------------------------------- walls --
def find_wall_edges(grid: Grid) -> Tuple[np.ndarray, np.ndarray]:
    """Every cell edge that has a wall on it, as (horizontal_wall, vertical_wall).

    An edge is a wall when
      * exactly one of the two cells beside it is walkable (the outline of the floor), or
      * both are walkable but the lattice has no step between them (a wall thinner than a cell,
        like the orchestra barrier).
    Edge (row, column) of horizontal_wall is the NORTH side of cell (row, column); the extra last
    row is the south side of the bottom cells.  Vertical edges work the same way on the WEST side."""
    num_levels, num_rows, num_columns = grid.shape
    horizontal_wall = np.zeros((num_levels, num_rows + 1, num_columns), bool)
    vertical_wall = np.zeros((num_levels, num_rows, num_columns + 1), bool)

    is_lattice_step = grid.edge_kind == EDGE_LATTICE
    step_keys = (grid.edge_from[is_lattice_step].astype(np.int64) * grid.num_nodes
                 + grid.edge_to[is_lattice_step].astype(np.int64))

    def has_lattice_step(node_a: np.ndarray, node_b: np.ndarray) -> np.ndarray:
        return np.isin(node_a.astype(np.int64) * grid.num_nodes + node_b.astype(np.int64), step_keys)

    for level_index in range(num_levels):
        walkable = grid.walkable[level_index]
        cell_nodes = grid.cell_node[level_index]

        # vertical edges: the cell to the west and the cell to the east of each edge
        walkable_west = np.zeros((num_rows, num_columns + 1), bool)
        walkable_east = np.zeros((num_rows, num_columns + 1), bool)
        walkable_west[:, 1:] = walkable
        walkable_east[:, :num_columns] = walkable
        vertical_wall[level_index] = walkable_west ^ walkable_east
        nodes_west, nodes_east = cell_nodes[:, :-1], cell_nodes[:, 1:]
        thin_wall = (nodes_west >= 0) & (nodes_east >= 0) & ~has_lattice_step(nodes_west, nodes_east)
        vertical_wall[level_index, :, 1:num_columns] |= thin_wall

        # horizontal edges: the cell to the north and the cell to the south of each edge
        walkable_north = np.zeros((num_rows + 1, num_columns), bool)
        walkable_south = np.zeros((num_rows + 1, num_columns), bool)
        walkable_north[1:, :] = walkable
        walkable_south[:num_rows, :] = walkable
        horizontal_wall[level_index] = walkable_north ^ walkable_south
        nodes_north, nodes_south = cell_nodes[:-1, :], cell_nodes[1:, :]
        thin_wall = (nodes_north >= 0) & (nodes_south >= 0) & ~has_lattice_step(nodes_north, nodes_south)
        horizontal_wall[level_index, 1:num_rows, :] |= thin_wall

    return horizontal_wall, vertical_wall


# ------------------------------------------------------ moving paint to a changed grid --
SEARCH_NORTH_SOUTH_M = 40.0      # how far old paint is slid north/south when the old north edge is unknown
MIN_OVERLAP_FRACTION = 0.9       # share of painted cells and wall edges that must land on the new floor and walls


def is_seat_layer(layer_name: str) -> bool:
    return layer_name.startswith(("seat_", "walkway_"))


def shift_flags(flags: np.ndarray, new_shape: Tuple[int, ...], row_shift: int, column_shift: int) -> np.ndarray:
    """Copy a (levels, rows, columns) flag array into a new array of `new_shape`, moving every flag by
    (row_shift, column_shift) cells.  Positive shifts go south and east.  Flags that land outside are lost."""
    moved = np.zeros(new_shape, bool)
    old_rows, old_columns = flags.shape[1:]
    new_rows, new_columns = new_shape[1:]
    first_row, last_row = max(0, row_shift), min(new_rows, old_rows + row_shift)
    first_column, last_column = max(0, column_shift), min(new_columns, old_columns + column_shift)
    if last_row > first_row and last_column > first_column:
        moved[:, first_row:last_row, first_column:last_column] = flags[
            :, first_row - row_shift:last_row - row_shift, first_column - column_shift:last_column - column_shift]
    return moved


def origin_shift(old_origin_m: Tuple[float, float], grid: Grid) -> Tuple[int, int]:
    """(row_shift, column_shift) that carries a cell of the old lattice to the cell at the same place in metres."""
    column_shift = (old_origin_m[0] - grid.x0) / grid.cell
    row_shift = (grid.y0 - old_origin_m[1]) / grid.cell
    if not (math.isclose(column_shift, round(column_shift), abs_tol=1e-6)
            and math.isclose(row_shift, round(row_shift), abs_tol=1e-6)):
        raise ValueError(f"the old lattice corner {old_origin_m} is not a whole number of {grid.cell:g} m cells "
                         f"away from the new one ({grid.x0:g}, {grid.y0:g}), so the paint cannot be moved cell for cell")
    return int(round(row_shift)), int(round(column_shift))


def find_row_shift(saved: Dict[str, np.ndarray], grid: Grid, column_shift: int,
                   wall_edges: Tuple[np.ndarray, np.ndarray], path: str) -> int:
    """Old file without a stored corner: slide the paint north/south until it sits on the new floor and walls."""
    shapes = layer_shapes(grid)
    horizontal_wall, vertical_wall = wall_edges
    painted_cells = saved["cell_has_led"] | saved["cell_is_barrier"]
    painted_horizontal = saved["wall_has_led_horizontal"]
    painted_vertical = saved["wall_has_led_vertical"]
    total = int(painted_cells.sum() + painted_horizontal.sum() + painted_vertical.sum())
    if total == 0:
        return 0                                     # no cells or walls to line up; only seat flags may exist
    reach = int(math.ceil(SEARCH_NORTH_SOUTH_M / grid.cell))
    candidates = list(range(-reach, reach + 1))
    hits = []
    for row_shift in candidates:
        moved_cells = shift_flags(painted_cells, shapes["cell_has_led"], row_shift, column_shift)
        moved_horizontal = shift_flags(painted_horizontal, shapes["wall_has_led_horizontal"], row_shift, column_shift)
        moved_vertical = shift_flags(painted_vertical, shapes["wall_has_led_vertical"], row_shift, column_shift)
        hits.append(int((moved_cells & grid.walkable).sum() + (moved_horizontal & horizontal_wall).sum()
                        + (moved_vertical & vertical_wall).sum()))
    hits = np.array(hits)
    best = int(hits.argmax())
    runner_up = int(np.delete(hits, best).max()) if len(hits) > 1 else 0
    guessed_x0 = -(saved["cell_has_led"].shape[2] // 2) * grid.cell
    how_to_fix = (f"Tell it where the old grid's north-west corner was: GridPaint.load(path, grid, "
                  f"old_origin_m=({guessed_x0:g}, <old north edge y in metres>)), or  --old-origin {guessed_x0:g} <y>.")
    if hits[best] < MIN_OVERLAP_FRACTION * total:
        raise ValueError(f"{path} was painted on a different grid and does not line up with this one: at the best "
                         f"north/south shift only {int(hits[best])} of {total} painted cells and wall edges land on "
                         f"the new floor and walls. {how_to_fix}")
    if hits[best] - runner_up < max(1, 0.02 * total):
        raise ValueError(f"{path} was painted on a different grid and could fit the new one at more than one "
                         f"north/south shift (best {int(hits[best])} and {runner_up} of {total}). {how_to_fix}")
    return candidates[best]


def shifted_layers(saved: Dict[str, np.ndarray], grid: Grid, row_shift: int, column_shift: int) -> Dict[str, np.ndarray]:
    """Every saved layer on this grid's lattice.  Per-seat layers are kept only if the seat count is unchanged."""
    shapes = layer_shapes(grid)
    layers = {}
    for name in LAYER_NAMES:
        if is_seat_layer(name):
            layers[name] = saved[name] if saved[name].shape == shapes[name] else np.zeros(shapes[name], bool)
        else:
            layers[name] = shift_flags(saved[name], shapes[name], row_shift, column_shift)
    return layers


# --------------------------------------------------------------------- paint --
@dataclass
class GridPaint:
    cell_has_led: np.ndarray
    cell_is_barrier: np.ndarray
    wall_has_led_horizontal: np.ndarray
    wall_has_led_vertical: np.ndarray
    seat_has_led: np.ndarray
    seat_is_barrier: np.ndarray
    walkway_has_led: np.ndarray
    walkway_is_barrier: np.ndarray

    # ------------------------------------------------------------ construction --
    @classmethod
    def empty(cls, grid: Grid) -> "GridPaint":
        return cls(**{name: np.zeros(shape, bool) for name, shape in layer_shapes(grid).items()})

    @classmethod
    def from_layers(cls, layers: Dict[str, np.ndarray], grid: Grid) -> "GridPaint":
        shapes = layer_shapes(grid)
        checked = {}
        for name in LAYER_NAMES:
            flags = np.asarray(layers[name]).astype(bool)
            if flags.shape != shapes[name]:
                raise ValueError(f"{name} has shape {flags.shape}, this grid needs {shapes[name]}")
            checked[name] = flags
        return cls(**checked)

    def clip_to_grid(self, grid: Grid, wall_edges: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> int:
        """Drop paint that sits on something that is not there (a cell that is not walkable, an edge
        with no wall).  Returns how many flags were dropped.  Matters when the grid was rebuilt."""
        horizontal_wall, vertical_wall = wall_edges or find_wall_edges(grid)
        before = self.count_flags()
        self.cell_has_led &= grid.walkable
        self.cell_is_barrier &= grid.walkable
        self.wall_has_led_horizontal &= horizontal_wall
        self.wall_has_led_vertical &= vertical_wall
        return before - self.count_flags()

    def count_flags(self) -> int:
        return int(sum(int(getattr(self, name).sum()) for name in LAYER_NAMES))

    # ------------------------------------------------------------- node arrays --
    def _node_array(self, grid: Grid, cell_flags: np.ndarray, seat_flags: np.ndarray,
                    walkway_flags: np.ndarray) -> np.ndarray:
        node_flags = np.zeros(grid.num_nodes, bool)
        node_flags[grid.cell_node[grid.walkable]] = cell_flags[grid.walkable]
        node_flags[grid.spawn] = seat_flags
        node_flags[grid.walkway_nodes] = walkway_flags
        return node_flags

    def node_has_led(self, grid: Grid) -> np.ndarray:
        """has_led for every graph node (lattice cells, then seats, then walkways)."""
        return self._node_array(grid, self.cell_has_led, self.seat_has_led, self.walkway_has_led)

    def node_is_barrier(self, grid: Grid) -> np.ndarray:
        """is_barrier for every graph node.  Pass it as `blocked=` to Grid.node_distance_field."""
        return self._node_array(grid, self.cell_is_barrier, self.seat_is_barrier, self.walkway_is_barrier)

    # -------------------------------------------------------------- wall LEDs --
    def wall_led_segments(self, grid: Grid) -> List[dict]:
        """Every wall edge with LEDs, as a line between two points in metres."""
        cell_size = grid.cell
        segments = []
        for level_index, row, column in zip(*np.nonzero(self.wall_has_led_horizontal)):
            y = grid.y0 - row * cell_size
            x_west = grid.x0 + column * cell_size
            segments.append(dict(level=LEVEL_NAMES[level_index], direction="east-west",
                                 x1_m=x_west, y1_m=y, x2_m=x_west + cell_size, y2_m=y, length_m=cell_size))
        for level_index, row, column in zip(*np.nonzero(self.wall_has_led_vertical)):
            x = grid.x0 + column * cell_size
            y_north = grid.y0 - row * cell_size
            segments.append(dict(level=LEVEL_NAMES[level_index], direction="north-south",
                                 x1_m=x, y1_m=y_north, x2_m=x, y2_m=y_north - cell_size, length_m=cell_size))
        return [{key: (round(float(value), 3) if isinstance(value, (float, np.floating)) else value)
                 for key, value in segment.items()} for segment in segments]

    # ------------------------------------------------------------------- files --
    def save(self, path: str, grid: Grid):
        temporary_path = path + ".tmp.npz"
        np.savez_compressed(
            temporary_path, paint_format_version=PAINT_FORMAT_VERSION, cell_size_m=grid.cell,
            origin_x_m=grid.x0, origin_y_m=grid.y0,      # west edge of column 0, north edge of row 0
            grid_shape=np.array(grid.shape), num_nodes=grid.num_nodes,
            node_has_led=self.node_has_led(grid), node_is_barrier=self.node_is_barrier(grid),
            **{name: getattr(self, name) for name in LAYER_NAMES})
        os.replace(temporary_path, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as summary_file:
            json.dump(summarize(grid, self), summary_file, indent=1)

    @classmethod
    def load(cls, path: str, grid: Grid, old_origin_m: Optional[Tuple[float, float]] = None,
             wall_edges: Optional[Tuple[np.ndarray, np.ndarray]] = None, verbose: bool = True) -> "GridPaint":
        """Open a paint file on `grid`, moving the paint onto the current lattice if the grid has changed.

        old_origin_m=(x0, y0) is the old lattice's west and north edge in metres; it is used only for files
        that did not store their own corner.  wall_edges (from find_wall_edges) saves recomputing them.
        verbose=False silences the note printed when paint had to be moved.  See the module docstring."""
        with np.load(path, allow_pickle=False) as archive:
            if "paint_format_version" not in archive.files or int(archive["paint_format_version"]) != PAINT_FORMAT_VERSION:
                raise ValueError(f"{path} is not a paint file this version can read")
            saved = {name: np.asarray(archive[name]).astype(bool) for name in LAYER_NAMES}
            saved_cell = float(archive["cell_size_m"])
            saved_origin = ((float(archive["origin_x_m"]), float(archive["origin_y_m"]))
                            if "origin_x_m" in archive.files else None)
        if not np.isclose(saved_cell, grid.cell):
            raise ValueError(f"{path} was painted with {saved_cell:g} m cells but this grid uses {grid.cell:g} m; "
                             "paint it again at the new cell size")
        saved_shape = tuple(saved["cell_has_led"].shape)
        new_shape = tuple(grid.shape)
        if saved_shape[0] != new_shape[0]:
            raise ValueError(f"{path} has {saved_shape[0]} levels but this grid has {new_shape[0]}")
        if saved_origin is None and old_origin_m is not None:
            saved_origin = (float(old_origin_m[0]), float(old_origin_m[1]))

        if saved_origin is not None:
            row_shift, column_shift = origin_shift(saved_origin, grid)
        elif saved_shape == new_shape:
            row_shift = column_shift = 0                      # old file, same lattice: assume the corner is unchanged
        else:
            column_shift = (new_shape[2] // 2) - (saved_shape[2] // 2)       # grid.py puts x = 0 on a cell edge
            row_shift = find_row_shift(saved, grid, column_shift, wall_edges or find_wall_edges(grid), path)

        paint = cls.from_layers(shifted_layers(saved, grid, row_shift, column_shift), grid)

        if verbose:
            notes = []
            if saved_shape != new_shape or row_shift or column_shift:
                lost = sum(int(saved[name].sum()) - int(getattr(paint, name).sum())
                           for name in LAYER_NAMES if not is_seat_layer(name))
                notes.append(f"lattice {saved_shape[1]}x{saved_shape[2]} -> {new_shape[1]}x{new_shape[2]}, paint moved "
                             f"{row_shift:+d} rows (+ = south) and {column_shift:+d} columns (+ = east)"
                             + (f", {lost} flags fell off the new lattice" if lost else ""))
            if saved["seat_has_led"].shape != layer_shapes(grid)["seat_has_led"]:
                notes.append(f"the seat count changed ({saved['seat_has_led'].shape[0]} -> {len(grid.spawn)}), "
                             "so the per-seat and row-walkway flags were dropped")
            painted_total = (int((paint.cell_has_led | paint.cell_is_barrier).sum())
                             + int(paint.wall_has_led_horizontal.sum()) + int(paint.wall_has_led_vertical.sum()))
            if painted_total:
                horizontal_wall, vertical_wall = wall_edges or find_wall_edges(grid)
                on_grid = (int(((paint.cell_has_led | paint.cell_is_barrier) & grid.walkable).sum())
                           + int((paint.wall_has_led_horizontal & horizontal_wall).sum())
                           + int((paint.wall_has_led_vertical & vertical_wall).sum()))
                if on_grid < MIN_OVERLAP_FRACTION * painted_total:
                    notes.append(f"only {on_grid} of {painted_total} painted cells and wall edges lie on the current floor "
                                 "and walls; open it in the painter and check it")
            if notes:
                print(f"{path}: the grid has changed since this was painted:\n  " + "\n  ".join(notes)
                      + "\n  Open it in the painter and press Save to store it for the new grid.")
        return paint


def paint_counts(grid: Grid, paint: GridPaint) -> dict:
    wall_led_count = int(paint.wall_has_led_horizontal.sum() + paint.wall_has_led_vertical.sum())
    return dict(led_floor_cells=int(paint.cell_has_led.sum()),
                barrier_floor_cells=int(paint.cell_is_barrier.sum()),
                led_wall_segments=wall_led_count,
                led_wall_length_m=round(wall_led_count * grid.cell, 3),
                led_seats=int(paint.seat_has_led.sum()), barrier_seats=int(paint.seat_is_barrier.sum()),
                led_row_walkways=int(paint.walkway_has_led.sum()),
                barrier_row_walkways=int(paint.walkway_is_barrier.sum()))


def summarize(grid: Grid, paint: GridPaint) -> dict:
    """Human readable list of everything painted, with positions in metres."""

    def floor_cells(flags: np.ndarray) -> List[dict]:
        levels, rows, columns = np.nonzero(flags)
        x_values, y_values = grid.xy(rows, columns)
        return [dict(level=LEVEL_NAMES[int(level)], row=int(row), column=int(column),
                     x_m=round(float(x), 3), y_m=round(float(y), 3))
                for level, row, column, x, y in zip(levels, rows, columns, x_values, y_values)]

    def nodes(flags: np.ndarray, node_ids: np.ndarray) -> List[dict]:
        items = []
        for seat_index in np.nonzero(flags)[0].tolist():
            node = int(node_ids[seat_index])
            item = dict(seat_index=seat_index, level=LEVEL_NAMES[int(grid.node_level[node])],
                        x_m=round(float(grid.node_x[node]), 3), y_m=round(float(grid.node_y[node]), 3))
            item["seat"] = f"{grid.row_names[int(grid.seat_row[seat_index])]} seat {int(grid.seat_number[seat_index])}"
            items.append(item)
        return items

    return dict(
        cell_size_m=grid.cell,
        counts=paint_counts(grid, paint),
        led_floor_cells=floor_cells(paint.cell_has_led),
        barrier_floor_cells=floor_cells(paint.cell_is_barrier),
        led_wall_segments=paint.wall_led_segments(grid),
        led_seats=nodes(paint.seat_has_led, grid.spawn),
        barrier_seats=nodes(paint.seat_is_barrier, grid.spawn),
        led_row_walkways=nodes(paint.walkway_has_led, grid.walkway_nodes),
        barrier_row_walkways=nodes(paint.walkway_is_barrier, grid.walkway_nodes))


# ------------------------------------------------------------------- layouts --
# A layout name becomes a file name, so keep it simple: letters, digits, space, underscore, dash and dot,
# starting with a letter or digit, at most 64 characters.
LAYOUT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


def valid_layout_name(name: str) -> bool:
    return bool(LAYOUT_NAME_PATTERN.fullmatch(name)) and not name.endswith(".") and not name.lower().endswith(".tmp")


def check_layout_name(name) -> str:
    """Strip and validate a layout name; raise ValueError with a readable message if it cannot be used."""
    cleaned = str(name).strip()
    if not valid_layout_name(cleaned):
        raise ValueError("a layout name must be 1-64 characters: letters, digits, spaces, _ - . "
                         "(starting with a letter or digit, not ending in '.' or '.tmp')")
    return cleaned


def layout_path(layouts_dir: str, name: str) -> str:
    return os.path.join(layouts_dir, name + ".npz")


def layout_summary_path(layouts_dir: str, name: str) -> str:
    return os.path.join(layouts_dir, name + ".json")


def list_layouts(grid: Grid, layouts_dir: str, wall_edges=None, old_origin_m=None) -> List[dict]:
    """Every layout file in the folder, with its counts (or the reason it cannot be read on this grid)."""
    items = []
    if not os.path.isdir(layouts_dir):
        return items
    for file_name in sorted(os.listdir(layouts_dir), key=str.lower):
        if not file_name.endswith(".npz") or file_name.endswith(".tmp.npz"):
            continue
        name = file_name[:-len(".npz")]
        if not valid_layout_name(name):
            continue
        path = os.path.join(layouts_dir, file_name)
        item = dict(name=name, modified=os.path.getmtime(path))
        try:
            paint = GridPaint.load(path, grid, old_origin_m=old_origin_m, wall_edges=wall_edges, verbose=False)
            item.update(ok=True, counts=paint_counts(grid, paint))
        except Exception as problem:
            item.update(ok=False, error=f"{type(problem).__name__}: {problem}")
        items.append(item)
    return items


def import_legacy_paint(grid: Grid, wall_edges, layouts_dir: str, old_origin_m=None) -> bool:
    """Copy the old single paint file (out/grid_paint_<cell>m.npz) in as the layout "default", once."""
    legacy_path = os.path.join(HERE, "out", f"grid_paint_{grid.cell:g}m.npz")
    target_path = layout_path(layouts_dir, DEFAULT_LAYOUT_NAME)
    if os.path.exists(target_path) or not os.path.exists(legacy_path):
        return False
    try:
        paint = GridPaint.load(legacy_path, grid, old_origin_m=old_origin_m, wall_edges=wall_edges)
        paint.clip_to_grid(grid, wall_edges)
        paint.save(target_path, grid)
    except ValueError as problem:
        print(f"could not import {legacy_path}: {problem}")
        return False
    print(f"copied the old paint file {legacy_path} into the layouts folder as '{DEFAULT_LAYOUT_NAME}'")
    return True


# --------------------------------------------------------------------- scene --
def build_scene(grid: Grid, wall_edges: Tuple[np.ndarray, np.ndarray]) -> dict:
    """Everything the web page needs to draw the grid (not the paint), as JSON-friendly values."""
    horizontal_wall, vertical_wall = wall_edges
    seat_nodes, walkway_nodes = grid.spawn, grid.walkway_nodes

    exit_labels = []
    for exit_index, label in enumerate(grid.exits):
        for level_index in range(grid.shape[0]):
            rows, columns = np.nonzero(grid.exit_id[level_index] == exit_index)
            if len(rows) == 0:
                continue
            x_values, y_values = grid.xy(rows, columns)
            # one label can cover two separate doors (the portico), so split at any gap in y
            by_y = np.argsort(y_values)
            for group in np.split(by_y, np.nonzero(np.diff(y_values[by_y]) > 2 * grid.cell)[0] + 1):
                exit_labels.append(dict(label=label, level=level_index,
                                        x=float(x_values[group].mean()), y=float(y_values[group].mean())))

    def node_positions(node_ids: np.ndarray) -> dict:
        return dict(x=np.round(grid.node_x[node_ids], 3).tolist(), y=np.round(grid.node_y[node_ids], 3).tolist(),
                    level=grid.node_level[node_ids].astype(int).tolist())

    seats = node_positions(seat_nodes)
    seats["label"] = [f"{grid.row_names[int(grid.seat_row[i])]} seat {int(grid.seat_number[i])}"
                      for i in range(len(seat_nodes))]
    walkways = node_positions(walkway_nodes)
    walkways["label"] = seats["label"]

    return dict(
        cellSizeMeters=grid.cell, westEdgeX=grid.x0, northEdgeY=grid.y0,
        numLevels=grid.shape[0], numRows=grid.shape[1], numColumns=grid.shape[2],
        levelNames=list(LEVEL_NAMES), exitNames=list(grid.exits),
        walkable=encode_flags(grid.walkable),
        exitIndex=base64.b64encode(np.where(grid.exit_id >= 0, grid.exit_id, 255).astype(np.uint8).tobytes()).decode("ascii"),
        isDoor=encode_flags((grid.door_id >= 0) & (grid.exit_id < 0)),
        isStairs=encode_flags(grid.walkable & (grid.cell_cost > COST_FLOOR)),
        horizontalWall=encode_flags(horizontal_wall), verticalWall=encode_flags(vertical_wall),
        exitLabels=exit_labels, seats=seats, walkways=walkways)


# The layout controls added to the page when grid_paint.html does not have its own (id="layoutBar").
# It only talks to the /api/layout... endpoints below, so it does not depend on how the page keeps its paint.
LAYOUT_BAR_SNIPPET = """
<div id="layoutBar" style="position:fixed;right:8px;bottom:8px;z-index:99999;display:flex;gap:6px;align-items:center;
     flex-wrap:wrap;max-width:96vw;padding:6px 8px;background:rgba(255,255,255,.95);color:#222;border:1px solid #888;
     border-radius:6px;font:13px/1.3 system-ui,sans-serif;box-shadow:0 1px 6px rgba(0,0,0,.25)">
  <b>Layout</b>
  <select id="layoutBarSelect" style="max-width:200px"></select>
  <button type="button" id="layoutBarOpen">Open</button>
  <button type="button" id="layoutBarSaveAs">Save as...</button>
  <button type="button" id="layoutBarRename">Rename</button>
  <button type="button" id="layoutBarDelete">Delete</button>
  <span id="layoutBarStatus" style="color:#444"></span>
</div>
<script>
(function () {
  const select = document.getElementById("layoutBarSelect");
  const status = document.getElementById("layoutBarStatus");
  async function call(path, body) {
    const options = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"},
                                               body: JSON.stringify(body)};
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok || result.ok === false) throw new Error(result.error || response.statusText);
    return result;
  }
  async function refresh() {
    try {
      const info = await call("/api/layouts");
      const keep = select.value;
      select.innerHTML = "";
      for (const layout of info.layouts) {
        const option = document.createElement("option");
        option.value = layout.name;
        option.textContent = layout.name + (layout.ok ? "" : " (unreadable)");
        select.appendChild(option);
      }
      const names = info.layouts.map(layout => layout.name);
      select.value = names.includes(keep) ? keep : (names.includes(info.current) ? info.current : "");
      status.textContent = "Save writes to: " + info.current + (names.includes(info.current) ? "" : " (new)");
    } catch (problem) {
      status.textContent = "layouts: " + problem.message;
    }
  }
  async function act(path, body, done) {
    try { await call(path, body); if (done) done(); else await refresh(); }
    catch (problem) { alert(problem.message); }
  }
  document.getElementById("layoutBarOpen").onclick = () => {
    if (!select.value) return;
    if (!confirm("Open '" + select.value + "'? Paint you have not saved will be lost.")) return;
    act("/api/layout/open", {name: select.value}, () => location.reload());
  };
  document.getElementById("layoutBarSaveAs").onclick = () => {
    const name = prompt("Name for the new layout (then press the page's Save button to write it):");
    if (name) act("/api/layout/target", {name: name});
  };
  document.getElementById("layoutBarRename").onclick = () => {
    if (!select.value) return;
    const newName = prompt("New name for '" + select.value + "':", select.value);
    if (newName && newName !== select.value) act("/api/layout/rename", {name: select.value, newName: newName});
  };
  document.getElementById("layoutBarDelete").onclick = () => {
    if (!select.value) return;
    if (!confirm("Delete the saved layout '" + select.value + "'? This cannot be undone.")) return;
    act("/api/layout/delete", {name: select.value});
  };
  refresh();
  setInterval(refresh, 3000);   // picks up the page's own Save button
})();
</script>
"""


def add_layout_bar(html_text: str) -> str:
    if 'id="layoutBar"' in html_text:
        return html_text
    body_end = html_text.lower().rfind("</body>")
    if body_end < 0:
        return html_text + LAYOUT_BAR_SNIPPET
    return html_text[:body_end] + LAYOUT_BAR_SNIPPET + html_text[body_end:]


# -------------------------------------------------------------------- server --
class PaintServer(ThreadingHTTPServer):
    def __init__(self, address, grid: Grid, wall_edges, layouts_dir: str, layout_name: str, old_origin_m=None):
        super().__init__(address, PaintRequestHandler)
        self.grid = grid
        self.wall_edges = wall_edges
        self.old_origin_m = old_origin_m            # corner of the old lattice, for files that did not store theirs
        self.layouts_dir = layouts_dir
        self.current_layout = layout_name           # the layout the page's Save button writes to
        self.lock = threading.Lock()
        self.static_scene = build_scene(grid, wall_edges)

    # ------------------------------------------------------------ layout files --
    def read_layout(self, name: str) -> GridPaint:
        """The saved layout, or an empty one if it has not been saved yet."""
        path = layout_path(self.layouts_dir, name)
        if not os.path.exists(path):
            return GridPaint.empty(self.grid)
        paint = GridPaint.load(path, self.grid, old_origin_m=self.old_origin_m, wall_edges=self.wall_edges)
        paint.clip_to_grid(self.grid, self.wall_edges)
        return paint

    def scene(self) -> dict:
        try:
            paint = self.read_layout(self.current_layout)
        except ValueError as problem:
            print(f"{problem}\nshowing an empty paint; saving will replace layout '{self.current_layout}'")
            paint = GridPaint.empty(self.grid)
        return dict(self.static_scene,
                    paint={camel_case(name): encode_flags(getattr(paint, name)) for name in LAYER_NAMES},
                    layoutName=self.current_layout, layoutsDir=self.layouts_dir,
                    savePath=layout_path(self.layouts_dir, self.current_layout))

    def layouts_info(self) -> dict:
        return dict(ok=True, current=self.current_layout, dir=self.layouts_dir,
                    layouts=list_layouts(self.grid, self.layouts_dir, self.wall_edges, self.old_origin_m))


class PaintRequestHandler(BaseHTTPRequestHandler):
    server: PaintServer

    def log_message(self, format, *args):    # keep the terminal quiet
        pass

    def _reply(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _reply_json(self, status: int, payload: dict):
        self._reply(status, json.dumps(payload).encode("utf-8"), "application/json")

    # --------------------------------------------------------------------- GET --
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            with open(HTML_PATH, "r", encoding="utf-8") as html_file:      # read each time so edits show on reload
                html_text = add_layout_bar(html_file.read())
            self._reply(200, html_text.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/scene":
            with self.server.lock:
                scene = self.server.scene()
            self._reply_json(200, scene)
        elif path == "/api/layouts":
            with self.server.lock:
                info = self.server.layouts_info()
            self._reply_json(200, info)
        elif path == "/favicon.ico":
            self._reply(204, b"", "image/x-icon")
        else:
            self._reply(404, b"not found", "text/plain")

    # -------------------------------------------------------------------- POST --
    def do_POST(self):
        routes = {"/api/save": self._save,
                  "/api/layout/open": self._open_layout,
                  "/api/layout/target": self._set_save_target,
                  "/api/layout/rename": self._rename_layout,
                  "/api/layout/delete": self._delete_layout}
        handler = routes.get(urlsplit(self.path).path)
        if handler is None:
            self._reply(404, b"not found", "text/plain")
            return
        try:
            body_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(body_length) or b"{}")
            with self.server.lock:
                result = handler(payload)
            self._reply_json(200, result)
        except FileNotFoundError as problem:
            self._reply_json(404, dict(ok=False, error=str(problem)))
        except Exception as problem:
            self._reply_json(400, dict(ok=False, error=f"{type(problem).__name__}: {problem}"))

    def _save(self, payload: dict) -> dict:
        """Save the painted layers.  "name" (optional) is the layout to write; without it the current one is used."""
        server = self.server
        grid = server.grid
        name = check_layout_name(payload["name"]) if payload.get("name") else server.current_layout
        shapes = layer_shapes(grid)
        layers = {layer: decode_flags(payload["layers"][camel_case(layer)], shapes[layer]) for layer in LAYER_NAMES}
        paint = GridPaint.from_layers(layers, grid)
        dropped = paint.clip_to_grid(grid, server.wall_edges)
        path = layout_path(server.layouts_dir, name)
        if os.path.exists(path):
            try:
                GridPaint.load(path, grid, old_origin_m=server.old_origin_m, wall_edges=server.wall_edges, verbose=False)
            except ValueError:
                shutil.copy2(path, path + ".bak")      # it does not fit this grid; keep the original before replacing it
        paint.save(path, grid)
        server.current_layout = name
        return dict(ok=True, name=name, path=path, summaryPath=layout_summary_path(server.layouts_dir, name),
                    droppedFlags=dropped, counts=paint_counts(grid, paint))

    def _open_layout(self, payload: dict) -> dict:
        """Make a saved layout the current one.  The page then reloads and gets its paint from /api/scene."""
        name = check_layout_name(payload.get("name", ""))
        if not os.path.exists(layout_path(self.server.layouts_dir, name)):
            raise FileNotFoundError(f"there is no layout called '{name}'")
        self.server.read_layout(name)             # raises ValueError if it was painted on a different grid
        self.server.current_layout = name
        return dict(ok=True, name=name)

    def _set_save_target(self, payload: dict) -> dict:
        """Choose the name the next Save writes to ("Save as").  Nothing is written yet."""
        name = check_layout_name(payload.get("name", ""))
        self.server.current_layout = name
        return dict(ok=True, name=name, exists=os.path.exists(layout_path(self.server.layouts_dir, name)))

    def _rename_layout(self, payload: dict) -> dict:
        server = self.server
        name = check_layout_name(payload.get("name", ""))
        new_name = check_layout_name(payload.get("newName", ""))
        old_path, new_path = layout_path(server.layouts_dir, name), layout_path(server.layouts_dir, new_name)
        if not os.path.exists(old_path):
            raise FileNotFoundError(f"there is no layout called '{name}'")
        if name.lower() != new_name.lower() and os.path.exists(new_path):
            raise ValueError(f"a layout called '{new_name}' already exists")
        os.replace(old_path, new_path)
        old_summary = layout_summary_path(server.layouts_dir, name)
        if os.path.exists(old_summary):
            os.replace(old_summary, layout_summary_path(server.layouts_dir, new_name))
        if server.current_layout == name:
            server.current_layout = new_name
        return dict(ok=True, name=new_name)

    def _delete_layout(self, payload: dict) -> dict:
        server = self.server
        name = check_layout_name(payload.get("name", ""))
        path = layout_path(server.layouts_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"there is no layout called '{name}'")
        os.remove(path)
        summary_path = layout_summary_path(server.layouts_dir, name)
        if os.path.exists(summary_path):
            os.remove(summary_path)
        # If it was the current layout, the name stays as the target of the next Save (it is now a new layout).
        return dict(ok=True, name=name)


# ---------------------------------------------------------------------- main --
def load_or_build_grid(grid_path: Optional[str], cell_m: float) -> Grid:
    if grid_path is None:
        default_path = os.path.join(HERE, "out", f"grid_hybrid_{cell_m:g}m.npz")
        grid_path = default_path if os.path.exists(default_path) else None
    if grid_path is not None:
        try:
            grid = Grid.load(grid_path)
            print(f"loaded grid {grid_path}")
            return grid
        except ValueError as problem:
            print(f"{problem}")
    print(f"building the {cell_m:g} m grid from model.py ...")
    return Grid.from_model(cell_m=cell_m)


def main():
    parser = argparse.ArgumentParser(description="Paint LEDs and barriers onto the Bailey Hall grid.")
    parser.add_argument("--grid", help="saved grid .npz (default: out/grid_hybrid_<cell>m.npz if it exists, else build)")
    parser.add_argument("--cell-m", type=float, default=0.5, help="cell size in metres when the grid is built")
    parser.add_argument("--layouts-dir", help="folder that holds the saved layouts (default: out/layouts_<cell>m)")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT_NAME,
                        help=f"layout to open first and save to (default: {DEFAULT_LAYOUT_NAME}); "
                             "a name that does not exist yet starts an empty layout")
    parser.add_argument("--old-origin", type=float, nargs=2, metavar=("X", "Y"),
                        help="west edge x and north edge y (metres) of the lattice that older layout files were "
                             "painted on, if the grid has changed and those files did not store it")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    arguments = parser.parse_args()

    grid = load_or_build_grid(arguments.grid, arguments.cell_m)
    wall_edges = find_wall_edges(grid)
    layouts_dir = os.path.abspath(arguments.layouts_dir or os.path.join(HERE, "out", f"layouts_{grid.cell:g}m"))
    os.makedirs(layouts_dir, exist_ok=True)
    layout_name = check_layout_name(arguments.layout)

    old_origin_m = tuple(arguments.old_origin) if arguments.old_origin else None
    import_legacy_paint(grid, wall_edges, layouts_dir, old_origin_m)

    server = PaintServer(("127.0.0.1", arguments.port), grid, wall_edges, layouts_dir, layout_name, old_origin_m)

    saved_layouts = list_layouts(grid, layouts_dir, wall_edges, old_origin_m)
    print(f"layouts folder {layouts_dir}: " + (", ".join(item["name"] for item in saved_layouts) or "no saved layouts yet"))
    if os.path.exists(layout_path(layouts_dir, layout_name)):
        try:
            server.read_layout(layout_name)
            print(f"opened layout '{layout_name}'")
        except ValueError as problem:
            print(f"{problem}\nstarting with an empty paint; saving will replace layout '{layout_name}'")
    else:
        print(f"layout '{layout_name}' is new; it is created when you press Save")

    address = f"http://127.0.0.1:{arguments.port}/"
    print(f"painter running at {address}   (Ctrl+C to stop)")
    if not arguments.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()