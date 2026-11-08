"""
Paint LEDs and barriers onto the Bailey Hall hybrid grid.

    python bailey_hall/grid_paint.py                       # load (or build) the 0.5 m grid and open the painter
    python bailey_hall/grid_paint.py --cell-m 0.25
    python bailey_hall/grid_paint.py --grid out/grid_hybrid_0.5m.npz --paint out/my_paint.npz

The painter is a small local web page (grid_paint.html, same folder).  This file builds the grid, serves
the page, and saves what you paint.  Put both files next to grid.py and model.py.

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
    paint = GridPaint.load("out/grid_paint_0.5m.npz", grid)
    paint.cell_has_led                    # bool (2, H, W)
    paint.wall_led_segments(grid)         # list of dicts with both end points in metres
    blocked = paint.node_is_barrier(grid) # bool (nodes,), accepted by grid.node_distance_field(blocked=...)
    field = grid.node_distance_field(blocked=blocked)

Saving writes <paint>.npz (arrays) and <paint>.json (a human readable list of every LED and barrier).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple

import numpy as np

from grid import COST_FLOOR, EDGE_LATTICE, LEVEL_NAMES, Grid

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "grid_paint.html")

PAINT_FORMAT_VERSION = 1

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
            grid_shape=np.array(grid.shape), num_nodes=grid.num_nodes,
            node_has_led=self.node_has_led(grid), node_is_barrier=self.node_is_barrier(grid),
            **{name: getattr(self, name) for name in LAYER_NAMES})
        os.replace(temporary_path, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as summary_file:
            json.dump(summarize(grid, self), summary_file, indent=1)

    @classmethod
    def load(cls, path: str, grid: Grid) -> "GridPaint":
        with np.load(path, allow_pickle=False) as archive:
            if "paint_format_version" not in archive.files or int(archive["paint_format_version"]) != PAINT_FORMAT_VERSION:
                raise ValueError(f"{path} is not a paint file this version can read")
            if tuple(archive["grid_shape"].tolist()) != tuple(grid.shape) or int(archive["num_nodes"]) != grid.num_nodes \
                    or not np.isclose(float(archive["cell_size_m"]), grid.cell):
                raise ValueError(f"{path} was painted on a different grid (shape, cell size or node count differ)")
            paint = cls.from_layers({name: archive[name] for name in LAYER_NAMES}, grid)
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


# --------------------------------------------------------------------- scene --
def build_scene(grid: Grid, wall_edges: Tuple[np.ndarray, np.ndarray], paint: GridPaint, paint_path: str) -> dict:
    """Everything the web page needs to draw the grid, as JSON-friendly values."""
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
        exitLabels=exit_labels, seats=seats, walkways=walkways,
        paint={camel_case(name): encode_flags(getattr(paint, name)) for name in LAYER_NAMES},
        savePath=paint_path)


# -------------------------------------------------------------------- server --
class PaintServer(ThreadingHTTPServer):
    def __init__(self, address, grid: Grid, wall_edges, paint: GridPaint, paint_path: str):
        super().__init__(address, PaintRequestHandler)
        self.grid = grid
        self.wall_edges = wall_edges
        self.paint_path = paint_path
        self.scene = build_scene(grid, wall_edges, paint, paint_path)


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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(HTML_PATH, "rb") as html_file:      # read each time so edits show on reload
                self._reply(200, html_file.read(), "text/html; charset=utf-8")
        elif self.path == "/api/scene":
            self._reply_json(200, self.server.scene)
        elif self.path == "/favicon.ico":
            self._reply(204, b"", "image/x-icon")
        else:
            self._reply(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._reply(404, b"not found", "text/plain")
            return
        try:
            body_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(body_length))
            grid = self.server.grid
            shapes = layer_shapes(grid)
            layers = {name: decode_flags(payload["layers"][camel_case(name)], shapes[name]) for name in LAYER_NAMES}
            paint = GridPaint.from_layers(layers, grid)
            dropped = paint.clip_to_grid(grid, self.server.wall_edges)
            paint.save(self.server.paint_path, grid)
            self._reply_json(200, dict(ok=True, path=self.server.paint_path,
                                       summaryPath=os.path.splitext(self.server.paint_path)[0] + ".json",
                                       droppedFlags=dropped, counts=paint_counts(grid, paint)))
        except Exception as problem:
            self._reply_json(400, dict(ok=False, error=f"{type(problem).__name__}: {problem}"))


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
    parser.add_argument("--paint", help="paint file to open and save (default: out/grid_paint_<cell>m.npz)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    arguments = parser.parse_args()

    grid = load_or_build_grid(arguments.grid, arguments.cell_m)
    wall_edges = find_wall_edges(grid)
    paint_path = arguments.paint or os.path.join(HERE, "out", f"grid_paint_{grid.cell:g}m.npz")
    os.makedirs(os.path.dirname(os.path.abspath(paint_path)), exist_ok=True)

    paint = GridPaint.empty(grid)
    if os.path.exists(paint_path):
        try:
            paint = GridPaint.load(paint_path, grid)
            dropped = paint.clip_to_grid(grid, wall_edges)
            print(f"opened {paint_path}" + (f" ({dropped} flags were on things that no longer exist and were dropped)"
                                           if dropped else ""))
        except ValueError as problem:
            print(f"{problem}\nstarting with an empty paint; saving will replace {paint_path}")

    server = PaintServer(("127.0.0.1", arguments.port), grid, wall_edges, paint, paint_path)
    address = f"http://127.0.0.1:{arguments.port}/"
    print(f"painter running at {address}   (Ctrl+C to stop)\nsaves to {paint_path}")
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