"""
Hybrid spatial grid of Bailey Hall: a 0.5 m lattice for open floor plus a graph for seat rows.
This file builds the space a later simulation runs on.  It contains no simulation.

    from grid import Grid
    grid = Grid.from_model(cell_m=0.5)
    grid.walkable        # bool    (2, H, W)  lattice cells; level 0 = first floor, 1 = balcony
    grid.exit_id         # int16   (2, H, W)  -1 = none, else index into grid.exits (11 labels)
    grid.door_id         # int16   (2, H, W)  -1 = none, else index into grid.doors
    grid.cell_cost       # float   (2, H, W)  cost of entering a cell (floor 1, stairs 2)
    grid.cell_node       # int32   (2, H, W)  -1 = none, else node id of that cell
    grid.dist            # float   (2, H, W)  cheapest cost to any exit from each lattice cell
    grid.node_dist       # float   (nodes,)   the same for every node, seats and walkways included
    grid.spawn           # int32   (seats,)   node id of each seat
    grid.seat_section / seat_row / seat_number / walkway_nodes     # per seat
    grid.node_kind / node_level / node_x / node_y / node_cost      # per node
    grid.edge_from / edge_to / edge_kind / edge_length / edge_cost # per directed edge
    grid.exit_mask("F3") / grid.door_mask("foyer_door_+0")         # masks for closing a gate
    grid.distance_field(targets, blocked=mask)                     # no rebuild needed
    grid.xy(row, column) / grid.ij(x, y)                           # lattice <-> metres
    grid.save(path) / Grid.load(path)

Every length, position and cost here is in METRES (a cost is metres weighted by the entry cost
of the destination).  model.py stays in feet, and from_model converts its geometry once, so
this file never holds a length in feet.

Lattice rows go north -> south and columns west -> east, like an image.  Both levels share one
coordinate frame and one lattice.

    python bailey_hall/grid.py [cell_m]     -> out/grid_hybrid_<cell>m.npz + .png + validation
"""
from __future__ import annotations

import heapq
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import shapely

from model import FT, Model

HERE = os.path.dirname(os.path.abspath(__file__))

FORMAT_VERSION = 2

FIRST_FLOOR, BALCONY = 0, 1
LEVEL_NAMES = ("first_floor", "balcony")

COST_FLOOR = 1.0        # also doors, exits and walkway nodes
COST_STAIRS = 2.0
COST_SEAT = 2.5

NODE_CELL, NODE_SEAT, NODE_WALKWAY = 0, 1, 2
NODE_KIND_NAMES = ("cell", "seat", "walkway")

EDGE_LATTICE, EDGE_SEAT_SEAT, EDGE_SEAT_WALKWAY, EDGE_WALKWAY_WALKWAY, EDGE_ROW_LINK, EDGE_STAIR = range(6)
EDGE_KIND_NAMES = ("lattice", "seat_seat", "seat_walkway", "walkway_walkway", "row_link", "stair")

# Exit label -> (level, model door names).  The portico has four physical doors but two labels.
EXIT_DOORS: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    "F1": (FIRST_FLOOR, ("front_door_-20",)),
    "F2": (FIRST_FLOOR, ("front_door_-10",)),
    "F3": (FIRST_FLOOR, ("front_door_+0",)),
    "F4": (FIRST_FLOOR, ("front_door_+10",)),
    "F5": (FIRST_FLOOR, ("front_door_+20",)),
    "PW": (FIRST_FLOOR, ("portico_door_125", "portico_door_141")),
    "PE": (FIRST_FLOOR, ("portico_door_40", "portico_door_55")),
    "SW": (FIRST_FLOOR, ("exit_W_room132",)),
    "SE": (FIRST_FLOOR, ("exit_E_room132",)),
    "BW": (BALCONY, ("balcony_exit_NW",)),
    "BE": (BALCONY, ("balcony_exit_NE",)),
}

# Stairs that get a one-way link.  The stage-side stairs have none because BW/BE end at the door.
STAIR_HEAD_DOORS = {"stair_SW": "balcony_exit_SW", "stair_SE": "balcony_exit_SE"}

# Smaller lattice islands (counted at 0.5 m cells) cannot be reached by anyone, so they are removed.  Most are slivers left
# by cutting out the seats, and the largest (14 cells) is the floor behind orchestra row T, whose
# only way in is an 8 cm gap at the row corners.  The 30-cell pockets between the north and south
# boxes must stay: they are real floor, reached through the ends of the box rows.
MIN_ISLAND_CELLS = 20

# A row end further than this from open floor faces a wall rather than an aisle.
MAX_ROW_LINK_M = 1.83

# Row A's strip is part of the front cross aisle, and cutting it out would leave that aisle one
# cell wide.  Its walkway nodes stay as nodes and the floor around them stays on the lattice.
OPEN_FRONT_ROW = "A"


def _to_metres(geometry):
    return shapely.transform(geometry, lambda coordinates: coordinates * FT)


def _connected_labels(num_nodes: int, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Component label per node, treating every edge as two-way."""
    parent = list(range(num_nodes))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for start, stop in zip(starts.tolist(), stops.tolist()):
        root_start, root_stop = find(start), find(stop)
        if root_start != root_stop:
            parent[root_start] = root_stop
    return np.array([find(node) for node in range(num_nodes)], np.int32)


def _segments(x_start, y_start, x_stop, y_stop):
    coordinates = np.stack([np.column_stack([x_start, y_start]), np.column_stack([x_stop, y_stop])], axis=1)
    return shapely.linestrings(coordinates)


def _open_steps(cell_mask: np.ndarray, floor, center_x: np.ndarray, center_y: np.ndarray):
    """Lattice steps that stay on the floor, as east, south and two diagonal step tables.

    Two neighbouring cell centres can both be walkable with a wall between them: the orchestra
    barrier and foyer wall are thinner than a cell.  So each step is tested as a segment, not
    just by its end points."""
    num_rows, num_columns = cell_mask.shape

    def open_step(row_step: int, column_step: int) -> np.ndarray:
        rows, columns = np.nonzero(cell_mask)
        next_rows, next_columns = rows + row_step, columns + column_step
        inside = (next_rows >= 0) & (next_rows < num_rows) & (next_columns >= 0) & (next_columns < num_columns)
        rows, columns, next_rows, next_columns = rows[inside], columns[inside], next_rows[inside], next_columns[inside]
        both_cells = cell_mask[next_rows, next_columns]
        rows, columns, next_rows, next_columns = rows[both_cells], columns[both_cells], next_rows[both_cells], next_columns[both_cells]
        result = np.zeros(cell_mask.shape, bool)
        if len(rows):
            lines = _segments(center_x[rows, columns], center_y[rows, columns],
                              center_x[next_rows, next_columns], center_y[next_rows, next_columns])
            result[rows, columns] = shapely.covers(floor, lines)
        return result

    east = open_step(0, 1)
    south = open_step(1, 0)
    south_east = open_step(1, 1)
    south_west = open_step(1, -1)

    # No corner cutting: a diagonal needs both cells beside it and all four sides of its square.
    south_east[:-1, :-1] &= (east[:-1, :-1] & south[:-1, 1:] & south[:-1, :-1] & east[1:, :-1])
    south_west[:-1, 1:] &= (south[:-1, 1:] & east[1:, :-1] & east[:-1, :-1] & south[:-1, :-1])
    return east, south, south_east, south_west


def _without_islands(cell_mask: np.ndarray, floor, center_x, center_y) -> np.ndarray:
    east, south, _, _ = _open_steps(cell_mask, floor, center_x, center_y)
    cell_index = np.full(cell_mask.shape, -1, np.int32)
    cell_index[cell_mask] = np.arange(int(cell_mask.sum()))
    rows_east, columns_east = np.nonzero(east)
    rows_south, columns_south = np.nonzero(south)
    starts = np.concatenate([cell_index[rows_east, columns_east], cell_index[rows_south, columns_south]])
    stops = np.concatenate([cell_index[rows_east, columns_east + 1], cell_index[rows_south + 1, columns_south]])
    labels = _connected_labels(int(cell_mask.sum()), starts, stops)
    sizes = np.bincount(labels, minlength=labels.max() + 1)
    kept = cell_mask.copy()
    kept[cell_mask] = sizes[labels] >= MIN_ISLAND_CELLS
    return kept


def _row_link(start_x: float, start_y: float, cell_x: np.ndarray, cell_y: np.ndarray, on_level: np.ndarray,
              walkable_area) -> Optional[Tuple[int, float]]:
    """Nearest lattice cell a row end can walk to in a straight line, or None at a dead end.

    Being near a cell is not enough: the closest one may sit behind the barrier or another row."""
    distances = np.hypot(cell_x - start_x, cell_y - start_y)
    candidates = np.nonzero(on_level & (distances <= MAX_ROW_LINK_M))[0]
    if len(candidates) == 0:
        return None
    candidates = candidates[np.argsort(distances[candidates], kind="stable")]
    lines = _segments(np.full(len(candidates), start_x), np.full(len(candidates), start_y),
                      cell_x[candidates], cell_y[candidates])
    clear = shapely.covers(walkable_area, lines)
    if not clear.any():
        return None
    chosen = candidates[np.argmax(clear)]
    return int(chosen), float(distances[chosen])


@dataclass
class Grid:
    cell: float                 # cell size in metres
    x0: float                   # x of the WEST edge of column 0 (metres)
    y0: float                   # y of the NORTH edge of row 0 (metres)
    walkable: np.ndarray
    exit_id: np.ndarray
    door_id: np.ndarray
    dist: np.ndarray
    spawn: np.ndarray
    seat_section: List[str]
    exits: List[str]
    doors: List[str]
    level: str
    cell_cost: np.ndarray
    cell_node: np.ndarray
    node_kind: np.ndarray
    node_level: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    node_cost: np.ndarray
    node_dist: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    edge_kind: np.ndarray
    edge_length: np.ndarray
    edge_cost: np.ndarray
    edge_corner_a: np.ndarray   # for a diagonal, the two cells it passes between; otherwise -1
    edge_corner_b: np.ndarray
    walkway_nodes: np.ndarray   # walkway node in front of each seat
    seat_row: np.ndarray        # index into row_names per seat
    seat_number: np.ndarray     # position within the row, from model.py (gaps where a seat was thinned)
    row_names: List[str]

    # ------------------------------------------------------------- conversions --
    def xy(self, row, column):
        """Centre of lattice cell (row, column) in metres."""
        return self.x0 + (np.asarray(column) + 0.5) * self.cell, self.y0 - (np.asarray(row) + 0.5) * self.cell

    def ij(self, x, y):
        """Lattice (row, column) containing a point given in metres."""
        return (np.floor((self.y0 - np.asarray(y)) / self.cell).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.cell).astype(int))

    @property
    def shape(self):
        return self.walkable.shape

    @property
    def cell_m(self):
        return self.cell

    @property
    def num_nodes(self):
        return len(self.node_kind)

    # --------------------------------------------------------------- building --
    @classmethod
    def from_model(cls, model: Optional[Model] = None, level: str = "both", cell_m: float = 0.5,
                   margin_m: float = 0.6, include_outdoor: bool = False,
                   cell_ft: Optional[float] = None) -> "Grid":
        if cell_ft is not None:
            raise TypeError("cell_ft was removed so feet and metres cannot be mixed; pass cell_m")
        if level != "both":
            raise ValueError(f"level={level!r} is no longer built: stairs need both levels in one "
                             "coordinate frame, so a Grid always holds both (level='both')")
        building = model or Model()

        walkable_areas = [building.walkable(name) for name in LEVEL_NAMES]
        if include_outdoor:
            walkable_areas[FIRST_FLOOR] = shapely.union_all(
                [walkable_areas[FIRST_FLOOR]] + list(building.outdoor.values())).buffer(0)
        walkable_areas = [_to_metres(area) for area in walkable_areas]

        lattice_floors = []
        for level_index, name in enumerate(LEVEL_NAMES):
            seat_regions = []
            for region in building.row_regions(name):
                seat_regions.append(region["poly"])
                is_open_front = region["section"].startswith("orchestra") and region["row"] == OPEN_FRONT_ROW
                if not is_open_front:
                    seat_regions.append(region["strip"])
            cut_out = _to_metres(shapely.union_all(seat_regions))
            lattice_floors.append(walkable_areas[level_index].difference(cut_out).buffer(0))

        # x = 0 falls on a cell edge so the lattice treats the east and west halves alike.
        bounds = np.array([area.bounds for area in walkable_areas])
        half_columns = int(math.ceil((np.abs(bounds[:, [0, 2]]).max() + margin_m) / cell_m))
        x0 = -half_columns * cell_m
        y0 = math.ceil((bounds[:, 3].max() + margin_m) / cell_m) * cell_m
        num_columns = 2 * half_columns
        num_rows = int(math.ceil((y0 - (bounds[:, 1].min() - margin_m)) / cell_m))
        columns_grid, rows_grid = np.meshgrid(np.arange(num_columns), np.arange(num_rows))
        center_x = x0 + (columns_grid + 0.5) * cell_m
        center_y = y0 - (rows_grid + 0.5) * cell_m
        lattice_shape = (2, num_rows, num_columns)

        def cells_in(polygon) -> np.ndarray:
            return shapely.contains_xy(polygon, center_x.ravel(), center_y.ravel()).reshape(num_rows, num_columns)

        walkable = np.zeros(lattice_shape, bool)
        for level_index, floor in enumerate(lattice_floors):
            walkable[level_index] = _without_islands(cells_in(floor), floor, center_x, center_y)

        # A door is often thinner than a cell, so grow it by half a cell to be sure it catches one.
        door_names = [door.name for door in building.doors]
        door_id = np.full(lattice_shape, -1, np.int16)
        for door_index, door in enumerate(building.doors):
            level_index = LEVEL_NAMES.index(door.level)
            grown = _to_metres(door.poly).buffer(cell_m * 0.5)
            door_id[level_index][cells_in(grown) & walkable[level_index]] = door_index

        exit_names = list(EXIT_DOORS)
        exit_id = np.full(lattice_shape, -1, np.int16)
        for exit_index, (level_index, names) in enumerate(EXIT_DOORS.values()):
            for name in names:
                door = building.doors[door_names.index(name)]
                grown = _to_metres(door.poly).buffer(cell_m * 0.5)
                exit_id[level_index][cells_in(grown) & walkable[level_index]] = exit_index

        cell_cost = np.full(lattice_shape, COST_FLOOR)
        for name, space in building.spaces["first_floor"].items():
            if name.startswith("stair_") or name.endswith(("_stair1", "_stair2")):
                cell_cost[FIRST_FLOOR][cells_in(_to_metres(space))] = COST_STAIRS

        # ---- nodes: lattice cells first, then seats, then walkways
        cell_node = np.full(lattice_shape, -1, np.int32)
        num_cells = int(walkable.sum())
        cell_node[walkable] = np.arange(num_cells)
        cell_levels, cell_rows, cell_columns = np.nonzero(walkable)

        seats, walkways, seat_levels = [], [], []
        for level_index, name in enumerate(LEVEL_NAMES):
            level_seats, level_walkways = building.seats(name), building.spawn_points(name)
            seats += level_seats
            walkways += level_walkways
            seat_levels += [level_index] * len(level_seats)
        num_seats = len(seats)
        seat_nodes = num_cells + np.arange(num_seats, dtype=np.int32)
        walkway_nodes = seat_nodes + num_seats
        seat_x = np.array([seat["x"] for seat in seats]) * FT
        seat_y = np.array([seat["y"] for seat in seats]) * FT
        walkway_x = np.array([walkway["x"] for walkway in walkways]) * FT
        walkway_y = np.array([walkway["y"] for walkway in walkways]) * FT
        seat_levels = np.array(seat_levels, np.int8)

        node_kind = np.concatenate([np.full(num_cells, NODE_CELL, np.int8), np.full(num_seats, NODE_SEAT, np.int8),
                                    np.full(num_seats, NODE_WALKWAY, np.int8)])
        node_level = np.concatenate([cell_levels.astype(np.int8), seat_levels, seat_levels])
        node_x = np.concatenate([center_x[cell_rows, cell_columns], seat_x, walkway_x])
        node_y = np.concatenate([center_y[cell_rows, cell_columns], seat_y, walkway_y])
        node_cost = np.concatenate([cell_cost[walkable], np.full(num_seats, COST_SEAT), np.full(num_seats, COST_FLOOR)])

        # ---- edges
        starts: List[np.ndarray] = []
        stops: List[np.ndarray] = []
        kinds: List[np.ndarray] = []
        lengths: List[np.ndarray] = []
        corners_a: List[np.ndarray] = []
        corners_b: List[np.ndarray] = []

        def add_edges(start, stop, kind, length, corner_a=None, corner_b=None):
            start = np.atleast_1d(start)
            stop = np.atleast_1d(stop)
            starts.append(start)
            stops.append(stop)
            kinds.append(np.full(len(start), kind, np.int8))
            lengths.append(np.broadcast_to(np.asarray(length, float), start.shape))
            no_corner = np.full(len(start), -1, np.int32)
            corners_a.append(no_corner if corner_a is None else corner_a)
            corners_b.append(no_corner if corner_b is None else corner_b)

        def add_two_way(node_a, node_b, kind, length, corner_a=None, corner_b=None):
            add_edges(node_a, node_b, kind, length, corner_a, corner_b)
            add_edges(node_b, node_a, kind, length, corner_a, corner_b)

        for level_index, floor in enumerate(lattice_floors):
            nodes = cell_node[level_index]
            east, south, south_east, south_west = _open_steps(walkable[level_index], floor, center_x, center_y)
            rows, columns = np.nonzero(east)
            add_two_way(nodes[rows, columns], nodes[rows, columns + 1], EDGE_LATTICE, cell_m)
            rows, columns = np.nonzero(south)
            add_two_way(nodes[rows, columns], nodes[rows + 1, columns], EDGE_LATTICE, cell_m)
            rows, columns = np.nonzero(south_east)
            add_two_way(nodes[rows, columns], nodes[rows + 1, columns + 1], EDGE_LATTICE, cell_m * math.sqrt(2),
                        nodes[rows, columns + 1], nodes[rows + 1, columns])
            rows, columns = np.nonzero(south_west)
            add_two_way(nodes[rows, columns], nodes[rows + 1, columns - 1], EDGE_LATTICE, cell_m * math.sqrt(2),
                        nodes[rows, columns - 1], nodes[rows + 1, columns])

        # Seats step out to the walkway but not back: the walkway is always the cheaper way along
        # a row, so an edge back into a seat would add nothing to the graph.
        seat_to_walkway = np.hypot(seat_x - walkway_x, seat_y - walkway_y)
        add_edges(seat_nodes, walkway_nodes, EDGE_SEAT_WALKWAY, seat_to_walkway)

        row_index_of: Dict[Tuple[str, str], int] = {}
        members: List[List[int]] = []
        for seat_index, seat in enumerate(seats):
            row_key = (seat["section"], seat["row"])
            if row_key not in row_index_of:
                row_index_of[row_key] = len(members)
                members.append([])
            members[row_index_of[row_key]].append(seat_index)

        for row_members in members:
            row_members.sort(key=lambda seat_index: seats[seat_index]["seat"])
            ordered = np.array(row_members)
            neighbours_a, neighbours_b = ordered[:-1], ordered[1:]
            if len(neighbours_a):
                add_two_way(seat_nodes[neighbours_a], seat_nodes[neighbours_b], EDGE_SEAT_SEAT,
                            np.hypot(seat_x[neighbours_a] - seat_x[neighbours_b], seat_y[neighbours_a] - seat_y[neighbours_b]))
                add_two_way(walkway_nodes[neighbours_a], walkway_nodes[neighbours_b], EDGE_WALKWAY_WALKWAY,
                            np.hypot(walkway_x[neighbours_a] - walkway_x[neighbours_b], walkway_y[neighbours_a] - walkway_y[neighbours_b]))

            for end_seat in {int(ordered[0]), int(ordered[-1])}:
                level_index = int(seat_levels[end_seat])
                link = _row_link(walkway_x[end_seat], walkway_y[end_seat], node_x[:num_cells],
                                 node_y[:num_cells], cell_levels == level_index, walkable_areas[level_index])
                if link is not None:
                    add_two_way(walkway_nodes[end_seat], link[0], EDGE_ROW_LINK, link[1])

        first_floor_cells = np.nonzero(cell_levels == FIRST_FLOOR)[0]
        for stair in building.stairs:
            if stair.name not in STAIR_HEAD_DOORS:
                continue
            head_door = door_names.index(STAIR_HEAD_DOORS[stair.name])
            head_cells = cell_node[BALCONY][door_id[BALCONY] == head_door]
            landing_x, landing_y = stair.p_to[0] * FT, stair.p_to[1] * FT
            nearest = np.argmin(np.hypot(node_x[first_floor_cells] - landing_x, node_y[first_floor_cells] - landing_y))
            landing_cells = np.full(len(head_cells), first_floor_cells[nearest], np.int32)
            add_edges(head_cells, landing_cells, EDGE_STAIR, stair.length * FT)

        edge_from = np.concatenate(starts).astype(np.int32)
        edge_to = np.concatenate(stops).astype(np.int32)
        edge_kind = np.concatenate(kinds)
        edge_length = np.concatenate(lengths)
        edge_cost = edge_length * node_cost[edge_to]
        # A stair's cost belongs to the whole flight, not to the floor cell it lands on.
        is_stair = edge_kind == EDGE_STAIR
        edge_cost[is_stair] = edge_length[is_stair] * COST_STAIRS

        row_names = [f"{section}/{row}" for section, row in row_index_of]
        seat_row = np.array([row_index_of[(seat["section"], seat["row"])] for seat in seats], np.int32)
        seat_number = np.array([seat["seat"] for seat in seats], np.int32)

        grid = cls(cell_m, x0, y0, walkable, exit_id, door_id, np.full(lattice_shape, np.inf),
                   seat_nodes, [seat["section"] for seat in seats], exit_names, door_names, "both",
                   cell_cost, cell_node, node_kind, node_level, node_x, node_y, node_cost,
                   np.full(len(node_kind), np.inf), edge_from, edge_to, edge_kind, edge_length, edge_cost,
                   np.concatenate(corners_a).astype(np.int32), np.concatenate(corners_b).astype(np.int32),
                   walkway_nodes, seat_row, seat_number, row_names)
        grid.node_dist = grid.node_distance_field()
        grid.dist = grid._lattice_values(grid.node_dist)
        return grid

    # ---------------------------------------------------------------- masks --
    def exit_mask(self, labels: Union[str, Iterable[str], None] = None) -> np.ndarray:
        """Lattice mask of the named exits, or of all exits.  Use it as a target or to block a gate."""
        if labels is None:
            return self.exit_id >= 0
        names = [labels] if isinstance(labels, str) else list(labels)
        return np.isin(self.exit_id, [self.exits.index(name) for name in names])

    def door_mask(self, names: Union[str, Iterable[str]]) -> np.ndarray:
        """Lattice mask of the named model doors, for closing an interior door."""
        names = [names] if isinstance(names, str) else list(names)
        return np.isin(self.door_id, [self.doors.index(name) for name in names])

    def _node_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, bool)
        if mask.shape == (self.num_nodes,):
            return mask
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} matches neither the lattice {self.shape} "
                             f"nor the node count ({self.num_nodes},)")
        node_mask = np.zeros(self.num_nodes, bool)
        node_mask[self.cell_node[self.walkable & mask]] = True
        return node_mask

    def _lattice_values(self, node_values: np.ndarray) -> np.ndarray:
        values = np.full(self.shape, np.inf)
        values[self.walkable] = node_values[self.cell_node[self.walkable]]
        return values

    # ------------------------------------------------------------- algorithms --
    def node_distance_field(self, targets: Optional[np.ndarray] = None,
                            blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """Cheapest cost from every node to the nearest target; inf where unreachable.

        `targets` and `blocked` are lattice masks (2, H, W) or node masks (nodes,).  Targets
        default to all exits.  Edges are directed, so the search runs backwards from the targets
        along incoming edges: searching forwards would let people climb the one-way stairs."""
        target_nodes = self._node_mask(self.exit_mask() if targets is None else targets)
        blocked_nodes = np.zeros(self.num_nodes, bool) if blocked is None else self._node_mask(blocked)
        is_blocked = blocked_nodes.tolist()

        order = np.argsort(self.edge_to, kind="stable")
        offsets = np.searchsorted(self.edge_to[order], np.arange(self.num_nodes + 1)).tolist()
        edge_starts = self.edge_from[order].tolist()
        edge_costs = self.edge_cost[order].tolist()
        corners_a = self.edge_corner_a[order].tolist()
        corners_b = self.edge_corner_b[order].tolist()

        best = [math.inf] * self.num_nodes
        heap = []
        for node in np.nonzero(target_nodes & ~blocked_nodes)[0].tolist():
            best[node] = 0.0
            heap.append((0.0, node))
        heapq.heapify(heap)
        while heap:
            cost, node = heapq.heappop(heap)
            if cost > best[node]:
                continue
            for edge in range(offsets[node], offsets[node + 1]):
                previous = edge_starts[edge]
                if is_blocked[previous]:
                    continue
                # A closed cell must also stop diagonals from slipping past its corner.
                corner_a, corner_b = corners_a[edge], corners_b[edge]
                if corner_a >= 0 and (is_blocked[corner_a] or is_blocked[corner_b]):
                    continue
                new_cost = cost + edge_costs[edge]
                if new_cost < best[previous]:
                    best[previous] = new_cost
                    heapq.heappush(heap, (new_cost, previous))
        return np.array(best)

    def distance_field(self, targets: Optional[np.ndarray] = None,
                       blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """Lattice view of node_distance_field, shape (2, H, W).  Seats and walkways are in
        node_distance_field.  Pass `blocked` to close a gate without rebuilding the grid."""
        return self._lattice_values(self.node_distance_field(targets, blocked))

    def components(self) -> np.ndarray:
        """Connected-component label per node, ignoring edge direction."""
        return _connected_labels(self.num_nodes, self.edge_from, self.edge_to)

    # ---------------------------------------------------------------- io / viz --
    def save(self, path: str):
        np.savez_compressed(
            path, format_version=FORMAT_VERSION, cell=self.cell, x0=self.x0, y0=self.y0,
            level=self.level, seat_section=np.array(self.seat_section), exits=np.array(self.exits),
            doors=np.array(self.doors), row_names=np.array(self.row_names),
            **{name: getattr(self, name) for name in _ARRAY_FIELDS})

    @classmethod
    def load(cls, path: str) -> "Grid":
        with np.load(path, allow_pickle=False) as archive:
            if "format_version" not in archive.files or int(archive["format_version"]) != FORMAT_VERSION:
                raise ValueError(f"{path} was written by an older grid.py (lengths in feet, no graph); "
                                 "rebuild it with Grid.from_model")
            return cls(cell=float(archive["cell"]), x0=float(archive["x0"]), y0=float(archive["y0"]),
                       level=str(archive["level"]), seat_section=archive["seat_section"].tolist(),
                       exits=archive["exits"].tolist(), doors=archive["doors"].tolist(),
                       row_names=archive["row_names"].tolist(),
                       **{name: archive[name] for name in _ARRAY_FIELDS})

    def render(self, path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, num_rows, num_columns = self.shape
        extent = (self.x0, self.x0 + num_columns * self.cell, self.y0 - num_rows * self.cell, self.y0)
        finite_costs = self.node_dist[np.isfinite(self.node_dist)]
        color_scale = dict(cmap="Blues_r", vmin=0.0, vmax=float(finite_costs.max()))
        figure, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=140, constrained_layout=True)
        image = None
        for level_index, axis in enumerate(axes):
            image = axis.imshow(np.where(self.walkable[level_index], self.dist[level_index], np.nan),
                                extent=extent, interpolation="nearest", **color_scale)
            on_level = self.node_level == level_index
            walkway = on_level & (self.node_kind == NODE_WALKWAY)
            seat = on_level & (self.node_kind == NODE_SEAT)
            axis.scatter(self.node_x[walkway], self.node_y[walkway], c=self.node_dist[walkway], s=5,
                         linewidths=0, **color_scale)
            axis.scatter(self.node_x[seat], self.node_y[seat], c="#8a8a8a", s=2.5, linewidths=0)

            is_stair = (self.edge_kind == EDGE_STAIR) & (self.node_level[self.edge_from] == level_index)
            axis.scatter(self.node_x[self.edge_from[is_stair]], self.node_y[self.edge_from[is_stair]],
                         marker="v", s=14, c="#111111", linewidths=0)
            for exit_index, label in enumerate(self.exits):
                rows, columns = np.nonzero(self.exit_id[level_index] == exit_index)
                if len(rows) == 0:
                    continue
                x_values, y_values = self.xy(rows, columns)
                axis.scatter(x_values, y_values, marker="s", s=9, c="#d1495b", linewidths=0)
                # A portico label covers two separate doors, so split the cells at any gap in y.
                by_y = np.argsort(y_values)
                door_groups = np.split(by_y, np.nonzero(np.diff(y_values[by_y]) > 2 * self.cell)[0] + 1)
                for group in door_groups:
                    axis.annotate(label, (x_values[group].mean(), y_values[group].mean()), xytext=(0, -15),
                                  textcoords="offset points", ha="center", fontsize=8, color="#222222")
            axis.set_title(f"{LEVEL_NAMES[level_index].replace('_', ' ')}: "
                           f"{int(self.walkable[level_index].sum())} cells, {int(seat.sum())} seats", fontsize=11)
            axis.set_xlabel("x east [m]")
            axis.set_ylabel("y north [m]")
            axis.set_aspect("equal")
            axis.set_facecolor("#f4f2ee")
            axis.tick_params(length=0, labelsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        figure.colorbar(image, ax=list(axes), shrink=0.7, label="weighted cost to the nearest exit [m]")
        figure.suptitle(f"Hybrid grid: {self.cell:g} m lattice with seat and walkway nodes. "
                        "Red squares are exits, black triangles are one-way stair heads.", fontsize=12)
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)

    def render_layout(self, path: str):
        """Plot only the three tile types: walls, walkable tiles and seats, with no distance shading."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        wall_color, walkable_color, seat_color = "#3a3f47", "#ece8df", "#d1495b"
        _, num_rows, num_columns = self.shape
        extent = (self.x0, self.x0 + num_columns * self.cell, self.y0 - num_rows * self.cell, self.y0)
        figure, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=140, constrained_layout=True)
        for level_index, axis in enumerate(axes):
            axis.imshow(np.where(self.walkable[level_index], 1.0, np.nan), extent=extent, interpolation="nearest",
                        cmap=ListedColormap([walkable_color]))
            on_level = self.node_level == level_index
            walkway = on_level & (self.node_kind == NODE_WALKWAY)
            seat = on_level & (self.node_kind == NODE_SEAT)
            # Row walkways are not lattice cells, so they are drawn as points in the walkable colour.
            axis.scatter(self.node_x[walkway], self.node_y[walkway], c=walkable_color, s=9, marker="s", linewidths=0)
            axis.scatter(self.node_x[seat], self.node_y[seat], c=seat_color, s=7, marker="s", linewidths=0)
            axis.set_title(f"{LEVEL_NAMES[level_index].replace('_', ' ')}: "
                           f"{int(self.walkable[level_index].sum())} cells, {int(seat.sum())} seats", fontsize=11)
            axis.set_xlabel("x east [m]")
            axis.set_ylabel("y north [m]")
            axis.set_aspect("equal")
            axis.set_facecolor(wall_color)
            axis.tick_params(length=0, labelsize=8)
            for spine in axis.spines.values():
                spine.set_visible(False)
        legend = [Patch(facecolor=wall_color, label="Wall or not walkable"),
                  Patch(facecolor=walkable_color, edgecolor="#999999", label="Walkable"),
                  Line2D([], [], marker="s", linestyle="", color=seat_color, label="Seat")]
        figure.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False, fontsize=10)
        figure.suptitle(f"Hybrid grid layout at {self.cell:g} m", fontsize=12)
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)


_ARRAY_FIELDS = ("walkable", "exit_id", "door_id", "dist", "spawn", "cell_cost", "cell_node", "node_kind",
                 "node_level", "node_x", "node_y", "node_cost", "node_dist", "edge_from", "edge_to", "edge_kind",
                 "edge_length", "edge_cost", "edge_corner_a", "edge_corner_b", "walkway_nodes", "seat_row",
                 "seat_number")


def validate(grid: Grid) -> List[str]:
    """Print the structural report and return a list of failures (empty = valid)."""
    failures: List[str] = []

    def check(passed: bool, description: str):
        print(f"  [{'ok' if passed else 'FAIL'}] {description}")
        if not passed:
            failures.append(description)

    print("seats")
    seat_mask = grid.node_kind == NODE_SEAT
    walkway_mask = grid.node_kind == NODE_WALKWAY
    seat_positions = {(int(level), round(float(x), 4), round(float(y), 4)) for level, x, y in
                      zip(grid.node_level[seat_mask], grid.node_x[seat_mask], grid.node_y[seat_mask])}
    check(len(np.unique(grid.spawn)) == 1329 and len(seat_positions) == 1329,
          f"{len(np.unique(grid.spawn))} seat nodes at {len(seat_positions)} distinct positions (want 1329)")
    section_counts = {group: 0 for group in Model.OFFICIAL_SEATS}
    for section in grid.seat_section:
        section_counts[Model.section_group(section)] += 1
    check(section_counts == Model.OFFICIAL_SEATS, f"section counts {section_counts}")
    check(int(walkway_mask.sum()) == int(seat_mask.sum()), f"{int(walkway_mask.sum())} walkway nodes, one per seat")

    print("nodes and edges")
    node_keys = np.column_stack([grid.node_kind, grid.node_level, np.round(grid.node_x, 4), np.round(grid.node_y, 4)])
    check(len(np.unique(node_keys, axis=0)) == grid.num_nodes, f"{grid.num_nodes} nodes, no duplicates")
    edge_keys = np.column_stack([grid.edge_from, grid.edge_to])
    check(len(np.unique(edge_keys, axis=0)) == len(edge_keys), f"{len(edge_keys)} directed edges, no duplicates")
    check(bool((grid.edge_from != grid.edge_to).all()), "no self-loops")
    for kind, name in enumerate(EDGE_KIND_NAMES):
        chosen = grid.edge_kind == kind
        print(f"       {name:16s}{int(chosen.sum()):7d} edges, length {grid.edge_length[chosen].min():.3f}"
              f"-{grid.edge_length[chosen].max():.3f} m")

    lattice_lengths = grid.edge_length[grid.edge_kind == EDGE_LATTICE]
    straight = np.isclose(lattice_lengths, grid.cell)
    diagonal = np.isclose(lattice_lengths, grid.cell * math.sqrt(2))
    check(bool((straight | diagonal).all()), f"lattice steps are {grid.cell:g} m or {grid.cell * math.sqrt(2):.3f} m")
    not_stairs = grid.edge_kind != EDGE_STAIR
    check(bool(np.allclose(grid.edge_cost[not_stairs],
                           grid.edge_length[not_stairs] * grid.node_cost[grid.edge_to[not_stairs]])),
          "edge cost = length x entry cost of the destination")
    stairs = ~not_stairs
    check(bool((grid.node_level[grid.edge_from[stairs]] == BALCONY).all()
               and (grid.node_level[grid.edge_to[stairs]] == FIRST_FLOOR).all()),
          f"{int(stairs.sum())} stair links, all one-way down (cost {grid.edge_cost[stairs].max():.2f})")
    crossing = grid.node_level[grid.edge_from] != grid.node_level[grid.edge_to]
    check(bool((grid.edge_kind[crossing] == EDGE_STAIR).all()), "only stair links change level")

    print("exits")
    check(len(grid.exits) == 11, f"{len(grid.exits)} exit labels")
    for exit_index, label in enumerate(grid.exits):
        cells = grid.exit_id == exit_index
        check(bool(cells.any()), f"{label}: {int(cells.sum())} cells")

    print("reachability")
    reachable = np.isfinite(grid.node_dist)
    unreachable = np.nonzero(~reachable)[0]
    for kind, name in enumerate(NODE_KIND_NAMES):
        for level, level_name in enumerate(LEVEL_NAMES):
            chosen = (grid.node_kind == kind) & (grid.node_level == level)
            print(f"       {name:8s} {level_name:12s}{int(chosen.sum()):6d} nodes, "
                  f"{int((chosen & ~reachable).sum())} unreachable")
    check(len(unreachable) == 0, f"{grid.num_nodes - len(unreachable)} of {grid.num_nodes} nodes reach an exit")
    for node in unreachable[:20].tolist():
        print(f"       unreachable: {NODE_KIND_NAMES[grid.node_kind[node]]} on {LEVEL_NAMES[grid.node_level[node]]} "
              f"at ({grid.node_x[node]:.2f}, {grid.node_y[node]:.2f}) m")
    seat_costs = grid.node_dist[grid.spawn]
    print(f"       seat cost to the nearest exit: median {np.median(seat_costs):.1f}, worst {seat_costs.max():.1f}")

    print("closing gates without a rebuild")
    for label in grid.exits:
        field = grid.node_distance_field(blocked=grid.exit_mask(label))
        stranded = int((~np.isfinite(field[grid.spawn])).sum())
        check(stranded == 0, f"{label} closed: {stranded} seats stranded")

    return failures


if __name__ == "__main__":
    cell_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    built_grid = Grid.from_model(cell_m=cell_size)
    print(f"lattice {built_grid.shape[2]}x{built_grid.shape[1]} at {cell_size:g} m on 2 levels, "
          f"{built_grid.num_nodes} nodes, {len(built_grid.edge_from)} directed edges, {len(built_grid.exits)} exits")
    problems = validate(built_grid)
    output_stem = os.path.join(HERE, "out", f"grid_hybrid_{cell_size:g}m")
    built_grid.save(output_stem + ".npz")
    built_grid.render(output_stem + ".png")
    built_grid.render_layout(output_stem + "_layout.png")
    print(output_stem + ".npz", output_stem + ".png", output_stem + "_layout.png")
    if problems:
        print(f"{len(problems)} validation failure(s):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print("all checks passed")
