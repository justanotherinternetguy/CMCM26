from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from animate import RUN_PATH, save_run
from grid import BALCONY, COST_STAIRS, FIRST_FLOOR, NODE_CELL, NODE_SEAT, Grid
from grid_paint import GridPaint
from model import FT, Model

CELL_M = 0.5
MAX_WALKING_SPEED_M_PER_S = 1.4
RANDOM_SEED = 42067
MAX_STEPS = 5000

# Alarm-to-first-movement delay, including immediate seat-neighbor influence.
RESPONSE_TIME_ENABLED = True
RESPONSE_LOG_MEAN = 3.93
RESPONSE_LOG_STD = 0.44
RESPONSE_SHUFFLE_FRACTION = 0.20
SOCIAL_RESPONSE_ENABLED = True
SOCIAL_REACTION_DELAY_S = 5.0

NORTH_STAIR_EXIT_LABELS = ("BW", "BE")
SIDE_EXIT_LABELS = ("PW", "PE")
INCLUDE_NORTH_STAIR_COST = True

NUM_DIRECTIONS = 8
SOCIAL_RADIUS_M = 2.0
HERD_STRENGTH = 2
REEVALUATE_EVERY_STEPS = 30
REEVALUATE_ON_LEAVING_AUDITORIUM = True   # also choose an exit the moment one steps out of the auditorium
TARGET_CHOICE_SHARPNESS = 1
VIEW_RANGE_M = 15.0

PAINT_PATH = f"out/grid_paint_{CELL_M:g}m.npz"
BARRIERS_ENABLED = True

CLOSED_EXIT_LABELS = ("F2", "F3")
DISCOVERY_RADIUS_M = 4.0          #distance at which an agent notices a closed exit
LED_KNOWS_CLOSURES = False

LED_ENABLED = True
LED_STRENGTH = 10.0
LED_TRUST_DECAY = 0.08    # trust = exp(-decay * (independence - 1))
LED_RELAX_WEIGHT = 0.5
LED_VIEW_SCALE_M = 4.0
LED_MAX_RANGE_M = 15.0
LED_INDEPENDENCE_GAIN = 1.7     # LED independence is multiplied by this for every closed exit the agent has noticed

# crowd managers (cms) stand at fixed posts. Someone within CROWD_MANAGER_RADIUS_M of a manager
# may be told to use that manager's exit instead of their own.


# The chance is a rate per second: it is
# CROWD_MANAGER_MAX_SWITCH_RATE_PER_S right next to the manager and falls linearly to 0 at the radius,
# so the longer someone stays close (e.g. stuck in a queue), the more likely they are to be redirected.
CROWD_MANAGER_RADIUS_M = 4.0
CROWD_MANAGER_MAX_SWITCH_RATE_PER_S = 1.0
ATTENDEES_PER_CROWD_MANAGER = 250    # NY State Fire Code: at least one manager per 250 attendees


@dataclass
class CrowdManagerPost:
    name: str
    level: int # FIRST_FLOOR/BALCONY
    x_m: float
    y_m: float
    target_exit_label: str # target exit that this manager will send everyone tos

# placement of crowd managers
CROWD_MANAGER_POSTS = (
    CrowdManagerPost("orchestra front W", FIRST_FLOOR, -8.0, 7.5, "SW"),
    CrowdManagerPost("orchestra front E", FIRST_FLOOR, 8.0, 7.5, "SE"),
    CrowdManagerPost("corridor W", FIRST_FLOOR, -17.0, -4.0, "PW"),
    CrowdManagerPost("corridor E", FIRST_FLOOR, 17.0, -4.0, "PE"),
    CrowdManagerPost("balcony W", BALCONY, -17.5, -2.0, "BW"),
    CrowdManagerPost("balcony E", BALCONY, 17.5, -2.0, "BE"),
)

MAIN_ENTRANCE_LABELS = ("F1", "F2", "F3", "F4", "F5")

KNOWN_AUDITORIUM_DOOR_NAMES = (
    "loge_door_W_aisle", "loge_door_E_aisle", "loge_door_centre", "loge_door_W_side", "loge_door_E_side",
    "balcony_exit_SW", "balcony_exit_SE", "balcony_exit_NW", "balcony_exit_NE",
)


@dataclass
class AudienceProfile:
    name: str
    speed_fraction_mean: float
    speed_fraction_std: float
    speed_fraction_min: float
    speed_fraction_max: float
    independence_mean: float
    independence_std: float
    independence_min: float
    knows_all_exits: bool = False
    has_crowd_managers: bool = False


CONCERT_PROFILE = AudienceProfile(
    "concert",
    speed_fraction_mean=0.8, speed_fraction_std=0.15, speed_fraction_min=0.3, speed_fraction_max=1.0,
    independence_mean=3.0, independence_std=1.0, independence_min=1.0,
    knows_all_exits=False, has_crowd_managers=True)

STUDENT_PROFILE = AudienceProfile(
    "students",
    speed_fraction_mean=0.8, speed_fraction_std=0.15, speed_fraction_min=0.3, speed_fraction_max=1.0,
    independence_mean=4.5, independence_std=1.0, independence_min=1.0,
    knows_all_exits=True, has_crowd_managers=False)

PROFILE = CONCERT_PROFILE


@dataclass
class DistanceFields:
    exit_labels: List[str]              # ALL exits; same row order in every matrix
    closed_exit_labels: Tuple[str, ...]
    is_closed: np.ndarray               # (num_exits,) bool
    cost_by_exit: np.ndarray            # BELIEVED: only painted barriers block
    best_cost: np.ndarray               # believed cost to the nearest exit (closed ones included)
    best_exit_index: np.ndarray
    blocked_nodes: np.ndarray           # painted barriers only (visible to everyone)
    closed_nodes: np.ndarray            # nodes of closed exits (physically impassable)
    true_cost_by_exit: np.ndarray       # TRUE: closed exits blocked; their rows are all inf
    true_best_cost: np.ndarray          # true cost to the nearest open exit

    def best_exit_label(self, node_id: int) -> str:
        return self.exit_labels[int(self.best_exit_index[node_id])]

def north_stair_extra_cost() -> float:
    north_stair_length_m = Model().p.stair_nw_length * FT
    return north_stair_length_m * COST_STAIRS

def mask_to_node_mask(grid: Grid, mask) -> np.ndarray:
    """Per-node boolean array for a per-cell mask, using the grid's own target handling."""
    if np.asarray(mask).ndim == 1:           # already per node
        return np.asarray(mask, bool)
    return grid.node_distance_field(targets=mask, blocked=None) == 0.0

def compute_distance_fields(grid: Grid, barrier_nodes: np.ndarray,
                            closed_exit_labels: Sequence[str] = ()) -> DistanceFields:
    closed_exit_labels = tuple(closed_exit_labels)
    exit_labels = list(grid.exits)
    unknown = set(closed_exit_labels) - set(exit_labels)
    if unknown:
        raise ValueError(f"unknown exit labels: {sorted(unknown)}; valid: {exit_labels}")
    is_closed = np.array([label in closed_exit_labels for label in exit_labels], dtype=bool)
    closed_nodes = (mask_to_node_mask(grid, grid.exit_mask(list(closed_exit_labels)))
                    if closed_exit_labels else np.zeros(grid.num_nodes, bool))
    if closed_exit_labels and not closed_nodes.any():
        raise ValueError(f"closed exits {closed_exit_labels} map to no nodes; check the cell-to-node mapping")
    extra_cost_for_north_stairs = north_stair_extra_cost() if INCLUDE_NORTH_STAIR_COST else 0.0

    def cost_matrix(blocked_nodes: np.ndarray, skip_closed: bool) -> np.ndarray:
        blocked_or_none = blocked_nodes if blocked_nodes.any() else None
        rows = []
        for label, closed in zip(exit_labels, is_closed):
            if skip_closed and closed:
                rows.append(np.full(grid.num_nodes, np.inf))
                continue
            cost = grid.node_distance_field(targets=grid.exit_mask(label), blocked=blocked_or_none)
            if label in NORTH_STAIR_EXIT_LABELS:
                cost = cost + extra_cost_for_north_stairs
            rows.append(cost)
        return np.vstack(rows)

    believed = cost_matrix(barrier_nodes, skip_closed=False)
    true = cost_matrix(barrier_nodes | closed_nodes, skip_closed=True)
    best_exit_index = believed.argmin(axis=0)
    best_cost = believed[best_exit_index, np.arange(grid.num_nodes)]
    return DistanceFields(exit_labels, closed_exit_labels, is_closed, believed, best_cost,
                          best_exit_index, barrier_nodes.copy(), closed_nodes, true, true.min(axis=0))


def compute_discovery_distance(grid: Grid, fields: DistanceFields, barrier_nodes: np.ndarray) -> np.ndarray:
    rows = np.full((len(fields.exit_labels), grid.num_nodes), np.inf)
    blocked_or_none = barrier_nodes if barrier_nodes.any() else None
    for row in np.flatnonzero(fields.is_closed):
        distance = grid.node_distance_field(targets=grid.exit_mask(fields.exit_labels[row]),
                                            blocked=blocked_or_none)
        distance[distance > DISCOVERY_RADIUS_M] = np.inf
        rows[row] = distance
    return rows

def compute_door_costs(grid: Grid, blocked_nodes: np.ndarray) -> np.ndarray:
    blocked_or_none = blocked_nodes if blocked_nodes.any() else None
    return np.vstack([grid.node_distance_field(targets=grid.door_mask(door_name), blocked=blocked_or_none)
                      for door_name in KNOWN_AUDITORIUM_DOOR_NAMES])


def compute_door_to_exit_costs(door_costs: np.ndarray, exit_costs: np.ndarray) -> np.ndarray:
    door_to_exit = np.full((door_costs.shape[0], exit_costs.shape[0]), np.inf)
    for door_row in range(door_costs.shape[0]):
        door_nodes = np.flatnonzero(door_costs[door_row] == 0.0)
        if door_nodes.size:
            door_to_exit[door_row] = exit_costs[:, door_nodes].min(axis=1)
    return door_to_exit


def compute_exit_goal_costs(exit_costs: np.ndarray) -> np.ndarray:
    return exit_costs.min(axis=1)


@dataclass
class LedGuidance:
    enabled: bool
    strength: float
    lit_nodes: np.ndarray             # True for every node touched by a painted LED
    cost_to_nearest_exit: np.ndarray  # what the strips point down: cost to the closest exit from each node
    visibility: np.ndarray
    # Optional: the exit the nearest strip points to from each node (row in exit_costs), -1 = nearest exit.
    target_exit_row_at_node: Optional[np.ndarray] = None
    exit_costs: Optional[np.ndarray] = None

    def cost_field_at(self, node: int) -> np.ndarray:
        """The cost field the LED strip seen from `node` points down."""
        if self.target_exit_row_at_node is not None:
            row = self.target_exit_row_at_node[node]
            if row >= 0:
                return self.exit_costs[row]
        return self.cost_to_nearest_exit


def painted_value_at(painted: np.ndarray, level_index: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    num_rows, num_cols = painted.shape[-2:]
    inside = (row >= 0) & (row < num_rows) & (col >= 0) & (col < num_cols)
    values = np.zeros(len(row), bool)
    if painted.ndim == 3:
        inside &= level_index < painted.shape[0]
        values[inside] = painted[level_index[inside], row[inside], col[inside]]
    else:
        values[inside] = painted[row[inside], col[inside]]
    return values


def node_is_lit(paint: GridPaint, grid: Grid) -> np.ndarray:
    """Nodes whose cell has an LED, or has an LED on any of its four walls."""
    if hasattr(paint, "node_has_led"):
        return np.asarray(paint.node_has_led(grid), bool)

    cell_row = np.floor(grid.node_y / grid.cell_m + 1e-9).astype(int)
    cell_col = np.floor(grid.node_x / grid.cell_m + 1e-9).astype(int)
    level_index = np.searchsorted(np.unique(grid.node_level), grid.node_level)

    lit = painted_value_at(paint.cell_has_led, level_index, cell_row, cell_col)
    lit |= painted_value_at(paint.wall_has_led_horizontal, level_index, cell_row, cell_col)
    lit |= painted_value_at(paint.wall_has_led_horizontal, level_index, cell_row + 1, cell_col)
    lit |= painted_value_at(paint.wall_has_led_vertical, level_index, cell_row, cell_col)
    lit |= painted_value_at(paint.wall_has_led_vertical, level_index, cell_row, cell_col + 1)
    return lit


def build_led_guidance(grid: Grid, paint: Optional[GridPaint], fields: DistanceFields,
                       enabled: bool = LED_ENABLED, strength: float = LED_STRENGTH,
                       lit_nodes: Optional[np.ndarray] = None,
                       target_exit_row_at_node: Optional[np.ndarray] = None) -> LedGuidance:
    """LEDs from the painted layout, or from `lit_nodes` (+ optional per-node target exit) if given.
    A node off the strips follows the target of the nearest lit node."""
    led_exit_costs = fields.true_cost_by_exit if LED_KNOWS_CLOSURES else fields.cost_by_exit
    led_cost = fields.true_best_cost if LED_KNOWS_CLOSURES else fields.best_cost
    if not enabled:
        return LedGuidance(False, strength, np.zeros(grid.num_nodes, bool), led_cost,
                           np.zeros(grid.num_nodes))
    if lit_nodes is None:
        lit_nodes = node_is_lit(paint, grid) if paint is not None else np.zeros(grid.num_nodes, bool)
    lit_nodes = np.asarray(lit_nodes, bool) & ~fields.blocked_nodes
    blocked_or_none = fields.blocked_nodes if fields.blocked_nodes.any() else None

    distance_to_lit = np.full(grid.num_nodes, np.inf)
    target_by_nearest = None
    if lit_nodes.any():
        if target_exit_row_at_node is None:
            distance_to_lit = grid.node_distance_field(targets=lit_nodes, blocked=blocked_or_none)
        else:
            # One search per distinct target, so nodes near a strip take that strip's exit.
            target_by_nearest = np.full(grid.num_nodes, -1, np.int32)
            for row in np.unique(target_exit_row_at_node[lit_nodes]):
                group = lit_nodes & (target_exit_row_at_node == row)
                distance = grid.node_distance_field(targets=group, blocked=blocked_or_none)
                closer = distance < distance_to_lit
                distance_to_lit[closer] = distance[closer]
                target_by_nearest[closer] = row
    visibility = np.where(distance_to_lit <= LED_MAX_RANGE_M,
                          np.exp(-distance_to_lit / LED_VIEW_SCALE_M), 0.0)
    return LedGuidance(True, strength, lit_nodes, led_cost, visibility, target_by_nearest, led_exit_costs)

@dataclass
class CrowdManagers:
    names: List[str]
    node: np.ndarray                # node each manager stands on (they do not block anyone)
    target_exit_row: np.ndarray
    target_exit_label: np.ndarray
    distance_to_manager: np.ndarray
    radius_m: float
    max_switch_rate_per_s: float

    @property
    def num_managers(self) -> int:
        return len(self.node)


def place_crowd_managers(grid: Grid, fields: DistanceFields, posts: Sequence[CrowdManagerPost],
                         cache: Optional[dict] = None) -> CrowdManagers:
    """`cache` (any dict, kept between calls) saves the per-post distance search when the same
    posts are placed again, which the optimizer does thousands of times."""
    names, nodes, exit_rows, distance_rows = [], [], [], []
    blocked = fields.blocked_nodes if fields.blocked_nodes.any() else None
    for post in posts:
        key = (post.level, post.x_m, post.y_m, post.target_exit_label, CROWD_MANAGER_RADIUS_M)
        if cache is not None and key in cache:
            node, exit_row, distance = cache[key]
        else:
            post_row = fields.exit_labels.index(post.target_exit_label)
            candidates = np.flatnonzero((grid.node_kind == NODE_CELL) & (grid.node_level == post.level)
                                        & ~fields.blocked_nodes & ~fields.closed_nodes
                                        & np.isfinite(fields.cost_by_exit[post_row]))
            node = int(candidates[np.argmin(np.hypot(grid.node_x[candidates] - post.x_m,
                                                     grid.node_y[candidates] - post.y_m))])
            exit_row = post_row
            if fields.is_closed[post_row] or not np.isfinite(fields.true_cost_by_exit[post_row, node]):
                exit_row = int(np.argmin(fields.true_cost_by_exit[:, node]))   # managers know every closure
            at_manager = np.zeros(grid.num_nodes, bool)
            at_manager[node] = True
            distance = grid.node_distance_field(targets=at_manager, blocked=blocked)
            distance[distance > CROWD_MANAGER_RADIUS_M] = np.inf
            if cache is not None:
                cache[key] = (node, exit_row, distance)
        names.append(post.name)
        nodes.append(node)
        exit_rows.append(exit_row)
        distance_rows.append(distance)
    distance_to_manager = np.vstack(distance_rows) if distance_rows else np.empty((0, grid.num_nodes))
    exit_rows = np.array(exit_rows, dtype=int)
    return CrowdManagers(names, np.array(nodes, dtype=np.int32), exit_rows,
                         np.array(fields.exit_labels)[exit_rows], distance_to_manager,
                         CROWD_MANAGER_RADIUS_M, CROWD_MANAGER_MAX_SWITCH_RATE_PER_S)


def assign_response_times(grid: Grid, rng: np.random.Generator) -> np.ndarray:
    """Assign seat-dependent response times using the paper's specified procedure.

    1. Independently draw N lognormal times (seconds).
    2. Order seats by (number of seats to aisle, number of rows to exit aisle),
       breaking ties randomly; give smaller sampled times to earlier seats.
    3. Shuffle the allocated times among a uniformly chosen 20% of occupants.

    For most seating sections, both ends of the section's row sequence are
    treated as adjacent to exit aisles. The two uppermost side-box sections
    on the first floor are exceptions: they have an aisle only below them,
    so their row distance increases from bottom to top.
    """
    n = len(grid.spawn)
    sampled = rng.lognormal(RESPONSE_LOG_MEAN, RESPONSE_LOG_STD, n)
    row_id_by_seat = np.asarray(grid.seat_row)
    seat_number = np.asarray(grid.seat_number)
    aisle_distance = np.zeros(n, dtype=np.int32)
    row_distance = np.zeros(n, dtype=np.int32)

    # Seats in the same (section, row) block are numbered along the row.
    # The first and last occupied seats are taken as adjacent to side aisles.
    for row_id in np.unique(row_id_by_seat):
        ids = np.flatnonzero(row_id_by_seat == row_id)
        ids = ids[np.argsort(seat_number[ids], kind="stable")]
        positions = np.arange(len(ids))
        aisle_distance[ids] = np.minimum(positions, len(ids) - 1 - positions)

    # The sequence of rows within each section is defined by grid.row_names.
    section_rows = {}
    for row_id, row_name in enumerate(grid.row_names):
        section = row_name.rsplit("/", 1)[0]
        section_rows.setdefault(section, []).append(row_id)
    for section, rows in section_rows.items():
        for position, row_id in enumerate(rows):
            row_distance[row_id_by_seat == row_id] = min(position, len(rows) - 1 - position)

    # The two uppermost side-box blocks have just ONE exit aisle, below them
    # (around y=5 m in the plotted first-floor geometry). For these seats,
    # rank rows from the bottom aisle upward rather than from either end.
    # Use actual seat coordinates so this works without relying on section names.
    seat_nodes = np.asarray(grid.spawn, dtype=np.int32)
    seat_x = np.asarray(grid.node_x)[seat_nodes]
    seat_y = np.asarray(grid.node_y)[seat_nodes]
    first_floor_seat = np.asarray(grid.node_level)[seat_nodes] == 0
    top_side_box = first_floor_seat & (np.abs(seat_x) > 10.0) & (seat_y > 5.0)

    for on_side in (seat_x < 0, seat_x > 0):
        box_seats = np.flatnonzero(top_side_box & on_side)
        if not len(box_seats):
            continue
        box_rows = np.unique(row_id_by_seat[box_seats])
        # Smaller y is closer to the only exit aisle, at the bottom.
        bottom_to_top = sorted(
            box_rows,
            key=lambda row_id: float(
                seat_y[box_seats[row_id_by_seat[box_seats] == row_id]].mean()
            ),
        )
        for distance, row_id in enumerate(bottom_to_top):
            same_row = box_seats[row_id_by_seat[box_seats] == row_id]
            row_distance[same_row] = distance

    random_tiebreakers = rng.random(n)
    seat_order = np.lexsort((random_tiebreakers, row_distance, aisle_distance))
    initially_assigned = np.empty(n, dtype=float)
    initially_assigned[seat_order] = np.sort(sampled)

    count = int(round(RESPONSE_SHUFFLE_FRACTION * n))
    chosen = rng.choice(n, size=count, replace=False)
    response_times = initially_assigned.copy()
    response_times[chosen] = initially_assigned[rng.permutation(chosen)]
    return response_times


def immediate_seat_neighbors(grid: Grid) -> np.ndarray:
    """For each pedestrian, return the IDs in the adjacent seats (or -1).

    Use the section/row identifiers and seat ordering already provided by
    grid.py. Neighbors never span different rows or seating sections.
    """
    n = len(grid.spawn)
    neighbors = np.full((n, 2), -1, dtype=np.int32)
    row_ids = np.asarray(grid.seat_row)
    seat_numbers = np.asarray(grid.seat_number)
    for row_id in np.unique(row_ids):
        occupants = np.flatnonzero(row_ids == row_id)
        occupants = occupants[np.argsort(seat_numbers[occupants], kind="stable")]
        neighbors[occupants[1:], 0] = occupants[:-1]   # left neighbor
        neighbors[occupants[:-1], 1] = occupants[1:]   # right neighbor
    return neighbors


@dataclass
class Agents:
    seat_index: np.ndarray
    start_node: np.ndarray
    current_node: np.ndarray
    section: List[str]
    speed_fraction: np.ndarray
    response_time: np.ndarray
    has_started_moving: np.ndarray
    target_door_name: np.ndarray        # door out of the auditorium
    target_exit_label: np.ndarray       # exit out of Bailey Hall
    knows_closed: np.ndarray            # (num_agents, num_exits) agent has noticed this exit is closed
    door_unusable: np.ndarray           # (num_agents, num_doors) doors the agent knows are closed
    has_exited: np.ndarray
    exit_step: np.ndarray

    target_door_row: np.ndarray         # row in door_costs
    target_exit_row: np.ndarray         # row in fields.cost_by_exit
    independence: np.ndarray
    movement_budget: np.ndarray         # people move at different speeds, so each accumulates a budget
    heading_to_door: np.ndarray         # True until the agent gets out of the auditorium
    last_move_direction: np.ndarray     # -1 if it has not moved
    has_left_seats: np.ndarray
    knows_all_exits: np.ndarray
    led_trust: np.ndarray
    led_guided: np.ndarray
    follows_manager: np.ndarray

    @property
    def num_agents(self) -> int:
        return len(self.start_node)


def spawn_agents(grid: Grid, profile: AudienceProfile, random_generator: np.random.Generator,
                 fields: DistanceFields, door_costs: np.ndarray, door_to_exit_costs: np.ndarray,
                 led: LedGuidance) -> Tuple[Agents, np.ndarray]:
    num_seats = len(grid.spawn)
    seat_indices = np.arange(num_seats)
    start_nodes = grid.spawn.astype(np.int32)

    speed_fractions = np.clip(
        random_generator.normal(profile.speed_fraction_mean, profile.speed_fraction_std, size=num_seats),
        profile.speed_fraction_min, profile.speed_fraction_max)

    independence = np.clip(
        random_generator.normal(profile.independence_mean, profile.independence_std, size=num_seats),
        profile.independence_min, None)

    if profile.knows_all_exits:
        # Cheapest exit of the whole building from the seat, then the door on the cheapest route to it.
        target_exit_row = fields.best_exit_index[start_nodes]
        target_exit_label = np.array(fields.exit_labels)[target_exit_row]

        cost_via_door = door_costs[:, start_nodes] + door_to_exit_costs[:, target_exit_row]
        cheapest_door_index = cost_via_door.argmin(axis=0)
        target_door_name = np.array(KNOWN_AUDITORIUM_DOOR_NAMES)[cheapest_door_index]

        direct_cost = fields.cost_by_exit[target_exit_row, start_nodes]
        heading_to_door = cost_via_door.min(axis=0) <= direct_cost + 1e-6
    else:
        # Cheapest known door from the seat, then the cheapest of the five main entrances.
        cheapest_door_index = door_costs[:, start_nodes].argmin(axis=0)
        target_door_name = np.array(KNOWN_AUDITORIUM_DOOR_NAMES)[cheapest_door_index]

        entrance_rows = [fields.exit_labels.index(label) for label in MAIN_ENTRANCE_LABELS]
        cost_to_entrances = fields.cost_by_exit[entrance_rows][:, start_nodes]
        cheapest_entrance_position = cost_to_entrances.argmin(axis=0)
        target_exit_label = np.array(MAIN_ENTRANCE_LABELS)[cheapest_entrance_position]
        target_exit_row = np.array(entrance_rows)[cheapest_entrance_position]
        heading_to_door = np.ones(num_seats, bool)

    led_trust = np.clip(np.exp(-LED_TRUST_DECAY * (independence - 1.0)), 0.0, 1.0)

    agents = Agents(
        seat_index=seat_indices,
        start_node=start_nodes,
        current_node=start_nodes.copy(),
        section=list(grid.seat_section),
        speed_fraction=speed_fractions,
        response_time=np.zeros(num_seats, dtype=float),
        has_started_moving=np.zeros(num_seats, dtype=bool),
        target_door_name=target_door_name,
        target_exit_label=target_exit_label,
        knows_closed=np.zeros((num_seats, len(fields.exit_labels)), bool),
        door_unusable=np.zeros((num_seats, len(KNOWN_AUDITORIUM_DOOR_NAMES)), bool),
        has_exited=np.zeros(num_seats, bool),
        exit_step=np.full(num_seats, -1, np.int32),
        target_door_row=cheapest_door_index,
        target_exit_row=target_exit_row,
        independence=independence,
        movement_budget=np.zeros(num_seats),
        heading_to_door=heading_to_door,
        last_move_direction=np.full(num_seats, -1, np.int8),
        has_left_seats=np.zeros(num_seats, bool),
        knows_all_exits=np.full(num_seats, profile.knows_all_exits, bool),
        led_trust=led_trust,
        led_guided=np.zeros(num_seats, bool),
        follows_manager=np.zeros(num_seats, bool),
    )

    occupant_at_node = np.full(grid.num_nodes, -1, np.int32)
    occupant_at_node[agents.current_node] = np.arange(num_seats, dtype=np.int32)
    return agents, occupant_at_node


@dataclass
class MovementTables:
    outgoing_edge_order: np.ndarray
    first_edge_of_node: np.ndarray
    edge_direction: np.ndarray
    nearby_nodes: np.ndarray            # nodes within SOCIAL_RADIUS_M of each node, stored linearly
    nearby_weights: np.ndarray          # exp(-distance) for each of those
    first_nearby_of_node: np.ndarray


def build_movement_tables(grid: Grid) -> MovementTables:
    outgoing_edge_order = np.argsort(grid.edge_from, kind="stable")
    first_edge_of_node = np.searchsorted(grid.edge_from[outgoing_edge_order], np.arange(grid.num_nodes + 1))

    step_x = grid.node_x[grid.edge_to] - grid.node_x[grid.edge_from]
    step_y = grid.node_y[grid.edge_to] - grid.node_y[grid.edge_from]
    edge_direction = np.round(np.arctan2(step_y, step_x) / (np.pi / 4)).astype(int) % NUM_DIRECTIONS

    coordinates = np.column_stack([grid.node_x, grid.node_y, grid.node_level * 1000.0])
    neighbor_lists = cKDTree(coordinates).query_ball_point(coordinates, r=SOCIAL_RADIUS_M)

    nearby_nodes, nearby_weights, first_nearby_of_node = [], [], [0]
    for node, neighbors in enumerate(neighbor_lists):
        neighbors = np.array([other for other in neighbors if other != node], dtype=np.int32)
        distances = np.hypot(grid.node_x[neighbors] - grid.node_x[node], grid.node_y[neighbors] - grid.node_y[node])
        nearby_nodes.append(neighbors)
        nearby_weights.append(np.exp(-distances))
        first_nearby_of_node.append(first_nearby_of_node[-1] + len(neighbors))

    return MovementTables(outgoing_edge_order, first_edge_of_node, edge_direction,
                          np.concatenate(nearby_nodes), np.concatenate(nearby_weights),
                          np.array(first_nearby_of_node))

def compute_led_trust(agents: Agents, agent_id: int) -> float:
    closed_seen = int(agents.knows_closed[agent_id].sum())
    led_independence = agents.independence[agent_id] * LED_INDEPENDENCE_GAIN ** closed_seen
    return float(np.exp(-LED_TRUST_DECAY * (led_independence - 1.0)))

def choose_next_edge(grid: Grid, agent_id: int, agents: Agents, occupant_at_node: np.ndarray,
                     cost_to_target: np.ndarray, led: LedGuidance, tables: MovementTables,
                     random_generator: np.random.Generator) -> Optional[int]:
    current_node = agents.current_node[agent_id]
    outgoing_edges = tables.outgoing_edge_order[tables.first_edge_of_node[current_node]:
                                                tables.first_edge_of_node[current_node + 1]]
    neighbor_nodes = grid.edge_to[outgoing_edges]

    distance_gained = cost_to_target[current_node] - cost_to_target[neighbor_nodes]
    can_enter = (occupant_at_node[neighbor_nodes] == -1) & np.isfinite(distance_gained)
    if agents.has_left_seats[agent_id]:
        can_enter &= grid.node_kind[neighbor_nodes] != NODE_SEAT
    if not can_enter.any():
        return None

    nearby = slice(tables.first_nearby_of_node[current_node], tables.first_nearby_of_node[current_node + 1])
    nearby_agents = occupant_at_node[tables.nearby_nodes[nearby]]
    is_present = nearby_agents >= 0
    nearby_weights = tables.nearby_weights[nearby][is_present]
    nearby_directions = agents.last_move_direction[nearby_agents[is_present]]
    has_moved = nearby_directions >= 0
    weight_by_direction = np.bincount(nearby_directions[has_moved], weights=nearby_weights[has_moved],
                                      minlength=NUM_DIRECTIONS)
    total_weight = nearby_weights.sum()
    share_by_direction = weight_by_direction / total_weight if total_weight > 0 else weight_by_direction
    herd_pull = share_by_direction[tables.edge_direction[outgoing_edges]]

    led_pull = np.zeros(len(outgoing_edges))
    led_weight = led.visibility[current_node] * compute_led_trust(agents, agent_id)
    led_cost = led.cost_field_at(current_node)
    if led_weight > 0.0 and np.isfinite(led_cost[current_node]):
        led_distance_gained = led_cost[current_node] - led_cost[neighbor_nodes]
        led_pull = (led_weight * led.strength
                    * np.where(np.isfinite(led_distance_gained), led_distance_gained, 0.0))

    independence = agents.independence[agent_id]
    selfish_pull = 3 * independence * distance_gained
    exponent = selfish_pull + 2 * (HERD_STRENGTH / independence ** 2) * herd_pull + led_pull

    weights = np.zeros(len(outgoing_edges))
    weights[can_enter] = np.exp(exponent[can_enter] - exponent[can_enter].max())
    probabilities = weights / weights.sum()
    return outgoing_edges[random_generator.choice(len(outgoing_edges), p=probabilities)]


def finish_if_at_exit(agents: Agents, agent_id: int, node: int, exit_costs: np.ndarray,
                      exit_goal_cost: np.ndarray, occupant_at_node: np.ndarray, step_number: int) -> bool:
    # Anyone standing in an open exit leaves through it, whatever exit they were aiming for. This matters
    # for the stage-side balcony doors, which are both auditorium doors and exits (BW/BE). Closed exits
    # are marked occupied, so nobody can stand in one. The agent's own target is checked first.
    target = int(agents.target_exit_row[agent_id])
    candidate_rows = [target] + [row for row in range(exit_costs.shape[0]) if row != target]
    for row in candidate_rows:
        if exit_costs[row, node] <= exit_goal_cost[row] + 1e-9:
            agents.has_exited[agent_id] = True
            agents.target_exit_row[agent_id] = row          # exit actually used
            agents.exit_step[agent_id] = step_number
            occupant_at_node[node] = -1
            return True
    return False


def move_agents_one_step(grid: Grid, agents: Agents, occupant_at_node: np.ndarray, door_costs: np.ndarray,
                         exit_costs: np.ndarray, exit_goal_cost: np.ndarray, led: LedGuidance,
                         tables: MovementTables, random_generator: np.random.Generator,
                         step_number: int, seat_neighbors: np.ndarray,
                         exit_options: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
    step_seconds = grid.cell_m / MAX_WALKING_SPEED_M_PER_S
    ready = (~agents.has_exited) & (step_number * step_seconds >= agents.response_time)
    agents.movement_budget[ready] += agents.speed_fraction[ready] * grid.cell_m

    for agent_id in random_generator.permutation(agents.num_agents):
        if not ready[agent_id]:
            continue

        if agents.heading_to_door[agent_id]:
            cost_to_target = door_costs[agents.target_door_row[agent_id]]
        else:
            cost_to_target = exit_costs[agents.target_exit_row[agent_id]]

        chosen_edge = choose_next_edge(grid, agent_id, agents, occupant_at_node, cost_to_target, led,
                                       tables, random_generator)
        if chosen_edge is None or agents.movement_budget[agent_id] < grid.edge_length[chosen_edge]:
            continue

        previous_node, next_node = agents.current_node[agent_id], grid.edge_to[chosen_edge]
        occupant_at_node[previous_node] = -1
        occupant_at_node[next_node] = agent_id
        agents.current_node[agent_id] = next_node
        agents.movement_budget[agent_id] -= grid.edge_length[chosen_edge]
        agents.last_move_direction[agent_id] = tables.edge_direction[chosen_edge]
        if grid.node_kind[next_node] != NODE_SEAT:
            agents.has_left_seats[agent_id] = True

        # A neighboring pedestrian's first actual move may shorten response delay.
        if not agents.has_started_moving[agent_id]:
            agents.has_started_moving[agent_id] = True
            if RESPONSE_TIME_ENABLED and SOCIAL_RESPONSE_ENABLED:
                for neighbor_id in seat_neighbors[agent_id]:
                    if neighbor_id >= 0 and not agents.has_started_moving[neighbor_id]:
                        agents.response_time[neighbor_id] = min(
                            agents.response_time[neighbor_id],
                            step_number * step_seconds + SOCIAL_REACTION_DELAY_S)

        # Any agent has left the auditorium at any door it knows is usable, not only its own.
        on_door = (door_costs[:, next_node] == 0.0) & ~agents.door_unusable[agent_id]
        reached_door = bool(on_door.any())
        agents.led_guided[agent_id] = (led.visibility[next_node] * compute_led_trust(agents, agent_id)
                                       >= LED_RELAX_WEIGHT)
        if agents.heading_to_door[agent_id] and reached_door:
            agents.heading_to_door[agent_id] = False
            if exit_options is not None:
                choose_exit_on_leaving_auditorium(agents, agent_id, next_node, exit_costs, exit_options,
                                                  random_generator)
        # Checked right away because some doors (balcony north) are also exits.  LED-guided people can be
        # led straight to an exit that is not behind any auditorium door (stage-side SW/SE), so they are
        # out as soon as they stand in any exit.
        if not agents.heading_to_door[agent_id] or agents.led_guided[agent_id]:
            finish_if_at_exit(agents, agent_id, next_node, exit_costs, exit_goal_cost, occupant_at_node, step_number)


def choose_exit_on_leaving_auditorium(agents: Agents, agent_id: int, node: int, exit_costs: np.ndarray,
                                      exit_options: Tuple[np.ndarray, np.ndarray],
                                      random_generator: np.random.Generator) -> None:
    """On stepping out of the auditorium, look around once and pick an exit by the usual rule, instead of
    waiting for the next scheduled re-evaluation. Someone told by a manager keeps the manager's exit."""
    if agents.follows_manager[agent_id]:
        return
    all_exit_rows, uninformed_exit_rows = exit_options
    rows = all_exit_rows if agents.knows_all_exits[agent_id] else uninformed_exit_rows
    rows = rows[~agents.knows_closed[agent_id, rows]]
    if rows.size == 0:
        return
    current_position = np.flatnonzero(rows == agents.target_exit_row[agent_id])
    current = int(current_position[0]) if current_position.size else None
    choice = pick_by_distance(exit_costs[rows, node], random_generator, current=current)
    if choice is not None:
        agents.target_exit_row[agent_id] = rows[choice]     # (like reevaluate_targets, only the row is used)


def send_agent_to_exit(agents: Agents, agent_id: int, exit_row: int, exit_label: str, door_costs: np.ndarray,
                       door_to_exit_costs: np.ndarray, exit_costs: np.ndarray) -> None:
    node = agents.current_node[agent_id]
    agents.target_exit_row[agent_id] = exit_row
    agents.target_exit_label[agent_id] = exit_label
    if agents.heading_to_door[agent_id]:
        cost_via_door = door_costs[:, node] + door_to_exit_costs[:, exit_row]
        cost_via_door = np.where(agents.door_unusable[agent_id], np.inf, cost_via_door)
        best_via_door = cost_via_door.min()
        if np.isfinite(best_via_door) and best_via_door <= exit_costs[exit_row, node] + 1e-6:
            agents.target_door_row[agent_id] = int(cost_via_door.argmin())
            agents.target_door_name[agent_id] = KNOWN_AUDITORIUM_DOOR_NAMES[int(cost_via_door.argmin())]
        else:
            agents.heading_to_door[agent_id] = False

def discover_closed_exits(agents: Agents, discovery_distance: np.ndarray, door_blocked_by_exit: np.ndarray,
                          door_costs: np.ndarray, door_to_exit_costs: np.ndarray, exit_costs: np.ndarray,
                          all_exit_rows: np.ndarray, uninformed_exit_rows: np.ndarray,
                          exit_labels: List[str]) -> int:
    if not np.isfinite(discovery_distance).any():
        return 0
    inside = np.flatnonzero(~agents.has_exited)
    in_view = np.isfinite(discovery_distance[:, agents.current_node[inside]]).T      # (inside, exits)
    newly = in_view & ~agents.knows_closed[inside]
    agents.knows_closed[inside] |= in_view
    learners = inside[newly.any(axis=1)]
    for agent_id in learners:
        known = agents.knows_closed[agent_id]
        agents.door_unusable[agent_id] = (door_blocked_by_exit & known).any(axis=1)
        rows = all_exit_rows if agents.knows_all_exits[agent_id] else uninformed_exit_rows
        rows = rows[~known[rows]]
        if rows.size == 0:
            rows = all_exit_rows[~known]
        node = agents.current_node[agent_id]
        target = int(agents.target_exit_row[agent_id])
        if known[target]:
            target = int(rows[np.argmin(exit_costs[rows, node])])    # nearest exit they still believe open
        send_agent_to_exit(agents, agent_id, target, exit_labels[target], door_costs,
                           door_to_exit_costs, exit_costs)
    return len(learners)

def apply_crowd_managers(agents: Agents, managers: Optional[CrowdManagers], door_costs: np.ndarray,
                         door_to_exit_costs: np.ndarray, exit_costs: np.ndarray,
                         random_generator: np.random.Generator, step_seconds: float) -> int:
    if managers is None or managers.num_managers == 0:
        return 0
    inside = np.flatnonzero(~agents.has_exited)
    nodes = agents.current_node[inside]
    distance = managers.distance_to_manager[:, nodes]
    nearest = distance.argmin(axis=0)
    distance_to_nearest = distance[nearest, np.arange(len(inside))]
    new_exit_rows = managers.target_exit_row[nearest]

    in_range = np.isfinite(distance_to_nearest) & np.isfinite(exit_costs[new_exit_rows, nodes])
    closeness = np.where(in_range, 1.0 - distance_to_nearest / managers.radius_m, 0.0)
    switch_probability = 1.0 - np.exp(-managers.max_switch_rate_per_s * closeness * step_seconds)
    told = in_range & (random_generator.random(len(inside)) < switch_probability)

    num_switched = 0
    for position in np.flatnonzero(told):
        agent_id, exit_row = inside[position], new_exit_rows[position]
        agents.follows_manager[agent_id] = True
        if agents.target_exit_row[agent_id] != exit_row:
            num_switched += 1
        send_agent_to_exit(agents, agent_id, exit_row, managers.target_exit_label[nearest[position]],
                           door_costs, door_to_exit_costs, exit_costs)
    return num_switched


def pick_by_distance(costs: np.ndarray, random_generator: np.random.Generator, current: Optional[int] = None,
                     visible_costs: Optional[np.ndarray] = None) -> Optional[int]:
    if visible_costs is None:
        visible_costs = costs
    visible = np.isfinite(costs) & (visible_costs <= VIEW_RANGE_M)
    if current is not None and np.isfinite(costs[current]):
        visible[current] = True
    if not visible.any():
        return None
    weights = np.zeros(len(costs))
    weights[visible] = np.exp(-TARGET_CHOICE_SHARPNESS * (costs[visible] - costs[visible].min()))
    return random_generator.choice(len(costs), p=weights / weights.sum())


def reevaluate_targets(agents: Agents, door_costs: np.ndarray, door_to_exit_costs: np.ndarray,
                       exit_costs: np.ndarray, exit_goal_cost: np.ndarray, all_exit_rows: np.ndarray,
                       main_entrance_rows: np.ndarray, occupant_at_node: np.ndarray,
                       random_generator: np.random.Generator, step_number: int) -> None:
    step_seconds = CELL_M / MAX_WALKING_SPEED_M_PER_S
    is_due = ((~agents.has_exited) & (step_number * step_seconds >= agents.response_time)
              & ((step_number + np.arange(agents.num_agents)) % REEVALUATE_EVERY_STEPS == 0))
    for agent_id in np.nonzero(is_due)[0]:
        node = agents.current_node[agent_id]
        knows_all = agents.knows_all_exits[agent_id]
        told_by_manager = agents.follows_manager[agent_id]
        unusable_doors = agents.door_unusable[agent_id]
        if agents.heading_to_door[agent_id]:
            if knows_all or told_by_manager:
                cost_via_door = door_costs[:, node] + door_to_exit_costs[:, agents.target_exit_row[agent_id]]
                cost_via_door = np.where(unusable_doors, np.inf, cost_via_door)
                choice = pick_by_distance(cost_via_door, random_generator, visible_costs=door_costs[:, node],
                                          current=int(agents.target_door_row[agent_id]))
            else:
                choice = pick_by_distance(np.where(unusable_doors, np.inf, door_costs[:, node]),
                                          random_generator, current=int(agents.target_door_row[agent_id]))
            if choice is not None:
                agents.target_door_row[agent_id] = choice
                if door_costs[choice, node] == 0.0:          # already standing in the door
                    agents.heading_to_door[agent_id] = False
                    finish_if_at_exit(agents, agent_id, node, exit_costs, exit_goal_cost, occupant_at_node, step_number)
        elif told_by_manager:
            continue                                        # keep the exit the manager gave
        else:
            rows = all_exit_rows if knows_all else main_entrance_rows
            rows = rows[~agents.knows_closed[agent_id, rows]]
            if rows.size == 0:
                continue
            current_position = np.flatnonzero(rows == agents.target_exit_row[agent_id])
            current = int(current_position[0]) if current_position.size else None
            choice = pick_by_distance(exit_costs[rows, node], random_generator, current=current)
            if choice is not None:
                agents.target_exit_row[agent_id] = rows[choice]


def time_to_fraction_out(agents: Agents, fraction: float, step_seconds: float) -> Optional[Tuple[int, float]]:
    needed = int(np.ceil(fraction * agents.num_agents))
    exit_steps = np.sort(agents.exit_step[agents.has_exited])
    if len(exit_steps) < needed:
        return None
    step = int(exit_steps[needed - 1])
    return step, step * step_seconds


@dataclass
class Scenario:
    """Everything that does not change between runs: grid, painted barriers, cost fields.
    Build it once with build_scenario() and pass it to run_simulation() as often as you like."""
    grid: Grid
    paint: Optional[GridPaint]
    fields: DistanceFields
    door_costs: np.ndarray
    door_to_exit_costs: np.ndarray
    exit_goal_cost: np.ndarray
    tables: MovementTables
    all_exit_rows: np.ndarray
    uninformed_exit_rows: np.ndarray     # main entrances plus the side exits
    discovery_distance: np.ndarray       # (num_exits, num_nodes) distance to a closed exit within view, else inf
    door_blocked_by_exit: np.ndarray     # (num_doors, num_exits) door shares nodes with that exit
    seat_neighbors: np.ndarray
    manager_cache: dict

    @property
    def exit_costs(self) -> np.ndarray:
        """Believed costs: closed exits look open until someone sees them."""
        return self.fields.cost_by_exit


def build_scenario(cell_m: float = CELL_M, paint_path: Optional[str] = PAINT_PATH,
                   barriers_enabled: bool = BARRIERS_ENABLED,
                   closed_exit_labels: Sequence[str] = CLOSED_EXIT_LABELS) -> Scenario:
    grid = Grid.from_model(cell_m=cell_m)
    paint = GridPaint.load(paint_path, grid) if paint_path else None
    if barriers_enabled and paint is not None:
        barrier_nodes = paint.node_is_barrier(grid)
    else:
        barrier_nodes = np.zeros(grid.num_nodes, bool)

    fields = compute_distance_fields(grid, barrier_nodes, closed_exit_labels)
    door_costs = compute_door_costs(grid, fields.blocked_nodes)
    door_to_exit_costs = compute_door_to_exit_costs(door_costs, fields.cost_by_exit)
    exit_goal_cost = compute_exit_goal_costs(fields.cost_by_exit)

    main_entrance_rows = np.array([fields.exit_labels.index(label) for label in MAIN_ENTRANCE_LABELS], dtype=int)
    side_exit_rows = np.array([fields.exit_labels.index(label) for label in SIDE_EXIT_LABELS], dtype=int)

    exit_node_masks = [mask_to_node_mask(grid, grid.exit_mask(label)) for label in fields.exit_labels]
    door_node_masks = [mask_to_node_mask(grid, grid.door_mask(door)) for door in KNOWN_AUDITORIUM_DOOR_NAMES]
    door_blocked_by_exit = np.array([[bool((door_nodes & exit_nodes).any()) for exit_nodes in exit_node_masks]
                                     for door_nodes in door_node_masks])

    return Scenario(grid, paint, fields, door_costs, door_to_exit_costs, exit_goal_cost,
                    build_movement_tables(grid), np.arange(len(fields.exit_labels)),
                    np.concatenate([main_entrance_rows, side_exit_rows]),
                    compute_discovery_distance(grid, fields, barrier_nodes), door_blocked_by_exit,
                    immediate_seat_neighbors(grid), {})


@dataclass
class RunResult:
    seed: int
    num_agents: int
    num_out: int
    evacuation_time_s: float             # last person out (inf if someone never got out)
    time_90_s: float
    time_95_s: float
    exit_counts: dict
    num_told_by_manager: int
    num_exit_changes_by_manager: int
    num_led_guided: int                  # people guided by the LEDs at some point
    num_noticed_closed_exit: int
    exit_times_s: np.ndarray             # per agent, inf if not out
    frames: Optional[List[dict]] = None
    manager_nodes: Optional[np.ndarray] = None
    exit_label_by_agent: Optional[np.ndarray] = None   # exit each agent used ("" if not out)

    def summary(self) -> dict:
        return dict(seed=self.seed, t100=self.evacuation_time_s, t95=self.time_95_s, t90=self.time_90_s,
                    num_out=self.num_out, num_agents=self.num_agents, exit_counts=self.exit_counts,
                    told_by_manager=self.num_told_by_manager)


def run_simulation(scenario: Scenario, seed: int = RANDOM_SEED, profile: AudienceProfile = PROFILE,
                   manager_posts: Optional[Sequence[CrowdManagerPost]] = CROWD_MANAGER_POSTS,
                   led: Optional[LedGuidance] = None, record_frames: bool = False,
                   max_steps: int = MAX_STEPS, verbose: bool = False) -> RunResult:
    """One evacuation.  `led=None` uses the painted LEDs; `manager_posts=None` or () means no managers
    (managers are also off when the profile has none)."""
    grid, fields = scenario.grid, scenario.fields
    exit_costs, door_costs = fields.cost_by_exit, scenario.door_costs
    if led is None:
        led = build_led_guidance(grid, scenario.paint, fields)

    random_generator = np.random.default_rng(seed)
    agents, occupant_at_node = spawn_agents(grid, profile, random_generator, fields,
                                            door_costs, scenario.door_to_exit_costs, led)
    occupant_at_node[fields.closed_nodes] = -2

    # A separate stream preserves the existing movement and target-choice RNG.
    response_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    if RESPONSE_TIME_ENABLED:
        agents.response_time = assign_response_times(grid, response_rng)
    initially_assigned = agents.response_time.copy()

    managers = None
    if profile.has_crowd_managers and manager_posts:
        managers = place_crowd_managers(grid, fields, manager_posts, cache=scenario.manager_cache)
    manager_random_generator = np.random.default_rng(np.random.SeedSequence([seed, 2]))

    if verbose:
        print(f"blocked nodes: {int(fields.blocked_nodes.sum())}, lit nodes: {int(led.lit_nodes.sum())}")
        print(f"Original response times: mean={initially_assigned.mean():.1f} s, "
              f"median={np.median(initially_assigned):.1f} s, maximum={initially_assigned.max():.1f} s")
        print(f"profile: {profile.name} (mean independence {agents.independence.mean():.2f})")
        for title, values in (("initial target doors", agents.target_door_name),
                              ("initial target exits", agents.target_exit_label)):
            names, counts = np.unique(values, return_counts=True)
            print(f"{title}:", dict(zip(names.tolist(), counts.tolist())))
        print("start without a door stage:", int((~agents.heading_to_door).sum()))
        if managers is not None:
            required = int(np.ceil(agents.num_agents / ATTENDEES_PER_CROWD_MANAGER))
            print(f"crowd managers: {managers.num_managers} (law requires at least {required})")
            for name, node, label in zip(managers.names, managers.node, managers.target_exit_label):
                print(f"  {name:20s} at ({grid.node_x[node]:6.1f}, {grid.node_y[node]:6.1f}) m -> {label}")

    time_step_seconds = grid.cell_m / MAX_WALKING_SPEED_M_PER_S
    num_redirected = 0
    num_discoveries = 0
    ever_led_guided = np.zeros(agents.num_agents, bool)
    frames = [] if record_frames else None     # one snapshot per step, for animate.py
    for step_number in range(1, max_steps + 1):
        num_discoveries += discover_closed_exits(agents, scenario.discovery_distance, scenario.door_blocked_by_exit,
                                                 door_costs, scenario.door_to_exit_costs, exit_costs,
                                                 scenario.all_exit_rows, scenario.uninformed_exit_rows,
                                                 fields.exit_labels)
        reevaluate_targets(agents, door_costs, scenario.door_to_exit_costs, exit_costs, scenario.exit_goal_cost,
                           scenario.all_exit_rows, scenario.uninformed_exit_rows, occupant_at_node,
                           random_generator, step_number)
        num_redirected += apply_crowd_managers(agents, managers, door_costs, scenario.door_to_exit_costs,
                                               exit_costs, manager_random_generator, time_step_seconds)
        move_agents_one_step(grid, agents, occupant_at_node, door_costs, exit_costs, scenario.exit_goal_cost,
                             led, scenario.tables, random_generator, step_number, scenario.seat_neighbors,
                             (scenario.all_exit_rows, scenario.uninformed_exit_rows)
                             if REEVALUATE_ON_LEAVING_AUDITORIUM else None)
        ever_led_guided |= agents.led_guided
        if record_frames:
            frames.append(dict(
                node=agents.current_node.copy(),
                exited=agents.has_exited.copy(),
                to_door=agents.heading_to_door.copy(),
                manager=agents.follows_manager.copy(),
            ))
        if agents.has_exited.all():
            break

    exit_names = np.array(fields.exit_labels)[agents.target_exit_row[agents.has_exited]]
    names, counts = np.unique(exit_names, return_counts=True)
    exit_times = np.where(agents.has_exited, agents.exit_step * time_step_seconds, np.inf)
    if INCLUDE_NORTH_STAIR_COST:
        # BW/BE end at the balcony door; add the time to walk down the stage-side stair at stair speed.
        north_rows = [fields.exit_labels.index(label) for label in NORTH_STAIR_EXIT_LABELS]
        via_north = agents.has_exited & np.isin(agents.target_exit_row, north_rows)
        stair_seconds = north_stair_extra_cost() / (agents.speed_fraction * MAX_WALKING_SPEED_M_PER_S)
        exit_times = np.where(via_north, exit_times + stair_seconds, exit_times)

    def fraction_time(fraction: float) -> float:
        needed = int(np.ceil(fraction * agents.num_agents))
        finished = np.sort(exit_times[np.isfinite(exit_times)])
        return float(finished[needed - 1]) if len(finished) >= needed else np.inf

    result = RunResult(
        seed=seed, num_agents=agents.num_agents, num_out=int(agents.has_exited.sum()),
        evacuation_time_s=fraction_time(1.0), time_90_s=fraction_time(0.9), time_95_s=fraction_time(0.95),
        exit_counts=dict(zip(names.tolist(), counts.tolist())),
        num_told_by_manager=int(agents.follows_manager.sum()), num_exit_changes_by_manager=num_redirected,
        num_led_guided=int(ever_led_guided.sum()),
        num_noticed_closed_exit=int(agents.knows_closed.any(axis=1).sum()),
        exit_times_s=exit_times, frames=frames,
        manager_nodes=managers.node if managers is not None else None,
        exit_label_by_agent=np.where(agents.has_exited, np.array(fields.exit_labels)[agents.target_exit_row], ""))

    if verbose:
        print(f"After social response: mean={agents.response_time.mean():.1f} s, "
              f"median={np.median(agents.response_time):.1f} s, "
              f"maximum={agents.response_time.max():.1f} s; "
              f"{np.count_nonzero(agents.response_time < initially_assigned)} shortened")
        print(f"agents who noticed a closed exit: {result.num_noticed_closed_exit} of {agents.num_agents}")
        print(f"guided by the LEDs at some point: {result.num_led_guided} of {agents.num_agents}")
        if managers is not None:
            print(f"told by a crowd manager: {result.num_told_by_manager} of {agents.num_agents} "
                  f"({num_redirected} exit changes)")
        print("exit used:", result.exit_counts)
        print(f"{result.num_out} of {agents.num_agents} out, last exit at step "
              f"{int(agents.exit_step.max())} ({agents.exit_step.max() * time_step_seconds:.0f} s)")
        if np.isfinite(result.time_90_s):
            print(f"90% out by {result.time_90_s:.0f} s")
    return result


if __name__ == "__main__":
    scenario = build_scenario()
    fields = scenario.fields
    print(fields.exit_labels, "closed:", fields.closed_exit_labels)
    if scenario.paint is not None:
        print(f"{int(scenario.paint.cell_has_led.sum())} painted LED cells")
    result = run_simulation(scenario, seed=RANDOM_SEED, profile=PROFILE, record_frames=True, verbose=True)
    time_step_seconds = scenario.grid.cell_m / MAX_WALKING_SPEED_M_PER_S
    closed = list(fields.closed_exit_labels)
    closed_exit_cells = (np.any([np.asarray(scenario.grid.exit_mask(label), bool) for label in closed], axis=0)
                         if closed else None)
    print("frames saved to", save_run(RUN_PATH, scenario.grid, result.frames, time_step_seconds,
                                      manager_nodes=result.manager_nodes, closed_exit_cells=closed_exit_cells))
