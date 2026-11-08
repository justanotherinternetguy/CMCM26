from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from animate import RUN_PATH, save_run
from grid import COST_STAIRS, NODE_SEAT, Grid
from grid_paint import GridPaint
from model import FT, Model

CELL_M = 0.5
MAX_WALKING_SPEED_M_PER_S = 1.4
RANDOM_SEED = 42067
MAX_STEPS = 5000

NORTH_STAIR_EXIT_LABELS = ("BW", "BE")
INCLUDE_NORTH_STAIR_COST = True

NUM_DIRECTIONS = 8
SOCIAL_RADIUS_M = 2.0
HERD_STRENGTH = 2
REEVALUATE_EVERY_STEPS = 30
TARGET_CHOICE_SHARPNESS = 1
VIEW_RANGE_M = 15.0

PAINT_PATH = f"out/grid_paint_{CELL_M:g}m.npz"
BARRIERS_ENABLED = True

LED_ENABLED = True
LED_STRENGTH = 10.0
LED_FOLLOW_PROBABILITY = 0.85

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
    independence_mean: float          # higher = cares more about the shortest route, less about the herd
    independence_std: float
    independence_min: float
    knows_all_exits: bool = False


CONCERT_PROFILE = AudienceProfile(
    "concert",
    speed_fraction_mean=0.8, speed_fraction_std=0.15, speed_fraction_min=0.3, speed_fraction_max=1.0,
    independence_mean=3.0, independence_std=1.0, independence_min=1.0,
    knows_all_exits=False)

STUDENT_PROFILE = AudienceProfile(
    "students",
    speed_fraction_mean=0.8, speed_fraction_std=0.15, speed_fraction_min=0.3, speed_fraction_max=1.0,
    independence_mean=4.5, independence_std=1.0, independence_min=1.0,
    knows_all_exits=True)

PROFILE = STUDENT_PROFILE


@dataclass
class DistanceFields:
    exit_labels: List[str]
    closed_exit_labels: Tuple[str, ...]
    cost_by_exit: np.ndarray          # cost from every node to each exit
    best_cost: np.ndarray             # cost to the nearest open exit
    best_exit_index: np.ndarray       # index into exit_labels
    blocked_nodes: np.ndarray         # painted barriers plus closed exits

    def best_exit_label(self, node_id: int) -> str:
        return self.exit_labels[int(self.best_exit_index[node_id])]


def north_stair_extra_cost() -> float:
    north_stair_length_m = Model().p.stair_nw_length * FT
    return north_stair_length_m * COST_STAIRS


def compute_distance_fields(grid: Grid, barrier_nodes: np.ndarray,
                            closed_exit_labels: Sequence[str] = ()) -> DistanceFields:
    closed_exit_labels = tuple(closed_exit_labels)
    blocked_nodes = barrier_nodes.copy()
    if closed_exit_labels:
        blocked_nodes |= grid.exit_mask(list(closed_exit_labels))
    blocked_or_none = blocked_nodes if blocked_nodes.any() else None

    open_exit_labels = [label for label in grid.exits if label not in closed_exit_labels]
    extra_cost_for_north_stairs = north_stair_extra_cost() if INCLUDE_NORTH_STAIR_COST else 0.0

    cost_rows = []
    for exit_label in open_exit_labels:
        cost_to_this_exit = grid.node_distance_field(targets=grid.exit_mask(exit_label), blocked=blocked_or_none)
        if exit_label in NORTH_STAIR_EXIT_LABELS:
            cost_to_this_exit = cost_to_this_exit + extra_cost_for_north_stairs
        cost_rows.append(cost_to_this_exit)
    cost_by_exit = np.vstack(cost_rows)

    best_exit_index = cost_by_exit.argmin(axis=0)
    best_cost = cost_by_exit[best_exit_index, np.arange(grid.num_nodes)]
    return DistanceFields(open_exit_labels, closed_exit_labels, cost_by_exit, best_cost, best_exit_index,
                          blocked_nodes)


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
    follow_probability: float
    lit_nodes: np.ndarray             # True for every node touched by a painted LED
    cost_to_nearest_exit: np.ndarray  # what the strips point down: cost to the closest exit from each node


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


def build_led_guidance(grid: Grid, paint: GridPaint, fields: DistanceFields) -> LedGuidance:
    if not LED_ENABLED:
        return LedGuidance(False, LED_STRENGTH, LED_FOLLOW_PROBABILITY,
                           np.zeros(grid.num_nodes, bool), fields.best_cost)
    lit_nodes = node_is_lit(paint, grid) & ~fields.blocked_nodes
    return LedGuidance(True, LED_STRENGTH, LED_FOLLOW_PROBABILITY, lit_nodes, fields.best_cost)


@dataclass
class Agents:
    seat_index: np.ndarray
    start_node: np.ndarray
    current_node: np.ndarray
    section: List[str]
    speed_fraction: np.ndarray
    target_door_name: np.ndarray        # door out of the auditorium
    target_exit_label: np.ndarray       # exit out of Bailey Hall
    knows_exit_is_blocked: np.ndarray
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
    follows_led: np.ndarray

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

    # Drawn even when LEDs are off so runs with and without LEDs start from the same audience.
    follows_led = led.enabled & (random_generator.random(num_seats) < led.follow_probability)

    agents = Agents(
        seat_index=seat_indices,
        start_node=start_nodes,
        current_node=start_nodes.copy(),
        section=list(grid.seat_section),
        speed_fraction=speed_fractions,
        target_door_name=target_door_name,
        target_exit_label=target_exit_label,
        knows_exit_is_blocked=np.zeros(num_seats, bool),
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
        follows_led=follows_led,
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


def choose_next_edge(grid: Grid, agent_id: int, agents: Agents, occupant_at_node: np.ndarray,
                     cost_to_target: np.ndarray, led: LedGuidance, tables: MovementTables,
                     random_generator: np.random.Generator) -> Optional[int]:
    current_node = agents.current_node[agent_id]
    outgoing_edges = tables.outgoing_edge_order[tables.first_edge_of_node[current_node]:
                                                tables.first_edge_of_node[current_node + 1]]
    neighbor_nodes = grid.edge_to[outgoing_edges]

    distance_gained = cost_to_target[current_node] - cost_to_target[neighbor_nodes]
    can_enter = (occupant_at_node[neighbor_nodes] < 0) & np.isfinite(distance_gained)
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
    if (agents.follows_led[agent_id] and led.lit_nodes[current_node]
            and np.isfinite(led.cost_to_nearest_exit[current_node])):
        led_distance_gained = led.cost_to_nearest_exit[current_node] - led.cost_to_nearest_exit[neighbor_nodes]
        led_pull = led.strength * np.where(np.isfinite(led_distance_gained), led_distance_gained, 0.0)

    independence = agents.independence[agent_id]
    selfish_pull = 3 * independence * distance_gained
    exponent = selfish_pull + 2 * (HERD_STRENGTH / independence ** 2) * herd_pull + led_pull

    weights = np.zeros(len(outgoing_edges))
    weights[can_enter] = np.exp(exponent[can_enter] - exponent[can_enter].max())
    probabilities = weights / weights.sum()
    return outgoing_edges[random_generator.choice(len(outgoing_edges), p=probabilities)]


def finish_if_at_exit(agents: Agents, agent_id: int, node: int, exit_costs: np.ndarray,
                      exit_goal_cost: np.ndarray, occupant_at_node: np.ndarray, step_number: int) -> bool:
    """People following the LEDs are out at any exit; everyone else only at their target exit."""
    if agents.follows_led[agent_id]:
        candidate_rows = range(exit_costs.shape[0])
    else:
        candidate_rows = [agents.target_exit_row[agent_id]]
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
                         step_number: int) -> None:
    agents.movement_budget += agents.speed_fraction * grid.cell_m

    for agent_id in random_generator.permutation(agents.num_agents):
        if agents.has_exited[agent_id]:
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

        # People following the LEDs have left the auditorium at any door, not only their own.
        reached_door = door_costs[agents.target_door_row[agent_id], next_node] == 0.0
        if agents.follows_led[agent_id]:
            reached_door = reached_door or door_costs[:, next_node].min() == 0.0
        if agents.heading_to_door[agent_id] and reached_door:
            agents.heading_to_door[agent_id] = False
        # Checked right away because some doors (balcony north) are also exits.
        if not agents.heading_to_door[agent_id]:
            finish_if_at_exit(agents, agent_id, next_node, exit_costs, exit_goal_cost, occupant_at_node, step_number)


def pick_by_distance(costs: np.ndarray, random_generator: np.random.Generator, current: Optional[int] = None,
                     visible_costs: Optional[np.ndarray] = None) -> Optional[int]:
    """Pick an index with probability falling off with `costs`, among options within VIEW_RANGE_M."""
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
    # Staggered by agent number so the crowd reevaluates in batches.
    is_due = ~agents.has_exited & ((step_number + np.arange(agents.num_agents)) % REEVALUATE_EVERY_STEPS == 0)
    for agent_id in np.nonzero(is_due)[0]:
        node = agents.current_node[agent_id]
        knows_all = agents.knows_all_exits[agent_id]
        if agents.heading_to_door[agent_id]:
            if knows_all:
                cost_via_door = door_costs[:, node] + door_to_exit_costs[:, agents.target_exit_row[agent_id]]
                choice = pick_by_distance(cost_via_door, random_generator, visible_costs=door_costs[:, node])
            else:
                choice = pick_by_distance(door_costs[:, node], random_generator)
            if choice is not None:
                agents.target_door_row[agent_id] = choice
                if door_costs[choice, node] == 0.0:          # already standing in the door
                    agents.heading_to_door[agent_id] = False
                    finish_if_at_exit(agents, agent_id, node, exit_costs, exit_goal_cost, occupant_at_node, step_number)
        else:
            rows = all_exit_rows if knows_all else main_entrance_rows
            current_position = np.flatnonzero(rows == agents.target_exit_row[agent_id])
            current = int(current_position[0]) if (knows_all and current_position.size) else None
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


if __name__ == "__main__":
    grid = Grid.from_model(cell_m=CELL_M)
    paint = GridPaint.load(PAINT_PATH, grid)
    barrier_nodes = paint.node_is_barrier(grid) if BARRIERS_ENABLED else np.zeros(grid.num_nodes, bool)

    fields = compute_distance_fields(grid, barrier_nodes)
    print(fields.exit_labels)
    exit_costs = fields.cost_by_exit
    door_costs = compute_door_costs(grid, fields.blocked_nodes)
    door_to_exit_costs = compute_door_to_exit_costs(door_costs, exit_costs)
    exit_goal_cost = compute_exit_goal_costs(exit_costs)

    led = build_led_guidance(grid, paint, fields)
    print(f"blocked nodes: {int(fields.blocked_nodes.sum())}, lit nodes: {int(led.lit_nodes.sum())} "
          f"({int(paint.cell_has_led.sum())} painted LED cells)")

    random_generator = np.random.default_rng(RANDOM_SEED)
    agents, occupant_at_node = spawn_agents(grid, PROFILE, random_generator, fields,
                                            door_costs, door_to_exit_costs, led)

    print(f"profile: {PROFILE.name} (mean independence {agents.independence.mean():.2f})")
    print(f"people following the LEDs: {int(agents.follows_led.sum())} of {agents.num_agents}")
    for title, values in (("initial target doors", agents.target_door_name),
                          ("initial target exits", agents.target_exit_label)):
        names, counts = np.unique(values, return_counts=True)
        print(f"{title}:", dict(zip(names.tolist(), counts.tolist())))
    print("start without a door stage:", int((~agents.heading_to_door).sum()))

    tables = build_movement_tables(grid)
    all_exit_rows = np.arange(len(fields.exit_labels))
    main_entrance_rows = np.array([fields.exit_labels.index(label) for label in MAIN_ENTRANCE_LABELS])

    frames = []     # one snapshot per step, for animate.py
    for step_number in range(1, MAX_STEPS + 1):
        reevaluate_targets(agents, door_costs, door_to_exit_costs, exit_costs, exit_goal_cost, all_exit_rows,
                           main_entrance_rows, occupant_at_node, random_generator, step_number)
        move_agents_one_step(grid, agents, occupant_at_node, door_costs, exit_costs, exit_goal_cost, led,
                             tables, random_generator, step_number)
        frames.append(dict(
            node=agents.current_node.copy(),
            exited=agents.has_exited.copy(),
            to_door=agents.heading_to_door.copy(),
        ))
        if agents.has_exited.all():
            break

    time_step_seconds = grid.cell_m / MAX_WALKING_SPEED_M_PER_S
    exit_names = np.array(fields.exit_labels)[agents.target_exit_row[agents.has_exited]]
    names, counts = np.unique(exit_names, return_counts=True)
    print("exit used:", dict(zip(names.tolist(), counts.tolist())))
    print(f"{int(agents.has_exited.sum())} of {agents.num_agents} out, last exit at step "
          f"{int(agents.exit_step.max())} ({agents.exit_step.max() * time_step_seconds:.0f} s)")
    ninety_percent = time_to_fraction_out(agents, 0.9, time_step_seconds)
    if ninety_percent is not None:
        print(f"90% out by {ninety_percent[1]:.0f} s")
    print("frames saved to", save_run(RUN_PATH, grid, frames, time_step_seconds))