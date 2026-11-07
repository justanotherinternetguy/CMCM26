from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from grid import COST_STAIRS, Grid
from model import FT, Model

MAX_WALKING_SPEED_M_PER_S = 1.4
RANDOM_SEED = 6767

NORTH_STAIR_EXIT_LABELS = ("BW", "BE")
INCLUDE_NORTH_STAIR_COST = True

NUM_DIRECTIONS = 8
SOCIAL_RADIUS = 2.0
HERD_STRENGTH = 3.0             
REEVALUATE_EVERY_STEPS = 30     
TARGET_CHOICE_SHARPNESS = 0.3   

# The five main entrances at the front.
MAIN_ENTRANCE_LABELS = ("F1", "F2", "F3", "F4", "F5")

# Everyone can see all the doors out of the auditorium, so they can choose among them.
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

CONCERT_PROFILE = AudienceProfile("concert", speed_fraction_mean=0.8, speed_fraction_std=0.15,
                                  speed_fraction_min=0.3, speed_fraction_max=1.0,
                                  independence_mean=3.0, independence_std=1.0, independence_min=1.0)

@dataclass
class DistanceFields:
    exit_labels: List[str]
    closed_exit_labels: Tuple[str, ...]
    cost_by_exit: np.ndarray        # cost from every node to each exit
    best_cost: np.ndarray           # cost to the nearest open exit
    best_exit_index: np.ndarray     # index into exit_labels

    def best_exit_label(self, node_id: int) -> str:
        return self.exit_labels[int(self.best_exit_index[node_id])]


def north_stair_extra_cost() -> float:
    north_stair_length_m = Model().p.stair_nw_length * FT
    return north_stair_length_m * COST_STAIRS


def compute_distance_fields(grid: Grid, closed_exit_labels: Sequence[str] = ()) -> DistanceFields:
    closed_exit_labels = tuple(closed_exit_labels)
    blocked_cells = grid.exit_mask(list(closed_exit_labels)) if closed_exit_labels else None
    open_exit_labels = [label for label in grid.exits if label not in closed_exit_labels]
    extra_cost_for_north_stairs = north_stair_extra_cost() if INCLUDE_NORTH_STAIR_COST else 0.0

    cost_rows = []
    for exit_label in open_exit_labels:
        cost_to_this_exit = grid.node_distance_field(targets=grid.exit_mask(exit_label), blocked=blocked_cells)
        if exit_label in NORTH_STAIR_EXIT_LABELS:
            cost_to_this_exit = cost_to_this_exit + extra_cost_for_north_stairs
        cost_rows.append(cost_to_this_exit)
    cost_by_exit = np.vstack(cost_rows)

    best_exit_index = cost_by_exit.argmin(axis=0)
    best_cost = cost_by_exit[best_exit_index, np.arange(grid.num_nodes)]
    return DistanceFields(open_exit_labels, closed_exit_labels, cost_by_exit, best_cost, best_exit_index)


def compute_door_costs(grid: Grid) -> np.ndarray:
    return np.vstack([grid.node_distance_field(targets=grid.door_mask(door_name)) for door_name in KNOWN_AUDITORIUM_DOOR_NAMES])


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
    movement_budget: np.ndarray         # cause people move different speeds so they move once they've accumulated a budget to move
    heading_to_door: np.ndarray         # True until the agent gets out the auditorium
    last_move_direction: np.ndarray     # index of the agent's last move, -1 if it has not moved

    @property
    def num_agents(self) -> int:
        return len(self.start_node)


def spawn_agents(grid: Grid, profile: AudienceProfile, random_generator: np.random.Generator, fields: DistanceFields, door_costs: np.ndarray) -> Tuple[Agents, np.ndarray]:
    num_seats = len(grid.spawn)
    seat_indices = np.arange(num_seats)
    start_nodes = grid.spawn.astype(np.int32)

    speed_fractions = np.clip(
        random_generator.normal(profile.speed_fraction_mean, profile.speed_fraction_std, size=num_seats),
        profile.speed_fraction_min, profile.speed_fraction_max)

    independence = np.clip(
        random_generator.normal(profile.independence_mean, profile.independence_std, size=num_seats),
        profile.independence_min, None)

    # Target 1: cheapest known auditorium door from seat.
    cheapest_door_index = door_costs[:, start_nodes].argmin(axis=0)
    target_door_name = np.array(KNOWN_AUDITORIUM_DOOR_NAMES)[cheapest_door_index]

    # Target 2: cheapest of the five main entrances from seat.
    entrance_rows = [fields.exit_labels.index(label) for label in MAIN_ENTRANCE_LABELS]
    cost_to_entrances = fields.cost_by_exit[entrance_rows][:, start_nodes]
    cheapest_entrance_position = cost_to_entrances.argmin(axis=0)
    target_exit_label = np.array(MAIN_ENTRANCE_LABELS)[cheapest_entrance_position]
    target_exit_row = np.array(entrance_rows)[cheapest_entrance_position]

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
        heading_to_door=np.ones(num_seats, bool),
        last_move_direction=np.full(num_seats, -1, np.int8),
    )

    occupant_at_node = np.full(grid.num_nodes, -1, np.int32)
    occupant_at_node[agents.current_node] = np.arange(num_seats, dtype=np.int32)
    return agents, occupant_at_node


@dataclass
class MovementTables:
    outgoing_edge_order: np.ndarray
    first_edge_of_node: np.ndarray
    edge_direction: np.ndarray         
    nearby_nodes: np.ndarray            # for each node, the nodes within SOCIAL_RADIUS_M, stored linearly
    nearby_weights: np.ndarray          # exp(-distance) for each of those
    first_nearby_of_node: np.ndarray


def build_movement_tables(grid: Grid) -> MovementTables:
    outgoing_edge_order = np.argsort(grid.edge_from, kind="stable")
    first_edge_of_node = np.searchsorted(grid.edge_from[outgoing_edge_order], np.arange(grid.num_nodes + 1))

    step_x = grid.node_x[grid.edge_to] - grid.node_x[grid.edge_from]
    step_y = grid.node_y[grid.edge_to] - grid.node_y[grid.edge_from]
    edge_direction = np.round(np.arctan2(step_y, step_x) / (np.pi / 4)).astype(int) % NUM_DIRECTIONS

    coordinates = np.column_stack([grid.node_x, grid.node_y, grid.node_level * 1000.0])
    neighbor_lists = cKDTree(coordinates).query_ball_point(coordinates, r=SOCIAL_RADIUS)

    nearby_nodes, nearby_weights, first_nearby_of_node = [], [], [0]
    for node, neighbors in enumerate(neighbor_lists):
        neighbors = np.array([other for other in neighbors if other != node], dtype=np.int32)
        distances = np.hypot(grid.node_x[neighbors] - grid.node_x[node], grid.node_y[neighbors] - grid.node_y[node])
        nearby_nodes.append(neighbors)
        nearby_weights.append(np.exp(-distances))
        first_nearby_of_node.append(first_nearby_of_node[-1] + len(neighbors))

    return MovementTables(outgoing_edge_order, first_edge_of_node, edge_direction, np.concatenate(nearby_nodes), np.concatenate(nearby_weights), np.array(first_nearby_of_node))

def choose_next_edge(grid, agent_id, agents, occupant_at_node, cost_to_target, tables, random_generator):
    current_node = agents.current_node[agent_id]
    outgoing_edges = tables.outgoing_edge_order[tables.first_edge_of_node[current_node]:
                                                tables.first_edge_of_node[current_node + 1]]
    neighbor_nodes = grid.edge_to[outgoing_edges]

    distance_gained = cost_to_target[current_node] - cost_to_target[neighbor_nodes]
    can_enter = (occupant_at_node[neighbor_nodes] < 0) & np.isfinite(distance_gained)
    if not can_enter.any():
        return None

    # Of the agents nearby, what weighted # last moved in each direction?
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

    # bigger independence weighs shortest route more and herd term less.
    independence = agents.independence[agent_id]
    exponent = independence * distance_gained + (HERD_STRENGTH / independence) * herd_pull

    weights = np.zeros(len(outgoing_edges))
    weights[can_enter] = np.exp(exponent[can_enter] - exponent[can_enter].max())
    probabilities = weights / weights.sum()
    return outgoing_edges[random_generator.choice(len(outgoing_edges), p=probabilities)]


def move_agents_one_step(grid, agents, occupant_at_node, door_costs, exit_costs, tables, random_generator, step_number):
    agents.movement_budget += agents.speed_fraction * grid.cell_m

    for agent_id in random_generator.permutation(agents.num_agents):
        if agents.has_exited[agent_id]:
            continue

        if agents.heading_to_door[agent_id]:
            cost_to_target = door_costs[agents.target_door_row[agent_id]]
        else:
            cost_to_target = exit_costs[agents.target_exit_row[agent_id]]

        chosen_edge = choose_next_edge(grid, agent_id, agents, occupant_at_node, cost_to_target,
                                       tables, random_generator)
        if chosen_edge is None or agents.movement_budget[agent_id] < grid.edge_length[chosen_edge]:
            continue

        previous_node, next_node = agents.current_node[agent_id], grid.edge_to[chosen_edge]
        occupant_at_node[previous_node] = -1
        occupant_at_node[next_node] = agent_id
        agents.current_node[agent_id] = next_node
        agents.movement_budget[agent_id] -= grid.edge_length[chosen_edge]
        agents.last_move_direction[agent_id] = tables.edge_direction[chosen_edge]    

        if cost_to_target[next_node] == 0.0:
            if agents.heading_to_door[agent_id]:
                agents.heading_to_door[agent_id] = False
            else:
                agents.has_exited[agent_id] = True
                agents.exit_step[agent_id] = step_number
                occupant_at_node[next_node] = -1

def pick_by_distance(costs, random_generator):
    reachable = np.isfinite(costs)
    if not reachable.any():
        return None
    weights = np.zeros(len(costs))
    weights[reachable] = np.exp(-TARGET_CHOICE_SHARPNESS * (costs[reachable] - costs[reachable].min()))
    return random_generator.choice(len(costs), p=weights / weights.sum())


def reevaluate_targets(agents, door_costs, exit_costs, reconsidered_exit_rows, random_generator, step_number):
    # Staggered by agent number so crowd reevaluates in batches
    is_due = ~agents.has_exited & ((step_number + np.arange(agents.num_agents)) % REEVALUATE_EVERY_STEPS == 0)
    for agent_id in np.nonzero(is_due)[0]:
        node = agents.current_node[agent_id]
        if agents.heading_to_door[agent_id]:
            choice = pick_by_distance(door_costs[:, node], random_generator)
            if choice is not None:
                agents.target_door_row[agent_id] = choice
                if door_costs[choice, node] == 0.0:          # already standing in door
                    agents.heading_to_door[agent_id] = False
        else:
            choice = pick_by_distance(exit_costs[reconsidered_exit_rows, node], random_generator)
            if choice is not None:
                agents.target_exit_row[agent_id] = reconsidered_exit_rows[choice]

if __name__ == "__main__":
    grid = Grid.from_model(cell_m=0.5)
    fields_gate_open = compute_distance_fields(grid)
    door_costs = compute_door_costs(grid)

    random_generator = np.random.default_rng(RANDOM_SEED)
    agents, occupant_at_node = spawn_agents(grid, CONCERT_PROFILE, random_generator, fields_gate_open, door_costs)

    agents, occupant_at_node = spawn_agents(grid, CONCERT_PROFILE, np.random.default_rng(RANDOM_SEED), fields_gate_open, door_costs)
    # print("every agent reaches a door:", bool(np.isfinite(door_costs[:, agents.start_node].min(axis=0)).all()))
    # for title, values in (("target doors", agents.target_door_name), ("target exits", agents.target_exit_label)):
    #     names, counts = np.unique(values, return_counts=True)
    #     print(f"{title}:", dict(zip(names.tolist(), counts.tolist())))
    tables = build_movement_tables(grid)
    reconsidered_exit_rows = np.array([fields_gate_open.exit_labels.index(label) for label in MAIN_ENTRANCE_LABELS])
    for step_number in range(1, 5001):
        reevaluate_targets(agents, door_costs, fields_gate_open.cost_by_exit, reconsidered_exit_rows,
                           random_generator, step_number)
        move_agents_one_step(grid, agents, occupant_at_node, door_costs, fields_gate_open.cost_by_exit,
                             tables, random_generator, step_number)
        if agents.has_exited.all():
            break
    for label in (MAIN_ENTRANCE_LABELS):
        for step_number in range(1, 5001):
            reevaluate_targets(agents, door_costs, fields_gate_open.cost_by_exit, reconsidered_exit_rows,
                            random_generator, step_number)
            move_agents_one_step(grid, agents, occupant_at_node, door_costs, fields_gate_open.cost_by_exit,
                                tables, random_generator, step_number)
            if agents.has_exited.all():
                break
    time_step_seconds = grid.cell_m / MAX_WALKING_SPEED_M_PER_S
    print(f"{int(agents.has_exited.sum())} of {agents.num_agents} out, last exit at step "
          f"{int(agents.exit_step.max())} ({agents.exit_step.max() * time_step_seconds:.0f} s)")