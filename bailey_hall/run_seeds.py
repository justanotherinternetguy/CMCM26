from types import SimpleNamespace

import numpy as np

import LED_Simulation as sim  

SEEDS = range(10)
PROFILES = (sim.CONCERT_PROFILE, sim.STUDENT_PROFILE)


def build_static() -> SimpleNamespace:
    """Everything that does not depend on the seed or the profile."""
    grid = sim.Grid.from_model(cell_m=sim.CELL_M)
    paint = sim.GridPaint.load(sim.PAINT_PATH, grid)
    barrier_nodes = (paint.node_is_barrier(grid) if sim.BARRIERS_ENABLED
                     else np.zeros(grid.num_nodes, bool))

    fields = sim.compute_distance_fields(grid, barrier_nodes, sim.CLOSED_EXIT_LABELS)
    exit_costs = fields.cost_by_exit
    door_costs = sim.compute_door_costs(grid, fields.blocked_nodes)
    door_to_exit_costs = sim.compute_door_to_exit_costs(door_costs, exit_costs)

    labels = fields.exit_labels
    main_entrance_rows = np.array([labels.index(l) for l in sim.MAIN_ENTRANCE_LABELS])
    side_exit_rows = np.array([labels.index(l) for l in sim.SIDE_EXIT_LABELS if l in labels], dtype=int)

    exit_node_masks = [sim.mask_to_node_mask(grid, grid.exit_mask(l)) for l in labels]
    door_node_masks = [sim.mask_to_node_mask(grid, grid.door_mask(d)) for d in sim.KNOWN_AUDITORIUM_DOOR_NAMES]

    return SimpleNamespace(
        grid=grid,
        fields=fields,
        exit_costs=exit_costs,
        door_costs=door_costs,
        door_to_exit_costs=door_to_exit_costs,
        exit_goal_cost=sim.compute_exit_goal_costs(exit_costs),
        nearby_exit_rows=sim.nearby_open_exit_rows(grid, fields),
        led=sim.build_led_guidance(grid, paint, fields),
        managers=sim.place_crowd_managers(grid, fields, sim.CROWD_MANAGER_POSTS),
        seat_neighbors=sim.immediate_seat_neighbors(grid),
        tables=sim.build_movement_tables(grid),
        all_exit_rows=np.arange(len(labels)),
        uninformed_exit_rows=np.concatenate([main_entrance_rows, side_exit_rows]),
        discovery_distance=sim.compute_discovery_distance(grid, fields, barrier_nodes),
        door_blocked_by_exit=np.array([[bool((d & e).any()) for e in exit_node_masks]
                                       for d in door_node_masks]),
        step_seconds=grid.cell_m / sim.MAX_WALKING_SPEED_M_PER_S,
    )


def run(profile, seed: int, s: SimpleNamespace):
    grid = s.grid
    rng = np.random.default_rng(seed)
    agents, occupant_at_node = sim.spawn_agents(grid, profile, rng, s.fields, s.door_costs,
                                                s.door_to_exit_costs, s.led)
    occupant_at_node[s.fields.closed_nodes] = -2

    if sim.RESPONSE_TIME_ENABLED:
        agents.response_time = sim.assign_response_times(
            grid, np.random.default_rng(np.random.SeedSequence([seed, 1])))
    manager_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    managers = s.managers if profile.has_crowd_managers else None

    for step in range(1, sim.MAX_STEPS + 1):
        sim.discover_closed_exits(agents, s.discovery_distance, s.door_blocked_by_exit, s.door_costs,
                                  s.door_to_exit_costs, s.exit_costs, s.all_exit_rows,
                                  s.uninformed_exit_rows, s.fields.exit_labels)
        sim.reevaluate_targets(agents, s.door_costs, s.door_to_exit_costs, s.exit_costs, s.exit_goal_cost,
                               s.all_exit_rows, s.uninformed_exit_rows, occupant_at_node, rng, step,
                               s.nearby_exit_rows)
        sim.apply_crowd_managers(agents, managers, s.door_costs, s.door_to_exit_costs, s.exit_costs,
                                 manager_rng, s.step_seconds)
        sim.move_agents_one_step(grid, agents, occupant_at_node, s.door_costs, s.exit_costs,
                                 s.exit_goal_cost, s.led, s.tables, rng, step, s.seat_neighbors,
                                 s.nearby_exit_rows)
        if agents.has_exited.all():
            break

    t90 = sim.time_to_fraction_out(agents, 0.9, s.step_seconds)
    t100 = sim.time_to_fraction_out(agents, 1.0, s.step_seconds)
    return (t90[1] if t90 else np.nan), (t100[1] if t100 else np.nan)


if __name__ == "__main__":
    static = build_static()
    for profile in PROFILES:
        print(f"\n{profile.name}")
        results = []
        for seed in SEEDS:
            t90, t100 = run(profile, seed, static)
            results.append((t90, t100))
            print(f"  seed {seed}: 90% = {t90:.0f} s, 100% = {t100:.0f} s")
        arr = np.array(results)
        print(f"  mean:   90% = {np.nanmean(arr[:, 0]):.0f} s, 100% = {np.nanmean(arr[:, 1]):.0f} s")