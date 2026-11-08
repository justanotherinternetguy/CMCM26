from __future__ import annotations

import os

#threading shit
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import csv
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import LED_Simulation as sim
from grid import NODE_CELL, NODE_SEAT, NODE_STAIR, NODE_WALKWAY, BALCONY, FIRST_FLOOR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out", "opt")

CANDIDATE_EXITS_PER_SPOT = 3
LED_BAND_HALF_WIDTH_M = 1.0      # assume: a strip is visible / "under foot" within 1 m of its line
MAIN_ENTRANCE_GROUP = "F"

# hand pick spots in aisles, cross aisles, corridors, and lobby..
MANAGER_SPOTS = (
    ("front cross W", FIRST_FLOOR, -8.0, 7.5), ("front cross E", FIRST_FLOOR, 8.0, 7.5),
    ("centre aisle W", FIRST_FLOOR, -5.0, -1.0), ("centre aisle E", FIRST_FLOOR, 5.0, -1.0),
    ("mid cross W", FIRST_FLOOR, -10.0, -2.0), ("mid cross E", FIRST_FLOOR, 10.0, -2.0),
    ("side corridor W north", FIRST_FLOOR, -17.0, 4.0), ("side corridor E north", FIRST_FLOOR, 17.0, 4.0),
    ("side corridor W", FIRST_FLOOR, -17.0, -4.0), ("side corridor E", FIRST_FLOOR, 17.0, -4.0),
    ("rear ring W", FIRST_FLOOR, -11.0, -14.5), ("rear ring E", FIRST_FLOOR, 11.0, -14.5),
    ("loge back centre", FIRST_FLOOR, 0.0, -16.5), ("lobby", FIRST_FLOOR, 0.0, -21.0),
    ("balcony W north", BALCONY, -17.5, 6.0), ("balcony E north", BALCONY, 17.5, 6.0),
    ("balcony W", BALCONY, -17.5, -2.0), ("balcony E", BALCONY, 17.5, -2.0),
    ("balcony SW", BALCONY, -11.0, -14.5), ("balcony SE", BALCONY, 11.0, -14.5),
    ("balcony rear", BALCONY, 0.0, -17.5),
)

LED_START_SPOTS = (
    ("front cross W", FIRST_FLOOR, -6.0, 7.5), ("front cross E", FIRST_FLOOR, 6.0, 7.5),
    ("centre aisle W", FIRST_FLOOR, -5.0, 0.0), ("centre aisle E", FIRST_FLOOR, 5.0, 0.0),
    ("mid cross W", FIRST_FLOOR, -11.0, -2.0), ("mid cross E", FIRST_FLOOR, 11.0, -2.0),
    ("side corridor W", FIRST_FLOOR, -17.0, 0.0), ("side corridor E", FIRST_FLOOR, 17.0, 0.0),
    ("rear ring W", FIRST_FLOOR, -12.0, -14.5), ("rear ring E", FIRST_FLOOR, 12.0, -14.5),
    ("loge back centre", FIRST_FLOOR, 0.0, -16.5),
    ("balcony W", BALCONY, -17.5, 0.0), ("balcony E", BALCONY, 17.5, 0.0),
    ("balcony SW", BALCONY, -11.0, -14.5), ("balcony SE", BALCONY, 11.0, -14.5),
    ("balcony rear", BALCONY, 0.0, -17.5),
)

PROFILES = {"concert": sim.CONCERT_PROFILE, "students": sim.STUDENT_PROFILE}


# ----------------------------------------------------------------------------------------- candidates --
@dataclass
class ManagerCandidate:
    id: str
    spot: str
    level: int
    x_m: float
    y_m: float
    target_exit: str

    def post(self) -> sim.CrowdManagerPost:
        return sim.CrowdManagerPost(self.id, self.level, self.x_m, self.y_m, self.target_exit)


@dataclass
class StripCandidate:
    id: str
    spot: str
    level: int
    x_m: float
    y_m: float
    target_exit: str
    length_m: float = 0.0


def exit_group(label: str) -> str:
    return MAIN_ENTRANCE_GROUP if label in sim.MAIN_ENTRANCE_LABELS else label


def snap_to_cell(scenario: sim.Scenario, level: int, x_m: float, y_m: float) -> int:
    grid = scenario.grid
    ok = np.flatnonzero((grid.node_kind == NODE_CELL) & (grid.node_level == level) & ~scenario.fields.blocked_nodes)
    return int(ok[np.argmin(np.hypot(grid.node_x[ok] - x_m, grid.node_y[ok] - y_m))])


def cheapest_exits(scenario: sim.Scenario, node: int, count: int) -> List[str]:
    costs = scenario.exit_costs[:, node]
    chosen, groups = [], set()
    for row in np.argsort(costs):
        label = scenario.fields.exit_labels[row]
        if not np.isfinite(costs[row]) or exit_group(label) in groups:
            continue
        chosen.append(label)
        groups.add(exit_group(label))
        if len(chosen) == count:
            break
    return chosen


def rank_spots_by_traffic(scenario: sim.Scenario, spots, profile, seed: int, radius_m: float) -> List[Tuple[tuple, int]]:
    grid = scenario.grid
    no_leds = sim.build_led_guidance(grid, None, scenario.fields, lit_nodes=np.zeros(grid.num_nodes, bool))
    run = sim.run_simulation(scenario, seed=seed, profile=profile, manager_posts=(), led=no_leds, record_frames=True)
    blocked = scenario.fields.blocked_nodes if scenario.fields.blocked_nodes.any() else None
    ranking = []
    for spot in spots:
        at_spot = np.zeros(grid.num_nodes, bool)
        at_spot[snap_to_cell(scenario, spot[1], spot[2], spot[3])] = True
        near = grid.node_distance_field(targets=at_spot, blocked=blocked) <= radius_m
        seen = np.zeros(run.num_agents, bool)
        for frame in run.frames:
            seen |= near[frame["node"]] & ~frame["exited"]
        ranking.append((spot, int(seen.sum())))
    return sorted(ranking, key=lambda item: -item[1])


def keep_busiest(scenario: sim.Scenario, spots, args, what: str):
    if not args.top_spots or args.top_spots >= len(spots):
        return spots
    ranking = rank_spots_by_traffic(scenario, spots, PROFILES[args.profile], args.seed0, sim.CROWD_MANAGER_RADIUS_M)
    print(f"{what} spots by people passing within {sim.CROWD_MANAGER_RADIUS_M:g} m (no managers, no LEDs):")
    for rank, (spot, count) in enumerate(ranking):
        print(f"  {'keep' if rank < args.top_spots else '    '}  {spot[0]:24s} {count:5d}")
    return [spot for spot, _ in ranking[:args.top_spots]]


def manager_candidates(scenario: sim.Scenario, exits_per_spot: int, spots=MANAGER_SPOTS) -> List[ManagerCandidate]:
    found = []
    for spot, level, x_m, y_m in spots:
        node = snap_to_cell(scenario, level, x_m, y_m)
        for label in cheapest_exits(scenario, node, exits_per_spot):
            found.append(ManagerCandidate(f"{spot} -> {label}", spot, level, x_m, y_m, label))
    return found


class StripBuilder:
    def __init__(self, scenario: sim.Scenario, half_width_m: float = LED_BAND_HALF_WIDTH_M):
        self.scenario, self.half_width_m = scenario, half_width_m
        grid = scenario.grid
        self.off_floor = np.isin(grid.node_kind, (NODE_SEAT, NODE_WALKWAY)) | scenario.fields.blocked_nodes
        self.order = np.argsort(grid.edge_from, kind="stable")
        self.first = np.searchsorted(grid.edge_from[self.order], np.arange(grid.num_nodes + 1))
        self._floor_cost: Dict[str, np.ndarray] = {}

    def floor_cost(self, exit_label: str) -> np.ndarray:
        if exit_label not in self._floor_cost:
            grid = self.scenario.grid
            self._floor_cost[exit_label] = grid.node_distance_field(targets=grid.exit_mask(exit_label),
                                                                    blocked=self.off_floor)
        return self._floor_cost[exit_label]

    def line(self, candidate: StripCandidate) -> Tuple[np.ndarray, float]:
        grid = self.scenario.grid
        cost = self.floor_cost(candidate.target_exit)
        node = snap_to_cell(self.scenario, candidate.level, candidate.x_m, candidate.y_m)
        if not np.isfinite(cost[node]):
            return np.empty(0, int), 0.0
        path, length = [node], 0.0
        while cost[node] > 1e-9 and len(path) < grid.num_nodes:
            edges = self.order[self.first[node]:self.first[node + 1]]
            edges = edges[~self.off_floor[grid.edge_to[edges]]]
            best = edges[np.argmin(cost[grid.edge_to[edges]])]
            if cost[grid.edge_to[best]] >= cost[node]:
                break
            node = int(grid.edge_to[best])
            length += float(grid.edge_length[best])
            path.append(node)
        return np.array(path), length

    def lit_nodes(self, candidate: StripCandidate) -> np.ndarray:
        grid = self.scenario.grid
        path, candidate.length_m = self.line(candidate)
        on_line = np.zeros(grid.num_nodes, bool)
        on_line[path] = True
        if self.half_width_m <= 0 or not on_line.any():
            return on_line
        near = grid.node_distance_field(targets=on_line, blocked=self.off_floor) <= self.half_width_m
        return near & ~self.off_floor


def strip_candidates(scenario: sim.Scenario, builder: StripBuilder, exits_per_spot: int,
                     spots=LED_START_SPOTS) -> Tuple[List[StripCandidate], Dict[str, np.ndarray]]:
    found, lit = [], {}
    for spot, level, x_m, y_m in spots:
        node = snap_to_cell(scenario, level, x_m, y_m)
        for label in cheapest_exits(scenario, node, exits_per_spot):
            candidate = StripCandidate(f"{spot} -> {label}", spot, level, x_m, y_m, label)
            nodes = builder.lit_nodes(candidate)
            if nodes.any():
                found.append(candidate)
                lit[candidate.id] = nodes
    return found, lit


STATE: dict = {}


def setup_state(args) -> None:
    scenario = sim.build_scenario(closed_exit_labels=args.closed_exit or ())
    builder = StripBuilder(scenario, args.band_half_width)
    manager_spots, strip_spots = MANAGER_SPOTS, LED_START_SPOTS
    if args.mode in ("managers", "candidates"):
        manager_spots = keep_busiest(scenario, MANAGER_SPOTS, args, "manager")
    if args.mode in ("leds", "candidates"):
        strip_spots = keep_busiest(scenario, LED_START_SPOTS, args, "LED start")
    managers = manager_candidates(scenario, args.exits_per_spot, manager_spots)
    strips, strip_lit = strip_candidates(scenario, builder, args.exits_per_spot, strip_spots)
    STATE.update(scenario=scenario, builder=builder, profile=PROFILES[args.profile],
                 managers={c.id: c for c in managers}, strips={c.id: c for c in strips}, strip_lit=strip_lit,
                 keep_painted_leds=args.keep_painted_leds)
    sim.place_crowd_managers(scenario.grid, scenario.fields, [c.post() for c in managers],
                             cache=scenario.manager_cache)


def add_external(kind: str, items: Sequence[dict]) -> List[str]:
    ids = []
    for item in items:
        if kind == "managers":
            candidate = ManagerCandidate(**{k: item[k] for k in ("id", "spot", "level", "x_m", "y_m", "target_exit")})
            STATE["managers"].setdefault(candidate.id, candidate)
        else:
            candidate = StripCandidate(**{k: item[k] for k in ("id", "spot", "level", "x_m", "y_m", "target_exit")})
            if candidate.id not in STATE["strips"]:
                STATE["strips"][candidate.id] = candidate
                STATE["strip_lit"][candidate.id] = STATE["builder"].lit_nodes(candidate)
        ids.append(candidate.id)
    return ids


def current_manager_ids() -> List[str]:
    return add_external("managers", [dict(id=f"{p.name} -> {p.target_exit_label}", spot=p.name, level=p.level,
                                          x_m=p.x_m, y_m=p.y_m, target_exit=p.target_exit_label)
                                     for p in sim.CROWD_MANAGER_POSTS])


def led_guidance(strip_ids: Sequence[str]) -> sim.LedGuidance:
    scenario = STATE["scenario"]
    grid, fields = scenario.grid, scenario.fields
    lit = np.zeros(grid.num_nodes, bool)
    target_row = np.full(grid.num_nodes, -1, np.int32)        # -1: point to the nearest exit (painted LEDs)
    if STATE["keep_painted_leds"] and scenario.paint is not None:
        lit |= sim.node_is_lit(scenario.paint, grid)
    for strip_id in strip_ids:
        nodes = STATE["strip_lit"][strip_id]
        lit |= nodes
        target_row[nodes] = fields.exit_labels.index(STATE["strips"][strip_id].target_exit)
    return sim.build_led_guidance(grid, None, fields, lit_nodes=lit, target_exit_row_at_node=target_row)


Config = Tuple[Tuple[str, ...], Tuple[str, ...]]

def run_one(task: Tuple[Config, int]) -> Tuple[Config, int, dict]:
    (manager_ids, strip_ids), seed = task
    posts = [STATE["managers"][i].post() for i in manager_ids]
    result = sim.run_simulation(STATE["scenario"], seed=seed, profile=STATE["profile"], manager_posts=posts,
                                led=led_guidance(strip_ids))
    cap = sim.MAX_STEPS * STATE["scenario"].grid.cell_m / sim.MAX_WALKING_SPEED_M_PER_S
    summary = result.summary()
    for key in ("t100", "t95", "t90"):
        summary[key] = min(summary[key], cap)
    return (manager_ids, strip_ids), seed, summary


class Evaluator:
    def __init__(self, args, pool):
        self.args, self.pool = args, pool
        self.cache: Dict[Tuple[Config, int], dict] = {}
        self.seeds = list(range(args.seed0, args.seed0 + args.confirm_seeds))
        self.screen_seeds = self.seeds[:args.screen_seeds]
        self.num_runs = 0

    def run(self, configs: Sequence[Config], seeds: Sequence[int]) -> None:
        tasks = [(config, seed) for config in dict.fromkeys(configs) for seed in seeds
                 if (config, seed) not in self.cache]
        if not tasks:
            return
        started = time.time()
        for config, seed, summary in self.pool.imap_unordered(run_one, tasks, chunksize=1):
            self.cache[(config, seed)] = summary
        self.num_runs += len(tasks)
        print(f"    {len(tasks)} runs in {time.time() - started:.0f} s", flush=True)

    def cost(self, config: Config) -> float:
        metres = sum(STATE["strips"][i].length_m for i in config[1])
        return self.args.cost_per_item * (len(config[0]) + len(config[1])) + self.args.cost_per_metre * metres

    def stats(self, config: Config, seeds: Sequence[int]) -> dict:
        rows = [self.cache[(config, seed)] for seed in seeds]
        values = np.array([row[self.args.metric] for row in rows])
        exit_totals: Dict[str, float] = {}
        for row in rows:
            for label, count in row["exit_counts"].items():
                exit_totals[label] = exit_totals.get(label, 0) + count / len(rows)
        half_ci = 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        return dict(mean=float(values.mean()), ci95=float(half_ci), score=float(values.mean()) + self.cost(config),
                    t100=float(np.mean([r["t100"] for r in rows])), t95=float(np.mean([r["t95"] for r in rows])),
                    t90=float(np.mean([r["t90"] for r in rows])),
                    told_by_manager=float(np.mean([r["told_by_manager"] for r in rows])),
                    exit_counts={k: round(v, 1) for k, v in sorted(exit_totals.items())}, num_seeds=len(rows))

    def best_of(self, configs: Sequence[Config]) -> Tuple[Config, dict, List[Config]]:
        configs = list(dict.fromkeys(configs))
        self.run(configs, self.screen_seeds)
        ranked = sorted(configs, key=lambda c: self.stats(c, self.screen_seeds)["score"])
        finalists = ranked[:self.args.confirm_top]
        self.run(finalists, self.seeds)
        best = min(finalists, key=lambda c: self.stats(c, self.seeds)["score"])
        return best, self.stats(best, self.seeds), ranked


def spot_of(kind: str, item_id: str) -> str:
    table = STATE["managers"] if kind == "managers" else STATE["strips"]
    return table[item_id].spot


def make_config(kind: str, items: Sequence[str], fixed: Sequence[str]) -> Config:
    return (tuple(items), tuple(fixed)) if kind == "managers" else (tuple(fixed), tuple(items))


def greedy(kind: str, evaluator: Evaluator, initial: List[str], fixed: List[str], pool_ids: List[str],
           max_items: int, swap_passes: int, shortlist: int = 0) -> Tuple[List[str], List[dict]]:
    selected = list(initial)
    config = make_config(kind, selected, fixed)
    evaluator.run([config], evaluator.seeds)
    steps = [dict(step=0, added="(start)", selected=list(selected), **evaluator.stats(config, evaluator.seeds))]
    print_step(kind, steps[-1])

    def allowed(chosen: Sequence[str]) -> List[str]:
        used = {spot_of(kind, i) for i in chosen}
        return [i for i in pool_ids if i not in chosen and spot_of(kind, i) not in used]

    for step in range(1, max_items + 1):
        options = allowed(selected)
        if not options:
            break
        print(f"step {step}: trying {len(options)} candidates", flush=True)
        configs = {make_config(kind, selected + [i], fixed): i for i in options}
        best, stats, ranked = evaluator.best_of(list(configs))
        if step == 1 and shortlist and len(pool_ids) > shortlist:
            pool_ids = [configs[c] for c in ranked[:shortlist]]
            print(f"  shortlist for later steps: {pool_ids}")
        selected.append(configs[best])
        steps.append(dict(step=step, added=configs[best], selected=list(selected), **stats))
        print_step(kind, steps[-1])

    num_initial = len(initial)
    for swap_pass in range(swap_passes):
        current = make_config(kind, selected, fixed)
        current_score = evaluator.stats(current, evaluator.seeds)["score"]
        changed = False
        for position in range(num_initial, len(selected)):
            others = selected[:position] + selected[position + 1:]
            configs = {make_config(kind, others[:position] + [i] + others[position:], fixed): i
                       for i in allowed(others)}
            configs[current] = selected[position]
            print(f"swap pass {swap_pass + 1}, item {position + 1}: trying {len(configs) - 1} replacements", flush=True)
            best, stats, _ = evaluator.best_of(list(configs))
            if best != current and stats["score"] < current_score - 1e-9:
                print(f"  swap {selected[position]!r} -> {configs[best]!r}: {current_score:.1f} -> {stats['score']:.1f}")
                selected = list(best[0] if kind == "managers" else best[1])
                current, current_score, changed = best, stats["score"], True
        steps.append(dict(step=f"swap {swap_pass + 1}", added="(swaps)", selected=list(selected),
                          **evaluator.stats(current, evaluator.seeds)))
        print_step(kind, steps[-1])
        if not changed:
            break
    return selected, steps


def print_step(kind: str, step: dict) -> None:
    metres = ""
    if kind == "leds":
        metres = f", {sum(STATE['strips'][i].length_m for i in step['selected']):.0f} m of LED"
    print(f"  [{step['step']}] {step['added']}: {len(step['selected'])} items{metres} | mean {step['mean']:.1f} s "
          f"± {step['ci95']:.1f} (t100 {step['t100']:.0f}, t95 {step['t95']:.0f}, t90 {step['t90']:.0f}) "
          f"| exits {step['exit_counts']}", flush=True)


EXIT_GROUP_COLORS = {"F": "#2a78d6", "PW": "#eb6834", "PE": "#1baf7a", "SW": "#eda100", "SE": "#e87ba4",
                     "BW": "#008300", "BE": "#7a5cd6"}   # fixed order, colour follows the exit not the rank
INK, MUTED, SURFACE = "#2b2b29", "#8a897f", "#fcfcfb"


def describe(manager_ids: Sequence[str], strip_ids: Sequence[str]) -> dict:
    return dict(managers=[asdict(STATE["managers"][i]) for i in manager_ids],
                led_strips=[asdict(STATE["strips"][i]) for i in strip_ids],
                led_metres=round(sum(STATE["strips"][i].length_m for i in strip_ids), 1))


def plot_curve(kind: str, steps: List[dict], metric: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    greedy_steps = [s for s in steps if isinstance(s["step"], int)]
    x = [len(s["selected"]) for s in greedy_steps]
    y = [s["mean"] for s in greedy_steps]
    error = [s["ci95"] for s in greedy_steps]
    fig, axis = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    axis.axhline(180, color=MUTED, lw=1, ls=(0, (4, 3)))
    axis.text(x[-1], 181, "180 s target", color=MUTED, ha="right", va="bottom", fontsize=9)
    axis.errorbar(x, y, yerr=error, color="#2a78d6", lw=2, marker="o", ms=6, capsize=3,
                  mec=SURFACE, mew=1.5)
    if kind == "leds":
        metres = [sum(STATE["strips"][i].length_m for i in s["selected"]) for s in greedy_steps]
        axis.set_xticks(x, [f"{n}\n{m:.0f} m" for n, m in zip(x, metres)])
        axis.set_xlabel("LED strips (and total length)", color=INK)
    else:
        axis.set_xticks(x)
        axis.set_xlabel("crowd managers", color=INK)
        required = int(np.ceil(len(STATE["scenario"].grid.spawn) / sim.ATTENDEES_PER_CROWD_MANAGER))
        axis.axvline(required, color=MUTED, lw=1, ls=":")
        axis.text(required, axis.get_ylim()[1], f" legal minimum ({required})", color=MUTED, va="top", fontsize=9)
    axis.set_ylabel(f"mean {metric} [s] (95% CI)", color=INK)
    axis.set_title(f"Greedy {kind} selection", color=INK, loc="left")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=INK)
    axis.grid(axis="y", color="#e4e3dd", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_layout(manager_ids: Sequence[str], strip_ids: Sequence[str], path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenario = STATE["scenario"]
    grid = scenario.grid
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=SURFACE)
    show_labels = len(manager_ids) + len(strip_ids) <= 24        # the all-candidates view would be unreadable
    for axis, level, name in ((axes[0], FIRST_FLOOR, "first floor"), (axes[1], BALCONY, "balcony")):
        axis.set_facecolor(SURFACE)
        on_level = grid.node_level == level
        cells = on_level & (grid.node_kind == NODE_CELL)
        seats = on_level & (grid.node_kind == NODE_SEAT)
        axis.scatter(grid.node_x[cells], grid.node_y[cells], s=7, marker="s", c="#e4e3dd", lw=0)
        axis.scatter(grid.node_x[seats], grid.node_y[seats], s=3, marker="s", c="#b9b8b0", lw=0)
        for strip_id in strip_ids:
            strip = STATE["strips"][strip_id]
            nodes = STATE["strip_lit"][strip_id] & on_level
            color = EXIT_GROUP_COLORS[exit_group(strip.target_exit)]
            axis.scatter(grid.node_x[nodes], grid.node_y[nodes], s=7, marker="s", c=color, lw=0)
            if show_labels and strip.level == level:
                axis.annotate(f"LED → {strip.target_exit}", (strip.x_m, strip.y_m), fontsize=8, color=INK,
                              xytext=(4, 4), textcoords="offset points")
        for manager_id in manager_ids:
            manager = STATE["managers"][manager_id]
            if manager.level != level:
                continue
            color = EXIT_GROUP_COLORS[exit_group(manager.target_exit)]
            axis.scatter([manager.x_m], [manager.y_m], s=90, marker="o", c=color, edgecolors=INK, lw=1.2, zorder=5)
            if show_labels:
                axis.annotate(f"CM → {manager.target_exit}", (manager.x_m, manager.y_m), fontsize=8, color=INK,
                              xytext=(6, -10), textcoords="offset points", zorder=6)
        for label in scenario.fields.exit_labels:
            nodes = scenario.grid._node_mask(grid.exit_mask(label)) & on_level
            if nodes.any():
                axis.annotate(label, (grid.node_x[nodes].mean(), grid.node_y[nodes].mean()), fontsize=9,
                              color=INK, weight="bold", ha="center", va="center")
        axis.set_title(name, color=INK)
        axis.set_aspect("equal")
        axis.axis("off")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=f"to {g}") for g, c in EXIT_GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_outputs(kind: str, args, selected_managers, selected_strips, steps, evaluator) -> str:
    folder = os.path.join(OUT_DIR, f"{kind}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(folder, exist_ok=True)
    final = make_config("managers", selected_managers, selected_strips)
    result = dict(kind=kind, args=vars(args), seeds=evaluator.seeds, num_runs=evaluator.num_runs,
                  final=evaluator.stats(final, evaluator.seeds), steps=steps,
                  **describe(selected_managers, selected_strips))
    with open(os.path.join(folder, "result.json"), "w") as handle:
        json.dump(result, handle, indent=1, default=float)
    with open(os.path.join(folder, "steps.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "added", "num_items", "led_metres", f"mean_{args.metric}", "ci95", "t100", "t95",
                         "t90", "told_by_manager", "exit_counts"])
        for s in steps:
            metres = sum(STATE["strips"][i].length_m for i in s["selected"]) if kind == "leds" else 0
            writer.writerow([s["step"], s["added"], len(s["selected"]), round(metres, 1), round(s["mean"], 2),
                             round(s["ci95"], 2), round(s["t100"], 1), round(s["t95"], 1), round(s["t90"], 1),
                             round(s["told_by_manager"], 1), json.dumps(s["exit_counts"])])
    plot_curve(kind, steps, args.metric, os.path.join(folder, "curve.png"))
    plot_layout(selected_managers, selected_strips, os.path.join(folder, "layout.png"),
                f"{kind}: mean {args.metric} {result['final']['mean']:.0f} s ± {result['final']['ci95']:.0f}")
    return folder


def load_items(spec: str, kind: str) -> List[str]:
    if spec == "none":
        return []
    if spec == "current":
        if kind != "managers":
            raise SystemExit("--leds takes none or a result.json (add --keep-painted-leds for the painted LEDs)")
        return current_manager_ids()
    with open(spec) as handle:
        data = json.load(handle)
    return add_external(kind, data["managers"] if kind == "managers" else data["led_strips"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("candidates", "managers", "leds", "evaluate"))
    parser.add_argument("--profile", choices=tuple(PROFILES), default="concert")
    parser.add_argument("--metric", choices=("t100", "t95", "t90"), default="t100")
    parser.add_argument("--max-items", type=int, default=8, help="greedy steps to take")
    parser.add_argument("--swap-passes", type=int, default=0, help="swap-improvement passes after the greedy steps")
    parser.add_argument("--start", default="none",
                        help="items the search starts from: none | current | <result.json>")
    parser.add_argument("--managers", default="current",
                        help="leds/evaluate mode: fixed managers, none | current | <result.json>")
    parser.add_argument("--leds", default="none", help="managers/evaluate mode: fixed LED strips, none | <result.json>")
    parser.add_argument("--config", help="evaluate mode: a result.json whose managers and strips are evaluated")
    parser.add_argument("--keep-painted-leds", action="store_true", help="also keep the LEDs painted in grid_paint")
    parser.add_argument("--closed-exit", action="append", help="close an exit (repeatable), e.g. --closed-exit PW")
    parser.add_argument("--screen-seeds", type=int, default=8)
    parser.add_argument("--confirm-seeds", type=int, default=24)
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument("--shortlist", type=int, default=20,
                        help="after step 1 only the best N single candidates stay in the pool (0 = all)")
    parser.add_argument("--seeds", type=int, help="evaluate mode: number of seeds (default --confirm-seeds)")
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--exits-per-spot", type=int, default=CANDIDATE_EXITS_PER_SPOT)
    parser.add_argument("--top-spots", type=int, default=0,
                        help="keep only the N spots the most people walk past in a run with no managers/LEDs")
    parser.add_argument("--band-half-width", type=float, default=LED_BAND_HALF_WIDTH_M)
    parser.add_argument("--cost-per-item", type=float, default=0.0, help="seconds added to the score per item")
    parser.add_argument("--cost-per-metre", type=float, default=0.0, help="seconds added per metre of LED")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if args.mode == "evaluate" and args.seeds:
        args.confirm_seeds = args.screen_seeds = args.seeds
    args.screen_seeds = min(args.screen_seeds, args.confirm_seeds)
    return args


def main():
    args = parse_args()
    print("building scenario and candidates ...", flush=True)
    setup_state(args)
    print(f"{len(STATE['managers'])} manager candidates, {len(STATE['strips'])} LED strip candidates "
          f"(profile {args.profile}, closed exits {args.closed_exit or 'none'})")

    if args.mode == "candidates":
        for candidate in STATE["managers"].values():
            print(f"  manager  {candidate.id}")
        for candidate in STATE["strips"].values():
            print(f"  LED      {candidate.id:40s} {candidate.length_m:5.1f} m")
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "candidates.png")
        plot_layout(list(STATE["managers"]), list(STATE["strips"]), path, "all candidates")
        print("drawn to", path)
        return

    if args.mode == "managers" and not PROFILES[args.profile].has_crowd_managers:
        raise SystemExit(f"profile {args.profile} has no crowd managers")

    kind = args.mode
    if kind == "evaluate":
        managers = load_items(args.config, "managers") if args.config else load_items(args.managers, "managers")
        strips = load_items(args.config, "leds") if args.config else load_items(args.leds, "leds")
    elif kind == "managers":
        fixed, pool_ids = load_items(args.leds, "leds"), list(STATE["managers"])
    else:
        fixed, pool_ids = load_items(args.managers, "managers"), list(STATE["strips"])
    initial = load_items(args.start, kind) if kind != "evaluate" else []

    context = mp.get_context("fork")         # workers inherit STATE (scenario, candidates, caches)
    with context.Pool(args.workers) as pool:
        evaluator = Evaluator(args, pool)
        if kind == "evaluate":
            config = make_config("managers", managers, strips)
            evaluator.run([config], evaluator.seeds)
            stats = evaluator.stats(config, evaluator.seeds)
            print(json.dumps(dict(**describe(managers, strips), stats=stats), indent=1, default=float))
            return
        selected, steps = greedy(kind, evaluator, initial, fixed, pool_ids, args.max_items, args.swap_passes,
                               args.shortlist)
    managers, strips = (selected, fixed) if kind == "managers" else (fixed, selected)
    folder = write_outputs(kind, args, managers, strips, steps, evaluator)
    print(f"{evaluator.num_runs} simulations; results in {folder}")


if __name__ == "__main__":
    main()
