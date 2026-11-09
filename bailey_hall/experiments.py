"""Scenario runs for the write-up: every configuration over the same seeds, one JSON of per-run results.

    python experiments.py list
    python experiments.py run students concert_none concert_baseline --seeds 24
    python experiments.py table                     # summary table from out/experiments/runs.json

Scenarios are defined in SCENARIOS below. Manager and LED configurations can come from an optimize.py
result.json (`managers="out/opt/.../result.json"`), from today's posts (`"current"`), or none.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import copy
import glob
import json
import multiprocessing as mp
import time
from typing import Dict, List

import numpy as np

import LED_Simulation_writeup as sim     # LED_Simulation.py (84c507f upload) lost the run harness
import optimize as opt
from barriers import barrier_nodes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "experiments")
RUNS_PATH = os.path.join(OUT, "runs.json")
PROFILES = {"concert": sim.CONCERT_PROFILE, "students": sim.STUDENT_PROFILE}
NORMAL_READY_WINDOW_S = 30.0     # ordinary departure: people are ready to leave within 30 s of the end


def latest(pattern: str) -> str:
    found = sorted(glob.glob(os.path.join(HERE, "out", "opt", pattern, "result.json")))
    return found[-1] if found else ""


def opt_managers() -> str:
    return latest("managers6_*")


def opt_leds() -> str:
    return latest("leds6_*")


def S(profile="concert", managers="none", leds="none", closed=(), barriers="none", normal=False, note="",
      params=None):
    return dict(profile=profile, managers=managers, leds=leds, closed=tuple(closed), barriers=barriers,
                normal=normal, note=note, params=dict(params or {}))


SCENARIOS: Dict[str, dict] = {
    # Q1 ordinary departure
    "normal_egress": S(normal=True, note="concert audience, no alarm, front doors and main balcony stairs only"),
    # Q3 students
    "students": S("students", note="lecture, students know every exit, no managers"),
    "students_F2F3": S("students", closed=("F2", "F3"), note="students, F2 and F3 blocked"),
    "students_barrier": S("students", barriers="portico", note="students with corridor barriers"),
    # Q2 concert
    "concert_none": S(note="concert, no managers, no LEDs"),
    "concert_baseline": S(managers="current", note="concert, 6 managers at today's posts"),
    "concert_baseline_F2F3": S(managers="current", closed=("F2", "F3"), note="baseline managers, F2+F3 blocked"),
    "concert_baseline_barrier": S(managers="current", barriers="portico", note="baseline managers + barriers"),
    "concert_none_barrier": S(barriers="portico", note="no managers + barriers"),
    # Q4 optimised
    "concert_opt": S(managers="@managers", note="6 optimised managers"),
    "concert_opt_F2F3": S(managers="@managers", closed=("F2", "F3"), note="6 optimised managers, F2+F3 blocked"),
    "concert_opt_PW": S(managers="@managers", closed=("PW",), note="6 optimised managers, PW blocked"),
    "concert_opt_barrier": S(managers="@managers", barriers="portico", note="6 optimised managers + barriers"),
    "concert_opt_led": S(managers="@managers", leds="@leds", note="6 optimised managers + LED strips"),
    "concert_opt_led_F2F3": S(managers="@managers", leds="@leds", closed=("F2", "F3"),
                              note="managers + LEDs, F2+F3 blocked"),
    "concert_opt_led_PW": S(managers="@managers", leds="@leds", closed=("PW",), note="managers + LEDs, PW blocked"),
    "concert_opt_led_barrier": S(managers="@managers", leds="@leds", barriers="portico",
                                 note="managers + LEDs + barriers"),
    "concert_baseline_PW": S(managers="current", closed=("PW",), note="baseline managers, PW blocked"),
    # sensitivity (optimised managers, one parameter changed)
    "sens_no_response": S(managers="@managers", params={"RESPONSE_TIME_ENABLED": False},
                          note="everyone ready at the alarm"),
    "sens_herd0": S(managers="@managers", params={"HERD_STRENGTH": 0}, note="no herding"),
    "sens_herd4": S(managers="@managers", params={"HERD_STRENGTH": 4}, note="herding doubled"),
    "sens_cm_r2": S(managers="@managers", params={"CROWD_MANAGER_RADIUS_M": 2.0}, note="manager radius 2 m"),
    "sens_cm_r6": S(managers="@managers", params={"CROWD_MANAGER_RADIUS_M": 6.0}, note="manager radius 6 m"),
    "sens_view10": S(managers="@managers", params={"VIEW_RANGE_M": 10.0}, note="view range 10"),
    "sens_view25": S(managers="@managers", params={"VIEW_RANGE_M": 25.0}, note="view range 25"),
    "sens_faster": S(managers="@managers", params={"SPEED_MEAN": 0.95}, note="mean speed fraction 0.95"),
}

SCEN: Dict[tuple, sim.Scenario] = {}
BUILDERS: Dict[tuple, opt.StripBuilder] = {}


def resolve(spec: str) -> str:
    return {"@managers": opt_managers(), "@leds": opt_leds()}.get(spec, spec)


def scenario_for(closed, barriers) -> sim.Scenario:
    key = (tuple(closed), barriers)
    if key not in SCEN:
        base = sim.build_scenario(closed_exit_labels=(), barriers_enabled=False)
        blocked = barrier_nodes(base.grid, barriers)
        if not blocked.any() and not closed:
            SCEN[key] = base
        else:
            grid = base.grid
            fields = sim.compute_distance_fields(grid, blocked, closed)
            door_costs = sim.compute_door_costs(grid, fields.blocked_nodes)
            door_to_exit = sim.compute_door_to_exit_costs(door_costs, fields.cost_by_exit)
            scenario = copy.copy(base)
            scenario.fields, scenario.door_costs, scenario.door_to_exit_costs = fields, door_costs, door_to_exit
            scenario.exit_goal_cost = sim.compute_exit_goal_costs(fields.cost_by_exit)
            scenario.discovery_distance = sim.compute_discovery_distance(grid, fields, blocked)
            scenario.manager_cache = {}
            SCEN[key] = scenario
    return SCEN[key]


def normal_version(scenario: sim.Scenario) -> sim.Scenario:
    """Ordinary departure: the stage-side balcony doors are not used and nobody considers the portico."""
    scenario = copy.copy(scenario)
    door_costs = scenario.door_costs.copy()
    for name in ("balcony_exit_NW", "balcony_exit_NE"):
        door_costs[sim.KNOWN_AUDITORIUM_DOOR_NAMES.index(name)] = np.inf
    scenario.door_costs = door_costs
    scenario.door_to_exit_costs = sim.compute_door_to_exit_costs(door_costs, scenario.fields.cost_by_exit)
    scenario.uninformed_exit_rows = np.array([scenario.fields.exit_labels.index(label)
                                              for label in sim.MAIN_ENTRANCE_LABELS])
    return scenario


def load_posts(spec: str) -> List[sim.CrowdManagerPost]:
    spec = resolve(spec)
    if spec == "none":
        return []
    if spec == "current":
        return list(sim.CROWD_MANAGER_POSTS)
    with open(spec) as handle:
        return [sim.CrowdManagerPost(m["id"], m["level"], m["x_m"], m["y_m"], m["target_exit"])
                for m in json.load(handle)["managers"]]


def led_for(scenario: sim.Scenario, key: tuple, spec: str) -> sim.LedGuidance:
    grid, fields = scenario.grid, scenario.fields
    spec = resolve(spec)
    lit = np.zeros(grid.num_nodes, bool)
    target = np.full(grid.num_nodes, -1, np.int32)
    if spec != "none":
        if key not in BUILDERS:
            BUILDERS[key] = opt.StripBuilder(scenario)
        with open(spec) as handle:
            strips = json.load(handle)["led_strips"]
        for item in strips:
            candidate = opt.StripCandidate(**{k: item[k] for k in ("id", "spot", "level", "x_m", "y_m", "target_exit")})
            nodes = BUILDERS[key].lit_nodes(candidate)
            lit |= nodes
            target[nodes] = fields.exit_labels.index(candidate.target_exit)
    return sim.build_led_guidance(grid, None, fields, lit_nodes=lit, target_exit_row_at_node=target)


def run_task(task):
    name, seed = task
    spec = SCENARIOS[name]
    key = (spec["closed"], spec["barriers"])
    scenario = scenario_for(*key)
    led = led_for(scenario, key, spec["leds"])
    original_response = sim.assign_response_times
    params = dict(spec.get("params", {}))
    profile = PROFILES[spec["profile"]]
    if "SPEED_MEAN" in params:
        profile = copy.copy(profile)
        profile.speed_fraction_mean = params.pop("SPEED_MEAN")
    saved = {name: getattr(sim, name) for name in params}
    for name, value in params.items():
        setattr(sim, name, value)
    if "CROWD_MANAGER_RADIUS_M" in params:
        scenario = copy.copy(scenario)
        scenario.manager_cache = {}
    if spec["normal"]:
        scenario = normal_version(scenario)
        sim.assign_response_times = lambda grid, rng: rng.uniform(0.0, NORMAL_READY_WINDOW_S, len(grid.spawn))
    try:
        result = sim.run_simulation(scenario, seed=seed, profile=profile,
                                    manager_posts=load_posts(spec["managers"]), led=led)
    finally:
        sim.assign_response_times = original_response
        for name, value in saved.items():
            setattr(sim, name, value)
    labels = result.exit_label_by_agent
    clear = {str(label): float(result.exit_times_s[labels == label].max()) for label in np.unique(labels) if label}
    return name, seed, dict(t100=result.evacuation_time_s, t95=result.time_95_s, t90=result.time_90_s,
                            num_out=result.num_out, exit_counts=result.exit_counts, exit_clear_s=clear,
                            told=result.num_told_by_manager, led_guided=result.num_led_guided,
                            noticed_closed=result.num_noticed_closed_exit)


def load_runs() -> dict:
    if os.path.exists(RUNS_PATH):
        with open(RUNS_PATH) as handle:
            return json.load(handle)
    return {}


def summarise(rows: List[dict]) -> dict:
    t = np.array([r["t100"] for r in rows])
    counts, clear = {}, {}
    for r in rows:
        for k, v in r["exit_counts"].items():
            counts[k] = counts.get(k, 0) + v / len(rows)
        for k, v in r["exit_clear_s"].items():
            clear.setdefault(k, []).append(v)
    return dict(n=len(t), mean=t.mean(), median=float(np.median(t)), sd=t.std(ddof=1) if len(t) > 1 else 0.0,
                ci95=1.96 * t.std(ddof=1) / np.sqrt(len(t)) if len(t) > 1 else 0.0,
                min=t.min(), max=t.max(), within_180=int((t <= 180).sum()),
                t95=np.mean([r["t95"] for r in rows]), t90=np.mean([r["t90"] for r in rows]),
                told=np.mean([r["told"] for r in rows]), led_guided=np.mean([r["led_guided"] for r in rows]),
                noticed_closed=np.mean([r["noticed_closed"] for r in rows]),
                exit_counts={k: round(v, 1) for k, v in sorted(counts.items())},
                exit_clear_s={k: round(float(np.mean(v)), 1) for k, v in sorted(clear.items())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("list", "run", "table"))
    parser.add_argument("names", nargs="*")
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.mode == "list":
        for name, spec in SCENARIOS.items():
            print(f"{name:28s} {spec['note']}")
        return
    runs = load_runs()
    if args.mode == "run":
        names = args.names or list(SCENARIOS)
        for name in names:                      # build every scenario before forking
            spec = SCENARIOS[name]
            key = (spec["closed"], spec["barriers"])
            led_for(scenario_for(*key), key, spec["leds"])
            print(f"{name}: managers={resolve(spec['managers'])} leds={resolve(spec['leds'])}", flush=True)
        seeds = range(args.seed0, args.seed0 + args.seeds)
        tasks = [(n, s) for n in names for s in seeds if str(s) not in runs.get(n, {})]
        started = time.time()
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, (name, seed, row) in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
                runs.setdefault(name, {})[str(seed)] = row
                if (i + 1) % 8 == 0 or i + 1 == len(tasks):
                    with open(RUNS_PATH, "w") as handle:
                        json.dump(runs, handle, default=float)
                    print(f"  {i + 1}/{len(tasks)} runs, {time.time() - started:.0f} s", flush=True)
    table = {name: summarise(list(rows.values())) for name, rows in runs.items()}
    with open(os.path.join(OUT, "summary.json"), "w") as handle:
        json.dump(table, handle, indent=1, default=float)
    for name in SCENARIOS:
        if name in table:
            s = table[name]
            print(f"{name:28s} n={s['n']:2d} mean {s['mean']:6.1f} sd {s['sd']:5.1f} [{s['min']:.0f}-{s['max']:.0f}] "
                  f"<=180: {s['within_180']}/{s['n']}  t95 {s['t95']:.0f} t90 {s['t90']:.0f} told {s['told']:.0f}")
            print(f"{'':28s} counts {s['exit_counts']}")
            print(f"{'':28s} clear  {s['exit_clear_s']}")


if __name__ == "__main__":
    main()
