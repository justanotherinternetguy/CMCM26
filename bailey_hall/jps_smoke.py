"""
JuPedSim smoke test: load the flattened single-plane layout, put one agent in front of every
seat (1,329 people), send each to the exit with the shortest walking path and run until the
building is empty.  This checks that the geometry is simulation-ready; it is NOT a calibrated
egress model (no pre-movement time, no crowd managers, everybody knows the nearest exit).

  python bailey_hall/jps_smoke.py [n_agents] [--exits name1,name2,...] [--max-time seconds]

Speed: about 0.1 ms per agent per 10 ms time step, i.e. a full house (1,314 agents) runs at
roughly 1/15 real time; use a few hundred agents while developing.

Writes bailey_hall/out/smoke_trajectories.sqlite and prints exit usage + evacuation time.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import jupedsim as jps
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon

from model import FT, Model

RADIUS = 0.15   # agent body radius [m]; row walkways are only ~0.4 m clear
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def to_m(g):
    return affinity.scale(g, FT, FT, origin=(0, 0))


def main(n_agents: int | None = None, only_exits: list[str] | None = None, seed: int = 1, max_time: float = 600.0):
    OUT.mkdir(exist_ok=True)
    m = Model()
    flat = m.flattened()
    walk = to_m(flat["walkable"])
    if isinstance(walk, MultiPolygon):
        raise SystemExit(f"walkable area is not connected ({len(walk.geoms)} parts)")
    walk = walk.simplify(0.03)   # 3 cm tolerance: fewer wall segments -> faster
    exits = [(n, to_m(g).buffer(-0.05).intersection(walk)) for n, g in flat["exits"]]
    exits = [(n, g) for n, g in exits if (only_exits is None or n in only_exits) and g.area > 0.2]
    print(f"geometry: {walk.area:.0f} m2, {len(exits)} exits: {[n for n, _ in exits]}", flush=True)

    spawn = [(s["x"] * FT, s["y"] * FT, s["section"]) for s in flat["spawn"]]
    boundary = walk.boundary
    ok = [s for s in spawn if walk.contains(Point(s[0], s[1])) and boundary.distance(Point(s[0], s[1])) > RADIUS + 0.02]
    if len(ok) < len(spawn):
        print(f"dropped {len(spawn) - len(ok)} spawn points closer than {RADIUS + 0.02:.2f} m to a wall", flush=True)
    spawn = ok
    if n_agents:
        import random
        random.Random(seed).shuffle(spawn)
        spawn = spawn[:n_agents]
    print(f"agents: {len(spawn)}", flush=True)

    traj = OUT / f"smoke_trajectories_{len(spawn)}.sqlite"
    sim = jps.Simulation(model=jps.CollisionFreeSpeedModel(),
                         geometry=walk,
                         trajectory_writer=jps.SqliteTrajectoryWriter(output_file=traj))
    exit_ids = {n: sim.add_exit_stage(g) for n, g in exits}
    journeys = {n: sim.add_journey(jps.JourneyDescription([eid])) for n, eid in exit_ids.items()}

    # route choice: shortest walking path (navigation mesh) to any exit
    router = jps.RoutingEngine(walk)
    targets = {n: g.representative_point() for n, g in exits}

    def path_len(src, dst):
        pts = router.compute_waypoints(src, (dst.x, dst.y))
        return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))

    params = jps.CollisionFreeSpeedModelAgentParameters
    speed_kw = "desired_speed" if "desired_speed" in params.__init__.__code__.co_varnames else "v0"
    usage = {n: 0 for n in exits and exit_ids}
    t0 = time.time()
    for (x, y, section) in spawn:
        best = min(targets, key=lambda n: path_len((x, y), targets[n]))
        usage[best] += 1
        kw = {speed_kw: 1.2, "radius": RADIUS}
        sim.add_agent(params(journey_id=journeys[best], stage_id=exit_ids[best], position=(x, y), **kw))
    print(f"routing + spawning took {time.time() - t0:.1f}s; planned exit usage: {usage}", flush=True)

    t0 = time.time()
    max_iter = int(max_time / sim.delta_time())
    n_start = sim.agent_count()
    while sim.agent_count() > 0 and sim.iteration_count() < max_iter:
        sim.iterate()
        if sim.iteration_count() % int(20 / sim.delta_time()) == 0:
            print(f"  t={sim.elapsed_time():5.0f}s  remaining {sim.agent_count():4d}/{n_start}  (wall {time.time() - t0:.0f}s)", flush=True)
    print(f"evacuated {n_start - sim.agent_count()} of {n_start} in {sim.elapsed_time():.0f} s "
          f"(wall {time.time() - t0:.0f}s, {sim.iteration_count()} iterations)")
    print(f"trajectories: {traj}")


if __name__ == "__main__":
    n = None
    ex = None
    mt = 600.0
    args = sys.argv[1:]
    if "--exits" in args:
        i = args.index("--exits"); ex = args[i + 1].split(","); del args[i:i + 2]
    if "--max-time" in args:
        i = args.index("--max-time"); mt = float(args[i + 1]); del args[i:i + 2]
    if args:
        n = int(args[0])
    main(n, ex, max_time=mt)
