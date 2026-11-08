# Bailey Hall 2-D model (CMCM 2026 "Bail(ey)ing Out")

A to-scale, parametric 2-D model of Cornell's Bailey Hall built from the evacuation plans and
chair layouts in `CMCM_2026_Bail(ey)ing_Out.pdf`, ready for pedestrian / egress simulation.

```
bailey_hall/
  model.py            geometry: Params (feet) -> shapely polygons, doors, seats, stairs, exports
  grid.py             hybrid grid for a simulation: 0.5 m lattice plus seat and walkway graph (metres)
  render.py           overlays on the PDF plans (sanity check) and clean figures / SVG
  Simulation.py       agent-based egress simulation on the hybrid grid
  bailey_hall.json    full export in METRES (walkable areas, spaces, obstacles, doors, seats,
                      spawn points, stairs, outdoor zones) - GeoJSON geometry objects
  bailey_hall_ft.json same in feet
  plans/              the two evacuation-plan images extracted from the PDF
  out/                rendered figures (overlay_*.png = model on top of the plan)
```

Set-up (already done in `.venv`): `uv venv --python 3.12 .venv && uv pip install shapely matplotlib scipy numpy pillow pymupdf`

```
.venv/bin/python bailey_hall/model.py            # areas, door count, seat counts
.venv/bin/python bailey_hall/render.py all       # out/overlay_*.png, model_*.png, model.svg
.venv/bin/python bailey_hall/grid.py             # build, validate, out/grid_hybrid_0.5m.npz + .png
.venv/bin/python bailey_hall/Simulation.py      # run the egress simulation (full house)
```

## Coordinates and units

* Origin = centre of the semicircular plan (centre of the loge / corridor arcs).
  x east, y north. The stage is north (y > +24 ft), the foyer and main entrance south (y < -63 ft).
* `Params` are in feet because every dimension on the drawings is imperial.
  `Model.to_dict("m")` / `bailey_hall.json` are in metres (`FT = 0.3048`).
* Two levels: `first_floor` (orchestra + loge + corridor + foyer) and `balcony`.
  `Model.flattened()` puts both in one plane for simulators without levels.

## How it was calibrated

| drawing | scale | centre (px) | how |
|---|---|---|---|
| first-floor plan (PDF p.4, 952x805) | 3.92 px/ft | (443.6, 300.4) | circle fit to the outer wall; scale from the 27'-0" centre block, loge row pitch and box rows |
| balcony plan (p.5, 617x801) | 3.92 px/ft | (299.0, 388.0) | same scale as p.4 (same drawing set); centre by symmetry + ring fit |
| orchestra/loge chairs (p.6) | 74.9 px/ft | (4040, 3424) after 180 deg rotation | 27'-0" rows = 2,025 px, 2'-10" row pitch = 212 px |
| balcony chairs (p.7) | 74.9 px/ft | (6096, 3175) after 180 deg rotation | 3'-0" row pitch = 224 px, 36'-4" side rows |

Pixels are isotropic on all four images (ellipse fit axis ratio 1.002). Expect about +-2 % on
positions, i.e. about 1 ft at the outer walls, and +-0.5 ft on door widths.
`out/overlay_first_floor.png` and `out/overlay_balcony.png` show the model drawn over the plans.

## What is in the model

First floor
* **Orchestra**: 18 rows A-T (no I, O), 27 ft straight centre block of 16 seats, two 3.5 ft
  radial aisles, fan-shaped side sections (rows A-K, a 3 ft cross passage at the 4" step, rows L-S),
  stage apron edge at y = +24 ft, 3 ft front cross aisle.
* **Loge / mezzanine**: barrier arc at r = 32.3 ft, crossed by four stair polygons (`stair_aisle_W/E`
  where the main aisles climb into the loge aisles, `stair_mid_W/E` from the side-section cross
  passage up to the loge side aisle = the six "access points" with the two front-corner stairs), rows AA-EE at r = 36.4 ... 48.3 ft, 4'-5" radial aisles 26 deg either side of
  south, a 4'-5" east-west side aisle at y = -10.4..-6 ft, wheelchair bay at the centre of DD/EE,
  side boxes (5 rows x 2 blocks per side, seats facing the hall, front rail at |x| = 32.4 ft).
* **Back wall** (inner face r = 48.5 ft, 3 ft thick): 5 openings - the two aisle doors, the two
  side-aisle doors, and the centre door behind the wheelchair bay. The brief says "nine regular
  exit doors"; the drawings show five openings, four of which are pairs of leaves, so the model
  uses 5 ft clear for those and 6 ft for the centre (`# ASSUMPTION`).
* **Corridor**: 9 ft wide (r = 51.5..60.5 ft), straight legs north to y = +29.4 ft where it enters
  room 132 / 10060 with an exterior door; bulges south into the vestibule (10041) which has the
  **three 5 ft doors to the foyer** at y = -63.2 ft.
* **Two portico doors** (4 ft) in the exterior wall at 39.5 and 149.6 deg, one per side, each facing an exterior stair (plan angles,
  clockwise from east); the portico and the two exterior stairs to street level are optional
  outdoor zones (`Model.outdoor`).
* **Foyer** 48 x 19 ft with **five 5.5 ft front doors** onto the monumental stair (outdoor zone).
* **Stage escape routes**: small stair at each front corner of the orchestra -> landing -> wide
  stair -> door in the side wall -> room 132/10060 -> exterior door (`# ASSUMPTION` for the
  exact route; red arrows on the plan).

Balcony
* U-shaped, front rail r = 46.4 ft, three curved rows AAA-CCC (r = 49.9, 52.8, 56.0 ft) with
  radial aisles 47.7 deg off south, 3 ft rear aisle, three straight rows per side above the
  corridor (|x| = 55.7, 53.0, 50.2 ft, y = -1.7..36 ft), columns ignored (they sit inside row CCC).
* Exits: doors into stairs 20001/20002 (30 deg off south, 4 ft, two-lane stairs that land in
  the vestibule corners next to the foyer doors, 30 ft walking length) and doors to the
  stage-side stairs 20004/20003 at the north end of the rear aisles (3.5 ft, land in room 132/10060).

Seats: rows are obstacles of depth `seat_depth` = 1.5 ft leaving 1.33 ft (orchestra) / 1.5 ft
(loge) clear between rows. `Model.seats()` samples seats at 20" along each row, packs a short
group by stepping the pitch down (19.0" orchestra, 19.4" loge), then thins evenly to exactly the
official capacities 527 / 486 / 316 (the chair layouts contain gaps for columns, 'MB' chairs and
wheelchairs that are not drawn). The balcony stays at 20" and is thinned from 399, so 75 of its
seat gaps are about 1 m rather than 0.5 m. `Model.spawn_points()` moves each seat into the
clear strip in front of its row - use those as agent start positions.

Judgement calls are tagged `# ASSUMPTION` in `model.py` (stage-escape route, stair lengths,
door widths, portico extent). Everything is a `Params` field, so change a number and re-render.

## Using it in a simulator

* `Model.walkable(level)` -> one shapely polygon (with holes) per level; `Model.exits(level)`.
* `Model.flattened()` -> single polygon with the balcony mirrored south of the foyer and the
  two main balcony stairs drawn as corridors (42 ft at level speed ~ 30 ft of stair at stair
  speed). The stage-side balcony stairs become direct exits (add ~30 s for the stair).
  `grid.py` no longer uses this layout, because it keeps both levels in one coordinate frame.
* `bailey_hall.json` has the same content as GeoJSON geometries for any other tool
  (Godot, a browser canvas, Mesa, Vadere...).

## Hybrid grid (`grid.py`)

`Grid.from_model(cell_m=0.5)` builds one directed graph over both levels. Everything is in metres.

* Open floor (aisles, corridors, foyer, stairs, doors) is a 0.5 m lattice with eight neighbours and
  no corner cutting. Each step is tested against the geometry, because the barrier and foyer wall
  are thinner than a cell.
* Seat rows are cut out of the lattice. Each of the 1,329 seats is a node at its real position, with
  one walkway node in front of it. Orchestra row A keeps its strip on the lattice as the front
  cross aisle.
* Entry costs: floor, door, exit and walkway 1, stairs 2, seats 2.5. Edge cost is true length times
  the entry cost of the destination. Seats step out to the walkway one-way, and seat to seat runs
  both ways.
* The two main balcony stairs are one-way links with a fixed cost of 18.29 (30 ft x 2.0), landing
  in the vestibule corners. The stage-side balcony stairs are not in the graph: BW and BE end at
  the balcony door, so add that stair's time in the simulation.
* 11 exit labels: F1-F5 (west to east), PW and PE (one portico door each), SW and SE (room 132),
  BW and BE. Close a gate with `grid.distance_field(blocked=grid.exit_mask("F3"))`.

`grid.dist` and `grid.node_dist` are weighted costs, not walking distances or times.

Pitfalls already handled (keep them in mind when editing `Params`): rows that end a few
centimetres short of a wall leave a hairline strip that a router can send people through;
spaces must overlap doors by `door_depth_pad`.
