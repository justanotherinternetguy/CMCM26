"""
Parametric 2-D model of Bailey Hall (Cornell) for pedestrian / egress simulation.

Coordinate system
-----------------
* Origin = centre of the semicircular plan (centre of the loge / corridor arcs).
* x east, y north (the stage is north, the foyer / main entrance south).
* All parameters are in FEET (the drawings are imperial); exports can be in metres.

Sources (CMCM_2026_Bail(ey)ing_Out.pdf)
---------------------------------------
* page 4 : first-floor evacuation plan   (952x805 px,   3.92 px/ft, centre (443.6,300.4) px)
* page 5 : balcony evacuation plan       (617x801 px,   3.92 px/ft, centre (299.0,388.0) px)
* page 6 : orchestra + loge chair layout (8479x7660 px, 74.9 px/ft, centre (4040,3424) px after 180deg rotation)
* page 7 : balcony chair layout          (10928x8302 px,74.9 px/ft, centre (6096,3175) px after 180deg rotation)

Scale was derived from the dimension strings on the chair layouts (27'-0" centre block,
2'-10" row pitch, 14'-0" box rows, 36'-4" balcony side rows) and transferred to the
evacuation plans through the loge row radii and the box rows.  Expect +-2 % (about 1 ft at
the outer walls).  Every number that is a judgement call rather than a measurement is tagged
`# ASSUMPTION`.

Level of detail
---------------
* Seat rows are obstacles (depth `seat_depth`) leaving a clear walkway between rows.
* Walls between spaces are modelled by leaving a gap between space polygons; doors are small
  rectangles that bridge the gap.  `walkable(level)` = union(spaces + doors) - obstacles.
* Stairs are 1-D connectors (polyline + walking length + width) between levels, plus a
  `flattened()` layout that lays the balcony beside the first floor and draws the stairs as
  corridors of the right walking length so a single-plane simulator can use it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping
from shapely.ops import unary_union

FT = 0.3048  # metres per foot


# --------------------------------------------------------------------------------------
# geometry helpers (angles in degrees, mathematical convention: 0 = east, CCW, y up)
# --------------------------------------------------------------------------------------
def arc_points(r: float, a0: float, a1: float, step: float = 1.0) -> List[Tuple[float, float]]:
    n = max(2, int(abs(a1 - a0) / step) + 1)
    return [(r * math.cos(math.radians(a0 + (a1 - a0) * i / (n - 1))),
             r * math.sin(math.radians(a0 + (a1 - a0) * i / (n - 1)))) for i in range(n)]


def annulus(r0: float, r1: float, a0: float = -180.0, a1: float = 0.0, step: float = 1.0) -> Polygon:
    """Annular sector between radii r0<r1 from angle a0 to a1 (CCW)."""
    outer = arc_points(r1, a0, a1, step)
    inner = arc_points(r0, a1, a0, step)
    return Polygon(outer + inner)


def disk(r: float, step: float = 1.0) -> Polygon:
    return Polygon(arc_points(r, 0, 360, step))


def rect(x0: float, x1: float, y0: float, y1: float) -> Polygon:
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def radial_strip(angle: float, width: float, r0: float, r1: float) -> Polygon:
    """Rectangle of the given width centred on the ray at `angle`, between radii r0 and r1."""
    p = rect(r0, r1, -width / 2, width / 2)
    return affinity.rotate(p, angle, origin=(0, 0))


def mirror_x(g):
    return affinity.scale(g, xfact=-1, yfact=1, origin=(0, 0))


def pol2cart(r: float, angle: float) -> Tuple[float, float]:
    return r * math.cos(math.radians(angle)), r * math.sin(math.radians(angle))


# --------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------
@dataclass
class Params:
    # ---- image calibration (needed only for overlays) --------------------------------
    plan_px_per_ft: float = 3.92
    plan_center_px: Tuple[float, float] = (443.6, 300.4)
    balcony_px_per_ft: float = 3.92
    balcony_center_px: Tuple[float, float] = (299.0, 388.0)

    # ---- seats ------------------------------------------------------------------------
    seat_pitch: float = 20.0 / 12.0   # seat width along the row (20" chairs dominate)
    row_pitch: float = 34.0 / 12.0    # orchestra back-to-back row spacing (2'-10" on the chart)
    loge_row_pitch: float = 3.0       # loge / balcony rows (measured 2.99 ft)
    seat_depth: float = 1.5           # ASSUMPTION: obstacle depth of a row of tipped-up seats
                                      # (leaves row_pitch-seat_depth clear = 1.33 ft orchestra)

    # ---- orchestra ---------------------------------------------------------------------
    centre_half_width: float = 13.5          # 27'-0" centre block
    aisle_width: float = 3.5                 # main radial aisles (measured 3.54 ft on chart)
    orch_rows: int = 18                      # A..T (no I, O)
    row_A_back_y: float = 19.36              # seat-back line of row A (chart)
    side_rows_front: int = 11                # A..K in the side sections
    side_rows_rear: int = 7                  # L..S/T, shortened by the loge
    side_rear_offset: float = 1.5            # rear side rows sit ~half a row further south
    cross_gap_y: Tuple[float, float] = (-9.0, -6.0)   # 4" step / 3 ft cross passage in the side sections
    side_row_arc_center_y: float = 116.6     # ASSUMPTION: side rows are arcs about (0, 116.6)
    box_rail_x: float = 32.4                 # front rail of the side boxes (|x|); plan 317 px, chart 1600 px
    r_barrier: float = 32.3                  # orchestra/loge barrier (arc about origin; row T corners at r=31.8)
    barrier_thickness: float = 0.5
    apron_center_y: float = 161.0            # stage apron edge = arc r 137 about (0,161)
    apron_radius: float = 137.0              #   -> y = 24.0 at the centre line, 28.2 at the boxes

    # ---- loge (mezzanine) ----------------------------------------------------------------
    loge_row_back_r: Tuple[float, ...] = (36.4, 39.4, 42.35, 45.4, 48.3)   # AA..EE
    loge_aisle_angle: float = 26.0           # radial aisles, degrees off south
    loge_aisle_width: float = 4.4            # 4'-5" (TYP)
    loge_side_aisle_y: Tuple[float, float] = (-10.4, -6.0)   # E-W aisle between boxes and loge side rows
    wheelchair_bay_half: float = 3.25        # DD/EE gap at the centre line (2 wheelchair spaces)
    box_row_back_x: Tuple[float, ...] = (48.8, 45.35, 42.25, 39.15, 35.95)   # |x| of seat backs (chart centre lines + 0.75); back row merged into the wall (48.5)
    box_north: Tuple[float, float] = (18.5, 32.2)   # north box block (8 seats / row)
    box_south: Tuple[float, float] = (-5.7, 13.7)   # south box block (~12 seats / row)
    mid_stair: Tuple[float, float, float, float] = (-36.6, -31.2, -9.0, -5.4)  # x0,x1,y0,y1 west mid stair

    # ---- walls / corridor / portico ----------------------------------------------------
    x_wall_in: float = 48.5     # inner face of the auditorium side/back wall (straight part and arc)
    x_wall_out: float = 51.5
    r_wall_in: float = 48.5
    r_wall_out: float = 51.5
    r_ext_in: float = 60.5      # corridor outer edge = exterior wall inner face
    r_ext_out: float = 62.7
    r_portico_out: float = 71.2  # portico edge (column line)
    corridor_north_y: float = 29.4   # straight corridors end at room 132 / 10060
    vestibule_half: float = 20.0     # the corridor bulges south to the foyer wall between the two stairs
    foyer_wall_y: float = -63.2
    loge_door_width: float = 5.0           # ASSUMPTION: pairs of single doors ~ 5 ft clear
    loge_center_door_width: float = 6.0
    loge_side_door_y: Tuple[float, float] = (-13.2, -8.6)   # side doors in the back wall (chart gap)
    # One side door each way, facing the exterior stairs.  The openings at 55 and 125 deg on the plan
    # are not exits: Google Maps shows only these two.  The west door is at the west stair's angle.
    portico_door_angles: Tuple[float, ...] = (-39.5, -149.6)
    portico_door_width: float = 4.0        # ASSUMPTION (gap measured 3.3-4.4 ft)
    foyer_door_xs: Tuple[float, ...] = (-11.2, 0.0, 11.2)
    foyer_door_width: float = 5.0
    foyer_x_half: float = 24.0
    foyer_south_y: float = -82.0
    front_door_xs: Tuple[float, ...] = (-20.4, -10.2, 0.0, 10.2, 20.4)
    front_door_width: float = 5.5
    front_steps: Tuple[float, float, float, float] = (-32.5, 32.5, -125.0, -84.8)
    ext_stair_angles: Tuple[float, float] = (-149.6, -38.0)   # portico -> street stairs (W, E)
    ext_stair_width: float = 10.0

    # ---- stage escape routes (front corners) -----------------------------------------
    esc_stair1: Tuple[float, float, float, float] = (-32.8, -27.7, 26.9, 32.0)
    esc_landing: Tuple[float, float, float, float] = (-36.6, -29.0, 32.0, 38.0)
    esc_stair2: Tuple[float, float, float, float] = (-44.3, -36.6, 32.0, 38.0)
    esc_landing2: Tuple[float, float, float, float] = (-48.5, -44.3, 32.0, 38.0)  # landing at the wall door
    esc_door_y: Tuple[float, float] = (35.5, 38.5)       # door in the side wall (x = -48.5..-51.5)
    room132: Tuple[float, float, float, float] = (-68.5, -51.5, 29.4, 37.1)
    room132_door_y: Tuple[float, float] = (30.7, 34.5)   # exterior door on its west wall
    corridor_to_132_x: Tuple[float, float] = (-59.0, -55.5)

    # ---- balcony ---------------------------------------------------------------------
    bal_rail_r: float = 46.4
    bal_row_back_r: Tuple[float, ...] = (49.9, 52.8, 56.0)   # AAA, BBB, CCC
    bal_aisle_angle: float = 47.7           # degrees off south (chart gap at 42.3 deg)
    bal_aisle_width: float = 3.5
    bal_arc_start_angle: float = 3.5        # arc rows start this many degrees below the E-W line
    bal_straight_back_x: Tuple[float, ...] = (55.7, 53.0, 50.2)
    bal_straight_y: Tuple[float, float] = (-1.7, 36.0)     # 36'-4" rows
    bal_north_y: float = 38.0
    bal_exit_sw: Tuple[float, float] = (29.0, -50.5)       # (|x|, y) door into stair 20001/20002 (~30 deg off south)
    bal_exit_sw_width: float = 4.0
    bal_exit_nw: Tuple[float, float] = (57.0, 38.0)        # (|x|, y) door to stair 20004/20003 at the end of the rear aisle
    bal_exit_nw_width: float = 3.5
    # stairs balcony -> first floor
    stair_sw_bottom: Tuple[float, float] = (17.0, -61.5)   # (|x|, y) lands in the corridor / vestibule corner
    stair_sw_length: float = 30.0          # ASSUMPTION: two flights + landing (walking length)
    stair_sw_width: float = 5.0            # two 2.5 ft lanes
    stair_nw_length: float = 32.0          # ASSUMPTION: two flights down to stair 10004 landing
    stair_nw_width: float = 4.0

    # ---- misc --------------------------------------------------------------------------
    door_depth_pad: float = 0.3  # doors overlap the spaces slightly so unions are watertight


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
@dataclass
class Door:
    name: str
    level: str
    poly: Polygon
    width: float
    kind: str = "door"        # door | exit (leads out of the building) | stair-head
    note: str = ""

    @property
    def center(self):
        c = self.poly.centroid
        return (c.x, c.y)


@dataclass
class Stair:
    name: str
    level_from: str
    p_from: Tuple[float, float]
    level_to: str
    p_to: Tuple[float, float]
    length: float
    width: float
    lanes: int = 2
    note: str = ""


class Model:
    LEVELS = ("first_floor", "balcony")

    def __init__(self, p: Optional[Params] = None):
        self.p = p or Params()
        self.spaces: Dict[str, Dict[str, Polygon]] = {lv: {} for lv in self.LEVELS}
        self.obstacles: Dict[str, Dict[str, Polygon]] = {lv: {} for lv in self.LEVELS}
        self.doors: List[Door] = []
        self.stairs: List[Stair] = []
        self.rows: Dict[str, List[dict]] = {lv: [] for lv in self.LEVELS}  # seat rows (for spawning)
        self.outdoor: Dict[str, Polygon] = {}
        self._build_first_floor()
        self._build_balcony()
        self._build_stairs()

    # ---------------------------------------------------------------- first floor ----
    def _build_first_floor(self):
        p = self.p
        lv = "first_floor"
        S, O = self.spaces[lv], self.obstacles[lv]

        # --- orchestra floor ---------------------------------------------------------
        apron = disk(p.apron_radius).buffer(0)
        apron = affinity.translate(apron, 0, p.apron_center_y)         # stage sits inside this disk
        half = rect(-p.box_rail_x, p.box_rail_x, -60, 60)
        orch = half.intersection(disk(p.r_barrier - p.barrier_thickness / 2).union(
            rect(-p.box_rail_x, p.box_rail_x, p.cross_gap_y[1], 60)))     # straight sides above the cross gap
        orch = orch.difference(apron)
        S["orchestra"] = orch

        # orchestra seat rows -------------------------------------------------------
        rows = []
        letters = "ABCDEFGHJKLMNPQRST"
        orch_clear = p.row_pitch - p.seat_depth
        for i in range(p.orch_rows):
            yb = p.row_A_back_y - i * p.row_pitch
            # centre block (straight)
            rows.append(dict(section="orchestra_centre", row=letters[i], facing=("vec", (0.0, 1.0)), clear=orch_clear,
                             poly=rect(-p.centre_half_width, p.centre_half_width, yb, yb + p.seat_depth),
                             line=LineString([(-p.centre_half_width, yb + p.seat_depth / 2),
                                              (p.centre_half_width, yb + p.seat_depth / 2)])))
        # side sections: arcs about (0, side_row_arc_center_y) that meet the centre rows at the aisle
        xa = p.centre_half_width + p.aisle_width
        for i in range(p.orch_rows):
            yb = p.row_A_back_y - i * p.row_pitch
            if i >= p.side_rows_front:
                if i - p.side_rows_front >= p.side_rows_rear:
                    continue
                yb -= p.side_rear_offset
            R = math.hypot(xa, p.side_row_arc_center_y - yb)
            for sgn, name in ((-1, "orchestra_west"), (1, "orchestra_east")):
                ring = annulus(R, R + p.seat_depth, 180, 360, step=0.5)  # lower half about the arc centre
                ring = affinity.translate(ring, 0, p.side_row_arc_center_y)
                side = rect(sgn * xa, sgn * 60, -60, 60)
                poly = ring.intersection(side).intersection(orch)
                # keep clear of the cross gap
                poly = poly.difference(rect(-60, 60, p.cross_gap_y[0], p.cross_gap_y[1]))
                poly = poly.buffer(0)
                if poly.is_empty or poly.area < 2:
                    continue
                ang0 = math.degrees(math.atan2(yb + p.seat_depth / 2 - p.side_row_arc_center_y, sgn * xa))
                # centre-line of the seats: arc at R + seat_depth/2 clipped to the row polygon
                a_far = -90 + sgn * 30
                line = LineString(arc_points(R + p.seat_depth / 2, ang0, a_far, 0.25))
                line = affinity.translate(line, 0, p.side_row_arc_center_y).intersection(poly.buffer(0.05))
                rows.append(dict(section=name, row=letters[i], poly=poly, line=line, clear=orch_clear,
                                 facing=("center", (0.0, p.side_row_arc_center_y))))
        for r in rows:
            O[f"row_{r['section']}_{r['row']}"] = r["poly"]
        self.rows[lv] += rows

        # barrier between orchestra and loge (arc) + box rails (straight) -----------
        barrier = annulus(p.r_barrier - p.barrier_thickness / 2, p.r_barrier + p.barrier_thickness / 2, -180, 0)
        barrier = barrier.intersection(rect(-60, 60, -60, p.cross_gap_y[1]))
        rail = rect(p.box_rail_x, p.box_rail_x + 0.5, p.cross_gap_y[1], p.box_north[1])
        barrier = unary_union([barrier, rail, mirror_x(rail)])
        # openings: main aisles (stairs up to the loge aisles) and the two mid stairs.  The stair
        # polygons are spaces that bridge the barrier ring (orchestra floor <-> loge floor).
        for sgn, tag in ((-1, "W"), (1, "E")):
            aisle_stair = rect(sgn * p.centre_half_width, sgn * (p.centre_half_width + p.aisle_width), -36.0, -26.0)
            ms = p.mid_stair
            mid_stair = rect(sgn * ms[0], sgn * (ms[1] + 1.0), ms[2], ms[3])     # reaches 1 ft into the orchestra floor
            barrier = barrier.difference(aisle_stair).difference(mid_stair)
            S[f"stair_aisle_{tag}"] = aisle_stair     # steps up from the orchestra aisle into the loge aisle
            S[f"stair_mid_{tag}"] = mid_stair         # steps up from the side-section cross passage to the loge side aisle
        O["barrier"] = barrier

        # --- loge (mezzanine) ---------------------------------------------------------
        loge_arc = annulus(p.r_barrier + p.barrier_thickness / 2, p.r_wall_in, -180, 0).intersection(
            rect(-60, 60, -60, p.cross_gap_y[1]))
        boxes_w = rect(-p.x_wall_in, -p.box_rail_x, p.cross_gap_y[1], p.box_north[1])
        # the E-W side aisle runs to the straight side wall (the arc would stop 1 ft short of the side doors)
        side_ext = rect(-p.x_wall_in, -p.box_rail_x, p.loge_side_door_y[0] - 0.3, p.cross_gap_y[1])
        loge = unary_union([loge_arc, boxes_w, mirror_x(boxes_w), side_ext, mirror_x(side_ext)]).buffer(0)
        S["loge"] = loge

        # loge rows AA..EE
        aisle_ang = -90 - p.loge_aisle_angle, -90 + p.loge_aisle_angle
        aisles = unary_union([radial_strip(a, p.loge_aisle_width, 30, 52) for a in aisle_ang])
        side_aisle = rect(-60, 60, p.loge_side_aisle_y[0], p.loge_side_aisle_y[1])
        names = ("AA", "BB", "CC", "DD", "EE")
        inner_r = (p.r_barrier + p.barrier_thickness / 2,) + p.loge_row_back_r[:-1]
        for k, rb in enumerate(p.loge_row_back_r):
            clear = (rb - p.seat_depth) - inner_r[k]
            rb_out = rb if k < len(p.loge_row_back_r) - 1 else max(rb, p.r_wall_in + 0.3)   # EE backs onto the wall: no sliver
            ring = annulus(rb - p.seat_depth, rb_out, -180, 0, 0.5).intersection(loge)
            ring = ring.intersection(rect(-60, 60, -60, p.loge_side_aisle_y[0]))   # rows stop at the side aisle
            ring = ring.difference(aisles)
            if k >= 3:  # DD, EE: wheelchair bay at the centre
                ring = ring.difference(rect(-p.wheelchair_bay_half, p.wheelchair_bay_half, -60, -30))
            ring = ring.buffer(0)
            geoms = list(ring.geoms) if isinstance(ring, MultiPolygon) else [ring]
            for j, g in enumerate(sorted(geoms, key=lambda g: g.centroid.x)):
                if g.area < 2:
                    continue
                cl = LineString(arc_points(rb - p.seat_depth / 2, -180, 0, 0.25)).intersection(g.buffer(0.05))
                rows_l = dict(section=f"loge_{names[k]}", row=f"{names[k]}{j}", poly=g, line=cl, clear=clear,
                              facing=("center", (0.0, 0.0)))
                self.rows[lv].append(rows_l)
                O[f"row_loge_{names[k]}_{j}"] = g
        # box rows (straight, seats face the centre line)
        box_inner = p.box_row_back_x[1:] + (p.box_rail_x + 0.5,)
        for k, xb in enumerate(p.box_row_back_x):
            clear = (xb - p.seat_depth) - box_inner[k]
            for blk, (y0, y1) in (("N", p.box_north), ("S", p.box_south)):
                for sgn, side in ((-1, "W"), (1, "E")):
                    x_in = sgn * (xb - p.seat_depth)
                    poly = rect(sgn * xb, x_in, y0, y1)
                    xm = sgn * (xb - p.seat_depth / 2)
                    self.rows[lv].append(dict(section=f"box_{side}{blk}", row=f"box{k}", facing=("vec", (-sgn * 1.0, 0.0)), clear=clear,
                                              poly=poly, line=LineString([(xm, y0), (xm, y1)])))
                    O[f"row_box_{side}{blk}_{k}"] = poly

        # --- stage-escape route (front corners): stair -> landing -> stair -> room 132 -> out
        for sgn, tag in ((1, "W"), (-1, "E")):   # params are given for the WEST side (negative x)
            def R(t):
                x0, x1, y0, y1 = t
                return rect(sgn * x0, sgn * x1, y0, y1)
            S[f"escape_{tag}_stair1"] = R(p.esc_stair1)
            S[f"escape_{tag}_landing"] = R(p.esc_landing)
            S[f"escape_{tag}_stair2"] = R(p.esc_stair2)
            S[f"escape_{tag}_landing2"] = R(p.esc_landing2)
            S[f"room132_{tag}"] = R(p.room132)
            # door through the auditorium side wall into room 132
            self.doors.append(Door(f"escape_{tag}_wall_door", lv,
                                   rect(-sgn * (p.x_wall_in - p.door_depth_pad), -sgn * (p.x_wall_out + p.door_depth_pad),
                                        p.esc_door_y[0], p.esc_door_y[1]),
                                   p.esc_door_y[1] - p.esc_door_y[0], "door",
                                   "stage escape route: landing -> room 132"))
            # exterior door of room 132  (EXIT)
            x_ext = sgn * p.room132[0]
            self.doors.append(Door(f"exit_{tag}_room132", lv,
                                   rect(x_ext + sgn * p.door_depth_pad, x_ext - sgn * 2.0,
                                        p.room132_door_y[0], p.room132_door_y[1]),
                                   p.room132_door_y[1] - p.room132_door_y[0], "exit",
                                   "ASSUMPTION: exterior door on the west/east wall of room 132 / 10060"))
            # corridor north end -> room 132
            self.doors.append(Door(f"corridor_{tag}_to_132", lv,
                                   rect(sgn * p.corridor_to_132_x[0], sgn * p.corridor_to_132_x[1],
                                        p.corridor_north_y - p.door_depth_pad, p.room132[2] + p.door_depth_pad),
                                   p.corridor_to_132_x[1] - p.corridor_to_132_x[0], "door",
                                   "ASSUMPTION: 3.5 ft door between the corridor end and room 132"))
            # The corridor and room 132 share an edge, so without this wall the whole 9 ft corridor
            # width stays open and the 3.5 ft door above would limit nothing.
            O[f"corridor_{tag}_132_wall"] = rect(-sgn * p.r_ext_in, -sgn * p.r_wall_out,
                                                 p.corridor_north_y - p.door_depth_pad,
                                                 p.corridor_north_y + p.door_depth_pad).difference(self.doors[-1].poly)
        # the first escape stair is a stair (slower); record as obstacle-free space but tag it
        # --- corridor + vestibule -------------------------------------------------------
        corr = annulus(p.r_wall_out, p.r_ext_in, -180, 0, 0.5)
        corr_n = rect(-p.r_ext_in, -p.r_wall_out, -0.1, p.corridor_north_y)
        vest = rect(-p.vestibule_half, p.vestibule_half, p.foyer_wall_y, -50).difference(disk(p.r_wall_out))
        S["corridor"] = unary_union([corr, corr_n, mirror_x(corr_n), vest]).buffer(0)

        # loge -> corridor doors (5)
        for a in aisle_ang:
            self.doors.append(Door(f"loge_door_{'W' if a < -90 else 'E'}_aisle", lv,
                                   radial_strip(a, p.loge_door_width, p.r_wall_in - p.door_depth_pad,
                                                p.r_wall_out + p.door_depth_pad),
                                   p.loge_door_width, "door", "loge aisle door (pair of leaves)"))
        self.doors.append(Door("loge_door_centre", lv,
                               radial_strip(-90, p.loge_center_door_width, p.r_wall_in - p.door_depth_pad,
                                            p.r_wall_out + p.door_depth_pad),
                               p.loge_center_door_width, "door", "centre door behind the wheelchair bay -> vestibule"))
        for sgn, tag in ((-1, "W"), (1, "E")):
            self.doors.append(Door(f"loge_door_{tag}_side", lv,
                                   rect(sgn * (p.x_wall_in - p.door_depth_pad), sgn * (p.x_wall_out + p.door_depth_pad),
                                        p.loge_side_door_y[0], p.loge_side_door_y[1]),
                                   p.loge_side_door_y[1] - p.loge_side_door_y[0], "door",
                                   "side door at the end of the loge side aisle"))
        # corridor -> portico (2 side doors)  (EXIT from the building envelope)
        for a in p.portico_door_angles:
            self.doors.append(Door(f"portico_door_{int(round(-a))}", lv,
                                   radial_strip(a, p.portico_door_width, p.r_ext_in - p.door_depth_pad,
                                                p.r_ext_out + p.door_depth_pad),
                                   p.portico_door_width, "exit", "rarely used side door to the portico"))
        # vestibule -> foyer (3)
        for x in p.foyer_door_xs:
            self.doors.append(Door(f"foyer_door_{x:+.0f}", lv,
                                   rect(x - p.foyer_door_width / 2, x + p.foyer_door_width / 2,
                                        p.foyer_wall_y - 1.0 - p.door_depth_pad, p.foyer_wall_y + p.door_depth_pad),
                                   p.foyer_door_width, "door", "vestibule -> foyer"))
        # --- foyer and front doors ----------------------------------------------------
        S["foyer"] = rect(-p.foyer_x_half, p.foyer_x_half, p.foyer_south_y, p.foyer_wall_y - 1.0)
        for x in p.front_door_xs:
            self.doors.append(Door(f"front_door_{x:+.0f}", lv,
                                   rect(x - p.front_door_width / 2, x + p.front_door_width / 2,
                                        p.front_steps[3] - p.door_depth_pad, p.foyer_south_y + p.door_depth_pad),
                                   p.front_door_width, "exit", "main entrance doors -> monumental staircase"))
        # --- outdoor zones (optional for simulations that continue to the street) ----
        self.outdoor["front_steps"] = rect(*p.front_steps)
        portico = annulus(p.r_ext_out, p.r_portico_out, -180, 0, 0.5)
        por_n = rect(-p.r_portico_out, -p.r_ext_out, -0.1, p.corridor_north_y)
        self.outdoor["portico"] = unary_union([portico, por_n, mirror_x(por_n)]).difference(
            rect(-40, 40, -130, -55))  # ASSUMPTION: portico stops at the foyer block
        for a in p.ext_stair_angles:
            self.outdoor[f"ext_stair_{int(round(-a))}"] = radial_strip(a, p.ext_stair_width, p.r_portico_out, p.r_portico_out + 8)

    # ------------------------------------------------------------------ balcony ------
    def _build_balcony(self):
        p = self.p
        lv = "balcony"
        S, O = self.spaces[lv], self.obstacles[lv]
        arc = annulus(p.bal_rail_r, p.r_ext_in, -180, 0, 0.5)
        straight = rect(-p.r_ext_in, -p.bal_rail_r, -0.1, p.bal_north_y)
        S["balcony"] = unary_union([arc, straight, mirror_x(straight)]).buffer(0)

        aisles = unary_union([radial_strip(-90 - p.bal_aisle_angle, p.bal_aisle_width, 40, 62),
                              radial_strip(-90 + p.bal_aisle_angle, p.bal_aisle_width, 40, 62)])
        names = ("AAA", "BBB", "CCC")
        inner_r = (p.bal_rail_r,) + p.bal_row_back_r[:-1]
        for k, rb in enumerate(p.bal_row_back_r):
            clear = (rb - p.seat_depth) - inner_r[k]
            ring = annulus(rb - p.seat_depth, rb, -180 + p.bal_arc_start_angle, -p.bal_arc_start_angle, 0.5)
            ring = ring.difference(aisles).buffer(0)
            geoms = list(ring.geoms) if isinstance(ring, MultiPolygon) else [ring]
            for j, g in enumerate(sorted(geoms, key=lambda g: g.centroid.x)):
                if g.area < 2:
                    continue
                cl = LineString(arc_points(rb - p.seat_depth / 2, -180, 0, 0.25)).intersection(g.buffer(0.05))
                self.rows[lv].append(dict(section=f"balcony_{names[k]}", row=f"{names[k]}{j}", poly=g, line=cl, clear=clear,
                                          facing=("center", (0.0, 0.0))))
                O[f"row_bal_{names[k]}_{j}"] = g
        inner_x = p.bal_straight_back_x[1:] + (p.bal_rail_r,)
        for k, xb in enumerate(p.bal_straight_back_x):
            clear = (xb - p.seat_depth) - inner_x[k]
            for sgn, side in ((-1, "W"), (1, "E")):
                poly = rect(sgn * xb, sgn * (xb - p.seat_depth), p.bal_straight_y[0], p.bal_straight_y[1])
                xm = sgn * (xb - p.seat_depth / 2)
                self.rows[lv].append(dict(section=f"balcony_side_{side}", row=f"S{k}", poly=poly, facing=("vec", (-sgn * 1.0, 0.0)), clear=clear,
                                          line=LineString([(xm, p.bal_straight_y[0]), (xm, p.bal_straight_y[1])])))
                O[f"row_bal_side_{side}_{k}"] = poly

        # balcony exits (stair heads)
        for sgn, tag in ((-1, "W"), (1, "E")):
            x, y = p.bal_exit_sw
            ang = math.degrees(math.atan2(y, sgn * x))
            self.doors.append(Door(f"balcony_exit_S{tag}", lv,
                                   radial_strip(ang, p.bal_exit_sw_width, p.r_ext_in - 2.0, p.r_ext_in + 1.5),
                                   p.bal_exit_sw_width, "stair-head", "door into stair 20001/20002 -> first-floor vestibule"))
            x, y = p.bal_exit_nw
            self.doors.append(Door(f"balcony_exit_N{tag}", lv,
                                   rect(sgn * (x - p.bal_exit_nw_width / 2), sgn * (x + p.bal_exit_nw_width / 2), y - 1.5, y + 1.5),
                                   p.bal_exit_nw_width, "stair-head", "door into stair 20004/20003 -> stair 10004 -> room 132 -> out"))

    # ------------------------------------------------------------------- stairs ------
    def _build_stairs(self):
        p = self.p
        for sgn, tag in ((-1, "W"), (1, "E")):
            self.stairs.append(Stair(f"stair_S{tag}", "balcony", (sgn * p.bal_exit_sw[0], p.bal_exit_sw[1]),
                                     "first_floor", (sgn * p.stair_sw_bottom[0], p.stair_sw_bottom[1]),
                                     p.stair_sw_length, p.stair_sw_width, 2,
                                     "two flights with a centre handrail (two lanes); lands beside the foyer doors"))
            self.stairs.append(Stair(f"stair_N{tag}", "balcony", (sgn * p.bal_exit_nw[0], p.bal_exit_nw[1]),
                                     "first_floor", (sgn * (-p.room132[0] - 3.0), p.room132[3] - 1.5),
                                     p.stair_nw_length, p.stair_nw_width, 1,
                                     "stage-side stair 20004/20003 -> 10004/10003; ASSUMPTION: discharges into room 132/10060"))

    # ------------------------------------------------------------------ derived ------
    def walkable(self, level: str) -> Polygon:
        """Union of spaces and doors minus obstacles, as a (Multi)Polygon in feet."""
        parts = list(self.spaces[level].values()) + [d.poly for d in self.doors if d.level == level]
        area = unary_union(parts).buffer(0)
        obs = unary_union(list(self.obstacles[level].values())).buffer(0)
        walk = area.difference(obs).buffer(0)
        if isinstance(walk, MultiPolygon):   # drop slivers (< 1 ft2) that unions sometimes leave behind
            parts = [g for g in walk.geoms if g.area >= 1.0]
            walk = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        return walk

    def exits(self, level: str) -> List[Door]:
        return [d for d in self.doors if d.level == level and d.kind in ("exit", "stair-head")]

    OFFICIAL_SEATS = {"orchestra": 527, "loge": 486, "balcony": 316}
    SEAT_PITCH_STEP = 0.005   # feet
    MIN_SEAT_PITCH = 1.25     # feet; 15" is narrower than any real chair, so stop searching there

    @staticmethod
    def section_group(section: str) -> str:
        if section.startswith("orchestra"):
            return "orchestra"
        if section.startswith("box") or section.startswith("loge"):
            return "loge"
        return "balcony"

    def seats(self, level: str, pitch: Optional[float] = None, trim: bool = True) -> List[dict]:
        """Seat centre points sampled along each row (for spawning agents).

        With trim=True the geometric seat count of each group (orchestra / loge / balcony) is
        thinned evenly to the official capacity (527 / 486 / 316): the chair layouts have
        gaps (columns, wheelchair bays, 'MB' movable chairs) that the model does not draw."""
        pitch = pitch or self.p.seat_pitch
        out = self._seats_raw(level, pitch)
        if not trim:
            return out
        res = []
        for grp in ("orchestra", "loge", "balcony"):
            grp_seats = [s for s in out if self.section_group(s["section"]) == grp]
            n, target = len(grp_seats), self.OFFICIAL_SEATS[grp]
            if 0 < n < target:   # a few seats short (e.g. 'MB' chairs on movable bases): pack the rows a little tighter
                # Seat counts jump in whole seats per row, so a proportional pitch estimate can
                # stall below the target; stepping down guarantees the first pitch that fits.
                packed_pitch = pitch
                while n < target and packed_pitch > self.MIN_SEAT_PITCH:
                    packed_pitch -= self.SEAT_PITCH_STEP
                    grp_seats = [s for s in self._seats_raw(level, packed_pitch) if self.section_group(s["section"]) == grp]
                    n = len(grp_seats)
            if n <= target:
                res += grp_seats
                continue
            keep = set(int(round(i * n / target)) for i in range(target))
            res += [s for i, s in enumerate(grp_seats) if i in keep]
        return res

    def _seats_raw(self, level: str, pitch: float) -> List[dict]:
        out = []
        for r in self.rows[level]:
            line = r["line"]
            if line.is_empty:
                continue
            lines = list(line.geoms) if hasattr(line, "geoms") else [line]
            for ln in lines:
                n = int(ln.length // pitch)
                if n < 1:
                    continue
                margin = (ln.length - n * pitch) / 2 + pitch / 2
                for i in range(n):
                    pt = ln.interpolate(margin + i * pitch)
                    out.append(dict(x=pt.x, y=pt.y, section=r["section"], row=r["row"], seat=i + 1,
                                    _facing=r.get("facing", ("vec", (0.0, 1.0))), _clear=r.get("clear", 1.3)))
        return out

    def spawn_points(self, level: str, trim: bool = True) -> List[dict]:
        """One point per seat, moved from the seat centre into the clear walkway in front of
        the row (seats themselves lie inside the row obstacles).  Use these as agent start
        positions."""
        p = self.p
        out = []
        for s in self.seats(level, trim=trim):
            ux, uy = self._front_direction(s["_facing"], s["x"], s["y"])
            off = p.seat_depth / 2 + s["_clear"] / 2   # centre of the clear strip in front of the row
            q = dict(s); q["x"] = s["x"] + ux * off; q["y"] = s["y"] + uy * off
            q["clear"] = s["_clear"]
            q.pop("_facing", None); q.pop("_clear", None)
            out.append(q)
        return out

    @staticmethod
    def _front_direction(facing, x: float, y: float) -> Tuple[float, float]:
        """Unit vector from a point on a row towards the front of that row."""
        kind, reference = facing
        if kind == "vec":
            return reference
        delta_x, delta_y = reference[0] - x, reference[1] - y
        distance = math.hypot(delta_x, delta_y) or 1.0
        return delta_x / distance, delta_y / distance

    def row_regions(self, level: str) -> List[dict]:
        """Footprint of each seat row for a lattice: the row polygon and its walkway strip.

        The strip is swept along the whole row, not just between its first and last spawn
        point, because stopping at those points would leave orphan lattice cells at row ends."""
        p = self.p
        regions = []
        for row in self.rows[level]:
            offset_points = []
            for x, y in row["line"].coords:
                front_x, front_y = self._front_direction(row["facing"], x, y)
                offset = p.seat_depth / 2 + row["clear"] / 2
                offset_points.append((x + front_x * offset, y + front_y * offset))
            strip = LineString(offset_points).buffer(row["clear"] / 2, cap_style="flat")
            regions.append(dict(section=row["section"], row=row["row"], poly=row["poly"], strip=strip))
        return regions

    def seat_counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for lv in self.LEVELS:
            for s in self.seats(lv, trim=False):
                key = self.section_group(s["section"])
                c[key] = c.get(key, 0) + 1
        return c

    # --------------------------------------------------------------- flattened -------
    def flattened(self, balcony_shift_y: float = -148.0) -> dict:
        """Single-plane layout for simulators without levels.

        The balcony is mirrored in y and placed south of the foyer so that its two main
        stairs (20001/20002) can be drawn as corridors from the balcony doors down to the
        first-floor landings at the vestibule corners.  Those corridors are ~42 ft long
        against ~30 ft of real stair; walked at level speed that is roughly the same time as
        the stair at stair speed, so no speed penalty is needed on them.
        The stage-side balcony stairs (20004/20003) do not interact with the main crowd and
        are represented as direct exits at the balcony doors (add ~30 s for the stair when
        reporting their egress times).
        Returns dict(walkable=Polygon[ft], exits=[(name, Polygon)], seats=[...], spawn=[...])."""
        p = self.p
        T = balcony_shift_y
        flip = lambda g: affinity.translate(affinity.scale(g, 1, -1, origin=(0, 0)), 0, T)
        fpt = lambda x, y: (x, -y + T)
        ff = self.walkable("first_floor")
        bal = flip(self.walkable("balcony"))
        parts = [ff, bal]
        stair_polys = []
        for st in self.stairs:
            if not st.name.startswith("stair_S"):
                continue
            sgn = -1 if st.name.endswith("W") else 1
            a = fpt(*st.p_from)                   # balcony door, flipped
            b = st.p_to                           # first-floor landing
            path = LineString([b, (sgn * 27.5, -68.0), (sgn * 27.5, a[1] + 6.0), a])
            poly = path.buffer(st.width / 2, cap_style=2, join_style=2)
            stair_polys.append((st.name, poly, path.length))
            parts.append(poly)
        walk = unary_union(parts).buffer(0)
        exits = [(d.name, d.poly) for d in self.doors if d.level == "first_floor" and d.kind == "exit"]
        exits += [(d.name, flip(d.poly)) for d in self.doors if d.level == "balcony" and d.name.startswith("balcony_exit_N")]
        seats = self.seats("first_floor") + [dict(s, **dict(zip(("x", "y"), fpt(s["x"], s["y"])))) for s in self.seats("balcony")]
        spawn = self.spawn_points("first_floor") + [dict(s, **dict(zip(("x", "y"), fpt(s["x"], s["y"])))) for s in self.spawn_points("balcony")]
        return dict(walkable=walk, exits=exits, seats=seats, spawn=spawn, stairs=stair_polys, balcony_shift_y=T)

    # ------------------------------------------------------------------ export -------
    def to_dict(self, units: str = "m") -> dict:
        k = FT if units == "m" else 1.0

        def g(geom):
            return mapping(affinity.scale(geom, k, k, origin=(0, 0)))

        d = dict(units=units, origin="centre of the semicircular plan; x east, y north; stage north",
                 params=asdict(self.p), levels={}, stairs=[], outdoor={})
        for lv in self.LEVELS:
            d["levels"][lv] = dict(
                walkable=g(self.walkable(lv)),
                spaces={n: g(s) for n, s in self.spaces[lv].items()},
                obstacles={n: g(o) for n, o in self.obstacles[lv].items()},
                doors=[dict(name=dr.name, kind=dr.kind, width=dr.width * k, note=dr.note,
                            center=[dr.center[0] * k, dr.center[1] * k], poly=g(dr.poly))
                       for dr in self.doors if dr.level == lv],
                seats=[{kk: (v * k if kk in ("x", "y") else v) for kk, v in s.items() if not kk.startswith("_")}
                       for s in self.seats(lv)],
                spawn_points=[dict(s, x=s["x"] * k, y=s["y"] * k, clear=s["clear"] * k) for s in self.spawn_points(lv)],
            )
        for st in self.stairs:
            d["stairs"].append(dict(name=st.name, level_from=st.level_from, p_from=[st.p_from[0] * k, st.p_from[1] * k],
                                    level_to=st.level_to, p_to=[st.p_to[0] * k, st.p_to[1] * k],
                                    length=st.length * k, width=st.width * k, lanes=st.lanes, note=st.note))
        for n, o in self.outdoor.items():
            d["outdoor"][n] = g(o)
        return d

    def save_json(self, path: str, units: str = "m"):
        with open(path, "w") as f:
            json.dump(self.to_dict(units), f)

    # --------------------------------------------------------- image registration ---
    def ft_to_px(self, level: str):
        """Return f(x_ft, y_ft) -> (px, py) on the corresponding evacuation-plan image."""
        if level == "first_floor":
            s, (cx, cy) = self.p.plan_px_per_ft, self.p.plan_center_px
        else:
            s, (cx, cy) = self.p.balcony_px_per_ft, self.p.balcony_center_px
        return lambda x, y: (cx + x * s, cy - y * s)


if __name__ == "__main__":
    m = Model()
    for lv in Model.LEVELS:
        w = m.walkable(lv)
        print(lv, "walkable area %.0f ft2 (%.0f m2), parts=%d" % (w.area, w.area * FT * FT,
              len(w.geoms) if hasattr(w, "geoms") else 1), "doors", len([d for d in m.doors if d.level == lv]))
    print("seat counts", m.seat_counts())
    print("stairs", [(s.name, s.length) for s in m.stairs])
