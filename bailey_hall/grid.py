"""
Rasterise the Bailey Hall model onto a dense square grid (for cellular-automaton / floor-field /
lattice-gas style simulations).

    from grid import Grid
    g = Grid.from_model(level="flat", cell_ft=0.5)        # "first_floor", "balcony" or "flat"
    g.walkable        # bool  (H, W)   True = floor a person can stand on
    g.exit_id         # int16 (H, W)   -1 = none, else index into g.exits (names)
    g.door_id         # int16 (H, W)   -1 = none, else index into g.doors (interior doors incl. exits)
    g.dist            # float (H, W)   walking distance to the nearest exit in FEET (inf = unreachable)
    g.spawn           # int32 (N, 2)   (row, col) of one cell per seat = where each person starts
    g.seat_section    # list[str]      section name per spawn cell
    g.xy(i, j) / g.ij(x, y)            # grid <-> feet conversions (cell centres)
    g.save("out/grid_flat_0.5ft.npz") / Grid.load(path)

Rows go north -> south (row 0 is the northern edge), columns west -> east, like an image.
Cell test = cell CENTRE inside the walkable polygon, so a corridor narrower than one cell can
disappear: keep cell_ft <= half the narrowest passage you care about (row walkways are
1.33 ft in the orchestra, 1.2-1.6 ft elsewhere) or run at 0.5 ft and let an agent occupy
several cells.

    python bailey_hall/grid.py [cell_ft] [level]     -> out/grid_<level>_<cell>ft.npz + .png
"""
from __future__ import annotations

import heapq
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon

from model import FT, Model

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Grid:
    cell: float                 # cell size in feet
    x0: float                   # x of the WEST edge of column 0 (feet)
    y0: float                   # y of the NORTH edge of row 0 (feet)
    walkable: np.ndarray
    exit_id: np.ndarray
    door_id: np.ndarray
    dist: np.ndarray
    spawn: np.ndarray
    seat_section: List[str]
    exits: List[str]
    doors: List[str]
    level: str = ""

    # ------------------------------------------------------------- conversions --
    def xy(self, i, j):
        """centre of cell (i, j) in feet"""
        return self.x0 + (np.asarray(j) + 0.5) * self.cell, self.y0 - (np.asarray(i) + 0.5) * self.cell

    def ij(self, x, y):
        return (np.floor((self.y0 - np.asarray(y)) / self.cell).astype(int),
                np.floor((np.asarray(x) - self.x0) / self.cell).astype(int))

    @property
    def shape(self):
        return self.walkable.shape

    @property
    def cell_m(self):
        return self.cell * FT

    # --------------------------------------------------------------- building --
    @classmethod
    def from_model(cls, model: Optional[Model] = None, level: str = "flat", cell_ft: float = 0.5,
                   margin_ft: float = 2.0, include_outdoor: bool = False) -> "Grid":
        m = model or Model()
        if level == "flat":
            flat = m.flattened()
            walk = flat["walkable"]
            exits = flat["exits"]                           # [(name, poly)]
            doors = [(d.name, d.poly) for d in m.doors if d.level == "first_floor"]
            spawn_pts = flat["spawn"]
        else:
            walk = m.walkable(level)
            exits = [(d.name, d.poly) for d in m.exits(level)]
            doors = [(d.name, d.poly) for d in m.doors if d.level == level]
            spawn_pts = m.spawn_points(level)
        if include_outdoor and level != "balcony":
            walk = shapely.union_all([walk] + list(m.outdoor.values())).buffer(0)

        minx, miny, maxx, maxy = walk.bounds
        x0, y1 = minx - margin_ft, maxy + margin_ft
        W = int(math.ceil((maxx + margin_ft - x0) / cell_ft))
        H = int(math.ceil((y1 - (miny - margin_ft)) / cell_ft))
        jj, ii = np.meshgrid(np.arange(W), np.arange(H))
        cx = x0 + (jj + 0.5) * cell_ft
        cy = y1 - (ii + 0.5) * cell_ft
        walkable = shapely.contains_xy(walk, cx.ravel(), cy.ravel()).reshape(H, W)

        def id_map(items):
            ids = np.full((H, W), -1, np.int16)
            for k, (_, poly) in enumerate(items):
                # a door is a thin rectangle; test cell centres against the slightly grown polygon
                hit = shapely.contains_xy(poly.buffer(cell_ft * 0.5), cx.ravel(), cy.ravel()).reshape(H, W)
                ids[hit & walkable] = k
            return ids

        exit_id = id_map(exits)
        door_id = id_map(doors)

        # spawn cells: snap each spawn point to its cell; if that cell is not walkable (coarse
        # grid) take the nearest walkable cell within 1 cell
        spawn, sections = [], []
        for s in spawn_pts:
            i, j = int((y1 - s["y"]) // cell_ft), int((s["x"] - x0) // cell_ft)
            if not walkable[i, j]:
                cand = [(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                        if 0 <= i + di < H and 0 <= j + dj < W and walkable[i + di, j + dj]]
                if not cand:
                    continue
                i, j = min(cand, key=lambda c: (c[0] - i) ** 2 + (c[1] - j) ** 2)
            spawn.append((i, j)); sections.append(s["section"])
        spawn = np.array(spawn, np.int32).reshape(-1, 2)

        g = cls(cell_ft, x0, y1, walkable, exit_id, door_id, np.full((H, W), np.inf),
                spawn, sections, [n for n, _ in exits], [n for n, _ in doors], level)
        g.dist = g.distance_field(exit_id >= 0)
        return g

    # ------------------------------------------------------------- algorithms --
    def distance_field(self, targets: np.ndarray, blocked: Optional[np.ndarray] = None) -> np.ndarray:
        """Dijkstra over walkable cells (8-connected, diagonal = sqrt 2) from all target cells.
        Returns distances in FEET; inf where unreachable.  Pass `blocked` (bool mask) to close
        doors / add barriers without rebuilding the grid."""
        H, W = self.shape
        free = self.walkable.copy()
        if blocked is not None:
            free &= ~blocked
        dist = np.full((H, W), np.inf)
        heap = []
        for i, j in zip(*np.nonzero(targets & free)):
            dist[i, j] = 0.0
            heap.append((0.0, int(i), int(j)))
        heapq.heapify(heap)
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
              (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        while heap:
            d, i, j = heapq.heappop(heap)
            if d > dist[i, j]:
                continue
            for di, dj, w in nb:
                a, b = i + di, j + dj
                if 0 <= a < H and 0 <= b < W and free[a, b]:
                    if w > 1 and not (free[i, b] and free[a, j]):   # no corner cutting through walls
                        continue
                    nd = d + w
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(heap, (nd, a, b))
        return dist * self.cell

    def components(self) -> np.ndarray:
        """label connected walkable regions (4-connected); 0 = not walkable"""
        from scipy import ndimage
        lab, _ = ndimage.label(self.walkable)
        return lab

    # ---------------------------------------------------------------- io / viz --
    def save(self, path: str):
        np.savez_compressed(path, cell=self.cell, x0=self.x0, y0=self.y0, walkable=self.walkable,
                            exit_id=self.exit_id, door_id=self.door_id, dist=self.dist, spawn=self.spawn,
                            seat_section=np.array(self.seat_section), exits=np.array(self.exits),
                            doors=np.array(self.doors), level=self.level)

    @classmethod
    def load(cls, path: str) -> "Grid":
        z = np.load(path, allow_pickle=False)
        return cls(float(z["cell"]), float(z["x0"]), float(z["y0"]), z["walkable"], z["exit_id"], z["door_id"],
                   z["dist"], z["spawn"], list(z["seat_section"]), list(z["exits"]), list(z["doors"]), str(z["level"]))

    def render(self, path: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        H, W = self.shape
        ext = (self.x0, self.x0 + W * self.cell, self.y0 - H * self.cell, self.y0)
        fig, ax = plt.subplots(figsize=(10, 10 * H / W), dpi=140)
        d = np.where(np.isfinite(self.dist), self.dist, np.nan)
        ax.imshow(np.where(self.walkable, 1.0, np.nan), extent=ext, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        im = ax.imshow(d, extent=ext, cmap="viridis", interpolation="nearest")
        ax.imshow(np.where(self.exit_id >= 0, 1.0, np.nan), extent=ext, cmap="autumn", vmin=0, vmax=1, interpolation="nearest")
        xs, ys = self.xy(self.spawn[:, 0], self.spawn[:, 1])
        ax.scatter(xs, ys, s=0.5, c="w")
        plt.colorbar(im, ax=ax, label="walking distance to nearest exit [ft]", shrink=0.6)
        ax.set_title(f"{self.level}: {W}x{H} cells of {self.cell:g} ft ({self.cell_m:.2f} m), "
                     f"{int(self.walkable.sum())} walkable cells, {len(self.spawn)} people")
        ax.set_xlabel("x east [ft]"); ax.set_ylabel("y north [ft]")
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    cell = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    level = sys.argv[2] if len(sys.argv) > 2 else "flat"
    g = Grid.from_model(level=level, cell_ft=cell)
    lab = g.components()
    comp_of_spawn = set(lab[g.spawn[:, 0], g.spawn[:, 1]].tolist())
    reach = np.isfinite(g.dist[g.spawn[:, 0], g.spawn[:, 1]])
    print(f"{level}: grid {g.shape[1]}x{g.shape[0]} @ {cell} ft, walkable {int(g.walkable.sum())} cells, "
          f"{len(g.exits)} exits, {len(g.spawn)} spawn cells in {len(comp_of_spawn)} component(s), "
          f"{int(reach.sum())} can reach an exit, farthest seat {np.nanmax(g.dist[g.spawn[:,0], g.spawn[:,1]][reach]):.0f} ft")
    out = os.path.join(HERE, "out", f"grid_{level}_{cell:g}ft")
    g.save(out + ".npz"); g.render(out + ".png"); print(out + ".npz", out + ".png")
