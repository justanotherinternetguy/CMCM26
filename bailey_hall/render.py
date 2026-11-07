"""
Render the Bailey Hall model.

  python bailey_hall/render.py overlay   -> bailey_hall/out/overlay_first_floor.png, overlay_balcony.png
                                            (model drawn on top of the PDF evacuation plans, for checking)
  python bailey_hall/render.py clean     -> bailey_hall/out/model_first_floor.png, model_balcony.png, model_flat.png
  python bailey_hall/render.py svg       -> bailey_hall/out/model.svg  (metres, both levels)
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, PathPatch
from matplotlib.path import Path
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from model import FT, Model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PLAN_IMG = {"first_floor": os.path.join(HERE, "plans", "first_floor_plan.png"),
            "balcony": os.path.join(HERE, "plans", "balcony_plan.png")}


def _polys(g):
    if g.is_empty:
        return []
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def draw_poly(ax, g, tf=lambda x, y: (x, y), **kw):
    for poly in _polys(g):
        verts, codes = [], []
        for ring in [poly.exterior] + list(poly.interiors):
            pts = [tf(x, y) for x, y in ring.coords]
            verts += pts
            codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), **kw))


def draw_level(ax, m: Model, level: str, tf=lambda x, y: (x, y), seats=True, labels=False, alpha=0.45):
    walk = m.walkable(level)
    draw_poly(ax, walk, tf, facecolor="#9ad0f5", edgecolor="#1f5f8b", lw=0.6, alpha=alpha)
    for name, o in m.obstacles[level].items():
        draw_poly(ax, o, tf, facecolor="#444444", edgecolor="none", alpha=alpha + 0.2)
    for d in m.doors:
        if d.level != level:
            continue
        col = {"door": "#2ca02c", "exit": "#d62728", "stair-head": "#ff7f0e"}[d.kind]
        draw_poly(ax, d.poly, tf, facecolor=col, edgecolor="k", lw=0.4, alpha=0.9)
        if labels:
            cx, cy = tf(*d.center)
            ax.annotate(d.name, (cx, cy), fontsize=5, color=col)
    if level == "first_floor":
        for n, o in m.outdoor.items():
            draw_poly(ax, o, tf, facecolor="#c8e6c9", edgecolor="#2e7d32", lw=0.5, alpha=0.35)
    if seats:
        pts = m.seats(level)
        xs, ys = zip(*[tf(s["x"], s["y"]) for s in pts]) if pts else ([], [])
        ax.scatter(xs, ys, s=1.2, c="#ffeb3b", edgecolors="none", zorder=5)


def overlay(m: Model):
    os.makedirs(OUT, exist_ok=True)
    for level in Model.LEVELS:
        img = Image.open(PLAN_IMG[level]).convert("RGB")
        W, H = img.size
        fig, ax = plt.subplots(figsize=(W / 60, H / 60), dpi=120)
        ax.imshow(img, extent=(0, W, H, 0))
        draw_level(ax, m, level, m.ft_to_px(level), seats=True, labels=True, alpha=0.35)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
        out = os.path.join(OUT, f"overlay_{level}.png")
        fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(out)


def clean(m: Model):
    os.makedirs(OUT, exist_ok=True)
    for level in Model.LEVELS:
        fig, ax = plt.subplots(figsize=(12, 12), dpi=130)
        draw_level(ax, m, level, seats=True, labels=True, alpha=0.8)
        ax.set_aspect("equal"); ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_xlabel("x east [ft]"); ax.set_ylabel("y north [ft]")
        ax.set_title(f"Bailey Hall — {level.replace('_', ' ')} (feet; stage at top)")
        ax.autoscale()
        out = os.path.join(OUT, f"model_{level}.png")
        fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(out)
    flat = m.flattened()
    fig, ax = plt.subplots(figsize=(16, 9), dpi=130)
    draw_poly(ax, flat["walkable"], facecolor="#9ad0f5", edgecolor="#1f5f8b", lw=0.5, alpha=0.8)
    for n, g in flat["exits"]:
        draw_poly(ax, g, facecolor="#d62728", edgecolor="k", lw=0.3)
    xs = [s["x"] for s in flat["seats"]]; ys = [s["y"] for s in flat["seats"]]
    ax.scatter(xs, ys, s=0.8, c="#ffeb3b")
    ax.set_aspect("equal"); ax.autoscale(); ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title("Flattened single-plane layout (balcony mirrored south of the foyer; main balcony stairs drawn as 42 ft corridors)")
    out = os.path.join(OUT, "model_flat.png"); fig.savefig(out, bbox_inches="tight"); plt.close(fig); print(out)


def svg(m: Model):
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, level in zip(axes, Model.LEVELS):
        draw_level(ax, m, level, tf=lambda x, y: (x * FT, y * FT), seats=True, labels=False, alpha=0.8)
        ax.set_aspect("equal"); ax.autoscale(); ax.grid(True, lw=0.3, alpha=0.5)
        ax.set_title(level); ax.set_xlabel("m"); ax.set_ylabel("m")
    out = os.path.join(OUT, "model.svg"); fig.savefig(out); plt.close(fig); print(out)


if __name__ == "__main__":
    m = Model()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("overlay", "all"):
        overlay(m)
    if what in ("clean", "all"):
        clean(m)
    if what in ("svg", "all"):
        svg(m)
