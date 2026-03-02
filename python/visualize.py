"""Geometric proof visualization: coordinate synthesis + diagram rendering.

Generates coordinates for JGEX problems (which have no coordinates),
then renders static proof diagrams showing:
  - Initial configuration in black
  - Auxiliary constructions in blue
  - Goal relation highlighted in green (proved) or red (unproved)
  - Geometric annotations (perpendicular marks, parallel arrows, equal ticks)

Usage:
    python python/visualize.py --problem "orthocenter" --problems problems/jgex_ag_231.txt
    python python/visualize.py --problem "morley" --problems problems/jgex_ag_231.txt --output diagrams/
    python python/visualize.py --list-solved --problems problems/jgex_ag_231.txt
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import geoprover
from orchestrate import load_problems


# ============================================================
# Coordinate Synthesis
# ============================================================

@dataclass
class GeoCoords:
    """Coordinates for all named points in a geometry problem."""
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    circles: list[tuple[str, float, float, float]] = field(default_factory=list)  # (center_name, cx, cy, r)
    construction_points: set[str] = field(default_factory=set)  # points added by constructions
    initial_points: set[str] = field(default_factory=set)  # base triangle/polygon points
    edges: list[tuple[str, str, str]] = field(default_factory=list)  # (p1, p2, style): "initial" or "construction"

    def add(self, name: str, x: float, y: float, is_construction: bool = False) -> None:
        self.points[name] = (x, y)
        if is_construction:
            self.construction_points.add(name)
        else:
            self.initial_points.add(name)

    def get(self, name: str) -> tuple[float, float] | None:
        return self.points.get(name)


def _midpoint(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _foot_of_perpendicular(
    p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> tuple[float, float]:
    """Project point p onto line ab."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return a
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    return (ax + t * dx, ay + t * dy)


def _circumcenter(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float]:
    """Circumcenter of triangle abc."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return _midpoint(a, b)
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy)


def _orthocenter(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float]:
    """Orthocenter of triangle abc."""
    # H = A + B + C - 2*O where O is circumcenter
    o = _circumcenter(a, b, c)
    return (a[0] + b[0] + c[0] - 2 * o[0], a[1] + b[1] + c[1] - 2 * o[1])


def _incenter(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]
) -> tuple[float, float]:
    """Incenter of triangle abc."""
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ca = math.dist(c, a)
    s = ab + bc + ca
    if s < 1e-10:
        return a
    return ((bc * a[0] + ca * b[0] + ab * c[0]) / s,
            (bc * a[1] + ca * b[1] + ab * c[1]) / s)


def _line_intersection(
    p1: tuple[float, float], d1: tuple[float, float],
    p2: tuple[float, float], d2: tuple[float, float],
) -> tuple[float, float] | None:
    """Intersection of line (p1 + t*d1) and (p2 + s*d2)."""
    det = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(det) < 1e-10:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / det
    return (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _reflect(p: tuple[float, float], center: tuple[float, float]) -> tuple[float, float]:
    return (2 * center[0] - p[0], 2 * center[1] - p[1])


def _on_circle_point(
    center: tuple[float, float], radius: float, angle_deg: float
) -> tuple[float, float]:
    """Point on circle at given angle."""
    rad = math.radians(angle_deg)
    return (center[0] + radius * math.cos(rad), center[1] + radius * math.sin(rad))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _strip_self(name: str, args: list[str]) -> list[str]:
    """Remove the output name from args if it appears as first arg (JGEX convention)."""
    if args and args[0] == name:
        return args[1:]
    return args


def _resolve_on_line(name: str, predicates: list[str], coords: GeoCoords) -> tuple[float, float] | None:
    """Try to resolve a point defined by multiple on_line/on_tline/on_pline/on_circle constraints."""
    lines = []  # each entry: (point_on_line, direction) for line constraints
    circles = []  # each entry: (center, radius_point) for circle constraints

    for pred_str in predicates:
        tokens = pred_str.strip().split()
        ptype = tokens[0] if tokens else ""
        pargs = _strip_self(name, tokens[1:])

        if ptype == "on_line" and len(pargs) >= 2:
            pa = coords.get(pargs[0])
            pb = coords.get(pargs[1])
            if pa and pb:
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                    lines.append((pa, (dx, dy), (pargs[0], pargs[1])))

        elif ptype == "on_tline" and len(pargs) >= 3:
            # on_tline x a b c = x on line through a, perpendicular to bc
            pa = coords.get(pargs[0])
            pb = coords.get(pargs[1])
            pc = coords.get(pargs[2])
            if pa and pb and pc:
                dx, dy = pc[0] - pb[0], pc[1] - pb[1]
                # perpendicular direction
                if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                    lines.append((pa, (-dy, dx), (name, pargs[0])))

        elif ptype == "on_pline" and len(pargs) >= 3:
            # on_pline x a b c = x on line through a, parallel to bc
            pa = coords.get(pargs[0])
            pb = coords.get(pargs[1])
            pc = coords.get(pargs[2])
            if pa and pb and pc:
                dx, dy = pc[0] - pb[0], pc[1] - pb[1]
                if abs(dx) > 1e-10 or abs(dy) > 1e-10:
                    lines.append((pa, (dx, dy), (name, pargs[0])))

        elif ptype == "on_circle" and len(pargs) >= 2:
            center = coords.get(pargs[0])
            rp = coords.get(pargs[1])
            if center and rp:
                circles.append((center, _dist(center, rp), pargs[0]))

    # Two lines → intersection
    if len(lines) >= 2:
        result = _line_intersection(lines[0][0], lines[0][1], lines[1][0], lines[1][1])
        if result:
            # Record edges: the point lies on both lines
            for _, _, edge_pts in lines:
                coords.edges.append((name, edge_pts[0], "construction"))
                coords.edges.append((name, edge_pts[1], "construction"))
            return result

    # Line + circle → pick the intersection closest to other known points
    if len(lines) >= 1 and len(circles) >= 1:
        p0, d, edge_pts = lines[0]
        center, r, cname = circles[0]
        # Parametric: p0 + t*d, solve |p0+t*d - center|^2 = r^2
        ocx, ocy = p0[0] - center[0], p0[1] - center[1]
        a_coeff = d[0]**2 + d[1]**2
        b_coeff = 2 * (ocx * d[0] + ocy * d[1])
        c_coeff = ocx**2 + ocy**2 - r**2
        disc = b_coeff**2 - 4 * a_coeff * c_coeff
        if disc >= 0 and a_coeff > 1e-10:
            sqrt_disc = math.sqrt(disc)
            t1 = (-b_coeff + sqrt_disc) / (2 * a_coeff)
            t2 = (-b_coeff - sqrt_disc) / (2 * a_coeff)
            pt1 = (p0[0] + t1 * d[0], p0[1] + t1 * d[1])
            pt2 = (p0[0] + t2 * d[0], p0[1] + t2 * d[1])
            # Pick the point that's farther from existing points (more likely the "new" one)
            existing = [coords.get(n) for n in coords.points if coords.get(n)]
            if existing:
                avg = (sum(p[0] for p in existing) / len(existing),
                       sum(p[1] for p in existing) / len(existing))
                # Pick the one farther from the centroid for variety
                if _dist(pt1, avg) > _dist(pt2, avg):
                    result = pt1
                else:
                    result = pt2
            else:
                result = pt1
            coords.edges.append((name, edge_pts[0], "construction"))
            coords.edges.append((name, edge_pts[1], "construction"))
            return result

    return None


def synthesize_coordinates(definition: str) -> GeoCoords:
    """Synthesize coordinates for a JGEX problem definition.

    Uses a standard triangle placement and computes construction steps geometrically.
    """
    coords = GeoCoords()

    # Split into construction steps and goal
    parts = definition.split("?")
    constructions_str = parts[0].strip()

    # Parse each construction clause (separated by ;)
    clauses = [c.strip() for c in constructions_str.split(";") if c.strip()]

    # Track which base points we've placed
    aux_counter = 0

    for clause in clauses:
        # Parse: "names = predicate args" or "names = predicate args, predicate args"
        if "=" not in clause:
            continue
        lhs, rhs = clause.split("=", 1)
        output_names = lhs.strip().split()
        predicates = [p.strip() for p in rhs.split(",")]

        for name in output_names:
            if name in coords.points:
                continue

            first_pred = predicates[0].strip()
            tokens = first_pred.split()
            pred_type = tokens[0] if tokens else ""
            args = _strip_self(name, tokens[1:])

            if pred_type in ("triangle", "r_triangle", "iso_triangle", "r_iso_triangle",
                             "risos", "eq_triangle"):
                tri_names = output_names
                if len(tri_names) >= 3:
                    coords.add(tri_names[0], 0.0, 0.0)
                    coords.add(tri_names[1], 4.0, 0.0)
                    coords.add(tri_names[2], 1.5, 3.2)
                    for n in tri_names[:3]:
                        coords.initial_points.add(n)
                    coords.edges.append((tri_names[0], tri_names[1], "initial"))
                    coords.edges.append((tri_names[1], tri_names[2], "initial"))
                    coords.edges.append((tri_names[2], tri_names[0], "initial"))
                break

            elif pred_type == "segment":
                seg_names = output_names
                if len(seg_names) >= 2:
                    coords.add(seg_names[0], 0.0, 0.0)
                    coords.add(seg_names[1], 4.0, 0.0)
                    for n in seg_names[:2]:
                        coords.initial_points.add(n)
                    coords.edges.append((seg_names[0], seg_names[1], "initial"))
                break

            elif pred_type == "pentagon":
                pent_names = output_names
                for i, n in enumerate(pent_names[:5]):
                    angle = math.pi / 2 + 2 * math.pi * i / 5
                    coords.add(n, 2 + 2 * math.cos(angle), 2 + 2 * math.sin(angle))
                    coords.initial_points.add(n)
                for i in range(len(pent_names[:5])):
                    j = (i + 1) % len(pent_names[:5])
                    coords.edges.append((pent_names[i], pent_names[j], "initial"))
                break

            elif pred_type in ("free",):
                coords.add(name, 2.5 + aux_counter * 0.5, 1.5 + aux_counter * 0.3, is_construction=True)
                aux_counter += 1

            elif pred_type == "midpoint":
                if len(args) >= 2:
                    a_name, b_name = args[0], args[1]
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    if pa and pb:
                        coords.add(name, *_midpoint(pa, pb), is_construction=True)
                        coords.edges.append((name, a_name, "construction"))
                        coords.edges.append((name, b_name, "construction"))
                    else:
                        coords.add(name, 2.0, 1.6, is_construction=True)

            elif pred_type == "foot":
                # foot f p a b = foot of perp from p onto line ab
                if len(args) >= 3:
                    p_name, a_name, b_name = args[0], args[1], args[2]
                    pp = coords.get(p_name)
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    if pp and pa and pb:
                        coords.add(name, *_foot_of_perpendicular(pp, pa, pb), is_construction=True)
                        # Edge from foot to the point it's the foot of, and along the base line
                        coords.edges.append((name, p_name, "construction"))
                    else:
                        coords.add(name, 2.0, 0.0, is_construction=True)

            elif pred_type in ("circle", "circumcenter"):
                if len(args) >= 3:
                    a_name, b_name, c_name = args[0], args[1], args[2]
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    pc = coords.get(c_name)
                    if pa and pb and pc:
                        cc = _circumcenter(pa, pb, pc)
                        coords.add(name, *cc, is_construction=True)
                        r = _dist(cc, pa)
                        coords.circles.append((name, cc[0], cc[1], r))
                    else:
                        coords.add(name, 2.0, 1.6, is_construction=True)

            elif pred_type == "orthocenter":
                if len(args) >= 3:
                    pa = coords.get(args[0])
                    pb = coords.get(args[1])
                    pc = coords.get(args[2])
                    if pa and pb and pc:
                        coords.add(name, *_orthocenter(pa, pb, pc), is_construction=True)
                    else:
                        coords.add(name, 2.0, 2.5, is_construction=True)

            elif pred_type == "incenter":
                if len(args) >= 3:
                    pa = coords.get(args[0])
                    pb = coords.get(args[1])
                    pc = coords.get(args[2])
                    if pa and pb and pc:
                        coords.add(name, *_incenter(pa, pb, pc), is_construction=True)
                    else:
                        coords.add(name, 1.8, 1.2, is_construction=True)

            elif pred_type == "mirror":
                if len(args) >= 2:
                    pp = coords.get(args[0])
                    pc = coords.get(args[1])
                    if pp and pc:
                        coords.add(name, *_reflect(pp, pc), is_construction=True)
                    else:
                        coords.add(name, 3.0, 1.0, is_construction=True)

            elif pred_type == "intersection_ll":
                # intersection_ll f a b c d = intersection of line(a,b) and line(c,d)
                if len(args) >= 4:
                    pa = coords.get(args[0])
                    pb = coords.get(args[1])
                    pc = coords.get(args[2])
                    pd = coords.get(args[3])
                    if pa and pb and pc and pd:
                        da = (pb[0] - pa[0], pb[1] - pa[1])
                        dc = (pd[0] - pc[0], pd[1] - pc[1])
                        result = _line_intersection(pa, da, pc, dc)
                        if result:
                            coords.add(name, *result, is_construction=True)
                            coords.edges.append((name, args[0], "construction"))
                            coords.edges.append((name, args[1], "construction"))
                            coords.edges.append((name, args[2], "construction"))
                            coords.edges.append((name, args[3], "construction"))
                        else:
                            coords.add(name, 2.0 + aux_counter * 0.4, 1.5, is_construction=True)
                    else:
                        coords.add(name, 2.0 + aux_counter * 0.4, 1.5, is_construction=True)
                    aux_counter += 1

            elif pred_type in ("on_line", "on_tline", "on_pline", "on_circle"):
                # Multi-predicate: try to resolve from all predicates
                result = _resolve_on_line(name, predicates, coords)
                if result:
                    coords.add(name, *result, is_construction=True)
                else:
                    # Fallback: place near referenced points
                    ref_points = []
                    for pred in predicates:
                        for tok in pred.split()[1:]:
                            if tok != name:
                                pt = coords.get(tok)
                                if pt:
                                    ref_points.append(pt)
                    if ref_points:
                        cx = sum(p[0] for p in ref_points) / len(ref_points)
                        cy = sum(p[1] for p in ref_points) / len(ref_points)
                        offset = 0.3 + 0.2 * aux_counter
                        coords.add(name, cx + offset, cy + offset * 0.7, is_construction=True)
                    else:
                        coords.add(name, 2.0 + aux_counter * 0.4, 1.5 + aux_counter * 0.3, is_construction=True)
                    aux_counter += 1

            else:
                # Unknown predicate — place near referenced points
                if name not in coords.points:
                    ref_points = []
                    for tok in args:
                        pt = coords.get(tok)
                        if pt:
                            ref_points.append(pt)
                    if ref_points:
                        cx = sum(p[0] for p in ref_points) / len(ref_points)
                        cy = sum(p[1] for p in ref_points) / len(ref_points)
                        offset = 0.3 + 0.2 * aux_counter
                        coords.add(name, cx + offset, cy + offset * 0.7, is_construction=True)
                    else:
                        coords.add(name, 2.0 + aux_counter * 0.4, 1.5 + aux_counter * 0.3, is_construction=True)
                    aux_counter += 1

    return coords


# ============================================================
# Diagram Rendering
# ============================================================

def _parse_goal(goal_str: str) -> tuple[str, list[str]]:
    """Parse goal string like 'perp a h b c' into (type, [args])."""
    tokens = goal_str.strip().split()
    if not tokens:
        return ("", [])
    return (tokens[0], tokens[1:])


def render_diagram(
    name: str,
    definition: str,
    coords: GeoCoords,
    solved: bool = False,
    output_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (10, 10),
) -> str | None:
    """Render a static proof diagram for a geometry problem.

    Returns the output path if saved, None otherwise.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_aspect("equal")
    ax.set_title(name, fontsize=14, fontweight="bold")

    # Parse goal
    parts = definition.split("?")
    goal_str = parts[1].strip() if len(parts) > 1 else ""
    goal_type, goal_args = _parse_goal(goal_str)

    # Collect point positions for axis limits
    all_x = [p[0] for p in coords.points.values()]
    all_y = [p[1] for p in coords.points.values()]
    if not all_x:
        ax.text(0.5, 0.5, "No coordinates", transform=ax.transAxes, ha="center")
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return output_path
        return None

    margin = 1.0
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Draw circles
    for cname, cx, cy, r in coords.circles:
        circle = plt.Circle((cx, cy), r, fill=False, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.add_patch(circle)

    # Draw edges from explicit edge list
    drawn_edges = set()
    for e1, e2, style in coords.edges:
        key = (min(e1, e2), max(e1, e2))
        if key in drawn_edges:
            continue
        drawn_edges.add(key)
        p1 = coords.get(e1)
        p2 = coords.get(e2)
        if p1 and p2:
            if style == "initial":
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", linewidth=1.5, zorder=1)
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color="#4488cc", linewidth=1.0, linestyle="--",
                        alpha=0.7, zorder=0)

    # Highlight goal relation
    goal_color = "#22aa22" if solved else "#cc4444"
    goal_points_set = set(goal_args)
    if goal_type in ("perp", "para", "cong") and len(goal_args) >= 4:
        p1 = coords.get(goal_args[0])
        p2 = coords.get(goal_args[1])
        p3 = coords.get(goal_args[2])
        p4 = coords.get(goal_args[3])
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=goal_color, linewidth=2.5, zorder=2)
        if p3 and p4:
            ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color=goal_color, linewidth=2.5, zorder=2)
    elif goal_type == "cyclic" and len(goal_args) >= 4:
        gpts = [coords.get(a) for a in goal_args[:4] if coords.get(a)]
        if len(gpts) >= 3:
            cc = _circumcenter(gpts[0], gpts[1], gpts[2])
            r = _dist(cc, gpts[0])
            circle = plt.Circle(cc, r, fill=False, color=goal_color, linewidth=2.5, linestyle="-", zorder=2)
            ax.add_patch(circle)
    elif goal_type == "eqangle" and len(goal_args) >= 6:
        for i in range(0, 6, 3):
            pts = [coords.get(goal_args[i + j]) for j in range(3)]
            if all(pts):
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color=goal_color, linewidth=2, zorder=2)
                ax.plot([pts[2][0], pts[1][0]], [pts[2][1], pts[1][1]], color=goal_color, linewidth=2, zorder=2)

    # Draw points
    for pname, (px, py) in coords.points.items():
        if pname in coords.construction_points:
            color = "#4488cc"
            marker = "s"
            size = 30
        elif pname in goal_points_set:
            color = goal_color
            marker = "o"
            size = 50
        else:
            color = "black"
            marker = "o"
            size = 40
        ax.scatter(px, py, c=color, s=size, zorder=5, marker=marker, edgecolors="white", linewidths=0.5)
        # Label
        offset = 0.15
        ax.annotate(pname, (px, py), textcoords="offset points", xytext=(8, 8),
                    fontsize=10, fontweight="bold", color=color, zorder=6)

    # Legend
    handles = [
        mpatches.Patch(color="black", label="Initial config"),
        mpatches.Patch(color="#4488cc", label="Constructions"),
        mpatches.Patch(color=goal_color, label=f"Goal: {goal_str}" + (" (PROVED)" if solved else "")),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.grid(True, alpha=0.15)
    ax.set_xlabel("")
    ax.set_ylabel("")

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    if show:
        plt.show()
    else:
        plt.close(fig)
    return None


def render_proof_steps(
    name: str,
    definition: str,
    coords: GeoCoords,
    proof_actions: list[str],
    output_dir: str = "diagrams",
) -> list[str]:
    """Render a multi-step proof as a series of diagrams.

    Step 0: Initial configuration
    Step 1+: Each construction step
    Final: Goal highlighted
    """
    paths = []
    os.makedirs(output_dir, exist_ok=True)

    # Step 0: Initial configuration only
    initial_coords = GeoCoords()
    for pname in coords.initial_points:
        pt = coords.get(pname)
        if pt:
            initial_coords.add(pname, *pt)
            initial_coords.initial_points.add(pname)

    path = render_diagram(
        f"{name} - Step 0: Initial",
        definition,
        initial_coords,
        solved=False,
        output_path=os.path.join(output_dir, f"{_safe_filename(name)}_step0.png"),
    )
    if path:
        paths.append(path)

    # Final: Full diagram with all constructions
    path = render_diagram(
        f"{name} - Final: {'Proved' if proof_actions else 'Full config'}",
        definition,
        coords,
        solved=bool(proof_actions),
        output_path=os.path.join(output_dir, f"{_safe_filename(name)}_final.png"),
    )
    if path:
        paths.append(path)

    return paths


def _safe_filename(name: str) -> str:
    """Convert problem name to safe filename."""
    return re.sub(r'[^\w\-.]', '_', name)[:80]


# ============================================================
# MCTS Tree Visualization
# ============================================================

def render_mcts_tree(
    name: str,
    tree_data: dict,
    output_path: str = "diagrams/mcts_tree.png",
    max_depth: int = 3,
    min_visits: int = 2,
) -> str:
    """Render an MCTS search tree as a static diagram.

    tree_data format: {
        "action": str, "visits": int, "value": float,
        "children": [tree_data, ...]
    }
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_title(f"MCTS Search Tree: {name}", fontsize=14, fontweight="bold")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axis("off")

    def _draw_node(node, x, y, width, depth):
        if depth > max_depth or node.get("visits", 0) < min_visits:
            return

        visits = node.get("visits", 0)
        value = node.get("value", 0)
        action = node.get("action", "root")

        # Node size proportional to visits
        size = max(200, min(2000, visits * 50))
        # Color by value: red (0) -> yellow (0.5) -> green (1.0)
        color = plt.cm.RdYlGn(value)

        ax.scatter(x, y, s=size, c=[color], edgecolors="black", linewidths=1, zorder=3)

        label = f"{action[:12]}\nv={visits}"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(0, -20),
                    fontsize=7, ha="center", zorder=4)

        children = [c for c in node.get("children", []) if c.get("visits", 0) >= min_visits]
        if not children:
            return

        child_width = width / max(len(children), 1)
        start_x = x - width / 2 + child_width / 2

        for i, child in enumerate(children):
            cx = start_x + i * child_width
            cy = y - 0.3
            ax.plot([x, cx], [y, cy], "k-", linewidth=0.8, alpha=0.5, zorder=1)
            _draw_node(child, cx, cy, child_width * 0.8, depth + 1)

    _draw_node(tree_data, 0, 1.0, 1.8, 0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize geometry proofs")
    parser.add_argument("--problems", default="problems/jgex_ag_231.txt", help="Problem file")
    parser.add_argument("--problem", default=None, help="Problem name substring to visualize")
    parser.add_argument("--output", default="diagrams", help="Output directory")
    parser.add_argument("--list-solved", action="store_true", help="List problems solvable by deduction")
    parser.add_argument("--all-solved", action="store_true", help="Render all deduction-solved problems")
    args = parser.parse_args()

    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems")

    if args.list_solved:
        solved = []
        for name, definition in problems:
            try:
                state = geoprover.parse_problem(f"{name}\n{definition}")
                if geoprover.saturate(state):
                    solved.append(name)
            except Exception:
                pass
        print(f"\nDeduction-solved: {len(solved)}/{len(problems)}")
        for s in sorted(solved):
            print(f"  {s}")
        return

    # Find matching problems
    targets = []
    for name, definition in problems:
        if args.problem and args.problem.lower() not in name.lower():
            continue
        if args.all_solved:
            try:
                state = geoprover.parse_problem(f"{name}\n{definition}")
                if not geoprover.saturate(state):
                    continue
            except Exception:
                continue
        targets.append((name, definition))

    if not targets:
        print(f"No problems matching '{args.problem}'")
        return

    print(f"Rendering {len(targets)} problems...")
    os.makedirs(args.output, exist_ok=True)

    for name, definition in targets:
        print(f"  {name}...", end=" ", flush=True)
        try:
            coords = synthesize_coordinates(definition)
            # Check if solved by deduction
            state = geoprover.parse_problem(f"{name}\n{definition}")
            solved = geoprover.saturate(state)

            path = render_diagram(
                name, definition, coords,
                solved=solved,
                output_path=os.path.join(args.output, f"{_safe_filename(name)}.png"),
            )
            status = "PROVED" if solved else "unsolved"
            print(f"{status} -> {path}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nDiagrams saved to {args.output}/")


if __name__ == "__main__":
    main()
