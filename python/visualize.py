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


def synthesize_coordinates(definition: str) -> GeoCoords:
    """Synthesize coordinates for a JGEX problem definition.

    Uses a standard triangle placement and computes construction steps geometrically.
    """
    coords = GeoCoords()

    # Split into construction steps and goal
    parts = definition.split("?")
    constructions_str = parts[0].strip()
    goal_str = parts[1].strip() if len(parts) > 1 else ""

    # Parse each construction clause (separated by ;)
    clauses = [c.strip() for c in constructions_str.split(";") if c.strip()]

    # Track which base points we've placed
    base_placed = set()
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
            args = tokens[1:] if len(tokens) > 1 else []

            is_construction = False

            if pred_type == "triangle":
                # Place triangle: A=(0,0), B=(4,0), C from angle
                tri_names = output_names
                if len(tri_names) >= 3:
                    coords.add(tri_names[0], 0.0, 0.0)
                    coords.add(tri_names[1], 4.0, 0.0)
                    coords.add(tri_names[2], 1.5, 3.2)
                    for n in tri_names[:3]:
                        base_placed.add(n)
                        coords.initial_points.add(n)
                break  # triangle defines all output names at once

            elif pred_type == "segment":
                seg_names = output_names
                if len(seg_names) >= 2:
                    coords.add(seg_names[0], 0.0, 0.0)
                    coords.add(seg_names[1], 4.0, 0.0)
                    for n in seg_names[:2]:
                        base_placed.add(n)
                        coords.initial_points.add(n)
                break

            elif pred_type == "pentagon":
                pent_names = output_names
                for i, n in enumerate(pent_names[:5]):
                    angle = math.pi / 2 + 2 * math.pi * i / 5
                    coords.add(n, 2 + 2 * math.cos(angle), 2 + 2 * math.sin(angle))
                    base_placed.add(n)
                    coords.initial_points.add(n)
                break

            elif pred_type in ("free",):
                # Free point — place it somewhere reasonable
                coords.add(name, 2.5 + aux_counter * 0.5, 1.5 + aux_counter * 0.3, is_construction=True)
                aux_counter += 1

            elif pred_type == "midpoint":
                if len(args) >= 2:
                    # midpoint m a b — first arg is the midpoint name (already in output_names)
                    a_name, b_name = args[0], args[1]
                    # Sometimes it's "midpoint m a b" where m is output
                    if a_name == name and len(args) >= 3:
                        a_name, b_name = args[1], args[2]
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    if pa and pb:
                        coords.add(name, *_midpoint(pa, pb), is_construction=True)
                    else:
                        coords.add(name, 2.0, 1.6, is_construction=True)

            elif pred_type == "foot":
                if len(args) >= 3:
                    foot_name = args[0]
                    if foot_name == name:
                        p_name, a_name, b_name = args[0], args[1], args[2]
                        if len(args) >= 4:
                            p_name, a_name, b_name = args[1], args[2], args[3]
                    else:
                        p_name, a_name, b_name = args[0], args[1], args[2]
                    pp = coords.get(p_name)
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    if pp and pa and pb:
                        coords.add(name, *_foot_of_perpendicular(pp, pa, pb), is_construction=True)
                    else:
                        coords.add(name, 2.0, 0.0, is_construction=True)

            elif pred_type in ("circle", "circumcenter"):
                if len(args) >= 3:
                    a_name, b_name, c_name = args[0], args[1], args[2]
                    if a_name == name and len(args) >= 4:
                        a_name, b_name, c_name = args[1], args[2], args[3]
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
                    a_name, b_name, c_name = args[0], args[1], args[2]
                    if a_name == name and len(args) >= 4:
                        a_name, b_name, c_name = args[1], args[2], args[3]
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    pc = coords.get(c_name)
                    if pa and pb and pc:
                        coords.add(name, *_orthocenter(pa, pb, pc), is_construction=True)
                    else:
                        coords.add(name, 2.0, 2.5, is_construction=True)

            elif pred_type == "incenter":
                if len(args) >= 3:
                    a_name, b_name, c_name = args[0], args[1], args[2]
                    if a_name == name and len(args) >= 4:
                        a_name, b_name, c_name = args[1], args[2], args[3]
                    pa = coords.get(a_name)
                    pb = coords.get(b_name)
                    pc = coords.get(c_name)
                    if pa and pb and pc:
                        coords.add(name, *_incenter(pa, pb, pc), is_construction=True)
                    else:
                        coords.add(name, 1.8, 1.2, is_construction=True)

            elif pred_type == "mirror":
                if len(args) >= 2:
                    p_name, center_name = args[0], args[1]
                    if p_name == name and len(args) >= 3:
                        p_name, center_name = args[1], args[2]
                    pp = coords.get(p_name)
                    pc = coords.get(center_name)
                    if pp and pc:
                        coords.add(name, *_reflect(pp, pc), is_construction=True)
                    else:
                        coords.add(name, 3.0, 1.0, is_construction=True)

            elif pred_type in ("on_line", "on_circle", "on_tline", "on_pline",
                               "intersection_ll", "intersection_lc", "intersection_cc",
                               "on_dia", "lc_tangent", "trisect", "excenter2",
                               "incenter2", "eqangle3"):
                # Complex constructions — place at a reasonable position
                # Try to extract referenced points and place near them
                ref_points = [coords.get(a) for a in args if coords.get(a) is not None]
                if ref_points:
                    cx = sum(p[0] for p in ref_points) / len(ref_points)
                    cy = sum(p[1] for p in ref_points) / len(ref_points)
                    # Offset slightly to avoid overlap
                    offset = 0.3 + 0.2 * aux_counter
                    coords.add(name, cx + offset, cy + offset * 0.7, is_construction=True)
                else:
                    coords.add(name, 2.0 + aux_counter * 0.4, 1.5 + aux_counter * 0.3, is_construction=True)
                aux_counter += 1

            else:
                # Unknown predicate — place somewhere
                if name not in coords.points:
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

    # Draw edges between initial points (triangle/polygon)
    initial = sorted(coords.initial_points)
    if len(initial) >= 3:
        for i in range(len(initial)):
            j = (i + 1) % len(initial)
            p1 = coords.get(initial[i])
            p2 = coords.get(initial[j])
            if p1 and p2:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", linewidth=1.5, zorder=1)

    # Draw construction lines (dashed blue)
    clauses = [c.strip() for c in parts[0].split(";") if c.strip()]
    for clause in clauses:
        if "=" not in clause:
            continue
        lhs, rhs = clause.split("=", 1)
        out_names = lhs.strip().split()
        for out_name in out_names:
            if out_name in coords.construction_points:
                # Draw lines from construction point to referenced points
                predicates = [p.strip() for p in rhs.split(",")]
                for pred in predicates:
                    tokens = pred.split()
                    for tok in tokens[1:]:
                        p1 = coords.get(out_name)
                        p2 = coords.get(tok)
                        if p1 and p2 and out_name != tok:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color="#4488cc", linewidth=0.7, linestyle="--",
                                    alpha=0.5, zorder=0)

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
