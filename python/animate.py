"""Animated proof walkthrough and MCTS search visualization.

Generates:
1. Step-by-step proof animation (matplotlib animation or frame sequence)
2. MCTS tree diagram showing search process

Usage:
    python python/animate.py --problem "9point" --problems problems/jgex_ag_231.txt --output diagrams/
    python python/animate.py --problem "morley" --mode mcts --output diagrams/
    python python/animate.py --problem "orthocenter" --mode steps --output diagrams/
"""

import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import geoprover
from orchestrate import MctsConfig, MctsNode, load_problems, mcts_search, _is_solved, _q_value
from visualize import (
    GeoCoords, synthesize_coordinates, _safe_filename,
    _circumcenter, _dist,
)


# ============================================================
# Step-by-step proof animation
# ============================================================

def _parse_clauses(definition: str) -> tuple[list[tuple[list[str], str]], str]:
    """Parse definition into (output_names, predicate_str) pairs and goal."""
    parts = definition.split("?")
    goal = parts[1].strip() if len(parts) > 1 else ""
    clauses_str = parts[0].strip()
    clauses = []
    for c in clauses_str.split(";"):
        c = c.strip()
        if not c or "=" not in c:
            continue
        lhs, rhs = c.split("=", 1)
        names = lhs.strip().split()
        clauses.append((names, rhs.strip()))
    return clauses, goal


def render_proof_animation(
    name: str,
    definition: str,
    coords: GeoCoords,
    solved: bool,
    output_path: str,
    fps: int = 1,
) -> str:
    """Render an animated GIF showing proof construction step by step.

    Each frame adds one construction step, with the final frame showing
    the goal relation highlighted.
    """
    clauses, goal_str = _parse_clauses(definition)

    # Build frames: each frame shows points up to step i
    frames_data = []

    # Frame 0: just initial points
    visible = set(coords.initial_points)
    frames_data.append((set(visible), "Initial configuration", False))

    # One frame per construction clause
    for out_names, pred_str in clauses:
        for n in out_names:
            if n in coords.construction_points:
                visible.add(n)
        desc = f"Construct: {out_names[0]} = {pred_str[:40]}"
        frames_data.append((set(visible), desc, False))

    # Final frame: show goal
    frames_data.append((set(visible), f"Goal: {goal_str}" + (" PROVED" if solved else ""), solved))

    # If very few frames, add the initial one twice for visibility
    if len(frames_data) < 3:
        frames_data.insert(0, frames_data[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    all_x = [p[0] for p in coords.points.values()]
    all_y = [p[1] for p in coords.points.values()]
    margin = 1.0
    xlim = (min(all_x) - margin, max(all_x) + margin)
    ylim = (min(all_y) - margin, max(all_y) + margin)

    goal_tokens = goal_str.split()
    goal_type = goal_tokens[0] if goal_tokens else ""
    goal_args = goal_tokens[1:] if len(goal_tokens) > 1 else []

    def _draw_frame(frame_idx):
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.15)

        visible_pts, desc, show_goal_highlight = frames_data[frame_idx]

        ax.set_title(f"{name}\nStep {frame_idx}/{len(frames_data)-1}: {desc}",
                     fontsize=12, fontweight="bold")

        # Draw circles
        for cname, cx, cy, r in coords.circles:
            if cname in visible_pts:
                circle = plt.Circle((cx, cy), r, fill=False, color="#cccccc",
                                    linewidth=0.8, linestyle="--")
                ax.add_patch(circle)

        # Draw edges (only between visible points)
        drawn_edges = set()
        for e1, e2, style in coords.edges:
            if e1 not in visible_pts or e2 not in visible_pts:
                continue
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

        # Goal highlight
        goal_color = "#22aa22" if show_goal_highlight and solved else "#cc4444"
        if show_goal_highlight and goal_type in ("perp", "para", "cong") and len(goal_args) >= 4:
            for i in range(0, 4, 2):
                p1 = coords.get(goal_args[i])
                p2 = coords.get(goal_args[i + 1])
                if p1 and p2:
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                            color=goal_color, linewidth=2.5, zorder=2)
        elif show_goal_highlight and goal_type == "cyclic" and len(goal_args) >= 4:
            gpts = [coords.get(a) for a in goal_args[:4] if coords.get(a)]
            if len(gpts) >= 3:
                cc = _circumcenter(gpts[0], gpts[1], gpts[2])
                r = _dist(cc, gpts[0])
                circle = plt.Circle(cc, r, fill=False, color=goal_color,
                                    linewidth=2.5, zorder=2)
                ax.add_patch(circle)

        # Draw points
        for pname in visible_pts:
            pt = coords.get(pname)
            if not pt:
                continue
            if pname in coords.construction_points:
                color = "#4488cc"
                marker = "s"
                size = 30
            else:
                color = "black"
                marker = "o"
                size = 40

            # Highlight newly added points
            is_new = frame_idx > 0 and pname not in frames_data[frame_idx - 1][0]
            if is_new:
                ax.scatter(pt[0], pt[1], c="orange", s=120, zorder=4,
                           marker="o", alpha=0.4)

            ax.scatter(pt[0], pt[1], c=color, s=size, zorder=5, marker=marker,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(pname, pt, textcoords="offset points", xytext=(8, 8),
                        fontsize=10, fontweight="bold", color=color, zorder=6)

        # Stats
        step_text = f"Points: {len(visible_pts)}"
        ax.text(0.02, 0.02, step_text, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Create animation
    anim = animation.FuncAnimation(fig, _draw_frame, frames=len(frames_data),
                                    interval=1000 // fps, repeat=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if output_path.endswith(".gif"):
        anim.save(output_path, writer="pillow", fps=fps, dpi=100)
    else:
        # Save as individual frames
        for i in range(len(frames_data)):
            _draw_frame(i)
            frame_path = output_path.replace(".png", f"_frame{i:02d}.png")
            fig.savefig(frame_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    print(f"    Animation saved: {output_path} ({len(frames_data)} frames)")
    return output_path


# ============================================================
# MCTS Tree Visualization
# ============================================================

def _collect_tree_data(node: MctsNode, depth: int = 0, max_depth: int = 3) -> dict:
    """Recursively collect MCTS tree data for visualization."""
    action_str = repr(node.action)[:30] if node.action else "root"
    data = {
        "action": action_str,
        "visits": node.visits,
        "value": _q_value(node),
        "terminal": node.terminal_value,
        "depth": depth,
        "children": [],
    }
    if depth < max_depth:
        for child in sorted(node.children, key=lambda c: c.visits, reverse=True):
            if child.visits >= 1:
                data["children"].append(_collect_tree_data(child, depth + 1, max_depth))
    return data


def render_mcts_tree_diagram(
    name: str,
    tree_data: dict,
    output_path: str,
    max_nodes: int = 50,
) -> str:
    """Render MCTS search tree as a hierarchical diagram.

    Node size = visit count, color = value estimate (red→green).
    Winning path highlighted in gold.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_title(f"MCTS Search Tree: {name}", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Flatten tree to get layout positions
    nodes = []
    edges = []

    def _flatten(node, parent_idx, depth, h_pos, h_width):
        idx = len(nodes)
        nodes.append({
            "x": h_pos,
            "y": 1.0 - depth * 0.25,
            "visits": node["visits"],
            "value": node["value"],
            "action": node["action"],
            "terminal": node.get("terminal"),
        })
        if parent_idx >= 0:
            edges.append((parent_idx, idx))

        children = node.get("children", [])
        if not children or len(nodes) >= max_nodes:
            return

        child_width = h_width / max(len(children), 1)
        start = h_pos - h_width / 2 + child_width / 2
        for i, child in enumerate(children):
            if len(nodes) >= max_nodes:
                break
            _flatten(child, idx, depth + 1, start + i * child_width, child_width * 0.9)

    _flatten(tree_data, -1, 0, 0.5, 0.9)

    if not nodes:
        ax.text(0.5, 0.5, "No MCTS data", transform=ax.transAxes, ha="center")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Draw edges
    for pi, ci in edges:
        ax.plot([nodes[pi]["x"], nodes[ci]["x"]],
                [nodes[pi]["y"], nodes[ci]["y"]],
                "k-", linewidth=0.5, alpha=0.3, zorder=0)

    # Find winning path
    winning = set()
    def _find_winning(node_data, path):
        if node_data.get("terminal") == 1.0:
            winning.update(range(len(path)))
            return True
        for child in node_data.get("children", []):
            if _find_winning(child, path + [child]):
                return True
        return False
    _find_winning(tree_data, [tree_data])

    # Draw nodes
    max_visits = max(n["visits"] for n in nodes) if nodes else 1
    for i, n in enumerate(nodes):
        size = max(100, min(1500, (n["visits"] / max(max_visits, 1)) * 1500))
        value = max(0, min(1, n["value"]))
        color = plt.cm.RdYlGn(value)

        edge_color = "#FFD700" if i in winning else "black"
        edge_width = 3 if i in winning else 1

        ax.scatter(n["x"], n["y"], s=size, c=[color], edgecolors=edge_color,
                   linewidths=edge_width, zorder=3)

        label = f"{n['action'][:15]}\nv={n['visits']}"
        if n.get("terminal") == 1.0:
            label += "\nPROVED"
        ax.annotate(label, (n["x"], n["y"]), textcoords="offset points",
                    xytext=(0, -18), fontsize=6, ha="center", zorder=4)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.3, label="Value estimate")

    # Stats
    total_visits = tree_data["visits"]
    total_nodes = len(nodes)
    ax.text(0.02, 0.98, f"Total visits: {total_visits}\nNodes shown: {total_nodes}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    MCTS tree saved: {output_path} ({total_nodes} nodes)")
    return output_path


def run_mcts_and_visualize(
    name: str,
    definition: str,
    coords: GeoCoords,
    output_dir: str,
    mcts_iterations: int = 50,
    device: str = "cpu",
) -> tuple[bool, str | None]:
    """Run MCTS on a problem and visualize the search tree."""
    from model import GeoTransformer

    problem_text = f"{name}\n{definition}"
    state = geoprover.parse_problem(problem_text)

    # Check deduction first
    proved = geoprover.saturate(state)
    if proved:
        print(f"    Solved by deduction (skipping MCTS)")
        return True, None

    # Re-parse (saturate mutates state)
    state = geoprover.parse_problem(problem_text)

    model = GeoTransformer()
    config = MctsConfig(
        num_iterations=mcts_iterations,
        max_children=20,
        max_depth=3,
        max_seq_len=128,
    )

    print(f"    Running MCTS ({mcts_iterations} iterations)...", end=" ", flush=True)
    t0 = time.time()
    result = mcts_search(state, model, config, device)
    elapsed = time.time() - t0
    print(f"{'SOLVED' if result.solved else 'unsolved'} ({elapsed:.1f}s)")

    # We need to re-run to get the tree structure
    # (mcts_search returns SearchResult but not the tree)
    # Re-run to capture tree
    state2 = geoprover.parse_problem(problem_text)
    from orchestrate import _evaluate, _expand, _select_leaf, _backprop

    root = MctsNode(state=state2)
    _evaluate(root, model, config, device)
    _expand(root, config, model, device)

    for _ in range(min(mcts_iterations, 30)):
        leaf = _select_leaf(root, config.c_puct)
        if leaf.terminal_value is not None:
            _backprop(leaf, leaf.terminal_value)
            if leaf.terminal_value == 1.0:
                break
            continue
        value = _evaluate(leaf, model, config, device)
        if leaf.terminal_value == 1.0:
            _backprop(leaf, 1.0)
            break
        if leaf.depth < config.max_depth:
            _expand(leaf, config, model, device)
        _backprop(leaf, value)

    tree_data = _collect_tree_data(root, max_depth=3)

    tree_path = os.path.join(output_dir, f"{_safe_filename(name)}_mcts_tree.png")
    render_mcts_tree_diagram(name, tree_data, tree_path)

    return result.solved, tree_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Animated proof visualization")
    parser.add_argument("--problems", default="problems/jgex_ag_231.txt")
    parser.add_argument("--problem", required=True, help="Problem name substring")
    parser.add_argument("--mode", choices=["steps", "mcts", "both"], default="both")
    parser.add_argument("--output", default="diagrams")
    parser.add_argument("--mcts-iterations", type=int, default=30)
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    problems = load_problems(args.problems)

    targets = [(n, d) for n, d in problems if args.problem.lower() in n.lower()]
    if not targets:
        print(f"No problems matching '{args.problem}'")
        return

    os.makedirs(args.output, exist_ok=True)

    for name, definition in targets:
        print(f"\n{name}:")
        coords = synthesize_coordinates(definition)

        # Check if solved
        try:
            state = geoprover.parse_problem(f"{name}\n{definition}")
            solved = geoprover.saturate(state)
        except Exception:
            solved = False

        if args.mode in ("steps", "both"):
            gif_path = os.path.join(args.output, f"{_safe_filename(name)}_anim.gif")
            render_proof_animation(name, definition, coords, solved, gif_path, fps=args.fps)

        if args.mode in ("mcts", "both"):
            run_mcts_and_visualize(name, definition, coords, args.output,
                                   mcts_iterations=args.mcts_iterations)

    print(f"\nAll outputs in {args.output}/")


if __name__ == "__main__":
    main()
