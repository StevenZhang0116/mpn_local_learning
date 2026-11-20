#!/usr/bin/env python
# coding: utf-8
"""
Illustration of the two-hop associative recall task (train_mpn.py --task twohop).

One stored history is drawn twice, as the task's paired held-out set asks it: once
with a one-hop query (a B symbol, one lookup) and once with a two-hop query (an A
symbol, two lookups). The two trials share the stored tables, the delay and the
answer, and differ only in the final step.

  top row     the two tables f: A -> B and g: B -> C, and the route to the answer
  bottom row  the input the network sees (one column per time step, one row per
              symbol channel) and the target at the last step

The trial comes from the task's own generator (associative_recall_tasks.
two_hop_generator), so the figure shows exactly what training feeds the network.

    python scripts/tasks/illustrations/plot_twohop_task.py                  # K=4, delay=4
    python scripts/tasks/illustrations/plot_twohop_task.py --symbols 8 --delay 15

Writes twohop_task.png next to this file (twohop_task_K8_d15.png for the second
command); --output chooses another path.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/tasks: the task module and its _bootstrap
from associative_recall_tasks import two_hop_generator  # noqa: E402

# Reference palette of the dataviz skill: categorical slots 1-3 (all-pairs safe), one
# per symbol set. Text and outlines stay in ink, never in a set color.
SET_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
SURFACE, TEXT, MUTED, GRID, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2", "#b5b4ae"
NODE_SIZE = 760          # scatter area in points^2
NODE_RADIUS = 15.0       # the same node's radius in points, for where arrows stop


def tint(color, weight):
    """Mix a hex color with the surface: weight 0 is the color, 1 is the surface."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple((1.0 - weight) * rgb + weight * np.array(matplotlib.colors.to_rgb(SURFACE)))


def example_trial(n_symbols, delay, seed):
    """One history asked both ways. Returns a dict with the two input arrays (T, 3K),
    the tables f and g, the queried symbols a and b = f[a], the answer c = g[b] and
    the steps at which the two pairs the answer depends on were shown."""
    X, target, info = two_hop_generator(1, n_symbols, delay, paired=True,
                                        rng=np.random.RandomState(seed))
    X = X.numpy()
    K = n_symbols
    a, b, c = int(info["a"][0]), int(info["b"][0]), int(target[0])
    return {"K": K, "delay": delay, "T": X.shape[1],
            "one_hop": X[0], "two_hop": X[1],
            "f": info["f"][0], "g": info["g"][0], "a": a, "b": b, "c": c,
            "step_ab": int(np.argmax(X[0, :K, a])),              # (A_a, B_b) on screen
            "step_bc": K + int(np.argmax(X[0, K:2 * K, K + b]))}  # (B_b, C_c) on screen


def draw_tables(ax, trial, two_hop):
    """The two tables as three columns of symbols, with the route of this query."""
    K, f, g = trial["K"], trial["f"], trial["g"]
    a, b, c = trial["a"], trial["b"], trial["c"]
    xs = {"A": 0.0, "B": 1.0, "C": 2.0}
    ax.set_xlim(-0.6, 2.6)       # set first: the lookup labels below read the transform
    ax.set_ylim(-1.3, K - 0.05)
    ax.axis("off")

    def y(index):
        return K - 1 - index     # symbol 0 at the top

    def edge(start, end, route):
        style = (dict(arrowstyle="-|>", lw=2.6, color=TEXT, mutation_scale=16) if route
                 else dict(arrowstyle="-", lw=1.1, color=EDGE))
        ax.annotate("", xy=end, xytext=start, zorder=3 if route else 1,
                    arrowprops=dict(shrinkA=NODE_RADIUS, shrinkB=NODE_RADIUS + (2 if route else 0),
                                    **style))

    for i in range(K):
        edge((xs["A"], y(i)), (xs["B"], y(f[i])), two_hop and i == a)
        edge((xs["B"], y(i)), (xs["C"], y(g[i])), i == b)

    on_route = {"A": {a} if two_hop else set(), "B": {b}, "C": {c}}
    for name, x in xs.items():
        for i in range(K):
            used = i in on_route[name]
            ax.scatter([x], [y(i)], s=NODE_SIZE, zorder=4,
                       facecolor=tint(SET_COLORS[name], 0.35 if used else 0.78),
                       edgecolor=TEXT if used else SET_COLORS[name],
                       linewidth=2.2 if used else 1.2)
            ax.text(x, y(i), f"{name}{i}", ha="center", va="center", zorder=5, fontsize=10,
                    color=TEXT, fontweight="bold" if used else "normal")
        ax.text(x, K - 0.5, f"set {name}", ha="center", va="bottom", fontsize=11,
                color=TEXT, fontweight="bold")
    captions = ((0.5, f"table f: A → B\nshown at steps 0–{K - 1}"),
                (1.5, f"table g: B → C\nshown at steps {K}–{2 * K - 1}"))
    for x, label in captions:
        ax.text(x, -0.62, label, ha="center", va="top", fontsize=8.5, color=MUTED,
                linespacing=1.25)

    def tag(text, xy, side):
        ax.annotate(text, xy=xy, xytext=(side * (NODE_RADIUS + 7), 0), zorder=6,
                    textcoords="offset points", ha="right" if side < 0 else "left",
                    va="center", fontsize=10, color=TEXT, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="none"))

    def lookup(text, start, end):
        """Label a route arrow: text along it, on its upper side."""
        mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        delta = ax.transData.transform(end) - ax.transData.transform(start)
        normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        normal = normal if normal[1] > 0 else -normal
        ax.annotate(text, xy=mid, xytext=tuple(11.0 * normal), textcoords="offset points",
                    ha="center", va="center", fontsize=9, color=TEXT, zorder=2,
                    rotation=np.degrees(np.arctan2(delta[1], delta[0])), rotation_mode="anchor",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    tag("answer", (xs["C"], y(c)), +1)
    if two_hop:
        tag("query", (xs["A"], y(a)), -1)
        lookup("lookup 1", (xs["A"], y(a)), (xs["B"], y(b)))
        lookup("lookup 2", (xs["B"], y(b)), (xs["C"], y(c)))
    else:
        tag("query", (xs["B"], y(b)), -1)
        lookup("one lookup", (xs["B"], y(b)), (xs["C"], y(c)))


def draw_inputs(ax, trial, two_hop):
    """The input raster (time x symbol channel) and the target at the last step."""
    K, T, delay = trial["K"], trial["T"], trial["delay"]
    X = trial["two_hop"] if two_hop else trial["one_hop"]
    names = [f"{s}{i}" for s in "ABC" for i in range(K)]
    needed = {trial["step_bc"]} | ({trial["step_ab"]} if two_hop else set())

    for t in range(T + 1):
        ax.plot([t, t], [0, 3 * K], color=GRID, lw=0.5, zorder=1)
    for r in range(3 * K + 1):
        boundary = r % K == 0
        ax.plot([0, T], [r, r], color=MUTED if boundary else GRID,
                lw=1.0 if boundary else 0.5, zorder=2 if boundary else 1)
    for t, r in zip(*np.nonzero(X)):
        ax.add_patch(Rectangle((t + 0.08, r + 0.08), 0.84, 0.84, zorder=3,
                               facecolor=SET_COLORS["ABC"[r // K]],
                               edgecolor=TEXT if t in needed else "none", linewidth=1.8))
    ax.add_patch(Rectangle((T - 1, 0), 1, 3 * K, fill=False, edgecolor=TEXT, linewidth=1.6,
                           zorder=4, clip_on=False))

    # The target: one of the K outputs, drawn beside the C rows it corresponds to.
    x0 = T + 1.0
    for i in range(K):
        hit = i == trial["c"]
        ax.add_patch(Rectangle((x0 + 0.08, 2 * K + i + 0.08), 0.84, 0.84, zorder=3, clip_on=False,
                               facecolor=SET_COLORS["C"] if hit else SURFACE,
                               edgecolor=TEXT if hit else GRID, linewidth=1.8 if hit else 0.8))
    ax.text(x0 + 0.5, 2 * K - 0.35, "target", ha="center", va="bottom", fontsize=9,
            color=TEXT, fontweight="bold", clip_on=False)
    ax.text(x0 + 0.5, 3 * K + 0.35, "scored at\nthe last\nstep only", ha="center", va="top",
            fontsize=7.5, color=MUTED, linespacing=1.1, clip_on=False)

    # Phases of the trial, bracketed above the raster.
    phases = [(0, K, "A–B pairs"), (K, 2 * K, "B–C pairs"), (T - 1, T, "query")]
    if delay > 0:
        phases.insert(2, (2 * K, 2 * K + delay, "delay (blank)" if delay >= 3 else "delay"))
    for start, stop, label in phases:
        ax.plot([start + 0.12, stop - 0.12], [-0.3, -0.3], color=MUTED, lw=1.2, clip_on=False)
        ax.text((start + stop) / 2.0, -0.5, label, ha="center", va="bottom", fontsize=9,
                color=TEXT, clip_on=False)

    ax.set_xlim(0, T + 2.4)     # room for the target strip
    ax.set_ylim(3 * K, 0)
    every = 1 if T <= 16 else 2 if T <= 34 else 4
    ax.set_xticks(np.arange(0, T, every) + 0.5)
    ax.set_xticklabels([str(t) for t in range(0, T, every)], fontsize=8, color=MUTED)
    ax.set_yticks(np.arange(3 * K) + 0.5)
    ax.set_yticklabels(names, fontsize=8 if K <= 6 else 7, color=TEXT)
    ax.set_xlabel("time step", fontsize=9, color=MUTED)
    ax.xaxis.set_label_coords(0.5 * T / (T + 2.4), -0.075)
    ax.set_ylabel("input channel", fontsize=9, color=MUTED)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot(trial):
    K, T = trial["K"], trial["T"]
    width = max(11.5, 2 * (1.7 + 0.27 * (T + 4)))
    table_height, raster_height = 0.62 * (K + 1.3), 0.24 * 3 * K + 0.6
    height = table_height + raster_height + 2.6
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[table_height, raster_height],
                            left=0.065, right=0.985, top=1 - 1.5 / height, bottom=1.0 / height,
                            hspace=0.6 / (0.5 * (table_height + raster_height)), wspace=0.16)
    a, b, c = trial["a"], trial["b"], trial["c"]
    headers = (f"One-hop query: B{b}  →  C{c}   (one lookup)",
               f"Two-hop query: A{a}  →  B{b}  →  C{c}   (two lookups)")
    for column, two_hop in enumerate((False, True)):
        tables = fig.add_subplot(grid[0, column], facecolor=SURFACE)
        draw_tables(tables, trial, two_hop)
        tables.set_title(headers[column], fontsize=12, color=TEXT, fontweight="bold", pad=8)
        draw_inputs(fig.add_subplot(grid[1, column], facecolor=SURFACE), trial, two_hop)

    fig.suptitle("Two-hop associative recall: one stored history, asked two ways",
                 x=0.065, y=1 - 0.2 / height, ha="left", va="top", fontsize=15, color=TEXT,
                 fontweight="bold")
    fig.text(0.065, 1 - 0.58 / height,
             "Both trials store the same two random tables and have the same answer. They "
             "differ only in the final\nquery step, so any gap between them is the cost of "
             "the second lookup.", ha="left", va="top", fontsize=10.5, color=MUTED,
             linespacing=1.3)
    handles = [Patch(facecolor=SET_COLORS[s], edgecolor="none", label=f"set {s}") for s in "ABC"]
    handles.append(Patch(facecolor=SURFACE, edgecolor=TEXT, linewidth=1.8,
                         label="outlined: the stored pairs this query needs"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.065, 0.1 / height), ncol=4,
               frameon=False, fontsize=9.5, labelcolor=TEXT, handlelength=1.3,
               columnspacing=2.0)
    return fig


def default_output(n_symbols, delay):
    suffix = "" if (n_symbols, delay) == (4, 4) else f"_K{n_symbols}_d{delay}"
    return HERE / f"twohop_task{suffix}.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw one two-hop recall history asked "
                                                 "as a one-hop and as a two-hop query.")
    parser.add_argument("--symbols", type=int, default=4, help="symbols per set, K (default 4)")
    parser.add_argument("--delay", type=int, default=4, help="blank steps before the query (default 4)")
    parser.add_argument("--seed", type=int, default=8, help="seed of the example history (default 8)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output image (default: twohop_task[_K<K>_d<delay>].png next to this script)")
    args = parser.parse_args(argv)
    output = args.output or default_output(args.symbols, args.delay)
    fig = plot(example_trial(args.symbols, args.delay, args.seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    main()
