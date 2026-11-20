#!/usr/bin/env python
# coding: utf-8
"""
Illustration of the nonlinear system identification tasks (train_mpn.py --task
wh_mid / wh_out).

One white-noise input is run through both hidden systems, as the paired tasks do.
The left column is wh_mid (the tanh sits between the two memories), the right
column wh_out (the tanh comes after both). Each column shows, top to bottom:

  the system      its three blocks, in order
  the input       x_t, the only thing the network sees (the same in both columns)
  inside          the signals between the blocks, and where on the tanh they fall
  the target      y_t, with the best causal linear filter's prediction: what that
                  filter misses is the part the reported skill measures

The signals come from the task's own code (nonlinear_system_tasks), so the figure
shows exactly what training asks of the network.

    python scripts/tasks/illustrations/plot_nonlinear_system_task.py
    python scripts/tasks/illustrations/plot_nonlinear_system_task.py --params b95_g2p5

--params takes the fragments of the task name (as in wh_mid_b95_g2p5). Writes
nonlinear_system_task.png next to this file (nonlinear_system_task_b95_g2p5.png for
the second command); --output chooses another path.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # scripts/tasks: the task modules and their _bootstrap
import _bootstrap  # noqa: E402,F401  (puts core/ on sys.path for the task registry)
import tasks  # noqa: E402
from nonlinear_system_tasks import fir_predict  # noqa: E402

# Reference palette of the dataviz skill: categorical slots 1-3 (all-pairs safe), one
# per signal along the system. Text and outlines stay in ink, never in a signal color.
FIRST, SECOND, OUTPUT = "#2a78d6", "#eb6834", "#1baf7a"   # after block 1, block 2, block 3
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d8d7d2"
TITLES = {"mid": "wh_mid: the nonlinearity sits between the two memories",
          "out": "wh_out: the nonlinearity comes after both memories"}


def tint(color, weight):
    """Mix a hex color with the surface: weight 0 is the color, 1 is the surface."""
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple((1.0 - weight) * rgb + weight * np.array(matplotlib.colors.to_rgb(SURFACE)))


def slow_filter(u, beta):
    """The slow memory of the system: r_t = beta * r_{t-1} + (1 - beta) * u_t, from 0."""
    out = np.zeros_like(u, dtype=np.float64)
    state = np.zeros(u.shape[0])
    for t in range(u.shape[1]):
        state = beta * state + (1.0 - beta) * u[:, t]
        out[:, t] = state
    return out


def second_signal(task, s):
    """The signal between the second and the third block: u_t = tanh(gain * s_t)
    for wh_mid, the slow state r_t for wh_out."""
    return np.tanh(task.gain * s) if task.variant == "mid" else slow_filter(s, task.beta)


def example(params="", seed=5, n_pool=512):
    """One input waveform through both systems. Returns {'x': (T,), 'mid': {...},
    'out': {...}}; each variant holds its task, the signals of the example trial
    (s, second, y, fir), and two statistics over a pool of n_pool other trials: the
    central 90 % range of the tanh's input and the linear filter's R^2."""
    suffix = f"_{params}" if params else ""
    made = {v: tasks.make_task(f"wh_{v}{suffix}") for v in ("mid", "out")}
    T = made["mid"].seq_len
    x = np.random.RandomState(seed).standard_normal((1, T))
    pool = np.random.RandomState(seed + 1).standard_normal((n_pool, T))
    result = {"x": x[0], "params": params}
    for variant, task in made.items():
        y, s = task.targets(x)
        y_pool, s_pool = task.targets(pool)
        before_tanh = s_pool if variant == "mid" else slow_filter(s_pool, task.beta)
        result[variant] = {
            "task": task, "s": s[0], "second": second_signal(task, s)[0], "y": y[0],
            "fir": fir_predict(x, *task.fir_coefficients())[0],
            "tanh_range": np.percentile(task.gain * before_tanh, [5.0, 95.0]),
            "r2": 1.0 - task.fir_mse(pool, y_pool) / float(y_pool.var())}
    return result


def number(value):
    return f"{value:g}"


def draw_system(ax, task):
    """The three blocks in order, with the signals between them."""
    variant = task.variant
    fast = ("fast memory", rf"$\alpha$ = {number(task.alpha)}", False)
    slow = ("slow memory", rf"$\beta$ = {number(task.beta)}", False)
    tanh = ("tanh", f"gain = {number(task.gain)}", True)
    blocks = (fast, tanh, slow) if variant == "mid" else (fast, slow, tanh)
    signals = ((r"$x_t$", MUTED), (r"$s_t$", FIRST),
               (r"$u_t$" if variant == "mid" else r"$r_t$", SECOND), (r"$y_t$", OUTPUT))
    lefts, width, y0, height = (0.10, 0.40, 0.70), 0.20, 0.36, 0.42
    for left, (name, detail, moved) in zip(lefts, blocks):
        ax.add_patch(FancyBboxPatch((left, y0), width, height, transform=ax.transAxes,
                                    boxstyle="round,pad=0,rounding_size=0.03", clip_on=False,
                                    facecolor=tint(MUTED, 0.86) if moved else SURFACE,
                                    edgecolor=TEXT if moved else MUTED,
                                    linewidth=2.0 if moved else 1.1))
        ax.text(left + width / 2, y0 + 0.63 * height, name, transform=ax.transAxes, ha="center",
                va="center", fontsize=10.5, color=TEXT, fontweight="bold")
        ax.text(left + width / 2, y0 + 0.27 * height, detail, transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=MUTED)
    edges = [0.0] + [v for left in lefts for v in (left, left + width)] + [1.0]
    for i, (label, color) in enumerate(signals):
        start, stop = edges[2 * i], edges[2 * i + 1]
        ax.annotate("", xy=(stop - 0.006, y0 + height / 2), xytext=(start + 0.006, y0 + height / 2),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=TEXT, mutation_scale=12,
                                    shrinkA=0, shrinkB=0))
        middle = (start + stop) / 2
        ax.text(middle, y0 + height / 2 + 0.2, label, transform=ax.transAxes, ha="center",
                va="center", fontsize=11.5, color=TEXT)
        ax.plot([middle - 0.022, middle + 0.022], [y0 + height / 2 + 0.07] * 2, color=color,
                lw=3.2, solid_capstyle="round", transform=ax.transAxes, clip_on=False)
    if variant == "mid":
        formula = (r"$s_t=\alpha s_{t-1}+(1-\alpha)x_t$      $u_t=\tanh(g\,s_t)$      "
                   r"$y_t=\beta y_{t-1}+(1-\beta)u_t$")
    else:
        formula = (r"$s_t=\alpha s_{t-1}+(1-\alpha)x_t$      $r_t=\beta r_{t-1}+(1-\beta)s_t$      "
                   r"$y_t=\tanh(g\,r_t)$")
    ax.text(0.5, 0.08, formula, transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
            color=MUTED)
    ax.axis("off")


def style_trace_axis(ax, T, limit, ticks, label, last):
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-limit, limit)
    ax.set_yticks(ticks)
    ax.axhline(0.0, color=GRID, lw=0.9, zorder=1)
    ax.set_ylabel(label, fontsize=9, color=MUTED)
    ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    ax.tick_params(axis="x", labelbottom=last)
    if last:
        ax.set_xlabel("time step", fontsize=9, color=MUTED)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(GRID)


def legend(ax, **kwargs):
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, handlelength=1.6, borderaxespad=0.2,
              loc="lower left", bbox_to_anchor=(0.0, 0.98), **kwargs)


def note(ax, text):
    ax.axis("off")
    ax.text(0.02, 0.5, text, transform=ax.transAxes, ha="left", va="center", fontsize=8.8,
            color=MUTED, linespacing=1.45)


def draw_column(fig, cell, data, x):
    task = data["task"]
    variant, T = task.variant, task.seq_len
    steps = np.arange(T)
    grid = cell.subgridspec(4, 2, height_ratios=[1.2, 1.0, 1.75, 1.75], width_ratios=[3.3, 1.2],
                            hspace=0.36, wspace=0.09)
    system = fig.add_subplot(grid[0, :], facecolor=SURFACE)
    draw_system(system, task)
    system.set_title(TITLES[variant], fontsize=12, color=TEXT, fontweight="bold", pad=6)

    # The input: the only thing the network sees.
    ax = fig.add_subplot(grid[1, 0], facecolor=SURFACE)
    ax.vlines(steps, 0.0, x, color=MUTED, lw=1.3, zorder=2)
    ax.plot(steps, x, "o", color=MUTED, ms=2.6, zorder=3)
    style_trace_axis(ax, T, 3.4, [-2, 0, 2], r"input $x_t$", last=False)
    note(fig.add_subplot(grid[1, 1]),
         "White noise, $x_t \\sim N(0, 1)$.\nThe only signal the network\nsees; the same waveform\n"
         "in both tasks.")

    # Inside the system: the two signals between the blocks.
    ax = fig.add_subplot(grid[2, 0], facecolor=SURFACE)
    second_label = (r"$u_t=\tanh(g\,s_t)$, after the tanh" if variant == "mid"
                    else r"$r_t$, slow memory of $s_t$")
    ax.plot(steps, data["s"], color=FIRST, lw=1.9, zorder=3, label=r"$s_t$, fast memory of $x_t$")
    ax.plot(steps, data["second"], color=SECOND, lw=1.9, zorder=3, label=second_label)
    style_trace_axis(ax, T, 1.25, [-1, 0, 1], "inside the system", last=False)
    legend(ax, ncol=2, columnspacing=1.4)

    # Where the tanh's input falls on the tanh.
    low, high = data["tanh_range"]
    entering = FIRST if variant == "mid" else SECOND
    ax = fig.add_subplot(grid[2, 1], facecolor=SURFACE)
    z = np.linspace(-8.0, 8.0, 321)
    ax.axvspan(low, high, color=tint(entering, 0.72), lw=0, zorder=1)
    ax.axhline(0.0, color=GRID, lw=0.9, zorder=2)
    ax.plot(z, np.tanh(z), color=TEXT, lw=1.8, zorder=3)
    ax.set_xlim(-8.0, 8.0)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([-6, 0, 6])
    ax.set_yticks([-1, 0, 1])
    ax.tick_params(length=0, labelsize=8, labelcolor=MUTED)
    ax.set_xlabel(r"input to tanh, $g\,s_t$" if variant == "mid" else r"input to tanh, $g\,r_t$",
                  fontsize=8.5, color=MUTED)
    span = f"{low:+.1f} to {high:+.1f}".replace("-", "−")
    ax.set_title(f"90 % of steps: {span}", fontsize=8.5, color=TEXT, pad=4)
    for name, spine in ax.spines.items():
        spine.set_visible(name == "bottom")
        spine.set_color(GRID)

    # The target, and how far a linear filter gets.
    ax = fig.add_subplot(grid[3, 0], facecolor=SURFACE)
    ax.plot(steps, data["y"], color=OUTPUT, lw=2.1, zorder=4, label=r"target $y_t$")
    ax.plot(steps, data["fir"], color=TEXT, lw=1.3, ls=(0, (4, 2)), zorder=3,
            label="best linear filter")
    ax.fill_between(steps, data["y"], data["fir"], color=tint(MUTED, 0.72), lw=0, zorder=2,
                    label="what the linear filter misses")
    style_trace_axis(ax, T, 1.25, [-1, 0, 1], "output", last=True)
    legend(ax, ncol=3, columnspacing=1.4)
    note(fig.add_subplot(grid[3, 1]),
         f"Causal linear filter,\n{task.fir_order} lags: $R^2$ = {data['r2']:.2f}.\n\n"
         "skill = 1 − MSE / MSE$_{linear}$\n0: this filter,  1: exact")


def plot(data):
    width, height = 14.5, 9.0
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    outer = fig.add_gridspec(1, 2, left=0.05, right=0.99, top=1 - 1.5 / height,
                             bottom=0.55 / height, wspace=0.1)
    for column, variant in enumerate(("mid", "out")):
        draw_column(fig, outer[0, column], data[variant], data["x"])
    fig.suptitle("Nonlinear system identification: one input, the nonlinearity in two places",
                 x=0.05, y=1 - 0.2 / height, ha="left", va="top", fontsize=15, color=TEXT,
                 fontweight="bold")
    fig.text(0.05, 1 - 0.58 / height,
             "The network sees only the input and must output $y_t$ at every step. Both tasks "
             "use the same input, the same two memories and the same tanh;\nonly the position "
             "of the tanh differs. A linear filter already explains most of the target, so the "
             "reported skill counts only what it misses.",
             ha="left", va="top", fontsize=10.5, color=MUTED, linespacing=1.3)
    return fig


def default_output(params=""):
    return HERE / f"nonlinear_system_task{'_' + params if params else ''}.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw one white-noise input through the "
                                                 "wh_mid and the wh_out system.")
    parser.add_argument("--params", default="",
                        help="task-name fragments shared by both tasks, e.g. b95_g2p5_T32 "
                             "(default: the tasks' defaults)")
    parser.add_argument("--seed", type=int, default=5, help="seed of the example input (default 5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="output image (default: nonlinear_system_task[_<params>].png next "
                             "to this script)")
    args = parser.parse_args(argv)
    output = args.output or default_output(args.params)
    fig = plot(example(args.params, args.seed))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    main()
