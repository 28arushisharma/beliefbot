#!/usr/bin/env python3
"""
Produce two publication-quality figures from simulation_results.csv.

Figure 1  (fig1_qs_by_round.png)
    Line chart: mean quadratic score per round, one line per bot config,
    with 95 % bootstrap confidence intervals.

Figure 2  (fig2_mode_divergence.png)
    Bar chart: mode-divergence rate (fraction of rounds where bot mode ≠
    Bayesian posterior mode) by bot configuration.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "simulation_results.csv"
FIG1_PATH   = RESULTS_DIR / "fig1_qs_by_round.png"
FIG2_PATH   = RESULTS_DIR / "fig2_mode_divergence.png"

# ── Display config ─────────────────────────────────────────────────────────

# Canonical order and display names for all three configs
CONFIG_ORDER = ["bayesian", "moderate_cb", "strong_cb"]
CONFIG_LABEL = {
    "bayesian":    "Bayesian  (λ$_c$=0.0, λ$_d$=0.0)",
    "moderate_cb": "Moderate CB  (λ$_c$=0.1, λ$_d$=0.6)",
    "strong_cb":   "Strong CB  (λ$_c$=0.0, λ$_d$=0.9)",
}
CONFIG_COLOR = {
    "bayesian":    "#2166ac",   # blue
    "moderate_cb": "#f4a582",   # peach
    "strong_cb":   "#d6604d",   # red
}
CONFIG_MARKER = {
    "bayesian":    "o",
    "moderate_cb": "s",
    "strong_cb":   "^",
}

N_BOOTSTRAP = 2_000
ALPHA       = 0.05      # for 95 % CI
RNG         = np.random.default_rng(1)

# ── Data loading ───────────────────────────────────────────────────────────

def load_csv(path: pathlib.Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def cast_row(row: dict) -> dict:
    return {
        "bot_config":      row["bot_config"],
        "session_id":      int(row["session_id"]),
        "round":           int(row["round"]),
        "modes_diverge":   int(row["modes_diverge"]),
        "quadratic_score": float(row["quadratic_score"]),
    }

# ── Bootstrap CI ───────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    stat: callable = np.mean,
    n: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: np.random.Generator = RNG,
) -> tuple[float, float, float]:
    """Return (point_estimate, ci_low, ci_high) via percentile bootstrap."""
    point  = float(stat(values))
    boots  = np.array([stat(rng.choice(values, size=len(values), replace=True)) for _ in range(n)])
    ci_low = float(np.percentile(boots, 100 * alpha / 2))
    ci_hi  = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, ci_low, ci_hi

# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(rows: list[dict]) -> dict:
    """
    Returns:
      qs_by_round[config][round]  -> np.ndarray of QS values (one per session)
      diverge_by_config[config]   -> np.ndarray of 0/1 per round across all sessions
    """
    # qs_by_round[config][round] -> list of floats
    qs_by_round: dict[str, dict[int, list[float]]] = {
        c: defaultdict(list) for c in CONFIG_ORDER
    }
    diverge_by_config: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        cfg = row["bot_config"]
        qs_by_round[cfg][row["round"]].append(row["quadratic_score"])
        diverge_by_config[cfg].append(row["modes_diverge"])

    # Convert to numpy
    return {
        "qs_by_round": {
            cfg: {rnd: np.array(vals) for rnd, vals in rd.items()}
            for cfg, rd in qs_by_round.items()
        },
        "diverge_by_config": {
            cfg: np.array(vals) for cfg, vals in diverge_by_config.items()
        },
    }

# ── Shared style ───────────────────────────────────────────────────────────

def _apply_minimal_style(ax: plt.Axes) -> None:
    """Remove chartjunk: no top/right spines, light grid on y only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9, color="#888888")

# ── Figure 1: QS by round ─────────────────────────────────────────────────

def plot_qs_by_round(agg: dict, path: pathlib.Path) -> None:
    qs_by_round = agg["qs_by_round"]
    rounds      = sorted(next(iter(qs_by_round.values())).keys())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _apply_minimal_style(ax)

    for cfg in CONFIG_ORDER:
        pts, los, his = [], [], []
        for rnd in rounds:
            pt, lo, hi = bootstrap_ci(qs_by_round[cfg][rnd])
            pts.append(pt)
            los.append(lo)
            his.append(hi)

        pts = np.array(pts)
        los = np.array(los)
        his = np.array(his)

        ax.plot(
            rounds, pts,
            color=CONFIG_COLOR[cfg],
            marker=CONFIG_MARKER[cfg],
            markersize=5,
            linewidth=1.8,
            label=CONFIG_LABEL[cfg],
            zorder=3,
        )
        ax.fill_between(
            rounds, los, his,
            color=CONFIG_COLOR[cfg],
            alpha=0.15,
            linewidth=0,
            zorder=2,
        )

    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Mean quadratic score", fontsize=10)
    ax.set_title(
        "Mean quadratic score by round\n"
        "(500 sessions per configuration; shaded band = 95 % bootstrap CI)",
        fontsize=10,
        loc="left",
        pad=10,
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(
        frameon=False,
        fontsize=8.5,
        loc="upper left",
        handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  →  {path}")

# ── Figure 2: mode divergence rate ────────────────────────────────────────

def plot_mode_divergence(agg: dict, path: pathlib.Path) -> None:
    diverge = agg["diverge_by_config"]

    rates  = []
    ci_los = []
    ci_his = []
    for cfg in CONFIG_ORDER:
        pt, lo, hi = bootstrap_ci(diverge[cfg])
        rates.append(pt * 100)
        ci_los.append(pt * 100 - lo * 100)
        ci_his.append(hi * 100 - pt * 100)

    labels = [CONFIG_LABEL[cfg] for cfg in CONFIG_ORDER]
    colors = [CONFIG_COLOR[cfg] for cfg in CONFIG_ORDER]
    x      = np.arange(len(CONFIG_ORDER))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _apply_minimal_style(ax)

    bars = ax.bar(
        x,
        rates,
        color=colors,
        width=0.5,
        yerr=[ci_los, ci_his],
        error_kw={
            "ecolor":    "#555555",
            "elinewidth": 1.2,
            "capsize":   4,
            "capthick":  1.2,
        },
        zorder=3,
    )

    # Value labels above each bar
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ci_his) * 0.15 + 0.4,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mode divergence rate (%)", fontsize=10)
    ax.set_title(
        "Mode divergence rate by bot configuration\n"
        "(fraction of rounds where bot mode ≠ Bayesian posterior mode;\n"
        " error bars = 95 % bootstrap CI)",
        fontsize=10,
        loc="left",
        pad=10,
    )
    ax.set_ylim(0, max(rates) * 1.35)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  →  {path}")

# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading  {CSV_PATH}")
    raw  = load_csv(CSV_PATH)
    rows = [cast_row(r) for r in raw]
    agg  = aggregate(rows)

    plot_qs_by_round(agg, FIG1_PATH)
    plot_mode_divergence(agg, FIG2_PATH)

    print("\nDone.")
