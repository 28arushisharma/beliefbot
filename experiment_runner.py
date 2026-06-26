#!/usr/bin/env python3
"""
Run many simulated sessions and aggregate results to CSV.

For each of three bot configurations, 500 independent sessions are run.
Each session uses a fresh random draw sequence (jar: 20 balls, 12 red,
5 rounds of 3 draws each without replacement from the remaining balls).

Per-round data saved to results/simulation_results.csv:
  bot_config, lam_confirm, lam_disconfirm, session_id, round,
  draw_red, draw_blue, true_bayes_mode, bot_mode, modes_diverge,
  quadratic_score

A summary table (mean QS and mode-divergence rate by config) is printed
after the CSV is written.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

import numpy as np

from simulate import (
    N_DRAWS_PER_ROUND,
    N_ROUNDS,
    run_bot_session,
    simulate_draws,
)

# ── Configuration ──────────────────────────────────────────────────────────

BOT_CONFIGS: list[tuple[str, float, float]] = [
    ("bayesian",    0.0, 0.0),
    ("moderate_cb", 0.1, 0.6),
    ("strong_cb",   0.0, 0.9),
]

N_SESSIONS  = 500
SEED        = 0

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
OUTPUT_CSV  = RESULTS_DIR / "simulation_results.csv"

CSV_FIELDS = [
    "bot_config",
    "lam_confirm",
    "lam_disconfirm",
    "session_id",
    "round",
    "draw_red",
    "draw_blue",
    "true_bayes_mode",
    "bot_mode",
    "modes_diverge",
    "quadratic_score",
]

# ── Data collection ────────────────────────────────────────────────────────

def run_all_sessions(rng: np.random.Generator) -> list[dict]:
    """
    Run N_SESSIONS sessions for each bot config and collect per-round rows.

    The rng advances through the full session loop so each session sees a
    distinct draw sequence regardless of bot configuration ordering.
    """
    rows: list[dict] = []

    for config_name, lam_c, lam_d in BOT_CONFIGS:
        # Fresh sub-generator per config so config order doesn't affect
        # the draw sequences seen within each config's sessions.
        config_rng = np.random.default_rng(rng.integers(2**31))

        for session_id in range(N_SESSIONS):
            draws   = simulate_draws(config_rng)
            results = run_bot_session(draws, lam_c, lam_d)

            for rnd_idx, r in enumerate(results):
                rows.append({
                    "bot_config":      config_name,
                    "lam_confirm":     lam_c,
                    "lam_disconfirm":  lam_d,
                    "session_id":      session_id,
                    "round":           rnd_idx + 1,
                    "draw_red":        r["k_red"],
                    "draw_blue":       N_DRAWS_PER_ROUND - r["k_red"],
                    "true_bayes_mode": r["bayes_mode"],
                    "bot_mode":        r["bot_mode"],
                    "modes_diverge":   int(r["bot_mode"] != r["bayes_mode"]),
                    "quadratic_score": r["qs"],
                })

    return rows

# ── Persistence ────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows):,} rows  →  {path}\n")

# ── Summary ────────────────────────────────────────────────────────────────

def _compute_summary(rows: list[dict]) -> dict[str, dict]:
    """Return per-config summary stats."""
    qs_per_round:    dict[str, list[float]] = defaultdict(list)
    diverge_per_rnd: dict[str, list[int]]   = defaultdict(list)
    session_totals:  dict[str, list[float]] = defaultdict(list)

    # Accumulate per-session totals in a temporary map
    sess_accum: dict[tuple[str, int], float] = defaultdict(float)

    for row in rows:
        cfg = row["bot_config"]
        qs_per_round[cfg].append(row["quadratic_score"])
        diverge_per_rnd[cfg].append(row["modes_diverge"])
        sess_accum[(cfg, row["session_id"])] += row["quadratic_score"]

    # Collect session totals per config
    for (cfg, _sid), total in sess_accum.items():
        session_totals[cfg].append(total)

    summary: dict[str, dict] = {}
    for config_name, lam_c, lam_d in BOT_CONFIGS:
        qs_rnd  = np.array(qs_per_round[config_name])
        div_rnd = np.array(diverge_per_rnd[config_name])
        qs_sess = np.array(session_totals[config_name])
        summary[config_name] = {
            "lam_confirm":     lam_c,
            "lam_disconfirm":  lam_d,
            "n_sessions":      len(qs_sess),
            "n_rounds":        len(qs_rnd),
            "mean_qs_round":   float(np.mean(qs_rnd)),
            "std_qs_round":    float(np.std(qs_rnd)),
            "mean_qs_session": float(np.mean(qs_sess)),
            "std_qs_session":  float(np.std(qs_sess)),
            "diverge_rate":    float(np.mean(div_rnd)),   # fraction in [0, 1]
        }
    return summary


def print_summary(rows: list[dict]) -> None:
    summary = _compute_summary(rows)

    hdrs = (
        f"  {'Configuration':<15}  {'λ_c':>4}  {'λ_d':>4}"
        f"  {'Mean QS/round':>13}  {'±SD':>7}"
        f"  {'Mean QS/session':>15}  {'±SD':>7}"
        f"  {'Mode diverge %':>14}"
    )
    div = f"  {'-'*15}  {'-'*4}  {'-'*4}  {'-'*13}  {'-'*7}  {'-'*15}  {'-'*7}  {'-'*14}"

    print("Summary  ─  500 sessions × 5 rounds per configuration\n")
    print(hdrs)
    print(div)

    for config_name, lam_c, lam_d in BOT_CONFIGS:
        s = summary[config_name]
        print(
            f"  {config_name:<15}  {lam_c:>4.1f}  {lam_d:>4.1f}"
            f"  {s['mean_qs_round']:>+13.4f}  {s['std_qs_round']:>7.4f}"
            f"  {s['mean_qs_session']:>+15.4f}  {s['std_qs_session']:>7.4f}"
            f"  {s['diverge_rate']*100:>13.1f}%"
        )

    print()
    print(
        "  Note: quadratic score range is (−1, 1]; higher is better.\n"
        "  A proper scoring rule maximises expected score at true beliefs;\n"
        "  confirmation bias (λ_d > 0) is expected to reduce mean score\n"
        "  and widen the gap between bot mode and Bayesian posterior mode."
    )
    print()

# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng  = np.random.default_rng(SEED)
    rows = run_all_sessions(rng)
    save_csv(rows, OUTPUT_CSV)
    print_summary(rows)
