#!/usr/bin/env python3
"""
Simulate a single jar-and-ball experiment session.

Setup
-----
Jar: 20 balls total, 12 red and 8 blue (true hypothesis h=12).
5 rounds, 3 draws per round without replacement from the remaining balls.

Participants
------------
Human  : Bayesian posterior perturbed by Dirichlet noise (κ=20).
Bot    : updates via asymmetric confirmation-bias model with configurable
         lam_confirm / lam_disconfirm.

Three bot configs are printed side by side:
  1. Perfectly Bayesian          (lam_confirm=0.0, lam_disconfirm=0.0)
  2. Moderate confirmation bias  (lam_confirm=0.1, lam_disconfirm=0.6)
  3. Strong confirmation bias    (lam_confirm=0.0, lam_disconfirm=0.9)

Likelihood note
---------------
Because balls are NOT replaced between rounds, the effective jar composition
changes each round.  Under initial-count hypothesis h, after removing
`prev_red_drawn` red balls in earlier rounds, the jar holds
(h − prev_red_drawn) red balls in `n_remaining` total.  We compute
hypergeometric likelihoods from this adjusted state for each round.
"""

from __future__ import annotations

import numpy as np
from belief_engine import hypergeometric_pmf

# ── Experiment constants ───────────────────────────────────────────────────

N_TOTAL           = 20
N_RED_TRUE        = 12
N_DRAWS_PER_ROUND = 3
N_ROUNDS          = 5
TRUE_IDX          = N_RED_TRUE        # index into HYPOTHESES list (0 … 20)

HYPOTHESES: list[int] = list(range(N_TOTAL + 1))
N_H                   = len(HYPOTHESES)
UNIFORM               = np.ones(N_H) / N_H

SEED = 42

# ── Core probability helpers ───────────────────────────────────────────────

def _round_liks(prev_red_drawn: int, n_remaining: int, k_red: int) -> np.ndarray:
    """
    Hypergeometric likelihoods for each hypothesis given this round's draw.

    Under hypothesis h (initial red count), after removing prev_red_drawn
    red balls in prior rounds, the jar holds (h − prev_red_drawn) red balls
    in n_remaining total.  hypergeometric_pmf returns 0 for infeasible states.
    """
    return np.array([
        hypergeometric_pmf(h - prev_red_drawn, n_remaining, N_DRAWS_PER_ROUND, k_red)
        for h in HYPOTHESES
    ])


def _bayes(belief: np.ndarray, liks: np.ndarray) -> np.ndarray:
    unnorm = liks * belief
    total  = unnorm.sum()
    return belief.copy() if total == 0 else unnorm / total


def _asym(
    current: np.ndarray,
    bayes_post: np.ndarray,
    lam_confirm: float,
    lam_disconfirm: float,
) -> np.ndarray:
    """Per-hypothesis asymmetric blend: lam_confirm where Bayes ≥ current."""
    lams   = np.where(bayes_post >= current, lam_confirm, lam_disconfirm)
    biased = (1.0 - lams) * bayes_post + lams * current
    return biased / biased.sum()


def _qs(probs: np.ndarray) -> float:
    """Quadratic (Brier) score against the true hypothesis."""
    p = np.asarray(probs, dtype=float)
    return float(2.0 * p[TRUE_IDX] - np.dot(p, p))


def _mode(probs: np.ndarray) -> int:
    return HYPOTHESES[int(np.argmax(probs))]


def _draw_str(k_red: int) -> str:
    return f"{k_red}R/{N_DRAWS_PER_ROUND - k_red}B"


# ── Draw simulation ────────────────────────────────────────────────────────

def simulate_draws(rng: np.random.Generator) -> list[tuple[int, int]]:
    """
    Sample the true draw sequence from the real jar.

    Returns list of (k_red_observed, n_jar_before_this_draw) per round.
    """
    jar_red   = N_RED_TRUE
    jar_total = N_TOTAL
    draws: list[tuple[int, int]] = []
    for _ in range(N_ROUNDS):
        k = int(rng.hypergeometric(jar_red, jar_total - jar_red, N_DRAWS_PER_ROUND))
        draws.append((k, jar_total))
        jar_red   -= k
        jar_total -= N_DRAWS_PER_ROUND
    return draws


# ── Session runners ────────────────────────────────────────────────────────

def run_bot_session(
    draws: list[tuple[int, int]],
    lam_confirm: float,
    lam_disconfirm: float,
) -> list[dict]:
    """
    Run one session for a bot with the given λ parameters.

    Also maintains a separate pure-Bayesian reference belief so the table
    can show the Bayesian posterior mode alongside the bot's biased mode.
    """
    bayes_belief = UNIFORM.copy()
    bot_belief   = UNIFORM.copy()
    prev_red     = 0
    results: list[dict] = []

    for k_red, n_remaining in draws:
        liks          = _round_liks(prev_red, n_remaining, k_red)
        bayes_belief  = _bayes(bayes_belief, liks)
        bot_bayes_pos = _bayes(bot_belief, liks)          # Bayes from bot's prior
        bot_belief    = _asym(bot_belief, bot_bayes_pos, lam_confirm, lam_disconfirm)

        results.append({
            "k_red":      k_red,
            "bayes_mode": _mode(bayes_belief),
            "bot_mode":   _mode(bot_belief),
            "qs":         _qs(bot_belief),
        })
        prev_red += k_red

    return results


def run_human_session(
    draws: list[tuple[int, int]],
    rng: np.random.Generator,
    noise_kappa: float = 20.0,
) -> list[dict]:
    """
    Simulate a human participant: true Bayesian posterior + Dirichlet noise.

    κ=20 means beliefs stay close to Bayesian but aren't identical.
    """
    belief   = UNIFORM.copy()
    prev_red = 0
    results: list[dict] = []

    for k_red, n_remaining in draws:
        liks    = _round_liks(prev_red, n_remaining, k_red)
        belief  = _bayes(belief, liks)
        # Dirichlet noise centred on Bayesian posterior
        human_b = rng.dirichlet(noise_kappa * belief + 1e-9)
        results.append({
            "k_red":      k_red,
            "bayes_mode": _mode(belief),
            "human_mode": _mode(human_b),
            "qs":         _qs(human_b),
        })
        prev_red += k_red

    return results


# ── Display ────────────────────────────────────────────────────────────────

def _print_human(human: list[dict]) -> None:
    total = sum(r["qs"] for r in human)
    print("Human participant  (noisy Bayesian, κ=20):")
    print(f"  {'Rnd':<4}  {'Draw':<5}  {'BayesMode':>9}  {'HumanMode':>9}  {'QS':>8}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}")
    for i, r in enumerate(human, 1):
        print(
            f"  {i:<4}  {_draw_str(r['k_red']):<5}"
            f"  {r['bayes_mode']:>9}  {r['human_mode']:>9}  {r['qs']:>+8.4f}"
        )
    print(f"  {'Total':>31}  {total:>+8.4f}")


def _print_bots(
    draws: list[tuple[int, int]],
    bot_configs: list[tuple[str, float, float]],
    all_bot: list[list[dict]],
) -> None:
    SEP = "─"
    COL = 30                          # width of each bot config column

    def _config_hdr(label: str, lc: float, ld: float) -> str:
        return f"{label}  λc={lc:.1f} λd={ld:.1f}"

    # Config header row
    prefix = f"  {'Rnd  Draw':<12}"
    print(prefix, end="")
    for label, lc, ld in bot_configs:
        hdr = _config_hdr(label, lc, ld)
        print(f"  │  {hdr:^{COL - 5}}", end="")
    print()

    # Sub-header
    print(f"  {'':12}", end="")
    for _ in bot_configs:
        print(f"  │  {'BayesMode':>9}  {'BotMode':>7}  {'QS':>8}", end="")
    print()

    # Divider
    print(f"  {SEP*12}", end="")
    for _ in bot_configs:
        print(f"  {SEP*COL}", end="")
    print()

    # Data rows
    for i in range(N_ROUNDS):
        k_red = draws[i][0]
        print(f"  {i+1:<4}  {_draw_str(k_red):<5}", end="")
        for bres in all_bot:
            r = bres[i]
            print(
                f"  │  {r['bayes_mode']:>9}  {r['bot_mode']:>7}  {r['qs']:>+8.4f}",
                end="",
            )
        print()

    # Total row
    print(f"  {SEP*12}", end="")
    for _ in bot_configs:
        print(f"  {SEP*COL}", end="")
    print()

    print(f"  {'Total':>12}", end="")
    for bres in all_bot:
        total = sum(r["qs"] for r in bres)
        print(f"  │  {'':>9}  {'':>7}  {total:>+8.4f}", end="")
    print()


def _print_interpretation(
    bot_configs: list[tuple[str, float, float]],
    all_bot: list[list[dict]],
) -> None:
    print("\nInterpretation:")
    for (label, lc, ld), bres in zip(bot_configs, all_bot):
        total       = sum(r["qs"] for r in bres)
        mode_drifts = sum(1 for r in bres if r["bot_mode"] != r["bayes_mode"])
        print(
            f"  {label:<15} λc={lc:.1f} λd={ld:.1f}"
            f"   total QS = {total:>+7.4f}"
            f"   mode ≠ Bayes in {mode_drifts}/{N_ROUNDS} rounds"
        )


def print_report(
    draws: list[tuple[int, int]],
    human: list[dict],
    bot_configs: list[tuple[str, float, float]],
) -> None:
    all_bot = [run_bot_session(draws, lc, ld) for _, lc, ld in bot_configs]

    bar = "=" * 78
    print(bar)
    print(
        f"  Jar: {N_TOTAL} balls  (TRUE: {N_RED_TRUE} red, {N_TOTAL - N_RED_TRUE} blue)"
        f"  │  {N_DRAWS_PER_ROUND} draws/round, {N_ROUNDS} rounds  │  seed={SEED}"
    )
    print(
        f"  Scoring: quadratic (Brier) rule  │  "
        f"true hypothesis h = {N_RED_TRUE}  │  range (−1, 1]"
    )
    print(bar)
    print()

    _print_human(human)
    print()
    _print_bots(draws, bot_configs, all_bot)
    _print_interpretation(bot_configs, all_bot)
    print()


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng   = np.random.default_rng(SEED)
    draws = simulate_draws(rng)

    human = run_human_session(draws, rng)

    bot_configs: list[tuple[str, float, float]] = [
        ("Bayesian",    0.0, 0.0),
        ("Moderate CB", 0.1, 0.6),
        ("Strong CB",   0.0, 0.9),
    ]

    print_report(draws, human, bot_configs)
