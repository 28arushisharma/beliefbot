"""
BeliefBot API
=============
Two endpoints manage the lifecycle of a single experiment session:

  POST /session/create       — initialise a session, return a session ID
  POST /session/{id}/update  — feed one draw signal, return updated beliefs

Session state is held in an in-process dictionary (SESSIONS).  Each value is
a SessionState dataclass that owns a BeliefEngine instance plus bookkeeping
fields needed for multi-round without-replacement scoring.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Path as FPath
from pydantic import BaseModel, Field, model_validator

# ── Path bootstrap ─────────────────────────────────────────────────────────
# Allow the API to import belief_engine whether run from api/ or from the
# project root (e.g. `uvicorn api.main:app`).
_BELIEFBOT_DIR = Path(__file__).resolve().parent.parent
if str(_BELIEFBOT_DIR) not in sys.path:
    sys.path.insert(0, str(_BELIEFBOT_DIR))

from belief_engine import BeliefEngine, hypergeometric_pmf  # noqa: E402

# ── In-memory store ────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """All mutable state for one experiment session."""

    # Jar / experiment config
    n_balls:          int
    n_red_true:       int
    n_rounds:         int
    draws_per_round:  int
    true_hyp_idx:     int       # index in engine.hypotheses for the true count

    # Bias parameters
    lam_confirm:     float
    lam_disconfirm:  float

    # Live state
    engine:          BeliefEngine
    round_number:    int = 0    # increments on each /update call
    prev_red_drawn:  int = 0    # cumulative red balls drawn so far (for WOR likelihoods)

    # History (one entry per completed round)
    history: list[dict] = field(default_factory=list)


# session_id (str) → SessionState
SESSIONS: dict[str, SessionState] = {}

# ── Pydantic models ────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    n_balls:         int   = Field(20,  ge=2,   description="Total balls in the jar")
    n_red_true:      int   = Field(12,  ge=0,   description="True number of red balls (used for scoring)")
    n_rounds:        int   = Field(5,   ge=1,   description="Number of draw rounds in the session")
    draws_per_round: int   = Field(3,   ge=1,   description="Balls drawn each round (without replacement)")
    lam_confirm:     float = Field(0.0, ge=0.0, le=1.0, description="Confirmation-bias weight for confirming evidence")
    lam_disconfirm:  float = Field(0.0, ge=0.0, le=1.0, description="Confirmation-bias weight for disconfirming evidence")

    @model_validator(mode="after")
    def _check_jar(self) -> "CreateSessionRequest":
        if self.n_red_true > self.n_balls:
            raise ValueError("n_red_true cannot exceed n_balls")
        if self.draws_per_round * self.n_rounds > self.n_balls:
            raise ValueError(
                f"draws_per_round × n_rounds ({self.draws_per_round * self.n_rounds}) "
                f"exceeds n_balls ({self.n_balls}); jar would run out of balls"
            )
        return self


class CreateSessionResponse(BaseModel):
    session_id:      str
    n_balls:         int
    n_red_true:      int
    n_rounds:        int
    draws_per_round: int
    lam_confirm:     float
    lam_disconfirm:  float
    n_hypotheses:    int
    message:         str


class UpdateRequest(BaseModel):
    n_draws: int = Field(..., ge=1, description="Number of balls drawn this round")
    k_red:   int = Field(..., ge=0, description="Number of red balls observed in this draw")

    @model_validator(mode="after")
    def _k_leq_n(self) -> "UpdateRequest":
        if self.k_red > self.n_draws:
            raise ValueError("k_red cannot exceed n_draws")
        return self


class UpdateResponse(BaseModel):
    session_id:       str
    round_number:     int
    rounds_remaining: int
    belief:           list[float]   = Field(description="Full posterior distribution over hypotheses 0 … n_balls")
    hypotheses:       list[int]     = Field(description="Hypothesis values corresponding to belief entries")
    posterior_mode:   int           = Field(description="Hypothesis with highest posterior probability")
    quadratic_score:  float         = Field(description="Brier score against the true hypothesis; range (−1, 1]")
    session_complete: bool


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BeliefBot API",
    description=(
        "Simulate Bayesian and confirmation-biased belief updating "
        "in a jar-and-ball experiment."
    ),
    version="0.1.0",
)

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/session/create", response_model=CreateSessionResponse, status_code=201)
def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
    """
    Initialise a new experiment session.

    Creates a BeliefEngine with a uniform prior over all possible red-ball
    counts (0 … n_balls) and stores it in the in-memory session store.
    Returns a UUID session_id that must be supplied to every subsequent
    /session/{session_id}/update call.

    The `lam_confirm` and `lam_disconfirm` parameters control asymmetric
    confirmation bias (see BeliefEngine.update_asymmetric).  Both 0.0 → pure
    Bayesian agent; lam_disconfirm close to 1.0 → strong resistance to
    evidence that contradicts the current belief.
    """
    engine = BeliefEngine(n_balls=req.n_balls)

    # Index of the true hypothesis in the engine's hypothesis list
    try:
        true_hyp_idx = engine.hypotheses.index(req.n_red_true)
    except ValueError:
        # Should be unreachable given the validator, but guard defensively
        raise HTTPException(status_code=422, detail="n_red_true not in hypothesis space")

    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = SessionState(
        n_balls          = req.n_balls,
        n_red_true       = req.n_red_true,
        n_rounds         = req.n_rounds,
        draws_per_round  = req.draws_per_round,
        true_hyp_idx     = true_hyp_idx,
        lam_confirm      = req.lam_confirm,
        lam_disconfirm   = req.lam_disconfirm,
        engine           = engine,
    )

    return CreateSessionResponse(
        session_id      = session_id,
        n_balls         = req.n_balls,
        n_red_true      = req.n_red_true,
        n_rounds        = req.n_rounds,
        draws_per_round = req.draws_per_round,
        lam_confirm     = req.lam_confirm,
        lam_disconfirm  = req.lam_disconfirm,
        n_hypotheses    = len(engine.hypotheses),
        message         = (
            f"Session created. Send up to {req.n_rounds} draw signals "
            f"to /session/{session_id}/update."
        ),
    )


@app.post("/session/{session_id}/update", response_model=UpdateResponse)
def update_session(
    session_id: Annotated[str, FPath(description="UUID returned by /session/create")],
    req: UpdateRequest,
) -> UpdateResponse:
    """
    Feed one draw signal to the session and return the updated belief state.

    Applies the asymmetric confirmation-bias update rule
    (BeliefEngine.update_asymmetric) using the session's configured λ
    parameters.  Likelihoods are computed via the hypergeometric distribution
    with the jar size adjusted for balls already removed in prior rounds
    (without-replacement semantics across rounds).

    The quadratic (Brier) score is computed against the true hypothesis
    supplied at session creation.  Calling this endpoint more times than
    n_rounds raises 409 Conflict.
    """
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    if state.round_number >= state.n_rounds:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session already completed {state.n_rounds} rounds. "
                "Create a new session to run another experiment."
            ),
        )

    # Validate draw size against session config and remaining jar
    n_remaining = state.n_balls - state.draws_per_round * state.round_number
    if req.n_draws != state.draws_per_round:
        raise HTTPException(
            status_code=422,
            detail=(
                f"This session expects {state.draws_per_round} draws per round; "
                f"got {req.n_draws}."
            ),
        )
    if req.n_draws > n_remaining:
        raise HTTPException(
            status_code=422,
            detail=f"Only {n_remaining} balls remain in the jar; cannot draw {req.n_draws}.",
        )

    # ── Likelihood computation (without-replacement across rounds) ──────────
    # Under hypothesis h, after prev_red_drawn red balls have been removed,
    # the jar now holds (h − prev_red_drawn) red balls in n_remaining total.
    liks = np.array([
        hypergeometric_pmf(
            h - state.prev_red_drawn,
            n_remaining,
            req.n_draws,
            req.k_red,
        )
        for h in state.engine.hypotheses
    ])

    # Pure-Bayes posterior from the current (possibly biased) belief
    current = state.engine.belief
    unnorm  = liks * current
    total   = unnorm.sum()
    bayes_from_current = current.copy() if total == 0.0 else unnorm / total

    # Asymmetric confirmation-bias blend
    lam_arr = np.where(
        bayes_from_current >= current,
        state.lam_confirm,
        state.lam_disconfirm,
    )
    biased = (1.0 - lam_arr) * bayes_from_current + lam_arr * current
    new_belief = biased / biased.sum()

    # Commit updated belief back to the engine
    state.engine.belief = new_belief

    # ── Score and bookkeeping ───────────────────────────────────────────────
    qs = float(
        state.engine.quadratic_score(new_belief, state.true_hyp_idx)
    )
    mode_val = state.engine.hypotheses[int(np.argmax(new_belief))]

    state.round_number   += 1
    state.prev_red_drawn += req.k_red
    state.history.append({
        "round":          state.round_number,
        "n_draws":        req.n_draws,
        "k_red":          req.k_red,
        "posterior_mode": mode_val,
        "quadratic_score": qs,
    })

    return UpdateResponse(
        session_id       = session_id,
        round_number     = state.round_number,
        rounds_remaining = state.n_rounds - state.round_number,
        belief           = new_belief.tolist(),
        hypotheses       = state.engine.hypotheses,
        posterior_mode   = mode_val,
        quadratic_score  = qs,
        session_complete = state.round_number >= state.n_rounds,
    )
