# Compatibility shim — source of truth has moved to beliefbot/belief_engine.py.
# Scripts that do `from belief_engine import ...` continue to work unchanged.
from beliefbot.belief_engine import BeliefEngine, hypergeometric_pmf  # noqa: F401

__all__ = ["BeliefEngine", "hypergeometric_pmf"]
