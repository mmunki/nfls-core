"""Small compatibility surface for the requested ``smt_verifier`` module.

The actual solver integration lives in :mod:`nfls.formal_verifier`; this module
exposes the exact shape the task asked for::

    status, payload = SMTVerifier().verify_pair(ast)
    # status == PROVED_SAT -> payload is the ExecutionGraph
    # status == UNSAT      -> payload is the CounterExample
    # status == UNKNOWN    -> payload is the full VerificationResult

``verify()`` returns the richer :class:`VerificationResult`, which keeps every
individual proof obligation visible.
"""

from __future__ import annotations

from .formal_verifier import (
    CounterExample,
    ExecutionGraph,
    FormalVerifier,
    VerificationResult,
    VerificationStatus,
)
from .schema import IntentAST


class SMTVerifier:
    """Pure Z3 integration for an :class:`IntentAST`."""

    def __init__(self, timeout_ms: int = 10_000):
        self.timeout_ms = timeout_ms

    def verify(self, intent: IntentAST) -> VerificationResult:
        return FormalVerifier(timeout_ms=self.timeout_ms).verify(intent)

    def verify_pair(
        self, intent: IntentAST
    ) -> tuple[
        VerificationStatus, ExecutionGraph | CounterExample | VerificationResult
    ]:
        result = self.verify(intent)
        if result.status is VerificationStatus.PROVED_SAT:
            return result.status, result.execution_graph
        if result.status is VerificationStatus.UNSAT:
            return result.status, result.counterexample or result
        return result.status, result


def verify(intent: IntentAST, timeout_ms: int = 10_000) -> VerificationResult:
    return SMTVerifier(timeout_ms=timeout_ms).verify(intent)
