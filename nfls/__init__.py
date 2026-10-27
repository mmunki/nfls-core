"""NFLS -- Neural-Formal Logic Synthesis.

Natural-language intent -> structured AST -> Z3 proofs -> executable code.
"""

from .code_gen import CodeGenerationError, CodeGenerator
from .formal_verifier import (
    CheckResult,
    CheckStatus,
    CounterExample,
    ExecutionGraph,
    FormalVerifier,
    VerificationError,
    VerificationResult,
    VerificationStatus,
    verify,
)
from .intent_parser import IntentParser, parse_intent
from .schema import Action, IntentAST, Sort, Variable
from .smt_verifier import SMTVerifier

__version__ = "0.1.0"

__all__ = [
    "Action",
    "CheckResult",
    "CheckStatus",
    "CodeGenerationError",
    "CodeGenerator",
    "CounterExample",
    "ExecutionGraph",
    "FormalVerifier",
    "IntentAST",
    "IntentParser",
    "SMTVerifier",
    "Sort",
    "Variable",
    "VerificationError",
    "VerificationResult",
    "VerificationStatus",
    "parse_intent",
    "verify",
    "__version__",
]
