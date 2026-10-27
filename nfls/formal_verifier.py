"""Z3-backed proof engine for the NFLS intent AST.

A proof obligation is checked by asking Z3 for a *violating* execution:

* initial state violates an invariant,
* an action starts in a legal state and leaves the invariant set,
* an action drives a variable outside its declared bounds,
* a postcondition does not hold after an action.

If that query is ``unsat`` the obligation is proved.  If it is ``sat`` the
model is a concrete counterexample and the intent is rejected.  This is the
standard way to prove inductive invariants and it is the part the original
task description got backwards: a raw ``unsat`` answer is normally the proof,
not the failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import z3

from .schema import Action, IntentAST, Sort, Variable, combine
from .smt_translator import SMTTranslator, TranslationError


class VerificationStatus(str, Enum):
    """Top-level verdict returned to the caller."""

    PROVED_SAT = "PROVED_SAT"
    UNSAT = "UNSAT"
    UNKNOWN = "UNKNOWN"


class CheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    WARNING = "WARNING"


class VerificationError(RuntimeError):
    """Raised for an AST that cannot be translated or verified at all."""


@dataclass
class CounterExample:
    """A concrete state (and optional transition) that breaks the intent."""

    property: str
    explanation: str
    action: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    next_state: dict[str, Any] = field(default_factory=dict)
    invariant: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "property": self.property,
            "explanation": self.explanation,
            "action": self.action,
            "state": self.state,
            "params": self.params,
            "next_state": self.next_state,
            "invariant": self.invariant,
        }


@dataclass
class CheckResult:
    name: str
    kind: str
    status: CheckStatus
    solver_result: str
    detail: str = ""
    action: str | None = None
    invariant: str | None = None
    counterexample: CounterExample | None = None

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status.value,
            "solver_result": self.solver_result,
            "detail": self.detail,
            "action": self.action,
            "invariant": self.invariant,
        }
        if self.counterexample is not None:
            data["counterexample"] = self.counterexample.to_dict()
        return data


@dataclass
class ActionNode:
    name: str
    params: list[dict[str, str]]
    guard: str | None
    updates: dict[str, str]
    reachable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "guard": self.guard,
            "updates": dict(self.updates),
            "reachable": self.reachable,
        }


@dataclass
class ExecutionGraph:
    """Symbolic graph of the verified transitions."""

    variables: dict[str, str]
    initial_state: dict[str, str]
    invariants: list[str]
    actions: list[ActionNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": dict(self.variables),
            "initial_state": dict(self.initial_state),
            "invariants": list(self.invariants),
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass
class VerificationResult:
    status: VerificationStatus
    checks: list[CheckResult]
    execution_graph: ExecutionGraph
    counterexample: CounterExample | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.PROVED_SAT

    @property
    def proved(self) -> bool:
        return self.ok

    def check(self, name: str) -> CheckResult | None:
        for result in self.checks:
            if result.name == name:
                return result
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "ok": self.ok,
            "message": self.message,
            "checks": [check.to_dict() for check in self.checks],
            "execution_graph": self.execution_graph.to_dict(),
            "counterexample": (
                self.counterexample.to_dict() if self.counterexample is not None else None
            ),
        }


def _conjunction(expressions: list[z3.BoolRef]) -> z3.BoolRef:
    if not expressions:
        return z3.BoolVal(True)
    if len(expressions) == 1:
        return expressions[0]
    return z3.And(*expressions)


def _disjunction(expressions: list[z3.BoolRef]) -> z3.BoolRef:
    if not expressions:
        return z3.BoolVal(False)
    if len(expressions) == 1:
        return expressions[0]
    return z3.Or(*expressions)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_mapping(values: Mapping[str, Any]) -> str:
    if not values:
        return "{}"
    return "{" + ", ".join(f"{key}={_format_value(value)}" for key, value in values.items()) + "}"


class FormalVerifier:
    """Execute every proof obligation of an :class:`IntentAST`."""

    def __init__(self, timeout_ms: int = 10_000):
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------ solver
    def _solve(self, assumptions: list[z3.BoolRef]) -> tuple[z3.CheckSatResult, z3.ModelRef | None, str]:
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)
        solver.add(*assumptions)
        result = solver.check()
        if result == z3.sat:
            return result, solver.model(), ""
        if result == z3.unknown:
            return result, None, solver.reason_unknown()
        return result, None, ""

    # ------------------------------------------------------------------ values
    def _value(self, model: z3.ModelRef, expression: z3.ExprRef) -> Any:
        value = model.eval(expression, model_completion=True)
        if z3.is_true(value):
            return True
        if z3.is_false(value):
            return False
        if z3.is_int_value(value):
            return value.as_long()
        if z3.is_rational_value(value):
            numerator = value.numerator_as_long()
            denominator = value.denominator_as_long()
            if denominator == 1:
                return numerator
            return numerator / denominator
        return str(value)

    def _state(self, model: z3.ModelRef, env: Mapping[str, z3.ExprRef]) -> dict[str, Any]:
        return {name: self._value(model, expression) for name, expression in env.items()}

    def _params(self, model: z3.ModelRef, env: Mapping[str, z3.ExprRef]) -> dict[str, Any]:
        return {name: self._value(model, expression) for name, expression in env.items()}

    def _holds(
        self,
        model: z3.ModelRef,
        expression: str,
        env: Mapping[str, z3.ExprRef],
        translator: SMTTranslator,
    ) -> bool:
        value = model.eval(translator.translate(expression, env), model_completion=True)
        return z3.is_true(value)

    def _violated_invariant(
        self,
        intent: IntentAST,
        translator: SMTTranslator,
        model: z3.ModelRef,
        env: Mapping[str, z3.ExprRef],
    ) -> str:
        for expression in intent.invariants:
            if not self._holds(model, expression, env, translator):
                return expression
        for variable in intent.state_variables:
            if variable.lower is not None:
                expression = f"{variable.name} >= {variable.lower}"
                if not self._holds(model, expression, env, translator):
                    return expression
            if variable.upper is not None:
                expression = f"{variable.name} <= {variable.upper}"
                if not self._holds(model, expression, env, translator):
                    return expression
        return combine(intent.invariants)

    # --------------------------------------------------------------- invariants
    def _invariant_constraints(
        self, intent: IntentAST, translator: SMTTranslator, env: Mapping[str, z3.ExprRef]
    ) -> list[z3.BoolRef]:
        constraints = translator.invariant_constraints(env)
        for variable in intent.state_variables:
            constraints.extend(translator.bound_constraints(variable, env))
        return constraints

    # ------------------------------------------------------------------- verify
    def verify(self, intent: IntentAST) -> VerificationResult:
        try:
            return self._verify(intent)
        except TranslationError as exc:
            raise VerificationError(str(exc)) from exc

    def _verify(self, intent: IntentAST) -> VerificationResult:
        translator = SMTTranslator(intent)
        current = translator.state("state__")
        checks: list[CheckResult] = []

        invariants = self._invariant_constraints(intent, translator, current)
        invariant_expr = combine(intent.invariants)

        # 1. The invariant set itself must have at least one model.  An unsat
        #    invariant set means the intent describes no legal state at all.
        result, model, reason = self._solve(invariants)
        if result == z3.sat:
            checks.append(
                CheckResult(
                    name="invariant-satisfiability",
                    kind="consistency",
                    status=CheckStatus.PASSED,
                    solver_result="sat",
                    detail="the invariant set admits at least one state",
                )
            )
        elif result == z3.unsat:
            explanation = (
                "Z3 proved that no state satisfies all invariants and declared bounds; "
                "the intent is self-contradictory."
            )
            counterexample = CounterExample(
                property="invariant-satisfiability",
                explanation=explanation,
                invariant=invariant_expr,
            )
            checks.append(
                CheckResult(
                    name="invariant-satisfiability",
                    kind="consistency",
                    status=CheckStatus.FAILED,
                    solver_result="unsat",
                    detail=explanation,
                    invariant=invariant_expr,
                    counterexample=counterexample,
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="invariant-satisfiability",
                    kind="consistency",
                    status=CheckStatus.UNKNOWN,
                    solver_result="unknown",
                    detail=f"Z3 returned unknown: {reason}",
                )
            )

        # 2. The initial state must establish the invariants and preconditions.
        initial = translator.initial_constraints(current)
        initial_state = self._initial_state(intent)

        result, model, reason = self._solve(initial + [z3.Not(_conjunction(invariants))])
        if result == z3.sat and model is not None:
            state = self._state(model, current)
            violated = self._violated_invariant(intent, translator, model, current)
            explanation = (
                f"The initial state {_format_mapping(state)} violates the invariant "
                f"'{violated}'; execution is halted before it starts."
            )
            counterexample = CounterExample(
                property="initial-invariant",
                explanation=explanation,
                state=state,
                invariant=violated,
            )
            checks.append(
                CheckResult(
                    name="initial-invariant",
                    kind="initialisation",
                    status=CheckStatus.FAILED,
                    solver_result="sat",
                    detail=explanation,
                    invariant=violated,
                    counterexample=counterexample,
                )
            )
        elif result == z3.unsat:
            checks.append(
                CheckResult(
                    name="initial-invariant",
                    kind="initialisation",
                    status=CheckStatus.PASSED,
                    solver_result="unsat",
                    detail="the initial state establishes every invariant",
                )
            )
        else:
            checks.append(
                CheckResult(
                    name="initial-invariant",
                    kind="initialisation",
                    status=CheckStatus.UNKNOWN,
                    solver_result="unknown" if result == z3.unknown else str(result),
                    detail=f"Z3 returned unknown: {reason}",
                )
            )

        if intent.preconditions:
            preconditions = translator.precondition_constraints(current)
            result, model, reason = self._solve(initial + [z3.Not(_conjunction(preconditions))])
            if result == z3.sat and model is not None:
                state = self._state(model, current)
                violated = next(
                    (
                        expression
                        for expression in intent.preconditions
                        if not self._holds(model, expression, current, translator)
                    ),
                    combine(intent.preconditions),
                )
                explanation = (
                    f"The initial state {_format_mapping(state)} violates the precondition "
                    f"'{violated}'."
                )
                counterexample = CounterExample(
                    property="initial-precondition",
                    explanation=explanation,
                    state=state,
                    invariant=violated,
                )
                checks.append(
                    CheckResult(
                        name="initial-precondition",
                        kind="initialisation",
                        status=CheckStatus.FAILED,
                        solver_result="sat",
                        detail=explanation,
                        invariant=violated,
                        counterexample=counterexample,
                    )
                )
            elif result == z3.unsat:
                checks.append(
                    CheckResult(
                        name="initial-precondition",
                        kind="initialisation",
                        status=CheckStatus.PASSED,
                        solver_result="unsat",
                        detail="the initial state satisfies every precondition",
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name="initial-precondition",
                        kind="initialisation",
                        status=CheckStatus.UNKNOWN,
                        solver_result="unknown" if result == z3.unknown else str(result),
                        detail=f"Z3 returned unknown: {reason}",
                    )
                )

        # 3. Every action is checked against every proof obligation.
        nodes: list[ActionNode] = []
        for action in intent.actions:
            node, action_checks = self._verify_action(
                intent, translator, action, current, invariants
            )
            nodes.append(node)
            checks.extend(action_checks)

        graph = ExecutionGraph(
            variables={variable.name: variable.sort.value for variable in intent.state_variables},
            initial_state=initial_state,
            invariants=list(intent.invariants),
            actions=nodes,
        )

        failures = [check for check in checks if check.status is CheckStatus.FAILED]
        unknowns = [check for check in checks if check.status is CheckStatus.UNKNOWN]
        if failures:
            counterexample = next(
                (check.counterexample for check in failures if check.counterexample is not None),
                None,
            )
            status = VerificationStatus.UNSAT
            message = failures[0].detail
        elif unknowns:
            counterexample = None
            status = VerificationStatus.UNKNOWN
            message = unknowns[0].detail
        else:
            counterexample = None
            status = VerificationStatus.PROVED_SAT
            message = (
                f"All {len(checks)} proof obligations were discharged; "
                f"{len(nodes)} action(s) are formally safe."
            )

        return VerificationResult(
            status=status,
            checks=checks,
            execution_graph=graph,
            counterexample=counterexample,
            message=message,
        )

    def _verify_action(
        self,
        intent: IntentAST,
        translator: SMTTranslator,
        action: Action,
        current: Mapping[str, z3.ExprRef],
        invariants: list[z3.BoolRef],
    ) -> tuple[ActionNode, list[CheckResult]]:
        checks: list[CheckResult] = []
        next_state = translator.state(f"next__{action.name}__")
        params = translator.params(action)
        param_constraints = translator.param_constraints(action, params)
        guard_constraints = translator.guard_constraints(action, dict(current) | dict(params))
        base = list(invariants) + param_constraints + guard_constraints
        transitions = translator.transition_constraints(action, current, next_state, params)

        node = ActionNode(
            name=action.name,
            params=[
                {"name": parameter.name, "sort": parameter.sort.value}
                for parameter in action.params
            ],
            guard=action.guard,
            updates=dict(action.updates),
        )

        # Reachability: an action whose guard can never be true under the
        # invariant is a warning (dead transition), not a proof failure.
        result, model, reason = self._solve(base)
        if result == z3.sat:
            node.reachable = True
            checks.append(
                CheckResult(
                    name=f"{action.name}:reachability",
                    kind="reachability",
                    status=CheckStatus.PASSED,
                    solver_result="sat",
                    detail="the action is reachable from a legal state",
                    action=action.name,
                )
            )
        elif result == z3.unsat:
            node.reachable = False
            checks.append(
                CheckResult(
                    name=f"{action.name}:reachability",
                    kind="reachability",
                    status=CheckStatus.WARNING,
                    solver_result="unsat",
                    detail=(
                        "the action is unreachable: no legal state satisfies its guard; "
                        "the transition is dead code"
                    ),
                    action=action.name,
                )
            )
        else:
            node.reachable = None
            checks.append(
                CheckResult(
                    name=f"{action.name}:reachability",
                    kind="reachability",
                    status=CheckStatus.UNKNOWN,
                    solver_result="unknown",
                    detail=f"Z3 returned unknown: {reason}",
                    action=action.name,
                )
            )

        # Inductive invariant: I(s) and guard(s) and T(s, s') imply I(s').
        next_invariants = self._invariant_constraints(intent, translator, next_state)
        result, model, reason = self._solve(
            base + transitions + [z3.Not(_conjunction(next_invariants))]
        )
        if result == z3.sat and model is not None:
            state = self._state(model, current)
            parameter_values = self._params(model, params)
            following = self._state(model, next_state)
            violated = self._violated_invariant(intent, translator, model, next_state)
            explanation = (
                f"Action '{action.name}' can start in {_format_mapping(state)}"
                + (f" with {_format_mapping(parameter_values)}" if parameter_values else "")
                + f" and produce {_format_mapping(following)}, violating the invariant "
                f"'{violated}'."
            )
            counterexample = CounterExample(
                property="invariant-preservation",
                explanation=explanation,
                action=action.name,
                state=state,
                params=parameter_values,
                next_state=following,
                invariant=violated,
            )
            checks.append(
                CheckResult(
                    name=f"{action.name}:invariant-preservation",
                    kind="invariant",
                    status=CheckStatus.FAILED,
                    solver_result="sat",
                    detail=explanation,
                    action=action.name,
                    invariant=violated,
                    counterexample=counterexample,
                )
            )
        elif result == z3.unsat:
            checks.append(
                CheckResult(
                    name=f"{action.name}:invariant-preservation",
                    kind="invariant",
                    status=CheckStatus.PASSED,
                    solver_result="unsat",
                    detail="every enabled transition preserves all invariants",
                    action=action.name,
                )
            )
        else:
            checks.append(
                CheckResult(
                    name=f"{action.name}:invariant-preservation",
                    kind="invariant",
                    status=CheckStatus.UNKNOWN,
                    solver_result="unknown",
                    detail=f"Z3 returned unknown: {reason}",
                    action=action.name,
                )
            )

        # Explicit overflow / underflow obligations for declared bounds.
        for variable in intent.state_variables:
            violations: list[z3.BoolRef] = []
            bounds: list[str] = []
            if variable.lower is not None:
                lower = translator.translate_term(variable.lower, current)
                violations.append(next_state[variable.name] < lower)
                bounds.append(f"{variable.name} >= {variable.lower}")
            if variable.upper is not None:
                upper = translator.translate_term(variable.upper, current)
                violations.append(next_state[variable.name] > upper)
                bounds.append(f"{variable.name} <= {variable.upper}")
            if not violations:
                continue

            check_name = f"{action.name}:{variable.name}-bounds"
            result, model, reason = self._solve(base + transitions + [_disjunction(violations)])
            if result == z3.sat and model is not None:
                state = self._state(model, current)
                parameter_values = self._params(model, params)
                following = self._state(model, next_state)
                value = following.get(variable.name)
                side = "bound"
                if variable.lower is not None:
                    lower_value = self._value(
                        model, translator.translate_term(variable.lower, current)
                    )
                    if isinstance(value, (int, float)) and isinstance(
                        lower_value, (int, float)
                    ):
                        if value < lower_value:
                            side = "underflow"
                if variable.upper is not None and side == "bound":
                    upper_value = self._value(
                        model, translator.translate_term(variable.upper, current)
                    )
                    if isinstance(value, (int, float)) and isinstance(
                        upper_value, (int, float)
                    ):
                        if value > upper_value:
                            side = "overflow"
                explanation = (
                    f"Action '{action.name}' can drive {variable.name} to {_format_value(value)}"
                    + (
                        f" from {_format_mapping(state)}"
                        + (f" with {_format_mapping(parameter_values)}" if parameter_values else "")
                    )
                    + f", outside the declared bound {' and '.join(bounds)} ({side})."
                )
                counterexample = CounterExample(
                    property=f"{variable.name}-bounds",
                    explanation=explanation,
                    action=action.name,
                    state=state,
                    params=parameter_values,
                    next_state=following,
                    invariant=" and ".join(bounds),
                )
                checks.append(
                    CheckResult(
                        name=check_name,
                        kind="bounds",
                        status=CheckStatus.FAILED,
                        solver_result="sat",
                        detail=explanation,
                        action=action.name,
                        invariant=" and ".join(bounds),
                        counterexample=counterexample,
                    )
                )
            elif result == z3.unsat:
                checks.append(
                    CheckResult(
                        name=check_name,
                        kind="bounds",
                        status=CheckStatus.PASSED,
                        solver_result="unsat",
                        detail=f"the action cannot violate {' and '.join(bounds)}",
                        action=action.name,
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name=check_name,
                        kind="bounds",
                        status=CheckStatus.UNKNOWN,
                        solver_result="unknown",
                        detail=f"Z3 returned unknown: {reason}",
                        action=action.name,
                    )
                )

        # Postconditions are evaluated in the post-state with the parameters.
        if intent.postconditions:
            post_env = dict(next_state)
            post_env.update(params)
            postconditions = [translator.translate(e, post_env) for e in intent.postconditions]
            result, model, reason = self._solve(
                base + transitions + [z3.Not(_conjunction(postconditions))]
            )
            if result == z3.sat and model is not None:
                state = self._state(model, current)
                parameter_values = self._params(model, params)
                following = self._state(model, next_state)
                violated = next(
                    (
                        expression
                        for expression in intent.postconditions
                        if not self._holds(model, expression, post_env, translator)
                    ),
                    combine(intent.postconditions),
                )
                explanation = (
                    f"Action '{action.name}' can start in {_format_mapping(state)}"
                    + (f" with {_format_mapping(parameter_values)}" if parameter_values else "")
                    + f" and produce {_format_mapping(following)}, violating the postcondition "
                    f"'{violated}'."
                )
                counterexample = CounterExample(
                    property="postcondition",
                    explanation=explanation,
                    action=action.name,
                    state=state,
                    params=parameter_values,
                    next_state=following,
                    invariant=violated,
                )
                checks.append(
                    CheckResult(
                        name=f"{action.name}:postcondition",
                        kind="postcondition",
                        status=CheckStatus.FAILED,
                        solver_result="sat",
                        detail=explanation,
                        action=action.name,
                        invariant=violated,
                        counterexample=counterexample,
                    )
                )
            elif result == z3.unsat:
                checks.append(
                    CheckResult(
                        name=f"{action.name}:postcondition",
                        kind="postcondition",
                        status=CheckStatus.PASSED,
                        solver_result="unsat",
                        detail="the action establishes every postcondition",
                        action=action.name,
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        name=f"{action.name}:postcondition",
                        kind="postcondition",
                        status=CheckStatus.UNKNOWN,
                        solver_result="unknown",
                        detail=f"Z3 returned unknown: {reason}",
                        action=action.name,
                    )
                )

        return node, checks

    def _initial_state(self, intent: IntentAST) -> dict[str, str]:
        state: dict[str, str] = {}
        for variable in intent.state_variables:
            if variable.initial is not None:
                state[variable.name] = variable.initial
            elif variable.sort is Sort.BOOL:
                state[variable.name] = "false"
            else:
                state[variable.name] = "0"
        return state


def verify(intent: IntentAST, timeout_ms: int = 10_000) -> VerificationResult:
    """Convenience function around :class:`FormalVerifier`."""

    return FormalVerifier(timeout_ms=timeout_ms).verify(intent)
