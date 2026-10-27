"""Translate the restricted intent expressions into Z3 terms.

The expression language is a safe subset of Python.  It is parsed with
:mod:`ast` and every node is checked against a whitelist before it becomes a
Z3 expression; ``eval`` is never used.
"""

from __future__ import annotations

import ast
from typing import Mapping

import z3

from .schema import Action, IntentAST, Sort, Variable


class TranslationError(ValueError):
    """Raised when an intent expression is outside the supported subset."""


_BOOL_OPS = {
    ast.And: z3.And,
    ast.Or: z3.Or,
}

_COMPARE_OPS = {
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


class ExpressionTranslator:
    """Compile one expression string against an environment of Z3 constants."""

    def __init__(self, env: Mapping[str, z3.ExprRef]):
        self.env = dict(env)

    def translate(self, expression: str, boolean: bool = True) -> z3.ExprRef:
        """Translate ``expression``.

        With ``boolean=True`` an arithmetic result is coerced to a boolean
        (``term != 0``), which is what a guard, invariant or precondition
        needs.  ``boolean=False`` keeps arithmetic terms intact for updates,
        initial values and numeric bounds.
        """

        text = str(expression).strip()
        if not text:
            raise TranslationError("empty expression")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise TranslationError(f"cannot parse expression {text!r}: {exc.msg}") from exc
        try:
            result = self._expr(tree.body)
        except TranslationError:
            raise
        except (TypeError, z3.Z3Exception) as exc:
            raise TranslationError(f"cannot translate expression {text!r}: {exc}") from exc
        if boolean and not z3.is_bool(result):
            result = result != 0
        return result

    def _expr(self, node: ast.AST) -> z3.ExprRef:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return z3.BoolVal(node.value)
            if isinstance(node.value, int):
                return z3.IntVal(node.value)
            if isinstance(node.value, float):
                return z3.RealVal(node.value)
            raise TranslationError(f"unsupported literal: {node.value!r}")

        if isinstance(node, ast.Name):
            name = node.id
            if name == "true":
                return z3.BoolVal(True)
            if name == "false":
                return z3.BoolVal(False)
            if name not in self.env:
                raise TranslationError(f"unknown symbol {name!r}")
            return self.env[name]

        if isinstance(node, ast.BinOp):
            left, right = self._expr(node.left), self._expr(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise TranslationError(f"unsupported binary operator: {type(node.op).__name__}")

        if isinstance(node, ast.UnaryOp):
            operand = self._expr(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return z3.Not(operand)
            raise TranslationError(f"unsupported unary operator: {type(node.op).__name__}")

        if isinstance(node, ast.BoolOp):
            function = _BOOL_OPS.get(type(node.op))
            if function is None:
                raise TranslationError(f"unsupported boolean operator: {type(node.op).__name__}")
            values = [self._expr(value) for value in node.values]
            if not values:
                raise TranslationError("empty boolean expression")
            return values[0] if len(values) == 1 else function(*values)

        if isinstance(node, ast.Compare):
            left = self._expr(node.left)
            clauses = []
            for operator, comparator in zip(node.ops, node.comparators):
                function = _COMPARE_OPS.get(type(operator))
                if function is None:
                    raise TranslationError(
                        f"unsupported comparison operator: {type(operator).__name__}"
                    )
                right = self._expr(comparator)
                clauses.append(function(left, right))
                left = right
            return clauses[0] if len(clauses) == 1 else z3.And(*clauses)

        if isinstance(node, ast.IfExp):
            condition = self._expr(node.test)
            body = self._expr(node.body)
            orelse = self._expr(node.orelse)
            return z3.If(condition, body, orelse)

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "If" and len(node.args) == 3:
                return z3.If(*(self._expr(arg) for arg in node.args))
            raise TranslationError("only the three-argument If(cond, then, else) call is supported")

        raise TranslationError(f"unsupported expression node: {type(node).__name__}")


def z3_sort(sort: Sort) -> z3.SortRef:
    if sort is Sort.INT:
        return z3.IntSort()
    if sort is Sort.BOOL:
        return z3.BoolSort()
    if sort is Sort.REAL:
        return z3.RealSort()
    raise TranslationError(f"unsupported sort: {sort!r}")


class SMTTranslator:
    """Build Z3 state, initial, guard and transition constraints for an AST."""

    def __init__(self, intent: IntentAST):
        self.intent = intent

    # ------------------------------------------------------------------ state
    def state(self, prefix: str = "") -> dict[str, z3.ExprRef]:
        return {
            variable.name: z3.Const(prefix + variable.name, z3_sort(variable.sort))
            for variable in self.intent.state_variables
        }

    def params(self, action: Action) -> dict[str, z3.ExprRef]:
        return {
            parameter.name: z3.Const(
                f"{action.name}__{parameter.name}", z3_sort(parameter.sort)
            )
            for parameter in action.params
        }

    # ------------------------------------------------------------- expressions
    def translate(self, expression: str, env: Mapping[str, z3.ExprRef]) -> z3.ExprRef:
        return ExpressionTranslator(env).translate(expression)

    def translate_term(self, expression: str, env: Mapping[str, z3.ExprRef]) -> z3.ExprRef:
        return ExpressionTranslator(env).translate(expression, boolean=False)

    def invariant_constraints(self, env: Mapping[str, z3.ExprRef]) -> list[z3.BoolRef]:
        return [self.translate(expression, env) for expression in self.intent.invariants]

    def precondition_constraints(self, env: Mapping[str, z3.ExprRef]) -> list[z3.BoolRef]:
        return [self.translate(expression, env) for expression in self.intent.preconditions]

    def postcondition_constraints(self, env: Mapping[str, z3.ExprRef]) -> list[z3.BoolRef]:
        return [self.translate(expression, env) for expression in self.intent.postconditions]

    # -------------------------------------------------------------- initial
    def initial_constraints(self, env: Mapping[str, z3.ExprRef]) -> list[z3.BoolRef]:
        """Return ``x == initial`` for every state variable.

        A missing initial value defaults to ``0`` (``false`` for booleans).
        """

        constraints: list[z3.BoolRef] = []
        for variable in self.intent.state_variables:
            if variable.initial is not None:
                value = self.translate_term(variable.initial, env)
            elif variable.sort is Sort.BOOL:
                value = z3.BoolVal(False)
            elif variable.sort is Sort.REAL:
                value = z3.RealVal(0)
            else:
                value = z3.IntVal(0)
            constraints.append(env[variable.name] == value)
        return constraints

    def initial_values(self, env: Mapping[str, z3.ExprRef]) -> list[z3.BoolRef]:
        return self.initial_constraints(env)

    # -------------------------------------------------------------- actions
    def param_constraints(
        self, action: Action, params: Mapping[str, z3.ExprRef]
    ) -> list[z3.BoolRef]:
        constraints: list[z3.BoolRef] = []
        for parameter in action.params:
            value = params[parameter.name]
            if parameter.lower is not None:
                constraints.append(value >= self.translate_term(parameter.lower, params))
            if parameter.upper is not None:
                constraints.append(value <= self.translate_term(parameter.upper, params))
        return constraints

    def guard_constraints(
        self, action: Action, env: Mapping[str, z3.ExprRef]
    ) -> list[z3.BoolRef]:
        if not action.guard:
            return []
        return [self.translate(action.guard, env)]

    def transition_constraints(
        self,
        action: Action,
        current: Mapping[str, z3.ExprRef],
        next_state: Mapping[str, z3.ExprRef],
        params: Mapping[str, z3.ExprRef],
    ) -> list[z3.BoolRef]:
        env = dict(current)
        env.update(params)
        constraints: list[z3.BoolRef] = []
        for variable in self.intent.state_variables:
            name = variable.name
            expression = action.updates.get(name)
            next_value = (
                self.translate_term(expression, env)
                if expression is not None
                else current[name]
            )
            constraints.append(next_state[name] == next_value)
        return constraints

    def bound_constraints(
        self, variable: Variable, env: Mapping[str, z3.ExprRef]
    ) -> list[z3.BoolRef]:
        """Declared lower/upper bounds of one state variable in ``env``."""

        constraints: list[z3.BoolRef] = []
        if variable.lower is not None:
            constraints.append(env[variable.name] >= self.translate_term(variable.lower, env))
        if variable.upper is not None:
            constraints.append(env[variable.name] <= self.translate_term(variable.upper, env))
        return constraints
