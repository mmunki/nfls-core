"""Emit executable target code from a *verified* intent.

The generator refuses to run unless :class:`nfls.formal_verifier.VerificationResult`
is ``PROVED_SAT``.  Generated code keeps the same guards and invariants as the
SMT model, so the runtime checks are the executable shadow of the proof.
"""

from __future__ import annotations

import ast
from typing import Iterable, Mapping

from .schema import Action, IntentAST, Sort, Variable
from .formal_verifier import VerificationResult, VerificationStatus, FormalVerifier


class CodeGenerationError(RuntimeError):
    """Raised when code is requested for an unverified or contradictory model."""


class _PythonSelfifier(ast.NodeTransformer):
    def __init__(self, state: set[str], local: set[str]):
        self.state = state
        self.local = local

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.state and node.id not in self.local:
            return ast.copy_location(
                ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=node.id,
                    ctx=node.ctx,
                ),
                node,
            )
        return node


def _python_expression(text: str, state: set[str], local: Iterable[str] = ()) -> str:
    tree = ast.parse(str(text).strip(), mode="eval")
    tree = _PythonSelfifier(state, set(local)).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _c_expression(node: ast.AST, local: set[str] | None = None) -> str:
    local = local or set()
    if isinstance(node, ast.Expression):
        return _c_expression(node.body, local)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return repr(node.value)
    if isinstance(node, ast.Name):
        if node.id == "true":
            return "true"
        if node.id == "false":
            return "false"
        if node.id in local:
            return node.id
        return f"s->{node.id}"
    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",
            ast.Mod: "%",
        }
        operator = operators.get(type(node.op))
        if operator is None:
            raise CodeGenerationError(f"unsupported C operator: {type(node.op).__name__}")
        return f"({_c_expression(node.left, local)} {operator} {_c_expression(node.right, local)})"
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return f"(-{_c_expression(node.operand, local)})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{_c_expression(node.operand, local)})"
        if isinstance(node.op, ast.Not):
            return f"(!{_c_expression(node.operand, local)})"
        raise CodeGenerationError(f"unsupported C unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BoolOp):
        operator = " && " if isinstance(node.op, ast.And) else " || "
        return "(" + operator.join(_c_expression(value, local) for value in node.values) + ")"
    if isinstance(node, ast.Compare):
        operators = {
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        parts = []
        left = node.left
        for operator, comparator in zip(node.ops, node.comparators):
            symbol = operators.get(type(operator))
            if symbol is None:
                raise CodeGenerationError(
                    f"unsupported C comparison: {type(operator).__name__}"
                )
            parts.append(
                f"({_c_expression(left, local)} {symbol} {_c_expression(comparator, local)})"
            )
            left = comparator
        return "(" + " && ".join(parts) + ")"
    if isinstance(node, ast.IfExp):
        return (
            f"({_c_expression(node.test, local)} ? {_c_expression(node.body, local)} : "
            f"{_c_expression(node.orelse, local)})"
        )
    raise CodeGenerationError(f"unsupported C expression node: {type(node).__name__}")


def _c_expr(text: str, local: Iterable[str] = ()) -> str:
    tree = ast.parse(str(text).strip(), mode="eval")
    ast.fix_missing_locations(tree)
    return _c_expression(tree, set(local))


def _effective_invariants(intent: IntentAST) -> list[str]:
    expressions: list[str] = []
    for expression in intent.invariants:
        if expression not in expressions:
            expressions.append(expression)
    for variable in intent.state_variables:
        if variable.lower is not None:
            expression = f"{variable.name} >= {variable.lower}"
            if expression not in expressions:
                expressions.append(expression)
        if variable.upper is not None:
            expression = f"{variable.name} <= {variable.upper}"
            if expression not in expressions:
                expressions.append(expression)
    return expressions


def _class_name(name: str) -> str:
    parts = [part for part in _split_words(name) if part]
    candidate = "".join(part.capitalize() for part in parts) or "Model"
    if candidate[0].isdigit():
        candidate = "Model" + candidate
    return candidate


def _module_name(name: str) -> str:
    candidate = "_".join(part.lower() for part in _split_words(name) if part)
    candidate = candidate or "model"
    if candidate[0].isdigit():
        candidate = "model_" + candidate
    return candidate


def _split_words(name: str) -> list[str]:
    words: list[str] = []
    current = ""
    for character in str(name):
        if character.isalnum():
            if character.isupper() and current:
                words.append(current)
                current = character
            else:
                current += character
        else:
            if current:
                words.append(current)
                current = ""
    if current:
        words.append(current)
    return words


def _python_type(variable: Variable) -> str:
    return variable.sort.python_type.__name__


class CodeGenerator:
    """Generate Python or C from a verified intent."""

    TARGETS = ("python", "c")

    def __init__(self, target: str = "python"):
        if target not in self.TARGETS:
            raise CodeGenerationError(
                f"unsupported target {target!r}; choose one of {', '.join(self.TARGETS)}"
            )
        self.target = target

    def generate(self, intent: IntentAST, verification: VerificationResult) -> str:
        if not isinstance(verification, VerificationResult):
            raise CodeGenerationError("a VerificationResult is required")
        if verification.status is not VerificationStatus.PROVED_SAT:
            explanation = verification.message or "the model was not verified"
            raise CodeGenerationError(
                f"refusing to generate code for an unverified model "
                f"({verification.status.value}): {explanation}"
            )
        if self.target == "python":
            return self._python(intent)
        return self._c(intent)

    def generate_verified(self, intent: IntentAST, timeout_ms: int = 10_000) -> str:
        verification = FormalVerifier(timeout_ms=timeout_ms).verify(intent)
        return self.generate(intent, verification)

    # ---------------------------------------------------------------- python
    def _python(self, intent: IntentAST) -> str:
        state = {variable.name for variable in intent.state_variables}
        class_name = _class_name(intent.name)
        invariants = _effective_invariants(intent)
        invariant_expression = " and ".join(
            f"({_python_expression(expression, state)})" for expression in invariants
        ) or "True"
        lines = [
            '"""Generated by nfls-core from a formally verified intent.',
            "",
            f"Intent: {intent.name}",
            f"Invariants: {'; '.join(invariants) or 'none'}",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "",
            f"class {class_name}:",
            f'    """Executable model generated from the intent {intent.name!r}."""',
            "",
            "    def __init__(self) -> None:",
        ]
        for variable in intent.state_variables:
            initial = variable.initial if variable.initial is not None else (
                "false" if variable.sort is Sort.BOOL else "0"
            )
            expression = _python_expression(initial, state)
            lines.append(f"        self.{variable.name}: {_python_type(variable)} = {expression}")
        lines.extend(
            [
                "",
                "    def invariants(self) -> bool:",
                f"        return {invariant_expression}",
                "",
                "    def assert_invariants(self) -> None:",
                "        if not self.invariants():",
                '            raise AssertionError("NFLS invariant violation")',
                "",
                "    def snapshot(self) -> dict[str, object]:",
                "        return {"
                + ", ".join(f'"{v.name}": self.{v.name}' for v in intent.state_variables)
                + "}",
                "",
                "    def __repr__(self) -> str:",
                '        return f"{self.__class__.__name__}({self.snapshot()!r})"',
            ]
        )

        for action in intent.actions:
            parameter_names = {parameter.name for parameter in action.params}
            parameters = ", ".join(
                f"{parameter.name}: {_python_type(parameter)}" for parameter in action.params
            )
            signature = f"    def {action.name}(self{', ' if parameters else ''}{parameters}) -> bool:"
            lines.extend(["", signature])
            if action.guard:
                guard = _python_expression(action.guard, state, parameter_names)
                lines.append(f"        if not ({guard}):")
                lines.append("            return False")
            if action.updates:
                for target, expression in action.updates.items():
                    value = _python_expression(expression, state, parameter_names)
                    lines.append(f"        new_{target} = {value}")
                for target in action.updates:
                    lines.append(f"        self.{target} = new_{target}")
            lines.append("        self.assert_invariants()")
            lines.append("        return True")
        lines.append("")
        return "\n".join(lines)

    # --------------------------------------------------------------------- C
    def _c(self, intent: IntentAST) -> str:
        module = _module_name(intent.name)
        invariants = _effective_invariants(intent)
        invariant_expression = " && ".join(
            f"({_c_expr(expression)})" for expression in invariants
        ) or "true"
        lines = [
            "/* Generated by nfls-core from a formally verified intent. */",
            f"/* Intent: {intent.name} */",
            f"/* Invariants: {'; '.join(invariants) or 'none'} */",
            "#ifndef NFLS_" + module.upper() + "_H",
            "#define NFLS_" + module.upper() + "_H",
            "",
            "#include <stdbool.h>",
            "",
            "typedef struct {",
        ]
        for variable in intent.state_variables:
            lines.append(f"    {variable.sort.c_type} {variable.name};")
        lines.extend(
            [
                f"}} {module}_state;",
                "",
                f"static inline bool {module}_invariants(const {module}_state *s) {{",
                f"    return {invariant_expression};",
                "}",
                "",
                f"static inline void {module}_init({module}_state *s) {{",
            ]
        )
        for variable in intent.state_variables:
            initial = variable.initial if variable.initial is not None else (
                "false" if variable.sort is Sort.BOOL else "0"
            )
            lines.append(f"    s->{variable.name} = {_c_expr(initial)};")
        lines.extend(["}", ""])

        for action in intent.actions:
            params = "".join(
                f", {parameter.sort.c_type} {parameter.name}" for parameter in action.params
            )
            lines.append(
                f"static inline bool {module}_{action.name}"
                f"({module}_state *s{params}) {{"
            )
            if action.guard:
                local = {parameter.name for parameter in action.params}
                lines.append(f"    if (!({_c_expr(action.guard, local)})) {{")
                lines.append("        return false;")
                lines.append("    }")
            for target, expression in action.updates.items():
                target_variable = intent.variable(target)
                local = {parameter.name for parameter in action.params}
                lines.append(
                    f"    {target_variable.sort.c_type} new_{target} = "
                    f"{_c_expr(expression, local)};"
                )
            for target in action.updates:
                lines.append(f"    s->{target} = new_{target};")
            lines.append(f"    return {module}_invariants(s);")
            lines.append("}")
            lines.append("")
        lines.append("#endif")
        lines.append("")
        return "\n".join(lines)


def generate(
    intent: IntentAST, verification: VerificationResult, target: str = "python"
) -> str:
    """Generate ``target`` code from a verified intent."""

    return CodeGenerator(target=target).generate(intent, verification)
