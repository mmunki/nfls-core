"""Turn a natural-language intent into the structured NFLS AST.

This is a deterministic, dependency-free front end.  It recognises the small
grammar that covers the examples in the task (counters, wallets, guarded
actions, explicit comparisons and common bound phrases) and leaves an
inspectable AST.  For production use the AST can also be authored directly as
JSON and loaded with :meth:`nfls.schema.IntentAST.load`.
"""

from __future__ import annotations

import re

from .schema import Action, IntentAST, Sort, Variable

_COMPARISON = re.compile(
    r"\b([A-Za-z_]\w*)\b\s*(<=|>=|==|!=|<|>)\s*(-?\d+|[A-Za-z_]\w*)\b"
)
_ACTION = re.compile(
    r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)\s*"
    r"(?:requires|requiring|when|if|precondition(?:s)?\s*:?)\s+([^.;]+)",
    re.IGNORECASE,
)
_INITIAL = re.compile(
    r"(?:starts?|begins?|initial(?:ly| value)?|init(?:ial)?\s+value)"
    r"\s*(?:at|with|to|=|:)?\s*(-?\d+)",
    re.IGNORECASE,
)
_CANNOT_EXCEED = re.compile(
    r"(?:cannot|can\s+not|may\s+not|must\s+not|never|should\s+not)\s+exceed\s+"
    r"(?:a\s+)?(?:max(?:imum)?\s+)?(?:capacity\s+|value\s+|limit\s+)?"
    r"(?:of\s+)?(-?\d+)",
    re.IGNORECASE,
)
_MAXIMUM = re.compile(
    r"(?:max(?:imum)?\s+(?:capacity\s+|value\s+|limit\s+)?(?:of\s+)?|at\s+most|"
    r"no\s+more\s+than|up\s+to)\s*(-?\d+)",
    re.IGNORECASE,
)
_MINIMUM = re.compile(
    r"(?:min(?:imum)?\s+(?:capacity\s+|value\s+|limit\s+)?(?:of\s+)?|at\s+least|"
    r"no\s+less\s+than)\s*(-?\d+)",
    re.IGNORECASE,
)
_INCREMENT = re.compile(
    r"\b(?:increments?|increases?|grows?)\s+(?:([A-Za-z_]\w*)\s+)?by\s+(\d+)",
    re.IGNORECASE,
)
_DECREMENT = re.compile(
    r"\b(?:decrements?|decreases?|shrinks?)\s+(?:([A-Za-z_]\w*)\s+)?by\s+(\d+)",
    re.IGNORECASE,
)
_ADDS = re.compile(
    r"\badds?\s+(\d+)\s+to\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)
_SUBTRACTS = re.compile(
    r"\bsubtracts?\s+(\d+)\s+from\s+([A-Za-z_]\w*)",
    re.IGNORECASE,
)
_NOUNS = (
    "counter",
    "wallet",
    "ledger",
    "balance",
    "account",
    "queue",
    "stack",
    "timer",
    "sensor",
    "buffer",
)
_DEBIT = {
    "transfer",
    "withdraw",
    "withdraws",
    "spend",
    "spends",
    "pay",
    "pays",
    "remove",
    "removes",
    "debit",
    "debits",
    "send",
    "sends",
    "take",
    "takes",
    "consume",
    "consumes",
}
_CREDIT = {
    "deposit",
    "deposits",
    "credit",
    "credits",
    "receive",
    "receives",
    "add",
    "adds",
    "topup",
    "top_up",
    "refund",
    "refunds",
    "mint",
    "mints",
}
_RESERVED = {
    "a",
    "an",
    "and",
    "at",
    "be",
    "by",
    "can",
    "cannot",
    "capacity",
    "create",
    "exceed",
    "for",
    "from",
    "if",
    "in",
    "initial",
    "initially",
    "is",
    "it",
    "least",
    "limit",
    "max",
    "maximum",
    "min",
    "minimum",
    "more",
    "most",
    "must",
    "never",
    "no",
    "non",
    "not",
    "of",
    "on",
    "or",
    "positive",
    "requires",
    "system",
    "than",
    "that",
    "the",
    "then",
    "to",
    "up",
    "value",
    "when",
    "with",
    "zero",
    "true",
    "false",
}


def _snake(name: str) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    return name.strip("_") or "action"


def _sort_from_hint(hint: str | None) -> Sort:
    if hint is None:
        return Sort.INT
    try:
        return Sort.parse(hint)
    except ValueError:
        return Sort.INT


def _parse_params(signature: str) -> list[Variable]:
    parameters: list[Variable] = []
    for chunk in signature.split(","):
        text = chunk.strip()
        if not text:
            continue
        hint: str | None = None
        if ":" in text:
            left, right = text.split(":", 1)
            text = left.strip()
            hint = right.strip().split()[0] if right.strip() else None
        tokens = re.findall(r"[A-Za-z_]\w*", text)
        if not tokens:
            continue
        if len(tokens) >= 2 and tokens[0].lower() in {
            "int",
            "integer",
            "long",
            "bool",
            "boolean",
            "real",
            "float",
            "double",
        }:
            hint = tokens[0]
            name = tokens[1]
        else:
            name = tokens[-1]
        parameters.append(Variable(name=name, sort=_sort_from_hint(hint), role="param"))
    return parameters


def _comparisons(text: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3)) for m in _COMPARISON.finditer(text)]


def _is_number(token: str) -> bool:
    return re.fullmatch(r"-?\d+", token) is not None


def _is_identifier(token: str) -> bool:
    return re.fullmatch(r"[A-Za-z_]\w*", token) is not None


def _candidate(token: str) -> bool:
    return (
        _is_identifier(token)
        and token.lower() not in _RESERVED
        and token not in {"true", "false"}
    )


class IntentParser:
    """Deterministic natural-language to :class:`IntentAST` parser."""

    def parse(self, text: str) -> IntentAST:
        source = " ".join(str(text).strip().split())
        if not source:
            raise ValueError("intent text is empty")

        action_matches = list(_ACTION.finditer(source))
        action_spans = [(match.start(), match.end()) for match in action_matches]

        explicit_actions: list[Action] = []
        param_names: set[str] = set()
        action_guard_tokens: list[str] = []
        for match in action_matches:
            name = _snake(match.group(1))
            parameters = _parse_params(match.group(2))
            param_names.update(parameter.name for parameter in parameters)
            guard_comparisons = _comparisons(match.group(3))
            guard = " and ".join(
                f"{left} {operator} {right}" for left, operator, right in guard_comparisons
            ) or None
            for left, _operator, right in guard_comparisons:
                action_guard_tokens.extend([left, right])
            explicit_actions.append(
                Action(
                    name=name,
                    params=parameters,
                    guard=guard,
                    description=match.group(0).strip(),
                )
            )

        masked = source
        for start, end in sorted(action_spans, reverse=True):
            masked = masked[:start] + " " * (end - start) + masked[end:]

        # ---------------------------------------------------------- variables
        state_names: list[str] = []
        explicit_invariants: list[str] = []
        for left, operator, right in _comparisons(masked):
            for token in (left, right):
                if _candidate(token) and token not in param_names and token not in state_names:
                    state_names.append(token)
            explicit_invariants.append(f"{left} {operator} {right}")

        for token in action_guard_tokens:
            if _candidate(token) and token not in param_names and token not in state_names:
                state_names.append(token)

        lowered = source.lower()
        noun = next((candidate for candidate in _NOUNS if candidate in lowered), None)
        if not state_names:
            state_names.append(noun or "state")

        primary = state_names[0]
        variables: dict[str, Variable] = {
            name: Variable(name=name, sort=Sort.INT) for name in state_names
        }

        # --------------------------------------------------------- invariants
        invariants: list[str] = []

        def add_invariant(expression: str) -> None:
            if expression not in invariants:
                invariants.append(expression)

        for expression in explicit_invariants:
            add_invariant(expression)

        lower_match = _MINIMUM.search(source)
        upper_match = _MAXIMUM.search(source) or _CANNOT_EXCEED.search(source)
        non_negative = re.search(r"\bnon[- ]negative\b", source, re.IGNORECASE)
        positive = re.search(r"\bpositive\b", source, re.IGNORECASE)

        lower: str | None = lower_match.group(1) if lower_match else None
        upper: str | None = upper_match.group(1) if upper_match else None
        if non_negative and lower is None:
            lower = "0"
        if positive and lower is None and not non_negative:
            lower = "1"

        if lower is not None:
            variables[primary].lower = lower
            add_invariant(f"{primary} >= {lower}")
        if upper is not None:
            variables[primary].upper = upper
            add_invariant(f"{primary} <= {upper}")

        # A simple comparison can also declare a bound, which makes the
        # explicit underflow/overflow proof use the same numbers.
        for left, operator, right in _comparisons(masked):
            target: str | None = None
            constant: str | None = None
            effective = operator
            if left in variables and _is_number(right):
                target, constant = left, right
            elif right in variables and _is_number(left):
                target, constant = right, left
                effective = {">": "<", ">=": "<=", "<": ">", "<=": ">="}.get(
                    operator, operator
                )
            if target is None or constant is None:
                continue
            if effective in {">", ">="}:
                variables[target].lower = constant
            elif effective in {"<", "<="}:
                variables[target].upper = constant

        initial_match = _INITIAL.search(source)
        for name, variable in variables.items():
            if name == primary:
                if initial_match:
                    variable.initial = initial_match.group(1)
                elif lower is not None:
                    variable.initial = lower
                elif variable.lower is not None:
                    variable.initial = variable.lower
                else:
                    variable.initial = "0"
            elif variable.lower is not None:
                variable.initial = variable.lower
            elif variable.sort is Sort.BOOL:
                variable.initial = "false"
            else:
                variable.initial = "0"

        # ------------------------------------------------------------ actions
        actions = explicit_actions
        used_names = {action.name for action in actions}

        def unique_action(base: str) -> str:
            name = base
            suffix = 2
            while name in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(name)
            return name

        for match in _ADD_EVENTS(source):
            target = match["target"] if match["target"] in variables else primary
            step = match["step"]
            guard = None
            if variables[target].upper is not None:
                guard = f"{target} + {step} <= {variables[target].upper}"
            actions.append(
                Action(
                    name=unique_action(match["kind"]),
                    guard=guard,
                    updates={target: f"{target} + {step}"},
                    description=match["text"],
                )
            )

        for match in _SUBTRACT_EVENTS(source):
            target = match["target"] if match["target"] in variables else primary
            step = match["step"]
            guard = None
            if variables[target].lower is not None:
                guard = f"{target} - {step} >= {variables[target].lower}"
            actions.append(
                Action(
                    name=unique_action(match["kind"]),
                    guard=guard,
                    updates={target: f"{target} - {step}"},
                    description=match["text"],
                )
            )

        for action in actions:
            if action.updates or not action.params:
                continue
            target = self._action_target(action, variables, primary)
            if target is None:
                continue
            parameter = self._amount_parameter(action)
            lowered_name = action.name.split("_")[0]
            if lowered_name in _DEBIT:
                action.updates[target] = f"{target} - {parameter}"
            elif lowered_name in _CREDIT:
                action.updates[target] = f"{target} + {parameter}"
            elif lowered_name in {"increment", "add"}:
                action.updates[target] = f"{target} + {parameter}"
            elif lowered_name in {"decrement", "subtract"}:
                action.updates[target] = f"{target} - {parameter}"

        name = noun or primary
        return IntentAST(
            name=name,
            variables=list(variables.values()),
            actions=actions,
            invariants=invariants,
            source=source,
            metadata={"parser": "nfls.intent_parser", "version": 1},
        )

    # ------------------------------------------------------------- heuristics
    def _action_target(
        self, action: Action, variables: dict[str, Variable], primary: str
    ) -> str | None:
        for left, _operator, right in _comparisons(action.guard or ""):
            if left in variables:
                return left
            if right in variables:
                return right
        return primary if primary in variables else None

    def _amount_parameter(self, action: Action) -> str:
        for parameter in action.params:
            if "amount" in parameter.name.lower():
                return parameter.name
        if action.params:
            return action.params[0].name
        return "1"


def _ADD_EVENTS(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for match in _INCREMENT.finditer(text):
        target = match.group(1)
        events.append(
            {
                "kind": "increment",
                "target": target or "",
                "step": match.group(2),
                "text": match.group(0),
            }
        )
    for match in _ADDS.finditer(text):
        events.append(
            {
                "kind": "add",
                "target": match.group(2),
                "step": match.group(1),
                "text": match.group(0),
            }
        )
    return events


def _SUBTRACT_EVENTS(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for match in _DECREMENT.finditer(text):
        target = match.group(1)
        events.append(
            {
                "kind": "decrement",
                "target": target or "",
                "step": match.group(2),
                "text": match.group(0),
            }
        )
    for match in _SUBTRACTS.finditer(text):
        events.append(
            {
                "kind": "subtract",
                "target": match.group(2),
                "step": match.group(1),
                "text": match.group(0),
            }
        )
    return events


def parse_intent(text: str) -> IntentAST:
    """Convenience wrapper for :class:`IntentParser`."""

    return IntentParser().parse(text)
