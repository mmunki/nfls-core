"""Structured intent AST used by every stage of the NFLS pipeline.

The AST is deliberately small and serialisable.  Expressions are kept as
strings in a restricted Python-expression syntax (integer/boolean literals,
state variables, action parameters, arithmetic, comparisons and ``and`` /
``or`` / ``not``).  ``nfls.smt_translator`` is the only component allowed to
turn those strings into solver terms, which keeps the parser free of any Z3
dependency.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Sort(str, Enum):
    """The sorts an intent variable may have."""

    INT = "Int"
    BOOL = "Bool"
    REAL = "Real"

    @classmethod
    def parse(cls, value: "Sort | str") -> "Sort":
        if isinstance(value, Sort):
            return value
        aliases = {
            "int": cls.INT,
            "integer": cls.INT,
            "long": cls.INT,
            "bool": cls.BOOL,
            "boolean": cls.BOOL,
            "real": cls.REAL,
            "float": cls.REAL,
            "double": cls.REAL,
        }
        try:
            return aliases[str(value).strip().lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported sort: {value!r}") from exc

    @property
    def python_type(self) -> type:
        return {Sort.INT: int, Sort.BOOL: bool, Sort.REAL: float}[self]

    @property
    def c_type(self) -> str:
        return {Sort.INT: "long long", Sort.BOOL: "bool", Sort.REAL: "double"}[self]


_GENERATED_NAMES = {"self", "invariants", "assert_invariants", "snapshot"}


def _check_identifier(value: str, what: str) -> None:
    if not value or not _IDENTIFIER.match(value):
        raise ValueError(f"{what} must be a valid identifier, got {value!r}")
    if keyword.iskeyword(value):
        raise ValueError(f"{what} cannot be a Python keyword: {value!r}")
    if value in _GENERATED_NAMES:
        raise ValueError(f"{what} is reserved by the code generator: {value!r}")


@dataclass
class Variable:
    """A state variable or an action parameter."""

    name: str
    sort: Sort = Sort.INT
    initial: str | None = None
    lower: str | None = None
    upper: str | None = None
    role: str = "state"
    description: str = ""

    def __post_init__(self) -> None:
        self.sort = Sort.parse(self.sort)
        _check_identifier(self.name, "variable name")
        if self.role not in {"state", "param"}:
            raise ValueError(f"role must be 'state' or 'param', got {self.role!r}")
        for label, value in (
            ("initial", self.initial),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if value is not None and not str(value).strip():
                raise ValueError(f"{self.name}.{label} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sort": self.sort.value,
            "initial": self.initial,
            "lower": self.lower,
            "upper": self.upper,
            "role": self.role,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Variable":
        return cls(
            name=data["name"],
            sort=Sort.parse(data.get("sort", "Int")),
            initial=data.get("initial"),
            lower=data.get("lower"),
            upper=data.get("upper"),
            role=data.get("role", "state"),
            description=data.get("description", ""),
        )


@dataclass
class Action:
    """A guarded transition.

    ``updates`` maps a state variable to the expression of its *next* value.
    Variables that are not mentioned are left unchanged (frame condition).
    """

    name: str
    params: list[Variable] = field(default_factory=list)
    guard: str | None = None
    updates: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        _check_identifier(self.name, "action name")
        seen: set[str] = set()
        for param in self.params:
            if param.role != "param":
                param.role = "param"
            if param.name in seen:
                raise ValueError(f"duplicate parameter {param.name!r} in action {self.name!r}")
            seen.add(param.name)
        for target, expression in self.updates.items():
            _check_identifier(target, "update target")
            if not str(expression).strip():
                raise ValueError(f"empty update expression for {target!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": [p.to_dict() for p in self.params],
            "guard": self.guard,
            "updates": dict(self.updates),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(
            name=data["name"],
            params=[Variable.from_dict(p) for p in data.get("params", [])],
            guard=data.get("guard"),
            updates=dict(data.get("updates", {})),
            description=data.get("description", ""),
        )


@dataclass
class IntentAST:
    """The complete formal intent handed to the SMT verifier."""

    name: str
    variables: list[Variable]
    actions: list[Action] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("intent name cannot be empty")
        if not self.variables:
            raise ValueError("an intent needs at least one state variable")
        names: set[str] = set()
        for variable in self.variables:
            if variable.role != "state":
                raise ValueError(f"{variable.name!r} is listed as state but has role {variable.role!r}")
            if variable.name in names:
                raise ValueError(f"duplicate state variable {variable.name!r}")
            names.add(variable.name)
        action_names: set[str] = set()
        for action in self.actions:
            if action.name in action_names:
                raise ValueError(f"duplicate action {action.name!r}")
            action_names.add(action.name)
            params = {p.name for p in action.params}
            overlap = params & names
            if overlap:
                raise ValueError(
                    f"action {action.name!r} parameters collide with state variables: "
                    f"{', '.join(sorted(overlap))}"
                )
            for target in action.updates:
                if target not in names:
                    raise ValueError(
                        f"action {action.name!r} updates unknown state variable {target!r}"
                    )
        for kind, values in (
            ("invariant", self.invariants),
            ("precondition", self.preconditions),
            ("postcondition", self.postconditions),
        ):
            for expression in values:
                if not str(expression).strip():
                    raise ValueError(f"empty {kind} expression")

    @property
    def state_variables(self) -> list[Variable]:
        return [v for v in self.variables if v.role == "state"]

    def variable(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(name)

    def action(self, name: str) -> Action:
        for action in self.actions:
            if action.name == name:
                return action
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variables": [v.to_dict() for v in self.variables],
            "actions": [a.to_dict() for a in self.actions],
            "invariants": list(self.invariants),
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntentAST":
        return cls(
            name=data["name"],
            variables=[Variable.from_dict(v) for v in data.get("variables", [])],
            actions=[Action.from_dict(a) for a in data.get("actions", [])],
            invariants=list(data.get("invariants", [])),
            preconditions=list(data.get("preconditions", [])),
            postconditions=list(data.get("postconditions", [])),
            source=data.get("source", ""),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "IntentAST":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str | Path) -> "IntentAST":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")


def combine(expressions: Iterable[str]) -> str:
    """Join non-empty expressions with ``and`` for compact human output."""

    return " and ".join(f"({e})" for e in expressions if str(e).strip()) or "true"
