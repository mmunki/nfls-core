"""Command line interface for the NFLS pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .code_gen import CodeGenerationError, CodeGenerator
from .formal_verifier import (
    CounterExample,
    FormalVerifier,
    VerificationError,
    VerificationResult,
    VerificationStatus,
)
from .intent_parser import parse_intent
from .schema import Action, IntentAST, Variable


class CLIError(RuntimeError):
    pass


def _read_input(args: argparse.Namespace) -> IntentAST:
    if args.intent is not None:
        return parse_intent(args.intent)
    path = args.schema
    if path == "-":
        text = sys.stdin.read()
    else:
        candidate = Path(path)
        if not candidate.is_file():
            raise CLIError(f"schema file not found: {candidate}")
        text = candidate.read_text(encoding="utf-8")
    try:
        return IntentAST.from_json(text)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CLIError(f"cannot load the intent schema: {exc}") from exc


def _describe_ast(intent: IntentAST) -> str:
    lines = [
        f"Intent: {intent.name}",
        f"Source: {intent.source or '(loaded from schema)'}",
        "",
        "State variables:",
    ]
    for variable in intent.state_variables:
        bounds = []
        if variable.lower is not None:
            bounds.append(f"lower={variable.lower}")
        if variable.upper is not None:
            bounds.append(f"upper={variable.upper}")
        suffix = f" ({', '.join(bounds)})" if bounds else ""
        lines.append(
            f"  - {variable.name}: {variable.sort.value} "
            f"initial={variable.initial if variable.initial is not None else '0'}{suffix}"
        )
    lines.append("")
    lines.append("Invariants:")
    lines.extend(f"  - {expression}" for expression in intent.invariants) if intent.invariants else lines.append("  (none)")
    if intent.preconditions:
        lines.append("Preconditions:")
        lines.extend(f"  - {expression}" for expression in intent.preconditions)
    if intent.postconditions:
        lines.append("Postconditions:")
        lines.extend(f"  - {expression}" for expression in intent.postconditions)
    lines.append("")
    lines.append("Actions:")
    if not intent.actions:
        lines.append("  (none)")
    for action in intent.actions:
        params = ", ".join(f"{p.name}: {p.sort.value}" for p in action.params)
        updates = ", ".join(f"{k} := {v}" for k, v in action.updates.items()) or "(no change)"
        lines.append(
            f"  - {action.name}({params}) guard={action.guard or 'true'} updates={updates}"
        )
    return "\n".join(lines)


def _describe_counterexample(counterexample: CounterExample) -> str:
    lines = [counterexample.explanation]
    if counterexample.state:
        state = ", ".join(f"{key}={value}" for key, value in counterexample.state.items())
        lines.append(f"  counterexample state: {state}")
    if counterexample.params:
        params = ", ".join(f"{key}={value}" for key, value in counterexample.params.items())
        lines.append(f"  parameters: {params}")
    if counterexample.next_state:
        following = ", ".join(
            f"{key}={value}" for key, value in counterexample.next_state.items()
        )
        lines.append(f"  next state: {following}")
    return "\n".join(lines)


def _describe_verification(result: VerificationResult) -> str:
    lines = [
        f"Verification: {result.status.value}",
        result.message,
        "",
        "Proof obligations:",
    ]
    for check in result.checks:
        marker = {
            "PASSED": "OK",
            "FAILED": "FAIL",
            "UNKNOWN": "?",
            "WARNING": "WARN",
        }[check.status.value]
        lines.append(f"  [{marker}] {check.name}: {check.detail}")
    if result.counterexample is not None:
        lines.extend(["", "Counterexample:", _describe_counterexample(result.counterexample)])
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfls",
        description=(
            "Neural-Formal Logic Synthesis: parse an intent, prove it with Z3, "
            "then emit target code."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_inputs(subparser: argparse.ArgumentParser) -> None:
        source = subparser.add_mutually_exclusive_group(required=True)
        source.add_argument("--intent", help="natural-language intent to parse")
        source.add_argument(
            "--schema",
            help="path to an IntentAST JSON file ('-' reads standard input)",
        )
        subparser.add_argument(
            "--timeout-ms",
            type=int,
            default=10_000,
            help="Z3 timeout per proof obligation in milliseconds (default: 10000)",
        )
        subparser.add_argument(
            "--json",
            action="store_true",
            help="emit a machine-readable JSON result instead of human text",
        )

    parse_command = subparsers.add_parser("parse", help="parse an intent into the AST")
    add_inputs(parse_command)

    verify_command = subparsers.add_parser(
        "verify", help="run every Z3 proof obligation for an intent"
    )
    add_inputs(verify_command)

    compile_command = subparsers.add_parser(
        "compile", help="verify an intent and emit executable target code"
    )
    add_inputs(compile_command)
    compile_command.add_argument(
        "--target",
        choices=CodeGenerator.TARGETS,
        default="python",
        help="output language (default: python)",
    )
    compile_command.add_argument(
        "--out",
        help="write the generated code to this file instead of standard output",
    )
    return parser


def _cmd_parse(args: argparse.Namespace) -> int:
    intent = _read_input(args)
    if args.json:
        print(intent.to_json())
    else:
        print(_describe_ast(intent))
    return 0


def _verify(intent: IntentAST, timeout_ms: int) -> VerificationResult:
    return FormalVerifier(timeout_ms=timeout_ms).verify(intent)


def _cmd_verify(args: argparse.Namespace) -> int:
    intent = _read_input(args)
    result = _verify(intent, args.timeout_ms)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(_describe_ast(intent))
        print()
        print(_describe_verification(result))
    return 0 if result.status is VerificationStatus.PROVED_SAT else 2


def _cmd_compile(args: argparse.Namespace) -> int:
    intent = _read_input(args)
    result = _verify(intent, args.timeout_ms)
    if result.status is not VerificationStatus.PROVED_SAT:
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(_describe_verification(result), file=sys.stderr)
        return 2

    code = CodeGenerator(target=args.target).generate(intent, result)
    if args.out:
        Path(args.out).write_text(code, encoding="utf-8")
        if args.json:
            print(
                json.dumps(
                    {
                        "status": result.status.value,
                        "target": args.target,
                        "output": str(args.out),
                    },
                    indent=2,
                )
            )
        else:
            print(f"wrote {args.target} code to {args.out}", file=sys.stderr)
    else:
        print(code, end="" if code.endswith("\n") else "\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "parse":
            return _cmd_parse(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "compile":
            return _cmd_compile(args)
        parser.error(f"unknown command {args.command!r}")
    except (CLIError, VerificationError, CodeGenerationError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
