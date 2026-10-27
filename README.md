# nfls-core

**Neural-Formal Logic Synthesis (NFLS)** turns a natural-language intent into a
structured state machine, proves the state machine with the Z3 SMT solver, and
only then emits executable code.

The project is deliberately small and dependency-light: Python 3.10+, one
runtime dependency (`z3-solver`), no LLM API in the loop. Natural-language
parsing is deterministic; the formal proof is the source of truth.

## Pipeline

```
intent text
    |
    v
intent_parser.py   -> IntentAST (variables, initial state, actions, invariants)
    |
    v
schema.py          -> validated dataclasses + JSON round-trip
    |
    v
smt_translator.py  -> Z3 terms, guards, transitions, bounds
    |
    v
formal_verifier.py -> proof obligations + counterexamples
    |                    |
    | PROVED_SAT          | UNSAT
    v                    v
code_gen.py          stop, explain the counterexample
    |
    v
Python / C source with runtime guards and invariant checks
```

| File | Role |
| --- | --- |
| `nfls/schema.py` | Intent AST: `IntentAST`, `Variable`, `Action`, sorts, JSON I/O |
| `nfls/intent_parser.py` | deterministic natural-language to AST front end |
| `nfls/smt_translator.py` | safe expression parser and Z3 environment builder |
| `nfls/formal_verifier.py` | proof obligations, execution graph, counterexamples |
| `nfls/smt_verifier.py` | `(status, ExecutionGraph / CounterExample)` compatibility API |
| `nfls/code_gen.py` | Python and C synthesis, refuses unverified models |
| `nfls/cli.py` | `parse`, `verify`, `compile` commands |

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For a one-off run without installing, put the checkout on `PYTHONPATH` and use
`python -m nfls`.

## Usage

```bash
python -m nfls verify \
  --intent "Create a non-negative counter that increments by 1 and cannot exceed max capacity 100"

python -m nfls compile \
  --intent "Create a wallet system with balance B >= 0. Transfer(amount) requires B >= amount." \
  --target python

python -m nfls compile --schema examples/wallet.json --target c --out wallet.h
```

`compile` writes diagnostics to stderr and code to stdout, so it composes:

```bash
python -m nfls compile --intent "..." > model.py
```

A rejected intent exits with status `2`, prints the failed obligation and the
Z3 counterexample, and emits no code:

```bash
python -m nfls verify --schema examples/overflowing-counter.invalid.json
# Verification: UNSAT
# Counterexample:
# Action 'increment' can start in {counter=100} and produce {counter=101},
# violating the invariant 'counter <= 100'.
```

## What is actually proved

The task description says "`sat` means valid, `unsat` means contradiction".
That is only true if the invariant is asserted directly. An SMT solver does not
prove a universal statement by returning `sat`; it finds one model. NFLS uses
the standard inductive-invariant query and asks Z3 for a **violating**
execution:

| Obligation | Query sent to Z3 | Proof |
| --- | --- | --- |
| Invariants are consistent | `I(s)` | `sat` (otherwise the intent is self-contradictory) |
| Initial state is legal | `init(s) and not I(s)` | `unsat` |
| Action preserves invariants | `I(s) and guard(s,p) and T(s,s',p) and not I(s')` | `unsat` |
| No underflow / overflow | same, plus `x' < lower` or `x' > upper` | `unsat` |
| Action is reachable | `I(s) and guard(s,p)` | `sat` (otherwise a dead-transition warning) |
| Postcondition holds | `I(s) and guard(s,p) and T(s,s',p) and not post(s',p)` | `unsat` |

The public name `PROVED_SAT` is kept for compatibility with the requested API:
it means "the model is satisfiable **and** every universal obligation was
discharged". `UNSAT` means Z3 found a satisfying counterexample, so compilation
stops. `UNKNOWN` is returned when the solver times out.

## Intent AST

The parser fills this structure:

```python
IntentAST(
    name="wallet",
    variables=[Variable("B", sort=Sort.INT, initial="0", lower="0")],
    invariants=["B >= 0"],
    actions=[
        Action(
            "transfer",
            params=[Variable("amount", role="param")],
            guard="B >= amount",
            updates={"B": "B - amount"},
        )
    ],
)
```

It round-trips through JSON. Expressions use a safe subset of Python syntax:
integer/real/boolean literals, state variables, action parameters, `+ - * / %`,
comparisons, `and`, `or`, `not`, parentheses and a three-argument `If(c, a, b)`.
`eval` is never used; `nfls.smt_translator` walks the AST with a whitelist.

For a precise intent, write the AST as JSON (`examples/wallet.json`) and pass it
with `--schema`. The natural-language front end understands the examples in the
task plus common variants:

* `non-negative` / `positive` lower bounds
* `cannot exceed max capacity 100`, `at most`, `up to`, `maximum`
* `at least`, `minimum`
* `starts at 10`
* `increments by 2`, `decrements by 1`, `adds 5 to counter`
* `Action(param) requires <comparison>` (e.g. `Transfer(amount) requires B >= amount`)
* debit verbs (`transfer`, `withdraw`, `spend`, ...) and credit verbs
  (`deposit`, `add`, `receive`, ...)

Anything richer should be expressed as an explicit JSON AST.

## Generated code

Python output is a class with the initial state, one method per action, the
same guards as the proof, and an `assert_invariants()` call after every
transition. C output is a header with a state struct, `*_init`, one
`*_action` function per transition and `*_invariants`.

Both targets are generated **only** from a `PROVED_SAT` result. The runtime
guards are defence in depth, not a replacement for the proof.

## Tests

```bash
python -m unittest discover -s tests -v
```

The suite covers parsing, JSON round-trips, inductive proofs, overflow,
underflow, contradictory invariants, unreachable actions, pre/postconditions,
Python execution semantics, C compilation when `gcc` is available, and the CLI.
CI runs Python 3.10, 3.11, 3.12 and 3.13 on Ubuntu x86_64, then builds the
wheel and sdist.

## Limits

* The NL parser is rule-based, not an LLM. Use the JSON AST for exact semantics.
* Z3 models mathematical integers; machine-word wrap-around of the generated C
  is not modelled. Declare explicit `lower`/`upper` bounds when the runtime type
  matters.
* Action parameters are universally chosen by Z3 unless they have explicit
  bounds. A guard such as `B >= amount` is what makes a transfer safe.
* Targets are Python and C; WebAssembly is not emitted by 0.1.0.
* A reachable action whose guard is never satisfiable is reported as a warning,
  not a hard failure.

## License

MIT.
