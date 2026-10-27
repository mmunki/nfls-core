import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from nfls.code_gen import CodeGenerationError, CodeGenerator
from nfls.formal_verifier import FormalVerifier, VerificationStatus
from nfls.intent_parser import parse_intent
from nfls.schema import Action, IntentAST, Variable


def _load_python(code: str, class_name: str):
    namespace: dict[str, object] = {}
    exec(compile(code, "<generated>", "exec"), namespace)
    return namespace[class_name]


class CodeGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.counter = parse_intent(
            "Create a non-negative counter that increments by 1 and cannot exceed "
            "max capacity 100"
        )
        self.counter_result = FormalVerifier().verify(self.counter)
        self.wallet = parse_intent(
            "Create a wallet system with balance B >= 0. "
            "Transfer(amount) requires B >= amount."
        )
        self.wallet_result = FormalVerifier().verify(self.wallet)

    def test_refuses_unverified_model(self):
        bad = IntentAST(
            name="counter",
            variables=[Variable("counter", initial="0", upper="10")],
            invariants=["counter <= 10"],
            actions=[Action("increment", updates={"counter": "counter + 1"})],
        )
        result = FormalVerifier().verify(bad)
        self.assertEqual(result.status, VerificationStatus.UNSAT)
        with self.assertRaises(CodeGenerationError):
            CodeGenerator().generate(bad, result)

    def test_python_counter_semantics(self):
        code = CodeGenerator("python").generate(self.counter, self.counter_result)
        Counter = _load_python(code, "Counter")
        model = Counter()
        self.assertTrue(model.invariants())
        for _ in range(100):
            self.assertTrue(model.increment())
        self.assertEqual(model.counter, 100)
        self.assertFalse(model.increment())
        self.assertEqual(model.counter, 100)
        self.assertTrue(model.invariants())

    def test_python_wallet_semantics(self):
        code = CodeGenerator("python").generate(self.wallet, self.wallet_result)
        Wallet = _load_python(code, "Wallet")
        model = Wallet()
        self.assertEqual(model.B, 0)
        self.assertFalse(model.transfer(1))
        model.B = 100
        self.assertTrue(model.transfer(40))
        self.assertEqual(model.B, 60)
        self.assertTrue(model.transfer(60))
        self.assertEqual(model.B, 0)
        self.assertFalse(model.transfer(1))
        self.assertTrue(model.invariants())

    def test_c_code_contains_guard_and_bounds(self):
        code = CodeGenerator("c").generate(self.wallet, self.wallet_result)
        self.assertIn("wallet_transfer", code)
        self.assertIn("s->B >= amount", code)
        self.assertIn("wallet_invariants", code)
        self.assertIn("return false;", code)

    @unittest.skipUnless(shutil.which("gcc"), "gcc is not installed")
    def test_c_code_compiles_and_enforces_invariants(self):
        header = CodeGenerator("c").generate(self.wallet, self.wallet_result)
        wrapper = r"""
#include <stdio.h>
#include "wallet.h"

int main(void) {
    wallet_state s;
    wallet_init(&s);
    s.B = 100;
    if (!wallet_transfer(&s, 40)) return 1;
    if (s.B != 60) return 2;
    if (!wallet_transfer(&s, 60)) return 3;
    if (s.B != 0) return 4;
    if (wallet_transfer(&s, 1)) return 5;
    if (!wallet_invariants(&s)) return 6;
    printf("B=%lld\n", s.B);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wallet.h").write_text(header, encoding="utf-8")
            (root / "main.c").write_text(wrapper, encoding="utf-8")
            binary = root / "wallet_test"
            compile_result = subprocess.run(
                ["gcc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-o", binary, "main.c"],
                cwd=root,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                compile_result.returncode,
                0,
                msg=compile_result.stderr,
            )
            run_result = subprocess.run([binary], text=True, capture_output=True)
            self.assertEqual(run_result.returncode, 0, msg=run_result.stderr)
            self.assertEqual(run_result.stdout.strip(), "B=0")

    def test_invalid_target_is_rejected(self):
        with self.assertRaises(CodeGenerationError):
            CodeGenerator("wasm")


if __name__ == "__main__":
    unittest.main()
