import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTER = (
    "Create a non-negative counter that increments by 1 and cannot exceed "
    "max capacity 100"
)
WALLET = (
    "Create a wallet system with balance B >= 0. "
    "Transfer(amount) requires B >= amount."
)


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "nfls", *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )


class CLITests(unittest.TestCase):
    def test_parse_emits_json(self):
        result = run_cli("parse", "--intent", WALLET, "--json")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["name"], "wallet")
        self.assertEqual(data["actions"][0]["guard"], "B >= amount")

    def test_verify_reports_proof_obligations(self):
        result = run_cli("verify", "--intent", COUNTER)
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Verification: PROVED_SAT", result.stdout)
        self.assertIn("[OK] increment:invariant-preservation", result.stdout)

    def test_verify_rejects_overflow(self):
        result = run_cli(
            "verify",
            "--schema",
            str(ROOT / "examples" / "overflowing-counter.invalid.json"),
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Verification: UNSAT", result.stdout)
        self.assertIn("Counterexample:", result.stdout)
        self.assertIn("counter=101", result.stdout)

    def test_compile_emits_python_to_stdout(self):
        result = run_cli("compile", "--intent", COUNTER, "--target", "python")
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("class Counter:", result.stdout)
        self.assertIn("def increment(self)", result.stdout)
        self.assertIn("assert_invariants", result.stdout)

    def test_compile_writes_file_and_json_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wallet.c"
            result = run_cli(
                "compile",
                "--intent",
                WALLET,
                "--target",
                "c",
                "--out",
                str(output),
                "--json",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("wallet_transfer", output.read_text(encoding="utf-8"))
            data = json.loads(result.stdout)
            self.assertEqual(data["status"], "PROVED_SAT")
            self.assertEqual(data["target"], "c")

    def test_schema_input(self):
        result = run_cli(
            "verify", "--schema", str(ROOT / "examples" / "wallet.json"), "--json"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["status"], "PROVED_SAT")
        self.assertTrue(data["ok"])

    def test_rejected_compile_does_not_emit_code(self):
        result = run_cli(
            "compile",
            "--schema",
            str(ROOT / "examples" / "overflowing-counter.invalid.json"),
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("counter <= 100", result.stderr)

    def test_missing_schema_is_a_clean_error(self):
        result = run_cli("verify", "--schema", "/definitely/not/here.json")
        self.assertEqual(result.returncode, 1)
        self.assertIn("schema file not found", result.stderr)


if __name__ == "__main__":
    unittest.main()
