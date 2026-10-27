import unittest

from nfls.formal_verifier import CheckStatus, FormalVerifier, VerificationStatus
from nfls.intent_parser import parse_intent
from nfls.schema import Action, IntentAST, Variable
from nfls.smt_verifier import SMTVerifier


class FormalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = FormalVerifier(timeout_ms=10_000)

    def test_counter_is_proved(self):
        intent = parse_intent(
            "Create a non-negative counter that increments by 1 and cannot exceed "
            "max capacity 100"
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.PROVED_SAT)
        self.assertTrue(result.ok)
        self.assertIsNone(result.counterexample)
        self.assertEqual(result.execution_graph.initial_state, {"counter": "0"})
        self.assertTrue(result.execution_graph.actions[0].reachable)
        self.assertEqual(result.check("increment:counter-bounds").status, CheckStatus.PASSED)

    def test_wallet_is_proved(self):
        intent = parse_intent(
            "Create a wallet system with balance B >= 0. "
            "Transfer(amount) requires B >= amount."
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.PROVED_SAT)
        self.assertEqual(result.check("transfer:invariant-preservation").solver_result, "unsat")

    def test_unguarded_increment_is_rejected_with_counterexample(self):
        intent = IntentAST(
            name="counter",
            variables=[Variable("counter", initial="0", lower="0", upper="100")],
            invariants=["counter >= 0", "counter <= 100"],
            actions=[Action("increment", updates={"counter": "counter + 1"})],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.UNSAT)
        self.assertFalse(result.ok)
        counterexample = result.counterexample
        self.assertIsNotNone(counterexample)
        self.assertEqual(counterexample.action, "increment")
        self.assertEqual(counterexample.state, {"counter": 100})
        self.assertEqual(counterexample.next_state, {"counter": 101})
        self.assertIn("counter <= 100", counterexample.explanation)

    def test_unguarded_transfer_is_rejected(self):
        intent = IntentAST(
            name="wallet",
            variables=[Variable("B", initial="0", lower="0")],
            invariants=["B >= 0"],
            actions=[
                Action(
                    "withdraw",
                    params=[Variable("amount", role="param")],
                    guard="B >= 0",
                    updates={"B": "B - amount"},
                )
            ],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.UNSAT)
        self.assertEqual(result.counterexample.property, "invariant-preservation")
        self.assertLess(result.counterexample.next_state["B"], 0)

    def test_initially_violated_invariant_is_rejected(self):
        intent = IntentAST(
            name="counter",
            variables=[Variable("counter", initial="5", lower="0", upper="3")],
            invariants=["counter <= 3"],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.UNSAT)
        self.assertEqual(result.counterexample.property, "initial-invariant")
        self.assertEqual(result.counterexample.state, {"counter": 5})

    def test_contradictory_invariants_are_rejected(self):
        intent = IntentAST(
            name="value",
            variables=[Variable("V", initial="0", lower="10", upper="5")],
            invariants=["V >= 10", "V <= 5"],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.UNSAT)
        self.assertEqual(result.counterexample.property, "invariant-satisfiability")
        self.assertIn("self-contradictory", result.counterexample.explanation)

    def test_unreachable_action_is_a_warning_not_a_failure(self):
        intent = IntentAST(
            name="wallet",
            variables=[Variable("B", initial="0", lower="0")],
            invariants=["B >= 0"],
            actions=[
                Action(
                    "impossible",
                    guard="B < 0",
                    updates={"B": "B"},
                )
            ],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.PROVED_SAT)
        check = result.check("impossible:reachability")
        self.assertEqual(check.status, CheckStatus.WARNING)
        self.assertFalse(result.execution_graph.actions[0].reachable)

    def test_precondition_and_postcondition(self):
        intent = IntentAST(
            name="counter",
            variables=[Variable("counter", initial="0", lower="0")],
            invariants=["counter >= 0"],
            preconditions=["counter == 0"],
            postconditions=["counter >= 0"],
            actions=[Action("increment", updates={"counter": "counter + 1"})],
        )
        result = self.verifier.verify(intent)
        self.assertEqual(result.status, VerificationStatus.PROVED_SAT)
        self.assertEqual(result.check("initial-precondition").status, CheckStatus.PASSED)
        self.assertEqual(result.check("increment:postcondition").status, CheckStatus.PASSED)

    def test_verify_pair_returns_graph_or_counterexample(self):
        good = parse_intent("Create a counter that cannot exceed 10 and increments by 1")
        status, payload = SMTVerifier().verify_pair(good)
        self.assertEqual(status, VerificationStatus.PROVED_SAT)
        self.assertEqual(payload.initial_state, {"counter": "0"})

        bad = IntentAST(
            name="counter",
            variables=[Variable("counter", initial="10", lower="0", upper="10")],
            invariants=["counter <= 10"],
            actions=[Action("increment", updates={"counter": "counter + 1"})],
        )
        status, payload = SMTVerifier().verify_pair(bad)
        self.assertEqual(status, VerificationStatus.UNSAT)
        self.assertEqual(payload.next_state, {"counter": 11})


if __name__ == "__main__":
    unittest.main()
