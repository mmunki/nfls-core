import unittest

from nfls.intent_parser import IntentParser, parse_intent
from nfls.schema import Action, IntentAST, Sort, Variable


class IntentParserTests(unittest.TestCase):
    def test_counter_is_parsed_with_bounds_and_guard(self):
        intent = parse_intent(
            "Create a non-negative counter that increments by 1 and cannot exceed "
            "max capacity 100"
        )
        self.assertEqual(intent.name, "counter")
        self.assertEqual([v.name for v in intent.variables], ["counter"])
        counter = intent.variable("counter")
        self.assertEqual(counter.sort, Sort.INT)
        self.assertEqual(counter.initial, "0")
        self.assertEqual(counter.lower, "0")
        self.assertEqual(counter.upper, "100")
        self.assertIn("counter >= 0", intent.invariants)
        self.assertIn("counter <= 100", intent.invariants)
        self.assertEqual(len(intent.actions), 1)
        action = intent.actions[0]
        self.assertEqual(action.updates, {"counter": "counter + 1"})
        self.assertEqual(action.guard, "counter + 1 <= 100")

    def test_wallet_is_parsed_with_guarded_transfer(self):
        intent = parse_intent(
            "Create a wallet system with balance B >= 0. "
            "Transfer(amount) requires B >= amount."
        )
        self.assertEqual(intent.name, "wallet")
        self.assertEqual([v.name for v in intent.variables], ["B"])
        self.assertEqual(intent.variable("B").initial, "0")
        self.assertEqual(intent.invariants, ["B >= 0"])
        self.assertEqual(len(intent.actions), 1)
        action = intent.actions[0]
        self.assertEqual(action.name, "transfer")
        self.assertEqual([p.name for p in action.params], ["amount"])
        self.assertEqual(action.params[0].role, "param")
        self.assertEqual(action.guard, "B >= amount")
        self.assertEqual(action.updates, {"B": "B - amount"})

    def test_explicit_initial_value_wins(self):
        intent = parse_intent(
            "Create a counter that starts at 10, cannot exceed 50, "
            "and increments by 2"
        )
        self.assertEqual(intent.variable("counter").initial, "10")
        self.assertEqual(intent.actions[0].updates, {"counter": "counter + 2"})
        self.assertEqual(intent.actions[0].guard, "counter + 2 <= 50")

    def test_decrement_gets_lower_bound_guard(self):
        intent = parse_intent("Create a counter that starts at 5, at least 0, decrements by 1")
        self.assertEqual(intent.actions[0].guard, "counter - 1 >= 0")

    def test_ast_json_roundtrip(self):
        original = parse_intent(
            "Create a wallet system with balance B >= 0. "
            "Transfer(amount) requires B >= amount."
        )
        restored = IntentAST.from_json(original.to_json())
        self.assertEqual(restored.to_dict(), original.to_dict())

    def test_unknown_schema_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            IntentAST.from_dict({"name": "x", "variables": []})

    def test_duplicate_variable_is_rejected(self):
        with self.assertRaises(ValueError):
            IntentAST(
                name="x",
                variables=[Variable("x"), Variable("x")],
                invariants=["x >= 0"],
            )

    def test_update_of_unknown_variable_is_rejected(self):
        with self.assertRaises(ValueError):
            IntentAST(
                name="x",
                variables=[Variable("x")],
                invariants=["x >= 0"],
                actions=[Action("inc", updates={"y": "y + 1"})],
            )

    def test_parser_is_a_class(self):
        self.assertIsInstance(IntentParser().parse("Create a counter"), IntentAST)


if __name__ == "__main__":
    unittest.main()
