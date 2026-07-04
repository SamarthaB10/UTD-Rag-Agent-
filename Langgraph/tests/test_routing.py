import unittest

from rag_agent.routing import Route, classify_question, routed_response


class RoutingTests(unittest.TestCase):
    def test_crisis_routing_is_deterministic(self):
        cases = {
            "There is an active shooter": Route.IMMEDIATE_DANGER,
            "I am thinking about suicide": Route.MENTAL_HEALTH,
            "Someone is not breathing": Route.MEDICAL_EMERGENCY,
            "How do I make a Title IX report?": Route.SEXUAL_MISCONDUCT,
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                route = classify_question(question).route
                self.assertEqual(route, expected)
                self.assertIsNotNone(routed_response(route))

    def test_personal_record_routes_away_from_rag(self):
        route = classify_question("Why is my financial aid status still pending?").route
        response = routed_response(route)
        self.assertEqual(route, Route.PERSONAL_RECORD)
        self.assertEqual(response.evidence_source, "unverified")
        self.assertTrue(response.needs_official_confirmation)

    def test_general_policy_question_uses_normal_workflow(self):
        self.assertEqual(
            classify_question("What is the policy for withdrawing from a class?").route,
            Route.NORMAL,
        )


if __name__ == "__main__":
    unittest.main()
