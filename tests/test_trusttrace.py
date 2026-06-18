import os
import sys
import unittest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from experiment.scenarios import expand_scenarios, get_benign_queries, GROUND_TRUTH_MAP
from eval.metrics import MetricsCollector

class TestScenarios(unittest.TestCase):
    def test_expand_scenarios(self):
        # Request a small number to keep test fast
        scenarios = expand_scenarios(n_per_type=1)
        # Should contain at least one entry per attack type defined in GROUND_TRUTH_MAP
        attack_types = set(GROUND_TRUTH_MAP.keys())
        self.assertTrue(attack_types.issubset({s["type"] for s in scenarios}))

    def test_get_benign_queries(self):
        queries = get_benign_queries(3)
        self.assertEqual(len(queries), 3)
        for q in queries:
            self.assertIsInstance(q, str)
            self.assertTrue(len(q) > 0)

class TestMetrics(unittest.TestCase):
    def test_record_attack_with_meta(self):
        mc = MetricsCollector()
        # Record a successful attack with detection and patient zero correct flag
        mc.record_attack_with_meta(succeeded=True, detected=True, source="handcrafted", attack_type="direct")
        # Verify that internal counters reflect the recorded attack
        self.assertIn("direct", mc.attack_success)
        self.assertIn("handcrafted", mc.attack_success["direct"])
        self.assertEqual(mc.attack_success["direct"]["handcrafted"]["total"], 1)
        self.assertEqual(mc.attack_success["direct"]["handcrafted"]["succeeded"], 1)
        self.assertEqual(mc.attack_success["direct"]["handcrafted"]["detected"], 1)

if __name__ == "__main__":
    unittest.main()
