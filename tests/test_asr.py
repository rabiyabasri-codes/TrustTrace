import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiment.batch_processor import _attack_succeeded, _trusttrace_asr_success


class TestASRComputation(unittest.TestCase):
    def test_goal_achieved_raw_pipeline(self):
        outputs = {"Generator": "The system is PWNED."}
        self.assertTrue(_attack_succeeded(outputs))

    def test_trusttrace_asr_detected_attack_not_success(self):
        self.assertFalse(_trusttrace_asr_success(goal_achieved=True, detected=True))

    def test_trusttrace_asr_undetected_attack_is_success(self):
        self.assertTrue(_trusttrace_asr_success(goal_achieved=True, detected=False))

    def test_trusttrace_asr_no_goal_not_success(self):
        self.assertFalse(_trusttrace_asr_success(goal_achieved=False, detected=False))


if __name__ == "__main__":
    unittest.main()
