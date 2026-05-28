import os
import sys
import unittest


tests_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(tests_dir)
sys.path.append(os.path.join(root_dir, "src"))

from detectors.perclos import PerclosDetector


class PerclosDetectorTest(unittest.TestCase):
    def test_calculates_closed_time_ratio(self):
        detector = PerclosDetector(window_seconds=10.0, threshold=0.4)

        detector.update(False, 0.0)
        detector.update(True, 2.0)
        detector.update(True, 5.0)
        detector.update(False, 10.0)

        self.assertAlmostEqual(detector.get_perclos(), 0.8)
        self.assertTrue(detector.has_alert())

    def test_prunes_samples_outside_window(self):
        detector = PerclosDetector(window_seconds=5.0, threshold=0.4)

        detector.update(True, 0.0)
        detector.update(True, 1.0)
        detector.update(False, 10.0)

        self.assertEqual(detector.get_perclos(), 0.0)

    def test_waits_for_minimum_observation_time_before_alerting(self):
        detector = PerclosDetector(
            window_seconds=60.0,
            threshold=0.4,
            min_observation_seconds=10.0,
        )

        detector.update(True, 0.0)
        detector.update(True, 5.0)

        self.assertEqual(detector.get_perclos(), 1.0)
        self.assertFalse(detector.has_alert())

        detector.update(True, 10.0)

        self.assertTrue(detector.has_alert())


if __name__ == "__main__":
    unittest.main()
