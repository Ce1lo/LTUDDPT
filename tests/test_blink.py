import os
import sys
import unittest


tests_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(tests_dir)
sys.path.append(os.path.join(root_dir, "src"))

from detectors.blink import BlinkDetector


class BlinkDetectorTest(unittest.TestCase):
    def test_counts_valid_blink_duration(self):
        detector = BlinkDetector(
            window_seconds=10,
            blink_threshold=10,
            debounce_frames=1,
            min_blink_duration=0.08,
            max_blink_duration=0.45,
            refractory_seconds=0.10,
        )

        detector.update(False, 0.00)
        detector.update(True, 0.10)   # close
        detector.update(False, 0.30)  # open, duration=0.20 -> valid

        self.assertEqual(detector.get_total_blinks(), 1)
        self.assertEqual(detector.get_blink_rate(0.30), 1)

    def test_ignores_too_short_or_too_long_closure(self):
        detector = BlinkDetector(
            window_seconds=10,
            blink_threshold=10,
            debounce_frames=1,
            min_blink_duration=0.08,
            max_blink_duration=0.45,
            refractory_seconds=0.10,
        )

        detector.update(False, 0.00)
        detector.update(True, 0.10)
        detector.update(False, 0.15)  # duration=0.05 -> too short

        detector.update(True, 1.00)
        detector.update(False, 1.70)  # duration=0.70 -> too long

        self.assertEqual(detector.get_total_blinks(), 0)

    def test_applies_refractory_gap(self):
        detector = BlinkDetector(
            window_seconds=10,
            blink_threshold=10,
            debounce_frames=1,
            min_blink_duration=0.08,
            max_blink_duration=0.45,
            refractory_seconds=0.20,
        )

        detector.update(False, 0.00)
        detector.update(True, 0.10)
        detector.update(False, 0.25)  # blink #1

        detector.update(True, 0.30)
        detector.update(False, 0.45)  # too close to previous blink -> ignored

        detector.update(True, 0.60)
        detector.update(False, 0.75)  # blink #2

        self.assertEqual(detector.get_total_blinks(), 2)


if __name__ == "__main__":
    unittest.main()
