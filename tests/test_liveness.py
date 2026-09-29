import unittest
import numpy as np
from core.liveness import (
    calculate_ear,
    verify_liveness
)

class TestLiveness(unittest.TestCase):
    def test_calculate_ear(self):
        # Open eye mock coordinates (width > height)
        # 6 points: p1, p2, p3, p4, p5, p6
        # p1=(0, 0), p2=(5, 5), p3=(10, 5), p4=(15, 0), p5=(10, -5), p6=(5, -5)
        open_eye = [
            (0, 0),
            (5, 5),
            (10, 5),
            (15, 0),
            (10, -5),
            (5, -5)
        ]
        ear = calculate_ear(open_eye)
        self.assertGreater(ear, 0.25)

        # Closed eye mock coordinates (height ≈ 0)
        closed_eye = [
            (0, 0),
            (5, 0.5),
            (10, 0.5),
            (15, 0),
            (10, -0.5),
            (5, -0.5)
        ]
        closed_ear = calculate_ear(closed_eye)
        self.assertLess(closed_ear, 0.15)
        self.assertLess(closed_ear, ear)

    def test_empty_frames_rejection(self):
        res = verify_liveness([])
        self.assertFalse(res["passed"])
        self.assertIn("No frames", res["reason"])

    def test_demo_mode_pass(self):
        # Demo mode allows testing without hardware camera
        res = verify_liveness(["dummy"], is_demo=True)
        self.assertTrue(res["passed"])
        self.assertEqual(res["challenge"], "demo_verified")

if __name__ == "__main__":
    unittest.main()
